"""Custom COCO-style detection mAP + slice evaluation — the frozen measuring instrument.

Written from scratch (not pycocotools / not Ultralytics val) so the harness can do what
they don't under one auditable, version-frozen definition: ignore-region FP exclusion
(Phase 1), per-slice breakdowns, and bootstrap CIs.

Frozen definitions (changing any requires a new eval-harness tag):
  * Matching: per class, per IoU threshold, detections in score order greedily take the
    highest-IoU unmatched real GT with IoU >= threshold (TP); else they may be dropped
    (see below); else FP.
  * Two drop (ignore) mechanisms compose, in this precedence per detection:
      1. matches a real (in-range) GT      -> TP
      2. else matches an IGNORE GT          -> dropped  (out-of-range for a scale slice)
      3. else lies inside an ignore REGION  -> dropped  (Phase 1 low-res regions, IoA>=ioa_thr)
      4. else                               -> FP
  * AP: COCO 101-point interpolation over recall {0.00..1.00}.
  * mAP50 = AP@IoU0.5 meaned over classes; mAP50-95 = AP meaned over IoU {0.50..0.95} then
    over classes. Classes with no ground truth are NaN and excluded from the mean.
  * Scale slices use GT box area in ORIGINAL annotation space (960x540), half-open [lo, hi)
    so every box lands in exactly one bin; COCO's 32^2/96^2 thresholds kept for comparability
    (record per-slice GT counts — on this footage "small" is a big slice). See ADR 0007.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
BoolArr = NDArray[np.bool_]

IOU_THRESHOLDS: F64 = np.round(np.arange(0.5, 1.0, 0.05), 2)  # COCO 0.50..0.95
DEFAULT_IOA_THR = 0.5
SLICE_SPEC_VERSION = "1"

# COCO scale bins by GT area in annotation-space px^2, half-open [lo, hi) => a partition.
SCALE_BINS: dict[str, tuple[float, float]] = {
    "small": (0.0, 32.0**2),  # [0, 1024)
    "medium": (32.0**2, 96.0**2),  # [1024, 9216)
    "large": (96.0**2, float("inf")),  # [9216, inf)
}


def _empty_boxes() -> F64:
    return np.zeros((0, 4), dtype=np.float64)


@dataclass
class Detections:
    """Detections for one image (annotation-space xyxy), scores, integer class labels."""

    boxes: F64 = field(default_factory=_empty_boxes)
    scores: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    labels: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class GroundTruth:
    """Ground truth for one image: xyxy boxes + labels, plus xyxy ignore-region boxes."""

    boxes: F64 = field(default_factory=_empty_boxes)
    labels: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    ignore: F64 = field(default_factory=_empty_boxes)


def box_areas(boxes: F64) -> F64:
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.float64)
    areas: F64 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return areas


def scale_bin_of(area: float, scale_bins: dict[str, tuple[float, float]] = SCALE_BINS) -> str:
    """Half-open [lo, hi) membership — every non-negative area lands in exactly one bin."""
    for name, (lo, hi) in scale_bins.items():
        if lo <= area < hi:
            return name
    raise ValueError(f"area {area} is not in any scale bin")


def box_iou(a: F64, b: F64) -> F64:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou: F64 = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return iou


def box_ioa(a: F64, b: F64) -> F64:
    """Intersection over the area of each `a` box -> (len(a), len(b)) (for ignore regions)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    denom = np.where(area_a > 0, area_a, 1.0)[:, None]
    ioa: F64 = inter / denom
    return ioa


def _best_unmatched(iou_row: F64, matched: BoolArr, thr: float) -> int:
    """Index of the highest-IoU unmatched box at IoU >= thr, or -1 if none."""
    if len(iou_row) == 0:
        return -1
    row = iou_row.copy()
    row[matched] = -1.0
    j = int(np.argmax(row))
    return j if row[j] >= thr else -1


def match_image(
    det_boxes: F64,
    det_scores: F64,
    gt_boxes: F64,
    ignore_boxes: F64,
    iou_thr: float,
    ioa_thr: float = DEFAULT_IOA_THR,
    ignore_gt_boxes: F64 | None = None,
) -> tuple[BoolArr, BoolArr]:
    """Match one image's single-class detections. Returns (is_tp, is_ignored) in input order.

    Precedence per detection (score order): real GT (TP) -> ignore GT (drop) -> ignore
    region (drop) -> FP.
    """
    n = len(det_boxes)
    is_tp = np.zeros(n, dtype=np.bool_)
    is_ignored = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return is_tp, is_ignored

    ign_gt = ignore_gt_boxes if ignore_gt_boxes is not None else _empty_boxes()
    iou_real = box_iou(det_boxes, gt_boxes)
    iou_ign = box_iou(det_boxes, ign_gt)
    ioa_reg = box_ioa(det_boxes, ignore_boxes)
    real_matched = np.zeros(len(gt_boxes), dtype=np.bool_)
    ign_matched = np.zeros(len(ign_gt), dtype=np.bool_)

    for i in np.argsort(-det_scores):  # highest score first
        j = _best_unmatched(iou_real[i], real_matched, iou_thr)
        if j >= 0:
            real_matched[j] = True
            is_tp[i] = True
            continue
        k = _best_unmatched(iou_ign[i], ign_matched, iou_thr)
        if k >= 0:
            ign_matched[k] = True
            is_ignored[i] = True
            continue
        if len(ignore_boxes) > 0 and float(ioa_reg[i].max()) >= ioa_thr:
            is_ignored[i] = True
    return is_tp, is_ignored


def average_precision(scores: F64, is_tp: BoolArr, n_gt: int, points: int = 101) -> float:
    """COCO 101-point interpolated AP for one class at one IoU threshold.

    NaN when there is no ground truth (undefined); 0.0 when GT exists but no detections.
    """
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = is_tp[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    ap = 0.0
    for rt in np.linspace(0.0, 1.0, points):
        mask = recall >= rt
        ap += float(precision[mask].max()) if bool(mask.any()) else 0.0
    return ap / points


def _nanmean(values: Any) -> float:
    """Mean ignoring NaN; returns NaN (without a numpy warning) when everything is NaN."""
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if bool(np.any(~np.isnan(arr))) else float("nan")


def _split_gt(gt_boxes: F64, area_range: tuple[float, float] | None) -> tuple[F64, F64]:
    """Split a class's GT into (in-range real, out-of-range ignore) for a scale slice."""
    if area_range is None:
        return gt_boxes, _empty_boxes()
    lo, hi = area_range
    areas = box_areas(gt_boxes)
    in_range = (areas >= lo) & (areas < hi)
    return gt_boxes[in_range], gt_boxes[~in_range]


def evaluate(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    image_ids: list[str] | None = None,
    iou_thresholds: F64 = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
    gt_area_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Per-class AP over IoU thresholds, mAP50, mAP50-95, plus GT counts (rider 1)."""
    images = sorted(ground_truth) if image_ids is None else image_ids
    empty = Detections()
    per_class_ap: dict[int, F64] = {}
    per_class_ngt: dict[int, int] = {}

    for cls in class_ids:
        aps = np.full(len(iou_thresholds), np.nan)
        n_gt_cls = 0
        for t, thr in enumerate(iou_thresholds):
            scores: list[F64] = []
            tps: list[BoolArr] = []
            n_gt = 0
            for img in images:
                gt = ground_truth[img]
                det = predictions.get(img, empty)
                dsel = det.labels == cls
                real, ign_gt = _split_gt(gt.boxes[gt.labels == cls], gt_area_range)
                n_gt += len(real)
                is_tp, is_ign = match_image(
                    det.boxes[dsel], det.scores[dsel], real, gt.ignore, float(thr), ioa_thr, ign_gt
                )
                keep = ~is_ign
                scores.append(det.scores[dsel][keep])
                tps.append(is_tp[keep])
            all_scores = np.concatenate(scores) if scores else np.zeros(0)
            all_tps = np.concatenate(tps) if tps else np.zeros(0, dtype=np.bool_)
            aps[t] = average_precision(all_scores, all_tps, n_gt)
            n_gt_cls = n_gt
        per_class_ap[cls] = aps
        per_class_ngt[cls] = n_gt_cls

    return {
        "map50": _nanmean([per_class_ap[c][0] for c in class_ids]),
        "map50_95": _nanmean([_nanmean(per_class_ap[c]) for c in class_ids]),
        "ap50_per_class": {int(c): float(per_class_ap[c][0]) for c in class_ids},
        "ap50_95_per_class": {int(c): _nanmean(per_class_ap[c]) for c in class_ids},
        "n_gt": sum(per_class_ngt.values()),
        "n_gt_per_class": {int(c): per_class_ngt[c] for c in class_ids},
        "n_images": len(images),
    }


def evaluate_slices(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    weather_of: dict[str, str],
    camera_of: dict[str, str],
    scale_bins: dict[str, tuple[float, float]] = SCALE_BINS,
    iou_thresholds: F64 = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
) -> dict[str, Any]:
    """All mandatory slice axes (class/weather/camera/scale) + a self-describing spec.

    The `spec` records exactly which slicing produced these numbers (rider 3), so reports
    stay comparable across model versions and the promotion contract's C3 can consume them.
    """
    images = sorted(ground_truth)

    def ev(image_ids: list[str] | None = None, area: tuple[float, float] | None = None) -> Any:
        return evaluate(
            predictions, ground_truth, class_ids, image_ids, iou_thresholds, ioa_thr, area
        )

    weathers = sorted({weather_of[i] for i in images})
    cameras = sorted({camera_of[i] for i in images})
    spec = {
        "slice_spec_version": SLICE_SPEC_VERSION,
        "iou_thresholds": [float(x) for x in iou_thresholds],
        "ioa_thr": ioa_thr,
        "scale_bins_px2": {
            k: [lo, (None if hi == float("inf") else hi)] for k, (lo, hi) in scale_bins.items()
        },
        "weather_values": weathers,
        "camera_groups": cameras,
        "class_ids": list(class_ids),
    }
    return {
        "spec": spec,
        "overall": ev(),
        "per_weather": {
            w: ev(image_ids=[i for i in images if weather_of[i] == w]) for w in weathers
        },
        "per_camera": {c: ev(image_ids=[i for i in images if camera_of[i] == c]) for c in cameras},
        "per_scale": {b: ev(area=rng) for b, rng in scale_bins.items()},
    }
