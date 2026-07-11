"""Custom COCO-style detection mAP — the project's frozen measuring instrument.

This is written from scratch (not pycocotools / not Ultralytics val) because the harness
must do things they don't: ignore-region false-positive exclusion (Phase 1), per-slice
breakdowns, and bootstrap CIs, all under one auditable, version-frozen definition.

Definitions (frozen at eval-harness-v1 — changing any of these requires a new tag):
  * Matching: per class, per IoU threshold, detections in score order greedily take the
    highest-IoU unmatched ground-truth box with IoU >= threshold (TP); otherwise FP.
  * Ignore regions: a would-be FP whose intersection-over-its-own-area with any ignore
    region is >= `ioa_thr` is DROPPED (neither TP nor FP). Matched detections always count.
  * AP: COCO 101-point interpolation over recall in {0.00, 0.01, ..., 1.00}.
  * mAP50 = AP@IoU0.5 meaned over classes; mAP50-95 = AP meaned over IoU
    {0.50, 0.55, ..., 0.95} then over classes. Classes with no ground truth are NaN and
    excluded from the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

# COCO IoU sweep 0.50..0.95 step 0.05.
IOU_THRESHOLDS: F64 = np.round(np.arange(0.5, 1.0, 0.05), 2)
DEFAULT_IOA_THR = 0.5


def _empty_boxes() -> F64:
    return np.zeros((0, 4), dtype=np.float64)


@dataclass
class Detections:
    """Detections for one image: xyxy boxes, confidence scores, integer class labels."""

    boxes: F64 = field(default_factory=_empty_boxes)
    scores: F64 = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    labels: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class GroundTruth:
    """Ground truth for one image: xyxy boxes + labels, plus xyxy ignore-region boxes."""

    boxes: F64 = field(default_factory=_empty_boxes)
    labels: I64 = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    ignore: F64 = field(default_factory=_empty_boxes)


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
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def box_ioa(a: F64, b: F64) -> F64:
    """Intersection over the area of each `a` box -> (len(a), len(b)).

    Used for ignore regions: 'how much of this detection sits inside an ignore box'.
    """
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


def match_image(
    det_boxes: F64,
    det_scores: F64,
    gt_boxes: F64,
    ignore_boxes: F64,
    iou_thr: float,
    ioa_thr: float = DEFAULT_IOA_THR,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Greedily match one image's detections (single class) to ground truth.

    Returns (is_tp, is_ignored), each aligned to the input detection order. A detection is
    TP if it takes an unmatched GT at IoU >= iou_thr; else it is a candidate FP, and it is
    'ignored' (dropped) if it lies mostly inside an ignore region (IoA >= ioa_thr).
    """
    n = len(det_boxes)
    is_tp = np.zeros(n, dtype=np.bool_)
    is_ignored = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return is_tp, is_ignored

    ious = box_iou(det_boxes, gt_boxes)
    ioas = box_ioa(det_boxes, ignore_boxes)
    gt_matched = np.zeros(len(gt_boxes), dtype=np.bool_)

    for i in np.argsort(-det_scores):  # highest score first
        if len(gt_boxes) > 0:
            row = ious[i].copy()
            row[gt_matched] = -1.0
            best = int(np.argmax(row))
            if row[best] >= iou_thr:
                gt_matched[best] = True
                is_tp[i] = True
                continue
        if len(ignore_boxes) > 0 and float(ioas[i].max()) >= ioa_thr:
            is_ignored[i] = True
    return is_tp, is_ignored


def average_precision(scores: F64, is_tp: NDArray[np.bool_], n_gt: int, points: int = 101) -> float:
    """COCO 101-point interpolated AP for one class at one IoU threshold.

    `scores`/`is_tp` cover the class's detections across all images (ignored ones already
    removed). Returns NaN when there is no ground truth (undefined), 0.0 when there are no
    detections but GT exists.
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


def evaluate(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    image_ids: list[str] | None = None,
    iou_thresholds: F64 = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
) -> dict[str, Any]:
    """Compute per-class AP (per IoU threshold), mAP50, and mAP50-95 over a set of images."""
    images = sorted(ground_truth) if image_ids is None else image_ids
    empty = Detections()
    per_class_ap: dict[int, F64] = {}

    for cls in class_ids:
        aps = np.full(len(iou_thresholds), np.nan)
        for t, thr in enumerate(iou_thresholds):
            scores: list[F64] = []
            tps: list[NDArray[np.bool_]] = []
            n_gt = 0
            for img in images:
                gt = ground_truth[img]
                det = predictions.get(img, empty)
                dsel = det.labels == cls
                gsel = gt.labels == cls
                n_gt += int(gsel.sum())
                is_tp, is_ign = match_image(
                    det.boxes[dsel],
                    det.scores[dsel],
                    gt.boxes[gsel],
                    gt.ignore,
                    float(thr),
                    ioa_thr,
                )
                keep = ~is_ign
                scores.append(det.scores[dsel][keep])
                tps.append(is_tp[keep])
            all_scores = np.concatenate(scores) if scores else np.zeros(0)
            all_tps = np.concatenate(tps) if tps else np.zeros(0, dtype=np.bool_)
            aps[t] = average_precision(all_scores, all_tps, n_gt)
        per_class_ap[cls] = aps

    ap5095 = {int(c): float(np.nanmean(per_class_ap[c])) for c in class_ids}
    map50 = float(np.nanmean([per_class_ap[c][0] for c in class_ids]))
    map5095 = float(np.nanmean([np.nanmean(per_class_ap[c]) for c in class_ids]))
    return {
        "map50": map50,
        "map50_95": map5095,
        "ap50_95_per_class": ap5095,
        "n_images": len(images),
    }
