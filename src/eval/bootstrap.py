"""Bootstrap percentile confidence intervals for every harness slice metric.

Resamples IMAGES with replacement (not boxes — boxes within a frame are correlated). Since
matching is per-image, each detection's (score, is_tp) is precomputed once and sorted by
score; a replicate only re-weights detections by their image's multiplicity and recomputes
the 101-point AP — no re-matching — so >=1000 resamples stay cheap. The point estimate
(all multiplicities = 1) equals harness.evaluate exactly (consistency test), so the CIs are
around the *same* frozen definition. See ADR 0007.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.eval.harness import (
    DEFAULT_IOA_THR,
    IOU_THRESHOLDS,
    SCALE_BINS,
    Detections,
    GroundTruth,
    _nanmean,
    _split_gt,
    match_image,
)

F64 = NDArray[np.float64]
DEFAULT_N_RESAMPLES = 2000
CI_ALPHA = 0.05  # 95% CI


def _weighted_ap(
    is_tp_sorted: F64, img_idx_sorted: NDArray[np.int64], m: F64, n_gt: float
) -> float:
    """101-point AP where each detection is weighted by its image's resample multiplicity."""
    if n_gt <= 0:
        return float("nan")
    if len(is_tp_sorted) == 0:
        return 0.0
    w = m[img_idx_sorted]
    tp_cum = np.cumsum(is_tp_sorted * w)
    fp_cum = np.cumsum((1.0 - is_tp_sorted) * w)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    p_env = np.maximum.accumulate(precision[::-1])[::-1]  # max precision to the right
    idx = np.searchsorted(recall, np.linspace(0.0, 1.0, 101), side="left")
    valid = idx < len(recall)
    return float(p_env[idx[valid]].sum()) / 101.0


def _prepare(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    images: list[str],
    iou_thresholds: F64,
    ioa_thr: float,
    area_range: tuple[float, float] | None,
) -> dict[str, Any]:
    """Precompute, per class and IoU threshold, score-sorted (is_tp, image index) + per-image GT."""
    n_thr = len(iou_thresholds)
    empty = Detections()
    ngt = {c: np.zeros(len(images)) for c in class_ids}
    sorted_tp: dict[int, list[F64]] = {c: [] for c in class_ids}
    sorted_img: dict[int, list[NDArray[np.int64]]] = {c: [] for c in class_ids}

    for c in class_ids:
        buckets_tp: list[list[F64]] = [[] for _ in range(n_thr)]
        buckets_img: list[list[NDArray[np.int64]]] = [[] for _ in range(n_thr)]
        buckets_score: list[list[F64]] = [[] for _ in range(n_thr)]
        for ii, im in enumerate(images):
            gt = ground_truth[im]
            det = predictions.get(im, empty)
            dsel = det.labels == c
            real, ign_gt = _split_gt(gt.boxes[gt.labels == c], area_range)
            ngt[c][ii] = len(real)
            for t, thr in enumerate(iou_thresholds):
                is_tp, is_ign = match_image(
                    det.boxes[dsel], det.scores[dsel], real, gt.ignore, float(thr), ioa_thr, ign_gt
                )
                keep = ~is_ign
                buckets_score[t].append(det.scores[dsel][keep])
                buckets_tp[t].append(is_tp[keep].astype(np.float64))
                buckets_img[t].append(np.full(int(keep.sum()), ii, dtype=np.int64))
        for t in range(n_thr):
            scores = np.concatenate(buckets_score[t]) if buckets_score[t] else np.zeros(0)
            tp = np.concatenate(buckets_tp[t]) if buckets_tp[t] else np.zeros(0)
            imgs = np.concatenate(buckets_img[t]) if buckets_img[t] else np.zeros(0, dtype=np.int64)
            order = np.argsort(-scores)
            sorted_tp[c].append(tp[order])
            sorted_img[c].append(imgs[order])
    return {"ngt": ngt, "sorted_tp": sorted_tp, "sorted_img": sorted_img, "n_thr": n_thr}


def _replicate(prep: dict[str, Any], class_ids: list[int], m: F64) -> dict[str, float]:
    """All metrics for one image-multiplicity vector m."""
    per_class: dict[int, F64] = {}
    for c in class_ids:
        n_gt = float((m * prep["ngt"][c]).sum())
        aps = np.array(
            [
                _weighted_ap(prep["sorted_tp"][c][t], prep["sorted_img"][c][t], m, n_gt)
                for t in range(prep["n_thr"])
            ]
        )
        per_class[c] = aps
    out: dict[str, float] = {
        "map50": _nanmean([per_class[c][0] for c in class_ids]),
        "map50_95": _nanmean([_nanmean(per_class[c]) for c in class_ids]),
    }
    for c in class_ids:
        out[f"ap50_class_{c}"] = float(per_class[c][0])
        out[f"ap50_95_class_{c}"] = _nanmean(per_class[c])
    return out


def bootstrap_metrics(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    image_ids: list[str] | None = None,
    iou_thresholds: F64 = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
    area_range: tuple[float, float] | None = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Point estimate + percentile 95% CI + n_gt for every metric over one slice."""
    images = sorted(ground_truth) if image_ids is None else list(image_ids)
    prep = _prepare(
        predictions, ground_truth, class_ids, images, iou_thresholds, ioa_thr, area_range
    )
    n_img = len(images)

    point = _replicate(prep, class_ids, np.ones(n_img))
    dist: dict[str, list[float]] = {k: [] for k in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        m = np.bincount(rng.integers(0, n_img, size=n_img), minlength=n_img).astype(np.float64)
        for k, v in _replicate(prep, class_ids, m).items():
            dist[k].append(v)

    total_gt = int(sum(int(prep["ngt"][c].sum()) for c in class_ids))
    n_gt_of: dict[str, int] = {"map50": total_gt, "map50_95": total_gt}
    for c in class_ids:
        n_gt_of[f"ap50_class_{c}"] = int(prep["ngt"][c].sum())
        n_gt_of[f"ap50_95_class_{c}"] = int(prep["ngt"][c].sum())

    result: dict[str, dict[str, float]] = {}
    for k, pv in point.items():
        arr = np.array([x for x in dist[k] if not np.isnan(x)])
        lo = float(np.percentile(arr, 100 * CI_ALPHA / 2)) if len(arr) else float("nan")
        hi = float(np.percentile(arr, 100 * (1 - CI_ALPHA / 2))) if len(arr) else float("nan")
        result[k] = {"point": pv, "ci_low": lo, "ci_high": hi, "n_gt": n_gt_of[k]}
    return result


def bootstrap_slices(
    predictions: dict[str, Detections],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    weather_of: dict[str, str],
    camera_of: dict[str, str],
    scale_bins: dict[str, tuple[float, float]] = SCALE_BINS,
    iou_thresholds: F64 = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile CIs for every metric on every slice axis (overall/weather/camera/scale)."""
    images = sorted(ground_truth)

    def boot(image_ids: list[str] | None = None, area: tuple[float, float] | None = None) -> Any:
        return bootstrap_metrics(
            predictions,
            ground_truth,
            class_ids,
            image_ids,
            iou_thresholds,
            ioa_thr,
            area,
            n_resamples,
            seed,
        )

    weathers = sorted({weather_of[i] for i in images})
    cameras = sorted({camera_of[i] for i in images})
    return {
        "n_resamples": n_resamples,
        "ci_alpha": CI_ALPHA,
        "overall": boot(),
        "per_weather": {
            w: boot(image_ids=[i for i in images if weather_of[i] == w]) for w in weathers
        },
        "per_camera": {
            c: boot(image_ids=[i for i in images if camera_of[i] == c]) for c in cameras
        },
        "per_scale": {b: boot(area=rng) for b, rng in scale_bins.items()},
    }
