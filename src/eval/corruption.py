"""Image corruptions + degradation-curve runner — applied on-the-fly, never materialized.

Five corruptions, severities 0..5 (0 == exact identity). Corruption happens in-memory at
eval time (never 20x disk copies; rider 4). Stochastic noise is seeded per-image from a
STABLE hash of (image id, corruption, severity) so a given corrupted image is bit-identical
on any machine, in any iteration order (rider 2). "brightness shift" is split into the two
signed, non-symmetric directions (rider 1) — darkening is the production-relevant one
(day->night drift). Additive pipeline is pinned float32 -> add -> clip[0,255] -> round ->
uint8. Severity ranges below are FROZEN: calibrated on the zero-shot COCO-YOLO floor
(monotone, non-degenerate curves) and frozen as part of eval-harness-v1. See ADR 0008;
changing any range now requires a new harness tag + historical re-run.

Cost note (rider 4): a full sweep is ~ (corruptions x severities) extra eval passes, so it
belongs in the report command and the promotion pipeline (C4 champion vs challenger), NOT
every training run's post-hoc eval.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.eval.harness import DEFAULT_IOA_THR, IOU_THRESHOLDS, Detections, GroundTruth, evaluate

U8 = NDArray[np.uint8]

MAX_SEVERITY = 5
# FROZEN ranges (index by severity-1). Calibrated on the zero-shot floor — see ADR 0008.
BLUR_SIGMA = (0.5, 1.0, 2.0, 3.0, 4.0)
NOISE_SIGMA = (5.0, 10.0, 20.0, 40.0, 60.0)
JPEG_QUALITY = (80, 60, 40, 25, 15)
BRIGHTNESS_DELTA = (20.0, 40.0, 60.0, 80.0, 100.0)


def stable_seed(image_id: str, corruption: str, severity: int) -> int:
    """Deterministic per-(image, corruption, severity) seed — independent of process/order."""
    digest = hashlib.sha256(f"{image_id}|{corruption}|{severity}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _add_clip_round(image: U8, delta: NDArray[np.float64] | float) -> U8:
    # pinned order: float32 -> add -> clip -> round -> uint8 (rider 1)
    out: U8 = np.round(np.clip(image.astype(np.float32) + delta, 0.0, 255.0)).astype(np.uint8)
    return out


def brightness_up(image: U8, severity: int, seed: int = 0) -> U8:
    return _add_clip_round(image, BRIGHTNESS_DELTA[severity - 1])


def brightness_down(image: U8, severity: int, seed: int = 0) -> U8:
    return _add_clip_round(image, -BRIGHTNESS_DELTA[severity - 1])


def gaussian_noise(image: U8, severity: int, seed: int = 0) -> U8:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, NOISE_SIGMA[severity - 1], image.shape)
    return _add_clip_round(image, noise)


def gaussian_blur(image: U8, severity: int, seed: int = 0) -> U8:
    import cv2  # lazy: keep module import cv2-free so CI can test the numpy corruptions

    sigma = BLUR_SIGMA[severity - 1]
    out: U8 = np.asarray(
        cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma), dtype=np.uint8
    )
    return out


def jpeg_compression(image: U8, severity: int, seed: int = 0) -> U8:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY[severity - 1]]
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    out: U8 = np.asarray(cv2.imdecode(buf, cv2.IMREAD_COLOR), dtype=np.uint8)
    return out


CORRUPTIONS: dict[str, Callable[[U8, int, int], U8]] = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
}
SEVERITY_PARAM: dict[str, tuple[float, ...]] = {
    "gaussian_blur": BLUR_SIGMA,
    "gaussian_noise": NOISE_SIGMA,
    "jpeg_compression": tuple(float(q) for q in JPEG_QUALITY),
    "brightness_up": BRIGHTNESS_DELTA,
    "brightness_down": BRIGHTNESS_DELTA,
}


def apply_corruption(image: U8, corruption: str, severity: int, image_id: str = "") -> U8:
    """Corrupt one image. Severity 0 is an exact identity (short-circuits before any op)."""
    if severity == 0:
        return image.copy()
    if not 1 <= severity <= MAX_SEVERITY:
        raise ValueError(f"severity must be 0..{MAX_SEVERITY}, got {severity}")
    return CORRUPTIONS[corruption](image, severity, stable_seed(image_id, corruption, severity))


def degradation_curves(
    image_ids: list[str],
    load_image: Callable[[str], U8],
    ground_truth: dict[str, GroundTruth],
    class_ids: list[int],
    detect: Callable[[str, U8], Detections],
    corruptions: list[str] | None = None,
    max_severity: int = MAX_SEVERITY,
    iou_thresholds: NDArray[np.float64] = IOU_THRESHOLDS,
    ioa_thr: float = DEFAULT_IOA_THR,
) -> dict[str, Any]:
    """mAP-vs-severity curves per corruption, corrupting each image on the fly at eval time."""
    names = list(CORRUPTIONS) if corruptions is None else corruptions
    curves: dict[str, dict[int, dict[str, Any]]] = {}
    for corr in names:
        curve: dict[int, dict[str, Any]] = {}
        for sev in range(max_severity + 1):
            preds = {
                iid: detect(iid, apply_corruption(load_image(iid), corr, sev, iid))
                for iid in image_ids
            }
            res = evaluate(preds, ground_truth, class_ids, image_ids, iou_thresholds, ioa_thr)
            curve[sev] = {
                "map50": res["map50"],
                "map50_95": res["map50_95"],
                "param": None if sev == 0 else SEVERITY_PARAM[corr][sev - 1],
            }
        curves[corr] = curve
    return {"severities": list(range(max_severity + 1)), "corruptions": names, "curves": curves}
