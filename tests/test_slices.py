"""Slice layer: half-open scale bins, composed ignore precedence, partition invariants,
and a self-describing slice spec. Numbers are hand-computed.
"""

import numpy as np

from src.eval.harness import (
    SLICE_SPEC_VERSION,
    Detections,
    GroundTruth,
    evaluate,
    evaluate_slices,
    match_image,
    scale_bin_of,
)


def _f(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)


def _det(boxes: list[list[float]], scores: list[float], labels: list[int]) -> Detections:
    return Detections(
        _f(boxes), np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64)
    )


def _gt(
    boxes: list[list[float]], labels: list[int], ignore: list[list[float]] | None = None
) -> GroundTruth:
    return GroundTruth(_f(boxes), np.array(labels, dtype=np.int64), _f(ignore or []))


# small box area 400 (<1024), large box area 40000 (>9216)
SMALL = [0.0, 0.0, 20.0, 20.0]
LARGE = [100.0, 100.0, 300.0, 300.0]
REGION = [500.0, 500.0, 600.0, 600.0]


def test_scale_bin_half_open_edges() -> None:
    assert scale_bin_of(0.0) == "small"
    assert scale_bin_of(1023.9) == "small"
    assert scale_bin_of(1024.0) == "medium"  # 32^2 -> medium (half-open [lo, hi))
    assert scale_bin_of(9215.9) == "medium"
    assert scale_bin_of(9216.0) == "large"  # 96^2 -> large
    assert scale_bin_of(1e9) == "large"


def test_every_area_lands_in_exactly_one_bin() -> None:
    # partition invariant: scale_bin_of returns for any non-negative area, never raises
    for area in [0.0, 1023.0, 1024.0, 5000.0, 9216.0, 100000.0]:
        assert scale_bin_of(area) in {"small", "medium", "large"}


def test_match_image_ignore_precedence() -> None:
    # real GT -> TP; ignore GT -> drop; ignore region -> drop; in that order
    dets = _f([LARGE, REGION, SMALL])
    scores = np.array([0.95, 0.90, 0.85])
    is_tp, is_ign = match_image(
        dets, scores, _f([SMALL]), _f([REGION]), iou_thr=0.5, ignore_gt_boxes=_f([LARGE])
    )
    assert list(is_tp) == [False, False, True]  # only the small-GT match is a TP
    assert list(is_ign) == [True, True, False]  # large->ignore-GT, region->ignore-region


def test_scale_slice_composes_both_ignore_mechanisms() -> None:
    # high-scored ignore candidates before the TP: mAP is 1.0 ONLY if both drops fire
    preds = {"img": _det([LARGE, REGION, SMALL], [0.95, 0.90, 0.85], [0, 0, 0])}
    gt = {"img": _gt([SMALL, LARGE], [0, 0], ignore=[REGION])}
    small = evaluate(preds, gt, [0], gt_area_range=(0.0, 1024.0))
    large = evaluate(preds, gt, [0], gt_area_range=(9216.0, float("inf")))
    medium = evaluate(preds, gt, [0], gt_area_range=(1024.0, 9216.0))
    assert abs(small["map50"] - 1.0) < 1e-9  # large GT match ignored, region det ignored
    assert small["n_gt"] == 1  # only the small GT counts (rider 1: per-slice GT count)
    assert abs(large["map50"] - 1.0) < 1e-9  # symmetric: small GT match becomes ignore-GT
    assert np.isnan(medium["map50"])  # no medium GT -> undefined, not zero


def test_slices_are_self_describing_and_partition() -> None:
    preds = {
        "a": _det([SMALL], [0.9], [0]),
        "b": _det([SMALL], [0.9], [0]),
        "c": _det([SMALL], [0.9], [0]),
    }
    gt = {k: _gt([SMALL], [0]) for k in ("a", "b", "c")}
    weather = {"a": "sunny", "b": "night", "c": "sunny"}
    camera = {"a": "cam_x", "b": "cam_x", "c": "cam_y"}
    r = evaluate_slices(preds, gt, [0], weather, camera)

    assert r["spec"]["slice_spec_version"] == SLICE_SPEC_VERSION
    assert r["spec"]["weather_values"] == ["night", "sunny"]
    assert r["spec"]["camera_groups"] == ["cam_x", "cam_y"]
    assert r["spec"]["scale_bins_px2"]["large"] == [9216.0, None]
    # weather and camera are partitions of the images -> subset counts sum to the whole
    assert sum(s["n_images"] for s in r["per_weather"].values()) == r["overall"]["n_images"]
    assert sum(s["n_images"] for s in r["per_camera"].values()) == r["overall"]["n_images"]
