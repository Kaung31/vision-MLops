"""Bootstrap CIs: the point estimate must equal the frozen evaluate() (same definition),
CIs bracket the point, a perfect slice collapses to [1,1], and results are deterministic.
"""

import numpy as np

from src.eval.bootstrap import bootstrap_metrics, bootstrap_slices
from src.eval.harness import Detections, GroundTruth, evaluate

BOX = [10.0, 10.0, 50.0, 50.0]
WRONG = [200.0, 200.0, 240.0, 240.0]


def _det(boxes: list[list[float]], scores: list[float], labels: list[int]) -> Detections:
    return Detections(
        np.array(boxes, dtype=np.float64),
        np.array(scores, dtype=np.float64),
        np.array(labels, dtype=np.int64),
    )


def _gt(boxes: list[list[float]], labels: list[int]) -> GroundTruth:
    return GroundTruth(np.array(boxes, dtype=np.float64), np.array(labels, dtype=np.int64))


def _mixed() -> tuple[dict[str, Detections], dict[str, GroundTruth]]:
    # three images hit, one misses (FP) -> mAP strictly between 0 and 1
    preds = {
        "1": _det([BOX], [0.9], [0]),
        "2": _det([BOX], [0.9], [0]),
        "3": _det([WRONG], [0.9], [0]),
        "4": _det([BOX], [0.9], [0]),
    }
    gt = {k: _gt([BOX], [0]) for k in ("1", "2", "3", "4")}
    return preds, gt


def test_point_estimate_equals_frozen_evaluate() -> None:
    preds, gt = _mixed()
    ev = evaluate(preds, gt, [0])
    bm = bootstrap_metrics(preds, gt, [0], n_resamples=200, seed=1)
    assert abs(bm["map50"]["point"] - ev["map50"]) < 1e-12
    assert abs(bm["map50_95"]["point"] - ev["map50_95"]) < 1e-12


def test_ci_brackets_point_and_has_width() -> None:
    preds, gt = _mixed()
    m = bootstrap_metrics(preds, gt, [0], n_resamples=500, seed=1)["map50"]
    assert m["ci_low"] <= m["point"] <= m["ci_high"]
    assert m["ci_high"] > m["ci_low"]  # resampling a mixed slice produces a non-degenerate CI


def test_perfect_slice_ci_collapses() -> None:
    preds = {k: _det([BOX], [0.9], [0]) for k in ("1", "2", "3")}
    gt = {k: _gt([BOX], [0]) for k in ("1", "2", "3")}
    m = bootstrap_metrics(preds, gt, [0], n_resamples=200, seed=1)["map50"]
    assert m["point"] == 1.0
    assert m["ci_low"] == 1.0
    assert m["ci_high"] == 1.0


def test_deterministic_under_seed() -> None:
    preds, gt = _mixed()
    a = bootstrap_metrics(preds, gt, [0], n_resamples=200, seed=42)
    b = bootstrap_metrics(preds, gt, [0], n_resamples=200, seed=42)
    assert a == b


def test_per_class_metrics_and_ngt_present() -> None:
    preds, gt = _mixed()
    bm = bootstrap_metrics(preds, gt, [0], n_resamples=50, seed=0)
    assert "ap50_class_0" in bm
    assert "ap50_95_class_0" in bm
    assert bm["map50"]["n_gt"] == 4  # four GT boxes
    assert bm["ap50_class_0"]["n_gt"] == 4


def test_bootstrap_slices_covers_axes() -> None:
    preds, gt = _mixed()
    weather = {"1": "sunny", "2": "sunny", "3": "night", "4": "night"}
    camera = {"1": "cam_a", "2": "cam_a", "3": "cam_b", "4": "cam_b"}
    r = bootstrap_slices(preds, gt, [0], weather, camera, n_resamples=50, seed=0)
    assert set(r["per_weather"]) == {"sunny", "night"}
    assert set(r["per_camera"]) == {"cam_a", "cam_b"}
    assert set(r["per_scale"]) == {"small", "medium", "large"}
    assert "point" in r["overall"]["map50"]
