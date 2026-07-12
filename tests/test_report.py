"""Report core, torch-free: YOLO-label -> GroundTruth denorm, ignore-region parse, the
point+CI merge driven through the real harness on synthetic detections, and the gap headline.
"""

import numpy as np

from src.eval.bootstrap import bootstrap_slices
from src.eval.harness import Detections, GroundTruth, evaluate_slices
from src.eval.report import _ignore_xyxy, _merge_point_ci, _yolo_to_gt, headline


def test_yolo_to_gt_denormalizes_to_annotation_space() -> None:
    # class 0, centre (480,270)px, size (96,108)px on 960x540 -> [432,216,528,324]
    gt = _yolo_to_gt("0 0.5 0.5 0.1 0.2\n", np.zeros((0, 4)))
    assert gt.labels.tolist() == [0]
    assert np.allclose(gt.boxes[0], [432.0, 216.0, 528.0, 324.0])
    assert gt.boxes.shape == (1, 4)


def test_yolo_to_gt_empty_is_shaped() -> None:
    gt = _yolo_to_gt("\n", np.zeros((0, 4)))
    assert gt.boxes.shape == (0, 4) and gt.labels.shape == (0,)


def test_ignore_xyxy(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "seq.json"
    p.write_text('{"ignore_regions": [{"left": 10, "top": 20, "width": 30, "height": 40}]}')
    boxes = _ignore_xyxy(p)
    assert np.allclose(boxes, [[10, 20, 40, 60]])  # x2=left+width, y2=top+height


def test_merge_point_ci_through_real_harness() -> None:
    box = [[10.0, 10.0, 50.0, 50.0]]
    gt = {
        "a": GroundTruth(np.array(box), np.array([0])),
        "b": GroundTruth(np.array(box), np.array([0])),
    }
    preds = {k: Detections(np.array(box), np.array([0.9]), np.array([0])) for k in gt}
    weather = {"a": "sunny", "b": "sunny"}
    camera = {"a": "cam1", "b": "cam1"}
    point = evaluate_slices(preds, gt, [0], weather, camera)
    ci = bootstrap_slices(preds, gt, [0], weather, camera, n_resamples=50, seed=0)
    merged = _merge_point_ci(point, ci)

    ov = merged["overall"]["metrics"]
    assert "spec" in merged
    assert abs(ov["map50"]["point"] - 1.0) < 1e-9  # perfect detections
    assert ov["map50"]["ci_low"] <= ov["map50"]["point"] <= ov["map50"]["ci_high"] + 1e-9
    assert merged["overall"]["n_gt"] == 2
    assert merged["per_weather"]["sunny"]["metrics"]["map50"]["point"] == 1.0


def test_headline_gap_is_size_weighted() -> None:
    def suite(role: str, m: float, n_gt: int) -> dict:  # type: ignore[type-arg]
        return {
            "role": role,
            "slices": {"overall": {"n_gt": n_gt, "metrics": {"map50_95": {"point": m}}}},
        }

    reports = {
        "ua-detrac": suite("in-distribution", 0.40, 100),
        "mio-tcd": suite("gated-eval", 0.30, 300),
        "rainsnow": suite("gated-eval", 0.10, 100),
    }
    h = headline(reports)
    # weighted gated = (0.30*300 + 0.10*100) / 400 = 0.25 ; gap = 0.40 - 0.25
    assert abs(h["gated_aggregate_map50_95"] - 0.25) < 1e-9
    assert abs(h["gap"] - 0.15) < 1e-9
    assert abs(h["per_suite"]["rainsnow"]["gap"] - 0.30) < 1e-9


def test_headline_empty_without_both_roles() -> None:
    only_gated = {"mio-tcd": {"role": "gated-eval", "slices": {"overall": {"n_gt": 1}}}}
    assert headline(only_gated) == {}
