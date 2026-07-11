"""Known-answer tests for the eval harness — the ruler must be provably correct before
it is frozen. Every number here is pencil-computed, not read off the implementation.
"""

import numpy as np

from src.eval.harness import (
    Detections,
    GroundTruth,
    average_precision,
    box_ioa,
    box_iou,
    evaluate,
)


def _f(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _det(boxes: list[list[float]], scores: list[float], labels: list[int]) -> Detections:
    return Detections(
        _f(boxes), np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int64)
    )


def _gt(
    boxes: list[list[float]], labels: list[int], ignore: list[list[float]] | None = None
) -> GroundTruth:
    ig = _f(ignore) if ignore else np.zeros((0, 4), dtype=np.float64)
    return GroundTruth(_f(boxes), np.array(labels, dtype=np.int64), ig)


def test_box_iou_known_values() -> None:
    iou = box_iou(_f([[0, 0, 10, 10]]), _f([[0, 0, 10, 10], [5, 0, 15, 10]]))
    assert iou[0, 0] == 1.0
    assert abs(iou[0, 1] - 1 / 3) < 1e-9  # inter 50 / union 150


def test_box_ioa_known_values() -> None:
    ioa = box_ioa(_f([[0, 0, 10, 10], [2, 2, 4, 4]]), _f([[0, 0, 5, 10]]))
    assert ioa[0, 0] == 0.5  # half the 10x10 detection sits in the region
    assert ioa[1, 0] == 1.0  # the 2x2 detection is fully inside


def test_average_precision_half() -> None:
    # a higher-scored FP before a TP, one GT -> precision-recall gives AP exactly 0.5
    ap = average_precision(np.array([0.9, 0.8]), np.array([False, True]), n_gt=1)
    assert abs(ap - 0.5) < 1e-9


def test_average_precision_perfect() -> None:
    ap = average_precision(np.array([0.9, 0.8]), np.array([True, True]), n_gt=2)
    assert ap == 1.0


def test_average_precision_no_gt_is_nan() -> None:
    assert np.isnan(average_precision(np.array([0.9]), np.array([True]), n_gt=0))


def test_evaluate_perfect_detection() -> None:
    preds = {"a": _det([[0, 0, 10, 10]], [0.9], [0])}
    gt = {"a": _gt([[0, 0, 10, 10]], [0])}
    r = evaluate(preds, gt, class_ids=[0])
    assert r["map50"] == 1.0
    assert r["map50_95"] == 1.0


def test_ignore_region_excludes_false_positive() -> None:
    # a high-scored FP at [20,20,30,30] plus a TP at [0,0,10,10]
    preds = {"a": _det([[20, 20, 30, 30], [0, 0, 10, 10]], [0.9, 0.8], [0, 0])}
    without_ignore = {"a": _gt([[0, 0, 10, 10]], [0])}
    with_ignore = {"a": _gt([[0, 0, 10, 10]], [0], ignore=[[20, 20, 30, 30]])}
    # the FP drags mAP50 to 0.5; declaring its area an ignore region restores 1.0
    assert abs(evaluate(preds, without_ignore, [0])["map50"] - 0.5) < 1e-9
    assert abs(evaluate(preds, with_ignore, [0])["map50"] - 1.0) < 1e-9
