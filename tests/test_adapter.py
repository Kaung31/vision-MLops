"""YOLO->harness adapter: canonical class mapping and array->Detections, without torch.

The mapping resolves canonical ids through taxonomy.yaml, so these also pin that car=0,
bus=1, van_truck=2 wiring end-to-end.
"""

import numpy as np

from src.eval.adapter import build_class_map, canonical_ids, detections_from_arrays

# COCO-80 slice ultralytics exposes as model.names (0-based), enough to exercise the map.
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def test_build_class_map_only_scored_classes() -> None:
    ids = canonical_ids()
    cmap = build_class_map(COCO_NAMES)
    assert cmap == {2: ids["car"], 5: ids["bus"], 7: ids["van_truck"]}
    # person/bicycle/motorcycle are not scored -> absent -> dropped, never a false positive
    assert 0 not in cmap and 1 not in cmap and 3 not in cmap


def test_detections_from_arrays_filters_and_relabels() -> None:
    cmap = build_class_map(COCO_NAMES)
    boxes = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 3, 3]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cls = np.array([2, 7, 0], dtype=np.int64)  # car, truck, person(drop)
    det = detections_from_arrays(boxes, scores, cls, cmap)
    assert det.labels.tolist() == [canonical_ids()["car"], canonical_ids()["van_truck"]]
    assert np.allclose(det.scores, [0.9, 0.8])
    assert det.boxes.shape == (2, 4)
    assert det.boxes.dtype == np.float64


def test_empty_and_all_dropped() -> None:
    cmap = build_class_map(COCO_NAMES)
    empty = detections_from_arrays(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64), cmap)
    assert len(empty.boxes) == 0 and len(empty.labels) == 0
    # a frame with only unscored detections -> no detections
    allp = detections_from_arrays(
        np.array([[0, 0, 1, 1]], dtype=np.float32),
        np.array([0.5]),
        np.array([0], dtype=np.int64),
        cmap,
    )
    assert len(allp.boxes) == 0
