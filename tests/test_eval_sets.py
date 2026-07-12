"""Cross-dataset eval-set builders: CSV/COCO parsing, taxonomy routing to scored/ignore
boxes, xywh->xyxy, bounds clipping and the xyxy-vs-xywh format guard, deterministic sampling.
No cv2 / no archives — the pure layer only.
"""

import pytest

from src.data.eval_sets import (
    parse_mio_gt,
    parse_rainsnow,
    rainsnow_boxes,
    route_boxes,
    sample_ids,
)
from src.data.taxonomy import load as load_taxonomy

TAX = load_taxonomy()
IDS = {"car": 0, "bus": 1, "van_truck": 2}


def test_parse_mio_gt_groups_by_image() -> None:
    csv_text = "00000000,car,10,20,30,40\n00000000,bus,0,0,5,5\n00000001,pickup_truck,1,2,3,4\n"
    gt = parse_mio_gt(csv_text)
    assert set(gt) == {"00000000", "00000001"}
    assert gt["00000000"][0] == ("car", 10.0, 20.0, 30.0, 40.0)


def test_route_boxes_map_ignore_exclude() -> None:
    rows = [
        ("car", 10, 10, 50, 50),  # -> scored car
        ("articulated_truck", 0, 0, 20, 20),  # -> scored van_truck
        ("motorized_vehicle", 5, 5, 15, 15),  # -> ignore
        ("pedestrian", 1, 1, 2, 2),  # -> excluded, dropped
    ]
    scored, ignore = route_boxes(TAX, "mio-tcd", rows, IDS, 640.0, 480.0)
    assert sorted(b[4] for b in scored) == [IDS["car"], IDS["van_truck"]]
    assert len(ignore) == 1
    assert ignore[0] == (5.0, 5.0, 15.0, 15.0)


def test_route_boxes_clips_and_drops_degenerate() -> None:
    rows = [
        ("car", -5, -5, 700, 500),  # clips to image
        ("car", 100, 100, 100, 200),  # zero width -> dropped
    ]
    scored, _ = route_boxes(TAX, "mio-tcd", rows, IDS, 640.0, 480.0)
    assert len(scored) == 1
    assert scored[0][:4] == (0.0, 0.0, 640.0, 480.0)


def test_route_boxes_rejects_xywh_misread() -> None:
    # a box whose "x2" is far past the width signals the CSV was misread as xywh
    with pytest.raises(ValueError, match="out of bounds"):
        route_boxes(TAX, "mio-tcd", [("car", 10, 10, 2000, 20)], IDS, 640.0, 480.0)


def test_sample_ids_deterministic_and_capped() -> None:
    ids = [f"{i:08d}" for i in range(100)]
    a = sample_ids(ids, 10, 1337)
    b = sample_ids(ids, 10, 1337)
    assert a == b and len(a) == 10 and a == sorted(a)
    assert sample_ids(ids[:5], 10, 1337) == sorted(ids[:5])  # fewer than target -> all


def test_rainsnow_bbox_xywh_to_xyxy_and_exclude() -> None:
    coco = {
        "images": [
            {"id": 0, "width": 640, "height": 480, "file_name": "Egensevej/E-1/cam1-00055.png"}
        ],
        "annotations": [
            {"image_id": 0, "category_id": 3, "bbox": [100, 50, 20, 30]},  # car
            {"image_id": 0, "category_id": 8, "bbox": [200, 60, 40, 40]},  # truck -> van_truck
            {"image_id": 0, "category_id": 1, "bbox": [0, 0, 5, 5]},  # person -> excluded
        ],
    }
    by_img, anns = parse_rainsnow(coco)
    scored, ignore = rainsnow_boxes(TAX, anns[0], IDS, 640.0, 480.0)
    assert (100.0, 50.0, 120.0, 80.0, IDS["car"]) in scored  # x2=x+w, y2=y+h
    assert (200.0, 60.0, 240.0, 100.0, IDS["van_truck"]) in scored
    assert len(scored) == 2 and ignore == []  # person dropped, no ignore class for rainsnow
