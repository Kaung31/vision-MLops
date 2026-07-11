"""YOLO conversion helpers (no cv2): label normalization, class routing, dedup."""

import xml.etree.ElementTree as ET

from src.data.convert import canonical_ids, frame_yolo_lines, to_yolo_line
from src.data.taxonomy import load as load_taxonomy

FRAME = """<frame num="1"><target_list>
  <target id="1">
    <box left="0" top="0" width="96" height="54"/><attribute vehicle_type="others"/>
  </target>
  <target id="2">
    <box left="100" top="100" width="50" height="50"/><attribute vehicle_type="car"/>
  </target>
  <target id="3">
    <box left="0.2" top="0.1" width="96" height="54"/><attribute vehicle_type="others"/>
  </target>
</target_list></frame>"""


def test_to_yolo_line_full_frame() -> None:
    assert to_yolo_line(2, (0, 0, 960, 540)) == "2 0.500000 0.500000 1.000000 1.000000"


def test_to_yolo_line_centered_small_box() -> None:
    assert to_yolo_line(0, (432, 243, 96, 54)) == "0 0.500000 0.500000 0.100000 0.100000"


def test_to_yolo_line_clips_out_of_bounds() -> None:
    parts = to_yolo_line(1, (900, 500, 400, 400)).split()
    assert all(0.0 <= float(p) <= 1.0 for p in parts[1:])


def test_canonical_ids() -> None:
    assert canonical_ids(load_taxonomy()) == {"car": 0, "bus": 1, "van_truck": 2}


def test_frame_yolo_lines_routes_and_dedups() -> None:
    tax = load_taxonomy()
    ids = canonical_ids(tax)
    lines = frame_yolo_lines(ET.fromstring(FRAME), tax, ids)
    # target 3 near-duplicates target 1 (both 'others') -> dedup drops it; 2 lines remain
    assert len(lines) == 2
    assert lines[0].startswith("2 ")  # others -> van_truck (id 2)
    assert lines[1].startswith("0 ")  # car -> id 0
