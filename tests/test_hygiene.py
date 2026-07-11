"""Ignore-region hygiene: parse (incl. empty), black-fill (incl. clip + float rounding),
and the self-describing export header. Hand-checked fixtures.
"""

import json
from pathlib import Path

import numpy as np

from src.data.hygiene import (
    COORD_CONVENTION,
    SCHEMA_VERSION,
    IgnoreBox,
    black_fill,
    export_ignore_regions,
    parse_ignore_regions,
)

FIXTURE_WITH = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_test">
   <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
   <ignored_region>
      <box left="2.0" top="3.0" width="4.0" height="5.0"/>
      <box left="778.75" top="24.75" width="181.75" height="63.5"/>
   </ignored_region>
   <frame num="1"><target_list></target_list></frame>
</sequence>
"""

# A real DETRAC case: some sequences have NO <ignored_region> element at all.
FIXTURE_WITHOUT = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_none">
   <sequence_attribute camera_state="stable" sence_weather="night"/>
   <frame num="1"><target_list></target_list></frame>
</sequence>
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def test_parse_returns_exact_boxes(tmp_path: Path) -> None:
    boxes = parse_ignore_regions(_write(tmp_path, "with.xml", FIXTURE_WITH))
    assert boxes == [
        IgnoreBox(2.0, 3.0, 4.0, 5.0),
        IgnoreBox(778.75, 24.75, 181.75, 63.5),
    ]


def test_parse_no_ignored_region_returns_empty(tmp_path: Path) -> None:
    # Must return [] and NOT crash — the failure mode most likely to bite at conversion.
    assert parse_ignore_regions(_write(tmp_path, "none.xml", FIXTURE_WITHOUT)) == []


def test_black_fill_inside_black_outside_untouched_and_copies() -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    out = black_fill(img, [IgnoreBox(2.0, 3.0, 4.0, 5.0)])  # x:[2,6) y:[3,8)
    assert (out[3:8, 2:6] == 0).all()
    assert (out[0:3, :] == 255).all()
    assert (out[:, 0:2] == 255).all()
    assert (img == 255).all()  # input not mutated


def test_black_fill_clips_partial_out_of_bounds() -> None:
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    out = black_fill(img, [IgnoreBox(8.0, 8.0, 5.0, 5.0)])  # would extend to 13, clip to 10
    assert (out[8:10, 8:10] == 0).all()
    assert out.shape == (10, 10, 3)  # no crash, no resize


def test_black_fill_float_rounding_is_pinned() -> None:
    # floor top-left, ceil bottom-right, half-open:
    # left=1.4 top=1.6 w=2.2 h=2.2 -> x0=1 y0=1 x1=ceil(3.6)=4 y1=ceil(3.8)=4 -> rows/cols 1,2,3
    img = np.full((6, 6), 255, dtype=np.uint8)
    out = black_fill(img, [IgnoreBox(1.4, 1.6, 2.2, 2.2)])
    assert (out[1:4, 1:4] == 0).all()
    assert out[0, 0] == 255  # floored edge: pixel 0 untouched
    assert out[4, 4] == 255  # ceiled edge is exclusive: pixel 4 untouched


def test_export_header_is_self_describing(tmp_path: Path) -> None:
    out = tmp_path / "MVI_test.json"
    export_ignore_regions("MVI_test", [IgnoreBox(2.0, 3.0, 4.0, 5.0)], 960, 540, out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["coordinate_convention"] == COORD_CONVENTION
    assert data["image_width"] == 960
    assert data["image_height"] == 540
    assert data["ignore_regions"] == [{"left": 2.0, "top": 3.0, "width": 4.0, "height": 5.0}]
