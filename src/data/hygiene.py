"""UA-DETRAC ignore-region hygiene: parse, black-fill (training), export (eval).

UA-DETRAC marks per-sequence "ignored regions" — rectangles holding vehicles too
low-resolution to label. One source, two consumers:
  * black_fill() zeroes those rectangles on TRAINING image copies (guide 2a), so the
    model never learns "background" on real-but-unlabeled vehicles.
  * export_ignore_regions() writes a self-describing JSON per sequence (guide 2b), so the
    Phase 2 eval harness can exclude detections inside them from false-positive counts.

Ignore regions are axis-aligned rectangles (DETRAC ``<ignored_region><box .../></>``),
not free-form polygons; we keep them as float xywh boxes and say so in the export header.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Export contract. Bump SCHEMA_VERSION on any breaking change to the JSON shape/meaning.
SCHEMA_VERSION = 1
# left,top,width,height in pixels; origin top-left; +x right, +y down; right/bottom edges
# EXCLUSIVE; values are floats (as DETRAC stores them).
COORD_CONVENTION = "xywh_px_topleft_origin_rb_exclusive_float"


@dataclass(frozen=True)
class IgnoreBox:
    left: float
    top: float
    width: float
    height: float


def parse_ignore_regions(xml_path: Path) -> list[IgnoreBox]:
    """Return the ignore-region boxes for one DETRAC sequence XML.

    Returns an EMPTY list when the sequence has no ``<ignored_region>`` element — this is
    common and must never raise (it is the case most likely to blow up mid-conversion).
    """
    root = ET.parse(xml_path).getroot()
    region = root.find("ignored_region")
    if region is None:
        return []
    return [
        IgnoreBox(
            left=float(box.attrib["left"]),
            top=float(box.attrib["top"]),
            width=float(box.attrib["width"]),
            height=float(box.attrib["height"]),
        )
        for box in region.findall("box")
    ]


def black_fill(image: NDArray[np.uint8], boxes: list[IgnoreBox]) -> NDArray[np.uint8]:
    """Return a copy of ``image`` with every ignore box zeroed (black).

    Pixel convention (pinned by tests): top-left floored, bottom-right ceiled, clipped to
    the image, filled as the half-open slice ``[y0:y1, x0:x1]``. This over-covers by up to
    a pixel at fractional edges — deliberate: under-masking would leak unlabeled vehicles
    into training, which is worse than blacking one extra edge pixel.
    """
    h, w = int(image.shape[0]), int(image.shape[1])
    out = image.copy()
    for b in boxes:
        x0 = max(0, floor(b.left))
        y0 = max(0, floor(b.top))
        x1 = min(w, ceil(b.left + b.width))
        y1 = min(h, ceil(b.top + b.height))
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = 0
    return out


def export_ignore_regions(
    sequence: str,
    boxes: list[IgnoreBox],
    image_width: int,
    image_height: int,
    out_path: Path,
) -> None:
    """Write a self-describing ignore-region JSON for one sequence (guide 2b).

    The header (schema version, coordinate convention, image dimensions) makes the file
    safe to consume blind in Phase 2 — no out-of-band assumptions about xywh/xyxy or origin.
    """
    payload = {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "coordinate_convention": COORD_CONVENTION,
        "image_width": image_width,
        "image_height": image_height,
        "ignore_regions": [
            {"left": b.left, "top": b.top, "width": b.width, "height": b.height} for b in boxes
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
