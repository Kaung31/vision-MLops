"""UA-DETRAC annotation hygiene: ignore-regions, dedup, and per-sequence audit.

Three jobs, one module (guide Phase 1 steps 2-3):
  * ignore regions -> black_fill() training copies + export_ignore_regions() JSON for eval;
  * dedup_frame() removes UA-DETRAC's duplicate-box defect (same-class, high IoU, per frame);
  * audit_sequence()/build_audit() produce the committed before/after audit report.

Ignore regions are axis-aligned rectangles (DETRAC ``<ignored_region><box .../></>``),
kept as float xywh boxes; the export header states the coordinate convention explicitly.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]

# Export contract. Bump SCHEMA_VERSION on any breaking change to the JSON shape/meaning.
SCHEMA_VERSION = 1
# left,top,width,height in pixels; origin top-left; +x right, +y down; right/bottom edges
# EXCLUSIVE; values are floats (as DETRAC stores them).
COORD_CONVENTION = "xywh_px_topleft_origin_rb_exclusive_float"

# UA-DETRAC frames are 960x540 (verified against the data).
UA_DETRAC_IMAGE_WIDTH = 960
UA_DETRAC_IMAGE_HEIGHT = 540
# Dedup IoU cut, calibrated on raw-v1: same-class pairs cluster at IoU>=0.9 (true dupes),
# well separated from legitimate occlusions (<0.7). See docs/audits/annotation-audit.md.
DEDUP_IOU_THRESHOLD = 0.9

Box = tuple[float, float, float, float]  # left, top, width, height


@dataclass(frozen=True)
class IgnoreBox:
    left: float
    top: float
    width: float
    height: float


def _box_to_pixels(b: IgnoreBox, w: int, h: int) -> tuple[int, int, int, int]:
    """Float box -> integer pixel range: top-left floored, bottom-right ceiled, clipped.

    Half-open [x0:x1, y0:y1] (right/bottom exclusive). Over-covers by up to a pixel at
    fractional edges — deliberate, so ignore regions never under-mask.
    """
    x0 = max(0, floor(b.left))
    y0 = max(0, floor(b.top))
    x1 = min(w, ceil(b.left + b.width))
    y1 = min(h, ceil(b.top + b.height))
    return x0, y0, x1, y1


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

    Pixel convention is pinned by tests (see ``_box_to_pixels``); under-masking would leak
    unlabeled vehicles into training, so we round outward.
    """
    h, w = int(image.shape[0]), int(image.shape[1])
    out = image.copy()
    for b in boxes:
        x0, y0, x1, y1 = _box_to_pixels(b, w, h)
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
    safe to consume blind in Phase 2 — no out-of-band xywh/xyxy or origin assumptions.
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


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two xywh boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def frame_targets(frame: ET.Element) -> list[tuple[str, Box]]:
    """Extract (native_vehicle_type, xywh box) for every target in a DETRAC frame."""
    out: list[tuple[str, Box]] = []
    # targets are nested under <target_list>, so iterate descendants, not direct children
    for tgt in frame.iter("target"):
        box = tgt.find("box")
        attr = tgt.find("attribute")
        if box is None or attr is None:
            continue
        out.append(
            (
                attr.get("vehicle_type", "?"),
                (
                    float(box.attrib["left"]),
                    float(box.attrib["top"]),
                    float(box.attrib["width"]),
                    float(box.attrib["height"]),
                ),
            )
        )
    return out


def dedup_frame(
    labeled: list[tuple[str, Box]], iou_thr: float = DEDUP_IOU_THRESHOLD
) -> tuple[list[tuple[str, Box]], int]:
    """Drop same-class boxes that near-duplicate an already-kept box (IoU >= threshold).

    Returns (kept, n_removed). Cross-class overlaps are left alone here — a same-geometry
    different-class pair is a label conflict, reported separately, not silently deleted.
    """
    kept: list[tuple[str, Box]] = []
    removed = 0
    for cls, box in labeled:
        if any(c == cls and iou(box, kb) >= iou_thr for c, kb in kept):
            removed += 1
        else:
            kept.append((cls, box))
    return kept, removed


def count_cross_class_overlaps(
    labeled: list[tuple[str, Box]], iou_thr: float = DEDUP_IOU_THRESHOLD
) -> int:
    """Count different-class box pairs that overlap at IoU >= threshold (label conflicts)."""
    n = 0
    for i in range(len(labeled)):
        ci, bi = labeled[i]
        for j in range(i + 1, len(labeled)):
            cj, bj = labeled[j]
            if ci != cj and iou(bi, bj) >= iou_thr:
                n += 1
    return n


def audit_sequence(
    xml_path: Path,
    image_width: int = UA_DETRAC_IMAGE_WIDTH,
    image_height: int = UA_DETRAC_IMAGE_HEIGHT,
    iou_thr: float = DEDUP_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Per-sequence audit: box counts before/after dedup, class histogram, ignore fraction."""
    root = ET.parse(xml_path).getroot()
    sa = root.find("sequence_attribute")
    weather = sa.get("sence_weather", "?") if sa is not None else "?"

    ignore = parse_ignore_regions(xml_path)
    mask = np.zeros((image_height, image_width), dtype=bool)
    for b in ignore:
        x0, y0, x1, y1 = _box_to_pixels(b, image_width, image_height)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True

    frames = before = after = removed = cross = 0
    hist: Counter[str] = Counter()
    for frame in root.iter("frame"):
        frames += 1
        labeled = frame_targets(frame)
        before += len(labeled)
        kept, rem = dedup_frame(labeled, iou_thr)
        after += len(kept)
        removed += rem
        cross += count_cross_class_overlaps(labeled, iou_thr)
        for c, _ in kept:
            hist[c] += 1

    return {
        "sequence": str(root.get("name", xml_path.stem)),
        "weather": weather,
        "frames": frames,
        "boxes_before": before,
        "boxes_after": after,
        "removed_duplicates": removed,
        "cross_class_overlaps": cross,
        "class_histogram": dict(hist),
        "ignore_region_count": len(ignore),
        "ignore_area_fraction": round(float(mask.mean()), 5),
    }


def build_audit(
    xml_dir: Path,
    image_width: int = UA_DETRAC_IMAGE_WIDTH,
    image_height: int = UA_DETRAC_IMAGE_HEIGHT,
    iou_thr: float = DEDUP_IOU_THRESHOLD,
) -> dict[str, Any]:
    """Audit every sequence XML in ``xml_dir`` and aggregate."""
    seqs = [
        audit_sequence(p, image_width, image_height, iou_thr) for p in sorted(xml_dir.glob("*.xml"))
    ]
    agg = {
        "sequences": len(seqs),
        "frames": sum(s["frames"] for s in seqs),
        "boxes_before": sum(s["boxes_before"] for s in seqs),
        "boxes_after": sum(s["boxes_after"] for s in seqs),
        "removed_duplicates": sum(s["removed_duplicates"] for s in seqs),
        "cross_class_overlaps": sum(s["cross_class_overlaps"] for s in seqs),
        "iou_threshold": iou_thr,
        "image_size": [image_width, image_height],
    }
    return {"aggregate": agg, "sequences": seqs}


def _render_markdown(report: dict[str, Any]) -> str:
    agg = report["aggregate"]
    w, h = agg["image_size"]
    lines = [
        "# UA-DETRAC annotation audit",
        "",
        f"Generated by `src/data/hygiene.py` over the {agg['sequences']} UA-DETRAC train "
        f"sequences (raw-v1). Image {w}x{h}; dedup IoU threshold **{agg['iou_threshold']}** "
        "(calibrated: same-class near-duplicates cluster at IoU>=0.9, cleanly separated "
        "from legitimate occlusions <0.7).",
        "",
        "## Totals",
        "",
        f"- sequences: **{agg['sequences']}**, frames: **{agg['frames']}**",
        f"- boxes before dedup: **{agg['boxes_before']}**, after: **{agg['boxes_after']}**, "
        f"duplicates removed: **{agg['removed_duplicates']}**",
        f"- cross-class high-overlap conflicts (flagged, NOT removed): "
        f"**{agg['cross_class_overlaps']}**",
        "",
        "## Per sequence",
        "",
        "| sequence | weather | frames | before | after | removed | cross | ignore% | histogram |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in report["sequences"]:
        hist = " ".join(f"{k}:{v}" for k, v in sorted(s["class_histogram"].items()))
        lines.append(
            f"| {s['sequence']} | {s['weather']} | {s['frames']} | {s['boxes_before']} | "
            f"{s['boxes_after']} | {s['removed_duplicates']} | {s['cross_class_overlaps']} | "
            f"{100 * s['ignore_area_fraction']:.2f} | {hist} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the UA-DETRAC annotation audit.")
    parser.add_argument("--xml-dir", type=Path, required=True, help="dir of DETRAC sequence XMLs")
    parser.add_argument(
        "--out-md", type=Path, default=REPO_ROOT / "docs/audits/annotation-audit.md"
    )
    parser.add_argument(
        "--out-json", type=Path, default=REPO_ROOT / "docs/audits/annotation-audit.json"
    )
    parser.add_argument("--width", type=int, default=UA_DETRAC_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=UA_DETRAC_IMAGE_HEIGHT)
    parser.add_argument("--iou", type=float, default=DEDUP_IOU_THRESHOLD)
    args = parser.parse_args(argv)

    report = build_audit(args.xml_dir, args.width, args.height, args.iou)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    args.out_md.write_text(_render_markdown(report))
    agg = report["aggregate"]
    print(
        f"audit: {agg['sequences']} seqs, {agg['boxes_before']}->{agg['boxes_after']} boxes "
        f"({agg['removed_duplicates']} dupes removed) -> {args.out_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
