"""Convert UA-DETRAC -> YOLO via the taxonomy; emit the three DVC dataset versions.

  train-v1        train + val, every-10th sampling, ignore-regions black-filled, + data.yaml
  eval-frozen-v1  test split, sampled + masked + ignore-region JSONs (FP exclusion). FROZEN
  prod-holdout-v1 full-rate frames REFERENCED from raw-v1 (not duplicated) via a
                  content-addressed manifest, + sequestered labels + ignore JSONs. FROZEN

Reuses hygiene (dedup, black-fill, ignore export), sample (every-10th), taxonomy (routing).
Needs the `data` group (cv2). Run: uv run --group data python -m src.data.convert
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import yaml

from src.data.datasets import DatasetDir, mark_readonly, open_dataset_dir
from src.data.hygiene import (
    Box,
    IgnoreBox,
    dedup_frame,
    frame_targets,
    ignore_regions_json,
    ignore_regions_of,
)
from src.data.sample import select_every_nth
from src.data.taxonomy import load as load_taxonomy
from src.data.taxonomy import route

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_ZIP = REPO_ROOT / "data" / "raw" / "ua-detrac" / "ua-detrac-orig.zip"
ANNO_ZIP = REPO_ROOT / "data" / "raw" / "ua-detrac" / "DETRAC-Train_Annotations-XML.zip"
SPLITS_YAML = REPO_ROOT / "configs" / "splits.yaml"
PROCESSED = REPO_ROOT / "data" / "processed"

IMAGE_W, IMAGE_H = 960, 540
IMAGE_ROOT = "DETRAC-Images/DETRAC-Images"  # frame dir inside the images zip
ANNO_ROOT = "DETRAC-Train-Annotations-XML"  # inside the annotations zip
FROZEN_REASON = "frozen eval set — see ADR 0006; no writer may modify (Law 4)"

# Anchors prod-holdout-v1 to an immutable raw-v1 (ADR 0006). Update only with raw-v1.
RAW_SOURCE = {
    "dvc_path": "data/raw/ua-detrac",
    "dvc_md5": "8e5708b5f7140c98baa14c0517b1dd39.dir",
    "git_tag": "raw-v1",
    "images_archive": "ua-detrac-orig.zip",
    "image_root_in_archive": IMAGE_ROOT,
}


def _clip01(v: float) -> float:
    return min(1.0, max(0.0, v))


def to_yolo_line(cls_id: int, box: Box, w: int = IMAGE_W, h: int = IMAGE_H) -> str:
    """One YOLO label line: class + normalized (cx, cy, w, h), clipped to [0, 1]."""
    left, top, bw, bh = box
    return (
        f"{cls_id} {_clip01((left + bw / 2) / w):.6f} {_clip01((top + bh / 2) / h):.6f} "
        f"{_clip01(bw / w):.6f} {_clip01(bh / h):.6f}"
    )


def canonical_ids(taxonomy: dict[str, Any]) -> dict[str, int]:
    return {str(c["name"]): int(c["id"]) for c in taxonomy["canonical"]}


def frame_yolo_lines(frame: ET.Element, taxonomy: dict[str, Any], ids: dict[str, int]) -> list[str]:
    """YOLO lines for one frame: dedup boxes, route native vehicle_type -> canonical id."""
    kept, _ = dedup_frame(frame_targets(frame))
    lines: list[str] = []
    for native, box in kept:
        verb, canonical = route(taxonomy, "ua-detrac", native)
        if verb == "map" and canonical is not None:
            lines.append(to_yolo_line(ids[canonical], box))
    return lines


def _mask_jpg(jpg_bytes: bytes, ignore: list[IgnoreBox]) -> bytes:
    import cv2  # lazy: keeps module import cv2-free so CI can import/test pure helpers
    import numpy as np

    from src.data.hygiene import black_fill

    img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed")
    ok, buf = cv2.imencode(".jpg", black_fill(img, ignore))
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return bytes(buf.tobytes())


def _seq_root(anno: zipfile.ZipFile, seq: str) -> ET.Element:
    return ET.fromstring(anno.read(f"{ANNO_ROOT}/{seq}.xml"))


def _frames_by_num(root: ET.Element) -> dict[int, ET.Element]:
    return {int(f.attrib["num"]): f for f in root.iter("frame")}


def _data_yaml(names: list[str], split_paths: dict[str, str]) -> str:
    doc: dict[str, Any] = {"path": ".", "names": dict(enumerate(names))}
    doc.update(split_paths)
    return yaml.safe_dump(doc, sort_keys=False)


def _write_masked_frames(
    dd: DatasetDir,
    split: str,
    seqs: list[str],
    anno: zipfile.ZipFile,
    images: zipfile.ZipFile,
    taxonomy: dict[str, Any],
    ids: dict[str, int],
    sample_it: bool,
    write_ignore_json: bool,
) -> int:
    written = 0
    for seq in seqs:
        root = _seq_root(anno, seq)
        ignore = ignore_regions_of(root)
        frames = _frames_by_num(root)
        nums = select_every_nth(sorted(frames)) if sample_it else sorted(frames)
        for num in nums:
            stem = f"{seq}_img{num:05d}"
            lines = frame_yolo_lines(frames[num], taxonomy, ids)
            dd.write_text(f"labels/{split}/{stem}.txt", "\n".join(lines) + "\n")
            jpg = images.read(f"{IMAGE_ROOT}/{seq}/img{num:05d}.jpg")
            dd.write_bytes(f"images/{split}/{stem}.jpg", _mask_jpg(jpg, ignore))
            written += 1
        if write_ignore_json:
            dd.write_text(
                f"ignore_regions/{seq}.json", ignore_regions_json(seq, ignore, IMAGE_W, IMAGE_H)
            )
    return written


def _zip_frame_counts(images: zipfile.ZipFile, seqs: list[str]) -> dict[str, int]:
    want = set(seqs)
    counts = dict.fromkeys(seqs, 0)
    for name in images.namelist():
        if not name.endswith(".jpg"):
            continue
        seq = next((p for p in name.split("/") if p.startswith("MVI_")), None)
        if seq in want:
            counts[seq] += 1
    return counts


def build_train_v1(
    root: Path,
    manifests: dict[str, list[str]],
    anno: zipfile.ZipFile,
    images: zipfile.ZipFile,
    taxonomy: dict[str, Any],
    ids: dict[str, int],
    names: list[str],
) -> None:
    dd = open_dataset_dir(root, "w")
    for split in ("train", "val"):
        n = _write_masked_frames(
            dd, split, manifests[split], anno, images, taxonomy, ids, True, False
        )
        print(f"  train-v1/{split}: {n} frames")
    dd.write_text("data.yaml", _data_yaml(names, {"train": "images/train", "val": "images/val"}))


def build_eval_frozen_v1(
    root: Path,
    manifests: dict[str, list[str]],
    anno: zipfile.ZipFile,
    images: zipfile.ZipFile,
    taxonomy: dict[str, Any],
    ids: dict[str, int],
    names: list[str],
) -> None:
    dd = open_dataset_dir(root, "w")
    n = _write_masked_frames(dd, "test", manifests["test"], anno, images, taxonomy, ids, True, True)
    print(f"  eval-frozen-v1/test: {n} frames")
    dd.write_text("data.yaml", _data_yaml(names, {"val": "images/test"}))
    mark_readonly(root, FROZEN_REASON)


def build_prod_holdout_v1(
    root: Path,
    manifests: dict[str, list[str]],
    anno: zipfile.ZipFile,
    images: zipfile.ZipFile,
    taxonomy: dict[str, Any],
    ids: dict[str, int],
) -> None:
    dd = open_dataset_dir(root, "w")
    seqs = manifests["prod_holdout"]
    counts = _zip_frame_counts(images, seqs)
    manifest: dict[str, Any] = {"raw_source": RAW_SOURCE, "sequences": {}}
    labeled = 0
    for seq in seqs:
        root_el = _seq_root(anno, seq)
        ignore = ignore_regions_of(root_el)
        dd.write_text(
            f"ignore_regions/{seq}.json", ignore_regions_json(seq, ignore, IMAGE_W, IMAGE_H)
        )
        for num, frame in _frames_by_num(root_el).items():
            dd.write_text(
                f"labels/{seq}/img{num:05d}.txt",
                "\n".join(frame_yolo_lines(frame, taxonomy, ids)) + "\n",
            )
            labeled += 1
        manifest["sequences"][seq] = {
            "frame_count": counts[seq],
            "archive_dir": f"{IMAGE_ROOT}/{seq}",
        }
    dd.write_text("frame_manifest.yaml", yaml.safe_dump(manifest, sort_keys=False))
    total = sum(counts.values())
    print(f"  prod-holdout-v1: {len(seqs)} seqs, {labeled} labeled frames, {total} full-rate refs")
    mark_readonly(root, FROZEN_REASON)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=PROCESSED)
    args = parser.parse_args(argv)

    manifests: dict[str, list[str]] = yaml.safe_load(SPLITS_YAML.read_text())["manifests"]
    taxonomy = load_taxonomy()
    ids = canonical_ids(taxonomy)
    names = [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])]

    with zipfile.ZipFile(ANNO_ZIP) as anno, zipfile.ZipFile(IMAGES_ZIP) as images:
        build_train_v1(args.out / "train-v1", manifests, anno, images, taxonomy, ids, names)
        build_eval_frozen_v1(
            args.out / "eval-frozen-v1", manifests, anno, images, taxonomy, ids, names
        )
        build_prod_holdout_v1(args.out / "prod-holdout-v1", manifests, anno, images, taxonomy, ids)
    print("conversion complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
