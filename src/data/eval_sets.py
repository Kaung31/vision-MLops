"""Build the two frozen cross-dataset eval sets from the local archives, via the taxonomy.

  mio-tcd-eval-v1   seeded ~2000-frame sample of MIO-TCD *train* (only split with public GT);
                    `motorized_vehicle` -> ignore-region (a correct detection there is not an FP).
  rainsnow-eval-v1  all annotated RGB frames of AAU RainSnow (pre-extracted PNGs in the zip; the
                    COCO json carries a bbox per instance, so no video decode, no mask->bbox).

Both are eval-only (never trained on) and written FROZEN (`.readonly`). Each is one directory:
`frames/<stem>.<ext>` + `annotations.json` (px xyxy boxes + per-image w/h + camera + ignore).
That compact manifest is exactly what the harness eats, so the report loads it with no
YOLO round-trip; the report unifies UA-DETRAC (YOLO on disk) and these at the GroundTruth
boundary, not on disk.

Pure parse/route helpers are cv2-free (CI-tested); frame extraction lazy-imports cv2 and needs
the `data` group + the raw archives.
Run: uv run --group data python -m src.data.eval_sets --dataset all
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.data.datasets import mark_readonly, open_dataset_dir
from src.data.taxonomy import canonical_names, route
from src.data.taxonomy import load as load_taxonomy

REPO_ROOT = Path(__file__).resolve().parents[2]
MIO_TAR = REPO_ROOT / "data" / "raw" / "mio-tcd" / "MIO-TCD-Localization.tar"
RAINSNOW_ZIP = REPO_ROOT / "data" / "raw" / "aau-rainsnow" / "aau-rainsnow.zip"
PROCESSED = REPO_ROOT / "data" / "processed"

MIO_MEMBER_PREFIX = "MIO-TCD-Localization"
RAINSNOW_RGB_JSON = "aauRainSnow-rgb.json"

SAMPLE_SEED = 1337
MIO_TARGET = 2000
FROZEN_REASON = "frozen cross-dataset eval set — Phase 2; no writer may modify (Law 4)"

Box5 = tuple[float, float, float, float, int]  # x1,y1,x2,y2,canonical_id
BoxI = tuple[float, float, float, float]  # ignore box xyxy


def _ids_map() -> dict[str, int]:
    tax = load_taxonomy()
    return {str(c["name"]): int(c["id"]) for c in tax["canonical"]}


# --- MIO-TCD --------------------------------------------------------------------------------


def parse_mio_gt(csv_text: str) -> dict[str, list[tuple[str, float, float, float, float]]]:
    """gt_train.csv -> {image_id: [(native_class, x1, y1, x2, y2), ...]}."""
    out: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
    for row in csv.reader(io.StringIO(csv_text)):
        if not row:
            continue
        img, cls, x1, y1, x2, y2 = row[0], row[1], *map(float, row[2:6])
        out[img].append((cls, x1, y1, x2, y2))
    return dict(out)


def route_boxes(
    tax: dict[str, Any],
    dataset: str,
    rows: Sequence[tuple[Any, float, float, float, float]],
    ids: dict[str, int],
    w: float,
    h: float,
) -> tuple[list[Box5], list[BoxI]]:
    """Split one image's native rows into (scored canonical boxes, ignore boxes) via taxonomy.

    Clips to the image, drops degenerate boxes, and fails loud if a box is grossly out of
    bounds (the xyxy-vs-xywh guard: a misread format would put x2 far past the width).
    """
    scored: list[Box5] = []
    ignore: list[BoxI] = []
    for native, x1, y1, x2, y2 in rows:
        # gross-violation guard (xyxy-vs-xywh misread); modest edge overhang clips silently
        if x2 > 1.5 * w or y2 > 1.5 * h or x1 < -0.5 * w or y1 < -0.5 * h:
            raise ValueError(f"{dataset}: box ({x1},{y1},{x2},{y2}) out of bounds for {w}x{h}")
        cx1, cy1 = max(0.0, min(x1, x2)), max(0.0, min(y1, y2))
        cx2, cy2 = min(w, max(x1, x2)), min(h, max(y1, y2))
        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
            continue
        verb, canonical = route(tax, dataset, native)
        if verb == "map" and canonical is not None:
            scored.append((cx1, cy1, cx2, cy2, ids[canonical]))
        elif verb == "ignore":
            ignore.append((cx1, cy1, cx2, cy2))
    return scored, ignore


def sample_ids(ids: list[str], target: int, seed: int) -> list[str]:
    """Deterministic seeded subset (or all, sorted, when fewer than target)."""
    ordered = sorted(ids)
    if len(ordered) <= target:
        return ordered
    rng = np.random.default_rng(seed)
    picked = rng.choice(np.array(ordered), size=target, replace=False)
    return sorted(picked.tolist())


def build_mio(out_root: Path) -> None:
    import cv2  # lazy: keeps the module importable + parse/route tests cv2-free in CI

    tax = load_taxonomy()
    ids = _ids_map()
    dd = open_dataset_dir(out_root, "w")
    with tarfile.open(MIO_TAR) as tar:
        gt_f = tar.extractfile(f"{MIO_MEMBER_PREFIX}/gt_train.csv")
        assert gt_f is not None, "gt_train.csv missing from MIO-TCD tar"
        gt = parse_mio_gt(gt_f.read().decode())
        chosen = sample_ids(list(gt), MIO_TARGET, SAMPLE_SEED)
        images: list[dict[str, Any]] = []
        for img_id in chosen:
            member = tar.extractfile(f"{MIO_MEMBER_PREFIX}/train/{img_id}.jpg")
            assert member is not None, f"frame {img_id}.jpg missing from tar"
            raw = member.read()
            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"cv2 failed to decode {img_id}.jpg")
            h, w = arr.shape[:2]
            scored, ignore = route_boxes(tax, "mio-tcd", gt[img_id], ids, float(w), float(h))
            dd.write_bytes(f"frames/{img_id}.jpg", raw)
            images.append(
                _image_record(img_id, f"frames/{img_id}.jpg", w, h, "unknown", scored, ignore)
            )
    _finalize(
        out_root,
        dd,
        "mio-tcd",
        MIO_TAR.name,
        "train (only split with public GT)",
        {"seed": SAMPLE_SEED, "target": MIO_TARGET, "selected": len(images)},
        images,
    )


# --- AAU RainSnow ---------------------------------------------------------------------------


def parse_rainsnow(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[Any]]]:
    """COCO json -> ({image_id: image_meta}, {image_id: [annotation, ...]})."""
    by_img = {int(im["id"]): im for im in coco["images"]}
    anns: dict[int, list[Any]] = defaultdict(list)
    for a in coco["annotations"]:
        anns[int(a["image_id"])].append(a)
    return by_img, dict(anns)


def rainsnow_boxes(
    tax: dict[str, Any], annotations: list[Any], ids: dict[str, int], w: float, h: float
) -> tuple[list[Box5], list[BoxI]]:
    """Route one image's COCO annotations (bbox = [x, y, w, h]) to scored/ignore xyxy boxes."""
    rows = [
        (
            int(a["category_id"]),
            a["bbox"][0],
            a["bbox"][1],
            a["bbox"][0] + a["bbox"][2],
            a["bbox"][1] + a["bbox"][3],
        )
        for a in annotations
    ]
    return route_boxes(tax, "aau-rainsnow", rows, ids, w, h)


def build_rainsnow(out_root: Path) -> None:
    tax = load_taxonomy()
    ids = _ids_map()
    dd = open_dataset_dir(out_root, "w")
    with zipfile.ZipFile(RAINSNOW_ZIP) as zf:
        coco = json.loads(zf.read(RAINSNOW_RGB_JSON))
        by_img, anns = parse_rainsnow(coco)
        names = set(zf.namelist())
        images: list[dict[str, Any]] = []
        for img_id in sorted(by_img):
            im = by_img[img_id]
            file_name = im["file_name"]
            assert file_name in names, f"annotated frame {file_name} missing from zip"
            stem = file_name.rsplit(".", 1)[0].replace("/", "__")
            ext = file_name.rsplit(".", 1)[1]
            camera = file_name.split("/", 1)[0]
            w, h = float(im["width"]), float(im["height"])
            scored, ignore = rainsnow_boxes(tax, anns.get(img_id, []), ids, w, h)
            dd.write_bytes(f"frames/{stem}.{ext}", zf.read(file_name))
            images.append(
                _image_record(stem, f"frames/{stem}.{ext}", int(w), int(h), camera, scored, ignore)
            )
    _finalize(
        out_root,
        dd,
        "aau-rainsnow",
        RAINSNOW_ZIP.name,
        "all annotated RGB frames",
        {"seed": None, "target": None, "selected": len(images)},
        images,
    )


# --- shared ---------------------------------------------------------------------------------


def _image_record(
    stem: str, file: str, w: int, h: int, camera: str, scored: list[Box5], ignore: list[BoxI]
) -> dict[str, Any]:
    return {
        "id": stem,
        "file": file,
        "width": w,
        "height": h,
        "camera": camera,
        "weather": "unknown",  # neither cross-dataset carries a per-frame weather label
        "boxes": [[round(x, 2) for x in b[:4]] + [b[4]] for b in scored],
        "ignore": [[round(x, 2) for x in b] for b in ignore],
    }


def _finalize(
    out_root: Path,
    dd: Any,
    dataset: str,
    archive: str,
    split: str,
    sample: dict[str, Any],
    images: list[dict[str, Any]],
) -> None:
    n_boxes = sum(len(im["boxes"]) for im in images)
    manifest = {
        "meta": {
            "dataset": dataset,
            "role": "gated-eval",
            "source_archive": archive,
            "split_used": split,
            "sample": sample,
            "class_names": canonical_names(load_taxonomy()),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "images": images,
    }
    dd.write_text("annotations.json", json.dumps(manifest, indent=2) + "\n")
    mark_readonly(out_root, FROZEN_REASON)
    print(f"  {dataset}: {len(images)} frames, {n_boxes} scored boxes -> {out_root.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mio", "rainsnow", "all"], default="all")
    parser.add_argument("--out", type=Path, default=PROCESSED)
    args = parser.parse_args(argv)

    if args.dataset in ("mio", "all"):
        build_mio(args.out / "mio-tcd-eval-v1")
    if args.dataset in ("rainsnow", "all"):
        build_rainsnow(args.out / "rainsnow-eval-v1")
    print("cross-dataset eval sets built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
