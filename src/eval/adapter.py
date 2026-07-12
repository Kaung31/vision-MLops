"""Ultralytics YOLO -> harness.Detections adapter (inference only; the zero-shot floor).

Two concerns kept separate:
  * ``detections_from_arrays`` — pure numpy: raw (boxes, scores, class-index) -> Detections,
    keeping only scored classes and relabelling to canonical ids. Torch-free, unit-tested.
  * ``YoloDetector`` — thin wrapper that loads a YOLO checkpoint (lazy import so the module,
    and its mapping test, need no torch) and extracts those arrays from a Results object.

Coordinate space: YOLO returns boxes in the INPUT array's pixel space (it letterboxes to
`imgsz` internally and maps back), and we feed each frame at its native resolution — which is
the space the GT is stored in — so there is no manual rescale. Model class NAMES are mapped to
canonical NAMES via the zero-shot table below (COCO has no 'van', so van_truck receives only
'truck'); canonical NAMES resolve to ids through taxonomy.yaml, so no canonical id is hardcoded.

mAP needs the full precision-recall curve, so predict at a LOW score threshold (0.001) with
NMS iou 0.7 — Ultralytics' predict default conf=0.25 would truncate the curve and understate AP.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.data.taxonomy import load as load_taxonomy
from src.eval.harness import Detections

# Zero-shot COCO-detector class NAME -> our canonical class NAME. The only three COCO classes
# we score; every other detection (person, etc.) is dropped, never counted as a false positive.
COCO_TO_CANONICAL_NAME: dict[str, str] = {"car": "car", "bus": "bus", "truck": "van_truck"}

DEFAULT_CONF = 0.001
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640


def canonical_ids() -> dict[str, int]:
    """Canonical class name -> id, straight from taxonomy.yaml (single source of truth)."""
    tax = load_taxonomy()
    return {str(c["name"]): int(c["id"]) for c in tax["canonical"]}


def build_class_map(model_names: dict[int, str]) -> dict[int, int]:
    """Model class index -> canonical id, for the scored classes only (others omitted = dropped).

    Two model kinds, resolved by NAME: a model fine-tuned on our data already uses canonical
    names (car/bus/van_truck) — matched first; a zero-shot COCO model uses COCO names —
    matched via the COCO table (unchanged from the frozen floor report). Anything else drops.
    """
    ids = canonical_ids()
    return {
        idx: ids[name if name in ids else COCO_TO_CANONICAL_NAME[name]]
        for idx, name in model_names.items()
        if name in ids or name in COCO_TO_CANONICAL_NAME
    }


def detections_from_arrays(
    boxes: NDArray[Any], scores: NDArray[Any], cls_idx: NDArray[Any], class_map: dict[int, int]
) -> Detections:
    """Raw YOLO arrays (xyxy px, scores, model class indices) -> Detections in canonical ids."""
    if len(cls_idx) == 0:
        return Detections()
    keep = np.array([int(c) in class_map for c in cls_idx], dtype=bool)
    labels = np.array([class_map[int(c)] for c in cls_idx[keep]], dtype=np.int64)
    return Detections(
        boxes[keep].astype(np.float64).reshape(-1, 4),
        scores[keep].astype(np.float64),
        labels,
    )


class YoloDetector:
    """Callable ``detector(image_bgr) -> Detections`` around an Ultralytics checkpoint."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "mps",
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]  # lazy: keep module torch-free

        self.model: Any = YOLO(str(weights))  # untyped boundary — treat results as Any
        self.class_map = build_class_map(dict(self.model.names))
        self.device, self.conf, self.iou, self.imgsz = device, conf, iou, imgsz

    def __call__(self, image_bgr: NDArray[np.uint8]) -> Detections:
        res = self.model.predict(
            image_bgr,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        b = res.boxes
        return detections_from_arrays(
            b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy(), self.class_map
        )
