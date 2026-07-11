#!/usr/bin/env python3
"""Lighting-invariant edge/structure re-pass over the median backgrounds (ADR 0005).

Colour correlation under-merges across weather; edges don't move with lighting. This:
  * builds a CLAHE->Canny edge map per sequence (structure only);
  * scores same-camera candidates by SPATIAL edge-map correlation (fixed structures line up);
  * confirms the row-3 and hedge-cluster merge suspicions with edge overlays;
  * shortlists 2-3 daytime matches per night/rain isolate for a targeted A-vs-B eyeball;
  * emits larger crops of the 63xxx / bottom-row sequences.

Reads data/camera_work/backgrounds/*.png (from derive_camera_groups.py). Outputs to
data/camera_work/edges/ (git-ignored: dataset imagery, Law 5).

Run: uv run --group data python scripts/edge_match.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
BG = REPO / "data/camera_work/backgrounds"
AUDIT_JSON = REPO / "docs/audits/annotation-audit.json"
OUT = REPO / "data/camera_work/edges"

ROW3 = ["MVI_40152", "MVI_40161", "MVI_40162", "MVI_40171", "MVI_40172"]
HEDGE = [
    "MVI_40201",
    "MVI_40204",
    "MVI_40211",
    "MVI_40212",
    "MVI_40213",
    "MVI_40241",
    "MVI_40243",
    "MVI_40244",
]
C63 = [
    "MVI_63521",
    "MVI_63525",
    "MVI_63544",
    "MVI_63552",
    "MVI_63553",
    "MVI_63554",
    "MVI_63561",
    "MVI_63562",
    "MVI_63563",
]


def meta() -> dict[str, str]:
    data = json.loads(AUDIT_JSON.read_text())
    return {s["sequence"]: s["weather"] for s in data["sequences"]}


def load_bg(seq: str) -> np.ndarray:
    return cv2.imread(str(BG / f"{seq}.png"))


def edge_map(bg: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)  # lift night structure
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    return cv2.dilate(edges, np.ones((2, 2), np.uint8))  # tolerate small misalignment


def edge_feat(e: np.ndarray) -> np.ndarray:
    v = cv2.resize(e, (120, 68)).astype(np.float32).ravel()
    v -= v.mean()
    s = float(v.std())
    return v / s if s > 0 else v


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def montage(imgs: list[np.ndarray], path: Path, cols: int = 5) -> None:
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * h, cols * w, 3), 30, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = im
    cv2.imwrite(str(path), canvas)


def overlay(ea: np.ndarray, eb: np.ndarray, path: Path) -> None:
    # edge A -> red, edge B -> green; aligned fixed structures show as yellow
    canvas = np.zeros((*ea.shape, 3), np.uint8)
    canvas[..., 2] = ea
    canvas[..., 1] = eb
    cv2.imwrite(str(path), canvas)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    weather = meta()
    seqs = list(weather)
    edges = {s: edge_map(load_bg(s)) for s in seqs}
    feats = np.stack([edge_feat(edges[s]) for s in seqs])
    sim = feats @ feats.T / feats.shape[1]
    idx = {s: i for i, s in enumerate(seqs)}

    def s5(a: str, b: str) -> float:
        return float(sim[idx[a], idx[b]])

    print("=== ROW-3 suspicion (one camera?) edge-similarity ===")
    for a in ROW3:
        print("  " + a[-5:] + ": " + " ".join(f"{b[-5:]}={s5(a, b):+.2f}" for b in ROW3 if b != a))
    print("\n=== HEDGE cluster edge-similarity ===")
    for a in HEDGE:
        print("  " + a[-5:] + ": " + " ".join(f"{b[-5:]}={s5(a, b):+.2f}" for b in HEDGE if b != a))

    day = [s for s in seqs if weather[s] in ("sunny", "cloudy")]
    dark = [s for s in seqs if weather[s] in ("night", "rainy")]
    print("\n=== NIGHT/RAIN isolates: top-3 DAYTIME candidates by edge structure ===")
    for s in dark:
        ranked = sorted(day, key=lambda d: -s5(s, d))[:3]
        print(
            f"  {s[-5:]} ({weather[s]}): " + ", ".join(f"{d[-5:]}={s5(s, d):+.2f}" for d in ranked)
        )
        montage(
            [label(cv2.cvtColor(edges[s], cv2.COLOR_GRAY2BGR), f"{s[-5:]} {weather[s]}")]
            + [label(cv2.cvtColor(edges[d], cv2.COLOR_GRAY2BGR), d[-5:]) for d in ranked],
            OUT / f"night_{s[-5:]}_candidates.png",
            cols=4,
        )

    # merge-suspicion edge montages + overlays
    montage(
        [label(cv2.cvtColor(edges[s], cv2.COLOR_GRAY2BGR), s[-5:]) for s in ROW3],
        OUT / "row3_edges.png",
        cols=5,
    )
    montage(
        [label(cv2.cvtColor(edges[s], cv2.COLOR_GRAY2BGR), s[-5:]) for s in HEDGE],
        OUT / "hedge_edges.png",
        cols=4,
    )
    montage(
        [label(cv2.cvtColor(edges[s], cv2.COLOR_GRAY2BGR), s[-5:]) for s in C63],
        OUT / "c63_edges.png",
        cols=5,
    )
    overlay(edges["MVI_40152"], edges["MVI_40171"], OUT / "overlay_40152_R_40171_G.png")
    overlay(edges["MVI_40241"], edges["MVI_40243"], OUT / "overlay_40241_R_40243_G.png")
    overlay(edges["MVI_40131"], edges["MVI_63521"], OUT / "overlay_40131_R_63521_G.png")

    # colour crops the user asked for (bottom rows / 63xxx)
    montage([label(load_bg(s), s[-5:]) for s in C63], OUT / "c63_backgrounds.png", cols=3)
    print(f"\nedge montages + overlays + crops -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
