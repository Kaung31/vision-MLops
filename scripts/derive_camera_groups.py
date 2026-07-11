#!/usr/bin/env python3
"""Derive DRAFT camera groups for UA-DETRAC, for human audit (guide Phase 1 ⑤ / ADR 0005).

Method (temporal-median backgrounds → similarity clustering → contact sheet):
  1. per sequence, sample K frames from the images zip and take the pixel-wise MEDIAN —
     moving vehicles cancel out, leaving the static background (road geometry, gantries,
     lane markings) which survives day/night/weather far better than a random frame;
  2. cluster sequences by standardized-grayscale-background correlation (a DRAFT only);
  3. write a contact sheet + nearest-neighbour report so a human can hunt UNDER-merges
     (same camera split into two groups = the only failure that causes leakage).

Output is a DRAFT under data/camera_work/ (git-ignored, contains dataset imagery — Law 5).
The committed configs/camera_groups.yaml is the hand-audited ground truth.

Run: uv run --group data python scripts/derive_camera_groups.py
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
ZIP = REPO / "data/raw/ua-detrac/ua-detrac-orig.zip"
AUDIT_JSON = REPO / "docs/audits/annotation-audit.json"
OUT = REPO / "data/camera_work"
K = 41  # frames sampled per sequence for the median background
THUMB = (320, 180)  # contact-sheet thumbnail (w, h)
CORR_THR = 0.80  # draft union threshold; over-merge is safe, human hunts under-merges


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def train_sequences() -> list[str]:
    data = json.loads(AUDIT_JSON.read_text())
    return [s["sequence"] for s in data["sequences"]]


def frames_by_seq(zf: zipfile.ZipFile, seqs: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if not name.lower().endswith(".jpg"):
            continue
        seq = next((p for p in name.split("/") if p.startswith("MVI_")), None)
        if seq is not None and seq in seqs:
            out[seq].append(name)
    for seq in out:
        out[seq].sort()
    return out


def median_background(zf: zipfile.ZipFile, paths: list[str]) -> np.ndarray:
    idx = np.linspace(0, len(paths) - 1, min(K, len(paths))).astype(int)
    frames = []
    for i in idx:
        img = cv2.imdecode(np.frombuffer(zf.read(paths[i]), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    return np.median(np.stack(frames).astype(np.float32), axis=0).astype(np.uint8)


def feature(bg: np.ndarray) -> np.ndarray:
    g = cv2.resize(cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY), (96, 54)).astype(np.float32).ravel()
    g -= g.mean()
    s = float(g.std())
    return g / s if s > 0 else g


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backgrounds").mkdir(exist_ok=True)
    seqs = train_sequences()
    print(f"deriving camera groups for {len(seqs)} sequences...")

    backgrounds: dict[str, np.ndarray] = {}
    with zipfile.ZipFile(ZIP) as zf:
        paths = frames_by_seq(zf, set(seqs))
        for n, seq in enumerate(seqs, 1):
            bg = median_background(zf, paths[seq])
            backgrounds[seq] = bg
            cv2.imwrite(str(OUT / "backgrounds" / f"{seq}.png"), bg)
            print(f"  [{n:>2}/{len(seqs)}] {seq}: median of {min(K, len(paths[seq]))} frames")

    feats = np.stack([feature(backgrounds[s]) for s in seqs])
    corr = feats @ feats.T / feats.shape[1]  # Pearson corr (rows are standardized)

    dsu = DSU(len(seqs))
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            if corr[i, j] > CORR_THR:
                dsu.union(i, j)
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(seqs)):
        members[dsu.find(i)].append(i)
    groups = {
        f"cam_{gi:02d}": [seqs[m] for m in sorted(ms)]
        for gi, (_, ms) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1])))
    }

    # nearest-neighbour report (the key audit aid: high-corr pairs = likely same camera)
    print(f"\n{len(groups)} draft groups. Nearest neighbours per sequence:")
    for i, s in enumerate(seqs):
        nn = [j for j in np.argsort(-corr[i]) if j != i][:2]
        pairs = ", ".join(f"{seqs[j][-5:]}={corr[i, j]:.2f}" for j in nn)
        print(f"  {s[-5:]}  NN: {pairs}")
    print("\ndraft groups (size>1 are the merges to scrutinise):")
    for g, ss in groups.items():
        if len(ss) > 1:
            print(f"  {g}: {[s[-5:] for s in ss]}")

    # contact sheet, ordered by group
    order = [(g, s) for g, ss in groups.items() for s in ss]
    cols = 6
    rows = ceil(len(order) / cols)
    tw, th = THUMB
    sheet = np.full((rows * th, cols * tw, 3), 40, np.uint8)
    for idx, (g, s) in enumerate(order):
        thumb = cv2.resize(backgrounds[s], (tw, th))
        cv2.putText(
            thumb, f"{g} {s[-5:]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        r, c = divmod(idx, cols)
        sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb
    cv2.imwrite(str(OUT / "camera_contact_sheet.png"), sheet)

    draft = OUT / "camera_groups.draft.yaml"
    draft.write_text(yaml.safe_dump({"groups": groups}, sort_keys=False))
    print(f"\ncontact sheet -> {OUT / 'camera_contact_sheet.png'}")
    print(f"draft groups  -> {draft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
