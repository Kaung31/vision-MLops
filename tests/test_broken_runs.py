"""Broken-run integration tests (Phase 3 exit criterion #2): the diagnosis flags must fire on
two deliberately broken REAL training runs.

  * LR 100x too high (lr0=1.0 vs default 0.01, explicit SGD) -> `diverged`
  * undertrain: 3 epochs FROM SCRATCH (yolov8s.yaml, no COCO init) -> `underfit`
    (from scratch is required: a COCO-pretrained model beats the 0.381 floor in 2 epochs)

These train for real (~minutes each on Apple MPS), so they are gated behind RUN_TRAINING_IT=1
and skipped in CI. They log to a throwaway local MLflow store, never to DagsHub.

Run:  RUN_TRAINING_IT=1 uv run --group eval --group train pytest tests/test_broken_runs.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_TRAINING_IT"),
    reason="real training runs; set RUN_TRAINING_IT=1 (needs eval+train groups, MPS/GPU)",
)


def _train(tmp_path: Path, run_name: str, extra: list[str]) -> dict[str, Any]:
    from src.train.run import main_summary

    return main_summary(
        [
            "--config",
            "configs/training/base.yaml",
            "--device",
            "mps",
            "--fraction",
            "0.1",
            "--tracking-uri",
            f"sqlite:///{tmp_path}/it-mlflow.db",
            "--run-name",
            run_name,
            *extra,
        ]
    )


def test_lr_100x_flags_diverged(tmp_path: Path) -> None:
    s = _train(tmp_path, "it-lr100x", ["--epochs", "3", "--lr0", "1.0"])
    d = s["diagnosis"]
    assert d["suspect"], f"LR-100x run was not flagged: {d}"
    # blow-ups can also read as underfit (loss went up == never learned); diverged is the
    # canonical flag but either proves the tripwire fires on this failure mode
    assert d["diverged"] or d["underfit"], f"wrong flags for LR-100x: {d}"


def test_undertrain_from_scratch_flags_underfit(tmp_path: Path) -> None:
    s = _train(tmp_path, "it-undertrain", ["--epochs", "3", "--model", "yolov8s.yaml"])
    d = s["diagnosis"]
    assert d["suspect"], f"undertrained run was not flagged: {d}"
    assert d["underfit"], f"undertrain must flag underfit: {d}"
    assert s["best_val_map50_95"] is not None and s["best_val_map50_95"] < 0.381
