"""Post-run training diagnosis: diverged / overfit / underfit -> `suspect`.

Pure functions. `run.py` calls this after a run and attaches the verdict to the MLflow run,
tagging a flagged run `suspect` (guide Phase 3.2). Two kinds of signal, kept honest:

  * Trajectory (shape only): the run's own per-epoch train/val loss and val-mAP curves — these
    are Ultralytics' internal metrics on train-v1's val split, used ONLY to read the shape of
    training (did it diverge? did val loss rebound while train fell?). They are never quoted as
    a result (Law 2).
  * The floor check: the run's best val mAP50-95 vs the frozen zero-shot floor (0.381,
    eval-harness-v1 on eval-frozen-v1). CAVEAT, stated honestly: at train time the mAP side
    comes from Ultralytics' evaluator on train-v1's val split — a different ruler and split
    than the floor's. That is fine for a *flag* (the guide prescribes exactly this check for
    underfit) but it is never quoted as a result; quotable numbers only ever come from the
    harness report (Law 2).

The two deliberately broken runs the exit criteria require map cleanly onto these:
  * LR 100x too high  -> loss blows up / non-finite             -> `diverged`
  * 3-epoch undertrain -> best val mAP below the zero-shot floor -> `underfit`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

DEFAULTS: dict[str, float] = {
    "overfit_rebound": 0.20,  # val loss > 20% above its own min (while train still falling)
    "min_learn_frac": 0.05,  # train loss must drop >=5% from epoch 0, else "barely learned"
    "window": 5,  # trailing epochs (reserved for plateau checks / callers)
    "floor_map50_95": 0.381,  # eval-harness-v1 zero-shot floor (reports/zero-shot-floor)
}


@dataclass
class Diagnosis:
    diverged: bool
    overfit: bool
    underfit: bool
    suspect: bool
    best_epoch: int
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(*arrays: list[float]) -> bool:
    return all(bool(np.all(np.isfinite(a))) for a in arrays if len(a))


def diagnose(
    train_losses: list[float],
    val_losses: list[float],
    val_map_curve: list[float] | None = None,
    best_val_map: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> Diagnosis:
    """Flag a run diverged/overfit/underfit from its loss curves + val-mAP-vs-floor check."""
    c = {**DEFAULTS, **(cfg or {})}
    n = len(train_losses)
    reasons: list[str] = []

    diverged = not _finite(train_losses, val_losses)
    if diverged:
        reasons.append("non-finite loss (NaN/Inf) during training")
    elif n >= 2 and train_losses[-1] >= train_losses[0]:
        diverged = True
        reasons.append(
            f"train loss did not decrease ({train_losses[0]:.4f} -> {train_losses[-1]:.4f})"
        )

    # overfit: val loss rebounded past its minimum while train loss kept improving
    overfit = False
    if not diverged and n >= 2:
        vmin_i = int(np.argmin(val_losses))
        vmin = val_losses[vmin_i]
        rebounded = val_losses[-1] > vmin * (1.0 + c["overfit_rebound"])
        train_still_falling = train_losses[-1] < train_losses[vmin_i]
        if rebounded and train_still_falling:
            overfit = True
            reasons.append(
                f"val loss rebounded {val_losses[-1]:.4f} from min {vmin:.4f} (epoch {vmin_i}) "
                f"while train loss still fell — overfitting"
            )

    # underfit: best val mAP below the zero-shot floor, or the model barely learned at all
    underfit = False
    if not diverged:
        if best_val_map is not None and best_val_map < c["floor_map50_95"]:
            underfit = True
            reasons.append(
                f"best val mAP50-95 {best_val_map:.4f} below floor {c['floor_map50_95']:.4f}"
            )
        learn_frac = (
            (train_losses[0] - train_losses[-1]) / abs(train_losses[0])
            if n >= 2 and train_losses[0] != 0
            else 0.0
        )
        if learn_frac < c["min_learn_frac"]:
            underfit = True
            reasons.append(
                f"train loss dropped only {learn_frac:.1%} (< {c['min_learn_frac']:.0%})"
            )

    best_epoch = int(np.argmax(val_map_curve)) if val_map_curve else max(n - 1, 0)
    suspect = diverged or overfit or underfit
    return Diagnosis(diverged, overfit, underfit, suspect, best_epoch, reasons)
