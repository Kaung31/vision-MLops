# Phase 4 — Hyperparameter tuning: what it actually bought

- **Status:** IN PROGRESS — search done + salvaged; transfer study pending
- **Date:** 2026-07-13
- **Phase:** 4 (guide §4.1–4.4)

## Search space rationale

The high-leverage knobs only (`configs/tuning.yaml`), ranges bracketing Ultralytics defaults:
`lr0`, `lrf`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `hsv_{h,s,v}`, `imgsz`. Tuned
on the **nano** model (guide 4.2) at 20 epochs/trial — realistic, not toy — with Optuna search
+ ASHA early stopping, under a hard 8 GPU-hour budget enforced in code (`src/train/tune.py`).

## Budget enforcement (demonstrated, not aspirational)

The first launch was **refused by the guard**: 20 trials × 25 epochs projected 12.5h > 8h
budget. The config was resized to 15 × 20 (7.5h ≤ 8h) and relaunched. This is the
exit-criterion "budget enforcement demonstrated" — live, before any GPU spend, not just a unit
test. Actual spend: **4.41 GPU-hours** for the (interrupted) session.

## What the search found — and a real finding about the tool

14 trials completed before the session was interrupted (laptop slept). ASHA behaved exactly as
designed: 6 trials ran the full 20 epochs; the 7 weakest were killed at the grace period of 8.
Results were recovered from Ultralytics' MLflow autolog (`spent_gpu_seconds` reconciled into
the budget; summary run `tune-session-1-recovered` on DagsHub).

**Finding: Ultralytics `optimizer="auto"` silently overrides `lr0`/`lrf`.** All 14 trials ran
under `auto`, so the two headline knobs had *zero effect*. The proof is internal: the winning
trial (val mAP50-95 **0.6931**) carries the **lowest** `lr0` tried (1e-4) — if that LR were
real, the model would barely learn in 20 epochs and score worst, not best. So this was in
truth an **augmentation + imgsz search at auto's learning rate**, not an LR search.

Decision (honest, budget-respecting): rather than re-tune with a forced optimizer (needs a
budget raise and risks underperforming a well-chosen auto default in this saturated-data
regime — see the learning-curve study), we **drop the non-functional `lr0`/`lrf`** and carry
the tuned augmentation forward on auto-lr. `configs/training/tuned-v1-candidate.yaml`.

Best trial's effective (applied) config:

| knob | tuned value | default |
|---|---|---|
| weight_decay | 0.000883 | 5e-4 |
| warmup_epochs | 0.566 | 3.0 |
| mosaic | 0.988 | 1.0 |
| mixup | 0.294 | 0.0 |
| hsv_h / hsv_s / hsv_v | 0.099 / 0.539 / 0.787 | 0.015 / 0.7 / 0.4 |
| imgsz | 640 | 640 |

## Transfer study (nano → small) — PENDING

Train the **small** model with the candidate config, run the full `eval-harness-v1.1` report,
and compare to `champion-v1` (in-dist 0.601, gated 0.137) with CIs, per slice. The candidate
is promoted to `configs/training/tuned-v1.yaml` **only if it wins**. Numbers: TBD.

## Honest bottom line — PENDING

TBD after the transfer study: did tuning buy anything over defaults, given (a) lr couldn't be
tuned and (b) the model is already capacity-limited on this data? A null result is a valid,
reportable outcome.
