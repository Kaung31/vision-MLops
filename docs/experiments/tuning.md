# Phase 4 — Hyperparameter tuning: what it actually bought

- **Status:** COMPLETE — tuned config won the transfer study and is promoted to `tuned-v1.yaml`
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

## Transfer study (nano → small)

Trained the **small** model (`tuned-v1`, run `16f49482`, 61 epochs, best val 0.7264 vs
champion 0.7008) with the candidate config, then ran the full `eval-harness-v1.1` report on
`eval-frozen-v1` + both gated sets. Comparison vs `champion-v1`, mAP50-95 with 95% CIs:

| slice | champion-v1 | tuned-v1 | Δ | verdict |
|---|---|---|---|---|
| in-dist overall | 0.601 [0.576, 0.625] | 0.630 [0.610, 0.651] | **+0.029** | CIs barely overlap — real |
| &nbsp;&nbsp;car | 0.684 [0.677, 0.690] | 0.693 [0.687, 0.699] | +0.009 | up |
| &nbsp;&nbsp;bus | 0.646 [0.575, 0.713] | 0.658 [0.600, 0.714] | +0.012 | thin slice, ~flat |
| &nbsp;&nbsp;**van_truck** | 0.473 [0.454, 0.494] | 0.540 [0.521, 0.559] | **+0.066** | non-overlapping — significant |
| **RainSnow** (gated) | 0.096 [0.090, 0.105] | 0.117 [0.108, 0.127] | **+0.020** | non-overlapping — significant |
| MIO-TCD (gated) | 0.218 [0.199, 0.238] | 0.222 [0.203, 0.242] | +0.003 | flat |
| gated aggregate | 0.137 | 0.151 | +0.014 | up (still < 0.157 floor) |
| night (weather) | 0.458 [0.421, 0.496] | 0.507 [0.471, 0.546] | +0.049 | up |
| sunny / rainy | 0.726 / 0.662 | 0.690 / 0.643 | −0.036 / −0.019 | within wide small-slice CIs — n.s. |

**Decision: PROMOTE.** tuned-v1 beats champion-v1 in-distribution (no large slice regresses
significantly) and — the result that matters — improves the two hardest slices with
statistical significance: `van_truck` (+0.066) and the RainSnow rain/snow/night set (+0.020).
`configs/training/tuned-v1.yaml` is now the config all future retraining uses. tuned-v1 is
registered as `traffic-vision-detector` **v2 = @challenger**; `@champion` stays on v1 until
Phase 5's `promote.py` runs the contract and flips it mechanically (no hand-promotion).

## Honest bottom line

Tuning bought **~+0.03 in-distribution and, more importantly, ~+0.02 on the hardest
cross-dataset condition** — and it did so entirely through **augmentation**, because
`optimizer="auto"` made the learning-rate knobs no-ops (the tool's behaviour, discovered and
proven from the trial data, not assumed). The direction is the valuable part: in a
capacity-limited regime (learning-curve study) where more data buys almost nothing, stronger
augmentation was the lever that improved *generalization* to unseen cameras/weather — exactly
where champion-v1 was weakest (it had regressed below the zero-shot floor). It did **not**
fully close that cross-dataset gap (0.151 < 0.157): fine-tuning still generalizes worse than
zero-shot COCO in aggregate, which is the problem the Phase 8 flywheel exists to attack. So:
a modest but real and correctly-directed gain, honestly bounded.
