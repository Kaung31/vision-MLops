# ADR 0008 — Corruption severity calibration and the `eval-harness-v1` freeze

- **Status:** Accepted
- **Date:** 2026-07-12
- **Phase:** 2 (Evaluation Harness) — the freeze

## Context

The corruption severity ranges (`src/eval/corruption.py`) were set **provisionally** in Phase 2
④, before any model existed, so they could not yet be judged against real degradation behaviour.
Phase 2 ⑥ requires calibrating them **once** on the zero-shot COCO-YOLO floor and then freezing
them as part of `eval-harness-v1` — after which changing any eval definition (AP math, ignore
precedence, slice bins, corruption ranges) requires a **new** harness tag and a re-run of every
historical report, because it is changing the ruler mid-experiment (Law 2).

## Calibration run

- **Model:** `yolov8n.pt` (COCO-pretrained, zero-shot), inference on Apple MPS (allowed: inference,
  not training — hardware routing).
- **Set:** `eval-frozen-v1` (UA-DETRAC test), 1080 frames, 10584 in-distribution boxes.
- **Provenance:** git `cfe6ccd`, seed 0, 2000 bootstrap resamples. Full report recorded at
  `reports/zero-shot-floor/`.

**Floor mAP50-95 vs severity (severity 0 = clean baseline = 0.381):**

| corruption | s0 | s1 | s2 | s3 | s4 | s5 | span |
|---|---|---|---|---|---|---|---|
| gaussian_blur | 0.381 | 0.381 | 0.377 | 0.347 | 0.312 | 0.264 | 0.117 |
| gaussian_noise | 0.381 | 0.370 | 0.351 | 0.289 | 0.141 | 0.060 | 0.321 |
| jpeg_compression | 0.381 | 0.380 | 0.375 | 0.367 | 0.350 | 0.293 | 0.088 |
| brightness_up | 0.381 | 0.379 | 0.377 | 0.373 | 0.372 | 0.360 | 0.021 |
| brightness_down | 0.381 | 0.378 | 0.368 | 0.354 | 0.314 | 0.253 | 0.128 |

## Decision

**Confirm the provisional ranges unchanged and freeze them.** The calibration criterion was: each
curve must be **monotone** and **non-degenerate** (not flat, not already zero at s1, not a cliff at
s1) so the promotion contract's C4 (corruption-AUC comparison) has resolution. All five curves meet
it. The frozen ranges (index by severity−1):

- `BLUR_SIGMA = (0.5, 1.0, 2.0, 3.0, 4.0)` — Gaussian σ px
- `NOISE_SIGMA = (5, 10, 20, 40, 60)` — additive Gaussian σ on 0–255
- `JPEG_QUALITY = (80, 60, 40, 25, 15)` — libjpeg quality
- `BRIGHTNESS_DELTA = (20, 40, 60, 80, 100)` — additive shift on 0–255, applied signed in both
  directions

## Honest caveats (Law 7 — recorded, not fixed away)

- **jpeg_compression (span 0.088) and brightness_up (span 0.021) are shallow axes.** This is
  *genuine YOLO robustness* to those perturbations, not a mis-set range. Steepening them by picking
  harsher params (quality → single digits, +140 brightness) would manufacture difficulty to make
  the curve look dramatic — dishonest ruler-shaping. They stay realistic; a shallow curve is a true
  measurement.
- **brightness is split into two signed directions** (a non-symmetric pair, decided in ④):
  darkening (`brightness_down`, span 0.128) is the production-relevant direction (day→night drift)
  and has good resolution; `brightness_up` is the control direction and is expected to be flat.
- **gaussian_noise reaches near-destruction at s5** (0.060). That is intentional — one
  near-failure anchor at max severity is standard for corruption benchmarks (cf. ImageNet-C) and
  bounds the severe end of the curve.
- **blur s1 (σ=0.5) is a near-no-op** (0.381→0.381 at 3 dp). Kept: σ=0.5 is a realistic
  barely-defocused frame, and a gentle low-severity anchor is fine.

## Consequences

- `src/eval/corruption.py` ranges are now **FROZEN**; its module docstring is updated from
  "PROVISIONAL" to frozen-with-this-ADR.
- The harness (`harness.py` + `bootstrap.py` + `corruption.py` + `report.py`) is tagged
  **`eval-harness-v1`** at this commit. From here, no metric is trusted unless produced by this
  ruler, and any eval-logic change needs a new tag + historical re-run.
- The two gated cross-dataset eval sets are frozen and versioned alongside: `mio-tcd-eval-v1`,
  `rainsnow-eval-v1`.

## Alternatives rejected

- **Re-scale jpeg / brightness_up to force steeper curves.** Rejected: fits the instrument to a
  desired shape before any model comparison — the exact failure Law 2 guards against.
- **Drop the two shallow axes.** Rejected: their flatness is a real, reportable robustness result;
  removing them hides a true measurement.

## Addendum — `eval-harness-v1.1` (2026-07-12, Phase 3)

`adapter.py`'s class map only knew COCO names; a checkpoint fine-tuned on train-v1 exposes
canonical names (`car`/`bus`/`van_truck`), and `van_truck` — not a COCO name — was silently
dropped, which would have zeroed that class in any fine-tuned model's report. Fixed before
first fine-tuned use: canonical names match first, the COCO table is an unchanged fallback.

Freeze protocol followed: eval code changed → new tag **`eval-harness-v1.1`** + historical
report re-run. The zero-shot floor was re-run in full under v1.1 and matched the frozen v1
report on every metric (COCO-name mapping is byte-identical; the fix touches only names v1
never scored). No frozen definition (AP math, ignore precedence, slices, corruption ranges)
changed. Equality evidence: `reports/zero-shot-floor/v1.1-equivalence.txt`.
