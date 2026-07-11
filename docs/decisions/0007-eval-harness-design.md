# ADR 0007 — Evaluation harness design (the frozen ruler)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 2 (Evaluation Harness)

## Context

The harness is the project's core IP and its measuring instrument. It is written from
scratch (stack §2: "Custom — you write it") and **frozen** at `eval-harness-v1`: once
tagged, changing any definition below requires a new tag and re-running historical
reports, because it is changing the ruler mid-experiment (Law 2).

## Decisions

- **AP = COCO 101-point interpolation** over recall {0.00..1.00}; greedy per-class matching
  (score order, highest-IoU unmatched GT at IoU ≥ threshold → TP). `mAP50` = AP@0.5 meaned
  over classes; `mAP50-95` = AP meaned over IoU {0.50..0.95, step 0.05} then over classes.
  Classes with no GT are NaN and excluded from the mean (not counted as 0).
- **Two ignore mechanisms compose, with fixed precedence** per detection: real GT → TP;
  else ignore-GT → dropped; else ignore-region (IoA ≥ 0.5) → dropped; else FP.
  - *Ignore regions* are Phase 1's low-resolution polygons (`box_ioa` ≥ 0.5): a would-be FP
    inside one is dropped so unlabelable vehicles don't inflate false positives.
  - *Ignore-GT* is the scale-slice mechanism (COCO `areaRng`): when scoring one scale bin,
    out-of-range GT are ignore-GT — a detection matching one is dropped, not an FP, and only
    in-range GT count toward recall.
- **Scale bins use GT box area in ORIGINAL annotation space (960×540)**, never post-resize.
  Membership must not change when `imgsz` changes between runs, or scale slices stop being
  comparable across model versions. Bins are **half-open `[lo, hi)`** so every box lands in
  exactly one bin (partition, asserted by test): small `[0, 32²)`, medium `[32², 96²)`,
  large `[96², ∞)`.
  - **Honest caveat:** COCO's 32²/96² thresholds were tuned for COCO's distribution. On
    960×540 traffic footage, distant vehicles dominate, so "small" is a *big* slice. We keep
    the standard thresholds for comparability but **record per-slice GT box counts** (`n_gt`
    in every slice's output) so a thin slice is never misread as a reliable one.
- **Reports are self-describing (versioned slice spec).** `evaluate_slices` emits a `spec`
  block: `slice_spec_version`, IoU/IoA thresholds, scale-bin edges, the weather values and
  camera groups actually present, and class ids. When a "dusk" weather value or a camera-map
  v2 appears later, reports declare which slicing produced them, so cross-version comparisons
  (and the promotion contract's C3, which consumes slices programmatically) do not silently
  break.

## Consequences

- The harness has zero model/GPU dependency — it is pure geometry + AP math, unit-tested
  against pencil-computed answers (perfect, 0.5, NaN-on-no-GT, IoU/IoA, composed-ignore
  precedence, half-open bin edges, partition invariants).
- Bootstrap CIs, corruption curves, and cross-dataset suites reuse this layer unchanged.

## Alternatives rejected

- **pycocotools / Ultralytics `val`:** neither exposes ignore-region FP exclusion nor our
  slice axes under one auditable, freezable definition.
- **Post-resize area for scale bins:** makes the same physical car change slice when `imgsz`
  changes — scale slices would stop being comparable across runs.
- **Inclusive `[lo, hi]` bin edges:** boxes at exactly 32²/96² would fall in two bins,
  breaking the partition invariant.
