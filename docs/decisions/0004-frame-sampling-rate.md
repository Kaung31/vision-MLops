# ADR 0004 — Frame sampling rate (every 10th)

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Annotation Hygiene, Taxonomy, Splits)

## Context

UA-DETRAC is 25 FPS video. Consecutive frames are near-duplicates — training on all of
them wastes compute and inflates apparent val performance (train/val frames from the same
clip are almost identical). A working training pool must sample frames down.

## Decision

`src/data/sample.py::select_every_nth` keeps **every 10th frame** (`SAMPLING_STRIDE = 10`),
the **first** of each group of ten, sampled **per sequence** and deterministically (sort +
stride). Applied to **train/val/test clips only**.

- 82,085 labeled frames → **~8.2k** training-pool frames — within the guide's 8–14k target.
- The **production-holdout keeps full-rate frames** (all 25 FPS) so the Phase 6 streaming
  replay is realistic; sampling is not applied there. This split is enforced at conversion
  (⑦), and called out so it cannot silently regress.

## Rationale

10× is standard practice for UA-DETRAC and removes temporal redundancy while preserving
scene/lighting diversity (sampling is within-clip, so every clip still contributes). Every
metric later traces to a fixed, reproducible pool rather than an arbitrary frame subset.

## Consequences

- The training pool is ~8k images. If the Phase 3 learning-curve study shows the model is
  **data-starved**, the stride is the first knob to revisit (documented, single constant).
- Full-rate frames remain on disk (raw-v1) for the replay producer; only the training pool
  is decimated.

## Alternatives rejected

- **Keep all frames:** 10× compute, and near-duplicate train/val frames make val mAP
  optimistic — exactly the overfitting blind spot this project exists to avoid.
- **Random sampling:** non-deterministic and harder to reproduce; stride sampling gives an
  even temporal spread for free.
