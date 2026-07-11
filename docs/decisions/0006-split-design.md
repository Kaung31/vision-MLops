# ADR 0006 — Split design: group-level, weather-constrained, leakage-guarded

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Annotation Hygiene, Taxonomy, Splits)

## Context

The split is where Laws 3 (splits by camera location & clip, never by frame) and 4 (frozen
eval never trained on) become executable code. It consumes the 27 hand-audited camera
groups (ADR 0005) and must produce train / val / test / production-holdout manifests that
no camera can leak across.

## Decisions

### Group-level assignment (stricter than the guide, deliberately)
Every camera group goes **entirely to one split** — all its clips move together. This is
stricter than the guide's "split train/val/test by clip within the training locations":
here no location appears in two splits at all, so test measures **held-out-camera**
generalization, not held-out-clip. It reconciles Law 3's "location and clip" cleanly —
"location" ≡ group (ADR 0005), and because clips never separate from their group, clips
never leak either. `src/data/split.py`.

### Sequence-count proportion targeting (not group count)
Groups range 1–6 sequences, so targeting *group* counts starved train (22/60 seqs, smaller
than the holdout). Instead we target **sequence** proportions — train 0.58 / prod_holdout
0.20 / val 0.12 / test 0.10 — and assign whole groups by **LPT** (largest group first, to
the split furthest below its sequence quota), reserving the *smallest* qualifying weather
groups first so big cameras stay free to balance. Result: **train 35 / prod_holdout 12 /
val 7 / test 6** sequences (=60). Deterministic from `configs/split.yaml` (seed).

### Weather coverage is a hard constraint
Rain is scarce — only **4** of 27 groups have rain. A naive shuffle can starve a split, and
then the harness weather slices and the Phase 7 drift playlist evaluate on nothing. So
`split.py` **fails loud** unless **test and prod_holdout each contain ≥1 night and ≥1 rain
group**. (val is left unconstrained by choice; it happened to get night but no rain — fine
for model selection.)

### Two guards, as code not memory
- **`assert_no_sequence_leakage`** (Law 3): no clip in two splits. A corrupted manifest
  makes it raise — proven by `test_corrupting_a_manifest_trips_the_leakage_guard`.
- **`assert_trainable`** (Law 4): the training data loader (Phase 3) calls this with the
  clips it is about to load; any clip outside the train split raises. Proven by
  `test_training_loader_guard_refuses_eval_clips`.

### Full-rate holdout
Per ADR 0004, sampling (every 10th frame) applies to **train/val/test only**; the
production-holdout keeps **full-rate** frames for a realistic Phase 6 streaming replay.
`split.py` emits clip lists; the frame-rate distinction is enforced at conversion (⑦).

## Consequences

- `configs/splits.yaml` is a committed, regenerable manifest; a test asserts it is not
  stale (equals a fresh `build_split()`), so hand-edits or seed drift are caught.
- If the Phase 3 learning-curve shows train is data-starved, the proportions are the knob
  (single config change → regenerate).

## Dataset materialization (⑦) and the read-only mechanism

`src/data/convert.py` emits three DVC versions from the manifests:

- **Read-only is a sentinel file, not chmod.** `eval-frozen-v1` and `prod-holdout-v1` each
  carry a `.readonly` file whose *contents* state why. Every dataset write routes through
  `datasets.open_dataset_dir(path, "w")`, which refuses a writable handle when the sentinel
  exists — a new writer cannot forget the check because it cannot get a handle without it.
  The sentinel survives `dvc pull`/checkout on Colab and the rented box; chmod bits do not.
  Local chmod is at most a bonus. This is Law 4 as code, not memory.
- **prod-holdout images are referenced, not duplicated.** Duplicating ~GB of full-rate
  frames that already sit checksummed in `raw-v1` buys nothing. `prod-holdout-v1` ships
  sequestered YOLO labels + ignore JSONs + a **content-addressed `frame_manifest.yaml`**
  that pins the exact archive dirs, the **raw-v1 DVC md5** (`8e5708b5…dir`) so it is
  anchored to an immutable version rather than "whatever raw-v1 is today", and per-sequence
  frame counts as a cheap integrity check the Phase 6 producer verifies at startup.
- **Operational notes.** (1) The producer must be **benchmarked reading from the zip at
  target rate** (4×25 = ~100 JPEG reads/s) on the rented box — measured, not assumed, since
  that machine's disk is unknown. (2) If zip random-access ever bottlenecks, the fix is a
  one-time **extract-to-scratch on the serving box — a runtime cache, not a fourth dataset
  version.** Nobody should "solve" it by materializing new frames.
- **val lives in `train-v1`** (standard YOLO `data.yaml` with train+val paths). Law 4
  protects test and prod-holdout, not val (it is an early-stopping/model-selection signal,
  never a gated surface). The one guard: a test asserts `train ∩ val = ∅` so sharing a
  directory never rots into leakage, and `assert_trainable` stays **clip-based**, so "it
  lives in train-v1" never becomes the test for "gradients may see it."

## Alternatives rejected

- **Frame-level split:** data leakage (adjacent 25 FPS frames are near-identical) — Law 3.
- **Clip-within-location (guide default):** allows the same camera in train and test;
  weaker generalization measure than held-out-camera.
- **Group-count targeting:** starves train (variable group sizes) — the reason we switched.
- **Random seed with no weather constraint:** can leave test/holdout with no rain, silently
  emptying weather slices.
