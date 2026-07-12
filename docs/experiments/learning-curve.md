# Experiment 1 — Learning curve: is the model data-starved or capacity-limited?

- **Status:** IN PROGRESS — runs launched, numbers pending
- **Date:** 2026-07-12
- **Phase:** 3 (mandatory experiment #1, guide §4/Phase 3.3)

## Question

With the sampled training pool (train-v1: 4,277 train / 861 val images), does yolov8s improve
with more data (data-starved → invest in the flywheel) or has it plateaued (capacity-limited →
more data won't help, the Phase 8 flywheel must mine *harder* examples, not just more)?
This answer drives every later data decision, including what the flywheel is for.

## Method

Four runs, identical config (`configs/training/base.yaml`, defaults; seed 0), differing only
in `fraction` ∈ {0.25, 0.5, 0.75, 1.0} of the training pool. All on Colab T4 via the pinned
`colab run` invocation (ADR 0009), all logged to DagsHub MLflow with full Law 6 provenance
(run names `lc-25`, `lc-50`, `lc-75`, `baseline`). Early stopping: patience 20 on val
mAP50-95, best-epoch checkpointing. `baseline` doubles as the 100% point and the
champion-v1 candidate; `baseline-repro` (same config+seed) measures run-to-run noise, giving
the reproducibility tolerance the curve must be read against.

**Metric caveat (Law 2, stated plainly):** the curve plots Ultralytics **val mAP50-95 on
train-v1's val split** — the guide prescribes exactly this for the study, and it is a
*relative* comparison between identically-measured runs, not a quotable accuracy claim. The
quotable number for the chosen champion comes from the frozen `eval-harness-v1` report only,
and appears in the results table below labelled as such.

## Results (pending)

| fraction | images | best val mAP50-95 | best epoch | epochs run | wall time |
|---|---|---|---|---|---|
| 0.25 | ~1069 | 0.5202 | ~4 | 25 | 12 min |
| 0.50 | ~2139 | TBD | TBD | TBD | TBD |
| 0.75 | ~3208 | TBD | TBD | TBD | TBD |
| 1.00 | 4277 | TBD | TBD | TBD | TBD |
| 1.00 (repro) | 4277 | TBD | — | — | TBD |

Reproducibility tolerance (|baseline − baseline-repro| on best val mAP50-95): **TBD**.

Champion (fraction 1.0) on the frozen harness (`eval-harness-v1`, eval-frozen-v1):
mAP50-95 **TBD** vs zero-shot floor 0.381.

## Early observation (lc-25)

25% of the pool already reaches best-val-mAP ~0.52 by ~epoch 4-5, then plateaus/oscillates —
strong COCO-pretrained transfer. If the curve stays this flat through 100%, the verdict leans
**capacity-/task-limited on in-distribution data**, and the flywheel's value must come from
*harder/rarer* examples (night, rain, small boxes), not volume. To be confirmed.

## Conclusion (pending)

TBD after all four points land: data-starved vs capacity-limited, with the CI-aware caveat
that differences smaller than the repro tolerance are noise.
