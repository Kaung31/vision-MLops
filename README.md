# traffic-vision-platform

A self-maintaining vehicle-detection MLOps platform: continuous per-slice generalization
evaluation, a gated learning flywheel, a statistical promotion contract, and operated
real-time serving under an SLO.

> **Status: Phase 0 — Foundation & Data Provenance (in progress).**
> This is a disciplined, phase-by-phase build. See
> [`docs/vision-mlops-platform-project-guide.md`](docs/vision-mlops-platform-project-guide.md)
> for the full contract and [`CLAUDE.md`](CLAUDE.md) for the working rules.

## Honest scope (read this)

- **Drift is scripted**, not organic — it is injected via playlist switches and documented as such.
- **The "human labeler" is held-back ground truth** surfaced through a review script.
- **Kafka is single-node**; the single point of failure is documented, not hidden.
- Numbers are only ever quoted from the machine class they were measured on
  (Mac = platform, Colab = training, rented Linux GPU = measurement).

## Non-goals

Not a novel model architecture. Not a labeling product. Not multi-cloud. Not organic-drift detection.

## Results

_Populated from Phase 3 onward — no metric exists until the eval harness is frozen (Phase 2, Law 2)._
