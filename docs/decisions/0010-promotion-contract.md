# ADR 0010 — The promotion contract (CI-overlap gating, no override)

- **Status:** Accepted
- **Date:** 2026-07-13
- **Phase:** 5 (The Promotion Contract)

## Context

A challenger model must never replace a better champion. The decision has to be **mechanical**
(reproducible, auditable, no human judgement in the loop) and **statistically honest** (robust
to the noise that a single mAP number hides). `src/registry/promote.py` +
`configs/promotion.yaml` encode it; it runs on two frozen `eval-harness-v1.1` reports, so the
contract has zero model/GPU dependency and is unit-tested with tiny fixtures.

## Decisions

### CI-overlap test, not point thresholds

A challenger is "significantly worse" on a metric **only when its entire 95% bootstrap CI sits
below the champion's point estimate** (`challenger.ci_high < champion.point`). This is the crux
of the whole phase:

- A **naive point threshold** (`challenger.point < champion.point − ε`) fires on noise. On a
  thin slice like `bus` (n≈105, CI width ≈ ±0.07), a run that is truly equal will land a few
  points either side by chance — a point threshold would reject perfectly good models half the
  time. The bootstrap CIs (Phase 2) exist precisely to separate signal from noise; the contract
  must *use* them, not throw the information away.
- Promotion still also requires the point estimate `≥ champion − ε` (ε default 0), so a
  challenger cannot be promoted on a technicality while being visibly, if not significantly,
  worse. "Not significantly worse **and** not pointwise worse" — both.

### Small-slice warning policy

C3 blocks only on **large** slices (≥ `c3_min_box_count`, default 500 GT boxes). A regression on
a slice below that threshold is reported as a **warning** (visible in the decision report) but
**never blocks**. Small slices have CIs too wide to trust; letting `bus` (n=105) or `rainy`
(n=203) veto a genuinely better model would make the contract hostage to noise. The warning
keeps the signal visible for a human to watch across cycles without giving it a veto.

### The five clauses

- **C1** in-distribution mAP50-95: CI-overlap + point ≥ champion − ε.
- **C2** gated cross-dataset: size-weighted aggregate not worse by more than ε, **and** no single
  gated suite drops more than `c2_catastrophic_drop` (0.05) — the catastrophic-regression guard.
- **C3** no large slice regresses significantly (per-class/weather/camera/scale); small slices warn.
- **C4** corruption robustness: challenger's degradation-curve area (mean mAP50-95 over
  corruptions × severities) within `c4_corruption_tolerance` of the champion's.
- **C5** latency/throughput SLO: **stubbed** (clearly marked) until Phase 6 builds the serving
  stack, so the contract is structurally complete and honest about what is not yet measured.

### No override flag

There is no flag, env var, or UI gesture that promotes a rejected challenger or blocks an
accepted one. To override a decision you **edit `configs/promotion.yaml` and commit it** — that
commit is the audit trail. Promotion reassigns the `@champion` alias to the challenger's
version; **rollback is the same call in reverse** (reassign `@champion` to the prior version),
one auditable line, no code redeploy. Every attempt logs a per-clause decision report (JSON +
human) to the `traffic-vision-promotion` MLflow experiment.

## Consequences

- **Live validation:** run on the real champion-v1 (v1) vs the tuned-v1 challenger (v2), the
  contract returned **PROMOTE** — all five clauses pass, C3 found 0 significant large-slice
  regressions (the sunny/rainy dips seen by eye were non-significant, wide-CI), C4 showed tuned-v1
  *more* corruption-robust (0.584 vs 0.533). The mechanism agreed with the by-eye read, and
  flipped `@champion` v1 → v2. Decision artifact attached to the attempt in MLflow.
- MLflow reachability for this (and all) tracking/registry is DagsHub — decided in ADR 0009.
- **CI automation deferred:** wiring the contract into a GitHub Actions `train.yml` on a
  self-hosted Mac runner + Colab GPU step is scoped to Phase 8's `retrain-trigger`, where it runs
  end-to-end naturally (Kaung's call, 2026-07-13). The contract already runs end-to-end via the
  CLI; Phase 8 only changes the trigger from manual to automated.

## Alternatives rejected

- **Naive point thresholds:** reject good models on noise; discard the CI information.
- **Block on any slice regardless of size:** thin slices (bus, rainy) would veto better models.
- **A human override flag "for convenience":** destroys the audit trail; the config-commit path
  is the only sanctioned override.
