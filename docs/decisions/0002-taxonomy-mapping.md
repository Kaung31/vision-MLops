# ADR 0002 — Canonical taxonomy and per-dataset class mapping

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Annotation Hygiene, Taxonomy, Splits)

## Context

The platform trains on UA-DETRAC (4 native classes) and evaluates cross-dataset on
MIO-TCD (11), AAU RainSnow (COCO-80), and VisDrone-DET (12 category ids). These must
collapse into one canonical class set, and every conversion must go through a single
config — no hardcoded class IDs (guide §Phase 1). The guide requires auditing what
UA-DETRAC's `others` contains **before** merging it, not assuming.

## Decision

### Canonical classes
`0 car`, `1 bus`, `2 van_truck`. Three coarse vehicle types the surveillance viewpoint
supports reliably, each large enough to evaluate with tight CIs.

### Routing model (`configs/taxonomy.yaml` + `src/data/taxonomy.py`)
Each dataset routes every native class under exactly one verb:
- **map** — native → canonical.
- **ignore** — don't-care: not a target *and* excluded from false-positive counting
  (treated like an ignore-region by the Phase 2 harness).
- **exclude** — dropped entirely (a genuinely different object).

Any native class **not listed fails loud** (`UnknownNativeClass`) at conversion. This is
the guard that makes "nothing is silently dropped" executable rather than aspirational —
a new/renamed native class stops the pipeline instead of vanishing.

### UA-DETRAC: `van` + `others` → `van_truck` (audited, not assumed)
Audit of the 60 train sequences (598,281 boxes):

| class | count | median box area | median height | in #seqs |
|---|---|---|---|---|
| car | 503,853 | 3,417 | 47 | 60 |
| van | 57,051 | 4,651 | 61 | 53 |
| **others** | **3,726 (0.62%)** | **8,961** | **82** | **27** |
| bus | 33,651 | 21,579 | 120 | 47 |

`others` sits geometrically between van and bus (closer to van), carries nothing small
(p10 area 1,808 > car's 842), and is scattered across 27/60 sequences — i.e. genuine
large/truck-like vehicles, not an artifact. At 0.62% it is far too small to be its own
reliable class, so it merges with `van` into `van_truck`.

### MIO-TCD: `motorized_vehicle` → **ignore** (not exclude)
`motorized_vehicle` (26k boxes) is MIO-TCD's ambiguous catch-all, used when the annotator
could not determine the specific vehicle type. Ignoring it means a correct
`car`/`van_truck` detection landing on one is **not** counted as a false positive on this
gated eval set; **excluding** it instead would let the model hallucinate a vehicle there
for free. Ignore is the honest, harder-on-the-model choice. Trucks
(`pickup_truck`, `work_van`, `single_unit_truck`, `articulated_truck`) → `van_truck`;
`pedestrian`/`bicycle`/`motorcycle`/`non-motorized_vehicle` → exclude.

### AAU RainSnow: COCO ids, `truck` → `van_truck`
RainSnow annotations use COCO-80 category ids; only 6 are actually annotated
(car, bicycle, truck, person, bus, motorbike). `3 car`→car, `6 bus`→bus, `8 truck`→
van_truck (COCO has no `van`); `person`/`bicycle`/`motorbike` → exclude. It is video +
instance masks, converted (frames + mask→bbox) as a frozen eval set in Phase 2.

### VisDrone: ids `0` and `11` → **ignore**
VisDrone's own DET protocol excludes category `0` (ignored regions) and `11` (others)
from scoring, so both are routed to `ignore`. `4 car`→car, `9 bus`→bus, `5 van`+`6 truck`
→van_truck; the person/bicycle/tricycle/motor ids → exclude. Ungated stretch, deferred.

## Consequences

- **`van_truck` is deliberately heterogeneous** (vans through articulated trucks). If a
  slice ever needs finer granularity we can split it, but per-class CIs would widen.
- **`ignore` classes feed the eval harness** (Phase 2): detections inside MIO-TCD
  `motorized_vehicle` / VisDrone 0,11 boxes are excluded from FP counts, exactly like
  UA-DETRAC ignore-regions.
- The fail-loud guard means RainSnow/VisDrone tables are safe to ship now: if the real
  labels contain a class we didn't enumerate, conversion stops rather than mis-scoring.

## Alternatives rejected

- **Keep `others`/trucks as a 4th class:** 0.62% of boxes → CIs too wide to gate on.
- **Exclude `motorized_vehicle`:** rewards hallucination on a gated set (see above).
- **Per-dataset `default: exclude` for unlisted classes:** unnecessary — every dataset's
  actually-used vocabulary is small and fully enumerable, so uniform fail-loud is simpler
  and safer.
