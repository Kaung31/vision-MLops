# ADR 0005 — Camera groups: the hand-audited "location" for UA-DETRAC

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Annotation Hygiene, Taxonomy, Splits)

## Context

Law 3 requires splitting "by camera location and clip," but UA-DETRAC publishes **no
sequence→location map**, the authors' host is dead, and neither sequence-number prefixes
(11 uneven session batches) nor ignore-region signatures (56/60 unique) recover the 24
locations. "Location" therefore has to be *derived and hand-audited*, then frozen as an
artifact the splitter treats as ground truth.

## Method

1. **Temporal-median background per sequence** — sample 41 frames, take the pixel-wise
   median; moving vehicles cancel, leaving the static scene (road geometry, medians,
   fences, gantries, buildings), which survives day/night/rain far better than any single
   frame. (`scripts/derive_camera_groups.py`)
2. **Draft clustering** by standardized-grayscale-background correlation — a *draft only*.
3. **Lighting-invariant edge re-pass** (`scripts/edge_match.py`): CLAHE→Canny edge maps,
   spatial edge overlays, and a night→daytime candidate shortlist.
4. **Human audit** over a contact sheet + edge overlays; the committed grouping is the
   audit result, not the clustering.

**Honest note on the edge scores.** The numeric edge-correlation shortlist came out
**unusable** (max ~0.28 even for confident same-camera pairs; near-zero elsewhere) —
sparse dilated edges plus small framing shifts between recordings destroy spatial
correlation. We reported the numbers as noise and fell back to **visual edge overlays**
(fixed structures show as aligned/yellow) rather than force matches from bad scores.

## The decision rule (and why the asymmetry makes the audit tractable)

Failure directions are **asymmetric**: merging two truly-different cameras only makes the
split *more conservative* (fewer independent groups) — it **cannot** cause leakage;
failing to merge two same-camera sequences puts one in train and one in eval — that **is**
the leakage. So: after honest eyeballing, **uncertain → merge**, and the human audit only
has to hunt one error type (under-merges). Every merge is anchored to **at least one fixed
structure** (fence line, median type, gantry, building silhouette, curvature) — geometry,
not "same kind of road." Weather/lighting change colour and texture; they do not move lane
counts, medians, or fences.

## Result: 27 groups (15 multi + 12 singletons)

Anchored merges:

| group | sequences | fixed-structure anchor |
|---|---|---|
| cam_wideave_complex | 20011,20012,20032,20033,20034,20035 | central median, one avenue (see override #3) |
| cam_redbarrier | 20061,20062,20063,20064,20065 | red roadside barriers |
| cam_fence5 | 40152,40161,40162,40171,40172 | top stall strip + white picket-fence median |
| cam_hedgewall | 63552,63553,63554,63561,63562,63563 | diagonal hedge + concrete wall + curvature |
| cam_metalmed | 40131,63521,63525 | central metal-post fence median (see override #4) |
| cam_bldgroad 39801,39811,39821 · cam_hedge_b 40211,40212,40213 · cam_hedge_c 40241,40243,40244 | | building/road-edge / hedge runs |
| cam_redbanner_a · cam_treemed · cam_greenmed · cam_hedge_a · cam_billboard · cam_nightblue · cam_pillar_r | (pairs) | banners / tree-median / green-median / hedge / gantry billboards / wet-blue road / right pillar |

Singletons (12): 39781, 39851, 39861, 39931, 40141, 40181, 40752, 40871, 40981, 41063,
41073, 63544 — night/rain isolates with no confident daytime match are left singleton
(leakage-safe: an unmatched clip in its own group cannot leak).

## The two override decisions

- **#4 — 40131 → cam_metalmed (weakest merge on the board).** The overlay shows the
  distinctive central metal-post median aligned along the road axis; the misaligned
  peripheries are what heavy rain does to vegetation/depth-of-field (63521/525 are rainy,
  40131 dry). "Borderline" is the rule's trigger — uncertain → merge; if wrong, the only
  cost is a slightly stricter split. Recorded as the weakest merge, anchored to the median.
- **#3 — cam_underpass folds into cam_wideave_complex.** A distinguishing structure (the
  tunnel mouth) proves the *fields of view* differ, not that the *cameras* differ, and not
  that the scenes are visually independent — which is what leakage cares about. "Same
  avenue" is the uncertainty admission; the rule has no carve-out. The name flags it:
  **over-grouped by design; may span two physical cameras on one avenue.**

## Honest residual

Conservative in one direction only: singletons might secretly share a camera we could not
visually confirm (disclosed; the direction that risks leakage — but every singleton is a
night/rain or unique-geometry clip we could not match, so the risk is small and named).
Visually-indistinguishable *different* cameras may have merged — harmless (stricter split).

## Consequence: weather coverage feeds split.py

Per-group `weathers` (auto-derived from DETRAC attributes) are stored in
`configs/camera_groups.yaml`. Distribution across the 27 groups: **10 night groups, only
4 rain groups** (cam_metalmed, cam_hedgewall, cam_40871, cam_63544). The scarcity of rain
is why `split.py` (⑤b) must **constraint-check** group assignment — test *and*
production-holdout each require ≥1 night and ≥1 rain group, or the harness weather slices
and the Phase 7 drift playlist evaluate on nothing. Recorded in ADR 0006 (split design).

**Law 5:** median backgrounds and contact sheets contain dataset imagery and are
git-ignored (`data/camera_work/`); only the derived grouping is committed.
