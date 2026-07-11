# ADR 0003 — Ignore-region handling and the replay-masking decision

- **Status:** Accepted
- **Date:** 2026-07-11
- **Phase:** 1 (Annotation Hygiene, Taxonomy, Splits)

## Context

UA-DETRAC annotates per-sequence **ignored regions** — rectangles containing vehicles
too low-resolution to label. They must be handled two ways (guide 2a/2b): blacked out on
training images, and exported so the eval harness can exclude detections inside them from
false-positive counts. Mishandling them corrupts both training and every future metric.

## Decisions

### Rectangles, kept as float `xywh`
DETRAC ignore regions are axis-aligned `<box left top width height/>` rectangles, not
free-form polygons. `src/data/hygiene.py::parse_ignore_regions` returns them as float
`IgnoreBox`es and **returns `[]` when a sequence has no `<ignored_region>`** (a real,
common case) rather than raising.

### Black-fill pixel convention (pinned by tests)
`black_fill()` maps each float box to pixels as **top-left floored, bottom-right ceiled,
clipped to the image, filled as the half-open slice `[y0:y1, x0:x1]`** (right/bottom
exclusive). This deliberately *over-covers* by up to one pixel at fractional edges:
under-masking would leak unlabeled vehicles into training, which is worse than blacking
one extra edge pixel. The convention is frozen by `test_black_fill_float_rounding_is_pinned`
— off-by-one is decided by a test, not by accident, because it would otherwise leak
silently into Phase 2 FP counting.

### Self-describing export (guide 2b)
`export_ignore_regions()` writes one JSON per sequence carrying a header:
`schema_version`, `coordinate_convention`
(`xywh_px_topleft_origin_rb_exclusive_float`), and the sequence's `image_width`/`image_height`
(UA-DETRAC frames verified at **960×540**). The Phase 2 harness consumes these blind weeks
from now; the header removes any xywh-vs-xyxy / origin ambiguity — the classic silent eval
bug. The stored boxes remain **float** (no rounding baked in), so FP-exclusion geometry is
exact; only the training black-fill rounds to pixels.

### Streaming-replay frames are UNMASKED (recorded now, not decided implicitly at ⑦)
The full-rate frames replayed as the simulated live stream in Phase 6 are served
**unmasked** — real production cameras do not emit black rectangles, so masking them would
make the "production" stream unrealistic. Eval fairness on the sequestered truth-sample is
already handled by 2b's FP-exclusion (detections inside ignore regions are not counted),
so masking the stream would buy nothing there. The consequence is a **deliberate
train/serve skew**: the model trains on ignore-masked images but serves on unmasked ones,
so it may fire inside ignore regions at serving time. That is accepted and called out here
(and will be in the README's honest-scope section); it is the realistic choice, and the
metric that matters is measured with FP-exclusion regardless. Masking-everywhere for
train/serve consistency was the reasonable alternative; it was rejected for realism, but
the point is that this is a recorded decision, not a default that fell out of `convert.py`.

## Consequences

- `black_fill` operates on numpy arrays (dtype/channel agnostic); image decode/encode
  happens in `convert.py` (⑦) — see the image-library note below.
- The ignore-region JSONs are derived artifacts, materialized under DVC alongside the
  converted dataset at ⑦, not committed to git.

## Note: one image library (OpenCV), introduced when first needed

`hygiene.py` needs only numpy for the fill. Image **decode/encode** (⑦ `convert.py`) and
**video decode** (Phase 6 `producer.py`) will both use **OpenCV** (`cv2`), which
Ultralytics pulls in anyway — so the project carries one image library, not Pillow *and*
cv2. cv2 is added at ⑦ when the first real image I/O happens, not before.
