# Dataset Licenses & Provenance

> **No dataset images or videos are redistributed in this repository (Law 5).**
> This file records the license terms, source, and provenance of every dataset the
> platform consumes. Frames are downloaded locally via `src/data/download.py` and
> tracked with DVC pointers only.

_Filled in at Phase 0 Step E, once dataset mirrors are verified (Step B) and downloaded (Step D)._

## UA-DETRAC (primary train/val/test)

- **License:** _TBD — record exact terms (non-commercial research)._
- **Official source:** original hosting is dead.
- **Mirror used:** _TBD (Step B) — URL + SHA256 pinned in `configs/datasets/`._
- **Provenance chain:** _TBD — how the mirror traces back to the original release._

## MIO-TCD (cross-dataset eval, gated)

- **License:** _TBD._
- **Source / availability:** _TBD (Step B). If unreachable, substitute BMD-45 + ADR._

## AAU RainSnow (cross-dataset eval, gated)

- **License:** _TBD._
- **Source / availability:** _TBD (Step B). If unreachable, substitute BMD-45 + ADR._

## VisDrone-DET (cross-dataset eval, ungated stretch)

- **License:** _TBD._
- **Source / availability:** _TBD (Step B)._

---

**Redistribution statement:** This repository contains code, configs, checksums, DVC
pointers, and download scripts only. It does not contain, host, or redistribute any
dataset frames. Each dataset is obtained by the user directly from its source under
that source's license.
