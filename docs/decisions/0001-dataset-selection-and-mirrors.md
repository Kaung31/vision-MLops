# ADR 0001 — Dataset selection, mirrors, and provenance

- **Status:** Accepted
- **Date:** 2026-07-10
- **Phase:** 0 (Foundation & Data Provenance)

## Context

The frozen stack (guide §2) mandates UA-DETRAC (primary), MIO-TCD + AAU RainSnow
(gated cross-dataset eval), and VisDrone-DET (ungated stretch). The guide requires
verifying availability **on Day 1** and, if either *gated* set is unreachable,
substituting BMD-45 with an ADR. Availability was checked on 2026-07-10.

## Availability findings (2026-07-10)

| Dataset | Role | Reachable? | Source | Access |
|---|---|---|---|---|
| UA-DETRAC | primary | yes | Kaggle `bratjay/ua-detrac-orig` (frames) + author Google Drive (annotations) | Kaggle login |
| MIO-TCD (Localization) | gated eval | yes — `HTTP 200`, 3.48 GiB | `https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar` | none |
| AAU RainSnow | gated eval | yes | Kaggle `aalborguniversity/aau-rainsnow` | Kaggle login |
| VisDrone-DET 2019 | ungated stretch | yes | Google Drive (official VisDrone GitHub) | none |

## Decisions

1. **No BMD-45 substitution.** Both *gated* cross-datasets (MIO-TCD, AAU RainSnow)
   are reachable, so the guide's substitution trigger does not fire.

2. **UA-DETRAC = two sources (frames + annotations).** The Kaggle mirror
   `bratjay/ua-detrac-orig` is **images only** (`DETRAC-Images/.../MVI_*/img*.jpg`,
   ~140k frames), verified by listing its files. The annotation XML — which carries
   the **ignore regions** and **weather attributes** Phase 1/2 depend on — is fetched
   separately from the original author's Google Drive release
   (`DETRAC-Train_Annotations-XML.zip`, id `12xJc8S0Z7lYaAadsi2CoSK3WqH2OkUBu`).

3. **Only the 60 UA-DETRAC train sequences are usable as labeled data.** Test-set
   annotations were withheld in the original benchmark and are not publicly released.
   The labeled working set is therefore the ~84k-frame train split spanning ~24 camera
   locations — which is exactly what Phase 1's location/clip split consumes. The
   unlabeled test frames are retained only as optional streaming-replay background.

4. **Provenance risk (dead official host, Google Drive) is mitigated by SHA256
   pinning, not trust.** Original RIT/Albany hosting is dead; we rely on a community
   mirror + Google Drive, both mutable. `src/data/download.py` pins each archive's
   SHA256 on first verified fetch (`configs/datasets/checksums.lock.yaml`) and refuses
   to proceed on any later mismatch — so a silently changed or rotted mirror fails
   loudly (Law 6).

5. **VisDrone is reported, never gates.** Aerial viewpoint ≈ different task; it stays
   in the harness for reporting only and never participates in the promotion contract.

## Consequences

- Two fetch mechanisms are needed: the `kaggle` CLI (UA-DETRAC frames, RainSnow) and
  direct/`gdown` HTTP (MIO-TCD, UA-DETRAC annotations, VisDrone). `kaggle` + `gdown`
  live in the uv `data` dependency group; CI does not install them.
- SHA256 values are pinned at Phase 0 Step D (first real download) and committed.
- `LICENSES.md` records per-dataset license terms and this provenance chain; **no
  frames are redistributed in the repo** (Law 5).

## Alternatives rejected

- **Roboflow/other re-exports of UA-DETRAC:** they reshuffle/re-split frames and drop
  the native XML (ignore regions, attributes), which would break the location/clip
  split (Law 3) and ignore-region handling. Rejected.
