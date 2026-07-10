# Dataset Licenses & Provenance

> **No dataset images or videos are redistributed in this repository (Law 5).**
> This repo holds only code, configs, checksums, DVC pointers, and download scripts.
> Frames are fetched locally by `src/data/download.py` and tracked as DVC pointers.
> Every archive's SHA256 is pinned in
> [`configs/datasets/checksums.lock.yaml`](configs/datasets/checksums.lock.yaml).
> Availability and mirror choices are recorded in
> [ADR 0001](docs/decisions/0001-dataset-selection-and-mirrors.md). Verified 2026-07-10.

## UA-DETRAC — primary train/val/test

- **License / terms:** Non-commercial academic research use (original UA-DETRAC terms).
- **Frames source:** Kaggle community mirror `bratjay/ua-detrac-orig` (images only).
- **Annotations source:** original author's Google Drive release
  `DETRAC-Train_Annotations-XML.zip` (id `12xJc8S0Z7lYaAadsi2CoSK3WqH2OkUBu`).
- **Provenance chain:** original RIT/Albany hosting is dead → author site
  `sites.google.com/view/daweidu/projects/ua-detrac` → Kaggle mirror (frames) + author
  Google Drive (annotations). Only the **60 training sequences** carry public XML
  annotations; test-set labels were withheld in the original benchmark.
- **Integrity verified against the paper (2026-07-10):**
  - Full benchmark in the frames archive: **100 sequences, 140,131 frames** (~100 / ~140k ✓).
  - Public labeled train set: **60 sequence XMLs, 82,085 labeled frames, 598,281 boxes**.
  - **4 vehicle classes** present: `car` (503,853), `van` (57,051), `bus` (33,651),
    `others` (3,726).
  - **4 weather categories** present (`sence_weather`): cloudy (19), night (16),
    sunny (15), rainy (10) sequences.
  - **Ignore regions** present (`<ignored_region>` per sequence).

## MIO-TCD (Localization) — cross-dataset eval (gated)

- **License / terms:** **CC BY-NC-SA 4.0**. Attribution required — cite Luo et al.,
  "MIO-TCD: A new benchmark dataset for vehicle classification and localization,"
  IEEE TIP, 2018.
- **Source:** official host `https://tcd.miovision.com/static/dataset/MIO-TCD-Localization.tar`
  (direct download, no login; verified `HTTP 200`, 3.48 GiB).
- **Contents:** 137,743 frames with bounding boxes across 11 traffic-object categories
  (mapped to our taxonomy in Phase 1).

## AAU RainSnow — cross-dataset eval (gated)

- **License / terms:** **CC BY 4.0** (Attribution 4.0 International, per the Kaggle
  dataset page). Cite Bahnsen & Moeslund, Aalborg University, 2018.
- **Source:** Kaggle `aalborguniversity/aau-rainsnow` (verified available; requires
  Kaggle login).
- **Contents:** instance-level annotations of road users in RGB + thermal video,
  22 five-minute clips across seven Danish intersections under rain/snow/night.

## VisDrone-DET 2019 — cross-dataset eval (ungated stretch, DEFERRED)

- **License / terms:** VisDrone dataset terms, academic/research use. Cite Zhu et al.
- **Source:** official VisDrone GitHub Google Drive links
  (`github.com/VisDrone/VisDrone-Dataset`).
- **Status:** **deferred.** On 2026-07-10 Google Drive refused automated download
  (quota/permission wall) — a known VisDrone issue. VisDrone is *ungated* (reported,
  never gates promotion) and is not consumed until the Phase 2 harness, so it is
  excluded from `raw-v1` and will be re-sourced then (Kaggle mirror / cookies / manual).

---

**Redistribution statement.** This repository does not contain, host, or redistribute
any dataset frames or videos. Each dataset is obtained by the user directly from its
source under that source's license. Datasets are used for non-commercial research only.
