# Production Vision MLOps Platform — The Strict Project Guide

**Project codename:** `traffic-vision-platform`
**Student:** Kaung Min Khant
**Discipline level:** This guide is a contract. Every phase has entry conditions, exit criteria, and forbidden shortcuts. You do not move to the next phase until the current phase's exit criteria are met and committed. If you catch yourself thinking "I'll come back to this later" — stop. That thought is how every abandoned portfolio project died.

---

## 0. The Laws (read before every work session)

These apply to the entire project. Violating any of them invalidates work built on top of it.

**Law 1 — The stack is frozen.** You use only what is listed in Section 2. No swapping Kafka for Redis Streams because you already know it. No "quick Streamlit dashboard instead." The stack was chosen deliberately after research; substitutions destroy the point of the project.

**Law 2 — No metric exists until the eval harness says it does.** You do not quote, log, or believe any mAP number produced before Phase 2 is frozen. Numbers from unaudited data are noise.

**Law 3 — Splits are by camera location and clip. Never by frame.** UA-DETRAC is 25 FPS video. Adjacent frames are near-duplicates. A frame-level split is data leakage and makes every result a lie. The splitter is code, with tests, and nothing bypasses it.

**Law 4 — The frozen eval sets are never trained on. Ever.** Not for fine-tuning, not for "just one experiment," not in the flywheel. A file-path guard in the training loader enforces this in code (assert no eval clip IDs in the training manifest) — not by your memory.

**Law 5 — No images in git.** The repo holds code, configs, DVC pointers, checksums, and download scripts only. Dataset licenses are non-commercial research; you document them and you never redistribute frames.

**Law 6 — Every training run is reproducible or it did not happen.** Config file + dataset version hash + git commit + seed, all logged to MLflow. A run you cannot reproduce is deleted.

**Law 7 — Honest documentation.** The drift in this system is scripted by you (playlist switches), not organic. The "human labeler" is you with held-back ground truth. The README says both, plainly. Overclaiming is how portfolios get dismantled in interviews.

**Law 8 — Vertical slices.** At the end of every phase the system runs end-to-end at its current depth. You never have two half-built layers at once.

---

## 1. Problem Statement and Goals (what you are building and why)

**Problem.** A vehicle-detection model trained once on fixed data degrades silently in production: camera, lighting and weather distributions shift; a single test-set mAP hides overfitting and generalization failure; and naive retraining on new data can silently make the model worse on conditions it used to handle.

**Goal.** A self-maintaining detection system with four provable properties:

1. **Measured generalization** — accuracy is continuously evaluated on data (cameras, datasets) the model never trained on, per-slice, not just in aggregate.
2. **A gated learning loop** — the system mines valuable frames from its own production stream, gets them labeled, retrains, and the retrained model passes a statistical promotion contract before serving.
3. **Monotonic quality** — no model version can replace a better one. The promotion contract makes this mechanical, not aspirational.
4. **Operated, not just deployed** — real-time serving under an SLO, with observability, drift detection, shadow rollout, and a written runbook for failure.

**Non-goals (write these in the README too).** Not a novel model architecture. Not a labeling product. Not multi-cloud. Not organic-drift detection — drift scenarios are injected deliberately and documented as such.

---

## 2. The Frozen Stack

### Data
| Purpose | Choice | Notes |
|---|---|---|
| Primary train/val/test | **UA-DETRAC** | Pinned community mirror (official hosting is dead). Record mirror URL + SHA256 in DVC. Verify counts vs paper (~100 sequences, ~140k frames, 4 classes, weather labels). |
| Cross-dataset eval (gated) | **MIO-TCD**, **AAU RainSnow** | Frozen eval-only. Verify download availability on Day 1 — if either is dead, replace with BMD-45 and document the substitution. |
| Cross-dataset eval (ungated stretch) | **VisDrone-DET** | Reported, never gates promotion (aerial viewpoint = near-different task). |
| Data versioning | **DVC** | Every dataset state is a tagged version with checksums. |
| Labeling | Held-back ground truth via a review script (Tier 1) → **Label Studio** (Tier 3) | |

### Model, training, tuning
| Purpose | Choice | Notes |
|---|---|---|
| Detector | **Ultralytics YOLO** (nano/small for tuning, small/medium final) | AGPL-3.0 — fine, repo is open source. |
| Experiment tracking + registry | **MLflow** | Runs, metrics, artifacts; model version **aliases** (`@champion`, `@challenger`) — do NOT use registry "stages", they are deprecated since MLflow 2.9. |
| Hyperparameter tuning | **Ultralytics `model.tune()` + Ray Tune + Optuna search + ASHA early stopping** | Hard GPU-hour cap in config. |
| Pipeline orchestration | **GitHub Actions** (Tier 1) → **Argo Workflows** (Tier 3) | |

### Serving and streaming
| Purpose | Choice | Notes |
|---|---|---|
| Stream transport | **Kafka** | JPEG-in-message first (tuned size limits); migrate to claim-check pattern (**MinIO** + pointers) as a documented scaling step. |
| Inference server | **NVIDIA Triton** | Dynamic batching; export YOLO to ONNX/TensorRT. |
| Streaming rollout | **Shadow deployment** (second consumer group) | Canary %-splitting does not apply to Kafka consumers — know why. |
| HTTP inference endpoint | **KServe** with canary (Tier 3) | Exists to demonstrate request-based canary where it genuinely applies. |
| Containers / infra | **Docker** → **Kubernetes (k3s local) + Terraform** (Tier 2) | Terraform manages in-cluster resources (kubernetes/helm providers) and/or a cloud-VM variant — it does not install local k3s itself. |

### Monitoring and evaluation
| Purpose | Choice | Notes |
|---|---|---|
| Eval harness | **Custom (you write it)** | Slices, cross-dataset, corruption curves, bootstrap CIs. This is the project's core IP. |
| Drift detection | **Evidently** on embedding features | You build the embedding-extraction step; Evidently consumes tabular features. |
| Metrics/observability | **Prometheus + Grafana** | SLO dashboards, alerting. |
| CI/CD | **GitHub Actions** | Lint (ruff), types (mypy), tests (pytest), pipeline triggers. |

### Tier rule
- **Tier 1 (core, mandatory):** DVC, MLflow, eval harness, promotion contract, Kafka, Triton, Docker, GitHub Actions, Prometheus/Grafana.
- **Tier 2:** k3s + Terraform, Evidently drift, shadow deployment.
- **Tier 3:** Argo, KServe HTTP canary, Label Studio, MinIO claim-check.
- You may not start a higher tier while a lower tier has failing exit criteria.

### Hardware strategy (Apple Silicon dev machine — no CUDA)

Three machines, three roles. Numbers are only ever quoted from the machine class they were measured on.

1. **MacBook (M-series) — the platform machine.** Repo, DVC, data hygiene, splitters, eval-harness code, MLflow server, and the full serving stack in Docker (Kafka, Prometheus, Grafana, consumers — all fine on ARM). Quick training smoke-tests use Ultralytics `device="mps"` natively (never in Docker — macOS containers cannot access the Apple GPU). Triton may be attempted CPU-only locally but is known to be unreliable under emulation on Apple Silicon (documented abort-on-request failures); it is a dev convenience only, never a source of numbers. If it won't run, develop the consumer against a thin ONNX Runtime stub exposing the same gRPC contract, and note it in an ADR.
2. **Colab Pro — the training machine.** All real training, tuning, learning-curve, and eval-suite runs execute on Colab GPU runtimes via the **Colab CLI** (`colab exec`), so `src/train/run.py` remains a plain script (no notebooks — Law 6 applies unchanged). Budget discipline: Pro is compute-unit metered (T4 ≈ ~1.8–2 CU/hr on a ~100 CU/month plan; A100 ≈ 15 CU/hr — reserve it for the final full-size runs only). Every run checkpoints and resumes; premium-GPU availability is not guaranteed, so nothing may *depend* on getting an L4/A100. Track CU spend in the Phase 9 cost accounting.
3. **Rented Linux GPU box — the measurement machine.** Phase 6's Triton dynamic-batching study, SLO validation, load tests, and GPU chaos tests run in short sessions on a cheap dedicated cloud GPU (T4/RTX-class, ~$0.2–0.5/hr). The docker-compose stack deploys there unchanged; benchmark scripts run; the box is torn down. Declared SLOs are defined against this machine class and stated as such in the README. A secondary "dev-mode" CPU SLO for the Mac may be recorded for regression-spotting but is never the headline number.

---

## 3. Repository Structure (create in Phase 0, exactly this)

```
traffic-vision-platform/
├── README.md                  # problem, honest scope, architecture, results
├── LICENSES.md                # per-dataset license terms + provenance
├── pyproject.toml             # python 3.12, ruff, mypy, pytest config
├── .github/workflows/         # ci.yml, train.yml, retrain-trigger.yml
├── configs/
│   ├── taxonomy.yaml          # canonical classes + per-dataset mappings
│   ├── datasets/              # per-dataset: mirror URL, sha256, version
│   ├── training/              # base.yaml, tuned.yaml (versioned artifacts)
│   ├── tuning.yaml            # search space, trial cap, GPU-hour cap
│   └── promotion.yaml         # the contract clauses + tolerances
├── src/
│   ├── data/
│   │   ├── download.py        # fetch + checksum-verify all datasets
│   │   ├── hygiene.py         # ignore-region masking, dedup, audits
│   │   ├── convert.py         # native formats → YOLO + canonical taxonomy
│   │   ├── split.py           # location/clip-level splitter + leakage guard
│   │   └── sample.py          # every-10th-frame sampler
│   ├── eval/
│   │   ├── harness.py         # slice eval, cross-dataset suites
│   │   ├── corruption.py      # blur/noise/compression/brightness curves
│   │   ├── bootstrap.py       # confidence intervals per slice
│   │   └── report.py          # versioned eval report generator
│   ├── train/
│   │   ├── run.py             # instrumented training entrypoint
│   │   ├── diagnose.py        # overfit/underfit auto-flags, learning curves
│   │   └── tune.py            # Ray Tune/Optuna/ASHA wrapper
│   ├── registry/
│   │   └── promote.py         # promotion contract enforcement
│   ├── streaming/
│   │   ├── producer.py        # video replay → Kafka (playlist-driven)
│   │   ├── consumer.py        # Kafka → Triton → results topic
│   │   └── shadow.py          # challenger consumer group + comparator
│   ├── flywheel/
│   │   ├── mine.py            # hard-example mining (4 signals)
│   │   ├── review.py          # pseudo-label review CLI (Tier 1 labeler)
│   │   └── ingest.py          # gated ingestion into new dataset version
│   └── monitoring/
│       ├── embeddings.py      # embedding extraction for drift
│       └── drift.py           # Evidently wrapper + alert rules
├── serving/
│   ├── triton/model_repository/
│   └── docker-compose.yml     # Tier 1 runtime (kafka, triton, prom, grafana)
├── infra/terraform/           # Tier 2
├── k8s/                       # Tier 2
├── tests/                     # mirrors src/; leakage tests are mandatory
└── docs/
    ├── runbook.md             # failure playbooks (Phase 8)
    ├── decisions/             # ADRs: one file per architectural decision
    └── experiments/           # experiment write-ups incl. flywheel study
```

Every architectural decision from this guide (JPEG-in-Kafka vs claim-check, shadow vs canary, frame sampling rate, tolerance policy) gets a one-page ADR in `docs/decisions/`. Interviews are won in that folder.

---

## 4. The Phases

Each phase lists: objective, the work, exit criteria (all must pass), and forbidden moves. Estimated calendar assumes part-time; the order is not negotiable even if the pace is.

---

### Phase 0 — Foundation and Data Provenance (Week 1)

**Objective.** A repo where every byte of data is pinned, verified, and legally documented.

**The work.**
1. Create the repo structure above. Configure ruff, mypy (strict), pytest, and a CI workflow that runs all three on every push. CI red = you fix it before anything else, from day one.
2. Verify availability of all four datasets **today**: pinned UA-DETRAC mirror, MIO-TCD, AAU RainSnow, VisDrone-DET. If MIO-TCD or RainSnow is unreachable, substitute BMD-45 now and write the ADR.
3. Write `download.py`: fetches each dataset, computes SHA256 per archive, refuses to proceed on mismatch. Record checksums in `configs/datasets/`.
4. Initialize DVC. Register raw datasets as version `raw-v1`.
5. Write `LICENSES.md`: license/terms for each dataset, provenance chain for the UA-DETRAC mirror, explicit statement that no images are redistributed.
6. Verify UA-DETRAC integrity against the paper: ~100 sequences, ~140k frames, 4 vehicle classes (car, bus, van, others), weather attributes present, ignore-region annotations present in the XML.

**Exit criteria.**
- [ ] `python -m src.data.download --all` completes with checksum verification on a clean machine.
- [ ] DVC `raw-v1` tagged; repo contains zero image files.
- [ ] CI green: ruff + mypy + a placeholder test suite.
- [ ] LICENSES.md complete; counts verified against the paper and recorded.

**Forbidden.** Training anything. Looking at model code. "Temporarily" committing sample images.

---

### Phase 1 — Annotation Hygiene, Taxonomy, and Splits (Weeks 1–2)

**Objective.** A clean, audited, leakage-proof dataset. This phase is where the project earns the word "production."

**The work.**
1. **Taxonomy first.** Define the canonical class set in `taxonomy.yaml`: `car`, `bus`, `van_truck` (map UA-DETRAC `van`+`others`→`van_truck` after auditing what `others` contains — audit before mapping, write the ADR). Add mapping tables for MIO-TCD, RainSnow, VisDrone native classes. Every conversion goes through this file; no hardcoded class IDs anywhere.
2. **Ignore-region handling (mandatory, non-negotiable).** UA-DETRAC annotates ignore regions containing vehicles too low-resolution to label. `hygiene.py` must: (a) black-fill ignore regions in every image copy used for training, and (b) export ignore-region polygons per sequence so the eval harness can exclude any detection falling inside them from false-positive counting. Unit-test both with hand-checked fixtures.
3. **Deduplication and audit.** IoU-filter duplicate boxes (known dataset defect). Produce an audit report per sequence: box counts before/after, class histogram, ignore-region area fraction. Commit the report.
4. **Frame sampling.** `sample.py` keeps every 10th frame (standard practice for this dataset; adjacent frames are near-duplicates). Sampled set becomes the working training pool (~8–14k images). Full-rate frames remain available for the streaming replay only.
5. **The split — the most important 100 lines in the repo.** `split.py` splits by camera location: from the 24 locations, assign roughly 16 → training pool, 8 → **production-holdout** (used only as the simulated live stream in Phases 6–7; their labels are sequestered as hidden ground truth). Within the 16 training locations, split train/val/test by *clip*, never by frame. Emit manifests (lists of clip IDs per split) as versioned artifacts.
6. **The leakage guard.** A test that fails if any clip ID appears in two splits, and a runtime assert in the training data loader that refuses any manifest containing eval or production-holdout clip IDs. This is Law 3 and Law 4 in executable form.
7. Convert everything to YOLO format via the taxonomy. Register as DVC `train-v1`, `eval-frozen-v1`, `prod-holdout-v1`.

**Exit criteria.**
- [ ] Hygiene unit tests pass, including ignore-region masking fixtures.
- [ ] Audit report committed; before/after numbers documented.
- [ ] Leakage tests pass; deliberately corrupting a manifest makes them fail (prove it, keep the proof as a test).
- [ ] Three DVC-tagged dataset versions exist; `eval-frozen-v1` and `prod-holdout-v1` are marked read-only in tooling.
- [ ] ADRs written: taxonomy mapping, sampling rate, split design.

**Forbidden.** Training. Frame-level splits "just to get a quick baseline." Touching production-holdout labels for anything except sequestered storage.

---

### Phase 2 — The Evaluation Harness (Weeks 2–3) — FROZEN BEFORE ANY TRAINING

**Objective.** The measuring instrument. Built, tested, and frozen before a single model exists, so no result can ever be curve-fit to it.

**The work.**
1. **Slice evaluation** (`harness.py`): given a model and an eval manifest, compute mAP50 and mAP50-95 per slice. Mandatory slices: per-class; weather (cloudy/night/sunny/rainy — the dataset provides labels); object scale (small/medium/large by box area); per-camera-location. Detections inside ignore regions are excluded from FP counts (Phase 1's polygons).
2. **Cross-dataset suites.** Frozen eval sets from MIO-TCD and AAU RainSnow (gated) and VisDrone (ungated stretch), all mapped through the taxonomy, each with its own manifest and version tag.
3. **Corruption benchmark** (`corruption.py`): apply parameterized blur, Gaussian noise, JPEG compression, brightness shift at 3–5 severities to the in-distribution test set; output degradation curves (mAP vs severity).
4. **Bootstrap confidence intervals** (`bootstrap.py`): resample images-with-replacement per slice (≥1000 resamples), emit 95% CIs for every slice metric. Small slices get wide CIs — that is the point; the promotion contract will use them.
5. **Report generator** (`report.py`): one command → a versioned eval report (JSON + human-readable) for any model checkpoint: all slices with CIs, cross-dataset numbers, corruption curves, in-distribution-vs-cross-dataset gap as a headline metric.
6. Test the harness itself: synthetic detections with known mAP verify the math; an ignore-region fixture verifies FP exclusion.

**Exit criteria.**
- [ ] `python -m src.eval.report --model X --suite all` produces a complete report on a stock COCO-pretrained YOLO (zero-shot numbers — your floor baseline, record it).
- [ ] Harness unit tests pass, including known-answer mAP tests.
- [ ] The harness code is tagged `eval-harness-v1`. From this commit on, changing eval logic requires a new version tag and re-running all historical reports. Treat it like changing a ruler mid-experiment — because it is.

**Forbidden.** Starting Phase 3 with any harness test red. "Small fixes" to the harness after seeing model results, without a version bump.

---

### Phase 3 — Instrumented Baseline Training (Weeks 3–4)

**Objective.** A reproducible, self-diagnosing training pipeline and an honest baseline.

**The work.**
1. `train/run.py`: single entrypoint, takes a config file, trains Ultralytics YOLO on `train-v1`, logs everything to MLflow — config, dataset version hash, git commit, seed, per-epoch train/val loss, per-class val AP, LR schedule. Early stopping on val mAP50-95; checkpoint the **best** epoch, never the last.
2. `train/diagnose.py`: post-run automatic diagnosis attached to the MLflow run — flags overfitting (val loss diverging from train loss beyond a threshold over a window) and underfitting (both losses plateaued high / val mAP below the zero-shot floor). A flagged run is marked `suspect` in MLflow.
3. **Learning-curve study (mandatory experiment #1).** Train on 25/50/75/100% of the training pool (same config, same seed policy). Plot val mAP vs data fraction. Write up in `docs/experiments/learning-curve.md`: is the model data-starved or capacity-limited? This answer drives every later data decision.
4. Baseline model: YOLO-small on the sampled training pool, default hyperparameters. Run the full Phase 2 report on it. This is `champion-v1` — register it in the MLflow registry and assign it the `@champion` alias (`set_registered_model_alias`). Serving code loads `models:/<name>@champion`, never a hardcoded version number.

**Exit criteria.**
- [ ] Re-running the baseline config reproduces val mAP within noise (document the tolerance).
- [ ] Diagnosis flags fire correctly on two deliberately broken runs (LR 100x too high; 3-epoch undertrain). Keep these as integration tests.
- [ ] Learning-curve write-up committed with the four data points and a stated conclusion.
- [ ] `champion-v1` in the registry with its full frozen-harness eval report attached as an artifact.

**Forbidden.** Cherry-picking the best of several unlogged runs as "the baseline." Quoting any number not produced by `eval-harness-v1`.

---

### Phase 4 — Hyperparameter Tuning as a Pipeline Stage (Weeks 4–5)

**Objective.** Systematic tuning under a hard budget, producing a versioned config — not a notebook full of vibes.

**The work.**
1. `configs/tuning.yaml`: search space limited to the high-leverage knobs — `lr0`, `lrf`, `weight_decay`, `warmup_epochs`, mosaic/mixup/HSV augmentation strengths, `imgsz`. Trial cap (15–25), ASHA early-stopping config, and a **hard GPU-hour budget** the wrapper enforces by refusing to launch past it.
2. `train/tune.py`: wraps Ultralytics `model.tune()` with `use_ray=True`, OptunaSearch on `metrics/mAP50-95(B)`, ASHA scheduler. Tune the **nano** model on the sampled pool with realistic epoch counts (heed the documented warning: short toy tuning runs transfer poorly). Every trial logs to MLflow.
3. Transfer study: take the best config, train the **small** model with it, full harness report. Compare against `champion-v1` — with CIs, per slice, not just the headline number.
4. If (and only if) it beats the baseline under the Phase 5 contract logic, the tuned config becomes `configs/training/tuned-v1.yaml` — the config all future retraining uses.

**Exit criteria.**
- [ ] Tuning completes within the GPU-hour budget; budget enforcement demonstrated (set budget to 0, watch it refuse).
- [ ] All trials visible in MLflow with configs and curves.
- [ ] `docs/experiments/tuning.md`: search space rationale, best config, nano→small transfer result, and an honest statement of how much tuning actually bought over defaults.

**Forbidden.** Widening the search space mid-run. Tuning on the full-rate frame set. Any manual "one more run with lr slightly lower" outside the logged pipeline.

---

### Phase 5 — The Promotion Contract (Week 5)

**Objective.** The mechanical guarantee that no model can replace a better one. Small code, highest concept density in the project.

**The work.**
1. `configs/promotion.yaml` encodes the clauses. A challenger is promoted only if **all** hold, evaluated by `eval-harness-v1` with bootstrap CIs:
   - **C1:** In-distribution mAP50-95 not significantly worse than champion (CI-overlap test), and point estimate ≥ champion − ε (ε from config, default 0).
   - **C2:** Gated cross-dataset aggregate (MIO-TCD + RainSnow, weighted by set size) not significantly worse; **and** no single gated suite drops more than 5 points (catastrophic-regression clause).
   - **C3:** No *large* slice (≥ a configured min box count) regresses with statistical significance. Small slices produce warnings in the report, never blocks.
   - **C4:** Corruption degradation curve not materially worse (area-under-curve comparison, tolerance in config).
   - **C5:** p95 latency and throughput within SLO on the actual serving stack (this clause activates in Phase 6; stubbed with a placeholder until then, clearly marked).
2. `registry/promote.py`: takes champion + challenger, runs the contract, emits a decision report (pass/fail per clause, with numbers), and reassigns (or refuses to reassign) the `@champion` alias to the challenger's version. Rollback is the same mechanism in reverse: reassign `@champion` to the previous version — one line, auditable, no redeploy of code. No human override flag exists in the code. If you want to override, you must change the config and commit it — that's the audit trail.
3. Wire into GitHub Actions: a `train.yml` workflow that trains a challenger from a config, evaluates, runs the contract, and promotes or rejects — end to end, no manual steps. **Hardware reality:** GitHub-hosted runners have no GPU, and this project's dev machine (Apple Silicon) has no CUDA. Register the Mac as a **self-hosted runner** for CPU jobs only (lint, tests, contract/gate logic, report generation); the training/eval step inside the workflow executes on a Colab GPU runtime via the **Colab CLI** (`colab exec` — verify the exact argument-passing syntax with `colab exec --help` before wiring the workflow; do not assume it), with checkpoint-and-resume so a session death never loses a run. Secure the runner: it must only run workflows from your own repo, never from forks. Write the ADR (including the MLflow reachability choice — tunneled local server vs free hosted).

**Exit criteria.**
- [ ] Integration tests: a deliberately degraded challenger (undertrained) is rejected with C1/C3 failures; a genuinely better one is promoted. Both kept as CI tests with small fixtures.
- [ ] A full decision report artifact is attached to every promotion attempt in MLflow.
- [ ] ADR: why CI-overlap tests instead of naive point thresholds; the small-slice warning policy.

**Forbidden.** Promoting anything by hand in the MLflow UI. Adding an override flag "for convenience."

---

### Phase 6 — Streaming Serving Path (Weeks 6–8)

**Objective.** Real-time inference under a declared SLO, with observability. The system becomes *operated*.

**The work.**
1. **Producer** (`streaming/producer.py`): replays production-holdout location videos as N simulated live camera streams into Kafka. Playlist-driven — a YAML playlist controls which sequences (and therefore which weather/lighting) stream when. This is your drift-injection mechanism; document it as such (Law 7). Frames are JPEG-compressed in-message; message size limits tuned and documented. Write the ADR now for the future claim-check (MinIO) migration and its trigger condition (e.g., stream count or broker throughput threshold).
2. **Triton deployment.** Export champion to ONNX, build the model repository, configure **dynamic batching**. Serve via `docker-compose.yml` (Tier 1 runtime: Kafka, Triton, Prometheus, Grafana). Per the hardware strategy: develop the compose stack on the Mac (Triton CPU-only if it runs; ONNX Runtime stub behind the same gRPC contract if it doesn't), then deploy the identical compose file to the rented GPU box for every measured run — batching study, SLO validation, load and chaos tests. TensorRT conversion happens on the GPU box (it is hardware-specific by design).
3. **Consumer** (`streaming/consumer.py`): Kafka → preprocess → Triton (gRPC) → postprocess (NMS, taxonomy labels) → results topic. Consumer lag, per-stage latency, and throughput exported to Prometheus.
4. **Declare the SLO** in the README before measuring, and pin it to the measurement machine: e.g., p95 end-to-end frame latency < 200 ms and p99 < 400 ms at 4 concurrent streams × 25 FPS **on the rented T4-class GPU box** (adjust the numbers to the hardware you actually rent, but declare *first*, measure second, and never quote Mac CPU numbers as the SLO).
5. **The batching study (mandatory experiment #2).** Sweep Triton `max_batch_size` and `max_queue_delay`; plot latency vs throughput vs GPU utilization. `docs/experiments/batching.md` with the curves and your chosen operating point, justified against the SLO.
6. **Backpressure policy.** Decide and implement what happens when inference falls behind (bounded consumer with frame-dropping policy vs unbounded lag), expose a dropped-frames metric, and write the ADR.
7. Grafana dashboards: SLO panel (latency percentiles vs target), consumer lag, GPU utilization, per-class detection-rate.
8. Activate promotion clause C5 with a scripted load test against the real stack.
9. **Tier 2 begins here:** stand up k3s locally (k3s is installed by its own installer — Terraform does not provision a local cluster). Terraform's honest role here is managing **in-cluster resources** via the `kubernetes`/`helm` providers (namespaces, deployments, Prometheus/Grafana charts) against the k3s kubeconfig; optionally add a second Terraform root that provisions a cloud GPU VM + k3s for the "real infra" variant. State this split in an ADR — claiming "Terraform provisions my laptop's cluster" is the kind of imprecision interviews catch. Compose remains as the dev-mode runtime.

**Exit criteria.**
- [ ] 4 streams × 25 FPS sustained for 30 minutes within the declared SLO; Grafana screenshot + metrics export committed as evidence.
- [ ] Batching study committed with curves and a justified operating point.
- [ ] Backpressure demonstrated: throttle Triton artificially, show the policy engaging and the dropped-frame metric counting, system recovering.
- [ ] Kill the consumer mid-stream; it resumes from committed offsets with no unprocessed-and-skipped frames (chaos test #1, scripted and repeatable).
- [ ] C5 wired into the contract with real measurements.
- [ ] Tier 2: the same exit tests pass on k3s, provisioned by `terraform apply`.

**Forbidden.** Raw uncompressed frames in Kafka. Declaring the SLO after seeing the numbers. A dashboard with no alert rules.

---

### Phase 7 — Drift Detection and Production Monitoring (Weeks 8–9)

**Objective.** The system notices its own degradation without ground-truth labels.

**The work.**
1. `monitoring/embeddings.py`: sample frames from the live stream (e.g., 1 per stream per N seconds), compute compact embeddings (the YOLO backbone's pooled features are sufficient — no new model), write to a feature store table (simple Postgres/parquet is fine — do not add a new heavy component).
2. `monitoring/drift.py`: Evidently compares rolling production windows against the training-distribution reference on those embedding features. Alert thresholds in config; alerts land in Prometheus Alertmanager.
3. Label-free proxy metrics alongside drift: per-class confidence distribution vs baseline, detection-count and class-mix distributions, and **temporal flicker rate** (objects appearing/disappearing between consecutive frames — an honest unsupervised quality signal for video).
4. The weekly truth sample: a small random draw of production frames scored against the sequestered ground truth (the one legitimate use of it, ADR-documented) → a true production-mAP time series on a dashboard panel.
5. **Drift scenario test (mandatory experiment #3):** playlist switch from sunny/cloudy locations to night/rainy ones. Show: embedding drift alert fires, proxy metrics move, truth-sample mAP dips — and record detection lag (time from playlist switch to alert). `docs/experiments/drift-scenario.md`.

**Exit criteria.**
- [ ] The scripted drift scenario reliably fires alerts; detection lag measured and documented.
- [ ] A no-drift control run (same-distribution playlist) does **not** fire alerts over a comparable window (false-positive check — as important as the hit).
- [ ] Production-mAP truth panel live on Grafana.

**Forbidden.** Tuning alert thresholds on the same scenario run you report (tune on one playlist, report on a fresh one). Claiming the drift is organic.

---

### Phase 8 — The Flywheel: Mining, Labeling, Gated Retraining, Shadow Rollout (Weeks 9–11)

**Objective.** Close the loop — and *measure* that closing it works better than the naive alternative.

**The work.**
1. **Mining** (`flywheel/mine.py`), four signals over production-holdout stream output: (a) low-confidence detections; (b) temporal flicker (tracker-detector disagreement); (c) frames inside drift-alert windows; (d) teacher disagreement — a larger YOLO (medium/large) trained on the same `train-v1` scores sampled frames; frames where teacher and champion disagree are mined. Write the honest ADR: the teacher shares training distribution and therefore blind spots; the location-holdout design partially mitigates this.
2. **Labeling** (`flywheel/review.py`, Tier 1): teacher pre-labels mined frames; a CLI shows pre-labels vs sequestered ground truth and "corrects" to ground truth, simulating a human labeler with a measurable budget (count of frames "labeled"). Because ground truth exists, you also get pseudo-label quality metrics for free (teacher precision/recall on mined frames) — report them. Tier 3 swaps the CLI for Label Studio with the same interface.
3. **Gated ingestion** (`flywheel/ingest.py`): label-format validation, class-balance sanity check, embedding-similarity dedup against the existing pool, and the leakage guard (nothing from frozen eval sets, ever). Output: DVC `train-v2` = train-v1 + mined-labeled frames + a replay sample of train-v1 (anti-forgetting mix, ratio in config).
4. **Triggered retraining:** drift alert (or manual dispatch) → GitHub Actions `retrain-trigger.yml` → train challenger on `train-v2` with `tuned-v1.yaml` → full harness report → promotion contract. (Tier 3: move the trigger to Argo Workflows; the workflow steps are identical.)
5. **Shadow rollout** (`streaming/shadow.py`): promoted challenger joins as a second Kafka consumer group over the same topics; a comparator job diffs champion-vs-challenger outputs (agreement rate, per-class deltas, latency) for a soak window; switchover is flipping the serving consumer group, rollback is flipping back. Both scripted.
6. **The flywheel study (mandatory experiment #4, the project's crown):** at an equal labeling budget (e.g., 300 frames), train (A) champion + mined frames vs (B) champion + random production frames. Full harness on both. Report whether active mining beat random sampling, per slice, with CIs. `docs/experiments/flywheel-value.md`. If mining does *not* win, report that too — a negative result honestly measured is worth more than a positive one asserted.
7. (Tier 3) KServe HTTP endpoint with request-based canary for single-image inference, plus the ADR explaining precisely why canary applies to the HTTP path and shadow to the streaming path.

**Exit criteria.**
- [ ] Full loop demonstrated end-to-end from a cold start: drift scenario → alert → mine → label (budgeted) → ingest (gates shown rejecting a poisoned fixture) → retrain → contract decision → shadow soak → switchover. Scripted so it can be re-run.
- [ ] Rollback demonstrated: force a bad challenger through shadow, show the comparator catching it and the switchover being refused/reverted.
- [ ] Flywheel study committed with the A/B numbers.
- [ ] Teacher pseudo-label quality metrics reported.

**Forbidden.** Using sequestered labels anywhere except the review-script correction step and the truth panel. Retraining without the anti-forgetting replay mix. Promoting via anything but `promote.py`.

---

### Phase 9 — Hardening and the Runbook (Weeks 11–12)

**Objective.** The difference between "it works" and "it's operated."

**The work.**
1. **Chaos suite**, scripted and repeatable: kill Triton mid-load (consumer behavior + recovery), kill a Kafka broker (single-node: document the SPOF honestly and what multi-broker would change), fill the disk MLflow writes to mid-training, feed the producer a corrupt video. Each test: expected behavior written *before* running, observed behavior after, gap analysis.
2. **`docs/runbook.md`** — written answers, with commands, to at minimum: drift alert fired but retraining failed the contract — now what? Consumer lag climbing and SLO breached — first three things to check? Shadow comparator shows challenger disagreeing on one class only — how to investigate? Checksum mismatch on dataset re-download — procedure? Promotion contract deadlocked (nothing passes for 3 cycles) — what's the escalation, and what tolerance change requires what evidence?
3. **Cost accounting:** total GPU-hours and £ spent, per phase, in the README. Production thinking treats cost as a metric.
4. **Final README:** problem, architecture diagram, honest-scope section (scripted drift, simulated labeler, single-node Kafka), headline results table — baseline vs tuned vs post-flywheel champion, in-distribution and cross-dataset, with CIs — links to the four experiment write-ups and every ADR.

**Exit criteria.**
- [ ] All chaos tests scripted, run, and written up with expected-vs-observed.
- [ ] Runbook reviewed by re-reading each scenario cold and executing the commands.
- [ ] README results table complete; a stranger can understand what was built, what was proven, and what was honestly out of scope in five minutes.

---

## 5. The Four Mandatory Experiments (recap)

You are not building a system that merely runs; you are building one that *proves things*. These four write-ups are the proof, and none are optional:

1. **Learning curves** (Phase 3) — data-starved or capacity-limited?
2. **Batching study** (Phase 6) — latency/throughput/GPU-utilization trade-off and the chosen operating point.
3. **Drift scenario** (Phase 7) — detection lag, plus the false-positive control.
4. **Flywheel value** (Phase 8) — active mining vs random sampling at equal budget, with CIs.

## 6. Definition of Done (the whole project)

The project is done when a single documented script can: start the stack, stream the production-holdout playlist, trigger the drift scenario, and carry a challenger through mine → label → ingest → retrain → contract → shadow → switchover — while Grafana shows the SLO held — and when every number in the README traces to a versioned eval-harness report, every decision to an ADR, and every claim survives the question "how do you know?"

Work the phases in order. Meet the exit criteria. Do not negotiate with the guide.
