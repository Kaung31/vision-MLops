# ADR 0009 — Colab CLI training invocation + DagsHub hosting

- **Status:** Accepted
- **Date:** 2026-07-12
- **Phase:** 3 (Instrumented Baseline Training)

## Context

Real training runs on Colab Pro (hardware strategy §2); the Mac only smoke-tests on MPS. The
guide forbids assuming the Colab CLI syntax — it had to be verified from `--help` and a live
dry-run before any workflow depends on it. Training data and MLflow tracking must be reachable
from an ephemeral Colab VM, which the Mac's local filesystem is not.

## Decisions

### 1. Hosting: DagsHub (free tier) for both the DVC remote and MLflow

One account provides both endpoints, reachable from Colab and the Mac:

- **DVC remote:** `https://dagshub.com/Kaung31/vision-MLops.dvc` — the **`.dvc` suffix is
  load-bearing.** Without it every transfer fails instantly (uploads hit the repo web path,
  not the storage API); this burned a full failed push of 10,274 files before diagnosis.
  Credentials live in gitignored `.dvc/config.local` (auth basic, user + token).
  `train-v1` is pushed (10,274 files, `-j 16` clean).
- **MLflow tracking:** `https://dagshub.com/Kaung31/vision-MLops.mlflow`, HTTP basic auth via
  `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` env vars. The **model-registry alias
  API is verified supported** (probe: create model → `set_registered_model_alias` → delete),
  so `@champion`/`@challenger` aliases (CLAUDE.md requirement) live on DagsHub too — no split
  registry.
- Not a stack change (Law 1): DagsHub only *hosts* the frozen tools (DVC, MLflow).
- MLflow 3.x note: the `./mlruns` file store is retired (raises on use); local runs default to
  `sqlite:///mlflow.db` (gitignored).

### 2. Colab CLI: `colab run` with a self-contained bootstrap

Verified behaviour (from `-h` and a live T4 dry-run, google-colab-cli, 2026-07-12):

- `colab run [--gpu T4] [--timeout SECS] SCRIPT [ARGS…]` provisions a **fresh ephemeral VM**,
  uploads **exactly one local .py file**, forwards the remaining args as `sys.argv[1:]`, and
  tears the VM down on exit. `--timeout` defaults to **30 s** — every training call must set
  it explicitly (hours). `colab exec` has the same 30 s default and needs a pre-existing
  session; `run` is the training vehicle.
- OAuth quirk: Google grants scopes non-deterministically ≠ requested set; without
  `OAUTHLIB_RELAX_TOKEN_SCOPE=1` exported, oauthlib hard-fails and credentials are never
  cached (auth loops forever). With it, auth succeeds and is saved.
- Output relay quirk: the script runs under a Jupyter kernel — only Python-level
  `sys.stdout` is forwarded to the local terminal. **Raw-fd subprocess output (git, pip, dvc,
  Ultralytics epochs) is invisible** unless piped and re-printed by the bootstrap, which
  `scripts/colab_train.py` therefore does for every child process.

The pinned invocation (learning-curve / baseline runs substitute fraction & run-name):

```bash
export OAUTHLIB_RELAX_TOKEN_SCOPE=1
colab run --gpu T4 --timeout 36000 "$PWD/scripts/colab_train.py" \
    --token <DAGSHUB_TOKEN> \
    --config configs/training/base.yaml --device 0 --fraction 1.0 --run-name baseline
```

`scripts/colab_train.py` (stdlib-only, no repo imports) then: shallow-clones `main` from
GitHub → pip-installs ultralytics/mlflow/dvc (torch stays as Colab ships it, CUDA-matched) →
`dvc pull`s the pinned train-v1 from DagsHub → execs `python -m src.train.run` with tracking
env set. It consumes its own flags and forwards everything else verbatim (`parse_known_args`,
no `--` separator — the CLI's parser eats it). Secrets are masked in echoed commands. Because
results (params, metrics, best.pt, diagnosis) land in DagsHub MLflow, nothing depends on the
VM surviving.

**Dry-run evidence:** run `colab-dryrun` (2 epochs, fraction 0.05, T4) FINISHED on DagsHub
with full Law 6 provenance (git commit, dataset md5, config sha256, seed), curves, and
artifacts (`weights/best.pt`, `results.csv`, `diagnosis.json`). End-to-end wall time incl.
provisioning + deps + data ≈ 5–10 min; `dvc pull` of train-v1 is acceptable at this scale.

### 3. Token handling

The DagsHub token is passed to the bootstrap as an argument (`colab run` forwards no env) and
into the VM's env for MLflow; it must be a **scoped, revocable DagsHub token**, never the
account password. Any token that leaks into terminal output or logs is rotated at
DagsHub → Settings → Tokens (this happened once during bring-up — the first bootstrap echoed
the dvc password line — hence the masking requirement above).

## Alternatives rejected

- **`colab exec` for training:** needs a managed long-lived session and separate
  upload/teardown choreography; 30 s default timeout; `run`'s fresh-VM-per-run matches the
  reproducibility posture (Law 6) better.
- **Google Drive for data:** unversioned, quota-bound, and already burned us in Phase 0
  (VisDrone GDrive blocks automated fetch).
- **Pushing raw UA-DETRAC (9 GB) to DagsHub:** training needs only processed `train-v1`
  (~127 MB); the raw archive stays Mac-local under DVC.
