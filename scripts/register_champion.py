"""Register a trained run's best.pt as a model version and point @champion at it (Phase 3.4).

One command, three effects, all on the DagsHub MLflow registry:
  1. downloads `weights/best.pt` from the named run (default: the latest run named
     `baseline`) to a local path, so the eval-harness report can be run against it;
  2. attaches the frozen-harness report (JSON+md, produced separately by src.eval.report)
     to that run as artifacts — the registry entry carries its own evidence;
  3. registers the run's best.pt as a version of `traffic-vision-detector` and assigns the
     `@champion` alias (model version aliases, NOT deprecated stages — CLAUDE.md).

Rollback is the same mechanism in reverse: reassign @champion to the previous version.
Two-step flow, because the report takes ~30 min of inference:

  # step 1 — fetch the checkpoint
  uv run --group train python scripts/register_champion.py --run-name baseline --fetch-only
  # step 2 — run the frozen report against it (Mac MPS, same rig as the zero-shot floor)
  uv run --group eval python -m src.eval.report --model <printed best.pt> --suite all \
      --device mps --out reports/champion-v1
  # step 3 — attach the report + register + alias
  uv run --group train python scripts/register_champion.py --run-name baseline \
      --report-dir reports/champion-v1

Creds: MLFLOW_TRACKING_URI/USERNAME/PASSWORD env vars (see ADR 0009); this script reads the
DagsHub token from .dvc/config.local automatically when the env vars are absent.
"""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "traffic-vision-detector"
DEFAULT_URI = "https://dagshub.com/Kaung31/vision-MLops.mlflow"
FETCH_DIR = REPO_ROOT / "runs" / "registry"


def dagshub_env() -> None:
    """Fill MLflow env vars from .dvc/config.local when not already set (never printed)."""
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD"):
        os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_URI)
        return
    cp = configparser.ConfigParser()
    cp.read(REPO_ROOT / ".dvc" / "config.local")
    # DVC section headers look like ['remote "origin"'] — quotes are part of the parsed name
    section = cp[next(s for s in cp.sections() if "origin" in s)]
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_URI)
    os.environ.setdefault("MLFLOW_TRACKING_USERNAME", section["user"])
    os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", section["password"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="baseline")
    p.add_argument("--alias", default="champion")
    p.add_argument("--fetch-only", action="store_true", help="download best.pt and stop")
    p.add_argument("--report-dir", type=Path, default=None, help="harness report to attach")
    args = p.parse_args(argv)

    dagshub_env()
    import mlflow  # after env is set

    c = mlflow.MlflowClient()
    runs = c.search_runs(
        ["0"],
        filter_string=f"attributes.run_name = '{args.run_name}'",
        order_by=["attributes.start_time DESC"],
    )
    if not runs:
        raise SystemExit(f"no run named {args.run_name!r} found")
    run = runs[0]
    if run.info.status != "FINISHED":
        raise SystemExit(f"run {args.run_name!r} is {run.info.status}, not FINISHED")
    if run.data.tags.get("suspect") == "true":
        raise SystemExit(f"run {args.run_name!r} is tagged suspect — refusing to register")

    FETCH_DIR.mkdir(parents=True, exist_ok=True)
    local = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path="weights/best.pt", dst_path=str(FETCH_DIR)
    )
    print(f"fetched: {local}")
    if args.fetch_only:
        return 0

    if args.report_dir is None:
        raise SystemExit("--report-dir is required to register (the entry carries evidence)")
    for f in ("report.json", "report.md"):
        path = args.report_dir / f
        if not path.exists():
            raise SystemExit(f"missing {path} — run src.eval.report first")
        c.log_artifact(run.info.run_id, str(path), artifact_path="harness-report")
    print(f"attached harness report from {args.report_dir}")

    mv = mlflow.register_model(f"runs:/{run.info.run_id}/weights/best.pt", MODEL_NAME)
    c.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)
    print(f"registered {MODEL_NAME} v{mv.version}  ->  @{args.alias}")
    print(f"serving loads: models:/{MODEL_NAME}@{args.alias}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
