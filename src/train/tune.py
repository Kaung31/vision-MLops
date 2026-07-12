"""Hyperparameter tuning as a pipeline stage: Ultralytics model.tune() + Ray Tune + Optuna
search + ASHA early stopping (frozen stack §2), under a HARD GPU-hour budget enforced in code.

Budget enforcement (guide 4.1 — demonstrable, not aspirational):
  * a JSON ledger records cumulative GPU-seconds across sessions;
  * `check_budget` refuses to launch when the ledger shows the budget spent, and refuses any
    session whose worst-case projection (trials x epochs x measured-minutes-per-epoch) would
    overrun the remainder — set the budget to 0 in configs/tuning.yaml and it refuses (the
    exit-criteria demo);
  * after every session the actual wall time is appended to the ledger. The Colab transport
    adds a second, outer wall-clock kill via `colab run --timeout`.

The search space is fixed by configs/tuning.yaml — widening it mid-run is forbidden (guide);
the wrapper only reads the committed config, so a wider space is a visible config commit.
Verified API (ultralytics 8.4.92 source): model.tune(use_ray=True, search_alg="optuna",
iterations=N, grace_period=G, gpu_per_trial=1, **train_args) with ASHA built in; per-trial
metric "metrics/mAP50-95(B)". Every trial is logged to MLflow as a nested run afterward.

Run (Colab GPU, via the bootstrap): python -m src.train.tune --device 0
Mac: budget/refusal logic only — never run trials here (no CUDA; MPS smoke is run.py's job).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.train.run import DEFAULT_DATA, config_sha256, dataset_md5, git_commit, guard_training_data

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tuning.yaml"
TUNE_METRIC = "metrics/mAP50-95(B)"
# conservative worst-case minutes per nano epoch on T4 (full sampled pool, imgsz<=640);
# refined against the ledger's measured rate after the first session.
FALLBACK_MIN_PER_EPOCH = 1.5


class BudgetExhausted(RuntimeError):
    """The hard GPU-hour budget forbids launching this tuning session."""


# --- pure, CI-tested budget logic -------------------------------------------------------------


def read_ledger(path: Path) -> dict[str, Any]:
    if path.exists():
        return dict(json.loads(path.read_text()))
    return {"spent_gpu_seconds": 0.0, "sessions": []}


def spent_from_runs(runs: list[Any]) -> float:
    """Cumulative GPU-seconds from prior tune-session runs' logged metrics.

    MLflow (DagsHub) is the budget's source of truth: the local JSON ledger dies with each
    ephemeral Colab VM, but every session logs `gpu_seconds` to the tune experiment, so the
    spend survives and is enforceable from any machine.
    """
    return sum(float(r.data.metrics.get("gpu_seconds", 0.0)) for r in runs)


def append_session(path: Path, seconds: float, note: str) -> dict[str, Any]:
    led = read_ledger(path)
    led["spent_gpu_seconds"] = float(led["spent_gpu_seconds"]) + seconds
    led["sessions"].append({"seconds": seconds, "note": note, "at": time.time()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(led, indent=2) + "\n")
    return led


def check_budget(
    budget_hours: float,
    spent_seconds: float,
    trials: int,
    epochs: int,
    min_per_epoch: float = FALLBACK_MIN_PER_EPOCH,
) -> float:
    """Refuse (raise BudgetExhausted) unless the session fits the remaining budget.

    Returns the remaining budget in hours. Worst case assumes ASHA kills nothing.
    """
    remaining = budget_hours - spent_seconds / 3600.0
    if remaining <= 0:
        raise BudgetExhausted(
            f"budget {budget_hours:.1f}h already spent ({spent_seconds / 3600.0:.2f}h) — refusing"
        )
    projected = trials * epochs * min_per_epoch / 60.0
    if projected > remaining:
        raise BudgetExhausted(
            f"projected worst case {projected:.1f}h exceeds remaining {remaining:.2f}h — "
            f"refusing (lower trials/epochs or raise the budget in configs/tuning.yaml)"
        )
    return remaining


def build_space(spec: dict[str, Any]) -> dict[str, Any]:
    """configs/tuning.yaml space spec -> ray.tune sample objects (import deferred to call)."""
    from ray import tune  # lazy: ray only exists on the tuning machine

    out: dict[str, Any] = {}
    for name, s in spec.items():
        if s["dist"] == "loguniform":
            out[name] = tune.loguniform(float(s["low"]), float(s["high"]))
        elif s["dist"] == "uniform":
            out[name] = tune.uniform(float(s["low"]), float(s["high"]))
        elif s["dist"] == "choice":
            out[name] = tune.choice(list(s["values"]))
        else:
            raise ValueError(f"unknown dist {s['dist']!r} for {name}")
    return out


# --- the session -------------------------------------------------------------------------------


def tune_session(args: argparse.Namespace) -> dict[str, Any]:
    import mlflow
    from ultralytics import YOLO

    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(args.data)
    guard_training_data(data_dir)  # Law 4, same guard as training

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    exp = mlflow.set_experiment(args.experiment)

    # budget source of truth = MLflow (survives ephemeral VMs); local ledger is a human record
    prior = mlflow.MlflowClient().search_runs([exp.experiment_id], max_results=500)
    spent = spent_from_runs(list(prior))
    ledger_path = REPO_ROOT / cfg["budget"]["ledger"]
    remaining = check_budget(
        float(cfg["budget"]["gpu_hours"]), spent, int(cfg["trials"]), int(cfg["epochs"])
    )
    print(f"budget ok: {remaining:.2f} GPU-hours remaining — launching {cfg['trials']} trials")

    resolved = yaml.safe_load((data_dir / "data.yaml").read_text())
    resolved["path"] = str(data_dir.resolve())
    resolved_yaml = REPO_ROOT / "runs" / "tune-data.yaml"
    resolved_yaml.parent.mkdir(parents=True, exist_ok=True)
    resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False))

    start = time.time()
    model = YOLO(str(cfg["model"]))
    results = model.tune(
        use_ray=True,
        space=build_space(cfg["space"]),
        iterations=int(cfg["trials"]),
        grace_period=int(cfg["grace_period"]),
        gpu_per_trial=int(cfg["gpu_per_trial"]),
        search_alg="optuna",
        data=str(resolved_yaml),
        epochs=int(cfg["epochs"]),
        seed=int(cfg["seed"]),
        device=args.device,
    )
    elapsed = time.time() - start
    append_session(ledger_path, elapsed, f"{cfg['trials']} trials x <= {cfg['epochs']} epochs")

    with mlflow.start_run(run_name=args.run_name) as parent:
        mlflow.log_params(
            {
                "tuning_config_sha256": config_sha256(Path(args.config)),
                "dataset_md5": dataset_md5(data_dir.parent / f"{data_dir.name}.dvc"),
                "git_commit": git_commit(),
                "trials": cfg["trials"],
                "epochs_per_trial": cfg["epochs"],
            }
        )
        mlflow.log_metric("gpu_seconds", elapsed)  # the budget's durable source of truth
        best_cfg: dict[str, Any] = {}
        best_metric = float("-inf")
        for i, res in enumerate(results):
            metric = float((res.metrics or {}).get(TUNE_METRIC, float("nan")))
            with mlflow.start_run(run_name=f"trial-{i:02d}", nested=True):
                mlflow.log_params({k: v for k, v in res.config.items() if k in cfg["space"]})
                if metric == metric:  # not NaN
                    mlflow.log_metric("map50_95", metric)
            if metric == metric and metric > best_metric:
                best_metric = metric
                best_cfg = {k: v for k, v in res.config.items() if k in cfg["space"]}

        out = REPO_ROOT / "runs" / "tuned-v1-candidate.yaml"
        out.write_text(
            "# Best tuning-trial config (candidate — becomes configs/training/tuned-v1.yaml\n"
            "# ONLY if the small-model transfer run beats champion-v1 under contract logic).\n"
            + yaml.safe_dump({"best_val_map50_95_nano": best_metric, **best_cfg}, sort_keys=False)
        )
        mlflow.log_artifact(str(out))
        print(f"best nano trial {TUNE_METRIC}={best_metric:.4f}; candidate written to {out}")
        return {"run_id": parent.info.run_id, "best_metric": best_metric, "best_config": best_cfg}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--device", default="0")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment", default="traffic-vision-tune")
    p.add_argument("--run-name", default="tune-session")
    args = p.parse_args(argv)
    tune_session(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
