"""Instrumented training entrypoint — one plain script for MPS smoke, Colab, anywhere (Law 6).

Every run is reproducible or it didn't happen: config file + its sha256, train-v1 .dvc md5,
git commit, and seed are all logged to MLflow before training starts. After training, the
per-epoch curves are logged, `diagnose()` runs, and a flagged run is tagged `suspect`.
Ultralytics checkpoints the BEST epoch (best.pt, by val fitness), never the last (guide 3.1);
its own MLflow callback is disabled so this script is the single explicit logger.

Law 4 guard: refuses to train on any directory carrying the frozen `.readonly` sentinel, and
refuses any dataset whose train/val sequences intersect the test or prod-holdout groups in
configs/splits.yaml.

Tracking server: --tracking-uri or MLFLOW_TRACKING_URI (DagsHub needs MLFLOW_TRACKING_USERNAME
+ MLFLOW_TRACKING_PASSWORD in the env — never hardcoded). Defaults to the local ./mlruns file
store so the MPS smoke needs no network.

Run (Mac smoke):   uv run --group eval --group train python -m src.train.run \
                     --config configs/training/base.yaml --device mps --epochs 2 --fraction 0.1
Run (real, Colab): same script, --device 0, full config, DagsHub tracking URI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from src.train.diagnose import diagnose

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "processed" / "train-v1"
SPLITS_YAML = REPO_ROOT / "configs" / "splits.yaml"
READONLY_SENTINEL = ".readonly"


class FrozenDataError(RuntimeError):
    """An attempt was made to train on a frozen eval/holdout dataset (Law 4)."""


# --- pure helpers (CI-tested, torch-free) -----------------------------------------------------


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_md5(dvc_pointer: Path) -> str:
    """The dataset version pin: outs[0].md5 from the DVC pointer file."""
    outs = yaml.safe_load(dvc_pointer.read_text())["outs"]
    return str(outs[0]["md5"])


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def guard_training_data(data_dir: Path, splits_yaml: Path = SPLITS_YAML) -> None:
    """Law 4 in executable form: no frozen dir, no eval/holdout sequences in the pool."""
    if (data_dir / READONLY_SENTINEL).exists():
        raise FrozenDataError(f"{data_dir} is a frozen eval set — training on it is forbidden")
    splits = yaml.safe_load(splits_yaml.read_text())["assignment"]
    groups = yaml.safe_load((REPO_ROOT / "configs" / "camera_groups.yaml").read_text())["groups"]
    forbidden = {
        seq
        for split in ("test", "prod_holdout")
        for grp in splits.get(split, [])
        for seq in groups[grp]["sequences"]
    }
    seen = {
        p.stem.rsplit("_img", 1)[0]
        for sub in ("train", "val")
        for p in (data_dir / "labels" / sub).glob("*.txt")
    }
    leaked = sorted(seen & forbidden)
    if leaked:
        raise FrozenDataError(f"training pool contains eval/prod-holdout sequences: {leaked}")


def parse_results_csv(text: str) -> dict[str, list[float]]:
    """Ultralytics results.csv -> per-epoch total train loss, total val loss, val mAP50-95.

    Column names are matched by prefix/substring (they carry suffixes like `(B)` and have
    varied across versions); loss = sum of the box/cls/dfl components.
    """
    rows = list(csv.DictReader(line for line in text.splitlines() if line.strip()))
    if not rows:
        return {"train_loss": [], "val_loss": [], "val_map50_95": []}
    keys = {k.strip(): k for k in rows[0]}
    train_keys = [v for k, v in keys.items() if k.startswith("train/") and k.endswith("_loss")]
    val_keys = [v for k, v in keys.items() if k.startswith("val/") and k.endswith("_loss")]
    map_key = next((v for k, v in keys.items() if "mAP50-95" in k), None)

    def total(row: dict[str, str], cols: list[str]) -> float:
        return sum(float(row[c]) for c in cols)

    return {
        "train_loss": [total(r, train_keys) for r in rows],
        "val_loss": [total(r, val_keys) for r in rows],
        "val_map50_95": [float(r[map_key]) for r in rows] if map_key else [],
    }


# --- the run ----------------------------------------------------------------------------------


def train(args: argparse.Namespace) -> dict[str, Any]:
    import mlflow  # lazy: train group only
    from ultralytics import YOLO  # type: ignore[attr-defined]  # lazy: eval group only
    from ultralytics import settings as ul_settings

    cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    data_dir = Path(args.data)
    guard_training_data(data_dir)
    # this script is the single explicit logger
    ul_settings.update({"mlflow": False})  # type: ignore[no-untyped-call]

    epochs = args.epochs or int(cfg["epochs"])
    fraction = args.fraction if args.fraction is not None else float(cfg.get("fraction", 1.0))
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    run_name = args.run_name or f"{Path(args.config).stem}-f{fraction:g}-s{seed}-{int(time.time())}"

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "config_file": str(args.config),
                "config_sha256": config_sha256(Path(args.config)),
                "dataset_version": cfg["data_version"],
                "dataset_md5": dataset_md5(data_dir.parent / f"{data_dir.name}.dvc"),
                "git_commit": git_commit(),
                "seed": seed,
                "model": cfg["model"],
                "imgsz": cfg["imgsz"],
                "epochs": epochs,
                "batch": cfg["batch"],
                "patience": cfg["patience"],
                "fraction": fraction,
                "device": args.device,
                **({"lr0": args.lr0} if args.lr0 is not None else {}),
            }
        )

        # Ultralytics resolves a relative `path:` against CWD, not the yaml's location — and
        # train-v1 is md5-pinned so we can't edit it in place. Write a run-scoped resolved
        # copy with an absolute path instead (portable to Colab mounts).
        resolved = yaml.safe_load((data_dir / "data.yaml").read_text())
        resolved["path"] = str(data_dir.resolve())
        resolved_yaml = REPO_ROOT / "runs" / f"{run_name}-data.yaml"
        resolved_yaml.parent.mkdir(parents=True, exist_ok=True)
        resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False))

        model = YOLO(cfg["model"])
        results = model.train(
            data=str(resolved_yaml),
            epochs=epochs,
            imgsz=int(cfg["imgsz"]),
            batch=int(cfg["batch"]),
            patience=int(cfg["patience"]),
            seed=seed,
            fraction=fraction,
            device=args.device,
            project=str(REPO_ROOT / "runs"),
            name=run_name,
            exist_ok=True,
            verbose=False,
            **({"lr0": args.lr0} if args.lr0 is not None else {}),
        )
        run_dir = Path(results.save_dir)

        curves = parse_results_csv((run_dir / "results.csv").read_text())
        for i in range(len(curves["train_loss"])):
            mlflow.log_metrics(
                {
                    "train_loss": curves["train_loss"][i],
                    "val_loss": curves["val_loss"][i],
                    **(
                        {"val_map50_95": curves["val_map50_95"][i]}
                        if curves["val_map50_95"]
                        else {}
                    ),
                },
                step=i,
            )

        best_map = max(curves["val_map50_95"]) if curves["val_map50_95"] else None
        diag = diagnose(
            curves["train_loss"],
            curves["val_loss"],
            curves["val_map50_95"] or None,
            best_val_map=best_map,
            cfg=cfg.get("diagnose"),
        )
        diag_path = run_dir / "diagnosis.json"
        diag_path.write_text(json.dumps(diag.as_dict(), indent=2) + "\n")
        mlflow.log_artifact(str(diag_path))
        mlflow.log_artifact(str(run_dir / "results.csv"))
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            mlflow.log_artifact(str(best_pt), artifact_path="weights")
        mlflow.set_tags(
            {
                "suspect": str(diag.suspect).lower(),
                "diagnosis": ";".join(diag.reasons) or "healthy",
            }
        )

        summary = {
            "run_id": run.info.run_id,
            "run_dir": str(run_dir),
            "best_pt": str(best_pt),
            "best_val_map50_95": best_map,
            "diagnosis": diag.as_dict(),
        }
        print(json.dumps(summary, indent=2))
        return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "training" / "base.yaml"))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--device", default=None, help="mps | 0 (cuda) | cpu | None=auto")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment", default="traffic-vision-train")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override config (smoke tests)")
    parser.add_argument("--fraction", type=float, default=None, help="override (learning curve)")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--lr0", type=float, default=None, help="override (broken-run tests)")
    args = parser.parse_args(argv)
    summary = train(args)
    return 1 if summary["diagnosis"]["suspect"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
