"""run.py pure layer: results.csv parsing, dataset version pin, and the Law 4 training guard
(frozen-sentinel refusal + eval/prod-holdout sequence leakage refusal). Torch/mlflow-free.
"""

from pathlib import Path

import pytest

from src.train.run import (
    FrozenDataError,
    dataset_md5,
    guard_training_data,
    parse_results_csv,
)

# realistic ultralytics 8.x header: suffixed metric names, box/cls/dfl loss components
RESULTS_CSV = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),"
    "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0\n"
    "0,10.0,1.0,2.0,0.5,0.5,0.4,0.3,0.2,1.5,2.5,1.0,0.001\n"
    "1,20.0,0.8,1.5,0.4,0.6,0.5,0.4,0.3,1.2,2.0,0.8,0.001\n"
)


def test_parse_results_csv_sums_losses_and_extracts_map() -> None:
    c = parse_results_csv(RESULTS_CSV)
    assert c["train_loss"] == [3.5, 2.7]  # box+cls+dfl
    assert c["val_loss"] == [5.0, 4.0]
    assert c["val_map50_95"] == [0.2, 0.3]


def test_parse_results_csv_empty() -> None:
    assert parse_results_csv("") == {"train_loss": [], "val_loss": [], "val_map50_95": []}


def test_dataset_md5(tmp_path: Path) -> None:
    p = tmp_path / "train-v1.dvc"
    p.write_text("outs:\n- md5: abc123.dir\n  path: train-v1\n")
    assert dataset_md5(p) == "abc123.dir"


def _pool(tmp_path: Path, sequences: list[str]) -> Path:
    d = tmp_path / "pool"
    (d / "labels" / "train").mkdir(parents=True)
    (d / "labels" / "val").mkdir(parents=True)
    for seq in sequences:
        (d / "labels" / "train" / f"{seq}_img00001.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    return d


def test_guard_refuses_frozen_dir(tmp_path: Path) -> None:
    d = _pool(tmp_path, ["MVI_20011"])
    (d / ".readonly").write_text("frozen\n")
    with pytest.raises(FrozenDataError, match="frozen"):
        guard_training_data(d)


def test_guard_refuses_eval_sequence(tmp_path: Path) -> None:
    # MVI_40141 is in the real test split (cam_40141); a pool containing it must be refused
    d = _pool(tmp_path, ["MVI_20011", "MVI_40141"])
    with pytest.raises(FrozenDataError, match="MVI_40141"):
        guard_training_data(d)


def test_guard_refuses_prod_holdout_sequence(tmp_path: Path) -> None:
    import yaml

    splits = yaml.safe_load(Path("configs/splits.yaml").read_text())["assignment"]
    groups = yaml.safe_load(Path("configs/camera_groups.yaml").read_text())["groups"]
    holdout_seq = groups[splits["prod_holdout"][0]]["sequences"][0]
    with pytest.raises(FrozenDataError, match=holdout_seq):
        guard_training_data(_pool(tmp_path, [holdout_seq]))


def test_guard_passes_clean_train_pool(tmp_path: Path) -> None:
    guard_training_data(_pool(tmp_path, ["MVI_20011", "MVI_20052"]))  # real train-split seqs


def test_guard_on_the_real_train_v1_if_present() -> None:
    real = Path("data/processed/train-v1")
    if not (real / "labels").exists():
        pytest.skip("train-v1 not materialized on this machine")
    guard_training_data(real)  # the actual pool must always pass its own guard


def test_merge_overrides_tuned_plus_lr() -> None:
    from src.train.run import merge_overrides

    cfg = {"overrides": {"mosaic": 0.98, "mixup": 0.3}}
    # tuned knobs pass through; a broken-run --lr0 override wins last
    assert merge_overrides(cfg, {}) == {"mosaic": 0.98, "mixup": 0.3}
    assert merge_overrides(cfg, {"lr0": 1.0, "optimizer": "SGD"}) == {
        "mosaic": 0.98,
        "mixup": 0.3,
        "lr0": 1.0,
        "optimizer": "SGD",
    }
    # no overrides block -> empty (base config unaffected)
    assert merge_overrides({}, {}) == {}
