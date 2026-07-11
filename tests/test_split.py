"""Split + leakage guard (Laws 3 & 4). Group-level, deterministic, weather-constrained,
and the mandatory proof that corrupting a manifest makes the guards fail.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.data.split import (
    SPLITS,
    Groups,
    LeakageError,
    assert_no_sequence_leakage,
    assert_trainable,
    assign_groups,
    build_manifests,
    build_split,
    load_config,
    load_groups,
    validate_assignment,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def groups() -> Groups:
    return load_groups()


@pytest.fixture
def config() -> dict[str, Any]:
    return load_config()


def test_assignment_is_deterministic(groups: Groups, config: dict[str, Any]) -> None:
    assert assign_groups(groups, config) == assign_groups(groups, config)


def test_every_group_assigned_exactly_once(groups: Groups, config: dict[str, Any]) -> None:
    assigned = [g for gs in assign_groups(groups, config).values() for g in gs]
    assert sorted(assigned) == sorted(groups)  # presence + disjointness at group level


def test_manifests_cover_all_60_with_no_clip_leakage(
    groups: Groups, config: dict[str, Any]
) -> None:
    manifests = build_manifests(groups, assign_groups(groups, config))
    seqs = [s for ss in manifests.values() for s in ss]
    assert len(seqs) == 60
    assert len(set(seqs)) == 60
    assert_no_sequence_leakage(manifests)  # Law 3: must not raise on a clean split


def test_weather_coverage_enforced(groups: Groups, config: dict[str, Any]) -> None:
    assignment = assign_groups(groups, config)
    validate_assignment(groups, assignment, config)  # raises if a constraint is unmet
    for split in ("test", "prod_holdout"):
        weathers = {w for g in assignment[split] for w in groups[g]["weathers"]}
        assert "night" in weathers and "rainy" in weathers


def test_train_is_the_largest_split(groups: Groups, config: dict[str, Any]) -> None:
    manifests = build_manifests(groups, assign_groups(groups, config))
    assert len(manifests["train"]) == max(len(manifests[s]) for s in SPLITS)


def test_corrupting_a_manifest_trips_the_leakage_guard(
    groups: Groups, config: dict[str, Any]
) -> None:
    # Law 3 proof: putting a holdout clip into train makes assert_no_sequence_leakage fail.
    manifests = build_manifests(groups, assign_groups(groups, config))
    corrupted = {**manifests, "train": manifests["train"] + [manifests["prod_holdout"][0]]}
    with pytest.raises(LeakageError):
        assert_no_sequence_leakage(corrupted)


def test_training_loader_guard_refuses_eval_clips(groups: Groups, config: dict[str, Any]) -> None:
    # Law 4 proof: assert_trainable passes on the clean train set, raises on an eval clip.
    manifests = build_manifests(groups, assign_groups(groups, config))
    assert_trainable(manifests["train"], manifests)
    with pytest.raises(LeakageError):
        assert_trainable([*manifests["train"], manifests["test"][0]], manifests)


def test_committed_splits_yaml_is_not_stale() -> None:
    live = build_split()
    committed: Any = yaml.safe_load((REPO / "configs" / "splits.yaml").read_text())
    assert committed["assignment"] == live["assignment"]
    assert committed["manifests"] == live["manifests"]
