"""Location/clip splitter + leakage guard — Laws 3 & 4 in executable form.

The 27 hand-audited camera groups (ADR 0005) are split into train / val / test /
prod_holdout at the GROUP level: a whole camera (all its clips) goes to exactly one split,
so no camera straddles train and eval. Assignment is deterministic (seeded) and
weather-constrained (test and prod_holdout each get >=1 night and >=1 rain group), because
rain is scarce (4 groups) and the harness weather slices + Phase 7 drift playlist would
otherwise evaluate on nothing.

Two guards make the Laws executable:
  * assert_no_sequence_leakage — no clip appears in two splits (Law 3);
  * assert_trainable — the training loader refuses any clip not in the train split, i.e.
    any eval/holdout clip (Law 4). A file-path guard, not memory.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_GROUPS = REPO_ROOT / "configs" / "camera_groups.yaml"
SPLIT_CONFIG = REPO_ROOT / "configs" / "split.yaml"
SPLITS_OUT = REPO_ROOT / "configs" / "splits.yaml"

SPLITS = ("train", "val", "test", "prod_holdout")

Groups = dict[str, dict[str, list[str]]]  # name -> {"sequences": [...], "weathers": [...]}
Assignment = dict[str, list[str]]  # split -> [group names]
Manifests = dict[str, list[str]]  # split -> [sequence ids]


class SplitError(RuntimeError):
    """The requested split is infeasible or internally inconsistent."""


class LeakageError(RuntimeError):
    """A clip or group leaks across splits (Law 3), or into training (Law 4)."""


def load_groups(path: Path = CAMERA_GROUPS) -> Groups:
    data: Any = yaml.safe_load(path.read_text())
    return {
        name: {"sequences": list(v["sequences"]), "weathers": list(v["weathers"])}
        for name, v in data["groups"].items()
    }


def load_config(path: Path = SPLIT_CONFIG) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text()))


def assign_groups(groups: Groups, config: dict[str, Any]) -> Assignment:
    """Deterministically assign every group to one split targeting SEQUENCE proportions.

    Weather-constraint groups are reserved first (the smallest qualifying ones, so big
    cameras stay free to balance); remaining groups are placed largest-first into whichever
    split is furthest below its sequence quota (LPT). Whole groups only — no camera splits.
    """
    rng = random.Random(config["seed"])
    names = sorted(groups)
    sizes = {g: len(groups[g]["sequences"]) for g in names}
    total = sum(sizes.values())
    props: dict[str, float] = config["proportions"]
    target = {s: round(props[s] * total) for s in ("prod_holdout", "val", "test")}
    target["train"] = total - sum(target.values())
    if target["train"] < 0:
        raise SplitError(f"proportions exceed 1.0 (train sequence target = {target['train']})")

    min_cov: dict[str, dict[str, int]] = config["constraints"]["min_weather_groups"]
    assignment: Assignment = {s: [] for s in SPLITS}
    cur: dict[str, int] = dict.fromkeys(SPLITS, 0)
    reserved: set[str] = set()

    def by_size(pool: list[str], largest_first: bool) -> list[str]:
        shuffled = list(pool)
        rng.shuffle(shuffled)  # seed breaks size ties deterministically
        return sorted(shuffled, key=lambda g: sizes[g], reverse=largest_first)

    def place(split: str, group: str) -> None:
        assignment[split].append(group)
        cur[split] += sizes[group]
        reserved.add(group)

    # 1) weather coverage: reserve the SMALLEST qualifying groups (keep big cameras free)
    for split in sorted(min_cov):
        for weather, need in sorted(min_cov[split].items()):
            pool = by_size(
                [g for g in names if g not in reserved and weather in groups[g]["weathers"]],
                largest_first=False,
            )
            if len(pool) < need:
                raise SplitError(
                    f"cannot give {split} {need} '{weather}' group(s): {len(pool)} free"
                )
            for g in pool[:need]:
                place(split, g)

    # 2) LPT: largest groups first, each to the split furthest below its sequence target
    for g in by_size([n for n in names if n not in reserved], largest_first=True):
        split = max(SPLITS, key=lambda s: (target[s] - cur[s], -SPLITS.index(s)))
        place(split, g)

    return {s: sorted(assignment[s]) for s in SPLITS}


def validate_assignment(groups: Groups, assignment: Assignment, config: dict[str, Any]) -> None:
    """Fail loud on group-level leakage, missing/extra groups, or unmet weather coverage."""
    assigned = [g for gs in assignment.values() for g in gs]
    if len(assigned) != len(set(assigned)):
        raise LeakageError("a group is assigned to more than one split")
    if set(assigned) != set(groups):
        raise LeakageError("assignment does not cover the groups exactly once")
    for split, reqs in config["constraints"]["min_weather_groups"].items():
        for weather, need in reqs.items():
            got = sum(1 for g in assignment[split] if weather in groups[g]["weathers"])
            if got < need:
                raise SplitError(f"{split} has {got} '{weather}' group(s), needs {need}")


def build_manifests(groups: Groups, assignment: Assignment) -> Manifests:
    return {s: sorted(seq for g in assignment[s] for seq in groups[g]["sequences"]) for s in SPLITS}


def assert_no_sequence_leakage(manifests: Manifests) -> None:
    """Law 3: no clip may appear in two splits."""
    seen: dict[str, str] = {}
    for split, seqs in manifests.items():
        for s in seqs:
            if s in seen:
                raise LeakageError(f"sequence {s} is in both '{seen[s]}' and '{split}'")
            seen[s] = split


def assert_trainable(training_sequences: Iterable[str], manifests: Manifests) -> None:
    """Law 4: the training loader must refuse any clip outside the train split.

    Call this in the training data loader with the sequences it is about to load. Any
    eval/holdout clip present raises — the guard is code, not memory.
    """
    train = set(manifests["train"])
    intruders = sorted(s for s in training_sequences if s not in train)
    if intruders:
        raise LeakageError(
            f"training manifest contains non-train (eval/holdout) clips: {intruders}"
        )


def build_split() -> dict[str, Any]:
    groups = load_groups()
    config = load_config()
    assignment = assign_groups(groups, config)
    validate_assignment(groups, assignment, config)
    manifests = build_manifests(groups, assignment)
    assert_no_sequence_leakage(manifests)
    return {
        "seed": config["seed"],
        "counts": {
            s: {"groups": len(assignment[s]), "sequences": len(manifests[s])} for s in SPLITS
        },
        "assignment": {s: assignment[s] for s in SPLITS},
        "manifests": {s: manifests[s] for s in SPLITS},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the versioned split manifest.")
    parser.add_argument("--out", type=Path, default=SPLITS_OUT)
    args = parser.parse_args(argv)

    split = build_split()
    header = (
        "# Versioned split manifest (ADR 0006). Generated by `python -m src.data.split`;\n"
        "# deterministic from configs/split.yaml (seed). Group-level, weather-constrained.\n"
        "# Do not hand-edit — change configs/split.yaml or configs/camera_groups.yaml and regen.\n"
    )
    args.out.write_text(header + yaml.safe_dump(split, sort_keys=False))
    for s in SPLITS:
        c = split["counts"][s]
        print(f"  {s:13} {c['groups']:>2} groups, {c['sequences']:>2} sequences")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
