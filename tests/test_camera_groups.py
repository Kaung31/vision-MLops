"""Camera-group completeness (ADR 0005): every UA-DETRAC train sequence appears exactly
once across the groups — asserted, never counted on screen. Three checks:
presence (all 60 accounted for), disjointness (no sequence in two groups), total == 60.
"""

import json
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]


def _members() -> list[str]:
    data: Any = yaml.safe_load((REPO / "configs/camera_groups.yaml").read_text())
    return [str(s) for g in data["groups"].values() for s in g["sequences"]]


def _expected_sequences() -> set[str]:
    audit: Any = json.loads((REPO / "docs/audits/annotation-audit.json").read_text())
    return {str(s["sequence"]) for s in audit["sequences"]}


def test_no_sequence_in_two_groups() -> None:
    members = _members()
    assert len(members) == len(set(members)), "a sequence is routed to more than one group"


def test_every_sequence_present_exactly_once() -> None:
    # set equality catches both missing and unexpected sequences
    assert set(_members()) == _expected_sequences()


def test_total_is_sixty() -> None:
    assert len(_members()) == 60
