"""Canonical taxonomy loader + native->canonical routing.

`configs/taxonomy.yaml` is the single source of truth for class IDs; nothing else in
the repo hardcodes a class ID. Every dataset conversion routes its native labels through
`route()`, which fails loud on any class not declared under map/ignore/exclude — so a
new or misspelled native class can never be silently dropped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"


class UnknownNativeClass(KeyError):
    """A native class is not routed (map/ignore/exclude) for its dataset."""


class TaxonomyError(ValueError):
    """taxonomy.yaml is internally inconsistent."""


def load(path: Path = TAXONOMY_PATH) -> dict[str, Any]:
    tax: Any = yaml.safe_load(path.read_text())
    validate(tax)
    return dict(tax)


def canonical_names(tax: dict[str, Any]) -> list[str]:
    return [str(c["name"]) for c in tax["canonical"]]


def route(tax: dict[str, Any], dataset: str, native: object) -> tuple[str, str | None]:
    """Route one native class label.

    Returns ('map', canonical_name), ('ignore', None), or ('exclude', None).
    Raises UnknownNativeClass if ``native`` is not declared for ``dataset``.
    """
    d = tax["datasets"][dataset]
    mapping = d.get("map") or {}
    if native in mapping:
        return ("map", str(mapping[native]))
    if native in (d.get("ignore") or []):
        return ("ignore", None)
    if native in (d.get("exclude") or []):
        return ("exclude", None)
    raise UnknownNativeClass(
        f"{dataset}: native class {native!r} is not routed in taxonomy.yaml "
        f"(add it to map, ignore, or exclude)"
    )


def validate(tax: dict[str, Any]) -> None:
    """Fail loud if taxonomy.yaml is inconsistent: duplicate canonical names, a map
    target that isn't canonical, or a native class routed by more than one verb.
    """
    names = [str(c["name"]) for c in tax["canonical"]]
    if len(names) != len(set(names)):
        raise TaxonomyError("duplicate canonical class names")
    canon = set(names)
    for ds, d in tax["datasets"].items():
        mapping = d.get("map") or {}
        ignore = list(d.get("ignore") or [])
        exclude = list(d.get("exclude") or [])
        bad_targets = set(mapping.values()) - canon
        if bad_targets:
            raise TaxonomyError(f"{ds}: map targets not in canonical set: {sorted(bad_targets)}")
        keys = list(mapping.keys()) + ignore + exclude
        dupes = sorted({k for k in keys if keys.count(k) > 1}, key=repr)
        if dupes:
            raise TaxonomyError(f"{ds}: native class(es) routed by multiple verbs: {dupes}")
