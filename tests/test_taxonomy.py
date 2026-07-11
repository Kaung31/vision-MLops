"""Taxonomy routing: completeness (nothing silently dropped) + fail-loud on unknown."""

from typing import Any

import pytest

from src.data.taxonomy import (
    TaxonomyError,
    UnknownNativeClass,
    canonical_names,
    load,
    route,
    validate,
)


def test_loads_and_canonical_order() -> None:
    tax = load()
    assert canonical_names(tax) == ["car", "bus", "van_truck"]


def test_known_routes() -> None:
    tax = load()
    # UA-DETRAC: the audited others->van_truck merge
    assert route(tax, "ua-detrac", "others") == ("map", "van_truck")
    assert route(tax, "ua-detrac", "car") == ("map", "car")
    # MIO-TCD: motorized_vehicle is a don't-care, not an exclude
    assert route(tax, "mio-tcd", "motorized_vehicle") == ("ignore", None)
    assert route(tax, "mio-tcd", "articulated_truck") == ("map", "van_truck")
    assert route(tax, "mio-tcd", "pedestrian") == ("exclude", None)
    # RainSnow: COCO ids (truck -> van_truck; no van in COCO)
    assert route(tax, "aau-rainsnow", 8) == ("map", "van_truck")
    assert route(tax, "aau-rainsnow", 1) == ("exclude", None)
    # VisDrone: ids 0 (ignored regions) and 11 (others) are don't-care per its protocol
    assert route(tax, "visdrone", 4) == ("map", "car")
    assert route(tax, "visdrone", 0) == ("ignore", None)
    assert route(tax, "visdrone", 11) == ("ignore", None)


def test_fail_loud_on_unknown_native_class() -> None:
    tax = load()
    with pytest.raises(UnknownNativeClass):
        route(tax, "ua-detrac", "spaceship")
    with pytest.raises(UnknownNativeClass):
        route(tax, "visdrone", 99)


def test_shipped_taxonomy_is_consistent() -> None:
    validate(load())  # must not raise on the committed file


def test_validate_rejects_double_routed_class() -> None:
    broken: dict[str, Any] = {
        "canonical": [{"id": 0, "name": "car"}],
        "datasets": {"x": {"map": {"a": "car"}, "ignore": ["a"], "exclude": []}},
    }
    with pytest.raises(TaxonomyError):
        validate(broken)


def test_validate_rejects_noncanonical_map_target() -> None:
    broken: dict[str, Any] = {
        "canonical": [{"id": 0, "name": "car"}],
        "datasets": {"x": {"map": {"a": "lorry"}, "ignore": [], "exclude": []}},
    }
    with pytest.raises(TaxonomyError):
        validate(broken)
