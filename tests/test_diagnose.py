"""Training diagnosis flags: healthy run is clean; diverged/overfit/underfit each fire on the
curve shape (or harness-vs-floor mAP) that defines them. The two broken-run *integration* tests
(real LR-100x and 3-epoch runs) live with run.py; these pin the pure logic."""

import numpy as np

from src.train.diagnose import diagnose


def _falling(n: int, start: float = 5.0, end: float = 1.0) -> list[float]:
    return list(np.linspace(start, end, n))


def test_healthy_run_is_not_suspect() -> None:
    train = _falling(30, 5.0, 0.8)
    val = _falling(30, 5.0, 1.2)  # falls and stays down
    vmap = _falling(30, 0.1, 0.55)  # improving val mAP
    d = diagnose(train, val, vmap, best_val_map=0.52)
    assert not d.suspect
    assert not (d.diverged or d.overfit or d.underfit)
    assert d.best_epoch == 29  # last epoch has the highest val mAP


def test_diverged_on_non_finite() -> None:
    d = diagnose([5.0, float("nan"), 3.0], [5.0, 4.0, 3.0], best_val_map=0.5)
    assert d.diverged and d.suspect
    assert any("non-finite" in r for r in d.reasons)


def test_diverged_when_loss_blows_up() -> None:
    # LR 100x too high: train loss increases overall
    d = diagnose([2.0, 8.0, 25.0], [2.0, 9.0, 30.0], best_val_map=0.5)
    assert d.diverged and d.suspect
    assert not d.overfit  # divergence takes precedence over the overfit/underfit checks


def test_overfit_on_val_rebound() -> None:
    train = _falling(20, 5.0, 0.5)  # keeps improving
    val = _falling(10, 5.0, 1.0) + _falling(10, 1.0, 2.5)  # min then rebound
    d = diagnose(train, val, best_val_map=0.5)
    assert d.overfit and d.suspect and not d.diverged
    assert any("rebounded" in r for r in d.reasons)


def test_underfit_below_floor() -> None:
    # 3-epoch undertrain: learned a little but harness mAP is below the zero-shot floor
    d = diagnose(
        [5.0, 4.0, 3.2], [5.0, 4.2, 3.6], val_map_curve=[0.05, 0.1, 0.15], best_val_map=0.15
    )
    assert d.underfit and d.suspect and not d.diverged
    assert any("below floor" in r for r in d.reasons)


def test_underfit_when_barely_learned() -> None:
    d = diagnose([5.0, 4.98, 4.97], [5.0, 4.99, 4.98], best_val_map=0.42)  # above floor but flat
    assert d.underfit and any("dropped only" in r for r in d.reasons)


def test_floor_from_config_override() -> None:
    # a harness mAP of 0.30 passes a low floor but fails a high one
    assert not diagnose(
        _falling(5), _falling(5), best_val_map=0.30, cfg={"floor_map50_95": 0.2}
    ).underfit
    assert diagnose(
        _falling(5), _falling(5), best_val_map=0.30, cfg={"floor_map50_95": 0.4}
    ).underfit
