"""Tuning budget enforcement (pure, CI): the ledger accounting and the two refusal paths —
budget spent and projection overrun. Guide 4 exit criterion: budget 0 => refuses to launch.
"""

from pathlib import Path

import pytest

from src.train.tune import BudgetExhausted, append_session, check_budget, read_ledger


def test_zero_budget_refuses() -> None:
    with pytest.raises(BudgetExhausted, match="refusing"):
        check_budget(budget_hours=0.0, spent_seconds=0.0, trials=1, epochs=1)


def test_spent_budget_refuses() -> None:
    with pytest.raises(BudgetExhausted, match="already spent"):
        check_budget(budget_hours=8.0, spent_seconds=8.5 * 3600, trials=1, epochs=1)


def test_projection_overrun_refuses() -> None:
    # 20 trials x 25 epochs x 1.5 min = 12.5h worst case > 1h remaining
    with pytest.raises(BudgetExhausted, match="projected worst case"):
        check_budget(budget_hours=1.0, spent_seconds=0.0, trials=20, epochs=25)


def test_fitting_session_allowed_and_returns_remaining() -> None:
    remaining = check_budget(budget_hours=8.0, spent_seconds=2 * 3600, trials=10, epochs=10)
    assert remaining == pytest.approx(6.0)


def test_ledger_roundtrip_accumulates(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    assert read_ledger(ledger)["spent_gpu_seconds"] == 0.0
    append_session(ledger, 3600.0, "s1")
    led = append_session(ledger, 1800.0, "s2")
    assert led["spent_gpu_seconds"] == pytest.approx(5400.0)
    assert [s["note"] for s in led["sessions"]] == ["s1", "s2"]
    # and the next check sees the spend
    remaining = check_budget(8.0, led["spent_gpu_seconds"], trials=2, epochs=10)
    assert remaining == pytest.approx(6.5)


def test_spent_from_runs_sums_metric_across_sessions() -> None:
    from types import SimpleNamespace

    from src.train.tune import spent_from_runs

    def run(**metrics: float) -> SimpleNamespace:
        return SimpleNamespace(data=SimpleNamespace(metrics=metrics))

    # two prior sessions + a trial run without the metric (nested runs don't carry it)
    runs = [run(gpu_seconds=3600.0), run(gpu_seconds=1800.0), run(map50_95=0.5)]
    assert spent_from_runs(runs) == pytest.approx(5400.0)
    assert spent_from_runs([]) == 0.0


def test_build_space_rejects_unknown_dist() -> None:
    pytest.importorskip("ray")
    from src.train.tune import build_space

    with pytest.raises(ValueError, match="unknown dist"):
        build_space({"lr0": {"dist": "normal", "low": 0, "high": 1}})
