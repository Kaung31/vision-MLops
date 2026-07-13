"""Promotion contract: the two exit-criteria integration cases (a degraded challenger is
rejected with C1/C3 failures; a genuinely better one is promoted) plus per-clause coverage.
Pure logic on tiny synthetic report fixtures — no MLflow, no models.
"""

from typing import Any

import yaml

from src.registry.promote import clause_c2, clause_c3, clause_c4, run_contract, sig_worse

CFG: dict[str, Any] = yaml.safe_load(open("configs/promotion.yaml"))


def m(point: float, half: float = 0.02) -> dict[str, float]:
    return {"point": point, "ci_low": point - half, "ci_high": point + half}


def report(
    indist: float,
    gated: dict[str, tuple[float, int]],
    classes: dict[str, tuple[float, int]],
    weather: dict[str, tuple[float, int]] | None = None,
    corr: list[float] | None = None,
) -> dict[str, Any]:
    per_class = {cid: {**m(pt), "n_gt": n} for cid, (pt, n) in classes.items()}
    ua: dict[str, Any] = {
        "slices": {
            "overall": {
                "n_gt": sum(n for _, n in classes.values()),
                "metrics": {"map50_95": m(indist), "ap50_95_per_class": per_class},
            },
            "per_weather": {
                w: {"n_gt": n, "metrics": {"map50_95": m(pt)}}
                for w, (pt, n) in (weather or {}).items()
            },
            "per_camera": {},
            "per_scale": {},
        }
    }
    if corr is not None:
        ua["corruption"] = {
            "curves": {"blur": {str(i): {"map50_95": v} for i, v in enumerate(corr)}}
        }
    suites: dict[str, Any] = {"ua-detrac": ua}
    for s, (pt, n) in gated.items():
        suites[s] = {"slices": {"overall": {"n_gt": n, "metrics": {"map50_95": m(pt)}}}}
    return {"suites": suites}


CHAMPION = report(
    indist=0.60,
    gated={"mio-tcd": (0.22, 5000), "rainsnow": (0.10, 11000)},
    classes={"0": (0.68, 9000), "1": (0.64, 100), "2": (0.47, 1400)},
    weather={"night": (0.46, 1700), "sunny": (0.72, 1000)},
    corr=[0.60, 0.55, 0.45, 0.30],
)


def _failed(decision: Any) -> list[str]:
    return [c.name for c in decision.clauses if not c.passed]


def test_better_challenger_is_promoted() -> None:
    better = report(
        indist=0.63,
        gated={"mio-tcd": (0.22, 5000), "rainsnow": (0.12, 11000)},
        classes={"0": (0.69, 9000), "1": (0.66, 100), "2": (0.54, 1400)},
        weather={"night": (0.51, 1700), "sunny": (0.72, 1000)},  # no large slice regresses
        corr=[0.63, 0.57, 0.46, 0.31],
    )
    d = run_contract(CHAMPION, better, CFG)
    assert d.promoted, _failed(d)


def test_undertrained_challenger_rejected_on_c1_and_c3() -> None:
    # much worse everywhere: in-dist CI fully below champion, and the large car slice regresses
    weak = report(
        indist=0.30,
        gated={"mio-tcd": (0.10, 5000), "rainsnow": (0.05, 11000)},
        classes={"0": (0.35, 9000), "1": (0.20, 100), "2": (0.15, 1400)},
        weather={"night": (0.20, 1700), "sunny": (0.30, 1000)},
        corr=[0.30, 0.25, 0.18, 0.10],
    )
    d = run_contract(CHAMPION, weak, CFG)
    assert not d.promoted
    failed = _failed(d)
    assert "C1_in_distribution" in failed
    assert "C3_large_slice_regression" in failed


def test_sig_worse_is_ci_overlap_not_point() -> None:
    assert sig_worse(m(0.60), m(0.50))  # 0.52 < 0.60 -> whole CI below champion point
    assert not sig_worse(m(0.60), m(0.59))  # 0.61 overlaps -> not significant
    assert not sig_worse(m(0.60), m(0.62))  # better


def test_c2_catastrophic_single_suite_drop_blocks() -> None:
    # aggregate fine (rainsnow is the big set), but mio-tcd craters > 5 points
    chal = report(
        indist=0.61,
        gated={"mio-tcd": (0.05, 5000), "rainsnow": (0.11, 11000)},
        classes={"0": (0.69, 9000)},
    )
    res = clause_c2(CHAMPION, chal, CFG)
    assert not res.passed and res.numbers["worst_suite"] == "mio-tcd"


def test_c3_small_slice_warns_never_blocks() -> None:
    # only regression is bus (n=100 < 500 min) and sunny (n=1000, kept EQUAL) -> warn, pass
    chal = report(
        indist=0.61,
        gated={"mio-tcd": (0.23, 5000), "rainsnow": (0.11, 11000)},
        classes={"0": (0.69, 9000), "1": (0.40, 100), "2": (0.48, 1400)},
        weather={"night": (0.47, 1700), "sunny": (0.72, 1000)},
        corr=[0.61, 0.56, 0.46, 0.31],
    )
    d = run_contract(CHAMPION, chal, CFG)
    c3 = next(c for c in d.clauses if c.name == "C3_large_slice_regression")
    assert c3.passed
    assert any("class:1" in w for w in c3.numbers["warnings"])
    assert d.promoted  # a small-slice dip alone must never block promotion


def test_c3_large_slice_dip_within_ci_does_not_block() -> None:
    # the real tuned-v1 case: a large slice dips a little but its wide CI overlaps the champion
    # -> non-significant -> must NOT block (else every noisy run is rejected).
    champ = report(0.60, {"mio-tcd": (0.22, 5000), "rainsnow": (0.10, 11000)}, {"0": (0.68, 9000)})
    chal = report(0.61, {"mio-tcd": (0.23, 5000), "rainsnow": (0.11, 11000)}, {"0": (0.69, 9000)})
    # inject a large weather slice that dips 0.72->0.69 but with a WIDE ci (half 0.05)
    champ["suites"]["ua-detrac"]["slices"]["per_weather"] = {
        "sunny": {"n_gt": 1000, "metrics": {"map50_95": m(0.72, half=0.05)}}
    }
    chal["suites"]["ua-detrac"]["slices"]["per_weather"] = {
        "sunny": {"n_gt": 1000, "metrics": {"map50_95": m(0.69, half=0.05)}}
    }
    res = clause_c3(champ, chal, CFG)
    assert res.passed and not res.numbers["blocks"]  # 0.74 ci_high >= 0.72 champ point


def test_c4_corruption_regression_blocks() -> None:
    worse_corr = report(
        indist=0.61,
        gated={"mio-tcd": (0.23, 5000), "rainsnow": (0.11, 11000)},
        classes={"0": (0.69, 9000)},
        corr=[0.40, 0.30, 0.20, 0.10],  # AUC far below champion
    )
    assert not clause_c4(CHAMPION, worse_corr, CFG).passed


def test_c4_skipped_when_no_curves() -> None:
    no_corr = report(
        0.61, {"mio-tcd": (0.23, 5000), "rainsnow": (0.11, 11000)}, {"0": (0.69, 9000)}
    )
    assert clause_c4(CHAMPION, no_corr, CFG).passed  # cannot judge -> does not block
