"""The promotion contract — the mechanical guarantee that no model replaces a better one.

`run_contract` is pure: two frozen-harness report dicts + the config -> a Decision with a
pass/fail (and numbers) for each of five clauses. It consumes reports, not models, so the
contract has zero GPU/inference dependency and is fully unit-tested with tiny fixtures.

The CLI resolves the current @champion / @challenger model versions, loads their harness
reports, runs the contract, writes a decision report (JSON + human) and logs it to MLflow, and
— iff every clause passes — reassigns the @champion alias to the challenger's version. Rollback
is the same call in reverse. There is NO override flag: overriding a decision means editing
configs/promotion.yaml and committing it (guide Phase 5; ADR 0010).

The statistical core is a CI-OVERLAP test, not a point threshold: a challenger is "significantly
worse" on a metric only when its whole 95% CI sits below the champion's (ci_high < ci_low). A
raw point-threshold would fire on noise, especially on thin slices — the exact failure the
bootstrap CIs exist to prevent.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "promotion.yaml"
Metric = dict[str, float]  # {"point", "ci_low", "ci_high", ("n_gt")}


@dataclass
class ClauseResult:
    name: str
    passed: bool
    detail: str
    numbers: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    promoted: bool
    clauses: list[ClauseResult]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- report accessors -------------------------------------------------------------------------


def _overall(report: dict[str, Any], suite: str) -> dict[str, Any]:
    return cast("dict[str, Any]", report["suites"][suite]["slices"]["overall"])


def _map(report: dict[str, Any], suite: str) -> Metric:
    return cast(Metric, _overall(report, suite)["metrics"]["map50_95"])


def sig_worse(champ: Metric, chal: Metric) -> bool:
    """Challenger significantly worse: its entire 95% CI sits below the champion's point.

    Uses champion's POINT (not ci_low) as the bar so a champion with its own wide CI can still
    be defended — the test asks "is the challenger's best plausible value below what we measured
    for the champion?"; a strictly CI-vs-CI test is available by swapping in champ["ci_low"].
    """
    return chal["ci_high"] < champ["point"]


# --- clauses ----------------------------------------------------------------------------------


def clause_c1(champ: dict[str, Any], chal: dict[str, Any], cfg: dict[str, Any]) -> ClauseResult:
    suite = cfg["in_distribution_suite"]
    cm, tm = _map(champ, suite), _map(chal, suite)
    worse = sig_worse(cm, tm)
    point_ok = tm["point"] >= cm["point"] - cfg["c1_epsilon"]
    passed = (not worse) and point_ok
    detail = (
        f"in-dist mAP50-95 {tm['point']:.3f} vs champion {cm['point']:.3f}; "
        f"{'significantly worse' if worse else 'CI-overlap ok'}; "
        f"point {'>=' if point_ok else '<'} champion-eps"
    )
    return ClauseResult("C1_in_distribution", passed, detail, {"champion": cm, "challenger": tm})


def clause_c2(champ: dict[str, Any], chal: dict[str, Any], cfg: dict[str, Any]) -> ClauseResult:
    gated = cfg["gated_suites"]
    worst_drop = 0.0
    worst_suite = ""
    wc = wt = 0.0  # size-weighted point sums
    n = 0.0
    all_sig_worse = True
    for s in gated:
        cm, tm = _map(champ, s), _map(chal, s)
        w = float(_overall(champ, s)["n_gt"])
        wc += cm["point"] * w
        wt += tm["point"] * w
        n += w
        drop = cm["point"] - tm["point"]
        if drop > worst_drop:
            worst_drop, worst_suite = drop, s
        all_sig_worse = all_sig_worse and sig_worse(cm, tm)
    champ_agg, chal_agg = wc / n, wt / n
    catastrophic = worst_drop > cfg["c2_catastrophic_drop"]
    agg_ok = chal_agg >= champ_agg - cfg["c2_epsilon"]
    passed = (not catastrophic) and agg_ok and (not all_sig_worse)
    detail = (
        f"gated aggregate {chal_agg:.3f} vs champion {champ_agg:.3f}; "
        f"worst single-suite drop {worst_drop:.3f} on {worst_suite or 'none'} "
        f"({'CATASTROPHIC' if catastrophic else 'ok'})"
    )
    return ClauseResult(
        "C2_gated_cross_dataset",
        passed,
        detail,
        {
            "champion_aggregate": champ_agg,
            "challenger_aggregate": chal_agg,
            "worst_drop": worst_drop,
            "worst_suite": worst_suite,
        },
    )


def _iter_slices(report: dict[str, Any], suite: str) -> Iterator[tuple[str, int, Metric]]:
    """Each slice bucket as (label, n_gt, map50_95 metric): class/weather/camera/scale."""
    ov = _overall(report, suite)["metrics"]
    for cid, m in ov["ap50_95_per_class"].items():
        yield f"class:{cid}", int(m["n_gt"]), m
    for axis in ("per_weather", "per_camera", "per_scale"):
        for bucket, blob in report["suites"][suite]["slices"].get(axis, {}).items():
            yield f"{axis}:{bucket}", int(blob["n_gt"]), blob["metrics"]["map50_95"]


def clause_c3(champ: dict[str, Any], chal: dict[str, Any], cfg: dict[str, Any]) -> ClauseResult:
    suite = cfg["in_distribution_suite"]
    min_box = cfg["c3_min_box_count"]
    champ_slices = {label: (n, m) for label, n, m in _iter_slices(champ, suite)}
    blocks: list[str] = []
    warnings: list[str] = []
    for label, n_gt, tm in _iter_slices(chal, suite):
        if label not in champ_slices:
            continue
        cm = champ_slices[label][1]
        if sig_worse(cm, tm):
            msg = f"{label} (n={n_gt}) {tm['point']:.3f} < champion {cm['point']:.3f}"
            (blocks if n_gt >= min_box else warnings).append(msg)
    passed = not blocks
    detail = (
        f"{len(blocks)} large-slice regression(s)"
        + (f": {'; '.join(blocks)}" if blocks else "")
        + f"; {len(warnings)} small-slice warning(s)"
    )
    return ClauseResult(
        "C3_large_slice_regression", passed, detail, {"blocks": blocks, "warnings": warnings}
    )


def _corruption_auc(report: dict[str, Any], suite: str) -> float:
    """Mean mAP50-95 over all corruptions x severities — the degradation-curve area proxy."""
    corr = report["suites"][suite].get("corruption")
    if not corr:
        return float("nan")
    vals = [pt["map50_95"] for curve in corr["curves"].values() for pt in curve.values()]
    return sum(vals) / len(vals) if vals else float("nan")


def clause_c4(champ: dict[str, Any], chal: dict[str, Any], cfg: dict[str, Any]) -> ClauseResult:
    suite = cfg["in_distribution_suite"]
    ca, ta = _corruption_auc(champ, suite), _corruption_auc(chal, suite)
    if ca != ca or ta != ta:  # NaN: no curves in a report -> cannot judge, do not block
        return ClauseResult(
            "C4_corruption",
            True,
            "no corruption curves — clause skipped",
            {"champion_auc": ca, "challenger_auc": ta},
        )
    passed = ta >= ca - cfg["c4_corruption_tolerance"]
    return ClauseResult(
        "C4_corruption",
        passed,
        f"corruption AUC {ta:.3f} vs champion {ca:.3f} (tol {cfg['c4_corruption_tolerance']})",
        {"champion_auc": ca, "challenger_auc": ta},
    )


def clause_c5(cfg: dict[str, Any]) -> ClauseResult:
    slo = cfg["c5_slo"]
    stubbed = slo.get("status") == "stubbed"
    return ClauseResult(
        "C5_slo",
        True,
        ("STUBBED — " if stubbed else "") + slo.get("note", ""),
        {"status": slo.get("status")},
    )


def run_contract(champ: dict[str, Any], chal: dict[str, Any], cfg: dict[str, Any]) -> Decision:
    clauses = [
        clause_c1(champ, chal, cfg),
        clause_c2(champ, chal, cfg),
        clause_c3(champ, chal, cfg),
        clause_c4(champ, chal, cfg),
        clause_c5(cfg),
    ]
    warnings = clauses[2].numbers.get("warnings", [])
    return Decision(all(c.passed for c in clauses), clauses, list(warnings))


# --- reporting + CLI --------------------------------------------------------------------------


def decision_markdown(decision: Decision, champ_v: str, chal_v: str) -> str:
    verdict = "PROMOTE" if decision.promoted else "REJECT"
    lines = [
        f"# Promotion decision: {verdict}",
        f"- champion v{champ_v}  →  challenger v{chal_v}",
        "",
        "| clause | result | detail |",
        "|---|---|---|",
    ]
    for c in decision.clauses:
        lines.append(f"| {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    if decision.warnings:
        lines += [
            "",
            "## Warnings (small slices — never block)",
            *(f"- {w}" for w in decision.warnings),
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--champion-report", required=True, type=Path)
    p.add_argument("--challenger-report", required=True, type=Path)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "promotion")
    p.add_argument(
        "--apply",
        action="store_true",
        help="execute the alias flip on a PROMOTE decision (default: report only)",
    )
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    champ = json.loads(args.champion_report.read_text())
    chal = json.loads(args.challenger_report.read_text())
    decision = run_contract(champ, chal, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "decision.json").write_text(json.dumps(decision.as_dict(), indent=2) + "\n")
    md = decision_markdown(decision, "?", "?")
    (args.out / "decision.md").write_text(md)
    print(md)

    if args.apply:
        rc = apply_and_log(cfg, decision, champ, chal, args.out)
        return rc
    print("(report only — pass --apply to execute the alias reassignment)")
    return 0 if decision.promoted else 1


def apply_and_log(
    cfg: dict[str, Any],
    decision: Decision,
    champ: dict[str, Any],
    chal: dict[str, Any],
    out: Path,
) -> int:
    """Log the attempt to MLflow and, iff promoted, flip @champion to the challenger version."""
    import mlflow

    client = mlflow.MlflowClient()
    model = cfg["model_name"]
    champ_v = client.get_model_version_by_alias(model, "champion").version
    chal_v = client.get_model_version_by_alias(model, "challenger").version
    (out / "decision.md").write_text(decision_markdown(decision, champ_v, chal_v))

    mlflow.set_experiment("traffic-vision-promotion")
    with mlflow.start_run(run_name=f"promote-v{chal_v}-over-v{champ_v}"):
        mlflow.log_params(
            {
                "champion_version": champ_v,
                "challenger_version": chal_v,
                "promoted": decision.promoted,
            }
        )
        for c in decision.clauses:
            mlflow.set_tag(c.name, "PASS" if c.passed else "FAIL")
        mlflow.log_artifact(str(out / "decision.json"))
        mlflow.log_artifact(str(out / "decision.md"))
        if decision.promoted:
            client.set_registered_model_alias(model, "champion", chal_v)
            print(f"PROMOTED: @champion -> v{chal_v} (was v{champ_v})")
        else:
            failed = [c.name for c in decision.clauses if not c.passed]
            print(f"REJECTED: @champion stays v{champ_v}. Failed: {', '.join(failed)}")
    return 0 if decision.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
