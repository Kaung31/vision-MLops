"""Versioned eval report: one command -> a full frozen-harness report for a checkpoint.

Per suite it computes the mandatory slices (class/weather/camera/scale) with bootstrap CIs, and
on the in-distribution suite the corruption degradation curves; across suites it reports the
in-distribution-vs-cross-dataset mAP gap as the headline. Every suite hands the report the
harness's in-memory types (GroundTruth + slice maps + a load_image callable), so the on-disk
difference between UA-DETRAC (YOLO on disk) and the cross-dataset sets (compact JSON) never
reaches the report core.

Point estimates come from `evaluate_slices` (which also emits the self-describing `spec`); the
CIs are merged in from `bootstrap_slices` (its point == evaluate's, by the consistency test).

Needs the `eval` group (ultralytics + cv2). Runs inference on this Mac's MPS for the zero-shot
floor (allowed: inference, not training). Run:
  uv run --group eval python -m src.eval.report --model yolov8n.pt --suite all --out reports/floor
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from src.data.taxonomy import canonical_names
from src.data.taxonomy import load as load_taxonomy
from src.eval.adapter import YoloDetector
from src.eval.bootstrap import DEFAULT_N_RESAMPLES, bootstrap_slices
from src.eval.corruption import degradation_curves
from src.eval.harness import Detections, GroundTruth, evaluate_slices

U8 = NDArray[np.uint8]
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = REPO_ROOT / "data" / "processed"
CAMERA_GROUPS = REPO_ROOT / "configs" / "camera_groups.yaml"
UA_IMG_W, UA_IMG_H = 960.0, 540.0


@dataclass
class Suite:
    name: str
    role: str  # "in-distribution" | "gated-eval"
    ground_truth: dict[str, GroundTruth]
    weather_of: dict[str, str]
    camera_of: dict[str, str]
    load_image: Callable[[str], U8]
    meta: dict[str, Any]


def _imread(path: Path) -> U8:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2 failed to read {path}")
    return np.asarray(img, dtype=np.uint8)


def _class_ids() -> list[int]:
    tax = load_taxonomy()
    return sorted(int(c["id"]) for c in tax["canonical"])


# --- suite loaders --------------------------------------------------------------------------


def _seq_maps() -> tuple[dict[str, str], dict[str, str]]:
    """(sequence -> group, group -> single weather) from the camera-group config."""
    groups = yaml.safe_load(CAMERA_GROUPS.read_text())["groups"]
    seq_to_group: dict[str, str] = {}
    group_weather: dict[str, str] = {}
    for g, spec in groups.items():
        weathers = spec["weathers"]
        group_weather[g] = weathers[0] if len(weathers) == 1 else "|".join(sorted(weathers))
        for seq in spec["sequences"]:
            seq_to_group[seq] = g
    return seq_to_group, group_weather


def _yolo_to_gt(text: str, ignore: NDArray[np.float64]) -> GroundTruth:
    boxes, labels = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        cid, cx, cy, w, h = (float(x) for x in line.split())
        boxes.append(
            [
                (cx - w / 2) * UA_IMG_W,
                (cy - h / 2) * UA_IMG_H,
                (cx + w / 2) * UA_IMG_W,
                (cy + h / 2) * UA_IMG_H,
            ]
        )
        labels.append(int(cid))
    return GroundTruth(
        np.array(boxes, dtype=np.float64).reshape(-1, 4),
        np.array(labels, dtype=np.int64),
        ignore,
    )


def _ignore_xyxy(json_path: Path) -> NDArray[np.float64]:
    regions = json.loads(json_path.read_text())["ignore_regions"]
    return np.array(
        [[r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"]] for r in regions],
        dtype=np.float64,
    ).reshape(-1, 4)


def load_ua_detrac(root: Path = PROCESSED / "eval-frozen-v1") -> Suite:
    """The in-distribution suite: UA-DETRAC eval-frozen-v1 (YOLO labels + ignore-region JSON)."""
    seq_to_group, group_weather = _seq_maps()
    ignore_cache: dict[str, NDArray[np.float64]] = {}
    gt: dict[str, GroundTruth] = {}
    weather_of: dict[str, str] = {}
    camera_of: dict[str, str] = {}
    for label in sorted((root / "labels" / "test").glob("*.txt")):
        stem = label.stem
        seq = stem.rsplit("_img", 1)[0]
        if seq not in ignore_cache:
            ignore_cache[seq] = _ignore_xyxy(root / "ignore_regions" / f"{seq}.json")
        gt[stem] = _yolo_to_gt(label.read_text(), ignore_cache[seq])
        group = seq_to_group[seq]
        weather = group_weather[group]
        if "|" in weather:  # the group->weather approximation must never silently apply
            raise ValueError(f"{seq}: group {group} multi-weather ({weather}); need per-seq")
        camera_of[stem], weather_of[stem] = group, weather

    def load(stem: str) -> U8:
        return _imread(root / "images" / "test" / f"{stem}.jpg")

    return Suite(
        "ua-detrac",
        "in-distribution",
        gt,
        weather_of,
        camera_of,
        load,
        {"source": "eval-frozen-v1", "n_images": len(gt)},
    )


def load_cross(root: Path, name: str) -> Suite:
    """A gated cross-dataset suite from a compact annotations.json (px xyxy boxes + ignore)."""
    manifest = json.loads((root / "annotations.json").read_text())
    gt: dict[str, GroundTruth] = {}
    weather_of: dict[str, str] = {}
    camera_of: dict[str, str] = {}
    file_of: dict[str, str] = {}
    for im in manifest["images"]:
        iid = im["id"]
        boxes = np.array([b[:4] for b in im["boxes"]], dtype=np.float64).reshape(-1, 4)
        labels = np.array([int(b[4]) for b in im["boxes"]], dtype=np.int64)
        ignore = np.array(im["ignore"], dtype=np.float64).reshape(-1, 4)
        gt[iid] = GroundTruth(boxes, labels, ignore)
        weather_of[iid], camera_of[iid], file_of[iid] = im["weather"], im["camera"], im["file"]

    def load(iid: str) -> U8:
        return _imread(root / file_of[iid])

    return Suite(name, "gated-eval", gt, weather_of, camera_of, load, dict(manifest["meta"]))


# --- evaluation -----------------------------------------------------------------------------


def predict_all(detector: YoloDetector, suite: Suite) -> dict[str, Detections]:
    ids = sorted(suite.ground_truth)
    preds: dict[str, Detections] = {}
    for i, iid in enumerate(ids, 1):
        preds[iid] = detector(suite.load_image(iid))
        if i % 250 == 0 or i == len(ids):
            print(f"    {suite.name}: predicted {i}/{len(ids)}", flush=True)
    return preds


def _merge_point_ci(point_slices: dict[str, Any], ci_slices: dict[str, Any]) -> dict[str, Any]:
    """Attach bootstrap ci_low/ci_high to each evaluate_slices metric, keyed by the axis buckets."""

    def merge_axis(pt: dict[str, Any], ci: dict[str, Any]) -> dict[str, Any]:
        out = {"n_gt": pt["n_gt"], "n_images": pt["n_images"], "metrics": {}}
        for metric in ("map50", "map50_95"):
            c = ci[metric]
            out["metrics"][metric] = {
                "point": pt[metric],
                "ci_low": c["ci_low"],
                "ci_high": c["ci_high"],
            }
        for key, cls in (("ap50_per_class", "ap50_class"), ("ap50_95_per_class", "ap50_95_class")):
            out["metrics"][key] = {
                int(cid): {
                    "point": val,
                    "ci_low": ci[f"{cls}_{cid}"]["ci_low"],
                    "ci_high": ci[f"{cls}_{cid}"]["ci_high"],
                    "n_gt": ci[f"{cls}_{cid}"]["n_gt"],
                }
                for cid, val in pt[key].items()
            }
        return out

    merged: dict[str, Any] = {"spec": point_slices["spec"]}
    merged["overall"] = merge_axis(point_slices["overall"], ci_slices["overall"])
    for axis in ("per_weather", "per_camera", "per_scale"):
        merged[axis] = {
            bucket: merge_axis(point_slices[axis][bucket], ci_slices[axis][bucket])
            for bucket in point_slices[axis]
        }
    return merged


def evaluate_suite(
    detector: YoloDetector,
    suite: Suite,
    class_ids: list[int],
    n_resamples: int,
    seed: int,
    with_corruption: bool = True,
) -> dict[str, Any]:
    preds = predict_all(detector, suite)
    point = evaluate_slices(preds, suite.ground_truth, class_ids, suite.weather_of, suite.camera_of)
    ci = bootstrap_slices(
        preds,
        suite.ground_truth,
        class_ids,
        suite.weather_of,
        suite.camera_of,
        n_resamples=n_resamples,
        seed=seed,
    )
    report: dict[str, Any] = {
        "role": suite.role,
        "meta": suite.meta,
        "n_resamples": n_resamples,
        "slices": _merge_point_ci(point, ci),
    }
    if suite.role == "in-distribution" and with_corruption:
        ids = sorted(suite.ground_truth)
        report["corruption"] = degradation_curves(
            ids,
            suite.load_image,
            suite.ground_truth,
            class_ids,
            lambda iid, im: detector(im),
        )
    return report


def _overall_map(suite_report: dict[str, Any], metric: str) -> dict[str, Any]:
    m: dict[str, Any] = suite_report["slices"]["overall"]["metrics"][metric]
    return m


def headline(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """In-distribution vs gated cross-dataset gap (size-weighted aggregate + per suite)."""
    in_dist = next((r for r in reports.values() if r["role"] == "in-distribution"), None)
    gated = {n: r for n, r in reports.items() if r["role"] == "gated-eval"}
    if in_dist is None or not gated:
        return {}
    base = _overall_map(in_dist, "map50_95")["point"]
    per_suite = {n: _overall_map(r, "map50_95")["point"] for n, r in gated.items()}
    weights = {n: r["slices"]["overall"]["n_gt"] for n, r in gated.items()}
    total = sum(weights.values()) or 1
    agg = sum(per_suite[n] * weights[n] for n in gated) / total
    return {
        "in_distribution_map50_95": base,
        "gated_aggregate_map50_95": agg,
        "gap": base - agg,
        "per_suite": {
            n: {"map50_95": v, "gap": base - v, "n_gt": weights[n]} for n, v in per_suite.items()
        },
    }


def _git(*args: str) -> str:
    """A git command's stripped stdout, or 'unknown' if git/repo is unavailable."""
    try:
        out = subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
        return out or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance(model: str, seed: int, n_resamples: int, device: str) -> dict[str, Any]:
    floor = "yolo" in Path(model).name.lower() and "coco" not in model.lower()
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_version": _git("describe", "--tags", "--match", "eval-harness-*", "--always"),
        "git_commit": _git("rev-parse", "HEAD"),
        "model": model,
        "device": device,
        "seed": seed,
        "n_resamples": n_resamples,
        "class_names": canonical_names(load_taxonomy()),
        "note": "zero-shot COCO-pretrained YOLO floor" if floor else "",
    }


# --- rendering ------------------------------------------------------------------------------


def _fmt(m: dict[str, Any]) -> str:
    return f"{m['point']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def to_markdown(full: dict[str, Any], names: list[str]) -> str:
    p = full["provenance"]
    out = [
        "# Eval report",
        "",
        f"- model: `{p['model']}`  | device: {p['device']}  | harness: `{p['harness_version']}`",
        f"- git: `{p['git_commit'][:10]}`  | seed: {p['seed']}  | resamples: {p['n_resamples']}",
        f"- generated: {p['generated_utc']}",
        "",
    ]
    if full.get("headline"):
        h = full["headline"]
        out += [
            "## Headline — generalization gap",
            "",
            f"- in-distribution mAP50-95: **{h['in_distribution_map50_95']:.3f}**",
            f"- gated cross-dataset (weighted) mAP50-95: **{h['gated_aggregate_map50_95']:.3f}**",
            f"- **gap: {h['gap']:.3f}**",
            "",
            "| gated suite | mAP50-95 | gap | n_gt |",
            "|---|---|---|---|",
        ]
        for n, s in h["per_suite"].items():
            out.append(f"| {n} | {s['map50_95']:.3f} | {s['gap']:.3f} | {s['n_gt']} |")
        out.append("")

    for name, r in full["suites"].items():
        sl = r["slices"]
        ov = sl["overall"]["metrics"]
        out += [
            f"## {name} ({r['role']})",
            "",
            f"- mAP50: {_fmt(ov['map50'])}  | mAP50-95: {_fmt(ov['map50_95'])}"
            f"  | n_gt {sl['overall']['n_gt']}",
            "",
            "| class | AP50 | AP50-95 | n_gt |",
            "|---|---|---|---|",
        ]
        for cid, m in ov["ap50_per_class"].items():
            m95 = ov["ap50_95_per_class"][cid]
            out.append(f"| {names[cid]} | {_fmt(m)} | {_fmt(m95)} | {m['n_gt']} |")
        out.append("")
        axes = (("per_weather", "weather"), ("per_camera", "camera"), ("per_scale", "scale"))
        for axis, title in axes:
            out += [f"### by {title}", "", "| bucket | mAP50-95 | n_gt |", "|---|---|---|"]
            for bucket, b in sl[axis].items():
                out.append(f"| {bucket} | {_fmt(b['metrics']['map50_95'])} | {b['n_gt']} |")
            out.append("")
        if "corruption" in r:
            out += [
                "### corruption (mAP50-95 by severity)",
                "",
                "| corruption | "
                + " | ".join(f"s{s}" for s in r["corruption"]["severities"])
                + " |",
                "|" + "---|" * (len(r["corruption"]["severities"]) + 1),
            ]
            sevs = r["corruption"]["severities"]
            for corr, curve in r["corruption"]["curves"].items():
                cells = " | ".join(f"{curve[s]['map50_95']:.3f}" for s in sevs)
                out.append(f"| {corr} | {cells} |")
            out.append("")
    return "\n".join(out)


# --- entrypoint -----------------------------------------------------------------------------

SUITE_LOADERS: dict[str, Callable[[], Suite]] = {
    "ua-detrac": load_ua_detrac,
    "mio-tcd": lambda: load_cross(PROCESSED / "mio-tcd-eval-v1", "mio-tcd"),
    "rainsnow": lambda: load_cross(PROCESSED / "rainsnow-eval-v1", "rainsnow"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=[*SUITE_LOADERS, "all"], default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--n-resamples", type=int, default=DEFAULT_N_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-corruption", action="store_true", help="skip the corruption sweep")
    parser.add_argument("--limit", type=int, default=0, help="debug: cap images per suite")
    args = parser.parse_args(argv)

    names = canonical_names(load_taxonomy())
    class_ids = _class_ids()
    detector = YoloDetector(args.model, device=args.device)
    suite_names = list(SUITE_LOADERS) if args.suite == "all" else [args.suite]

    reports: dict[str, dict[str, Any]] = {}
    for name in suite_names:
        print(f"  loading suite {name}...", flush=True)
        suite = SUITE_LOADERS[name]()
        if args.limit:
            keep = sorted(suite.ground_truth)[: args.limit]
            suite.ground_truth = {k: suite.ground_truth[k] for k in keep}
        reports[name] = evaluate_suite(
            detector,
            suite,
            class_ids,
            args.n_resamples,
            args.seed,
            with_corruption=not args.no_corruption,
        )

    full = {
        "provenance": provenance(args.model, args.seed, args.n_resamples, args.device),
        "headline": headline(reports),
        "suites": reports,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(full, indent=2) + "\n")
    (args.out / "report.md").write_text(to_markdown(full, names))
    print(f"wrote {args.out / 'report.json'} and {args.out / 'report.md'}")
    if full["headline"]:
        print(f"HEADLINE gap (in-dist - gated): {full['headline']['gap']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
