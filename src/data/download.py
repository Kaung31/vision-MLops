"""Fetch + checksum-verify every dataset archive.

Law 5: archives land in ``data/raw/`` (gitignored); nothing but pointers/checksums enter git.
Law 6: the first verified download *pins* each archive's SHA256 into
``configs/datasets/checksums.lock.yaml``; every run after re-verifies and refuses to
proceed on mismatch, so a mutated/rotted mirror fails loudly instead of silently.

Usage (needs the ``data`` dependency group for kaggle/gdown):
    uv run --group data python -m src.data.download --all
    uv run --group data python -m src.data.download --dataset mio-tcd
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "datasets"
LOCK_FILE = CONFIG_DIR / "checksums.lock.yaml"
DATA_ROOT = REPO_ROOT / "data" / "raw"
_CHUNK = 1 << 20  # 1 MiB
_PROGRESS_EVERY = 1 << 28  # 256 MiB


class ChecksumMismatch(RuntimeError):
    """A downloaded archive's SHA256 does not match the pinned value."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_pin(locks: dict[str, str | None], key: str, computed: str) -> str:
    """Pin on first sight, verify thereafter. Returns 'pinned' or 'verified'.

    Raises ChecksumMismatch if a pinned value disagrees with ``computed``.
    """
    pinned = locks.get(key)
    if pinned is None:
        locks[key] = computed
        return "pinned"
    if pinned != computed:
        raise ChecksumMismatch(f"{key}: expected {pinned}, got {computed}")
    return "verified"


def load_locks() -> dict[str, str | None]:
    if not LOCK_FILE.exists():
        return {}
    data: Any = yaml.safe_load(LOCK_FILE.read_text())
    return dict(data) if data else {}


def save_locks(locks: dict[str, str | None]) -> None:
    header = "# Auto-managed by src/data/download.py — pinned SHA256 per dataset archive.\n"
    LOCK_FILE.write_text(header + yaml.safe_dump(locks, sort_keys=True))


def _download_http(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "traffic-vision-platform"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        next_mark = _PROGRESS_EVERY
        while chunk := resp.read(_CHUNK):
            out.write(chunk)
            done += len(chunk)
            if done >= next_mark:
                pct = f" ({done * 100 // total}%)" if total else ""
                print(f"  ... {done >> 20} MiB{pct}", flush=True)
                next_mark += _PROGRESS_EVERY


def _download_gdrive(file_id: str, dest: Path) -> None:
    subprocess.run(
        ["gdown", f"https://drive.google.com/uc?id={file_id}", "-O", str(dest)],
        check=True,
    )


def _download_kaggle(slug: str, dest_dir: Path) -> None:
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest_dir), "--force"],
        check=True,
    )


def _fetch(archive: dict[str, Any], dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / str(archive["filename"])
    method = archive["method"]
    if method == "http":
        _download_http(archive["source"], dest)
    elif method == "gdrive":
        _download_gdrive(archive["source"], dest)
    elif method == "kaggle":
        _download_kaggle(archive["source"], dest_dir)  # writes <slug-tail>.zip == filename
    else:
        raise ValueError(f"unknown download method: {method!r}")
    if not dest.exists():
        raise FileNotFoundError(f"expected {dest} after fetch but it is missing")
    return dest


def process_dataset(cfg: dict[str, Any], locks: dict[str, str | None]) -> None:
    name = cfg["name"]
    dest_dir = DATA_ROOT / name
    print(f"== {name} ({cfg.get('role', '?')}) ==")
    for archive in cfg["archives"]:
        key = archive["id"]
        dest = dest_dir / archive["filename"]
        if dest.exists() and locks.get(key) is not None and sha256_of(dest) == locks[key]:
            print(f"[skip] {key}: already present and verified")
            continue
        path = _fetch(archive, dest_dir)
        status = verify_or_pin(locks, key, sha256_of(path))
        save_locks(locks)
        print(f"[{status}] {key}: {locks[key]}")


def _dataset_configs() -> list[Path]:
    return sorted(p for p in CONFIG_DIR.glob("*.yaml") if p.name != LOCK_FILE.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="fetch every dataset")
    group.add_argument("--dataset", help="fetch one dataset by name (config stem)")
    parser.add_argument(
        "--skip-stretch",
        action="store_true",
        help="with --all, skip ungated-stretch datasets (e.g. VisDrone)",
    )
    args = parser.parse_args(argv)

    locks = load_locks()
    if args.dataset:
        configs = [CONFIG_DIR / f"{args.dataset}.yaml"]
        if not configs[0].exists():
            parser.error(f"no dataset config: {configs[0]}")
    else:
        configs = _dataset_configs()

    failed_optional: list[str] = []
    for cfg_path in configs:
        cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text())
        role = cfg.get("role")
        if args.skip_stretch and role == "ungated-stretch":
            print(f"[skip] {cfg['name']}: ungated-stretch (--skip-stretch)")
            continue
        try:
            process_dataset(cfg, locks)
        except (subprocess.CalledProcessError, OSError, ChecksumMismatch) as exc:
            # An ungated-stretch set (e.g. VisDrone) never gates anything and is only
            # consumed by the Phase 2 harness; a flaky third-party mirror must not fail
            # the whole provenance run. Core datasets still hard-fail.
            if role == "ungated-stretch":
                print(
                    f"[warn] {cfg['name']}: optional stretch dataset unavailable — skipped ({exc})"
                )
                failed_optional.append(str(cfg["name"]))
                continue
            raise

    if failed_optional:
        print(f"Core datasets verified. Deferred optional stretch: {', '.join(failed_optional)}")
    else:
        print("All requested datasets fetched and checksum-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
