"""Dataset-directory access with a code-enforced read-only guard (Law 4).

Frozen datasets (eval-frozen-v1, prod-holdout-v1) carry a ``.readonly`` sentinel whose
*contents* say why. Every dataset write routes through ``open_dataset_dir(path, "w")``,
which refuses to return a writable handle when the sentinel is present — a new writer
cannot forget the check, because it cannot obtain a write handle without passing it.

Why a sentinel file and not chmod: the sentinel travels through ``dvc pull`` /checkout on
Colab or the rented box; chmod bits do not survive checkout across machines/filesystems.
Code-level enforcement is also the guide's philosophy (Law 4: a code guard, not memory).
Local chmod is at most a bonus, never the mechanism of record.
"""

from __future__ import annotations

from pathlib import Path

READONLY_SENTINEL = ".readonly"


class ReadOnlyDatasetError(RuntimeError):
    """An attempt was made to write to a frozen dataset directory."""


class DatasetDir:
    """A dataset directory handle. Only a writable handle can write files."""

    def __init__(self, root: Path, writable: bool) -> None:
        self.root = root
        self._writable = writable

    def path(self, relative: str) -> Path:
        return self.root / relative

    def _guard(self) -> None:
        if not self._writable:
            raise ReadOnlyDatasetError(f"{self.root} is read-only; refusing to write")

    def write_text(self, relative: str, text: str) -> None:
        self._guard()
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def write_bytes(self, relative: str, data: bytes) -> None:
        self._guard()
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def open_dataset_dir(root: Path, mode: str) -> DatasetDir:
    """Open a dataset directory. mode='w' fails loud if the readonly sentinel is present."""
    root = Path(root)
    if mode not in ("r", "w"):
        raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")
    sentinel = root / READONLY_SENTINEL
    if mode == "w":
        if sentinel.exists():
            raise ReadOnlyDatasetError(f"{root} is frozen: {sentinel.read_text().strip()}")
        root.mkdir(parents=True, exist_ok=True)
        return DatasetDir(root, writable=True)
    return DatasetDir(root, writable=False)


def mark_readonly(root: Path, reason: str) -> None:
    """Freeze a dataset directory by writing the sentinel with a human-readable reason."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / READONLY_SENTINEL).write_text(reason.rstrip() + "\n")
