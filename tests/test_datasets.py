"""Read-only dataset guard: a frozen dir cannot be reopened for writing, and a read
handle cannot write. Belt-and-braces for Law 4 (eval sets never modified/trained on).
"""

from pathlib import Path

import pytest

from src.data.datasets import ReadOnlyDatasetError, mark_readonly, open_dataset_dir


def test_write_then_freeze_then_refuse(tmp_path: Path) -> None:
    ds = open_dataset_dir(tmp_path / "eval", "w")
    ds.write_text("labels/a.txt", "0 0.5 0.5 0.1 0.1\n")
    assert (tmp_path / "eval" / "labels" / "a.txt").exists()

    mark_readonly(tmp_path / "eval", "frozen eval set — see ADR 0006; no writer may modify")

    # can no longer obtain a write handle
    with pytest.raises(ReadOnlyDatasetError):
        open_dataset_dir(tmp_path / "eval", "w")


def test_read_handle_cannot_write(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir()
    ds = open_dataset_dir(tmp_path / "eval", "r")
    with pytest.raises(ReadOnlyDatasetError):
        ds.write_text("x.txt", "nope")


def test_sentinel_carries_reason(tmp_path: Path) -> None:
    mark_readonly(tmp_path / "d", "because reasons")
    assert "because reasons" in (tmp_path / "d" / ".readonly").read_text()
