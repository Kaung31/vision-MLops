"""Checksum pin/verify logic — the Law 6 guard, tested without any network."""

import hashlib
from pathlib import Path

import pytest

from src.data.download import ChecksumMismatch, sha256_of, verify_or_pin


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"traffic-vision" * 1000)
    assert sha256_of(f) == hashlib.sha256(f.read_bytes()).hexdigest()


def test_pins_when_absent() -> None:
    locks: dict[str, str | None] = {}
    assert verify_or_pin(locks, "k", "abc") == "pinned"
    assert locks["k"] == "abc"


def test_verifies_when_matching() -> None:
    locks: dict[str, str | None] = {"k": "abc"}
    assert verify_or_pin(locks, "k", "abc") == "verified"


def test_raises_on_mismatch() -> None:
    locks: dict[str, str | None] = {"k": "abc"}
    with pytest.raises(ChecksumMismatch):
        verify_or_pin(locks, "k", "def")
