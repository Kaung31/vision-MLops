"""Every-Nth sampling: positions, stride, and edge cases."""

import pytest

from src.data.sample import SAMPLING_STRIDE, select_every_nth


def test_default_stride_keeps_first_of_each_group() -> None:
    kept = select_every_nth(range(1, 101))  # frames 1..100, stride 10
    assert kept == [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
    assert len(kept) == 10


def test_custom_stride() -> None:
    assert select_every_nth(range(1, 21), stride=5) == [1, 6, 11, 16]


def test_empty_in_empty_out() -> None:
    assert select_every_nth([]) == []


def test_fewer_than_stride_keeps_only_first() -> None:
    assert select_every_nth([3, 1, 2]) == [1]  # <stride -> first only, and sorted


def test_deduplicates_and_sorts() -> None:
    assert select_every_nth([5, 5, 1, 1, 1]) == [1]


def test_stride_must_be_positive() -> None:
    with pytest.raises(ValueError):
        select_every_nth(range(10), stride=0)


def test_default_stride_is_ten() -> None:
    assert SAMPLING_STRIDE == 10
