"""Every-Nth frame sampling for the working training pool.

UA-DETRAC is 25 FPS video: adjacent frames are near-duplicates, so the training pool
keeps every 10th frame (guide Phase 1 step 4). This shrinks ~82k labeled frames to ~8k
without losing scene diversity. Applied to train/val/test clips only — the
production-holdout keeps full-rate frames for realistic streaming replay (enforced at
conversion, not here).
"""

from __future__ import annotations

from collections.abc import Iterable

SAMPLING_STRIDE = 10


def select_every_nth(frame_numbers: Iterable[int], stride: int = SAMPLING_STRIDE) -> list[int]:
    """Keep every ``stride``-th frame: the first of each group, in sorted order.

    Deduplicates and sorts first, so the result is deterministic regardless of input
    order. ``[]`` in -> ``[]`` out; fewer than ``stride`` frames -> just the first.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return sorted(set(frame_numbers))[::stride]
