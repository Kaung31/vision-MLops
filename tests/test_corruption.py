"""Corruption transforms: exact severity-0 identity, per-image-stable seeding, monotone
effects, the on-the-fly curve runner, and the ignore-region exemption composed with noise.
"""

import numpy as np
import pytest

from src.eval.corruption import (
    BRIGHTNESS_DELTA,
    CORRUPTIONS,
    apply_corruption,
    degradation_curves,
    gaussian_noise,
)
from src.eval.harness import Detections, GroundTruth, evaluate

GRAY = np.full((64, 64, 3), 128, dtype=np.uint8)


def test_severity_zero_is_exact_identity() -> None:
    for name in CORRUPTIONS:
        out = apply_corruption(GRAY, name, 0, "img")
        assert np.array_equal(out, GRAY)


def test_noise_seed_is_stable_regardless_of_order() -> None:
    a1 = apply_corruption(GRAY, "gaussian_noise", 3, "img_a")
    # corrupt other things in between — a global RNG would make a2 differ
    apply_corruption(GRAY, "gaussian_noise", 2, "img_b")
    apply_corruption(GRAY, "gaussian_noise", 5, "img_c")
    a2 = apply_corruption(GRAY, "gaussian_noise", 3, "img_a")
    assert np.array_equal(a1, a2)


def test_brightness_directions() -> None:
    assert apply_corruption(GRAY, "brightness_up", 5, "x").mean() > 128
    assert apply_corruption(GRAY, "brightness_down", 5, "x").mean() < 128


def test_noise_variance_grows_with_severity() -> None:
    def resid_std(sev: int) -> float:
        c = gaussian_noise(GRAY, sev, seed=1).astype(np.int64) - GRAY.astype(np.int64)
        return float(c.std())

    assert resid_std(1) > 0
    assert resid_std(5) > resid_std(1)


def test_blur_reduces_contrast() -> None:
    cv2 = pytest.importorskip("cv2")
    assert cv2 is not None
    checker = (np.indices((64, 64)).sum(0) % 2 * 255).astype(np.uint8)[..., None].repeat(3, 2)
    blurred = apply_corruption(checker, "gaussian_blur", 5, "x")
    assert blurred.var() < checker.var()


def test_jpeg_changes_bytes_but_keeps_shape() -> None:
    pytest.importorskip("cv2")
    out = apply_corruption(GRAY, "jpeg_compression", 5, "x")
    assert out.shape == GRAY.shape
    assert out.dtype == np.uint8


def test_degradation_curve_structure_and_baseline() -> None:
    box = [[10.0, 10.0, 50.0, 50.0]]
    gt = {"i": GroundTruth(np.array(box), np.array([0]))}
    perfect = Detections(np.array(box), np.array([0.9]), np.array([0]))
    curves = degradation_curves(
        ["i"],
        lambda iid: GRAY,
        gt,
        [0],
        lambda iid, im: perfect,
        corruptions=["gaussian_noise", "brightness_down"],
    )
    assert curves["severities"] == [0, 1, 2, 3, 4, 5]
    assert curves["curves"]["gaussian_noise"][0]["map50"] == 1.0  # sev 0 == clean baseline
    assert curves["curves"]["brightness_down"][5]["param"] == BRIGHTNESS_DELTA[4]


def test_ignore_exemption_survives_corruption() -> None:
    # a black-filled ignore rectangle; heavy noise injects texture there
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    img[10:40, 10:40] = 0
    corrupted = apply_corruption(img, "gaussian_noise", 5, "x")
    assert corrupted[10:40, 10:40].std() > 0  # texture injected into the once-black region

    # a detection in that ignore region must still be FP-exempt (composition pinned)
    preds = {
        "x": Detections(
            np.array([[10.0, 10.0, 40.0, 40.0], [60.0, 60.0, 80.0, 80.0]]),
            np.array([0.9, 0.8]),
            np.array([0, 0]),
        )
    }
    gt = {
        "x": GroundTruth(
            np.array([[60.0, 60.0, 80.0, 80.0]]),
            np.array([0]),
            ignore=np.array([[10.0, 10.0, 40.0, 40.0]]),
        )
    }
    assert abs(evaluate(preds, gt, [0])["map50"] - 1.0) < 1e-9
