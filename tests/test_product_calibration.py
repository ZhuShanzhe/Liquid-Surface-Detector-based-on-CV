from __future__ import annotations

import cv2
import numpy as np
import pytest

from liquid_depth.rail_calibration import (
    fit_rail_calibration,
    intersect_curve_with_rail,
    rail_depth_from_y,
)
from liquid_depth.rail_runtime import estimate_reference_motion_px


def test_three_arbitrary_points_fit_affine_rail() -> None:
    points = [
        {"x_px": 101.0, "y_px": 70.0, "depth_m": 0.24},
        {"x_px": 99.0, "y_px": 220.0, "depth_m": 0.54},
        {"x_px": 100.0, "y_px": 405.0, "depth_m": 0.91},
    ]
    calibration = fit_rail_calibration(points)

    assert calibration["model"] == "affine"
    assert calibration["rail_x_px"] == pytest.approx(100.0)
    assert calibration["loocv_mae_m"] < 1e-12
    assert rail_depth_from_y(calibration, 300.0) == pytest.approx(0.7)


def test_five_points_can_select_projective_mapping() -> None:
    y_values = np.array([20.0, 95.0, 180.0, 310.0, 460.0])
    depth_values = (0.02 * y_values + 1.0) / (0.006 * y_values + 2.0)
    points = [
        {"x_px": 200.0 + index, "y_px": y, "depth_m": depth}
        for index, (y, depth) in enumerate(zip(y_values, depth_values, strict=True))
    ]
    calibration = fit_rail_calibration(points)

    assert calibration["model"] == "projective"
    expected = (0.02 * 250.0 + 1.0) / (0.006 * 250.0 + 2.0)
    assert rail_depth_from_y(calibration, 250.0) == pytest.approx(expected, abs=1e-8)


def test_invalid_rail_points_are_rejected() -> None:
    non_monotonic = [
        {"x_px": 100.0, "y_px": 50.0, "depth_m": 0.1},
        {"x_px": 100.0, "y_px": 150.0, "depth_m": 0.4},
        {"x_px": 100.0, "y_px": 250.0, "depth_m": 0.3},
    ]
    with pytest.raises(ValueError, match="monotonically"):
        fit_rail_calibration(non_monotonic)

    wide = [
        {"x_px": 40.0, "y_px": 50.0, "depth_m": 0.1},
        {"x_px": 100.0, "y_px": 150.0, "depth_m": 0.2},
        {"x_px": 160.0, "y_px": 250.0, "depth_m": 0.3},
    ]
    with pytest.raises(ValueError, match="reference rail"):
        fit_rail_calibration(wide)


def test_curve_intersection_uses_local_confidence() -> None:
    curve = np.array([[20.0, 30.0], [60.0, 50.0], [100.0, 70.0]])
    confidence = np.array([0.2, 0.8, 0.6])

    y_value, local_confidence = intersect_curve_with_rail(curve, confidence, 80.0)

    assert y_value == pytest.approx(60.0)
    assert local_confidence == pytest.approx(0.7)


def _textured_image() -> np.ndarray:
    image = np.zeros((260, 340, 3), dtype=np.uint8)
    generator = np.random.default_rng(17)
    for _ in range(180):
        center = tuple(int(value) for value in generator.integers([5, 5], [335, 255]))
        color = tuple(int(value) for value in generator.integers(60, 255, size=3))
        cv2.circle(image, center, 2, color, -1)
    return image


def test_reference_motion_detects_camera_shift() -> None:
    reference = _textured_image()
    matrix = np.float32([[1.0, 0.0, 7.0], [0.0, 1.0, 4.0]])
    shifted = cv2.warpAffine(reference, matrix, (reference.shape[1], reference.shape[0]))
    crop = (15, 15, 325, 245)

    stationary = estimate_reference_motion_px(reference, reference, crop)
    motion = estimate_reference_motion_px(reference, shifted, crop)

    assert stationary is not None and stationary < 0.3
    assert motion is not None and motion == pytest.approx(np.hypot(7.0, 4.0), abs=1.5)
