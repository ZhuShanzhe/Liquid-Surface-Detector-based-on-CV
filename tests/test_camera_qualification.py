from __future__ import annotations

import numpy as np

from liquid_depth.camera_qualification import (
    analyze_plane_frame,
    load_plane_capture_directory,
    qualify_plane_captures,
    simulate_plane_captures,
)
from liquid_depth.site_calibration_simulation import CAMERA_ERROR_PROFILES
from liquid_depth.system_runtime import _camera_depth_correction, _depth_input


def test_plane_frame_uses_center_roi_and_rejects_invalid_pixels() -> None:
    depth = np.full((20, 30), 1000, dtype=np.uint16)
    depth[:5] = 0
    result = analyze_plane_frame(
        depth,
        1.0,
        depth_scale_to_m=0.001,
        roi_fraction=0.5,
    )
    assert result["accepted"]
    assert result["median_m"] == 1.0
    assert result["within_tolerance"]


def test_simulated_plane_calibration_reduces_systematic_error() -> None:
    profile = CAMERA_ERROR_PROFILES["long_baseline_stereo"]
    captures = simulate_plane_captures(
        profile,
        (0.3, 1.0, 3.0, 5.0, 8.0),
        frames_per_distance=12,
        seed=9,
        shape=(32, 40),
    )
    report = qualify_plane_captures(
        captures,
        depth_scale_to_m=1.0,
        calibration_frames_per_distance=6,
        profile=profile,
    )
    assert report["calibration"]["samples"] == 5
    assert report["corrected_all"]["mae_m"] < report["raw_all"]["mae_m"]


def test_tof_excludes_out_of_spec_distances_from_fit() -> None:
    profile = CAMERA_ERROR_PROFILES["tof_typical"]
    captures = simulate_plane_captures(
        profile,
        (0.3, 1.0, 3.0, 5.0, 8.0),
        frames_per_distance=10,
        seed=4,
        shape=(24, 32),
    )
    report = qualify_plane_captures(
        captures,
        depth_scale_to_m=1.0,
        calibration_frames_per_distance=5,
        profile=profile,
    )
    assert report["calibration"]["samples"] == 3
    assert not report["by_distance"]["0.3"]["used_for_calibration"]
    assert not report["by_distance"]["8"]["used_for_calibration"]


def test_load_plane_capture_directory(tmp_path) -> None:
    frame = tmp_path / "1m" / "frame-001"
    frame.mkdir(parents=True)
    np.save(frame / "depth.npy", np.full((2, 3), 1000, dtype=np.uint16))
    captures = load_plane_capture_directory(tmp_path)
    assert list(captures) == [1.0]
    assert captures[1.0][0].shape == (2, 3)


def test_depth_input_applies_verified_correction_without_reviving_holes() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.asarray([[1000, 0], [2000, 1000]], dtype=np.uint16)
    inputs = _depth_input(
        rgb,
        depth,
        (0, 0, 2, 2),
        (2, 2),
        0.001,
        10.0,
        depth_correction_scale=2.0,
        depth_correction_offset_m=0.1,
    )
    assert np.isclose(inputs[0, 0, 3], 0.21)
    assert inputs[0, 1, 3] == 0.0
    assert inputs[0, 1, 4] == 0.0
    assert inputs[1, 0, 4] == 1.0


def test_only_verified_camera_correction_is_activated() -> None:
    simulated = {"depth_correction": {"scale": 2.0, "offset_m": 0.1, "status": "simulation_only"}}
    verified = {"depth_correction": {"scale": 2.0, "offset_m": 0.1, "status": "verified"}}
    assert _camera_depth_correction(simulated) == (1.0, 0.0)
    assert _camera_depth_correction(verified) == (2.0, 0.1)
