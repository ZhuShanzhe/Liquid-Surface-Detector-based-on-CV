from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import fit_output_calibration
from .site_calibration_simulation import (
    CameraErrorProfile,
    _camera_measurement,
    _sample_site_error,
)

DEFAULT_QUALIFICATION_DISTANCES_M = (0.3, 1.0, 3.0, 5.0, 8.0)
_DISTANCE_PATTERN = re.compile(r"^(?:distance[_-]?)?([0-9]+(?:\.[0-9]+)?)m?$")


def measurement_tolerance_m(truth_m: float) -> float:
    """Industrial research target: max(3 mm, 1% of the reference distance)."""

    return max(0.003, 0.01 * float(truth_m))


def analyze_plane_frame(
    depth: np.ndarray,
    truth_m: float,
    *,
    depth_scale_to_m: float,
    roi_fraction: float = 0.5,
) -> dict[str, float | int | bool]:
    if depth.ndim != 2:
        raise ValueError("Depth frame must be a two-dimensional array")
    if truth_m <= 0 or depth_scale_to_m <= 0:
        raise ValueError("Truth distance and depth scale must be positive")
    if not 0 < roi_fraction <= 1:
        raise ValueError("roi_fraction must be in (0, 1]")

    height, width = depth.shape
    roi_height = max(1, round(height * roi_fraction))
    roi_width = max(1, round(width * roi_fraction))
    top = (height - roi_height) // 2
    left = (width - roi_width) // 2
    values = depth[top : top + roi_height, left : left + roi_width].astype(np.float64)
    values *= depth_scale_to_m
    valid = np.isfinite(values) & (values > 0)
    valid_count = int(np.count_nonzero(valid))
    total_count = int(values.size)
    if valid_count == 0:
        return {
            "accepted": False,
            "valid_pixels": 0,
            "total_pixels": total_count,
            "valid_fraction": 0.0,
            "median_m": float("nan"),
            "signed_error_m": float("nan"),
            "absolute_error_m": float("nan"),
            "robust_sigma_m": float("nan"),
            "within_tolerance": False,
        }
    selected = values[valid]
    median = float(np.median(selected))
    robust_sigma = float(1.4826 * np.median(np.abs(selected - median)))
    error = median - truth_m
    valid_fraction = valid_count / total_count
    accepted = valid_fraction >= 0.80
    return {
        "accepted": accepted,
        "valid_pixels": valid_count,
        "total_pixels": total_count,
        "valid_fraction": valid_fraction,
        "median_m": median,
        "signed_error_m": error,
        "absolute_error_m": abs(error),
        "robust_sigma_m": robust_sigma,
        "within_tolerance": accepted and abs(error) <= measurement_tolerance_m(truth_m),
    }


def _summary(
    estimates: Sequence[float],
    truths: Sequence[float],
) -> dict[str, float | int | None]:
    if len(estimates) != len(truths):
        raise ValueError("Estimate and truth counts do not match")
    if not estimates:
        return {
            "samples": 0,
            "mae_m": None,
            "rmse_m": None,
            "max_abs_error_m": None,
            "abs_rel": None,
            "within_tolerance_rate": 0.0,
        }
    estimate = np.asarray(estimates, dtype=np.float64)
    truth = np.asarray(truths, dtype=np.float64)
    error = estimate - truth
    absolute = np.abs(error)
    tolerance = np.maximum(0.003, 0.01 * truth)
    return {
        "samples": len(estimates),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_m": float(absolute.max()),
        "abs_rel": float(np.mean(absolute / truth)),
        "within_tolerance_rate": float(np.mean(absolute <= tolerance)),
    }


def qualify_plane_captures(
    captures: Mapping[float, Sequence[np.ndarray]],
    *,
    depth_scale_to_m: float,
    calibration_frames_per_distance: int,
    roi_fraction: float = 0.5,
    validation_window_frames: int = 5,
    profile: CameraErrorProfile | None = None,
) -> dict[str, Any]:
    """Fit on the first frames at each distance and validate on untouched frames."""

    if calibration_frames_per_distance < 2:
        raise ValueError("At least two calibration frames per distance are required")
    if validation_window_frames < 1:
        raise ValueError("validation_window_frames must be positive")
    per_distance: dict[str, Any] = {}
    calibration_measured, calibration_truth = [], []
    validation: list[tuple[float, dict[str, float | int | bool]]] = []
    for truth_m in sorted(float(value) for value in captures):
        frames = captures[truth_m]
        if len(frames) <= calibration_frames_per_distance:
            raise ValueError(
                f"Distance {truth_m:g} m needs more than {calibration_frames_per_distance} frames"
            )
        results = [
            analyze_plane_frame(
                frame,
                truth_m,
                depth_scale_to_m=depth_scale_to_m,
                roi_fraction=roi_fraction,
            )
            for frame in frames
        ]
        tuning = [
            float(item["median_m"]) for item in results[:calibration_frames_per_distance] if item["accepted"]
        ]
        in_profile_range = profile is None or profile.min_range_m <= truth_m <= profile.max_range_m
        used_for_calibration = in_profile_range and len(tuning) >= 2
        if used_for_calibration:
            calibration_measured.append(float(np.median(tuning)))
            calibration_truth.append(truth_m)
        held_out = results[calibration_frames_per_distance:]
        validation.extend((truth_m, item) for item in held_out if item["accepted"])
        per_distance[f"{truth_m:g}"] = {
            "truth_m": truth_m,
            "in_profile_range": in_profile_range,
            "used_for_calibration": used_for_calibration,
            "captured_frames": len(frames),
            "calibration_accepted_frames": len(tuning),
            "validation_accepted_frames": sum(bool(item["accepted"]) for item in held_out),
            "mean_valid_fraction": float(np.mean([float(item["valid_fraction"]) for item in held_out])),
            "median_spatial_sigma_m": float(np.median([float(item["robust_sigma_m"]) for item in held_out])),
        }
    if len(calibration_measured) < 3:
        raise ValueError("At least three in-range distances must pass calibration")
    calibration = fit_output_calibration(calibration_measured, calibration_truth)
    raw_estimates, corrected_estimates, truths = [], [], []
    for truth_m, item in validation:
        measured = float(item["median_m"])
        raw_estimates.append(measured)
        corrected_estimates.append(calibration["scale"] * measured + calibration["offset_m"])
        truths.append(truth_m)

    temporal_estimates, temporal_truths = [], []
    for value in per_distance.values():
        truth_m = float(value["truth_m"])
        distance_validation = [
            float(item["median_m"]) for current_truth, item in validation if current_truth == truth_m
        ]
        corrected = [calibration["scale"] * item + calibration["offset_m"] for item in distance_validation]
        windowed = [
            float(np.median(distance_validation[start : start + validation_window_frames]))
            for start in range(
                0,
                len(distance_validation) - validation_window_frames + 1,
                validation_window_frames,
            )
        ]
        corrected_windowed = [calibration["scale"] * item + calibration["offset_m"] for item in windowed]
        temporal_estimates.extend(corrected_windowed)
        temporal_truths.extend([truth_m] * len(corrected_windowed))
        value["raw"] = _summary(distance_validation, [truth_m] * len(distance_validation))
        value["corrected"] = _summary(corrected, [truth_m] * len(corrected))
        value["corrected_temporal"] = _summary(
            corrected_windowed,
            [truth_m] * len(corrected_windowed),
        )

    in_range_indices = [
        index
        for index, truth_m in enumerate(truths)
        if profile is None or profile.min_range_m <= truth_m <= profile.max_range_m
    ]
    in_range_temporal_indices = [
        index
        for index, truth_m in enumerate(temporal_truths)
        if profile is None or profile.min_range_m <= truth_m <= profile.max_range_m
    ]
    return {
        "protocol": {
            "distances_m": sorted(float(value) for value in captures),
            "calibration_frames_per_distance": calibration_frames_per_distance,
            "validation_frames_per_distance": {
                key: int(value["validation_accepted_frames"]) for key, value in per_distance.items()
            },
            "roi_fraction": roi_fraction,
            "minimum_valid_fraction": 0.80,
            "validation_window_frames": validation_window_frames,
            "tolerance": "max(3 mm, 1% of reference distance)",
        },
        "calibration": calibration,
        "raw_all": _summary(raw_estimates, truths),
        "corrected_all": _summary(corrected_estimates, truths),
        "corrected_in_profile_range": _summary(
            [corrected_estimates[index] for index in in_range_indices],
            [truths[index] for index in in_range_indices],
        ),
        "corrected_temporal_in_profile_range": _summary(
            [temporal_estimates[index] for index in in_range_temporal_indices],
            [temporal_truths[index] for index in in_range_temporal_indices],
        ),
        "corrected_temporal_all": _summary(temporal_estimates, temporal_truths),
        "by_distance": per_distance,
    }


def simulate_plane_captures(
    profile: CameraErrorProfile,
    distances_m: Iterable[float],
    *,
    frames_per_distance: int,
    seed: int,
    shape: tuple[int, int] = (120, 160),
) -> dict[float, list[np.ndarray]]:
    """Create diffuse-plane frames with one fixed virtual camera unit."""

    if frames_per_distance < 1:
        raise ValueError("frames_per_distance must be positive")
    rng = np.random.default_rng(seed)
    site = _sample_site_error(profile, rng)
    captures: dict[float, list[np.ndarray]] = {}
    for raw_truth in distances_m:
        truth_m = float(raw_truth)
        frames = []
        outside = max(profile.min_range_m - truth_m, truth_m - profile.max_range_m, 0.0)
        outside_fraction = outside / max(profile.max_range_m - profile.min_range_m, 1e-6)
        invalid_probability = min(0.85, 0.005 + 1.5 * outside_fraction)
        for _ in range(frames_per_distance):
            center = _camera_measurement(truth_m, profile, site, rng)
            sigma = profile.random_sigma_m + profile.random_sigma_fraction * truth_m
            global_jitter = float(rng.normal(0.0, 0.15 * sigma))
            frame = center + global_jitter + rng.normal(0.0, sigma, size=shape)
            invalid = rng.random(shape) < invalid_probability
            gross = rng.random(shape) < 0.001
            frame[gross] += rng.normal(0.0, max(0.03, 0.03 * truth_m), size=int(gross.sum()))
            frame[invalid] = 0.0
            frames.append(frame.astype(np.float32))
        captures[truth_m] = frames
    return captures


def load_plane_capture_directory(root: str | Path) -> dict[float, list[np.ndarray]]:
    source = Path(root).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    captures: dict[float, list[np.ndarray]] = {}
    for distance_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        match = _DISTANCE_PATTERN.fullmatch(distance_dir.name.lower())
        if match is None:
            continue
        distance_m = float(match.group(1))
        frames = [np.load(path, allow_pickle=False) for path in sorted(distance_dir.glob("*/depth.npy"))]
        if frames:
            captures[distance_m] = frames
    if not captures:
        raise ValueError("No captures found. Expected <root>/<distance>m/<frame_id>/depth.npy")
    return captures


def save_qualification_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
