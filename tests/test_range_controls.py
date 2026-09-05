from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from test_surface_memory import fixture

from liquid_depth.range_calibration import RANGE_EDGES, SCORE_EDGES, RangeNoiseCalibration
from liquid_depth.surface_memory import MetricSurfaceMemory
from liquid_depth.verified_tracking import VerifiedSurfaceTracker


def profile():
    shape = (len(RANGE_EDGES) - 1, len(SCORE_EDGES) - 1)
    return {
        "schema_version": 1,
        "range_edges": RANGE_EDGES,
        "score_edges": SCORE_EDGES,
        "sensors": {
            "tof": {
                "sigma_coefficients": [0.001, 0.03],
                "probabilities": np.full(shape, 0.99).tolist(),
                "wilson_lower": np.full(shape, 0.97).tolist(),
                "counts": np.full(shape, 1000).tolist(),
            }
        },
    }


def cue(level=0.3, bound=0.002):
    return {
        "available": True,
        "level_m": level,
        "uncertainty_proxy_m": bound / 2,
        "depth_input_used": False,
        "resolution_checked": True,
        "error_bound_proxy_m": bound,
    }


def test_calibrated_low_model_scores_can_supply_spatial_support():
    args = list(fixture())
    args[2] = {**args[2], "confidence": np.full_like(args[1], 0.25)}
    assert not MetricSurfaceMemory().estimate(*args)["accepted"]
    result = MetricSurfaceMemory(range_calibration=RangeNoiseCalibration(profile(), "tof")).estimate(*args)
    assert result["accepted"]
    assert result["calibrated_confidence_used"]
    assert json.loads(json.dumps(result))["range_calibration_available"] is True


def test_no_extrapolation_or_empty_bin_acceptance():
    p = profile()
    p["sensors"]["tof"]["counts"] = np.zeros_like(p["sensors"]["tof"]["counts"]).tolist()
    policy = RangeNoiseCalibration(p, "tof")
    raw = np.ones((8, 8))
    selected, _ = policy.select(raw, np.ones_like(raw, bool), np.ones_like(raw))
    assert not selected.any()
    selected, diag = policy.select(raw * 20, np.ones_like(raw, bool), np.ones_like(raw))
    assert not selected.any() and not diag["range_calibration_available"]


def test_distance_noise_gate_does_not_only_loosen_near_range():
    policy = RangeNoiseCalibration(profile(), "tof")

    def gate(distance):
        raw = np.full((8, 8), distance)
        return policy.select(raw, np.ones_like(raw, bool), np.ones_like(raw))[1]["plane_gate_m"]

    assert gate(0.1) == 0.004
    assert gate(3) > gate(1) > gate(0.1)
    assert gate(10) <= 0.25


def test_noise_scaled_plane_acceptance_has_unchanged_tilt_constraint():
    args = list(fixture())
    rng = np.random.default_rng(42)
    raw = args[1].copy()
    raw[raw > 0] += rng.normal(0, 0.03, size=(raw > 0).sum())
    args[1] = raw
    assert not MetricSurfaceMemory().estimate(*args)["accepted"]
    result = MetricSurfaceMemory(range_calibration=RangeNoiseCalibration(profile(), "tof")).estimate(*args)
    assert result["accepted"]
    assert result["normalized_plane_residual"] < 1


def test_invalid_profile_is_rejected():
    p = profile()
    p["sensors"]["tof"]["sigma_coefficients"][0] = float("nan")
    with pytest.raises(ValueError):
        RangeNoiseCalibration(p, "tof")
    with pytest.raises(ValueError):
        RangeNoiseCalibration(profile(), "unknown")


def test_strict_rgb_uncertainty_cannot_widen_acceptance():
    tracker = VerifiedSurfaceTracker(strict_rgb=True)
    for bound in (0.006, 0.01, 0.1):
        out = tracker.process(*fixture(), witness=cue(bound=bound))
        assert out["reasons"] == ["independent_rgb_resolution_insufficient"]
    for i in range(5):
        out = tracker.process(*fixture(), witness=cue())
        assert out["accepted"] == (i == 4)
    trusted = tracker.metric.last_level_m
    args = list(fixture(0.294))
    out = tracker.process(*args, witness=cue())
    assert not out["accepted"]
    assert tracker.metric.last_level_m == trusted


def test_strict_reacquisition_and_severe_failure_policy():
    tracker = VerifiedSurfaceTracker(strict_rgb=True)
    for _ in range(5):
        assert tracker.process(*fixture(), witness=cue()) is not None
    args = list(fixture())
    args[1] = np.zeros_like(args[1])
    for _ in range(600):
        assert tracker.process(*args, witness=cue())["reasons"] == [
            "unsupported_95_100_percent_depth_failure"
        ]
    for i in range(5):
        out = tracker.process(*fixture(0.34), witness=cue(0.34))
        assert out["accepted"] == (i == 4)


def test_strict_requires_actual_resolution_check():
    value = copy.deepcopy(cue())
    value.pop("resolution_checked")
    assert VerifiedSurfaceTracker(strict_rgb=True).process(*fixture(), witness=value)["reasons"] == [
        "independent_rgb_resolution_unverified"
    ]
