from pathlib import Path

import numpy as np
import pytest

from liquid_depth.accuracy import (
    allowed_error_m,
    evaluate_accuracy_profile,
    find_accuracy_band,
    load_accuracy_profile,
)

PROFILE_PATH = Path(__file__).parents[1] / "configs" / "accuracy_profile_industrial_v1.yaml"


def test_profile_loads_and_boundaries_are_unambiguous():
    profile = load_accuracy_profile(PROFILE_PATH)

    assert find_accuracy_band(profile, 0.019) is None
    assert find_accuracy_band(profile, 0.02)["name"] == "near_zero_stress"
    assert find_accuracy_band(profile, 0.20)["name"] == "close"
    assert find_accuracy_band(profile, 1.00)["name"] == "medium"
    assert find_accuracy_band(profile, 5.00)["name"] == "long_range"
    assert find_accuracy_band(profile, 10.00)["name"] == "long_range"
    assert find_accuracy_band(profile, 10.01) is None


def test_allowed_error_uses_absolute_floor_and_relative_term():
    specification = {"absolute_floor_mm": 2.0, "relative_percent": 0.5}
    tolerance = allowed_error_m(np.asarray([0.2, 1.0, 2.0]), specification)

    assert tolerance == pytest.approx([0.002, 0.005, 0.010])


def test_profile_evaluation_passes_target_tolerances():
    profile = load_accuracy_profile(PROFILE_PATH)
    truth = np.asarray([0.2, 0.5, 1.0, 2.0, 5.0, 8.0])
    prediction = truth + np.asarray([0.001, -0.002, 0.005, -0.010, 0.025, -0.040])

    report = evaluate_accuracy_profile(truth, prediction, profile, level="target")

    assert report["coverage"] == 1.0
    assert report["outside_profile_samples"] == 0
    assert report["passes_profile"]
    assert report["bands"]["medium"]["passes_error_gate"]
    assert report["bands"]["long_range"]["passes_error_gate"]


def test_profile_evaluation_fails_large_errors_and_low_coverage():
    profile = load_accuracy_profile(PROFILE_PATH)
    truth = np.asarray([0.5, 2.0, 8.0])
    prediction = np.asarray([0.55, 2.2, 9.0])
    confidence = np.asarray([1.0, 0.1, 1.0])

    report = evaluate_accuracy_profile(
        truth,
        prediction,
        profile,
        confidence=confidence,
        confidence_threshold=0.5,
        level="deployment",
    )

    assert report["coverage"] == pytest.approx(2 / 3)
    assert not report["passes_coverage_gate"]
    assert not report["passes_profile"]
    assert not report["bands"]["close"]["passes_error_gate"]


def test_out_of_scope_truth_prevents_certification():
    profile = load_accuracy_profile(PROFILE_PATH)
    report = evaluate_accuracy_profile([0.01, 11.0], [0.01, 11.0], profile)

    assert report["outside_profile_samples"] == 2
    assert not report["passes_profile"]
