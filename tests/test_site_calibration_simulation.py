from __future__ import annotations

import numpy as np

from liquid_depth.site_calibration_simulation import (
    CAMERA_ERROR_PROFILES,
    build_empirical_pools,
    calibration_levels,
    simulate_site_calibration,
)


def _records(relative_error: float = 0.0) -> list[dict]:
    values = []
    for scenario in ("ordinary", "transparent"):
        for index in range(20):
            values.append(
                {
                    "sample_id": f"{scenario}-{index:03d}",
                    "scenario": scenario,
                    "accepted": True,
                    "relative_error": relative_error,
                    "error_m": relative_error,
                }
            )
    return values


def test_market_camera_profiles_cover_stereo_and_tof() -> None:
    assert set(CAMERA_ERROR_PROFILES) == {
        "stereo_2pct",
        "long_baseline_stereo",
        "tof_typical",
    }
    assert CAMERA_ERROR_PROFILES["stereo_2pct"].max_range_m == 10.0
    assert CAMERA_ERROR_PROFILES["tof_typical"].random_sigma_m == 0.017


def test_empirical_pool_split_does_not_reuse_tuning_half() -> None:
    records = _records(0.02)
    pools = build_empirical_pools(records)
    assert pools["ordinary"]["tuning_samples"] == 10
    assert pools["ordinary"]["assessment_samples"] == 10
    assert pools["ordinary"]["tuning_absolute_bias_m"] == 0.02
    assert pools["ordinary"]["assessment_acceptance"] == 1.0


def test_hybrid_calibration_levels_include_range_endpoints() -> None:
    profile = CAMERA_ERROR_PROFILES["stereo_2pct"]
    levels = calibration_levels(profile, 7, "hybrid")
    assert len(levels) == 7
    assert levels[0] == profile.min_range_m
    assert levels[-1] == profile.max_range_m
    assert np.all(np.diff(levels) > 0)


def test_repeated_site_calibration_corrects_fixed_camera_error() -> None:
    result = simulate_site_calibration(
        _records(),
        CAMERA_ERROR_PROFILES["long_baseline_stereo"],
        trials=40,
        seed=17,
        calibration_level_count=7,
        calibration_frames=5,
        validation_frames=5,
        minimum_accepted_frames=2,
        design="hybrid",
    )
    assert result["site_success_rate"] == 1.0
    assert result["global"]["abs_rel"] < 0.01
    assert result["global"]["within_tolerance_rate"] > 0.75

