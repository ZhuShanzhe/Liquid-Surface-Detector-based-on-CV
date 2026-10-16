"""Checks for offline formal partial-loss stratification."""
from summarize_partial_loss_formal_v62 import depth_band, loss_band, stats


def test_depth_bands_keep_empty_separate_from_shallow():
    assert depth_band(0) == "empty"
    assert depth_band(.02) == "0-0.1m"
    assert depth_band(.5) == "0.1-1m"
    assert depth_band(2.4) == "2-3m"
    assert depth_band(8.9) == "8-9m"


def test_loss_bands_use_measured_area_near_nominal_levels():
    assert loss_band(0) == "0-<5%"
    assert loss_band(9.995) == "5-<15%"
    assert loss_band(20.004) == "15-<25%"
    assert loss_band(49.999) == "45-50%"


def test_stats_count_all_frames_and_use_liquid_height_tolerance():
    rows = [
        {"truth_m": .1, "prediction": .119,
         "measured_loss_pct": 10.0},
        {"truth_m": .1, "prediction": .121,
         "measured_loss_pct": 20.0},
        {"truth_m": .1, "prediction": None,
         "measured_loss_pct": 30.0},
    ]
    result = stats(rows, "prediction")
    assert result["frames"] == 3
    assert result["numeric"] == 2
    assert result["all_frame_pass_pct"] == 100 / 3
    assert result["coverage_pct"] == 200 / 3
    assert abs(result["abs_rel_pct_mean"] - 20) < 1e-10
