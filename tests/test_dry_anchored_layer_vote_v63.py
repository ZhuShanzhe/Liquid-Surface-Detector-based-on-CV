"""Regression checks for the dry-anchored return-layer selector."""
import numpy as np

from liquid_depth.dry_anchored_layer_vote_v63 import vote_dry_anchored_layer


def test_bottom_dominance_can_select_supported_wet_layer():
    rng = np.random.default_rng(63)
    values = np.r_[rng.normal(0, .025, 2400),
                   rng.normal(1.4, .045, 900)]
    result = vote_dry_anchored_layer(
        values, rim_height_m=1.6, primary_height_m=.1,
        container_quantile_height_m=1.42)
    assert result["selected"]
    assert result["candidate"] == "wet_surface_peak"
    assert abs(result["height_m"] - 1.42) < .01


def test_no_plausible_wet_layer_uses_observed_bottom_peak_without_snap():
    rng = np.random.default_rng(64)
    values = rng.normal(.04, .025, 2400)
    result = vote_dry_anchored_layer(
        values, rim_height_m=.16, primary_height_m=-1.2,
        container_quantile_height_m=.5)
    assert result["selected"]
    assert result["candidate"] == "dry_anchored_near_zero_peak"
    assert abs(result["height_m"] - .04) < .01


def test_weak_dry_layer_keeps_original_fit():
    rng = np.random.default_rng(65)
    values = np.r_[rng.normal(0, .03, 120),
                   rng.normal(1.0, .08, 1400)]
    result = vote_dry_anchored_layer(
        values, rim_height_m=1.2, primary_height_m=1.01,
        container_quantile_height_m=1.0)
    assert not result["selected"]
    assert result["height_m"] == 1.01
