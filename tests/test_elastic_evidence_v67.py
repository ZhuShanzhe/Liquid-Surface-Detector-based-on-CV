"""Regression checks for the optional far-range elastic evidence window."""
import numpy as np

from liquid_depth.elastic_evidence_v67 import elastic_evidence_window


def test_asymmetric_outlier_tail_uses_local_evidence():
    rng = np.random.default_rng(67)
    values = np.r_[rng.normal(.8, .025, 520), rng.uniform(1.3, 2.5, 480)]
    result = elastic_evidence_window(
        values, camera_to_bottom_m=9., primary_height_m=1.4)
    assert result["applied"]
    assert abs(result["height_m"] - .8) < .05
    assert result["retained_fraction"] < .6


def test_unimodal_evidence_keeps_primary():
    rng = np.random.default_rng(68)
    values = rng.normal(.8, .025, 1000)
    result = elastic_evidence_window(
        values, camera_to_bottom_m=9., primary_height_m=.81)
    assert not result["applied"]
    assert result["height_m"] == .81


def test_dry_vote_takes_precedence_and_radius_scales_with_distance():
    rng = np.random.default_rng(69)
    values = np.r_[rng.normal(.8, .025, 520), rng.uniform(1.3, 2.5, 480)]
    blocked = elastic_evidence_window(
        values, camera_to_bottom_m=9., primary_height_m=0.,
        dry_layer_already_selected=True)
    near = elastic_evidence_window(
        values, camera_to_bottom_m=8., primary_height_m=1.4)
    far = elastic_evidence_window(
        values, camera_to_bottom_m=12., primary_height_m=1.4)
    assert not blocked["applied"]
    assert far["radius_m"] < near["radius_m"]
