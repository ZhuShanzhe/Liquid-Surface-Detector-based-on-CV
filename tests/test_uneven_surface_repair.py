"""Regression checks for echo-layer conflicts and supported visible peaks."""

from types import SimpleNamespace

import numpy as np
import pytest

from liquid_depth import uneven_surface_repair as module


def evidence(candidate, quantile, se, tiles=5):
    pose = np.eye(4)
    pose[2, 3] = 9.6
    return {
        "cal": SimpleNamespace(camera_to_world_cv=pose, bottom_z_m=0.0),
        "paired_height_m": candidate,
        "raw_quantile_m": quantile,
        "paired_tile_se_m": se,
        "paired_tiles": tiles,
        "paired_support": 76,
    }


def test_deep_bottom_vote_conflict_restores_observed_wet_quantile():
    ev = evidence(5.017, 5.202, 0.167)
    value, reason = module.repaired_liquid(
        ev, 0.018, "dry_anchored_layer_vote_v64:dry_anchored_near_zero_peak"
    )
    assert value == pytest.approx(5.202)
    assert reason == "dry_layer_contradicted_by_two_wet_candidates"


def test_dry_noise_is_not_promoted_to_wet_layer():
    ev = evidence(0.648, 0.699, 0.067, tiles=28)
    value, reason = module.repaired_liquid(
        ev, 0.007, "dry_anchored_layer_vote_v64:dry_anchored_near_zero_peak"
    )
    assert value == pytest.approx(0.007)
    assert reason == "current_preserved"


def test_spatial_shallow_evidence_can_resolve_fixed_gate_boundary():
    ev = evidence(0.028, 0.233, 0.01, tiles=39)
    value, reason = module.repaired_liquid(ev, 0.233, "container_return_quantile")
    assert value == pytest.approx(0.028)
    assert reason == "spatially_consistent_shallow_override"


def test_missing_evidence_does_not_snap_negative_height():
    ev = evidence(None, None, None)
    value, _ = module.repaired_liquid(ev, -0.02, "container_return_quantile")
    assert value == -0.02


def test_peak_rejects_isolated_echo_and_keeps_coherent_float(monkeypatch):
    ev = evidence(None, None, None)
    height = np.full((40, 40), 0.5)
    height[15:24, 15:24] = 0.56
    height[4, 4] = 1.1
    ev.update(height=height, valid=np.ones(height.shape, dtype=bool))
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, diag = module.repair_measurement(None, None, None, 0.5, "", 1.1)
    assert peak == pytest.approx(0.56)
    assert diag["peak_reason"] == "supported_mean_suppresses_isolated_high_echo"
    _, preserved, _ = module.repair_measurement(None, None, None, 0.5, "", 0.56)
    assert preserved == 0.56


def test_sparse_peak_uses_supported_neighbors(monkeypatch):
    ev = evidence(None, None, None)
    height = np.full((40, 40), 0.5)
    valid = np.zeros(height.shape, dtype=bool)
    valid[::2, ::2] = True
    ev.update(height=height, valid=valid)
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, diag = module.repair_measurement(None, None, None, 0.5, "", None)
    assert peak == pytest.approx(0.5)
    assert diag["peak_reason"] == "supported_mean_rescues_missing_peak"


def test_unsupported_peak_is_still_explicitly_missing(monkeypatch):
    ev = evidence(None, None, None)
    height = np.full((40, 40), 0.5)
    valid = np.zeros(height.shape, dtype=bool)
    valid[0, :2] = True
    ev.update(height=height, valid=valid)
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, _ = module.repair_measurement(None, None, None, 0.5, "", None)
    assert peak is None


def test_sparse_wider_support_recovers_wrong_low_layer(monkeypatch):
    ev = evidence(None, None, None)
    height = np.full((40, 40), 0.5)
    valid = np.zeros(height.shape, dtype=bool)
    valid[10:21:5, 10:21:5] = True
    ev.update(height=height, valid=valid)
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, diag = module.repair_measurement(None, None, None, 0.5, "", 0.05)
    assert peak == pytest.approx(0.5)
    assert diag["peak_sparse_fallback"]

def test_near_bottom_peak_uses_measured_dry_pair_support(monkeypatch):
    ev = evidence(0.003, 0.003, 0.001)
    ev["paired_support"] = 1600
    height = np.full((40, 40), 0.003)
    height[4, 4] = 0.10
    ev.update(
        height=height,
        valid=np.ones(height.shape, dtype=bool),
        paired_mask=np.ones(height.shape, dtype=bool),
        paired_diff=height,
    )
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    monkeypatch.setattr(module, "peak_candidates", lambda *args, **kwargs: {"local9_max": 0.07, "selected_count": 1600})
    _, peak, diag = module.repair_measurement(None, None, None, 0.003, "", 0.07)
    assert 0.003 <= peak < 0.010
    assert diag["peak_reason"] == "paired_near_bottom_supported_peak"


def test_wide_search_recovers_separate_supported_layer(monkeypatch):
    ev = evidence(2.35, 2.59, 0.10)
    height = np.full((40, 40), 2.41)
    ev.update(height=height, valid=np.ones(height.shape, dtype=bool))
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, diag = module.repair_measurement(None, None, None, 2.59, "", 1.36)
    assert peak == pytest.approx(2.41)
    assert diag["peak_wide_fallback"]


def test_common_mode_peak_offset_requires_consistent_dry_pair(monkeypatch):
    ev = evidence(1.15, 1.22, 0.01, tiles=12)
    ev["paired_support"] = 500
    height = np.full((40, 40), 1.32)
    ev.update(height=height, valid=np.ones(height.shape, dtype=bool))
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    _, peak, diag = module.repair_measurement(None, None, None, 1.22, "", 1.32)
    assert peak == pytest.approx(1.32 - 0.07 / (1 + (0.01 / 0.03) ** 2))
    assert diag["peak_common_mode_offset_m"] == pytest.approx(-0.063)


def test_close_range_coherent_shallow_float_survives_empty_noise_correction(monkeypatch):
    ev = evidence(0.015, 0.015, 0.001)
    ev["cal"].camera_to_world_cv[2, 3] = 0.6
    ev["paired_support"] = 1600
    height = np.full((40, 40), 0.019)
    ev.update(
        height=height,
        valid=np.ones(height.shape, dtype=bool),
        paired_mask=np.ones(height.shape, dtype=bool),
        paired_diff=height,
    )
    monkeypatch.setattr(module, "prepare_evidence", lambda *args: ev)
    monkeypatch.setattr(module, "peak_candidates", lambda *args, **kwargs: {"local9_max": 0.054, "selected_count": 1600})
    _, peak, diag = module.repair_measurement(None, None, None, 0.015, "", 0.068)
    assert peak == pytest.approx(0.054)
    assert diag["peak_reason"] == "near_bottom_coherent_peak_preserved"

