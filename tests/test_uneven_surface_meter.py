"""Contract tests for the research two-height wrapper."""

import numpy as np
import pytest

from liquid_depth import uneven_surface_meter as module
from liquid_depth.liquid_height_metrics import HeightCalibration


class Base:
    def __init__(self):
        self.bundle = {"sensor_family": "tof"}
        self.dry_reference = object()
        self.last_depth = None

    def measure(self, scene, *, depth_m):
        assert scene == "ordinary_diffuse"
        assert depth_m.shape == (2, 2)
        self.last_depth = depth_m.copy()
        return {
            "liquid_depth_m": 1.0,
            "numeric_output": True,
            "diagnostics": {
                "value_source": "test_source",
                "quality_flags": [],
                "selected_pixels": 4,
            },
            "route": "test_route",
        }


def test_two_heights_and_order_flag(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (0.8, 20, "static_empty_roi"))
    result = module.UnevenSurfaceMeter(Base()).measure(
        np.ones((2, 2)), dry_paired_shallow_enabled=False,
        evidence_repair_enabled=False,
    )
    assert result["liquid_surface_height_m"] == 1.0
    assert result["highest_visible_height_m"] == 0.8
    assert "height_order_inconsistent" in result["quality_flags"]
    assert "few_highest_roi_returns" in result["quality_flags"]
    assert not result["field_metrology_qualified"]


def test_missing_peak_is_explicit(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (None, 0, "static_empty_roi"))
    result = module.UnevenSurfaceMeter(Base()).measure(
        np.ones((2, 2)), dry_paired_shallow_enabled=False,
        evidence_repair_enabled=False,
    )
    assert result["liquid_numeric"]
    assert not result["highest_numeric"]
    assert result["highest_visible_height_m"] is None
    assert "highest_support_unavailable" in result["quality_flags"]


def test_empty_reference_rejected():
    base = Base()
    base.dry_reference = None
    with pytest.raises(ValueError, match="empty_geometry"):
        module.UnevenSurfaceMeter(base)


def test_temporal_liquid_keeps_current_peak(monkeypatch):
    seen = []

    def peak(base, depth):
        seen.append(depth.copy())
        return 1.1, 20, "static_empty_roi"

    monkeypatch.setattr(module, "_top_envelope", peak)
    base = Base()
    current = np.ones((2, 2), dtype=float)
    history = [np.full((2, 2), 2.0), np.full((2, 2), 3.0)]
    result = module.UnevenSurfaceMeter(base).measure(
        current,
        recent_depth_frames=history,
        stationary_confirmed=True,
        dry_paired_shallow_enabled=False,
        evidence_repair_enabled=False,
    )
    np.testing.assert_allclose(base.last_depth, 2.0)
    np.testing.assert_allclose(seen[0], current)
    assert result["temporal_liquid_frames_used"] == 3


def test_temporal_requires_aligned_frames(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (1.0, 20, "static_empty_roi"))
    with pytest.raises(ValueError, match="aligned_recent"):
        module.UnevenSurfaceMeter(Base()).measure(
            np.ones((2, 2)),
            recent_depth_frames=[np.ones((3, 3))],
            stationary_confirmed=True,
            dry_paired_shallow_enabled=False,
        )


def test_paired_shallow_uses_bottom_pixels_without_mutation(monkeypatch):
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    k = np.array([[100.0, 0.0, 2.5], [0.0, 100.0, 2.5], [0.0, 0.0, 1.0]])
    cal = HeightCalibration(k, pose, 0.0, (1.0, 1.0))
    monkeypatch.setattr(module, "height_calibration", lambda bundle: cal)
    base = Base()
    base.ordinary_container_roi = np.ones((6, 6), dtype=bool)
    dry = np.ones((6, 6))
    wet = np.full((6, 6), 0.8)
    result = module._dry_paired_shallow(base, dry, wet, 0.25)
    assert result["applied"]
    assert result["height_m"] == pytest.approx(0.2)
    assert result["support_pixels"] == 36
    assert base.ordinary_container_roi.all()
    rejected = module._dry_paired_shallow(base, dry, wet, 0.6)
    assert not rejected["applied"]
    assert rejected["reason"] == "paired_candidate_disagrees_with_primary"


def test_paired_mode_requires_dry_image(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (1.0, 20, "static_empty_roi"))
    with pytest.raises(ValueError, match="dry_depth_required"):
        module.UnevenSurfaceMeter(Base()).measure(np.ones((2, 2)))


def test_matched_dry_defaults_to_paired_route(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (1.1, 30, "static_empty_roi"))
    monkeypatch.setattr(
        module,
        "_dry_paired_shallow",
        lambda base, dry, wet, preliminary: {"applied": True, "height_m": 0.2},
    )
    base = Base()
    base.dry_reference = type("DryReference", (), {"image_shape": (2, 2)})()
    meter = module.UnevenSurfaceMeter(base, np.ones((2, 2)))
    selected = meter.measure(np.ones((2, 2)), evidence_repair_enabled=False)
    assert selected["liquid_surface_height_m"] == 0.2
    assert selected["liquid_route"].endswith("+dry_paired_shallow_v1")
    legacy = meter.measure(
        np.ones((2, 2)), dry_paired_shallow_enabled=False,
        evidence_repair_enabled=False,
    )
    assert legacy["liquid_surface_height_m"] == 1.0
    assert legacy["dry_paired_shallow"] is None


def test_merged_route_uses_evidence_repair_by_default(monkeypatch):
    monkeypatch.setattr(module, "_top_envelope", lambda base, depth: (1.1, 30, "static_empty_roi"))
    monkeypatch.setattr(
        module,
        "repair_measurement",
        lambda base, dry, depth, liquid, source, peak: (
            0.9,
            1.2,
            {
                "peak_reason": "supported_mean_resolves_low_echo_layer",
                "peak_selected_returns": 300,
                "peak_sparse_fallback": False,
                "peak_wide_fallback": False,
                "echo_layer_conflict": False,
                "paired_tile_se_m": 0.005,
                "liquid_reason": "current_preserved",
            },
        ),
    )
    base = Base()
    base.dry_reference = type("DryReference", (), {"image_shape": (2, 2)})()
    result = module.UnevenSurfaceMeter(base, np.ones((2, 2))).measure(
        np.ones((2, 2)), dry_paired_shallow_enabled=False
    )
    assert result["liquid_surface_height_m"] == 0.9
    assert result["highest_visible_height_m"] == 1.2
    assert result["liquid_route"].endswith("+uneven_echo_evidence_v3")
    assert result["software_route_status"] == "merged_uneven_v11_research"
    assert not result["field_metrology_qualified"]
