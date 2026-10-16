from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from liquid_depth.illumination_scene_meter import IlluminationSceneMeter


class FakeReturnMeter:
    bundle: ClassVar[dict] = {"sensor_family": "structured_light"}
    dry_reference = object()
    ordinary_container_roi = np.ones((4, 5), dtype=bool)

    def measure(self, scene, **kwargs):
        assert scene == "ordinary_diffuse"
        assert kwargs["depth_m"].shape == (4, 5)
        return {"scene": scene, "route": "ordinary_shared_robust_returns_v2",
                "liquid_depth_m": -0.004, "numeric_output": True,
                "reasons": [], "diagnostics": {"value_source": "test"},
                "field_metrology_qualified": False}


def test_illumination_route_preserves_unclipped_depth_and_reports_rgb_quality():
    meter = IlluminationSceneMeter(FakeReturnMeter())
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[0] = 255
    result = meter.measure("glare", rgb_bgr=rgb,
                           depth_m=np.ones((4, 5), np.float32))

    assert result["scene"] == "glare"
    assert result["liquid_depth_m"] == -0.004
    assert result["software_route_status"] == "merged_illumination_v1_research"
    assert result["diagnostics"]["illumination_quality"][
        "saturated_fraction_any_channel_ge_250"] == pytest.approx(.25)
    assert result["diagnostics"]["rgb_used_for_depth_value"] is False

def test_severe_illumination_warns_without_suppressing_number():
    meter = IlluminationSceneMeter(FakeReturnMeter())
    depth = np.ones((4, 5), np.float32)
    dark = meter.measure("low_light", rgb_bgr=np.zeros((4, 5, 3), np.uint8),
                         depth_m=depth)
    saturated = meter.measure("glare", rgb_bgr=np.full((4, 5, 3), 255, np.uint8),
                              depth_m=depth)
    assert dark["liquid_depth_m"] == saturated["liquid_depth_m"] == -0.004
    assert dark["diagnostics"]["illumination_observability_flags"] == [
        "rgb_severely_underexposed"]
    assert saturated["diagnostics"]["illumination_observability_flags"] == [
        "rgb_severely_saturated"]


def test_illumination_route_requires_explicit_scene_and_aligned_rgb():
    meter = IlluminationSceneMeter(FakeReturnMeter())
    with pytest.raises(ValueError, match="explicit_glare"):
        meter.measure("ordinary_diffuse", rgb_bgr=np.zeros((4, 5, 3), np.uint8),
                      depth_m=np.ones((4, 5)))
    with pytest.raises(ValueError, match="aligned_uint8"):
        meter.measure("low_light", rgb_bgr=np.zeros((3, 5, 3), np.uint8),
                      depth_m=np.ones((4, 5)))


def test_illumination_route_requires_dry_geometry():
    with pytest.raises(ValueError, match="empty_geometry"):
        IlluminationSceneMeter(SimpleNamespace(bundle=None, dry_reference=None))
