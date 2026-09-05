from __future__ import annotations

import numpy as np
import pytest
from test_surface_memory import fixture

from liquid_depth.surface_memory import MetricSurfaceMemory
from liquid_depth.surface_video_runtime import UniversalSurfaceVideoSystem
from liquid_depth.verified_tracking import VerifiedSurfaceTracker


class Predictor:
    def predict(self, rgb, raw):
        self.shape = rgb.shape
        return fixture()[2]


class Witness:
    def estimate(self, rgb, k, pose, *, resolution_checks=False):
        self.shape, self.k, self.pose = rgb.shape, k, pose
        return {
            "available": True,
            "level_m": 0.3,
            "uncertainty_proxy_m": 0.001,
            "depth_input_used": False,
            "resolution_checked": resolution_checks,
            "error_bound_proxy_m": 0.002,
        }


def system():
    result = UniversalSurfaceVideoSystem.__new__(UniversalSurfaceVideoSystem)
    result.predictor = Predictor()
    result.rgb_witness = Witness()
    result.strict_rgb = True
    result.memory_options = {}
    result.memory = MetricSurfaceMemory()
    result.verified_tracker = VerifiedSurfaceTracker(strict_rgb=True)
    return result


def test_high_resolution_cue_does_not_resize_model_or_depth_input():
    s = system()
    rgb, raw, _, k, pose, bottom = fixture()
    big = np.repeat(np.repeat(rgb, 4, axis=0), 4, axis=1)
    big_k = k.copy()
    big_k[:2] *= 4
    frame = {"rgb_bgr": big, "intrinsics": big_k, "camera_to_world_cv": pose, "synchronized": True}
    for i in range(5):
        out = s.process(rgb, raw, k, pose, bottom, witness_frame=frame)
        assert out["accepted"] == (i == 4)
    assert s.predictor.shape == rgb.shape
    assert s.rgb_witness.shape == big.shape
    np.testing.assert_allclose(s.rgb_witness.k, big_k)
    s.reset_reference()
    assert s.verified_tracker.strict_rgb
    assert not s.process(rgb, raw, k, pose, bottom, witness_frame=frame)["accepted"]


def test_unsynchronized_independent_frame_and_removing_strict_cue_fail():
    s = system()
    rgb, raw, _, k, pose, bottom = fixture()
    with pytest.raises(ValueError, match="synchronization"):
        s.process(rgb, raw, k, pose, bottom, witness_frame={"rgb_bgr": rgb})
    with pytest.raises(ValueError):
        s.set_rgb_witness(None)
