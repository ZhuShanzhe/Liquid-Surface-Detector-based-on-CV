"""Research candidates never override the trusted-output API."""

import numpy as np
import pytest
from test_rgb_witness import reference
from test_surface_memory import fixture
from test_surface_video_range_runtime import system

from liquid_depth.super_resolution import scaled_intrinsics
from liquid_depth.surface_candidates import SurfaceCandidateEstimator, area_statistics


def footprint():
    x, y = np.meshgrid(np.linspace(-0.25, 0.25, 21), np.linspace(-0.18, 0.18, 21))
    return np.column_stack((x.ravel(), y.ravel()))


def run(engine, args=None, **kwargs):
    return engine.estimate(*(args or fixture()), footprint(), (0.25, 0.18), **kwargs)


def test_relaxed_outputs_remain_unverified():
    out = run(SurfaceCandidateEstimator(mode="free"))
    assert out["candidate_available"] and not out["accepted"]
    assert out["requires_independent_verification"]
    assert abs(out["statistics"]["mean_depth_m"] - 0.3) < 1e-6


def test_tilt_is_a_diagnostic_not_a_candidate_rejection():
    args = list(fixture())
    x = np.indices(args[1].shape)[1]
    args[1] = np.where(args[2]["mask"], 1 / (1 + 0.5 * (x - 60) / 120), 0)
    out = run(SurfaceCandidateEstimator(mode="free"), args)
    assert out["candidate_available"]
    assert "tilt_over_12" in out["quality_flags"]
    assert not out["accepted"]


def test_early_history_is_used_but_height_tracks_fresh_observations():
    engine = SurfaceCandidateEstimator()
    run(engine)
    out = run(engine, fixture(0.32))
    assert out["early_history_used"]
    assert abs(out["level_m"] - 0.32) < 1e-5


def test_no_stale_height_with_complete_or_95_percent_failure():
    engine = SurfaceCandidateEstimator()
    run(engine)
    for count in (0, 100, 240):  # mask has 4800 pixels
        args = list(fixture())
        raw = np.zeros_like(args[1])
        coords = np.argwhere(args[2]["mask"])[:count]
        raw[coords[:, 0], coords[:, 1]] = 1
        args[1] = raw
        out = run(engine, args)
        assert not out["candidate_available"]
        assert out["level_m"] is None


def test_invalid_pose_clears_early_memory():
    engine = SurfaceCandidateEstimator()
    run(engine)
    assert engine.history
    assert not run(engine, pose_valid=False)["candidate_available"]
    assert not engine.history


def test_history_expires_without_refresh():
    engine = SurfaceCandidateEstimator(history_age=2)
    run(engine)
    for _ in range(3):
        args = list(fixture())
        args[1] = np.zeros_like(args[1])
        run(engine, args)
    assert not engine.history


def test_wave_statistics_describe_a_nonplanar_surface():
    args = list(fixture())
    x = (np.indices(args[1].shape)[1] - 60) / 120
    # World z = 0.2 X^2; ray X=x*(1-z); solve positive ray depth.
    a = 0.2 * x * x
    depth = 2 / (1 + np.sqrt(1 + 4 * a))
    args[1] = np.where(args[2]["mask"], depth, 0)
    out = run(SurfaceCandidateEstimator(surface_mode="waves"), args)
    truth = area_statistics(0.3 + 0.2 * footprint()[:, 0] ** 2)
    assert out["candidate_available"] and not out["accepted"]
    for key in ("min_depth_m", "mean_depth_m", "max_depth_m"):
        assert abs(out["statistics"][key] - truth[key]) < 0.001
    assert out["statistics"]["peak_to_peak_m"] > 0.01


def test_invalid_footprint_and_configuration():
    with pytest.raises(ValueError):
        SurfaceCandidateEstimator(max_points=12)
    for area, radii in (([], (1, 1)), ([[0, np.nan]], (1, 1)), ([[0, 0]], (1, 0))):
        with pytest.raises(ValueError):
            SurfaceCandidateEstimator().estimate(*fixture(), area, radii)
    out = SurfaceCandidateEstimator().estimate(*fixture(), footprint().tolist(), (0.25, 0.18))
    assert out["candidate_available"]


def test_pixel_center_intrinsics_roundtrip():
    k = fixture()[3]
    np.testing.assert_allclose(scaled_intrinsics(scaled_intrinsics(k, 4), 0.25), k)
    assert scaled_intrinsics(k, 4)[0, 2] == (k[0, 2] + 0.5) * 4 - 0.5
    with pytest.raises(ValueError):
        scaled_intrinsics(k, 0)


def test_sr_cannot_silently_shrink_source_pixel_ambiguity():
    witness, rgb, k, pose = reference()
    naive = witness.estimate(rgb, k, pose, resolution_checks=True)
    guarded = witness.estimate(rgb, k, pose, resolution_checks=True, source_pixel_scale=4)
    assert guarded["error_bound_proxy_m"] >= naive["error_bound_proxy_m"]
    with pytest.raises(ValueError):
        witness.estimate(rgb, k, pose, source_pixel_scale=0)


def test_runtime_candidate_route_and_reference_reset():
    s = system()
    rgb, raw, _, k, pose, bottom = fixture()
    out = s.process_surface_candidates(rgb, raw, k, pose, bottom, area_xy=footprint(), radii=(0.25, 0.18))
    assert out["candidate_available"] and not out["accepted"]
    assert out["route"] == "experimental_unverified_surface_candidate"
    assert s._surface_candidate_engines
    s.reset_reference()
    assert not s._surface_candidate_engines


def test_runtime_propagates_source_pixel_scale():
    s = system()
    rgb, raw, _, k, pose, bottom = fixture()
    frame = {
        "rgb_bgr": rgb,
        "intrinsics": k,
        "camera_to_world_cv": pose,
        "synchronized": True,
        "source_pixel_scale": 4,
    }
    s.process(rgb, raw, k, pose, bottom, witness_frame=frame)
    assert s.rgb_witness.source_pixel_scale == 4
