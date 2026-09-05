import numpy as np
import pytest
from test_rgb_witness import reference
from test_surface_candidates import footprint
from test_surface_memory import fixture
from test_surface_video_range_runtime import system

from liquid_depth.rgb_continuous import RGBContinuousWitness
from liquid_depth.rgb_witness import RGBContourWitness
from liquid_depth.simulation import render_geometric_labels, sample_scene, simulate_raw_depth
from liquid_depth.surface_refinement import RefinedSurfaceEstimator, StereoNoiseModel, censored_clearance


@pytest.mark.parametrize("mode", ["balanced", "sensor", "partition"])
def test_refinement_is_unverified_and_tracks_fresh_height(mode):
    engine = RefinedSurfaceEstimator(mode=mode)
    out = engine.estimate(*fixture(0.32), footprint(), (0.25, 0.18))
    assert out["candidate_available"] and not out["accepted"]
    assert abs(out["level_m"] - 0.32) < 1e-5


@pytest.mark.parametrize("mode", ["balanced", "sensor", "partition"])
def test_refinement_never_fills_complete_depth_failure(mode):
    args = list(fixture())
    args[1] = np.zeros_like(args[1])
    out = RefinedSurfaceEstimator(mode=mode).estimate(*args, footprint(), (0.25, 0.18))
    assert not out["candidate_available"] and out["level_m"] is None


def test_conditional_intervals_are_explicit_and_contain_flat_surface():
    engine = RefinedSurfaceEstimator(mode="partition", max_surface_slope=0.5)
    out = engine.estimate(*fixture(), footprint(), (0.25, 0.18))
    assert out["statistics_intervals"]
    assert not out["interval_is_statistically_calibrated"]
    for key in ("mean_depth_m", "min_depth_m", "max_depth_m"):
        lo, hi = out["statistics_intervals"][key]
        assert lo <= 0.3 <= hi
    assert 0 < out["observed_area_fraction"] <= 1


def test_no_global_interval_without_slope_prior():
    out = RefinedSurfaceEstimator(mode="partition").estimate(*fixture(), footprint(), (0.25, 0.18))
    assert out["statistics_intervals"] is None
    assert "global_bounds_require_validated_slope_prior" in out["quality_flags"]


def test_truncated_disparity_likelihood_reduces_boundary_bias():
    model = StereoNoiseModel.simulation_proxy("active_stereo")
    rng = np.random.default_rng(11851)
    fx, true = 120.0, 10.0
    disparity = rng.normal(fx * model.baseline_m / true, model.disparity_sigma_px, 30000)
    depth = fx * model.baseline_m / np.maximum(disparity, 1e-6)
    depth = depth[(depth >= model.min_depth_m) & (depth <= model.max_depth_m)]
    fitted = censored_clearance(depth, np.ones(len(depth)), np.ones(len(depth)) / len(depth), fx, model)
    assert abs(fitted - true) < 0.15
    assert abs(fitted - true) < 0.25 * abs(np.median(depth) - true)


def test_tof_cannot_request_stereo_proxy():
    with pytest.raises(ValueError):
        StereoNoiseModel.simulation_proxy("tof")
    with pytest.raises(ValueError):
        StereoNoiseModel(-1, 0.2, 0.001, 0.001)


def test_sensor_switch_defaults_are_bitwise_compatible():
    scene = sample_scene(3, seed=11953, width=96, height=54)
    labels = render_geometric_labels(scene)
    a = simulate_raw_depth(scene, labels)
    b = simulate_raw_depth(
        scene,
        labels,
        components={k: True for k in ("depth_noise", "disparity_noise", "quantization", "range_cutoff")},
    )
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    with pytest.raises(ValueError):
        simulate_raw_depth(scene, labels, components={"truth_correction": True})


def continuous_reference():
    old, rgb, k, pose = reference()
    mask = old._project_mask(0.3, k, pose, rgb.shape[:2])
    witness = RGBContinuousWitness()
    witness.calibrate(rgb, mask, 0.3, k, pose, -0.3, 0.25, 0.18)
    return witness, rgb, k, pose


def test_continuous_calibration_roundtrip_requires_matching_fitter():
    witness, rgb, k, pose = continuous_reference()
    result = witness.estimate(rgb, k, pose, resolution_checks=True)
    assert result["available"] and abs(result["level_m"] - 0.3) < 1e-6
    restored = RGBContinuousWitness.from_dict(witness.to_dict())
    again = restored.estimate(rgb, k, pose, resolution_checks=True)
    for key, value in result.items():
        if isinstance(value, float):
            assert again[key] == pytest.approx(value, abs=1e-8)
        else:
            assert again[key] == value
    with pytest.raises(ValueError):
        RGBContourWitness.from_dict(witness.to_dict())


def test_continuous_darkness_and_sr_budget():
    witness, rgb, k, pose = continuous_reference()
    assert not witness.estimate(np.zeros_like(rgb), k, pose)["available"]
    a = witness.estimate(rgb, k, pose, resolution_checks=True)
    b = witness.estimate(rgb, k, pose, resolution_checks=True, source_pixel_scale=4)
    assert b["error_bound_proxy_m"] >= a["error_bound_proxy_m"]


def test_runtime_refinement_does_not_change_trusted_policy():
    s = system()
    rgb, raw, _, k, pose, bottom = fixture()
    out = s.process_refined_surface(rgb, raw, k, pose, bottom, area_xy=footprint(), radii=(0.25, 0.18))
    assert out["candidate_available"] and not out["accepted"]
    assert not s.process(rgb, raw, k, pose, bottom)["accepted"]


@pytest.mark.parametrize("distance", [0.2, 0.5, 1.0, 3.0])
def test_sensor_fit_does_not_skip_narrow_inlier_basin(distance):
    model = StereoNoiseModel.simulation_proxy("active_stereo")
    depth = np.full(256, distance)
    result = censored_clearance(depth, np.ones(256), np.ones(256) / 256, 120, model)
    assert result == pytest.approx(distance, abs=0.001)


def test_invalid_pose_and_invalid_geometry_have_no_refined_output():
    out = RefinedSurfaceEstimator().estimate(*fixture(), footprint(), (0.25, 0.18), pose_valid=False)
    assert not out["candidate_available"]
    with pytest.raises(ValueError):
        RefinedSurfaceEstimator().estimate(*fixture(), footprint(), (0, 1))
