import numpy as np
import pytest

from liquid_depth.illumination_scene_meter import IlluminationSceneMeter
from liquid_depth.illumination_surface_evidence import dry_changed_evidence


def installation():
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 2.
    return {'source': 'empty_rgbd_plus_operator_metric_anchor',
            'sensor_family': 'tof', 'image_size': [80, 60], 'rim_z_m': 1.,
            'baseline_m': 0., 'calibration': {
                'intrinsics': [[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]],
                'camera_to_world_cv': pose.tolist(), 'bottom_z_m': 0.,
                'radii_m': [1., 1.], 'footprint_exponent': 2.}}


def observations():
    dry = np.full((60, 80), 2., np.float32)
    wet = dry.copy()
    wet[25:30, 34:40] = 1.6
    return dry, wet


def test_localized_changed_returns_recover_a_minority_liquid_layer():
    dry, wet = observations()
    result = dry_changed_evidence(installation(), wet, dry, np.ones(dry.shape, bool))
    assert result['height_m'] == pytest.approx(.4, abs=1e-6)
    assert result['changed_pixels'] == 30
    assert result['layer_tiles'] < 4


def test_empty_observation_does_not_create_a_positive_liquid_candidate():
    dry, _ = observations()
    wet = dry + np.random.default_rng(4).normal(0, .001, dry.shape)
    result = dry_changed_evidence(installation(), wet, dry, np.ones(dry.shape, bool))
    assert result['height_m'] is None
    assert result['changed_pixels'] == 0


def test_stationary_history_is_explicit_and_does_not_hide_the_source():
    dry, wet = observations()
    meter = IlluminationSceneMeter.from_empty_geometry(
        installation(), dry, refinement_enabled=True)
    rgb = np.full((*dry.shape, 3), 255, np.uint8)
    current = meter.measure('glare', rgb_bgr=rgb, depth_m=dry,
                            recent_depth_frames=[wet], stationary_confirmed=False)
    assert current['liquid_depth_m'] == pytest.approx(0., abs=1e-6)
    remembered = meter.measure('glare', rgb_bgr=rgb, depth_m=dry,
                               recent_depth_frames=[wet], stationary_confirmed=True)
    assert remembered['liquid_depth_m'] == pytest.approx(.4, abs=1e-6)
    details = remembered['diagnostics']['illumination_refinement']
    assert details['temporal_used']
    assert details['value_source'] == 'stationary_dry_changed_memory'
    stale = meter.measure('glare', rgb_bgr=rgb, depth_m=dry,
                          recent_depth_frames=[wet, dry, dry, dry], stationary_confirmed=True)
    assert stale['liquid_depth_m'] == pytest.approx(0., abs=1e-6)


def test_negative_value_is_not_snapped_to_zero_by_refinement():
    dry, _ = observations()
    meter = IlluminationSceneMeter.from_empty_geometry(
        installation(), dry, refinement_enabled=True)
    result = meter.measure('low_light', rgb_bgr=np.zeros((*dry.shape, 3), np.uint8),
                            depth_m=dry+.004)
    assert result['liquid_depth_m'] == pytest.approx(-.004, abs=1e-6)


def test_bottom_vote_cannot_override_an_overwhelming_current_wet_distribution():
    dry, _ = observations()
    bundle = installation()
    bundle.update(sensor_family='structured_light', baseline_m=.035)
    meter = IlluminationSceneMeter.from_empty_geometry(bundle, dry, refinement_enabled=True)
    wet = np.full(dry.shape, 1.6, np.float32)
    value, details = meter._refine_frame('glare', wet, -.005, {
        'layer_vote': {'applied': True}, 'selected_pixels': 4000,
        'height_p10_m': .39, 'quantile_percent': 58})
    assert value == pytest.approx(.4, abs=1e-6)
    assert details['value_source'] == 'contradicted_dry_vote_reverted'
    unchanged, _ = meter._refine_frame('glare', dry, -.005, {
        'layer_vote': {'applied': True}, 'selected_pixels': 4000,
        'height_p10_m': -.01, 'quantile_percent': 58})
    assert unchanged == -.005
