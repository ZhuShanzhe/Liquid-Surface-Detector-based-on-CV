import numpy as np
import pytest

from liquid_depth.supported_scene_meter import FormalSceneMeter


def bundle(sensor='active_stereo'):
    k = [[100.,0.,40.],[0.,100.,30.],[0.,0.,1.]]
    pose = np.diag([1.,-1.,-1.,1.])
    pose[2,3] = 6.
    return {'source':'test_empty_geometry','sensor_family':sensor,
            'image_size':[80,60],'rim_z_m':2.,
            'calibration':{'intrinsics':k,'camera_to_world_cv':pose.tolist(),
                           'bottom_z_m':0.,'radii_m':[1.,1.],
                           'footprint_exponent':2.}}


def test_partial_loss_outputs_without_segmentation_support():
    meter = FormalSceneMeter.from_empty_geometry(
        bundle(), np.full((60,80),6.,np.float32))
    out = meter.measure('partial_depth_loss',
                        depth_m=np.full((60,80),5.,np.float32),
                        mask_probability=np.zeros((60,80),np.float32))
    assert out['numeric_output']
    assert abs(out['liquid_depth_m'] - 1.) < 1e-6
    assert out['route'] == 'partial_loss_ungated_v52'
    assert out['diagnostics']['quantile_percent'] == 58


def test_tof_shallow_uses_dry_paired_zero_reference():
    meter = FormalSceneMeter.from_empty_geometry(
        bundle('tof'), np.full((60,80),6.02,np.float32))
    out = meter.measure('partial_depth_loss',
                        depth_m=np.full((60,80),5.97,np.float32))
    assert out['numeric_output']
    assert abs(out['liquid_depth_m'] - .05) < 1e-5
    assert out['diagnostics']['value_source'] == 'tof_shallow_dry_paired_center'


def test_no_clipping_and_no_fabrication_without_any_current_return():
    meter = FormalSceneMeter.from_empty_geometry(
        bundle(), np.full((60,80),6.,np.float32))
    negative = meter.measure('partial_depth_loss',
                             depth_m=np.full((60,80),6.5,np.float32))
    assert negative['liquid_depth_m'] < 0
    empty = meter.measure('partial_depth_loss',
                          depth_m=np.zeros((60,80),np.float32))
    assert not empty['numeric_output']
    assert 'no_current_return_in_calibrated_container' in empty['reasons']


def test_camera_geometry_change_requires_recalibration():
    meter = FormalSceneMeter.from_empty_geometry(
        bundle(), np.full((60,80),6.,np.float32))
    meter.bundle['rim_z_m'] = 2.1
    with pytest.raises(ValueError, match='stale_dry_reference'):
        meter.measure('partial_depth_loss',
                      depth_m=np.full((60,80),5.,np.float32))


def test_optional_disparity_fit_uses_declared_camera_range():
    spec = bundle()
    spec['baseline_m'] = .055
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    result = meter.measure(
        'partial_depth_loss', depth_m=np.full((60, 80), 5., np.float32),
        mask_probability=np.zeros((60, 80), np.float32))
    assert result['numeric_output']
    assert result['route'] == 'partial_loss_censored_disparity_v56'
    assert abs(result['liquid_depth_m'] - 1.) < .02
    assert result['diagnostics']['mask_ignored_for_value']
    assert result['diagnostics']['disparity_fit']['max_return_depth_m'] == 10.


def test_optional_disparity_fit_falls_back_without_enough_returns():
    spec = bundle()
    spec['baseline_m'] = .055
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    wet = np.zeros((60, 80), np.float32)
    yy, xx = np.argwhere(meter.ordinary_container_roi)[0]
    wet[yy, xx] = 5.
    result = meter.measure('partial_depth_loss', depth_m=wet)
    assert result['numeric_output']
    assert result['route'] == 'partial_loss_ungated_v52'
    assert 'few_current_container_returns' in result['diagnostics']['quality_flags']


def test_camera_max_range_must_be_positive():
    with pytest.raises(ValueError, match='positive_camera_max_return_depth'):
        FormalSceneMeter.from_empty_geometry(
            bundle(), np.full((60, 80), 6., np.float32),
            max_return_depth_m=0.)


def test_camera_range_mismatch_uses_numeric_fallback():
    spec = bundle()
    spec['baseline_m'] = .055
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    wet = np.full((60, 80), 5., np.float32)
    yy, xx = np.argwhere(meter.ordinary_container_roi)[0]
    wet[yy, xx] = 10.01
    result = meter.measure('partial_depth_loss', depth_m=wet)
    assert result['numeric_output']
    assert result['route'] == 'partial_loss_ungated_v52'
    assert 'camera_range_profile_mismatch_quantile_fallback' in result['diagnostics']['quality_flags']


def test_joint_disparity_requires_stationary_confirmation():
    spec = bundle('structured_light')
    spec['baseline_m'] = .035
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    current = np.full((60, 80), 5., np.float32)
    prior = np.full((60, 80), 5., np.float32)
    ignored = meter.measure('partial_depth_loss', depth_m=current,
                            recent_depth_frames=[prior])
    assert ignored['route'] == 'partial_loss_censored_disparity_v56'
    assert 'temporal_frames_ignored_without_stationary_confirmation' in (
        ignored['diagnostics']['quality_flags'])
    joint = meter.measure('partial_depth_loss', depth_m=current,
                          recent_depth_frames=[prior],
                          stationary_confirmed=True)
    assert joint['route'] == 'partial_loss_joint_disparity_v59'
    assert joint['diagnostics']['formal_partial_loss_policy'] == 'selective_stationary_v61'
    assert joint['diagnostics']['joint_fit_used']
    assert joint['diagnostics']['disparity_fit']['frames_used'] == 2
    assert abs(joint['liquid_depth_m'] - 1.) < .02


def test_joint_disparity_rejects_misaligned_history():
    spec = bundle('structured_light')
    spec['baseline_m'] = .035
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    with pytest.raises(ValueError, match='aligned_recent_depth_frames'):
        meter.measure('partial_depth_loss',
                      depth_m=np.full((60, 80), 5., np.float32),
                      recent_depth_frames=[np.ones((30, 40), np.float32)],
                      stationary_confirmed=True)


def test_active_stereo_keeps_single_frame_when_history_supplied():
    spec = bundle('active_stereo')
    spec['baseline_m'] = .055
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.,
        layer_vote_enabled=False)
    current = np.full((60, 80), 5., np.float32)
    result = meter.measure(
        'partial_depth_loss', depth_m=current,
        recent_depth_frames=[current], stationary_confirmed=True)
    assert result['route'] == 'partial_loss_censored_disparity_v56'
    assert 'joint_disparity_not_enabled_for_sensor' in (
        result['diagnostics']['quality_flags'])


def test_formal_partial_loss_default_uses_roi_quantile_with_camera_profile():
    spec = bundle('structured_light')
    spec['baseline_m'] = .035
    meter = FormalSceneMeter.from_empty_geometry(
        spec, np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.)
    result = meter.measure(
        'partial_depth_loss', depth_m=np.full((60, 80), 5., np.float32))
    assert result['route'] == 'partial_loss_quantile_v64'
    assert result['diagnostics']['formal_partial_loss_policy'] == 'dry_anchored_quantile_v64'
    assert result['diagnostics']['disparity_fit'] is None

def test_ordinary_uses_shared_robust_return_estimator():
    meter = FormalSceneMeter.from_empty_geometry(
        bundle(), np.full((60, 80), 6., np.float32),
        max_return_depth_m=10.)
    frame = np.full((60, 80), 5., np.float32)
    ordinary = meter.measure('ordinary_diffuse', depth_m=frame)
    partial = meter.measure('partial_depth_loss', depth_m=frame)
    assert ordinary['liquid_depth_m'] == partial['liquid_depth_m']
    assert ordinary['route'] == 'ordinary_shared_robust_returns_v2'
    assert ordinary['diagnostics']['underlying_return_route'] == partial['route']
