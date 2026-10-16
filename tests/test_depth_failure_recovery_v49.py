import numpy as np
import pytest

import liquid_depth.depth_failure_recovery_v49 as recovery


def bundle():
    k = np.array([[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]])
    pose = np.diag([1., -1., -1., 1.])
    pose[2, 3] = 6.
    return {'source':'empty_rim_plus_independent_metric_and_gravity',
            'sensor_family':'tof','image_size':[80,60],'rim_z_m':2.,
            'calibration':{'intrinsics':k.tolist(),
                           'camera_to_world_cv':pose.tolist(),
                           'bottom_z_m':0.,'radii_m':[1.,1.],
                           'footprint_exponent':2.}}


def fake_candidate(b, rgb, raw, mask, *, scene):
    value = 1.01 if int(mask.sum()) else None
    return {'scene':scene, 'numeric_candidate_m':value,
            'reasons':[] if value is not None else ['insufficient_current_depth'],
            'measurement_route':'surface_sensor','accepted':False,
            'production_qualified':False,'research_only':True,
            'input_support':{'geometry_consistent_pixels':int(mask.sum())}}


def test_fallback_uses_dry_center_and_weak_probability_without_clipping(monkeypatch):
    monkeypatch.setattr(recovery,'surface_candidate',fake_candidate)
    b = bundle()
    dry = np.full((60,80),6.,np.float32)
    wet = np.full((60,80),5.,np.float32)
    ref = recovery.prepare_dry_center_reference(b,dry)
    probability = np.full((60,80),.02,np.float32)
    out = recovery.recover_depth_failure(b,np.zeros((60,80,3),np.uint8),wet,
                                         probability,ref)
    assert out['recovery_status']=='low_probability_candidate_consistent'
    assert abs(out['numeric_candidate_m']-1.)<1e-6
    assert out['candidate_center_disagreement_m']<.02
    assert not out['production_qualified']


def test_stereo_keeps_sensor_likelihood_candidate(monkeypatch):
    monkeypatch.setattr(recovery,'surface_candidate',fake_candidate)
    b = bundle()
    b['sensor_family']='active_stereo'
    ref = recovery.prepare_dry_center_reference(b,np.full((60,80),6.,np.float32))
    out = recovery.recover_depth_failure(
        b,np.zeros((60,80,3),np.uint8),np.full((60,80),5.,np.float32),
        np.full((60,80),.02,np.float32),ref)
    assert out['numeric_candidate_m']==1.01


def test_disagreement_and_stale_reference_keep_no_output(monkeypatch):
    monkeypatch.setattr(recovery,'surface_candidate',fake_candidate)
    b = bundle()
    dry = np.full((60,80),6.,np.float32)
    ref = recovery.prepare_dry_center_reference(b,dry)
    probability = np.full((60,80),.02,np.float32)
    wrong = np.full((60,80),5.5,np.float32)
    out = recovery.recover_depth_failure(b,np.zeros((60,80,3),np.uint8),wrong,
                                         probability,ref)
    assert out['numeric_candidate_m'] is None
    assert out['recovery_status']=='candidate_center_disagreement'
    b['rim_z_m']=1.9
    with pytest.raises(ValueError,match='stale_dry_reference'):
        recovery.recover_depth_failure(b,np.zeros((60,80,3),np.uint8),wrong,
                                       probability,ref)


def test_existing_numeric_is_unchanged(monkeypatch):
    monkeypatch.setattr(recovery,'surface_candidate',fake_candidate)
    b = bundle()
    ref = recovery.prepare_dry_center_reference(b,np.full((60,80),6.,np.float32))
    raw = np.full((60,80),5.,np.float32)
    probability = np.full((60,80),.8,np.float32)
    out = recovery.recover_depth_failure(b,np.zeros((60,80,3),np.uint8),raw,
                                         probability,ref)
    assert out['numeric_candidate_m']==1.01
    assert out['recovery_status']=='baseline_unchanged'
