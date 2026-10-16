import numpy as np
import pytest
from liquid_depth.liquid_height_metrics import (HeightCalibration, estimate_height,
    world_points, score_frame, summarize_frames, field_statistics, observable_features)


def calibration(distance=1., bottom=-.2):
    t = np.diag([1., -1., -1., 1.]); t[2, 3] = distance
    return HeightCalibration(np.array([[100.,0,39.5],[0,100.,39.5],[0,0,1.]]), t, bottom, (.35,.35))


def test_bottom_reference_not_camera_distance():
    c = calibration(); z = np.ones((80,80)); mask = np.ones_like(z, bool)
    p = estimate_height(z, mask, c)
    assert p['accepted'] and p['statistics']['mean_m'] == pytest.approx(.2)
    r = score_frame(estimate_height(z+.01, mask, c), field_statistics([.2]))
    assert r['absolute_error_m'] == pytest.approx(.01)
    assert r['abs_rel'] == pytest.approx(.05)
    assert not r['within_tolerance']


def test_blender_conversion():
    c = calibration()
    m = dict(camera_to_world=np.diag([1.,1.,1.,1.]).tolist(), camera_intrinsics=c.intrinsics,
             container_bottom_z_m=-.2,surface_radius_x_m=.35,surface_radius_y_m=.35,container_exponent=2)
    converted = HeightCalibration.from_simulation(m)
    assert world_points(np.ones((80,80)), converted)[40,40,2] == pytest.approx(-1)


def test_rejections_not_zero_errors():
    c = calibration(); p=estimate_height(np.zeros((80,80)),np.ones((80,80),bool),c)
    s=summarize_frames([score_frame(p,field_statistics([.2]))])
    assert s['mae_mm'] is None and s['output_coverage']==0
    assert s['effective_pass_coverage']==0 and s['tolerance_pass_rate'] is None


def test_coverage_denominator_includes_rejections():
    c=calibration(); mask=np.ones((80,80),bool)
    good=score_frame(estimate_height(np.ones((80,80)),mask,c),field_statistics([.2]))
    bad=score_frame(estimate_height(np.zeros((80,80)),mask,c),field_statistics([.2]))
    s=summarize_frames([good,bad]);assert s['output_coverage']==.5
    assert s['tolerance_pass_rate']==1 and s['effective_pass_coverage']==.5


def test_prediction_not_intersected_with_truth_mask():
    c=calibration(); mask=np.ones((80,80),bool)
    p=estimate_height(np.full((80,80),.8),mask,c)
    assert p['statistics']['mean_m']==pytest.approx(.4)
    r=score_frame(p,field_statistics([.2]));assert r['abs_rel']==pytest.approx(1)


def test_tilted_camera_world_transform():
    c=calibration(); angle=.2
    rot=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t=c.camera_to_world_cv.copy();t[:3,:3]=rot@t[:3,:3]
    c=HeightCalibration(c.intrinsics,t,c.bottom_z_m,c.radii_m)
    v,u=np.indices((80,80));ray=np.stack([(u-39.5)/100,(v-39.5)/100,np.ones((80,80))],-1)
    z=-t[2,3]/(ray@t[:3,:3].T)[...,2]
    p=estimate_height(z,np.ones_like(z,bool),c)
    assert p['statistics']['mean_m']==pytest.approx(.2,abs=1e-8)


def test_features_no_reference_argument():
    c=calibration();raw=np.ones((80,80));mask=np.ones_like(raw,bool)
    p=estimate_height(raw,mask,c)
    f,d=observable_features(p,p,raw,mask,c)
    assert len(f)==12 and np.isfinite(f).all() and d==pytest.approx(1.2)


def test_bottom_calibration_bias_propagates_without_scale_trick():
    c=calibration(bottom=-.195)
    p=estimate_height(np.ones((80,80)),np.ones((80,80),bool),c)
    assert p['statistics']['mean_m']==pytest.approx(.195)
    result=score_frame(p,field_statistics([.2]))
    assert result['absolute_error_m']==pytest.approx(.005)


def test_height_features_invariant_to_world_origin_z():
    c=calibration();t=c.camera_to_world_cv.copy();t[2,3]+=17
    shifted=HeightCalibration(c.intrinsics,t,c.bottom_z_m+17,c.radii_m)
    raw=np.ones((80,80));mask=np.ones_like(raw,bool)
    a=estimate_height(raw,mask,c);b=estimate_height(raw,mask,shifted)
    assert a['statistics']['mean_m']==pytest.approx(b['statistics']['mean_m'])
    fa,da=observable_features(a,a,raw,mask,c);fb,db=observable_features(b,b,raw,mask,shifted)
    assert np.allclose(fa,fb) and da==pytest.approx(db)


def test_invalid_pose_and_spatial_support():
    t=np.eye(4);t[2,2]=2
    with pytest.raises(ValueError):HeightCalibration(np.eye(3),t,0,(1,1))
    c=calibration();mask=np.zeros((80,80),bool);mask[35:45,35:45]=True
    p=estimate_height(np.ones((80,80)),mask,c)
    assert not p['accepted']
