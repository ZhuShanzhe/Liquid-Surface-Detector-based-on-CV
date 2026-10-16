import numpy as np
import pytest
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.ordinary_surface_v35 import interior_ray_roi,estimate_ordinary

def calibration():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2.
    return HeightCalibration(np.array([[100.,0,49.5],[0,100.,49.5],[0,0,1.]]),pose,0.,(.6,.5))

def test_geometry_roi_inside_both_planes():
    c=calibration();roi=interior_ray_roi(c,1.,(100,100))
    assert 100<roi.sum()<10000
    y,x=np.nonzero(roi)
    for z in (0.,.5,1.):
        xy=np.column_stack(((x-49.5)/100,(y-49.5)/100))*(2-z)
        assert np.max((xy[:,0]/.6)**2+(xy[:,1]/.5)**2)<.85**2+1e-9

def test_negative_empty_measurement_not_forced_zero():
    c=calibration();raw=np.full((100,100),2.014)
    p=estimate_ordinary(np.zeros((100,100,3),np.uint8),raw,c,1.,'tof')
    assert p['statistics']['mean_m']==pytest.approx(-.014)
    assert not p['qualified']

def test_background_outside_roi_does_not_change_height():
    c=calibration();roi=interior_ray_roi(c,1.,(100,100));rgb=np.zeros((100,100,3),np.uint8)
    raw=np.full((100,100),1.7);a=estimate_ordinary(rgb,raw,c,1.,'tof')
    raw[~roi]=.2;b=estimate_ordinary(rgb,raw,c,1.,'tof')
    assert a['statistics']==b['statistics']

def test_no_depth_has_explicit_no_solution():
    p=estimate_ordinary(np.zeros((100,100,3),np.uint8),np.zeros((100,100)),calibration(),1.,'tof')
    assert p['statistics'] is None and p['reasons']

def test_invalid_rim_rejected():
    with pytest.raises(ValueError):interior_ray_roi(calibration(),-1.,(100,100))

def test_adaptive_fallback_keeps_deep_narrow_vessel_observable():
    from dataclasses import replace
    c=replace(calibration(),bottom_z_m=-8.,radii_m=(.2,.2))
    p=estimate_ordinary(np.zeros((100,100,3),np.uint8),np.full((100,100),1.8),c,1.,'tof',0)
    assert p['statistics']['mean_m']==pytest.approx(8.2)
    assert 'opening_only_geometry_fallback' in p['reasons']

def test_world_origin_shift_cannot_reveal_liquid_label():
    from dataclasses import replace
    c=calibration();pose=c.camera_to_world_cv.copy();pose[2,3]+=7.3
    shifted=replace(c,camera_to_world_cv=pose,bottom_z_m=c.bottom_z_m+7.3)
    raw=np.full((100,100),1.8);rgb=np.zeros((100,100,3),np.uint8)
    a=estimate_ordinary(rgb,raw,c,1.,'tof',0)
    b=estimate_ordinary(rgb,raw,shifted,8.3,'tof',0)
    assert a['statistics']['mean_m']==pytest.approx(b['statistics']['mean_m'])
