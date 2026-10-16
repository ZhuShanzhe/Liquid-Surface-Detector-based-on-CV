from dataclasses import replace
import numpy as np
import pytest
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference, estimate_v39
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.surface_refinement import StereoNoiseModel


def setup(bottom=8.5, bias=.025, sigma=.18, height=.3, seed=3911):
    rng=np.random.default_rng(seed);fx=629.;fb=fx*.1;shape=(120,160)
    k=np.array([[fx,0,79.5],[0,fx,59.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=bottom
    cal=HeightCalibration(k,pose,0.,(2.,2.))
    model=StereoNoiseModel(.1,.18,1e-9,1e-9,.4,10.)
    def image(distance, echo=False):
        z=np.full(shape,distance)
        if echo:z[rng.random(shape)<.12]=bottom
        disp=np.round((fb/z+bias+rng.normal(0,sigma,shape))*16)/16
        d=np.round((fb/disp)/.001)*.001
        return np.where((d>=.4)&(d<=10),d,0).astype(np.float32)
    return cal,model,image(bottom),image(bottom-height,True)


def test_dry_reference_recovers_bias_and_noise():
    cal,m,dry,_=setup()
    r=calibrate_dry_reference(dry,cal,.5,m)
    assert r.disparity_bias_px==pytest.approx(.025,abs=.006)
    assert r.disparity_sigma_px==pytest.approx(.18,abs=.006)


def test_noisy_echo_and_bias_recovery():
    cal,m,dry,wet=setup()
    r=calibrate_dry_reference(dry,cal,.5,m)
    p=estimate_v39(wet,cal,.5,m,r)
    assert p['statistics']['mean_m']==pytest.approx(.3,abs=.025)
    assert not p['qualified']


@pytest.mark.parametrize('height',[-.05,.8])
def test_signed_and_over_capacity_outputs_retained(height):
    cal,m,dry,wet=setup(bottom=2.,height=height,sigma=.06)
    r=calibrate_dry_reference(dry,cal,.5,m)
    p=estimate_v39(wet,cal,.5,m,r)
    assert p['statistics']['mean_m']==pytest.approx(height,abs=.004)


def test_reference_installation_mismatch():
    cal,m,dry,wet=setup();r=calibrate_dry_reference(dry,cal,.5,m)
    with pytest.raises(ValueError):estimate_v39(wet,replace(cal,bottom_z_m=.2),.5,m,r)


def test_reference_pose_and_intrinsics_mismatch():
    cal,m,dry,wet=setup();r=calibrate_dry_reference(dry,cal,.5,m)
    moved=cal.camera_to_world_cv.copy();moved[0,3]+=.01
    with pytest.raises(ValueError):estimate_v39(wet,replace(cal,camera_to_world_cv=moved),.5,m,r)
    k=cal.intrinsics.copy();k[0,2]+=.5
    with pytest.raises(ValueError):estimate_v39(wet,replace(cal,intrinsics=k),.5,m,r)


def test_absent_current_data_not_replaced_by_empty_prior():
    cal,m,dry,wet=setup();r=calibrate_dry_reference(dry,cal,.5,m)
    assert estimate_v39(wet*0,cal,.5,m,r)['statistics'] is None


def test_invalid_empty_calibration():
    cal,m,dry,_=setup()
    with pytest.raises(ValueError):calibrate_dry_reference(dry*0,cal,.5,m)


def test_no_reference_is_explicit_and_does_not_use_truth():
    cal,m,dry,wet=setup();p=estimate_v39(wet,cal,.5,m)
    assert p['statistics'] is not None
    assert 'no_independent_bias_reference' in p['reasons']


def test_range_cutoff_empty_reference():
    cal,m,dry,wet=setup(bottom=9.95,sigma=.36,height=.5)
    r=calibrate_dry_reference(dry,cal,.7,m)
    assert r.disparity_bias_px==pytest.approx(.025,abs=.035)
    assert estimate_v39(wet,cal,.7,m,r)['statistics'] is not None
