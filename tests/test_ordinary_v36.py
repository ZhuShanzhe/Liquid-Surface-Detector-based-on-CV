import numpy as np
import pytest
from liquid_depth.surface_refinement import StereoNoiseModel
from liquid_depth.ordinary_surface_v36 import discrete_clearance,normal_mass,estimate_v36
from liquid_depth.liquid_height_metrics import HeightCalibration

def test_normal_mass_stable_in_tails():
    assert normal_mass(np.array([8.]),np.array([9.]))[0]>0
    assert normal_mass(np.array([-1.]),np.array([1.]))[0]==pytest.approx(.68268949)

def sample():
    rng=np.random.default_rng(3621);fx=314.;model=StereoNoiseModel.simulation_proxy('structured_light')
    disp=fx*model.baseline_m/8.9+rng.normal(0,.26,12000);disp=np.round(disp*16)/16
    d=fx*model.baseline_m/np.maximum(disp,1e-3);d=np.round(d[(d>=.12)&(d<=10.5)]/.001)*.001
    return d,fx,model

def test_native_pixel_unit_rescaling_is_invariant():
    from dataclasses import replace
    d,fx,m=sample();c=np.ones(len(d));w=c/len(c)
    a=discrete_clearance(d,c,w,fx,m)
    b=discrete_clearance(d,c,w,fx/2,replace(m,disparity_sigma_px=m.disparity_sigma_px/2),disparity_step=1/32)
    assert a==pytest.approx(b,abs=1e-6)

def test_censored_quantized_estimate_recovers_distance():
    d,fx,m=sample();c=np.ones(len(d));w=c/len(c)
    assert abs(discrete_clearance(d,c,w,fx,m)-8.9)<.08

def test_few_samples_no_output():
    assert discrete_clearance(np.ones(8),np.ones(8),np.ones(8),314.,StereoNoiseModel.simulation_proxy('structured_light')) is None

def test_tof_negative_height_not_zeroed():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2.
    cal=HeightCalibration(np.array([[100.,0,49.5],[0,100.,49.5],[0,0,1.]]),pose,0.,(.6,.5))
    p=estimate_v36(None,np.full((100,100),2.014),cal,1.,'tof')
    assert p['statistics']['mean_m']==pytest.approx(-.014)

def test_structured_light_negative_height_is_retained():
    d,fx,model=sample();raw=np.resize(d,(80,80))
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=8.6
    cal=HeightCalibration(np.array([[fx,0,39.5],[0,fx,39.5],[0,0,1.]]),pose,0.,(2.,2.))
    p=estimate_v36(None,raw,cal,.5,'structured_light',uniform=True)
    assert p['statistics']['mean_m']<-.1
    assert not p['qualified']

def test_measurement_can_exceed_known_capacity():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=2.
    cal=HeightCalibration(np.array([[100.,0,49.5],[0,100.,49.5],[0,0,1.]]),pose,0.,(.6,.5))
    p=estimate_v36(None,np.full((100,100),1.7),cal,.2,'tof',uniform=True)
    assert p['statistics']['mean_m']==pytest.approx(.3)

def test_invalid_quantization_rejected():
    with pytest.raises(ValueError):estimate_v36(None,None,None,1.,'structured_light',disparity_step=float('nan'))
