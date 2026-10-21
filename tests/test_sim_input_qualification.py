import numpy as np
import pytest
from liquid_depth.sim_input_qualification import qualify_simulated_input,measure_if_eligible

def test_ordinary_valid_input():
    q=qualify_simulated_input(np.ones((10,10)),np.full((10,10),2),material='ordinary_diffuse')
    assert q['eligible'] and q['liquid_valid_pixels']==100

def test_invalid_input_never_calls_meter_or_classifier():
    class ExplodingMeter:
        def measure(self,obs):raise AssertionError('measurement must not execute')
    q=qualify_simulated_input(np.zeros((10,10)),np.full((10,10),2),material='ordinary_diffuse')
    assert not q['eligible']
    assert measure_if_eligible(ExplodingMeter(),object(),q) is None

def test_walls_do_not_inflate_liquid_availability():
    d=np.ones((30,30));s=np.zeros((30,30),int);s[:10,:10]=2;d[:10,:10]=0
    q=qualify_simulated_input(d,s,material='ordinary_diffuse')
    assert not q['eligible'] and q['liquid_invalid_fraction']==1.

def test_half_missing_allowed_but_more_rejected():
    d=np.ones((10,10));s=np.full(d.shape,2);d[:5]=0
    assert qualify_simulated_input(d,s,material='ordinary_diffuse')['eligible']
    d[5,0]=0
    assert not qualify_simulated_input(d,s,material='ordinary_diffuse')['eligible']

def test_float_support_required():
    d=np.ones((10,10));s=np.full(d.shape,2)
    assert not qualify_simulated_input(d,s,material='ordinary_diffuse',has_floats=True)['eligible']
    s[:2,:2]=3
    assert qualify_simulated_input(d,s,material='ordinary_diffuse',has_floats=True)['eligible']

def test_line_of_pixels_is_not_spatial_support():
    d=np.zeros((40,40));s=np.full(d.shape,2);d[0,:]=1
    q=qualify_simulated_input(d,s,material='ordinary_diffuse')
    assert 'insufficient_visible_liquid_returns' in q['reasons']

def test_refractive_uses_phase_not_opaque_depth_validity():
    c=np.ones((3,4,10,10));c[:,0]=2
    q=qualify_simulated_input(np.zeros((20,20)),np.zeros((20,20)),material='homogeneous_transparent',correlations=c,phase_roi=np.ones((10,10),bool))
    assert q['eligible']
    c[:,:,:6]=1200
    assert not qualify_simulated_input(np.ones((20,20)),np.zeros((20,20)),material='homogeneous_transparent',correlations=c,phase_roi=np.ones((10,10),bool))['eligible']

def test_no_upper_range_clipping():
    q=qualify_simulated_input(np.full((10,10),10.1),np.full((10,10),2),material='ordinary_diffuse')
    assert q['eligible']

def test_allowed_input_calls_exactly_once_without_modification():
    class Meter:
        count=0
        def measure(self,obs):self.count+=1;return obs
    meter=Meter();obs=object();q={'eligible':True}
    assert measure_if_eligible(meter,obs,q) is obs and meter.count==1
