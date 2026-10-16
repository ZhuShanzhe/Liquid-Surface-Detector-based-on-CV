import numpy as np
import pytest
from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.ordinary_surface_v36 import estimate_v36
from liquid_depth.ordinary_temporal_v37 import estimate_window
from liquid_depth.correlated_noise_v37 import fields
from liquid_depth.ordinary_temporal_v37 import estimate_guarded4

def inputs():
    k=np.array([[314.,0,39.5],[0,314.,39.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=8.6
    cal=HeightCalibration(k,pose,0.,(2.,2.));rng=np.random.default_rng(371)
    out=[]
    for _ in range(4):
        disparity=314*.035/8.9+rng.normal(0,.26,(80,80));disparity=np.round(disparity*16)/16
        raw=314*.035/np.maximum(disparity,1e-3);raw[(raw<.12)|(raw>10.5)]=0;out.append(np.round(raw/.001)*.001)
    return cal,out

def test_single_matches_frozen_v36():
    c,raw=inputs();a=estimate_window([raw[-1]],c,.5,'structured_light')
    b=estimate_v36(None,raw[-1],c,.5,'structured_light',uniform=True)
    assert a['statistics']==b['statistics']

def test_duplicate_frames_are_not_artificial_extra_information():
    c,raw=inputs();a=estimate_window([raw[-1]],c,.5,'structured_light')
    b=estimate_window([raw[-1]]*4,c,.5,'structured_light')
    assert a['statistics']['mean_m']==pytest.approx(b['statistics']['mean_m'],abs=1e-5)
    assert not b['qualified'] and 'correlation_uncertainty_not_calibrated' in b['reasons']

def test_negative_joint_height_not_snapped():
    c,raw=inputs();a=estimate_window(raw,c,.5,'structured_light',timestamps=[0,1/30,2/30,3/30])
    assert a['statistics']['mean_m']<-.1 and a['frames_used']==4

@pytest.mark.parametrize('kw',[{'motion_detected':True},{'fixed_pose':False},{'timestamps':[0,.2,.4,.6]}])
def test_invalid_history_falls_back_to_current_not_reject(kw):
    c,raw=inputs();p=estimate_window(raw,c,.5,'structured_light',**kw)
    current=estimate_window([raw[-1]],c,.5,'structured_light')
    assert p['frames_used']==1 and p['statistics']==current['statistics']

def test_missing_current_does_not_report_historical_value():
    c,raw=inputs();raw[-1]=np.zeros_like(raw[-1]);p=estimate_window(raw,c,.5,'structured_light')
    assert p['statistics'] is None and p['reasons']==['no_current_support']

def test_timestamp_order_must_be_valid():
    c,raw=inputs()
    with pytest.raises(ValueError):estimate_window(raw,c,.5,'structured_light',timestamps=[0,.1,.05,.2])

def test_noise_variance_and_correlation():
    seq=list(fields((160,240),40,3719));result={}
    for mode in ('iid','spatial','persistent'):
        a=np.stack([f[mode] for f in seq]);var=a.var()
        assert .9<var<1.1
        horizontal=np.corrcoef(a[:,:,:-1].ravel(),a[:,:,1:].ravel())[0,1]
        temporal=np.corrcoef(a[:-1].ravel(),a[1:].ravel())[0,1]
        result[mode]=(horizontal,temporal,a.mean(axis=(1,2)).std())
    assert abs(result['iid'][0])<.02 and abs(result['iid'][1])<.02
    assert .38<result['spatial'][0]<.5 and abs(result['spatial'][1])<.03
    assert .38<result['persistent'][1]<.52
    assert result['spatial'][2]>2*result['iid'][2]

def test_guarded_identical_far_frames_use_current():
    c,raw=inputs();p=estimate_guarded4([raw[0]]*4,c,.5,'structured_light')
    q=estimate_window([raw[0]],c,.5,'structured_light')
    assert p['frames_used']==1 and p['statistics']==q['statistics']
    assert p['temporal_block_correlation']>.99

def test_guarded_near_preserves_single_exactly():
    c,raw=inputs();raw=[np.where(x>0,1.5,x) for x in raw]
    p=estimate_guarded4(raw,c,.5,'tof');q=estimate_window(raw[-1:],c,.5,'tof')
    assert p['statistics']==q['statistics'] and p['frames_used']==1

def test_guarded_empty_history_is_explicit_error():
    c,_=inputs()
    with pytest.raises(ValueError):estimate_guarded4([],c,.5,'structured_light')
