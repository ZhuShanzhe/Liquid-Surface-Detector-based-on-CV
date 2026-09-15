import numpy as np
import pytest
from scipy.ndimage import spline_filter,map_coordinates
from liquid_depth.refraction_inverse_v20 import reduce_burst,estimate_layers
from liquid_depth.refraction_evidence_v20 import linear_rgb

def observations():
    rng=np.random.default_rng(20001);h,w=100,160
    empty=.6+rng.normal(0,.001,(h,w))
    base=np.full((h,w),.65);base[:25]=.45
    fixed=rng.normal(0,.00025,(h,w))
    raw=base+fixed+rng.normal(0,.001,(64,h,w))
    camera=dict(sensor_model='nondispersive_coaxial_tof_proxy',fx=200.,fy=200.,cx=80.,cy=50.,depth_scale_to_m=1.)
    return raw,empty,camera

def test_temporal_reduction_counts_pixels_not_frames():
    raw,empty,camera=observations()
    z,d=reduce_burst(raw,empty,camera)
    assert z.shape==empty.shape
    assert d['independent_unit']=='spatial_pixel'
    assert d['stable_pixels']<=empty.size
    assert .0003<=d['spatial_noise_m']<.0005

def test_persistent_bias_is_not_divided_by_sqrt_frames():
    raw,empty,camera=observations()
    _,a=reduce_burst(raw,empty,camera)
    _,b=reduce_burst(np.tile(raw,(4,1,1)),empty,camera)
    assert b['spatial_noise_m']==pytest.approx(a['spatial_noise_m'])

def test_rejects_sequence_drift():
    raw,empty,camera=observations()
    raw+=np.linspace(0,.01,len(raw))[:,None,None]
    z,d=reduce_burst(raw,empty,camera)
    assert z is None and d['status']=='moving_surface'

def test_switching_returns_do_not_create_averaged_interface():
    raw,empty,camera=observations()
    raw[::2,40:50,40:50]=.45;raw[1::2,40:50,40:50]=.65
    z,d=reduce_burst(raw,empty,camera)
    assert not z[40:50,40:50].any()

def test_depth_units_are_converted_before_statistics():
    raw,empty,camera=observations()
    z,a=reduce_burst(raw,empty,camera)
    mm,b=reduce_burst(raw*1000,empty*1000,dict(camera,depth_scale_to_m=.001))
    assert np.allclose(z,mm)
    assert a['spatial_noise_m']==pytest.approx(b['spatial_noise_m'])

def test_single_frame_is_not_silently_promoted_to_stable_burst():
    raw,empty,camera=observations()
    z,d=reduce_burst(raw[0],empty,camera)
    assert z is None and d['status']=='insufficient_stable_frames'

def test_continuous_interpolation_resolves_below_one_32nd_pixel():
    y,x=np.indices((32,64));ref=np.sin(x*.2)+np.cos(y*.15)
    coef=spline_filter(ref,order=3)
    a=map_coordinates(coef,[[15.],[25.101]],order=3,prefilter=False)
    b=map_coordinates(coef,[[15.],[25.105]],order=3,prefilter=False)
    assert abs(float(a[0]-b[0]))>1e-5

def test_srgb_inverse_transfer():
    values=linear_rgb(np.array([0.,.04045,.5,1.]))
    assert values==pytest.approx([0.,.00313080495,.21404114048,1.])

def test_unsupported_sensor_rejected():
    raw,empty,camera=observations();rgb=np.zeros((*empty.shape,3))
    r=estimate_layers(rgb,raw,rgb,empty,dict(camera,sensor_model='active_stereo'))
    assert not r['accepted'] and r['status']=='unsupported_sensor_model'

def test_invalid_scale_rejected():
    raw,empty,camera=observations()
    with pytest.raises(ValueError):
        reduce_burst(raw,empty,dict(camera,depth_scale_to_m=-1))

def test_invalid_rgb_rejected_before_photometric_fitting():
    raw,empty,camera=observations();rgb=np.ones((*empty.shape,3))*255
    with pytest.raises(ValueError):
        estimate_layers(rgb,raw,rgb,empty,camera)
