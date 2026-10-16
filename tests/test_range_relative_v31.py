import cv2
import numpy as np
import pytest
from liquid_depth.range_images_v31 import RangeScene
from liquid_depth import range_relative_v31 as model
from liquid_depth import finite_relative_v30 as base
from experiment_range_v31 import specs,summary


def cal():
    return dict(fx=450.,cx=80.,cy=64.,bottom_m=10.,container_max_height_m=9.4)


def pair(s,correlation=1.):
    p=np.array([[-30.,-20.],[30.,-20.],[-30.,20.],[30.,20.],[0.,20.],[20.,0.]])+[80,64]
    return base.package(p,(p-[80,64])*np.exp(s)+[80,64],cal(),'left',dict(correlation=correlation))


def test_curve_known_nodes_and_unclipped_extrapolation():
    hs=[0.,1.4,3.7,6.1,8.4];latent=np.asarray(hs)*.025
    ps=[(a,b,pair(latent[b]-latent[a])) for a in range(5) for b in range(a+1,5)]
    prior=model.fit_curve(ps,hs,cal(),'transparent')
    for h in (0.,.2,4.,9.,11.):
        groups=[(a,pair(.025*h-latent[a])) for a in range(5)]
        result=model.infer_curve(groups,prior,cal())
        assert result['height_m']==pytest.approx(h,abs=1e-4)
        assert not result['confidence_gate_enabled']
    assert 'outside_container_capacity' in result['reasons']
    assert not prior['physical_index_identified']


def test_disconnected_priors_do_not_use_test_truth():
    hs=[0.,1.4,3.7,6.1,8.4]
    ps=[(a,b,pair((hs[b]-hs[a])*.025)) for a,b in [(0,1),(1,2)]]
    prior=model.fit_curve(ps,hs,cal(),'translucent_mild')
    assert prior['nodes']==[0,1,2]
    result=model.infer_curve([(2,pair((9.-hs[2])*.025))],prior,cal())
    assert result['height_m']==pytest.approx(9.,abs=1e-4)
    assert model.infer_curve([],prior,cal())['height_m'] is None


def test_wide_scale_and_external_roi_invariance():
    rng=np.random.default_rng(43)
    a=np.uint8(np.clip(cv2.GaussianBlur(rng.normal(128,45,(128,160)).astype(np.float32),(0,0),1.6),0,255))
    transform=np.array([[1.28,0,-.28*80],[0,1.28,-.28*64]],dtype=float)
    b=cv2.warpAffine(a,transform,(160,128),flags=cv2.INTER_CUBIC)
    mask=np.zeros_like(a);mask[12:-12,12:-12]=255
    fit=model.wide_pair(a,b,cal(),mask,'left')
    assert base.observed_log_scale(fit)==pytest.approx(np.log(1.28),abs=.001)
    a2=a.copy();b2=b.copy();a2[mask==0]=255;b2[mask==0]=0
    other=model.wide_pair(a2,b2,cal(),mask,'left')
    assert base.observed_log_scale(other)==pytest.approx(base.observed_log_scale(fit),abs=1e-10)


def test_scene_fixed_native_sensor_and_independent_zero():
    scene=RangeScene(11,10.,9.4,128,1.33,.05,.035,'translucent_mild')
    empty,t0=scene.capture(0.,0);repeat,t1=scene.capture(0.,1000);deep,t9=scene.capture(9.,1072)
    assert t9['height_m']==9.
    assert not np.array_equal(empty['left'],repeat['left'])
    assert empty['left'].shape==deep['left'].shape==scene.masks['left'].shape
    assert scene.cal['crop_x0']>=0 and scene.cal['crop_y0']>=0
    assert scene.cal['crop_x0']+scene.cal['width']<=480
    assert scene.cal['crop_y0']+scene.cal['height']<=320
    assert 'index' not in scene.cal and 'height_m' not in scene.cal
    with pytest.raises(ValueError):RangeScene(1,10.1,9.4,128,1.33,.05,0,'transparent')
    with pytest.raises(ValueError):RangeScene(1,10.,9.4,32,1.33,.05,0,'transparent')


def test_uniform_actual_range_disjoint_split_and_zero_metric():
    dev=specs('development');test=specs('test')
    assert not {c['seed'] for c in dev}&{c['seed'] for c in test}
    for c in dev+test:
        assert c['bottom']<=10 and c['capacity']>9 and c['width_px']>=96
        assert c['heights'][0]==0 and c['heights'][-1]==9
        assert np.allclose(np.diff(c['heights']),np.diff(c['heights'])[0])
    s=summary([dict(predicted_height_m=.001,true_height_m=0,absolute_error_m=.001)])
    assert s['absrel_positive_pct'] is None and s['zero_height_frames']==1 and s['pass_pct']==100
