import cv2
import numpy as np
import pytest
from liquid_depth import finite_relative_v30 as model
from liquid_depth.finite_refraction_images_v30 import geometry, render
from liquid_depth.refraction_images_v23 import bottom_map, inverse_bottom


def test_footprint_and_fixed_roi():
    near,_=geometry(2.6,.12,.4);far,masks=geometry(9.6,.12,.4)
    assert near['empty_width_px']/far['empty_width_px']==pytest.approx(9.6/2.6)
    a,ca,ma,_=render(101,0,1.33,9.6,.12,.4,'transparent')
    b,cb,mb,_=render(101,.108,1.33,9.6,.12,.4,'translucent_mild')
    assert ca==cb and a['left'].shape==b['left'].shape
    for ch in masks:assert np.array_equal(ma[ch],mb[ch])
    p=model.baseline_pair(a['left'],b['left'],ca,ma['left'],'left')
    assert p['regions']==0  # V29's 28px margins cannot fit this ROI.


def test_scope_and_prior_budget():
    with pytest.raises(ValueError):render(1,.1,1.33,9,.3,.5,'strong_scatter')
    cal,_=geometry(9,.3,.5)
    with pytest.raises(ValueError):model.fit([], [0,.1,.2],cal,'transparent')
    with pytest.raises(ValueError):model.fit([], [0,.05,.1,.2,.3],cal,'transparent')
    with pytest.raises(ValueError):model.fit([], [0,.05,.1,.2,.27],cal,'transparent')


def test_direct_ignores_external_pixels():
    rng=np.random.default_rng(52);a=rng.integers(30,220,(45,48),dtype=np.uint8)
    b=a.copy();mask=np.zeros_like(a);mask[8:37,9:39]=255
    x=model.direct_one(a,b,mask)
    a[mask==0]=255;b[mask==0]=0
    y=model.direct_one(a,b,mask)
    assert x is not None and y is not None
    assert np.allclose(x[0],y[0]) and np.allclose(x[1],y[1],atol=1e-7)


def test_native_small_zoom():
    rng=np.random.default_rng(108)
    a=cv2.GaussianBlur(rng.random((68,72)).astype(np.float32),(0,0),1.1)
    a=np.uint8(a*255);centre=np.array([35.5,33.5]);scale=1.003
    mat=np.c_[np.eye(2)*scale,centre*(1-scale)]
    b=cv2.warpAffine(a,mat,(72,68),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT)
    mask=np.zeros_like(a);mask[7:-7,7:-7]=255
    result=model.direct_one(a,b,mask)
    assert result is not None
    # Quantization and resampling are real errors; test only coarse consistency.
    assert abs(result[2]['log_scale']-np.log(scale))<.003


def test_small_tracker_ignores_external_pixels():
    rng=np.random.default_rng(192)
    a=np.uint8(cv2.GaussianBlur(rng.random((60,64)).astype(np.float32),(0,0),.7)*255)
    mask=np.zeros_like(a);mask[8:52,9:55]=255;b=a.copy();cal={'fx':450.,'cx':32.,'cy':30.}
    p=model.small_lk(a,b,cal,mask,'left')
    a[mask==0]=255;b[mask==0]=0
    q=model.small_lk(a,b,cal,mask,'left')
    assert p['regions']>6 and p['regions']==q['regions']
    assert np.allclose(p['anchor_q'],q['anchor_q'])
    assert np.allclose(p['current_q'],q['current_q'],atol=2e-5)


def test_degenerate_support_not_fabricated():
    cal={'fx':450.,'cx':0.,'cy':0.}
    p=model.package([[i,0] for i in range(8)],[[i,0] for i in range(8)],cal,'left')
    assert not model.valid(p)
    a=np.full((24,24),100,np.uint8)
    assert model.direct_one(a,a,np.ones_like(a)*255) is None


def test_geometry_inference_and_no_capacity_clip():
    cal,_=geometry(8,.3,.8);hs=np.array([0,.045,.12,.195,.27]);n=1.37
    xx,yy=np.meshgrid(np.linspace(-.035,.035,9),np.linspace(-.025,.025,8))
    world=np.c_[xx.ravel(),yy.ravel()]*8
    def pair(a,b):
        p=inverse_bottom(world,hs[a],n,8)
        q=inverse_bottom(world,b,n,8)
        return dict(anchor_q=p,current_q=q,weights=np.ones(len(p)),regions=len(p),channel='left')
    pairs=[(a,b,pair(a,hs[b])) for a in range(5) for b in range(a+1,5)]
    prior=model.fit(pairs,hs,cal,'transparent')
    assert prior['index']==pytest.approx(n,abs=1e-5)
    for target in (.093,.37):
        pred=model.infer([(a,pair(a,target)) for a in range(5)],prior,cal)
        assert pred['height_m']==pytest.approx(target,abs=1e-5)
        assert pred['qualified'] is False
    assert 'outside_container_capacity' in pred['reasons']
    assert model.infer([],prior,cal)['height_m'] is None
    response=model.fit_response(pairs,hs,cal,'transparent')
    assert response['physical_index_identified'] is False
    pred=model.infer_response([(a,pair(a,.093)) for a in range(5)],response,cal)
    assert pred['height_m']==pytest.approx(.093,abs=.001)


def test_split_isolation_and_budget():
    from experiment_finite_v30 import specifications,LEVELS
    dev=specifications('development');test=specifications('test')
    assert len(dev)==64 and len(test)==144 and len(LEVELS)==5 and LEVELS[0]==0
    assert not {x['seed'] for x in dev}&{x['seed'] for x in test}
    for key in ('bottom','width','capacity','index','grain'):
        assert not {x[key] for x in dev}&{x[key] for x in test}
    for x in dev+test:
        assert not set(x['unknown'])&set(LEVELS)
        assert all(0<=r*x['capacity']<x['capacity'] for r in LEVELS)
