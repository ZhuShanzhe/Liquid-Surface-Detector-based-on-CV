import numpy as np
import pytest
from liquid_depth import transparent_refinement_v26 as v26
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from liquid_depth.refraction_matching_v26 import decode_structured
from liquid_depth.refraction_matching_v23 import decode_structured as legacy_decode

CAL=dict(fx=450.,bottom_m=1.,baseline_m=.055)

def observation(h,coefficient=0.):
    x,y=np.meshgrid(np.linspace(-.35,.35,22),np.linspace(-.22,.22,16))
    q=np.column_stack((x.ravel(),y.ravel()))
    r=inverse_bottom(bottom_map(q,h,1.42,1.),h,1.42,1.,(.055,0))
    target=r*450+[240,160];observed=target.copy()
    for _ in range(15):
        observed[:,0]-=(observed[:,0]+coefficient*np.sin(2*np.pi*observed[:,0])-target[:,0])/(1+2*np.pi*coefficient*np.cos(2*np.pi*observed[:,0]))
    return dict(left_q=q,right_q=(observed-[240,160])/450,weights=np.ones(len(q)),
                left_pixels=q*450+[240,160],right_pixels=observed,regions=len(q))

def test_bounded_gain_and_constant_tail_are_continuous():
    p=dict(x=[.1,.3,.6],y=[.15,.25,.7])
    grid=np.linspace(.01,.9,2001)
    output=np.array([v26.bounded_height(x,p,CAL,1.,.3,'constant') for x in grid])
    derivative=np.diff(output)/np.diff(grid)
    assert derivative.min()>=.7-1e-8 and derivative.max()<=1.3+1e-8
    assert abs((output[-1]-grid[-1])-(output[-2]-grid[-2]))<1e-10
    for boundary in p['x']:
        assert abs(v26.bounded_height(boundary-1e-8,p,CAL)-v26.bounded_height(boundary+1e-8,p,CAL))<3e-8

def test_zero_shrink_keeps_physical_estimate():
    assert v26.bounded_height(.72,dict(x=[.1,.3,.6],y=[.2,.4,.8]),CAL,0.)==.72

def test_joint_pixel_bias_corrects_unknown_heights():
    hs=[.1,.22,.35,.5,.65]
    p=v26.fit_prior([observation(h,.03) for h in hs],hs,CAL,'transparent',1,.005)
    assert abs(p['pixel_bias_coefficients'][0]-.03)<.003
    for h in [.08,.18,.42,.72]:
        r=v26.infer(observation(h,.03),p,CAL,dict(kind='none'))
        assert r['finite'] and abs(r['height_m']-h)<.003

def test_scope_and_test_truth_invariance():
    hs=[.1,.35,.65];obs=[observation(h) for h in hs]
    with pytest.raises(ValueError,match='scope'):v26.fit_prior(obs,hs,CAL,'translucent_strong')
    p=v26.fit_prior(obs,hs,CAL,'translucent_mild');o=observation(.42)
    a=v26.infer(o,p,CAL,dict(kind='none'));o['height_truth']=9.
    assert v26.infer(o,p,CAL,dict(kind='none'))==a
    assert a['qualified'] is False and not a['confidence_gate_enabled']

def phase_images(noise):
    yy,xx=np.indices((320,480));coords={'x':xx+5.2,'y':yy+3.1}
    rng=np.random.default_rng(123)
    def image(a):return np.uint8(np.clip(a+rng.normal(0,noise,a.shape),0,255))
    images={'black':np.full(xx.shape,30,np.uint8),'white':np.full(xx.shape,210,np.uint8)}
    for axis,coord in coords.items():
        cell=np.floor(coord/16).astype(int);gray=cell^(cell>>1)
        for b in range(5):
            bit=(gray>>b)&1
            images[f'{axis}_g{b}']=image(30+180*bit)
            images[f'{axis}_i{b}']=image(210-180*bit)
        for k in range(4):images[f'{axis}_p{k}']=image(120+80*np.cos(2*np.pi*coord/16+k*np.pi/2))
    return images

def test_block_phase_reduces_noisy_coordinate_error():
    cal=dict(fx=450,cx=240,cy=160,reference_cx=256,reference_cy=192)
    images=phase_images(8.)
    old=legacy_decode(images,cal,True);new=decode_structured(images,cal,.8,16)
    error=lambda o:np.median(np.linalg.norm(o['right_pixels']-o['left_pixels']-[5.2,3.1],axis=1))
    assert new['regions']>=20 and error(new)<error(old)
