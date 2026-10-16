import inspect
import cv2,numpy as np,pytest
from liquid_depth import range_registration_v33 as reg
from liquid_depth import finite_relative_v30 as base
from liquid_depth import range_relative_v32 as height
from experiment_sr_v33 import specs,levels
from liquid_depth.range_runtime_v33 import FivePriorImageMeter

def texture_pair(scale=0.,translation=(0.,0.)):
    rng=np.random.default_rng(47117)
    # Evaluate both frames from one continuous band-limited scene. OpenCV's
    # quantized cubic warp is not an exact subpixel ground-truth generator.
    centre=np.array([87.5,71.5]);yy,xx=np.indices((144,176));p=np.stack((xx,yy),-1)
    angles=np.linspace(0,np.pi,12,endpoint=False)
    directions=np.stack((np.cos(angles),np.sin(angles)),-1)*2*np.pi/18
    phases=rng.uniform(-np.pi,np.pi,len(angles))
    def field(q):return (125+8*np.cos(q@directions.T+phases).sum(-1)).astype(np.float32)
    image=field(p);target=field(centre+(p-centre-translation)*np.exp(-scale))
    mask=np.zeros_like(image,dtype=np.uint8);mask[8:-8,8:-8]=255
    cal=dict(fx=450.,cx=87.5,cy=71.5,bottom_m=10.,container_max_height_m=9.4)
    return image,target,mask,cal

@pytest.mark.parametrize('name',['midpoint_650','midpoint_dense','midpoint_spline5','native_dense','bicubic_x2','midpoint_bicubic2'])
def test_subpixel_scale_registration(name):
    a,b,m,c=texture_pair(.004,(.11,-.08))
    if reg.OPTIONS[name]['factor']==2:
        a=cv2.resize(a,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        b=cv2.resize(b,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        m=cv2.resize(m,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    p=reg.pair(a,b,c,m,'left',name)
    assert base.valid(p)
    assert base.observed_log_scale(p)==pytest.approx(.004,abs=.00025)
    assert p['correlated_dense_samples'] and not p['independent_samples_claimed']

def test_midpoint_inverse_symmetry():
    a,b,m,c=texture_pair(.013,(.2,-.15))
    p=reg.pair(a,b,c,m,'left','midpoint_dense');q=reg.pair(b,a,c,m,'left','midpoint_dense')
    assert base.observed_log_scale(p)==pytest.approx(-base.observed_log_scale(q),abs=2e-6)

def test_pixel_centres_and_scale_geometry():
    cal=dict(fx=450.,cx=80.,cy=64.,width=160,height=128)
    c=reg.upscale_geometry(cal,2)
    assert c['fx']==900 and c['cx']==160.5 and c['cy']==128.5
    p=np.array([[0.,0.],[80.,64.],[159.,127.]])
    np.testing.assert_allclose(reg.native_coordinates((p+.5)*2-.5,2),p)
    np.testing.assert_allclose(((p+.5)*2-.5-[c['cx'],c['cy']])/c['fx'],(p-[80,64])/450)

def test_missing_images_explicit_and_no_truth_inputs():
    a,b,m,c=texture_pair();m[:]=0
    assert not base.valid(reg.pair(a,b,c,m,'left','midpoint_dense'))
    assert list(inspect.signature(reg.pair).parameters)==['a','b','cal','mask','channel','name']
    assert list(inspect.signature(reg.SwinIRX2.__call__).parameters)==['self','images']

def test_budget_and_disjoint_splits():
    dev=specs('development');test=specs('test')
    assert not {j['seed'] for j in dev}&{j['seed'] for j in test}
    assert all(j['width_px']>=128 and j['bottom']<=10 for j in dev+test)
    assert len(levels('test'))==167
    assert sum(l['height']==0 for l in levels('test'))==24
    assert max(l['height'] for l in levels('test'))==9

def test_runtime_signed_heights_and_no_test_memory(monkeypatch):
    a,b,m,c=texture_pair();hs=[0.,1.4,3.7,6.1,8.4]
    def images(h):return {ch:np.full(a.shape,h,dtype=float) for ch in ('left','right')}
    def fake_pair(a,b,cal,mask,ch,name):
        p=np.array([[20,20],[20,100],[100,20],[100,100],[60,20],[20,60]],float)
        centre=[cal['cx'],cal['cy']];s=.025*(float(b[0,0])-float(a[0,0]))
        return base.package(p,(p-centre)*np.exp(s)+centre,cal,ch,dict(correlation=1.))
    monkeypatch.setattr(reg,'pair',fake_pair)
    meter=FivePriorImageMeter(c,{'left':m,'right':m},[images(h) for h in hs],hs,'translucent_mild')
    for h in [-.02,.002,.01,.5,2.5,11.]:
        pred=meter.measure(images(h));assert pred['height_m']==pytest.approx(h,abs=1e-8)
        assert pred['prior_groups']==5 and not pred['measurement_frame_memory']
        assert not pred['zero_snapping'] and not pred['capacity_clipping']
    with pytest.raises(ValueError):meter.measure({'left':a})
    with pytest.raises(ValueError):FivePriorImageMeter(c,{'left':m,'right':m},[images(h) for h in hs[:4]],hs,'translucent_mild')
