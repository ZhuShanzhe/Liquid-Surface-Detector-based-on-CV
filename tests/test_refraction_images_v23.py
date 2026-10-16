import numpy as np
import pytest
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom,render_capture
from liquid_depth.refraction_matching_v23 import decode_structured,stereo_matches
from liquid_depth.refraction_inverse_v23 import fit_prior,infer


def test_two_dimensional_ray_roundtrip():
    q=np.array([[.2,.3],[-.4,.1],[0,0]])
    x=bottom_map(q,.4,1.38,1.,(.055,0))
    assert np.allclose(inverse_bottom(x,.4,1.38,1.,(.055,0)),q,atol=1e-10)


def test_refraction_induces_vertical_disparity():
    q=np.array([[.3,.25]])
    x=bottom_map(q,.5,1.4,1.)
    r=inverse_bottom(x,.5,1.4,1.,(.055,0))
    assert abs(r[0,1]-q[0,1])>1e-5


def observed(h):
    x,y=np.meshgrid(np.linspace(-.35,.35,12),np.linspace(-.2,.2,8));q=np.column_stack((x.ravel(),y.ravel()))
    qr=inverse_bottom(bottom_map(q,h,1.38,1.),h,1.38,1.,(.055,0))
    return dict(left_q=q,right_q=qr,weights=np.ones(len(q)),left_pixels=q*450+[240,160],regions=len(q),visibility=.5)


def test_known_frames_then_unknown_height():
    cal=dict(fx=450.,bottom_m=1.,baseline_m=.055)
    prior=fit_prior([observed(h) for h in [.1,.35,.65]],[.1,.35,.65],cal)
    assert prior['index']==pytest.approx(1.38,abs=1e-4)
    assert infer(observed(.42),prior,cal)['height_m']==pytest.approx(.42,abs=1e-5)


def test_empty_images_no_invented_matches():
    cal=dict(fx=450.,bottom_m=1.,baseline_m=.055,cx=240,cy=160,reference_cx=240,reference_cy=160)
    assert stereo_matches(np.zeros((320,480),np.uint8),np.zeros((320,480),np.uint8),cal)['regions']==0


def test_active_stereo_reciprocal_tracking_keeps_observed_matches():
    images,cal,truth=render_capture(91234,.25,1.38,.8,.055,'transparent','active_stereo')
    obs=stereo_matches(images['left'],images['right'],cal,True)
    assert obs['regions']>=20
    points=obs['left_pixels'].astype(int)
    reference=np.column_stack((truth['right_u'][points[:,1],points[:,0]],truth['right_v'][points[:,1],points[:,0]]))
    assert np.median(np.linalg.norm(obs['right_pixels']-reference,axis=1))<.15


def test_current_truth_keys_do_not_affect_inverse():
    obs=observed(.42);cal=dict(fx=450.,bottom_m=1.,baseline_m=.055)
    prior=dict(index=1.38,bias_px=0.,index_sd=0.,bias_sd=0.)
    first=infer(obs,prior,cal)
    obs.update(ground_truth_height_m=9.,ground_truth_index=1.2)
    assert infer(obs,prior,cal)==first


def test_structured_images_decode_without_truth():
    images,cal,truth=render_capture(123,.25,1.38,.8,.035,'transparent','structured_light')
    obs=decode_structured(images,cal,True)
    assert obs['regions']>=20
    p=obs['left_pixels'].astype(int)
    reference=np.column_stack((truth['projector_u'][p[:,1],p[:,0]],truth['projector_v'][p[:,1],p[:,0]]))
    assert np.median(np.linalg.norm(obs['right_pixels']-reference,axis=1))<.5


def test_same_level_prior_rejects():
    with pytest.raises(ValueError,match='span'):
        fit_prior([observed(.3)]*3,[.3]*3,dict(fx=450.,bottom_m=1.,baseline_m=.055))
