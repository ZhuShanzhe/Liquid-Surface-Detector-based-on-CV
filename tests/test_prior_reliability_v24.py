import copy
import numpy as np
import pytest
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from liquid_depth.prior_reliability_v24 import fit_prior,infer,optical_coordinate
from liquid_depth.prior_reliability_v24 import route

CAL=dict(fx=450.,bottom_m=1.,baseline_m=.055)

def observation(h):
    x,y=np.meshgrid(np.linspace(-.35,.35,28),np.linspace(-.22,.22,20))
    q=np.column_stack((x.ravel(),y.ravel()))
    r=inverse_bottom(bottom_map(q,h,1.42,1.),h,1.42,1.,(.055,0))
    return dict(left_q=q,right_q=r,weights=np.ones(len(q)),left_pixels=q*450+[240,160],regions=len(q))

@pytest.mark.parametrize('mode',['affine','piecewise','pchip'])
def test_unknown_heights_from_known_levels(mode):
    hs=[.1,.22,.35,.5,.65]
    p=fit_prior([observation(h) for h in hs],hs,CAL,mode)
    for h in [.08,.15,.28,.42,.58,.72]:
        pred=infer(observation(h),p,CAL)
        assert pred['observable']
        assert abs(pred['height_m']-h)<.005

def test_failed_frame_does_not_invalidate_five_good_level_plan():
    hs=[.1,.22,.35,.5,.65];obs=[observation(h) for h in hs]
    obs[1]['regions']=0
    p=fit_prior(obs,hs,CAL,'pchip')
    assert len(p['used_frames'])==4 and len(p['discarded'])==1
    assert abs(infer(observation(.42),p,CAL)['height_m']-.42)<.005

def test_two_remaining_priors_are_not_enough():
    obs=[observation(h) for h in [.1,.35,.65]];obs[0]['regions']=0
    with pytest.raises(ValueError,match='three_usable'):fit_prior(obs,[.1,.35,.65],CAL)

def test_contradictory_level_labels_reject():
    with pytest.raises(ValueError,match='nonmonotonic'):
        fit_prior([observation(h) for h in [.1,.35,.65]],[.1,.65,.35],CAL)

def test_no_geometry_and_excessive_extrapolation_reject():
    obs=observation(.3);obs['right_q']=obs['left_q'].copy()
    with pytest.raises(ValueError,match='geometry'):optical_coordinate(obs,CAL)
    hs=[.2,.3,.4];p=fit_prior([observation(h) for h in hs],hs,CAL)
    assert not infer(observation(.72),p,CAL)['observable']

def test_test_truth_cannot_change_prediction():
    hs=[.1,.35,.65];p=fit_prior([observation(h) for h in hs],hs,CAL)
    obs=observation(.42);before=infer(obs,p,CAL)
    other=copy.deepcopy(obs);other['ground_truth_height_m']=7.;other['ground_truth_index']=1.9
    assert infer(other,p,CAL)==before

def test_route_preserves_primary_and_rejects_uncertain_fallback():
    primary=dict(height_m=.4,uncertainty_m=.001,observable=True)
    fallback=dict(height_m=.5,uncertainty_m=.001,observable=True)
    assert route(primary,fallback,2,2)['height_m']==.4
    primary['observable']=False
    assert route(primary,fallback,2,2)['source']=='calibrated_fallback'
    fallback['uncertainty_m']=.2
    assert not route(primary,fallback,2,2)['accepted']
