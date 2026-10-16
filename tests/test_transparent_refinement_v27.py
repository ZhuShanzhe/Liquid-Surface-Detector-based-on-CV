import numpy as np
import pytest
from liquid_depth import transparent_refinement_v27 as v27
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom

CAL=dict(fx=450.,bottom_m=8.,baseline_m=.055)
SETTING=dict(harmonics=1,period_px=16.,spatial=True,fit_y=True,ridge=1e-6)

def observation(h,inject=True):
    x,y=np.meshgrid(np.linspace(-.35,.35,22),np.linspace(-.22,.22,16))
    q=np.column_stack((x.ravel(),y.ravel()))
    r=inverse_bottom(bottom_map(q,h,1.42,8.),h,1.42,8.,(.055,0))
    target=r*450+[256,192];observed=target.copy()
    o=dict(left_q=q,right_q=r,weights=np.ones(len(q)),left_pixels=q*450+[240,160],right_pixels=observed,regions=len(q))
    if inject:
        for _ in range(20):
            o['right_pixels']=observed
            sx=v27.bias_basis(o,SETTING,0)@np.array([.025,-.015,.008,-.006,.005])
            sy=v27.bias_basis(o,SETTING,1)@np.array([.004,.015,-.01,.005,.003,-.004])
            observed=target-np.column_stack((sx,sy))-[.02,0.]
        o['right_pixels']=observed;o['right_q']=(observed-[256,192])/450
    return o

def test_injected_field_and_phase_bias_unknown_levels():
    hs=[.24,.64,1.76,3.84,6.08]
    p=v27.fit_prior([observation(h) for h in hs],hs,CAL,'translucent_mild',SETTING)
    for h in [.12,.52,2.4,6.4]:
        r=v27.infer(observation(h),p,CAL,dict(kind='none'))
        assert r['finite'] and abs(r['height_m']-h)<.004
        assert r['residual_px']<.001

def test_exact_five_budget_and_scope():
    with pytest.raises(ValueError,match='exactly_five'):
        v27.fit_prior([observation(1.)]*4,[1.]*4,CAL,'transparent',SETTING)
    with pytest.raises(ValueError,match='scope'):
        v27.fit_prior([observation(1.)]*5,[1.]*5,CAL,'translucent_strong',SETTING)

def test_truth_invariance_and_no_output_gate():
    hs=[.24,.64,1.76,3.84,6.08]
    p=v27.fit_prior([observation(h,False) for h in hs],hs,CAL,'transparent',SETTING)
    o=observation(.12,False);a=v27.infer(o,p,CAL,dict(kind='none'))
    o['evaluator_truth']=7.;o['confidence']=0.
    assert v27.infer(o,p,CAL,dict(kind='none'))==a
    assert 'outside_prior_span' in a['reasons'] and a['height_m'] is not None
    assert not a['qualified'] and not a['confidence_gate_enabled'] and not a['coverage_gate_enabled']

def test_invalid_observations_report_missing_not_fabricated():
    hs=[.24,.64,1.76,3.84,6.08]
    p=v27.fit_prior([observation(h,False) for h in hs],hs,CAL,'transparent',SETTING)
    o=observation(.5,False);o['regions']=0
    r=v27.infer(o,p,CAL,dict(kind='none'))
    assert r['height_m'] is None and not r['finite']

def test_all_layouts_have_identical_five_frame_budget():
    from experiment_transparent_v27 import LAYOUTS
    for values in LAYOUTS.values():
        assert len(values)==5 and len(set(values))==5
        assert all(0<v<.98 for v in values) and values==sorted(values)

def test_missing_frames_remain_in_pass_rate_denominator():
    from experiment_transparent_v27 import summary
    rs=[dict(predicted_height_m=.1,absolute_error_m=0.,true_height_m=.1),
        dict(predicted_height_m=None,absolute_error_m=None,true_height_m=.1)]
    m=summary(rs)
    assert m['frames']==2 and m['outputs']==1 and m['passed']==1
    assert m['pass_pct']==50. and m['numeric_output_pct']==50.
