import numpy as np
import pytest
from capacity_priors_v28 import LAYOUTS,validate_layout,specifications,result_rows,summary
from audit_capacity_v28 import oracle_observation
from liquid_depth import transparent_diagnostic_v25 as base

def test_five_prior_budget_never_exceeds_capacity():
    for cap in (.15,.20,.35,.50,.75):
        for b in (2.2,3.,6.5,8.5,9.8):
            for values in LAYOUTS.values():
                hs=validate_layout(values,cap,b)
                assert len(hs)==5 and max(hs)<cap and min(hs)>0 and np.ptp(hs)>.03

def test_invalid_or_duplicate_capacity_levels_are_explicit():
    for values in ([.1,.2,.3,.4],[.1,.1,.3,.5,.9],[.1,.2,.3,.5,1.1]):
        with pytest.raises(ValueError):validate_layout(values,.2,8.)
    with pytest.raises(ValueError):validate_layout(LAYOUTS['capacity_uniform'],9.,8.)

def test_holdout_specifications_and_seeds_are_new():
    dev=specifications('development');test=specifications('test')
    assert len(dev)==32 and len(test)==72
    for key in ('bottom','capacity','index','seed'):
        assert not set(j[key] for j in dev)&set(j[key] for j in test)
    for job in dev+test:
        assert max(job['unknown'])<1 and min(job['unknown'])>0
        assert job['capacity']<job['bottom']

def test_out_of_capacity_prediction_remains_visible(tmp_path):
    p=tmp_path/'truth.npz';np.savez(p,height_m=.1,index=1.4)
    c=dict(id='toy',family='active_stereo',condition='transparent',camera_to_bottom_m=9.8,
        container_max_height_m=.15,tests=[dict(evaluator_truth=str(p))])
    r=result_rows(c,[dict(height_m=.25,reasons=[],qualified=False)],'test','capacity_uniform')[0]
    assert r['predicted_height_m']==.25 and abs(r['absolute_error_m']-.15)<1e-12
    assert 'outside_container_capacity' in r['diagnostics']['reasons']
    assert r['tolerance_m']==.005 and not r['qualified']

def test_no_output_is_counted_in_denominator():
    rs=[dict(predicted_height_m=.1,absolute_error_m=0.,true_height_m=.1),
        dict(predicted_height_m=None,absolute_error_m=None,true_height_m=.1)]
    s=summary(rs)
    assert s['frames']==2 and s['pass_pct']==50. and s['numeric_output_pct']==50.

def test_oracle_correspondence_is_geometry_consistent_and_does_not_mutate_input():
    q=np.array([[-.2,-.1],[.1,.2],[.3,-.2]])
    o=dict(left_q=q,right_q=q.copy(),right_pixels=q*450+[256,192])
    cal=dict(fx=450.,bottom_m=9.8,baseline_m=.035,reference_cx=256.,reference_cy=192.)
    clean=oracle_observation(o,.12,1.42,cal)
    assert np.max(abs(base.pixel_residual(.12,1.42,0.,clean,cal)))<1e-10
    assert np.array_equal(o['right_q'],q)
