import copy,inspect
import numpy as np
import pytest
from liquid_depth import transparent_relative_v29 as rel
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom

CAL=dict(bottom_m=8.,fx=450.,cx=240.,cy=160.,container_max_height_m=.2,baseline_m=.055)
HS=[0.,.03,.08,.13,.18]

def pair(ha,hb,index=1.37):
    x,y=np.meshgrid(np.linspace(-.42,.42,12),np.linspace(-.26,.26,8));q=np.c_[x.ravel(),y.ravel()]
    target=inverse_bottom(bottom_map(q,ha,index,CAL['bottom_m']),hb,index,CAL['bottom_m'])
    return dict(anchor_q=q,current_q=target,weights=np.ones(len(q)),regions=len(q),channel='left')

def pairs():return [(a,b,pair(HS[a],HS[b])) for a in range(5) for b in range(a+1,5)]

def test_empty_budget_validation():
    assert np.array_equal(rel.validate_heights(HS,CAL),HS)
    for invalid in ([.01,.03,.08,.13,.18],[0.,.03,.08,.13],[0.,.03,.03,.13,.18],[0.,.03,.08,.13,.21]):
        with pytest.raises(ValueError):rel.validate_heights(invalid,CAL)

@pytest.mark.parametrize('mode',['empty','multi'])
def test_relative_prior_and_unknown_height(mode):
    ps=pairs();prior=rel.fit_relative(ps,HS,CAL,'transparent',mode)
    assert abs(prior['index']-1.37)<1e-5
    out=rel.infer_relative([(a,pair(h,.047)) for a,h in enumerate(HS)],prior,CAL)
    assert abs(out['height_m']-.047)<1e-6
    assert out['qualified'] is False and out['confidence_gate_enabled'] is False

def test_finite_outside_capacity_is_not_clipped():
    prior=rel.fit_relative(pairs(),HS,CAL,'transparent')
    out=rel.infer_relative([(a,pair(h,.24)) for a,h in enumerate(HS)],prior,CAL)
    assert abs(out['height_m']-.24)<1e-6
    assert 'outside_container_capacity' in out['reasons']

def test_translation_nuisance_and_no_mutation():
    p=pair(0.,.12);p['current_q']+=np.array([.1,-.05])/CAL['fx'];before=copy.deepcopy(p)
    residual=rel.relative_residual(p,0.,.12,1.37,CAL)
    assert np.max(np.abs(residual))<1e-9
    for k in ('anchor_q','current_q','weights'):np.testing.assert_array_equal(p[k],before[k])

def test_empty_absolute_prior_supported():
    obs=[]
    for h in HS:
        p=pair(0.,0.);q=p['anchor_q'];world=bottom_map(q,h,1.37,CAL['bottom_m'])
        other=inverse_bottom(world,h,1.37,CAL['bottom_m'],(CAL['baseline_m'],0))
        obs.append(dict(left_q=q,right_q=other,left_pixels=q*CAL['fx']+[240,160],
            right_pixels=other*CAL['fx']+[240,160],weights=np.ones(len(q)),regions=len(q)))
    prior=rel.fit_absolute_empty(obs,HS,CAL,'transparent',0,.02)
    assert abs(prior['index']-1.37)<1e-5
    assert prior['y'][0]==0. and len(prior['used_frames'])==5

def test_no_features_missing_output_not_fabricated():
    blank=np.zeros((80,100),np.uint8)
    p=rel.track_pair(blank,blank,CAL)
    assert p['regions']==0
    out=rel.infer_relative([(0,p)],dict(heights=HS,mode='multi',index=1.37),CAL)
    assert out['height_m'] is None

def test_strong_scattering_excluded():
    with pytest.raises(ValueError):rel.fit_relative(pairs(),HS,CAL,'translucent_strong')

def test_new_splits_and_unknown_levels_disjoint():
    from experiment_relative_v29 import specifications,LAYOUTS
    dev=specifications('development');test=specifications('test')
    for key in ('bottom','capacity','index','seed'):
        assert not ({x[key] for x in dev}&{x[key] for x in test})
    for x in dev+test:
        assert not (set(x['unknown']) & {v for vs in LAYOUTS.values() for v in vs})
    assert all(len(v)==5 for v in LAYOUTS.values())

def test_inference_module_has_no_evaluator_truth_inputs():
    source=inspect.getsource(rel)
    assert 'evaluator_truth' not in source and 'true_height' not in source
