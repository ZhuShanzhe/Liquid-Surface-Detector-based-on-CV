import copy
import numpy as np
import pytest
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth.scene_boundaries import resolve_measurement_route
from liquid_depth.scenario_policy import ComplexScenePolicy,SceneSignals

CAL=dict(fx=450.,bottom_m=1.,baseline_m=.055)
def observation(h):
    x,y=np.meshgrid(np.linspace(-.35,.35,22),np.linspace(-.22,.22,16))
    q=np.column_stack((x.ravel(),y.ravel()))
    r=inverse_bottom(bottom_map(q,h,1.42,1.),h,1.42,1.,(.055,0))
    return dict(left_q=q,right_q=r,weights=np.ones(len(q)),left_pixels=q*450+[240,160],regions=len(q))

@pytest.mark.parametrize('mode',['none','affine','pchip'])
def test_unfiltered_known_level_calibration(mode):
    hs=[.1,.22,.35,.5,.65]
    p=v25.fit_prior([observation(h) for h in hs],hs,CAL,'transparent',mode)
    for h in [.08,.15,.28,.42,.58,.72]:
        r=v25.infer(observation(h),p,CAL)
        assert abs(r['height_m']-h)<.001
        assert r['finite'] and not r['qualified']
        if h in [.08,.72]:assert 'outside_prior_span' in r['reasons']

def test_truth_invariance_and_scope():
    hs=[.1,.35,.65];obs=[observation(h) for h in hs]
    with pytest.raises(ValueError,match='scope'):v25.fit_prior(obs,hs,CAL,'translucent_strong')
    p=v25.fit_prior(obs,hs,CAL,'translucent_mild')
    o=observation(.42);a=v25.infer(o,p,CAL)
    o['true_height_m']=9.;o['index']=1.2
    assert v25.infer(o,p,CAL)==a
    o['regions']=0;assert v25.infer(o,p,CAL)['height_m'] is None

def test_invalid_prior_labels():
    hs=[.1,.35,.65];obs=[observation(h) for h in hs]
    with pytest.raises(ValueError):v25.fit_prior(obs,[.1,float('nan'),.65],CAL,'transparent')

def test_strong_scatter_surface_boundary():
    ctx=dict(measurement_route='surface',liquid_optics='strong_scattering')
    assert not resolve_measurement_route(ctx)['allowed']
    assert resolve_measurement_route(dict(ctx,surface_return_verified=True))['allowed']
    assert not resolve_measurement_route(dict(ctx,measurement_route='homogeneous_refraction'))['allowed']

def test_unintegrated_research_never_certified():
    ctx=dict(measurement_route='homogeneous_refraction',liquid_optics='transparent',bottom_pattern_visible=True)
    assert not resolve_measurement_route(ctx)['allowed']
    assert 'research_route_not_integrated' in resolve_measurement_route(ctx)['reasons']

def test_explicit_surface_does_not_trigger_transparent_and_resets_hold():
    p=ComplexScenePolicy(dict(mode='auto'),available_variants=['transparent_multilayer'])
    signals=SceneSignals(.9,.5,.1,0.,.4)
    assert p.decide(signals,dict(transparent_liquid=True)).activated
    d=p.decide(signals,dict(measurement_route='surface',liquid_optics='strong_scattering',
                          transparent_liquid=True,surface_return_verified=True))
    assert not d.requested and not d.held_by_hysteresis and d.result_allowed
    d=p.decide(signals,dict(measurement_route='surface',surface_return_verified=False))
    assert not d.result_allowed
    assert d.rejection_reasons()==['surface_return_not_verified']

def test_modifiers_not_suppressed_by_surface_route():
    p=ComplexScenePolicy(dict(mode='auto'),available_variants=['low_light'])
    d=p.decide(SceneSignals(.9,.1,.8,0.,.4),dict(measurement_route='surface',surface_return_verified=True))
    assert d.model_variant=='low_light'
