import numpy as np
import pytest
from liquid_depth.unified_tof import ToFInstallation,ToFObservation,make_plan
from liquid_depth.scene_console_web import validate_controls


def install(material='ordinary_diffuse',phase=None):
    return ToFInstallation('test',{'sensor_family':'tof'},np.ones((8,8)),material,phase)


def prediction(material='ordinary_diffuse',tags=(),ambiguous=False):
    return dict(material=material,tags=list(tags),material_ambiguous=ambiguous)


def test_all_opaque_tags_have_only_one_liquid_core():
    plan=make_plan(prediction(tags=['low_light','glare','floating_objects','uneven_surface']),install(),{'container_invalid_fraction':.3})
    assert plan['liquid_route']=='geometric' and plan['visual_peak']
    assert plan['modules'].count('frozen_symmetric_liquid')==1
    assert 'five_prior_amcw_refraction' not in plan['modules']
    assert plan['composition_qualification']=='experimental_holdout_required'


def test_phase_never_falls_back_without_priors():
    plan=make_plan(prediction('homogeneous_transparent'),install('homogeneous_transparent'),{'container_invalid_fraction':.1})
    assert 'five_phase_priors_required_no_geometric_fallback' in plan['blocked_reasons']
    assert 'frozen_symmetric_liquid' not in plan['modules']


def test_phase_float_is_explicitly_unsupported():
    plan=make_plan(prediction('homogeneous_transparent',['floating_objects']),install('homogeneous_transparent',{}),{'container_invalid_fraction':.1})
    assert 'refractive_float_phase_occlusion_not_validated' in plan['blocked_reasons']
    assert not plan['visual_peak']


def test_noisy_image_tags_do_not_override_confirmed_optical_scope():
    prior=install('homogeneous_transparent',{})
    prior.confirmed_refractive_planar_unoccluded=True
    plan=make_plan(prediction('homogeneous_transparent',['uneven_surface']),prior,{'container_invalid_fraction':.1})
    assert plan['liquid_route']=='phase' and not plan['blocked_reasons']
    assert 'rgbd_nonplanarity_suspected_review_confirmed_scope' in plan['warnings']


def test_material_confirmation_and_measurement_confidence_are_distinct():
    plan=make_plan(prediction('homogeneous_translucent',ambiguous=True),install(),{'container_invalid_fraction':.2})
    assert plan['material']=='ordinary_diffuse'
    assert plan['warnings']==['rgb_material_disagrees_with_confirmed_calibration']


def test_ambiguous_unconfirmed_requires_initialization_confirmation():
    plan=make_plan(prediction(ambiguous=True),install(None),{'container_invalid_fraction':0.})
    assert plan['blocked_reasons']==['confirm_liquid_material_once_at_initialization']


def test_even_confident_rgb_cannot_bypass_physical_initialization():
    plan=make_plan(prediction(ambiguous=False),install(None),{'container_invalid_fraction':0.})
    assert 'confirm_liquid_material_once_at_initialization' in plan['blocked_reasons']


def test_uniform_input_allows_missing_optional_phase_but_not_missing_rgb():
    obs=ToFObservation('f',np.zeros((8,8,3),np.uint8),np.ones((8,8)),0.)
    obs.validate((8,8))
    with pytest.raises(ValueError,match='shape_changed'):obs.validate((9,9))


@pytest.mark.parametrize('bad',[{'height_m':float('nan')},{'loss':.51},{'float_count':.2},{'path':'/etc/passwd'},{'wave_m':.1,'height_m':0.}])
def test_controls_reject_invalid_or_nonphysical_inputs(bad):
    with pytest.raises(ValueError):validate_controls(bad)


def test_no_ground_truth_field_in_public_observation():
    with pytest.raises(TypeError):ToFObservation('f',np.zeros((8,8,3),np.uint8),np.ones((8,8)),0.,true_height_m=.2)


def test_preloading_assets_never_measures_an_image():
    from types import SimpleNamespace
    from liquid_depth.unified_tof import UnifiedToFMeter
    sentinel=object();calls=[]
    holder=SimpleNamespace(_float_meter=None,_factory=lambda:calls.append('load') or sentinel)
    meter=object.__new__(UnifiedToFMeter)
    meter.geometric=SimpleNamespace(backend=SimpleNamespace(backend=holder))
    meter.preload_optional_peak_models();meter.preload_optional_peak_models()
    assert calls==['load'] and holder._float_meter is sentinel
