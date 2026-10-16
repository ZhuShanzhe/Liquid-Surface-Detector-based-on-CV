import numpy as np
import pytest
from liquid_depth.homogeneous_refraction_v22 import coefficient, fit_material_prior, infer_height


def frame(h, n=1.38, bias=.001, model='nondispersive_coaxial_tof_proxy'):
    q = np.linspace(.02, .3, 64)
    return dict(q2=q, value_m=h*coefficient(n, q, model)+bias)


@pytest.mark.parametrize('model', ['nondispersive_coaxial_tof_proxy', 'rgb_reference_snell'])
def test_known_prior_unknown_height(model):
    hs = [.1, .3, .6]
    prior = fit_material_prior([frame(h, model=model) for h in hs], hs,
        sensor_model=model, bottom_distance_m=1, observation_sigma_m=.0001)
    assert abs(prior.index-1.38) < 1e-4
    assert abs(prior.bias_m-.001) < 1e-5
    result = infer_height(frame(.42, model=model), prior)
    assert result['accepted']
    assert abs(result['height_m']-.42) < 1e-5
    assert not prior.production_enabled


def test_repeated_level_is_not_identifiable():
    with pytest.raises(ValueError, match='height span'):
        fit_material_prior([frame(.3)]*3, [.3]*3,
            sensor_model='nondispersive_coaxial_tof_proxy', bottom_distance_m=1)


def test_stereo_is_not_tof():
    with pytest.raises(ValueError, match='Unsupported'):
        coefficient(1.33, [0], 'active_stereo')


def test_normal_incidence_formula():
    assert coefficient(1.33, [0], 'nondispersive_coaxial_tof_proxy')[0] == pytest.approx(.33)
    assert coefficient(1.33, [0], 'rgb_reference_snell')[0] == pytest.approx(1-1/1.33)


def test_no_bottom_rejects():
    hs = [.1, .3, .6]
    prior = fit_material_prior([frame(h) for h in hs], hs,
        sensor_model='nondispersive_coaxial_tof_proxy', bottom_distance_m=1)
    result = infer_height(dict(q2=np.linspace(.02, .3, 64), value_m=np.full(64, np.nan)), prior)
    assert not result['accepted']


def test_current_labels_do_not_enter_inverse():
    hs = [.1, .3, .6]
    prior = fit_material_prior([frame(h) for h in hs], hs,
        sensor_model='nondispersive_coaxial_tof_proxy', bottom_distance_m=1)
    observed = frame(.4)
    first = infer_height(observed, prior)
    observed.update(ground_truth_height=9, event_mask=np.zeros(64), refractive_index=1.5)
    assert infer_height(observed, prior) == first
