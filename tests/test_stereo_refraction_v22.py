import numpy as np
import pytest
from liquid_depth.stereo_refraction_v22 import apparent_contraction, fit_stereo_prior, infer_stereo_height


def f(h, n=1.38, b=.055):
    q = np.linspace(-.4, .4, 64)
    return dict(qx=q, value_m=apparent_contraction(h, n, q, 1., b)+.001)


@pytest.mark.parametrize('family,b', [('active_stereo', .055), ('structured_light', .035)])
def test_separate_triangulation_prior(family, b):
    hs = [.1, .3, .6]
    prior = fit_stereo_prior([f(h, b=b) for h in hs], hs,
        sensor_family=family, bottom_m=1, baseline_m=b, sigma_m=.0001)
    assert abs(prior.index-1.38) < 1e-4
    result = infer_stereo_height(f(.42, b=b), prior)
    assert result['accepted']
    assert abs(result['height_m']-.42) < 1e-5
    assert not prior.production_enabled


def test_paraxial_limit_and_empty_reference():
    assert apparent_contraction(0, 1.333, [0], 1., .055)[0] == pytest.approx(0, abs=1e-10)
    # Tiny baseline and central ray approach H*(1-1/n), NOT H*(n-1).
    assert apparent_contraction(.3, 1.333, [0], 1., 1e-5)[0] == pytest.approx(.3*(1-1/1.333), abs=1e-8)


def test_repeated_prior_level_rejected():
    with pytest.raises(ValueError, match='diversity'):
        fit_stereo_prior([f(.3)]*3, [.3]*3, sensor_family='active_stereo',
                         bottom_m=1, baseline_m=.055, sigma_m=.001)


def test_tof_not_accepted():
    with pytest.raises(ValueError, match='ToF'):
        fit_stereo_prior([], [], sensor_family='tof', bottom_m=1, baseline_m=.055, sigma_m=.001)
