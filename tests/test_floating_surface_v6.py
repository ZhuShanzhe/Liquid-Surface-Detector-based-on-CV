import numpy as np
import pytest

from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_surface_v6 import fit_observed_surface
from liquid_depth.liquid_height_metrics import HeightCalibration


def calibration():
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 3.
    return HeightCalibration(np.array([[200., 0., 50.], [0., 200., 50.], [0., 0., 1.]]), pose, 0., (1., 1.))


def sample():
    x, y = np.meshgrid(np.linspace(30, 70, 12), np.linspace(30, 70, 12))
    return np.column_stack([x.ravel(), y.ravel()])


def test_flat_surface_with_wrong_near_returns():
    rng = np.random.default_rng(61); pixels = sample(); raw = 2.2+rng.normal(0, .002, len(pixels))
    raw[::10] -= 1.2
    result = fit_observed_surface(raw, pixels, np.ones(len(raw)), calibration(), 'tof')
    assert result['valid'] and abs(result['height_m']-.8) < .015
    assert result['rejected_observed_points'] >= 10


def test_curved_top_is_not_forced_to_plane():
    pixels = sample(); uv = (pixels-50)/20
    height = .8+.15*(1-(uv*uv).sum(1)); raw = 3-height
    result = fit_observed_surface(raw, pixels, np.ones(len(raw)), calibration(), 'tof')
    assert result['valid'] and result['degree'] == 2
    assert abs(result['height_m']-.95) < .015


@pytest.mark.parametrize('n', [0, 1, 5, 11])
def test_smooth_appearance_does_not_make_five_points_sufficient(n):
    result = fit_observed_surface(np.ones(n), sample()[:n], np.ones(n), calibration(), 'tof')
    assert not result['valid']


def test_collinear_support_is_not_surface_evidence():
    points = np.column_stack([np.arange(30)+20, np.full(30, 50.)])
    assert not fit_observed_surface(np.full(30, 2.), points, np.ones(30), calibration(), 'tof')['valid']


def test_dense_queries_do_not_increase_observed_point_count():
    pixels = sample(); result = fit_observed_surface(np.full(len(pixels), 2.), pixels, np.ones(len(pixels)), calibration(), 'tof')
    assert result['valid'] and result['inferred_query_count'] > 0
    assert result['observed_points'] == len(pixels)
    assert result['effective_weighted_points'] <= len(pixels)+1e-6


def test_stereo_uses_inverse_depth_not_height_quantile():
    pixels = sample(); raw = np.full(len(pixels), 2.)
    result = fit_observed_surface(raw, pixels, np.ones(len(raw)), calibration(), 'structured_light')
    assert result['valid'] and result['inverse_depth_domain']
    assert abs(result['height_m']-1.) < 1e-8


def test_no_zero_snap_and_no_rim_clipping():
    pixels = sample(); result = fit_observed_surface(np.full(len(pixels), 3.05), pixels, np.ones(len(pixels)), calibration(), 'tof')
    assert result['valid'] and result['height_m'] < -.049


@pytest.mark.parametrize('scene', ['ordinary_diffuse', 'glare', 'low_light', 'partial_depth_loss',
    'uneven_surface', 'homogeneous_transparent', 'homogeneous_translucent'])
def test_surface_factory_cannot_run_outside_float_highest(scene):
    class Legacy:
        def measure(self, *args, **kwargs): return {'liquid_depth_m': .3, 'highest_height_m': .4, 'diagnostics': {}}
    def forbidden(): raise AssertionError('surface module must not load')
    result = FloatingPeakOnlyMeter(Legacy(), forbidden).measure(scene, visual_enabled=True)
    assert result['highest_height_m'] == .4 and result['liquid_depth_m'] == .3
    assert not result['visual_module_executed']
