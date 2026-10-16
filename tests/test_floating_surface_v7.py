import numpy as np
import pytest

from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.floating_apex_v5 import candidates_v5, observed_peak_roi
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.floating_surface_v7 import (
    fast_candidates,
    fit_candidate_models,
    fit_patch,
    model_allowed,
    select_models,
)
from liquid_depth.supported_scene_meter import FormalSceneMeter


def base_meter():
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 3.
    bundle = {'source': 'test_empty_geometry', 'sensor_family': 'tof', 'image_size': [80, 60], 'rim_z_m': 1.,
        'calibration': {'intrinsics': [[200., 0., 40.], [0., 200., 30.], [0., 0., 1.]],
            'camera_to_world_cv': pose.tolist(), 'bottom_z_m': 0., 'radii_m': [1., 1.], 'footprint_exponent': 2.}}
    return FormalSceneMeter.from_empty_geometry(bundle, np.full((60, 80), 3., np.float32))


def prediction(mask):
    sem = np.zeros((4, *mask.shape)); sem[2] = 1.; sem[3, mask] = .95
    origin = np.zeros((7, *mask.shape)); origin[2] = .9
    return {'semantic': sem, 'origin': origin, 'reliability': np.ones((1, *mask.shape))}


def test_fast_features_and_masks_match_reference():
    base = base_meter(); rng = np.random.default_rng(71); raw = 2.2+rng.normal(0, .005, (60, 80))
    mask = np.zeros(raw.shape, bool); mask[20:32, 30:42] = True; mask[35:44, 50:60] = True
    rgb = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8); p = prediction(mask); dry = np.full_like(raw, 3.)
    roi = observed_peak_roi(base, raw)
    old = candidates_v5(base, dry, rgb, raw, p, .8, roi); new = fast_candidates(base, dry, rgb, raw, p, .8, roi)
    assert len(old) == len(new)
    for a, b in zip(old, new, strict=True):
        np.testing.assert_array_equal(a['mask'], b['mask'])
        np.testing.assert_allclose(a['features'], b['features'], atol=1e-6, rtol=1e-6)


def test_fourteen_sigma_mismatch_is_not_accepted_as_smooth():
    good = {'valid': True, 'cv_sigma': 14., 'residual_sigma': 12., 'inlier_fraction': 1.,
        'occupied_3px_cells': 8, 'witness_cells': 8}
    assert not model_allowed(good)
    good.update(cv_sigma=.7, residual_sigma=.7)
    assert model_allowed(good)


def test_five_points_or_one_spatial_cell_cannot_validate_high_peak():
    bad = {'valid': True, 'cv_sigma': .2, 'residual_sigma': .2, 'inlier_fraction': 1.,
        'occupied_3px_cells': 1, 'witness_cells': 1}
    assert not model_allowed(bad)
    base = base_meter(); p = np.column_stack([np.arange(5)+30, np.arange(5)+20])
    assert not fit_patch(np.full(5, 1.), p, np.ones(5), height_calibration(base.bundle), base.bundle)['valid']


def test_large_real_height_is_not_clipped_by_rim_or_range():
    base = base_meter(); x, y = np.meshgrid(np.arange(25, 45), np.arange(20, 40)); p = np.column_stack([x.ravel(), y.ravel()])
    model = fit_patch(np.full(len(p), .5), p, np.ones(len(p)), height_calibration(base.bundle), base.bundle)
    assert model_allowed(model) and abs(model['height_m']-2.5) < 1e-8
    assert model['height_m'] > base.bundle['rim_z_m']


def test_guarded_raw_requires_witnesses_for_its_own_height():
    base = base_meter(); raw = np.full((60, 80), 2.2); mask = np.zeros(raw.shape, bool); mask[20:40, 30:50] = True
    c = {'mask': mask, 'features': np.zeros(43), 'object_score': .99, 'pixels': int(mask.sum()), 'corrected_raw_m': 2., 'noise_m': .003}
    models = fit_candidate_models([c], raw, prediction(mask), base, piecewise=False)
    guarded = next(m for m in models if m['kind'] == 'guarded_raw')
    assert guarded['model']['witness_cells'] == 0 and not model_allowed(guarded['model'])


def test_facets_preserve_observed_ridge_top():
    base = base_meter(); raw = np.full((60, 80), 2.5); mask = np.zeros(raw.shape, bool); mask[20:40, 30:50] = True
    yy, xx = np.nonzero(mask); raw[yy, xx] = 3-(.5+.3*(1-abs(xx-40)/10))
    c = {'mask': mask, 'features': np.zeros(43), 'object_score': .99, 'pixels': int(mask.sum()), 'corrected_raw_m': .77, 'noise_m': .003}
    models = fit_candidate_models([c], raw, prediction(mask), base, piecewise=True)
    facets = [m for m in models if m['kind'] == 'local_facet']
    assert facets
    chosen, _ = select_models(facets, {'mode': 'surface', 'max_cv': 3., 'min_cells': 3, 'threshold': .5, 'interval_factor': 2.})
    assert chosen is not None and abs(chosen['height_m']-.8) < .015


def test_model_rejection_does_not_force_zero_or_numeric_value():
    assert select_models([], {'mode': 'surface'}) == (None, [])


@pytest.mark.parametrize('scene', ['ordinary_diffuse', 'partial_depth_loss', 'low_light', 'glare',
    'uneven_surface', 'homogeneous_transparent', 'homogeneous_translucent'])
def test_scope_still_blocks_new_model_for_all_nonfloating_scenes(scene):
    class Legacy:
        def measure(self, *args, **kwargs): return {'liquid_depth_m': .4, 'highest_height_m': .5, 'diagnostics': {}}
    def forbidden(): raise AssertionError('V7 must not load')
    result = FloatingPeakOnlyMeter(Legacy(), forbidden).measure(scene, visual_enabled=True)
    assert result['liquid_depth_m'] == .4 and result['highest_height_m'] == .5
    assert not result['visual_module_executed']
