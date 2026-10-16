import numpy as np
import pytest

from liquid_depth.floating_apex_v5 import candidates_v5, observed_peak_roi, score_v5, select_v5
from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter
from liquid_depth.supported_scene_meter import FormalSceneMeter


def base_meter():
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 6.
    bundle = {'source': 'test_empty_geometry', 'sensor_family': 'tof', 'image_size': [80, 60], 'rim_z_m': 2.,
        'calibration': {'intrinsics': [[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]],
            'camera_to_world_cv': pose.tolist(), 'bottom_z_m': 0., 'radii_m': [1., 1.], 'footprint_exponent': 2.}}
    return FormalSceneMeter.from_empty_geometry(bundle, np.full((60, 80), 6., np.float32))


def candidate(height=.5, correction=-.2, score=.99, mask=None):
    if mask is None: mask = np.ones((4, 4), bool)
    return {'height_raw_m': height, 'correction_m': correction, 'object_score': score,
        'interval_width_m': .01, 'mask': mask, 'pixels': int(mask.sum())}


def policy(fraction=0.):
    return {'threshold': .7, 'interval_factor': 2., 'correction_fraction': fraction, 'consensus_quantile': .5}


def test_shrinkage_only_applies_selected_fraction():
    assert select_v5([candidate()], policy(0.))[0]['height_m'] == .5
    assert select_v5([candidate()], policy(.5))[0]['height_m'] == .4
    assert select_v5([candidate()], policy(1.))[0]['height_m'] == .3


def test_duplicate_masks_use_consensus_not_max_trial():
    chosen, groups = select_v5([candidate(.5), candidate(.6)], policy())
    assert len(groups) == 1 and chosen['proposal_agreement'] == 2
    assert chosen['height_m'] == .55


def test_disjoint_objects_keep_highest_group():
    a = np.zeros((4, 4), bool); a[:, :2] = True
    b = ~a
    chosen, groups = select_v5([candidate(.5, mask=a), candidate(.8, mask=b)], policy())
    assert len(groups) == 2 and chosen['height_m'] == .8


def test_rejection_preserves_no_visual_choice():
    assert select_v5([candidate(score=.1)], policy()) == (None, [])
    assert select_v5([], policy()) == (None, [])


def test_peak_roi_can_include_above_rim_points_without_mutating_liquid_roi():
    base = base_meter(); original = base.ordinary_container_roi.copy()
    raw = np.full((60, 80), 6.); raw[30, 60] = 2.  # world x=.4,z=4; inside vessel projection, above rim=2
    roi = observed_peak_roi(base, raw)
    assert roi[30, 60] and not original[30, 60]
    np.testing.assert_array_equal(original, base.ordinary_container_roi)


def test_missing_depth_does_not_expand_geometry_roi():
    base = base_meter()
    np.testing.assert_array_equal(observed_peak_roi(base, np.zeros((60, 80))), base.ordinary_container_roi)


def test_proposals_are_original_observed_pixels_with_43_features():
    base = base_meter(); raw = np.full((60, 80), 4.6); raw[25:32, 35:42] = 4.35
    sem = np.zeros((4, 60, 80)); sem[2] = 1; sem[3, 25:32, 35:42] = .9
    origin = np.zeros((7, 60, 80)); origin[2] = .8
    prediction = {'semantic': sem, 'origin': origin, 'reliability': np.ones((1, 60, 80))}
    roi = observed_peak_roi(base, raw)
    proposals = candidates_v5(base, np.full_like(raw, 6.), np.zeros((60, 80, 3), np.uint8), raw, prediction, 1.4, roi)
    assert proposals
    for c in proposals:
        assert c['features'].shape == (43,)
        assert (raw[c['mask']] > 0).all()
        assert abs(c['height_raw_m']-1.65) < 1e-10


def test_no_depth_cannot_create_metric_candidate():
    base = base_meter(); raw = np.zeros((60, 80))
    pred = {'semantic': np.ones((4, 60, 80)), 'origin': np.ones((7, 60, 80)), 'reliability': np.ones((1, 60, 80))}
    assert candidates_v5(base, np.ones_like(raw), np.zeros((60, 80, 3), np.uint8), raw, pred, None,
        observed_peak_roi(base, raw)) == []


def test_crossed_old_intervals_remain_positive():
    class Constant:
        def __init__(self, value): self.value = value
        def predict(self, x): return np.full(len(x), self.value)
        def predict_proba(self, x): return np.tile([0., 1.], (len(x), 1))
    c = {**candidate(), 'features': np.zeros(43), 'noise_m': .1}
    previous = {'regressor': Constant(0.), 'lower': Constant(2.), 'upper': Constant(-2.)}
    scored = score_v5([c], Constant(1.), previous)
    assert scored[0]['interval_width_m'] == .4


@pytest.mark.parametrize('scene', ['ordinary_diffuse', 'low_light', 'glare', 'partial_depth_loss',
    'uneven_surface', 'homogeneous_transparent', 'homogeneous_translucent'])
def test_v5_factory_still_confined_by_existing_scene_gate(scene):
    class Legacy:
        def measure(self, *args, **kwargs):
            assert kwargs['visual_mode'] == 'off' and 'rgb_bgr' not in kwargs
            return {'liquid_depth_m': .4, 'highest_height_m': .4, 'diagnostics': {}}
    def forbidden(): raise AssertionError('must not load V5')
    r = FloatingPeakOnlyMeter(Legacy(), forbidden).measure(scene, depth_m=np.ones((60, 80)), visual_enabled=True)
    assert r['liquid_depth_m'] == .4 and not r['visual_module_executed']
