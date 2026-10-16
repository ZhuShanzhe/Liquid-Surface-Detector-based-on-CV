import numpy as np
import pytest

from liquid_depth.floating_peak_only_v4 import APPLICATION_SCENES, FloatingPeakOnlyMeter
from liquid_depth.floating_rank_v8 import score_candidates, select_ranked


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class ConstantClassifier:
    def predict_proba(self, x):
        return np.tile([.01, .99], (len(x), 1))


def candidate(height, offset=0, risk=.01, gap=0., size=12):
    indices = np.arange(size)+offset
    return {'indices': indices, 'pixels': size, 'height_m': height, 'height_raw_m': height,
        'noise_m': .01, 'physical_score': .99, 'object_score': .99, 'risk_m': risk,
        'top_gap_m': gap, 'cells': 4, 'bbox_xyxy_px': [0, 0, size, 3]}


def policy(**extra):
    return {'min_pixels': 4, 'min_cells': 2, 'physical_threshold': .9, 'max_risk_m': .25,
        'risk_weight': .5, 'gap_weight': 1., **extra}


def test_highest_instance_is_not_largest_semantic_mass():
    lower = candidate(1., size=400)
    higher = candidate(1.3, offset=800, size=8)
    chosen, _ = select_ranked([lower, higher], policy())
    assert chosen['height_m'] == 1.3


def test_risk_can_prevent_a_noisy_high_trial_from_winning():
    high = candidate(1.2, risk=.24)
    supported = candidate(1.15, offset=100, risk=.01)
    chosen, _ = select_ranked([high, supported], policy())
    assert chosen['height_m'] == 1.15


def test_nested_trials_do_not_maximize_the_same_instance():
    a = candidate(1.0)
    b = candidate(1.2)
    chosen, instances = select_ranked([a, b], policy())
    assert len(instances) == 1 and chosen['height_m'] == 1.1


def test_one_cell_peak_is_rejected_without_zero_output_substitution():
    bad = candidate(8., size=5)
    bad['cells'] = 1
    chosen, instances = select_ranked([bad], policy())
    assert chosen is None and not instances


def test_no_range_or_rim_clipping_and_risk_does_not_replace_reported_height():
    chosen, _ = select_ranked([candidate(10.2, risk=.2)], policy())
    assert chosen['height_m'] == 10.2 and chosen['utility'] < chosen['height_m']


def test_continuous_correction_does_not_snap_negative_height_to_zero():
    c = candidate(-.03)
    assert 'mask' not in c  # Compact training caches intentionally omit image masks.
    bundle = {'object_classifier': ConstantClassifier(), 'correction': ConstantRegressor(-2.),
        'risk': ConstantRegressor(.5), 'top_gap': ConstantRegressor(0.)}
    result = score_candidates([c], np.zeros((1, 67)), bundle)[0]
    assert result['height_m'] == pytest.approx(-.05)
    assert result['risk_m'] == pytest.approx(.005)


@pytest.mark.parametrize('scene', sorted(APPLICATION_SCENES-{'floating_objects'}))
def test_other_scenes_never_instantiate_or_receive_vision(scene):
    class Legacy:
        def measure(self, name, **kwargs):
            assert kwargs['visual_mode'] == 'off' and 'rgb_bgr' not in kwargs
            return {'liquid_depth_m': .42, 'highest_height_m': .5, 'diagnostics': {}}

    def forbidden():
        raise AssertionError('model must not load')
    result = FloatingPeakOnlyMeter(Legacy(), forbidden).measure(scene, depth_m=np.ones((4, 4)), visual_enabled=True)
    assert result['liquid_depth_m'] == .42 and not result['visual_module_executed']


def test_untrusted_auxiliary_cannot_override_liquid_depth():
    class Legacy:
        def measure(self, name, **kwargs):
            return {'liquid_depth_m': .42, 'highest_height_m': .5, 'diagnostics': {}}

    class Aux:
        def measure(self, **kwargs):
            return {'liquid_depth_m': 99., 'highest_height_m': .9, 'visual_guidance_used': True, 'diagnostics': {}}
    result = FloatingPeakOnlyMeter(Legacy(), Aux).measure('floating_objects', depth_m=np.ones((4, 4)), rgb_bgr=np.zeros((4, 4, 3)), visual_enabled=True)
    assert result['liquid_depth_m'] == .42 and result['highest_height_m'] == .9
