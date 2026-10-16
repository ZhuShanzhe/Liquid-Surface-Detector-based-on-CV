import numpy as np
import pytest

from liquid_depth.floating_joint_v9 import (
    CausalSupport,
    choose_supported,
    instance_candidates,
    support_check,
    weak_eligible,
)
from liquid_depth.floating_peak_only_v4 import APPLICATION_SCENES, FloatingPeakOnlyMeter


def candidate(*, pixels=6, height=1., identity=.999, risk=.01, offset=0):
    f = np.zeros(43)
    f[[18, 20, 23]] = .95
    return {'pixels': pixels, 'indices': np.arange(pixels)+offset, 'height_m': height, 'risk_m': risk,
        'conditional_risk_m': .001, 'physical_score': identity, 'top_gap_m': 0., 'features': f,
        'cells': 3, 'bbox_xyxy_px': [0, 0, 3, 3], 'utility': height-risk}


def policy(**extra):
    return {'identity_threshold': .8, 'risk_cap_m': .25, 'risk_weight': .5, 'confirm_frames': 2, **extra}


def test_v8_six_point_high_confidence_peak_cannot_pass_single_frame_hard_support():
    c = candidate(identity=.9999, height=4.66)
    chosen, _, _ = choose_supported([c], policy(), lambda _: {'valid': True})
    assert chosen is None


def test_joint_risk_not_conditional_risk_controls_selection():
    c = candidate(pixels=24, risk=1.2)
    assert instance_candidates([c], policy()) == []


def test_weak_support_diagnostics_are_native_json_scalars():
    import json
    count = sum(weak_eligible(c, policy()) for c in [candidate(), candidate(pixels=20)])
    assert type(count) is int
    assert json.loads(json.dumps({'weak_eligible_instances': count}))['weak_eligible_instances'] == 1


def test_small_current_object_needs_distinct_past_depth_frames():
    memory = CausalSupport()
    raw = np.ones((8, 8))
    c = candidate()
    valid, _ = memory.begin(('installation_a', 0, 0.), raw)
    assert choose_supported([c], policy(), lambda _: {'valid': True}, memory, valid)[0] is None
    raw[7, 7] += .001
    valid, _ = memory.begin(('installation_a', 1, .1), raw)
    chosen = choose_supported([c], policy(), lambda _: {'valid': True}, memory, valid)[0]
    assert chosen['support_kind'] == 'causal_small_object' and chosen['confirmations'] == 2


def test_duplicate_depth_buffer_does_not_create_confirmation():
    memory = CausalSupport()
    raw = np.ones((8, 8))
    for frame in range(4):
        valid, _ = memory.begin(('a', frame, frame*.1), raw)
        assert choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)[0] is None


@pytest.mark.parametrize('context', [('b', 1, .1), ('a', 0, .1), ('a', 1, 2.), None])
def test_stream_change_repeat_gap_or_missing_context_resets_confirmation(context):
    memory = CausalSupport()
    raw = np.ones((8, 8))
    valid, _ = memory.begin(('a', 0, 0.), raw)
    choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)
    raw[7, 7] += .001
    valid, _ = memory.begin(context, raw)
    assert choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)[0] is None


def test_history_without_current_object_does_not_hold_old_highest_height():
    memory = CausalSupport()
    raw = np.ones((8, 8))
    valid, _ = memory.begin(('a', 0, 0.), raw)
    choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)
    raw[0, 0] += .001
    valid, _ = memory.begin(('a', 1, .1), raw)
    assert choose_supported([], policy(), lambda _: {'valid': True}, memory, valid)[0] is None


def test_old_history_is_not_reused_beyond_half_second():
    memory = CausalSupport()
    raw = np.ones((8, 8))
    valid, _ = memory.begin(('a', 0, 0.), raw)
    choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)
    raw[0, 0] += .001
    valid, _ = memory.begin(('a', 1, .4), raw)
    choose_supported([], policy(), lambda _: {'valid': True}, memory, valid)
    raw[0, 0] += .001
    valid, _ = memory.begin(('a', 2, .8), raw)
    assert choose_supported([candidate()], policy(), lambda _: {'valid': True}, memory, valid)[0] is None


def test_large_mismatched_region_cannot_use_small_object_escape():
    chosen = choose_supported([candidate(pixels=100)], policy(), lambda _: {'valid': False})[0]
    assert chosen is None


def test_no_range_rim_clipping_or_zero_snap_of_supported_height():
    c = candidate(pixels=20, height=10.2)
    chosen = choose_supported([c], policy(), lambda _: {'valid': True})[0]
    assert chosen['height_m'] == 10.2


def test_spatial_support_rejects_thin_six_point_patch_before_fitting():
    c = candidate(pixels=6)
    result = support_check(c, np.ones((4, 4)), None, {})
    assert not result['valid'] and result['reason'] == 'hard_spatial_support'


@pytest.mark.parametrize('scene', sorted(APPLICATION_SCENES-{'floating_objects'}))
def test_other_scenes_cannot_load_or_use_peak_vision(scene):
    class Legacy:
        def measure(self, name, **kwargs):
            assert kwargs['visual_mode'] == 'off' and 'rgb_bgr' not in kwargs
            return {'liquid_depth_m': .3, 'highest_height_m': .4, 'diagnostics': {}}
    def forbidden():
        raise AssertionError('wrong scene model load')
    result = FloatingPeakOnlyMeter(Legacy(), forbidden).measure(scene, depth_m=np.ones((3, 3)), visual_enabled=True)
    assert result['liquid_depth_m'] == .3 and not result['visual_module_executed']
