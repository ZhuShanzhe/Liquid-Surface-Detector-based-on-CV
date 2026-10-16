import numpy as np
import pytest

from liquid_depth.floating_peak_only_v4 import FloatingPeakOnlyMeter


class Legacy:
    def __init__(self): self.calls = []
    def measure(self, scene, **kwargs):
        self.calls.append((scene, kwargs))
        return {'scene': scene, 'liquid_depth_m': .4, 'highest_height_m': .45,
            'numeric_output': True, 'diagnostics': {'legacy_field': 1}}


class Auxiliary:
    def __init__(self): self.calls = 0
    def measure(self, **kwargs):
        self.calls += 1
        return {'liquid_depth_m': 999., 'numeric_output': False, 'highest_height_m': .7,
            'visual_guidance_used': True, 'diagnostics': {'verified_regions': [{'pixels': 8}]}}


@pytest.mark.parametrize('scene', ['ordinary_diffuse', 'low_light', 'glare', 'uneven_surface',
    'partial_depth_loss', 'homogeneous_transparent', 'homogeneous_translucent'])
def test_nonfloating_request_cannot_load_or_invoke_visual(scene):
    def forbidden(): raise AssertionError('visual model must not load')
    legacy = Legacy(); meter = FloatingPeakOnlyMeter(legacy, forbidden)
    result = meter.measure(scene, depth_m=np.ones((5, 5)), visual_enabled=True)
    assert not result['visual_module_executed']
    assert not result['highest_visual_guidance_used']
    assert result['highest_height_m'] == .45
    assert not result['highest_visual_guidance_used']
    assert result['visual_application_policy']['disabled_reason'] == 'scene_not_floating_objects'
    assert legacy.calls[0][1]['visual_mode'] == 'off'
    assert 'rgb_bgr' not in legacy.calls[0][1]


def test_floating_still_requires_operator_opt_in():
    def forbidden(): raise AssertionError('default must be visual off')
    result = FloatingPeakOnlyMeter(Legacy(), forbidden).measure('floating_objects', depth_m=np.ones((5, 5)))
    assert not result['visual_module_executed']
    assert result['highest_height_m'] == .45


def test_auxiliary_can_only_change_highest_height():
    auxiliary = Auxiliary(); meter = FloatingPeakOnlyMeter(Legacy(), lambda: auxiliary)
    result = meter.measure('floating_objects', depth_m=np.ones((5, 5)), rgb_bgr=np.zeros((5, 5, 3)), visual_enabled=True)
    assert auxiliary.calls == 1
    assert result['highest_height_m'] == .7
    assert result['liquid_depth_m'] == .4
    assert result['numeric_output']
    assert not result['liquid_override_used']
    assert not result['liquid_visual_guidance_used']


def test_loaded_float_model_is_never_called_after_switching_scene():
    auxiliary = Auxiliary(); loads = []
    def factory(): loads.append(1); return auxiliary
    meter = FloatingPeakOnlyMeter(Legacy(), factory)
    meter.measure('floating_objects', depth_m=np.ones((5, 5)), rgb_bgr=np.zeros((5, 5, 3)), visual_enabled=True)
    meter.measure('glare', depth_m=np.ones((5, 5)), visual_enabled=True)
    meter.measure('floating_objects', depth_m=np.ones((5, 5)), rgb_bgr=np.zeros((5, 5, 3)), visual_enabled=True)
    assert loads == [1]
    assert auxiliary.calls == 2


def test_missing_rgb_does_not_load_weights():
    def forbidden(): raise AssertionError('must validate RGB first')
    with pytest.raises(ValueError, match='aligned_RGB'):
        FloatingPeakOnlyMeter(Legacy(), forbidden).measure('floating_objects', visual_enabled=True)


def test_unknown_scene_cannot_fall_back_to_floating():
    legacy = Legacy()
    with pytest.raises(ValueError, match='explicit_supported'):
        FloatingPeakOnlyMeter(legacy).measure('floating_objects_glare_compound', visual_enabled=True)
    assert legacy.calls == []


def test_transparent_route_keeps_its_stereo_inputs_but_never_visual():
    legacy = Legacy(); frames = {'left': 'left', 'right': 'right'}
    result = FloatingPeakOnlyMeter(legacy).measure('homogeneous_transparent', stereo_images=frames, visual_enabled=True)
    assert legacy.calls[0][1]['stereo_images'] is frames
    assert not result['visual_module_executed']


def test_nonfinite_auxiliary_peak_preserves_legacy_output():
    class Invalid(Auxiliary):
        def measure(self, **kwargs):
            value = super().measure(**kwargs); value['highest_height_m'] = float('nan'); return value
    result = FloatingPeakOnlyMeter(Legacy(), Invalid).measure('floating_objects', rgb_bgr=np.zeros((5, 5, 3)), visual_enabled=True)
    assert result['highest_height_m'] == .45
    assert not result['highest_visual_guidance_used']
