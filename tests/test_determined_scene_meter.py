import numpy as np
import pytest

from liquid_depth import determined_scene_meter as route


class Backend:
    def __init__(self):
        self.calls = []

    def measure(self, scene, **kwargs):
        self.calls.append((scene, kwargs))
        return {'liquid_depth_m': -.013, 'highest_height_m': 12.1, 'numeric_output': True}


@pytest.mark.parametrize('scene', sorted(route.OPAQUE_SCENES))
def test_facade_preserves_existing_values(scene):
    backend = Backend()
    result = route.DeterminedSceneMeter(backend, kind='empty_geometry').measure(scene, depth_m=np.ones((2, 2)))
    assert result['liquid_depth_m'] == -.013
    assert result['highest_height_m'] == 12.1
    assert not backend.calls[0][1]['visual_enabled']


def test_pins_selected_v8_policy(monkeypatch):
    calls = []
    monkeypatch.setattr(route, 'build_v8', lambda *a, **kw: calls.append(kw) or Backend())
    route.DeterminedSceneMeter.from_empty_geometry({}, np.ones((2, 2)))
    assert calls[0]['policy_variant'] == 'selected'
    assert calls[0]['max_return_depth_m'] == 10.5


@pytest.mark.parametrize('scene', sorted(route.REFRACTION_SCENES))
def test_refraction_never_receives_visual_or_depth(scene):
    backend = Backend()
    meter = route.DeterminedSceneMeter(backend, kind='five_priors', refraction_scene=scene)
    frame = {'left': np.ones((2, 2)), 'right': np.ones((2, 2))}
    result = meter.measure(scene, stereo_images=frame, visual_enabled=True, depth_m=np.ones((2, 2)))
    assert backend.calls[0][1] == {'stereo_images': frame}
    assert not result['visual_module_executed']
    assert result['liquid_depth_m'] == -.013


def test_wrong_initialization_and_unknown_scene_rejected():
    meter = route.DeterminedSceneMeter(Backend(), kind='empty_geometry')
    with pytest.raises(ValueError, match='five_prior'):
        meter.measure('homogeneous_transparent')
    with pytest.raises(ValueError, match='explicit_supported'):
        meter.measure('multilayer_glare_compound')
    meter = route.DeterminedSceneMeter(Backend(), kind='five_priors', refraction_scene='homogeneous_transparent')
    with pytest.raises(ValueError, match='empty_geometry'):
        meter.measure('ordinary_diffuse')
    with pytest.raises(ValueError, match='scene_specific'):
        meter.measure('homogeneous_translucent')
