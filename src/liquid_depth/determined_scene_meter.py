"""Fixed all-scene facade: unchanged geometric liquid, optional V8 float peak."""
from pathlib import Path

from .floating_rank_application_v8 import from_empty_geometry as build_v8
from .supported_scene_meter import FormalSceneMeter
from .runtime_paths import resource_root

OPAQUE_SCENES = frozenset({'ordinary_diffuse', 'partial_depth_loss', 'low_light',
                           'glare', 'floating_objects', 'uneven_surface'})
REFRACTION_SCENES = frozenset({'homogeneous_transparent', 'homogeneous_translucent'})
SCENES = OPAQUE_SCENES | REFRACTION_SCENES
PROFILE = resource_root()/'configs/floating_rank_v8.json'
VERSION = 'determined_all_scenes_v1_v8_peak_20261016'


class DeterminedSceneMeter:
    """Only adapts constructors and metadata; never changes measured values.

    Scene selection is explicit. No autonomous classification, temporal recovery,
    V9 switch, zero snap, or new numerical coverage gate is introduced here.
    """

    def __init__(self, backend, *, kind, refraction_scene=None):
        if kind not in {'empty_geometry', 'five_priors'}:
            raise ValueError('unknown_initialization_kind')
        self.backend = backend
        self.kind = kind
        self.refraction_scene = refraction_scene

    @classmethod
    def from_empty_geometry(cls, bundle, dry_depth_m, *, profile_path=PROFILE,
                            max_return_depth_m=10.5):
        backend = build_v8(bundle, dry_depth_m, profile_path=profile_path,
                           policy_variant='selected', max_return_depth_m=max_return_depth_m)
        return cls(backend, kind='empty_geometry')

    @classmethod
    def from_five_priors(cls, calibration, masks, prior_images, known_heights_m, *, scene):
        backend = FormalSceneMeter.from_five_priors(calibration, masks, prior_images,
                                                    known_heights_m, scene=scene)
        return cls(backend, kind='five_priors', refraction_scene=scene)

    def measure(self, scene, *, depth_m=None, rgb_bgr=None, stereo_images=None,
                visual_enabled=False):
        if scene not in SCENES:
            raise ValueError('explicit_supported_application_scene_required')
        if scene in REFRACTION_SCENES:
            if self.kind != 'five_priors' or scene != self.refraction_scene:
                raise ValueError('scene_specific_five_prior_initialization_required')
            result = self.backend.measure(scene, stereo_images=stereo_images)
            result = {**result, 'visual_module_executed': False,
                      'highest_visual_guidance_used': False, 'liquid_visual_guidance_used': False,
                      'visual_guidance_used': False, 'liquid_override_used': False}
        else:
            if self.kind != 'empty_geometry':
                raise ValueError('empty_geometry_initialization_required')
            result = self.backend.measure(scene, depth_m=depth_m, rgb_bgr=rgb_bgr,
                                          visual_enabled=visual_enabled)
        return {**result, 'determined_version': VERSION, 'selected_visual_backend': 'V8_selected',
                'field_metrology_qualified': False}
