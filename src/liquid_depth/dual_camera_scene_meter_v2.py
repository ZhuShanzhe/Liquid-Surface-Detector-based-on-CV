"""Formal dual-mechanism facade. The former all-scene facade stays frozen."""
from .determined_scene_meter import DeterminedSceneMeter,OPAQUE_SCENES,REFRACTION_SCENES,SCENES
from .phase_refraction_v2 import PhasePriorMeter

VERSION='dual_camera_v2_20261017'
SENSORS=frozenset({'tof','structured_light'})


class DualCameraSceneMeter:
    def __init__(self,backend,*,sensor,kind,scene=None):
        if sensor not in SENSORS:raise ValueError('dual_release_only_tof_or_structured_light')
        self.backend=backend;self.sensor=sensor;self.kind=kind;self.scene=scene

    @classmethod
    def from_empty_geometry(cls,bundle,dry_depth_m,**kwargs):
        sensor=bundle['sensor_family']
        if sensor not in SENSORS:raise ValueError('dual_release_only_tof_or_structured_light')
        return cls(DeterminedSceneMeter.from_empty_geometry(bundle,dry_depth_m,**kwargs),sensor=sensor,kind='empty_geometry')

    @classmethod
    def from_phase_priors(cls,*,sensor,scene,profile,ray_coordinates,roi,prior_correlations,known_heights_m):
        backend=PhasePriorMeter.fit(sensor,scene,profile,ray_coordinates,roi,prior_correlations,known_heights_m)
        return cls(backend,sensor=sensor,kind='five_phase_priors',scene=scene)

    def measure(self,scene,*,depth_m=None,rgb_bgr=None,correlations=None,visual_enabled=False):
        if scene not in SCENES:raise ValueError('unsupported_scene')
        if scene in REFRACTION_SCENES:
            if self.kind!='five_phase_priors' or self.scene!=scene:raise ValueError('matching_five_phase_priors_required')
            if correlations is None:raise ValueError('raw_tof_or_projector_phase_correlations_required_not_depth_only')
            value=self.backend.measure(correlations)
        else:
            if self.kind!='empty_geometry':raise ValueError('empty_geometry_required')
            value=self.backend.measure(scene,depth_m=depth_m,rgb_bgr=rgb_bgr,visual_enabled=visual_enabled)
        return {**value,'release_version':VERSION,'sensor_family':self.sensor,'field_metrology_qualified':False}
