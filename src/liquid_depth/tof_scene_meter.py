"""Formal ToF v2: calibrated compound evidence, highest-only optional vision."""
from .determined_scene_meter import DeterminedSceneMeter,OPAQUE_SCENES,REFRACTION_SCENES,SCENES
from .tof_phase_refraction import PhasePriorMeter

VERSION='tof_formal_v2_20261020'
SENSORS=frozenset({'tof'})


class ToFSceneMeter:
    def __init__(self,backend,*,sensor,kind,scene=None,repair=None):
        if sensor not in SENSORS:raise ValueError('formal_release_only_tof_supported')
        self.backend=backend;self.sensor=sensor;self.kind=kind;self.scene=scene
        self.repair=repair

    @classmethod
    def from_empty_geometry(cls,bundle,dry_depth_m,*,evidence_repair=True,repair_policy=None,**kwargs):
        sensor=bundle['sensor_family']
        if sensor not in SENSORS:raise ValueError('formal_release_only_tof_supported')
        backend=DeterminedSceneMeter.from_empty_geometry(bundle,dry_depth_m,**kwargs)
        repair=None
        if evidence_repair:
            from .supported_scene_meter import FormalSceneMeter
            from .compound_tof_repair import CompoundToFRepair
            repair=CompoundToFRepair(FormalSceneMeter.from_empty_geometry(bundle,dry_depth_m),dry_depth_m,repair_policy)
        return cls(backend,sensor=sensor,kind='empty_geometry',repair=repair)

    @classmethod
    def from_phase_priors(cls,*,sensor,scene,profile,ray_coordinates,roi,prior_correlations,known_heights_m):
        if sensor not in SENSORS:raise ValueError('formal_release_only_tof_supported')
        backend=PhasePriorMeter.fit(sensor,scene,profile,ray_coordinates,roi,prior_correlations,known_heights_m)
        return cls(backend,sensor=sensor,kind='five_phase_priors',scene=scene)

    def measure(self,scene,*,depth_m=None,rgb_bgr=None,correlations=None,visual_enabled=False,capture_validity=None):
        if scene not in SCENES:raise ValueError('unsupported_scene')
        if scene in REFRACTION_SCENES:
            if self.kind!='five_phase_priors' or self.scene!=scene:raise ValueError('matching_five_phase_priors_required')
            if correlations is None:raise ValueError('raw_tof_phase_correlations_required_not_depth_only')
            value=self.backend.measure(correlations)
        else:
            if self.kind!='empty_geometry':raise ValueError('empty_geometry_required')
            if self.repair is None:
                value=self.backend.measure(scene,depth_m=depth_m,rgb_bgr=rgb_bgr,visual_enabled=visual_enabled)
            else:
                from .compound_tof_repair import sanitize_capture
                depth,audit=sanitize_capture(depth_m,self.repair.policy.minimum_sensor_depth_m,capture_validity)
                use_visual=scene=='floating_objects' and bool(visual_enabled)
                value=self.backend.measure(scene,depth_m=depth,rgb_bgr=rgb_bgr,visual_enabled=False)
                holder=self.backend.backend
                if use_visual and holder._float_meter is None:holder._float_meter=holder._factory()
                value=self.repair.repair(value,depth,rgb_bgr=rgb_bgr,float_backend=holder._float_meter,visual_peak=use_visual)
                if use_visual and value['diagnostics']['compound_repair']['peak_legacy_fallback']:
                    previous=self.backend.measure(scene,depth_m=depth,rgb_bgr=rgb_bgr,visual_enabled=True)
                    value['highest_height_m']=previous.get('highest_height_m')
                value['input_validity_audit']=audit
        return {**value,'release_version':VERSION,'sensor_family':self.sensor,'field_metrology_qualified':False}
