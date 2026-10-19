"""One observation contract, one authoritative liquid path, composable auxiliaries.

This opt-in integration does not replace the independently validated manual facade.
RGB proposes labels. Calibration controls the optical family: a single RGB image
cannot identify a refractive index or make missing phase channels observable.
"""
from dataclasses import dataclass, field
import time
import cv2
import numpy as np
from .tof_scene_meter import ToFSceneMeter

VERSION = 'unified_tof_console_v1_20261019'
MATERIALS = frozenset(('ordinary_diffuse', 'homogeneous_transparent', 'homogeneous_translucent'))
TAGS = frozenset(('low_light', 'glare', 'uneven_surface', 'floating_objects', 'partial_depth_loss'))


@dataclass(frozen=True)
class ToFObservation:
    frame_id: str
    rgb_bgr: np.ndarray
    depth_m: np.ndarray
    timestamp_s: float
    correlations: np.ndarray | None = None
    sensor_family: str = 'tof'
    installation_id: str = 'default'

    def validate(self, expected_shape):
        if self.sensor_family != 'tof':
            raise ValueError('only_tof_supported')
        if np.asarray(self.depth_m).shape != tuple(expected_shape):
            raise ValueError('depth_shape_changed_reinitialize')
        a = np.asarray(self.rgb_bgr)
        if a.shape != (*expected_shape, 3) or a.dtype != np.uint8:
            raise ValueError('aligned_uint8_bgr_required')
        if not np.isfinite(self.timestamp_s):
            raise ValueError('finite_timestamp_required')
        if self.correlations is not None:
            c = np.asarray(self.correlations)
            if c.ndim != 4 or c.shape[:2] != (3, 4) or not np.isfinite(c).all():
                raise ValueError('three_frequency_four_phase_capture_required')


@dataclass
class ToFInstallation:
    installation_id: str
    geometry: dict
    dry_depth_m: np.ndarray
    confirmed_material: str | None = None
    # Same envelope for all materials; unused channels may be absent, never fabricated.
    phase: dict | None = None
    provenance: dict = field(default_factory=dict)
    confirmed_refractive_planar_unoccluded: bool = False

    def validate(self):
        if self.geometry.get('sensor_family') != 'tof':
            raise ValueError('tof_geometry_required')
        if self.confirmed_material is not None and self.confirmed_material not in MATERIALS:
            raise ValueError('unknown_material')
        if np.asarray(self.dry_depth_m).ndim != 2:
            raise ValueError('empty_metric_depth_required')


def observed_quality(observation, roi):
    d = np.asarray(observation.depth_m)
    gray = cv2.cvtColor(observation.rgb_bgr, cv2.COLOR_BGR2GRAY)
    pixels = gray[roi]
    return dict(container_invalid_fraction=float(np.mean(~np.isfinite(d[roi]) | (d[roi] <= 0))),
                median_luma=float(np.median(pixels) / 255),
                saturated_fraction=float(np.mean(pixels >= 250)),
                # This is observable ROI loss, NOT hidden true liquid-surface loss.
                loss_measure='calibrated_container_ROI_not_true_liquid_mask')


def make_plan(prediction, installation, quality):
    proposed = prediction['material']
    material = installation.confirmed_material or proposed
    tags = set(prediction['tags']) & TAGS
    warnings, blocked = [], []
    if installation.confirmed_material and proposed != material:
        warnings.append('rgb_material_disagrees_with_confirmed_calibration')
    if installation.confirmed_material is None:
        blocked.append('confirm_liquid_material_once_at_initialization')
    if quality['container_invalid_fraction'] > .02:
        tags.add('partial_depth_loss')
    if quality['container_invalid_fraction'] > .50:
        warnings.append('observed_container_loss_above_validated_0_50pct_scope')
    phase = material != 'ordinary_diffuse'
    visual_peak=not phase and 'floating_objects' in tags and prediction.get('float_peak_recommended',True)
    if 'floating_objects' in tags and not phase and not visual_peak:
        warnings.append('float_detected_visual_peak_not_recommended_by_validation')
    if phase and installation.phase is None:
        blocked.append('five_phase_priors_required_no_geometric_fallback')
    if phase and 'uneven_surface' in tags:
        if installation.confirmed_refractive_planar_unoccluded:
            warnings.append('rgbd_nonplanarity_suspected_review_confirmed_scope')
        else:
            blocked.append('refractive_nonplanar_surface_not_validated')
    # A full-ROI phase mean cannot safely treat an unknown occlusion as liquid.
    if phase and 'floating_objects' in tags:
        if installation.confirmed_refractive_planar_unoccluded:
            warnings.append('rgbd_float_suspected_review_confirmed_scope')
        else:
            blocked.append('refractive_float_phase_occlusion_not_validated')
    modules = ['empty_container_geometry', 'rgb_multilabel_router']
    modules += ['five_prior_amcw_refraction'] if phase else ['frozen_symmetric_liquid']
    modules += [f'{t}_observation_audit' for t in sorted(tags & {'low_light', 'glare', 'partial_depth_loss'})]
    if not phase and 'uneven_surface' in tags:
        modules.append('geometric_surface_envelope')
    if visual_peak:
        modules.append('v8_tof_v5_float_highest_only')
    modifiers = tags & {'low_light', 'glare', 'partial_depth_loss', 'uneven_surface'}
    composite = len(tags) > 1
    return dict(material=material, proposed_material=proposed, tags=sorted(tags), modules=modules,
                liquid_route='phase' if phase else 'geometric',
                visual_peak=visual_peak,
                blocked_reasons=blocked, warnings=warnings,
                composite=composite, composition_qualification='experimental_holdout_required' if composite else 'base_route_simulation_only',
                modifiers_do_not_stack_numerical_corrections=bool(modifiers),
                mutually_exclusive_liquid_backends=True)


class UnifiedToFMeter:
    def __init__(self, installation, classifier, *, profile_path):
        installation.validate()
        self.installation, self.classifier = installation, classifier
        self.shape = np.asarray(installation.dry_depth_m).shape
        self.geometric = ToFSceneMeter.from_empty_geometry(installation.geometry,
                         installation.dry_depth_m, profile_path=profile_path)
        # Use the existing calibrated empty-vessel ROI, never current truth.
        from .supported_scene_meter import FormalSceneMeter
        base = FormalSceneMeter.from_empty_geometry(installation.geometry, installation.dry_depth_m)
        self.roi = np.asarray(base.ordinary_container_roi, bool)
        if not self.roi.any():
            raise ValueError('empty_calibrated_container_roi')
        self.phase_cache = {}
        self.last_timestamp = None

    def preload_optional_peak_models(self):
        """Load trusted assets at initialization; do NOT measure any non-float image."""
        start=time.perf_counter()
        holder=self.geometric.backend.backend
        if holder._float_meter is None:
            if holder._factory is None:raise ValueError('optional_float_factory_missing')
            holder._float_meter=holder._factory()
        return (time.perf_counter()-start)*1000

    def measure(self, observation):
        start = time.perf_counter()
        observation.validate(self.shape)
        if observation.installation_id != self.installation.installation_id:
            raise ValueError('installation_id_changed_recalibrate')
        if self.last_timestamp is not None and observation.timestamp_s < self.last_timestamp:
            raise ValueError('out_of_order_capture')
        self.last_timestamp = observation.timestamp_s
        classification = self.classifier.predict(observation.rgb_bgr,depth_m=observation.depth_m,
                          dry_depth_m=self.installation.dry_depth_m,roi=self.roi)
        quality = observed_quality(observation, self.roi)
        plan = make_plan(classification, self.installation, quality)
        if plan['liquid_route'] == 'phase' and observation.correlations is None:
            plan['blocked_reasons'].append('raw_amcw_correlations_missing')
        if plan['blocked_reasons']:
            result = dict(liquid_depth_m=None, highest_height_m=None, numeric_output=False,
                          reasons=plan['blocked_reasons'], visual_module_executed=False)
        elif plan['liquid_route'] == 'phase':
            material = plan['material']
            if material not in self.phase_cache:
                self.phase_cache[material] = ToFSceneMeter.from_phase_priors(sensor='tof', scene=material,
                    **self.installation.phase)
            result = self.phase_cache[material].measure(material, correlations=observation.correlations)
        else:
            # All opaque scenes already share exactly this numerical liquid core.
            scene = 'floating_objects' if plan['visual_peak'] else 'uneven_surface' if 'uneven_surface' in plan['tags'] else 'ordinary_diffuse'
            result = self.geometric.measure(scene, depth_m=observation.depth_m,
                                            rgb_bgr=observation.rgb_bgr, visual_enabled=plan['visual_peak'])
        liquid=result.get('liquid_depth_m');highest=result.get('highest_height_m')
        consistency=[]
        if liquid is not None and highest is not None and highest<liquid-.005:
            consistency.append('highest_below_estimated_liquid_review_required')
        # Diagnose contradictions rather than silently clipping two measurements.
        plan['warnings']+=consistency
        plan['executed_modules']=plan['modules'] if not plan['blocked_reasons'] else ['empty_container_geometry','rgb_multilabel_router']
        return {**result, 'unified_version': VERSION, 'frame_id': observation.frame_id,
                'classification': classification, 'module_plan': plan, 'observed_quality': quality,
                'pipeline_ms': (time.perf_counter()-start)*1000,
                'physical_consistency_flags':consistency,
                'liquid_visual_guidance_used': False, 'field_metrology_qualified': False}
