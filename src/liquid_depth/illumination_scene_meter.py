"""Research-only glare and low-light liquid-depth route.

RGB quality is observable but does not by itself move the depth estimate:
saturated highlights and darkness are unreliable segmentation evidence. The
metric estimate reuses the calibrated empty-vessel return path.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .empty_geometry_v41 import height_calibration
from .illumination_surface_evidence import dry_changed_evidence, estimate_surface_evidence
from .liquid_height_metrics import world_points
from .supported_scene_meter import FormalSceneMeter

ILLUMINATION_SCENES = frozenset({"glare", "low_light"})


class IlluminationSceneMeter:
    def __init__(self, ordinary_meter: FormalSceneMeter, *, dry_depth_m=None,
                 refinement_enabled=False):
        if ordinary_meter.bundle is None or ordinary_meter.dry_reference is None:
            raise ValueError("empty_geometry_and_dry_reference_required")
        self._ordinary_meter = ordinary_meter
        self.refinement_enabled = bool(refinement_enabled)
        self._dry_depth_m = None if dry_depth_m is None else np.asarray(dry_depth_m).copy()
        if self.refinement_enabled and self._dry_depth_m is None:
            raise ValueError('dry_frame_required_for_illumination_refinement')

    @classmethod
    def from_empty_geometry(
        cls, bundle: dict, dry_depth_m: np.ndarray, *,
        max_return_depth_m: float = 10.0,
        quantized_disparity_profile: dict | None = None,
        refinement_enabled: bool = False,
    ) -> IlluminationSceneMeter:
        return cls(FormalSceneMeter.from_empty_geometry(
            bundle, dry_depth_m,
            max_return_depth_m=max_return_depth_m,
            quantized_disparity_profile=quantized_disparity_profile,
        ), dry_depth_m=dry_depth_m, refinement_enabled=refinement_enabled)

    def _refine_frame(self, scene, depth, base, base_diagnostics=None):
        meter = self._ordinary_meter
        value = base
        info = {'value_source': 'v1_returns', 'trusted_changed_height_m': None}
        diagnostic = base_diagnostics or {}
        vote = diagnostic.get('layer_vote') or {}
        if (vote.get('applied') and diagnostic.get('selected_pixels', 0) >= 64 and
                diagnostic.get('height_p10_m', -np.inf) > max(.05, .02*meter.bundle['rim_z_m'])):
            cal = height_calibration(meter.bundle)
            valid = meter.ordinary_container_roi & np.isfinite(depth) & (depth > 0)
            heights = world_points(depth, cal)[..., 2][valid]-cal.bottom_z_m
            if len(heights) >= 64:
                value = float(np.percentile(heights, diagnostic.get('quantile_percent', 58)))
                info.update(value_source='contradicted_dry_vote_reverted',
                            dry_vote_candidate_m=base, restored_return_quantile_m=value)
        if scene == 'low_light' and meter.bundle['sensor_family'] != 'tof':
            evidence = estimate_surface_evidence(
                meter.bundle, depth, selection_mask=meter.ordinary_container_roi,
                max_return_depth_m=meter.max_return_depth_m or 10.)
            candidate, noise = evidence['height_m'], evidence.get('median_noise_m', np.inf)
            if (candidate is not None and value is not None and candidate <= value and
                    evidence['layer_pixels'] >= 64 and evidence.get('layer_tiles', 0) >= 4 and
                    noise <= .2 and abs(candidate-value) <= max(.03, .6*noise)):
                value = candidate
                info['value_source'] = 'lowlight_sensor_bias_refinement'
            info['sensor_fit'] = {k: v for k, v in evidence.items() if not k.endswith('_mask')}
        if meter.bundle['sensor_family'] == 'tof':
            evidence = dry_changed_evidence(
                meter.bundle, depth, self._dry_depth_m, meter.ordinary_container_roi)
            candidate = evidence['height_m']
            rim = float(meter.bundle['rim_z_m'])
            margin = max(.03, .02*rim)
            plausible = candidate is not None and -margin <= candidate <= rim+margin
            spatial_fraction = evidence['layer_pixels']/max(evidence['dry_reference_pixels'], 1)
            evidence['candidate_physically_plausible'] = plausible
            evidence['layer_fraction_of_dry_reference'] = spatial_fraction
            if plausible and evidence['dry_reference_pixels'] >= 24 and spatial_fraction >= .005:
                info['trusted_changed_height_m'] = candidate
                if value is None or abs(candidate-value) > .1:
                    value = candidate
                    info['value_source'] = 'dry_changed_tof_layer'
            info['dry_changed_layer'] = {k: v for k, v in evidence.items() if not k.endswith('_mask')}
        return value, info

    def measure(
        self, scene: str, *, rgb_bgr: np.ndarray,
        depth_m: np.ndarray, mask_probability: np.ndarray | None = None,
        recent_depth_frames: list[np.ndarray] | None = None,
        stationary_confirmed: bool = False,
    ) -> dict[str, Any]:
        if scene not in ILLUMINATION_SCENES:
            raise ValueError("explicit_glare_or_low_light_scene_required")
        rgb = np.asarray(rgb_bgr)
        depth = np.asarray(depth_m)
        if rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("aligned_uint8_rgb_depth_required")
        roi = self._ordinary_meter.ordinary_container_roi
        if roi is None or not np.any(roi):
            raise ValueError("calibrated_container_roi_required")
        grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        brightness = grey[roi]
        quality = {
            "roi_pixels": int(roi.sum()),
            "median_luma_8bit": float(np.median(brightness)),
            "dark_fraction_luma_lt_24": float(np.mean(brightness < 24)),
            "saturated_fraction_any_channel_ge_250": float(
                np.mean(np.max(rgb[roi], axis=-1) >= 250)),
        }
        ordinary = self._ordinary_meter.measure(
            "ordinary_diffuse", depth_m=depth,
            mask_probability=mask_probability,
            recent_depth_frames=recent_depth_frames,
            stationary_confirmed=stationary_confirmed,
        )
        diagnostics = dict(ordinary["diagnostics"])
        diagnostics["illumination_quality"] = quality
        observability_flags = []
        if scene == "low_light" and quality["dark_fraction_luma_lt_24"] >= .8:
            observability_flags.append("rgb_severely_underexposed")
        if scene == "glare" and quality[
                "saturated_fraction_any_channel_ge_250"] >= .5:
            observability_flags.append("rgb_severely_saturated")
        diagnostics["illumination_observability_flags"] = observability_flags
        diagnostics["illumination_flags_are_not_calibrated_confidence"] = True
        diagnostics["underlying_return_route"] = diagnostics.get(
            "underlying_return_route", ordinary["route"])
        diagnostics["rgb_used_for_depth_value"] = False
        diagnostics["dry_depth_prior_required"] = True
        if self.refinement_enabled:
            value, evidence = self._refine_frame(
                scene, depth, ordinary['liquid_depth_m'], ordinary['diagnostics'])
            evidence['temporal_used'] = False
            severe = (quality['median_luma_8bit'] <= 2 if scene == 'low_light' else
                      quality['saturated_fraction_any_channel_ge_250'] >= .5)
            if severe and stationary_confirmed and recent_depth_frames:
                past = [np.asarray(frame) for frame in recent_depth_frames[-3:]]
                if any(frame.shape != depth.shape for frame in past):
                    raise ValueError('aligned_recent_depth_frames_required')
                prior = []
                for frame in past:
                    reading = self._ordinary_meter.measure('ordinary_diffuse', depth_m=frame)
                    prior.append(self._refine_frame(
                        scene, frame, reading['liquid_depth_m'], reading['diagnostics']))
                trusted = [d['trusted_changed_height_m'] for _, d in prior
                           if d['trusted_changed_height_m'] is not None]
                numbers = [v for v, _ in prior if v is not None and np.isfinite(v)]
                if evidence['trusted_changed_height_m'] is None and trusted:
                    memory = float(np.median(trusted))
                    if value is None or abs(value-memory) > max(.05, .02*abs(memory)):
                        value = memory
                        evidence.update(temporal_used=True, value_source='stationary_dry_changed_memory')
                elif evidence['trusted_changed_height_m'] is None and len(numbers) >= 2:
                    center = float(np.median(numbers))
                    agreement = max(.03, .02*abs(center))
                    consistent = [v for v in numbers if abs(v-center) <= agreement]
                    if len(consistent) >= 2 and (value is None or abs(value-center) > max(.05, 2*agreement)):
                        value = float(np.median(consistent))
                        evidence.update(temporal_used=True, value_source='stationary_consensus_memory')
            flags = []
            current_count = diagnostics.get('selected_pixels', 0)
            roi_count = diagnostics.get('static_roi_pixels', 0)
            if current_count < 24 or current_count/max(roi_count, 1) < .01:
                flags.append('few_current_container_returns')
            if severe and evidence['trusted_changed_height_m'] is None:
                flags.append('liquid_layer_not_independently_confirmed')
            if evidence['temporal_used']:
                flags.append('recent_stationary_history_used')
            evidence['evidence_flags'] = flags
            evidence['history_frames_considered'] = min(3, len(recent_depth_frames or []))
            evidence['evidence_flags_are_not_calibrated_confidence'] = True
            diagnostics['illumination_refinement'] = evidence
            ordinary = {**ordinary, 'liquid_depth_m': value,
                        'numeric_output': value is not None and bool(np.isfinite(value)),
                        'reasons': [] if value is not None else ordinary['reasons']}
        return {
            **ordinary,
            "scene": scene,
            "route": ("illumination_evidence_v2" if self.refinement_enabled else
                      "illumination_dry_geometry_returns_v1"),
            "field_metrology_qualified": False,
            "software_route_status": ("illumination_v2_experimental" if self.refinement_enabled else
                                      "merged_illumination_v1_research"),
            "diagnostics": diagnostics,
        }
