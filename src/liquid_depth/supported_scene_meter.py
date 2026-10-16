"""Supported three-scene liquid-height measurement interface.

The code path is integrated, but field metrology qualification is intentionally
false until independent camera/container tests are performed. Scene selection is
explicit; missing-return percentage is not observable from RGB-D alone.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .censored_disparity_v56 import fit_censored_disparity
from .depth_failure_recovery_v49 import (
    _signature,
    prepare_dry_center_reference,
)
from .dry_anchored_layer_vote_v63 import vote_dry_anchored_layer
from .elastic_evidence_v67 import elastic_evidence_window
from .empty_geometry_v41 import height_calibration
from .liquid_height_metrics import world_points
from .ordinary_surface_v35 import interior_ray_roi
from .quantized_disparity_v72 import fit_quantized_disparity
from .range_runtime_v33 import FivePriorImageMeter

SCENES = frozenset({'ordinary_diffuse', 'homogeneous_transparent',
                    'homogeneous_translucent', 'partial_depth_loss'})


def _measurement(scene: str, route: str, value: float | None,
                 diagnostics: dict[str, Any]) -> dict[str, Any]:
    numeric = value is not None and np.isfinite(value)
    return {
        'schema_version': 1,
        'scene': scene,
        'route': route,
        'liquid_depth_m': float(value) if numeric else None,
        'numeric_output': bool(numeric),
        'reasons': [] if numeric else list(diagnostics.get('reasons', ['no_numeric_evidence'])),
        'field_metrology_qualified': False,
        'software_route_status': 'supported_three_scene_v3',
        'diagnostics': diagnostics,
    }


class FormalSceneMeter:
    """Explicit dispatcher for ordinary, single-layer transparent and <=50% loss.

    Construct with ``from_empty_geometry`` for ordinary/partial-loss or
    ``from_five_priors`` for transparent/translucent. The latter requires
    rectified left/right observed images and independently selected ROIs.

    Partial loss uses the dry-container ROI and always returns a numeric estimate
    when current depth evidence exists. A calibrated structured-light disparity
    profile enables the V72 quantization correction and internal V67/V64 fallback.
    Other camera families keep the calibrated quantile route. None of these
    simulation results constitutes field-metrology qualification.
    """

    def __init__(self, *, bundle: dict | None = None,
                 dry_reference: Any | None = None,
                 ordinary_container_roi: np.ndarray | None = None,
                 refraction_meter: FivePriorImageMeter | None = None,
                 refraction_scene: str | None = None,
                 max_return_depth_m: float | None = None,
                 layer_vote_enabled: bool = True,
                 elastic_evidence_enabled: bool | None = None,
                 quantized_disparity_profile: dict | None = None) -> None:
        self.bundle = bundle
        self.dry_reference = dry_reference
        self.ordinary_container_roi = ordinary_container_roi
        self.refraction_meter = refraction_meter
        self.refraction_scene = refraction_scene
        self.max_return_depth_m = max_return_depth_m
        self.layer_vote_enabled = bool(layer_vote_enabled)
        self.elastic_evidence_enabled = (quantized_disparity_profile is not None
                                         if elastic_evidence_enabled is None else bool(elastic_evidence_enabled))
        if quantized_disparity_profile is not None:
            required = {'disparity_step_px', 'disparity_noise_px'}
            if (not required.issubset(quantized_disparity_profile) or
                    any(not np.isfinite(quantized_disparity_profile[key]) or
                        quantized_disparity_profile[key] <= 0 for key in required)):
                raise ValueError('valid_quantized_disparity_profile_required')
            if not self.elastic_evidence_enabled:
                raise ValueError('quantized_profile_requires_elastic_evidence')
            if max_return_depth_m is None:
                raise ValueError('quantized_profile_requires_camera_max_return_depth')
            if bundle is None or bundle.get('sensor_family') != 'structured_light':
                raise ValueError('quantized_profile_requires_structured_light')
        self.quantized_disparity_profile = quantized_disparity_profile

    @classmethod
    def from_empty_geometry(cls, bundle: dict, dry_depth_m: np.ndarray,
                            *, max_return_depth_m: float | None = None,
                            layer_vote_enabled: bool = True,
                            elastic_evidence_enabled: bool | None = None,
                            quantized_disparity_profile: dict | None = None) -> FormalSceneMeter:
        if max_return_depth_m is not None and (
                not np.isfinite(max_return_depth_m) or max_return_depth_m <= 0):
            raise ValueError('positive_camera_max_return_depth_required')
        dry = np.asarray(dry_depth_m)
        expected = tuple(reversed(bundle['image_size']))
        if dry.shape != expected:
            raise ValueError('dry_image_shape_changed_recalibration_required')
        cal = height_calibration(bundle)
        rim = float(bundle['rim_z_m'])
        opening_plane = replace(cal, bottom_z_m=rim - 1e-5)
        opening = interior_ray_roi(opening_plane, rim, dry.shape, .90)
        dry_xyz = world_points(dry, cal)
        normalized = dry_xyz[..., :2] / np.asarray(cal.radii_m)
        inside = (abs(normalized) ** cal.footprint_exponent).sum(axis=-1) \
                 <= 1.05 ** cal.footprint_exponent
        margin = max(.03, .02 * rim)
        plausible = ((dry_xyz[..., 2] >= cal.bottom_z_m - margin) &
                     (dry_xyz[..., 2] <= rim + margin))
        return cls(bundle=bundle,
                   dry_reference=prepare_dry_center_reference(bundle, dry),
                   ordinary_container_roi=opening & inside & plausible,
                   max_return_depth_m=max_return_depth_m,
                   layer_vote_enabled=layer_vote_enabled,
                   elastic_evidence_enabled=elastic_evidence_enabled,
                   quantized_disparity_profile=quantized_disparity_profile)

    @classmethod
    def from_five_priors(cls, cal: dict, masks: dict,
                         prior_images: list[dict], known_heights_m: list[float],
                         *, scene: str) -> FormalSceneMeter:
        if scene not in {'homogeneous_transparent', 'homogeneous_translucent'}:
            raise ValueError('five_priors_only_for_single_layer_transparent_liquid')
        known = np.asarray(known_heights_m, dtype=float)
        if known.shape != (5,) or not np.isfinite(known).all() or known[0] != 0 or \
                np.any(known[1:] <= 0) or len(np.unique(known)) != 5:
            raise ValueError('one_empty_plus_four_distinct_known_liquid_depths_required')
        condition = ('transparent' if scene == 'homogeneous_transparent'
                     else 'translucent_mild')
        method = ('baseline_v32' if condition == 'transparent'
                  else 'midpoint_dense')
        meter = FivePriorImageMeter(cal, masks, prior_images, known, condition,
                                    method=method)
        return cls(refraction_meter=meter, refraction_scene=scene)

    def measure(self, scene: str, *, rgb_bgr: np.ndarray | None = None,
                depth_m: np.ndarray | None = None,
                mask_probability: np.ndarray | None = None,
                stereo_images: dict | None = None,
                recent_depth_frames: list[np.ndarray] | None = None,
                stationary_confirmed: bool = False) -> dict[str, Any]:
        if scene not in SCENES:
            raise ValueError('unsupported_scene_route')
        if scene.startswith('homogeneous_'):
            if self.refraction_meter is None or scene != self.refraction_scene:
                raise ValueError('scene_specific_five_prior_calibration_required')
            if stereo_images is None:
                raise ValueError('current_rectified_left_right_images_required')
            result = self.refraction_meter.measure(stereo_images)
            return _measurement(scene, 'five_prior_refraction_' + self.refraction_meter.method,
                                result.get('height_m'), result)

        if self.bundle is None or self.dry_reference is None:
            raise ValueError('empty_geometry_and_dry_depth_reference_required')
        if depth_m is None:
            raise ValueError('current_metric_depth_required')
        depth = np.asarray(depth_m)
        if depth.shape != self.dry_reference.image_shape:
            raise ValueError('image_shape_changed_recalibration_required')
        if _signature(self.bundle) != self.dry_reference.geometry_signature:
            raise ValueError('stale_dry_reference_recalibration_required')
        probability = None
        if mask_probability is not None:
            probability = np.asarray(mask_probability)
            if probability.shape != depth.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('finite_aligned_mask_probability_required')
        cal = height_calibration(self.bundle)
        xyz = world_points(depth, cal)
        valid = self.ordinary_container_roi & np.isfinite(depth) & (depth > 0)
        source = 'empty_container_roi'
        if not valid.any():
            rim = float(self.bundle['rim_z_m'])
            opening = interior_ray_roi(replace(cal, bottom_z_m=rim-1e-5), rim, depth.shape, .90)
            normalized = xyz[..., :2] / np.asarray(cal.radii_m)
            inside = (abs(normalized) ** cal.footprint_exponent).sum(axis=-1) <= 1.05 ** cal.footprint_exponent
            margin = max(.03, .02 * rim)
            valid = opening & np.isfinite(depth) & (depth > 0) & inside & (xyz[..., 2] >= cal.bottom_z_m - margin) & (xyz[..., 2] <= rim + margin)
            source = 'dynamic_geometry_rescue'
        count = int(valid.sum())
        quantile = 50 if self.bundle['sensor_family'] == 'tof' else 58
        heights = xyz[..., 2][valid] - cal.bottom_z_m
        value = float(np.percentile(heights, quantile)) if count else None
        flags = []
        if (count and self.max_return_depth_m is not None and
                np.any(depth[valid] > self.max_return_depth_m + .002)):
            flags.append('camera_range_profile_mismatch_quantile_fallback')
        if (self.layer_vote_enabled and recent_depth_frames and
                stationary_confirmed):
            flags.append('temporal_history_not_used_by_quantile_v64')
        if recent_depth_frames and not stationary_confirmed:
            flags.append('temporal_frames_ignored_without_stationary_confirmation')
        if (recent_depth_frames and stationary_confirmed and
                self.bundle['sensor_family'] != 'structured_light'):
            flags.append('joint_disparity_not_enabled_for_sensor')
        value_source = 'container_return_quantile'
        disparity_fit = None
        joint_fit_used = False
        if (not self.layer_vote_enabled and count >= 24 and
                self.bundle['sensor_family'] in
                {'active_stereo', 'structured_light'} and
                self.max_return_depth_m is not None):
            fit_input = depth
            fit_roi = valid
            if (stationary_confirmed and recent_depth_frames and
                    self.bundle['sensor_family'] == 'structured_light'):
                previous = [np.asarray(frame) for frame in recent_depth_frames[-3:]]
                if any(frame.shape != depth.shape for frame in previous):
                    raise ValueError('aligned_recent_depth_frames_required')
                fit_input = np.stack([*previous, depth])
                fit_roi = (self.ordinary_container_roi if source == 'empty_container_roi'
                           else valid)
                joint_fit_used = True
            disparity_fit = fit_censored_disparity(
                fit_input, fit_roi, cal, float(self.bundle['baseline_m']),
                self.max_return_depth_m)
            if disparity_fit is not None:
                value = disparity_fit['height_m']
                value_source = ('range_censored_disparity_joint_plane'
                                if joint_fit_used else 'range_censored_disparity_plane')
                if not disparity_fit['optimizer_converged']:
                    flags.append('disparity_optimizer_not_converged')
        layer_vote = None
        if (self.layer_vote_enabled and count >= 24 and
                self.bundle['sensor_family'] == 'structured_light' and
                self.max_return_depth_m is not None and value is not None):
            layer_vote = vote_dry_anchored_layer(
                heights, rim_height_m=float(self.bundle['rim_z_m']),
                primary_height_m=value,
                container_quantile_height_m=float(np.percentile(heights, quantile)))
            layer_vote['applied'] = bool(
                layer_vote['selected'] and
                layer_vote['candidate'] == 'dry_anchored_near_zero_peak' and
                layer_vote['wet_peak_m'] is None and
                value >= .25 and
                abs(value - layer_vote['height_m']) > .15)
            if layer_vote['applied']:
                value = layer_vote['height_m']
                value_source = 'dry_anchored_layer_vote_v64:' + layer_vote['candidate']
                flags.append('dry_anchored_layer_candidate_selected')
        elastic_evidence = None
        quantized_fit = None
        pre_elastic_value = value
        if (self.elastic_evidence_enabled and self.layer_vote_enabled and
                self.bundle['sensor_family'] == 'structured_light' and
                self.max_return_depth_m is not None and count >= 24 and
                value is not None):
            camera_distance = self.bundle.get('bottom_plane_camera', {}).get(
                'perpendicular_distance_m', float('nan'))
            elastic_evidence = elastic_evidence_window(
                heights, camera_to_bottom_m=float(camera_distance),
                primary_height_m=value,
                dry_layer_already_selected=bool(layer_vote and layer_vote['applied']))
            if elastic_evidence['applied']:
                value = elastic_evidence['height_m']
                value_source = 'elastic_evidence_window_v67'
                flags.append('elastic_evidence_window_selected')
            if (self.quantized_disparity_profile is not None and
                    camera_distance >= 8. and
                    .05 < pre_elastic_value < .25 * camera_distance and
                    not (layer_vote and layer_vote['applied'])):
                k = np.asarray(self.bundle['calibration']['intrinsics'])
                pose = np.asarray(self.bundle['calibration']['camera_to_world_cv'])
                raw_returns = depth[valid]
                scale = float(k[0, 0]) * float(self.bundle['baseline_m'])
                step = float(self.quantized_disparity_profile['disparity_step_px'])
                primary_index = int(np.argmin(np.abs(heights - pre_elastic_value)))
                primary_bin = int(np.rint(scale / raw_returns[primary_index] / step))
                adjacent = np.abs(scale / raw_returns - (primary_bin - 1) * step) < min(.001, .04 * step)
                coarse_candidate = (float(np.median(heights[adjacent]))
                                    if primary_bin > 1 and adjacent.sum() >= 24
                                    else float(pre_elastic_value))
                if elastic_evidence['applied']:
                    coarse_candidate = float(elastic_evidence['height_m'])
                quantized_fit = fit_quantized_disparity(
                    depth[valid], heights,
                    camera_z_m=float(pose[2, 3]),
                    bottom_z_m=float(cal.bottom_z_m),
                    fx_px=float(k[0, 0]),
                    baseline_m=float(self.bundle['baseline_m']),
                    disparity_step_px=float(self.quantized_disparity_profile['disparity_step_px']),
                    disparity_noise_px=float(self.quantized_disparity_profile['disparity_noise_px']),
                    max_return_depth_m=float(self.max_return_depth_m),
                    coarse_height_m=coarse_candidate,
                    upper_height_m=float(pre_elastic_value),
                    radius_m=.18)
                if quantized_fit['ambiguous']:
                    if quantized_fit['applied']:
                        value = quantized_fit['height_m']
                        value_source = 'quantized_disparity_likelihood_v72'
                        flags.append('adjacent_disparity_bins_resolved')
                    else:
                        value = pre_elastic_value
                        value_source = 'quantization_ambiguity_guard_v71'
                        flags.append('adjacent_disparity_bins_fit_unavailable')
        paired_support = 0
        if self.bundle['sensor_family'] == 'tof' and value is not None and value <= 1.0:
            dry_center = self.dry_reference.dry_height_median_m
            central = self.dry_reference.central_roi & np.isfinite(depth) & (depth > 0)
            paired_support = int(central.sum())
            if dry_center is not None and paired_support >= 24:
                wet_center = float(np.median(xyz[..., 2][central] - cal.bottom_z_m))
                value = wet_center - dry_center
                value_source = 'tof_shallow_dry_paired_center'
        if count < 24:
            flags.append('few_current_container_returns')
        if source != 'empty_container_roi':
            flags.append('static_roi_has_no_current_return')
        if value is not None and (value < 0 or value > self.bundle['rim_z_m']):
            flags.append('estimate_outside_container_height_unclipped')
        route = ('partial_loss_quantized_disparity_v72' if quantized_fit is not None
                 and quantized_fit['ambiguous'] and quantized_fit['applied'] else
                 'partial_loss_quantization_guard_v71' if quantized_fit is not None
                 and quantized_fit['ambiguous'] else
                 'partial_loss_elastic_evidence_v67' if elastic_evidence is not None
                 and elastic_evidence['applied'] else
                 'partial_loss_dry_anchored_vote_v64' if layer_vote is not None
                 and layer_vote['applied'] else
                 'partial_loss_quantile_v64' if self.layer_vote_enabled
                 and self.max_return_depth_m is not None else
                 'partial_loss_joint_disparity_v59' if disparity_fit is not None
                 and joint_fit_used else
                 'partial_loss_censored_disparity_v56' if disparity_fit is not None
                 else 'partial_loss_ungated_v52')
        diagnostics = {
            'measurement_route': route,
            'formal_partial_loss_policy': ('quantized_disparity_v72_supported'
                                           if self.quantized_disparity_profile is not None else
                                           'elastic_evidence_v67_experimental' if self.elastic_evidence_enabled else
                                           'dry_anchored_quantile_v64' if self.layer_vote_enabled
                                           and self.max_return_depth_m is not None else
                                           'ungated_quantile_v52' if self.layer_vote_enabled else 'selective_stationary_v61'),
            'selected_pixels': count, 'static_roi_pixels': int(self.ordinary_container_roi.sum()),
            'selection_source': source, 'quantile_percent': quantile,
            'value_source': value_source, 'paired_center_support': paired_support,
            'height_p10_m': float(np.percentile(heights, 10)) if count else None,
            'height_p90_m': float(np.percentile(heights, 90)) if count else None,
            'mask_ge_0_5_valid_pixels': int(np.count_nonzero(
                (probability >= .5) & valid)) if probability is not None else None,
            'mask_ignored_for_value': True, 'quality_flags': flags,
            'disparity_fit': disparity_fit,
            'elastic_evidence': elastic_evidence,
            'quantized_disparity': quantized_fit,
            'layer_vote': layer_vote,
            'stationary_confirmed': bool(stationary_confirmed),
            'joint_fit_used': bool(joint_fit_used and disparity_fit is not None),
            'reasons': [] if count else ['no_current_return_in_calibrated_container'],
            'research_only': True, 'production_qualified': False,
        }
        if scene == 'ordinary_diffuse':
            diagnostics['underlying_return_route'] = route
            diagnostics['formal_ordinary_policy'] = 'shared_robust_returns_v2'
            route = diagnostics['measurement_route'] = 'ordinary_shared_robust_returns_v2'
        return _measurement(scene, route, value, diagnostics)
