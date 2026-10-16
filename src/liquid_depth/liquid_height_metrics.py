"""Bottom-referenced metrology. Inference functions never read surface labels.

Calibration is an ideal, known horizontal bottom and fixed vessel footprint in
this simulation protocol; it is not an inferred or hardware-qualified bottom.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HeightCalibration:
    intrinsics: np.ndarray
    camera_to_world_cv: np.ndarray
    bottom_z_m: float
    radii_m: tuple[float, float]
    footprint_exponent: float = 2.0

    def __post_init__(self):
        k, t = np.asarray(self.intrinsics), np.asarray(self.camera_to_world_cv)
        if k.shape != (3, 3) or t.shape != (4, 4):
            raise ValueError('Metric intrinsics and 4x4 pose required')
        if not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError('Nonfinite calibration')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError('Invalid camera calibration')
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5):
            raise ValueError('Pose rotation must be orthonormal')
        if np.linalg.det(t[:3, :3]) < .999 or not np.isfinite(self.bottom_z_m):
            raise ValueError('Invalid frame convention or bottom')
        if min(self.radii_m) <= 0 or self.footprint_exponent < 1:
            raise ValueError('Invalid calibrated footprint')

    @classmethod
    def from_simulation(cls, metadata):
        # Blender axes are x right, y up, -z forward, unlike OpenCV.
        pose = np.asarray(metadata['camera_to_world'], float) @ np.diag([1, -1, -1, 1])
        return cls(np.asarray(metadata['camera_intrinsics'], float), pose,
                   float(metadata['container_bottom_z_m']),
                   (float(metadata['surface_radius_x_m']), float(metadata['surface_radius_y_m'])),
                   float(metadata['container_exponent']))

    def area_grid(self, size=24):
        axis = (np.arange(size) + .5) / size * 2 - 1
        x, y = np.meshgrid(axis, axis)
        valid = abs(x)**self.footprint_exponent + abs(y)**self.footprint_exponent <= 1
        return np.column_stack((x[valid]*self.radii_m[0], y[valid]*self.radii_m[1]))


def world_points(depth, calibration):
    z = np.asarray(depth, float)
    v, u = np.indices(z.shape)
    k, t = calibration.intrinsics, calibration.camera_to_world_cv
    camera = np.stack(((u-k[0, 2])/k[0, 0]*z, (v-k[1, 2])/k[1, 1]*z, z), -1)
    return camera @ t[:3, :3].T + t[:3, 3]


def reference_statistics(metadata, calibration):
    """Evaluator-only analytic ground truth on a uniform horizontal area grid."""
    from .simulation import SyntheticScene, surface_height_and_gradient
    from dataclasses import fields
    keys = {f.name for f in fields(SyntheticScene)}
    scene = SyntheticScene(**{k: metadata[k] for k in keys})
    xy = calibration.area_grid()
    z, _, _ = surface_height_and_gradient(scene, xy[:, 0], xy[:, 1])
    return field_statistics(z-calibration.bottom_z_m)


def field_statistics(values):
    a = np.asarray(values, float)
    return {name: float(fun(a)) for name, fun in (
        ('mean_m', np.mean), ('min_m', np.min), ('max_m', np.max),
        ('p05_m', lambda x: np.percentile(x, 5)), ('p95_m', lambda x: np.percentile(x, 95)))}


def estimate_height(depth, predicted_mask, calibration, confidence=None, threshold=None):
    """Prediction-only local surface reconstruction, independent of GT masks.

    Median per fixed XY cell, then four-neighbour inverse-distance interpolation
    across the calibrated footprint. An extrapolated field is an estimate, not
    evidence that unobserved waves or the liquid interface were truly seen.
    Gate constants are fixed before evaluation and do not use scene labels.
    """
    z = np.asarray(depth, float)
    mask = np.asarray(predicted_mask, bool)
    if z.ndim != 2 or mask.shape != z.shape:
        raise ValueError('Aligned 2-D metric depth and predicted mask required')
    use = mask & np.isfinite(z) & (z > 0)
    if threshold is not None:
        if confidence is None or np.asarray(confidence).shape != z.shape:
            raise ValueError('Confidence array required')
        use &= np.asarray(confidence) >= threshold
    out = dict(accepted=False, statistics=None, reasons=[], predicted_points=int(use.sum()),
               occupied_cells=0, footprint_support_fraction=0., calibration='ideal_simulation')
    if use.sum() < 64:
        out['reasons'] = ['insufficient_predicted_support']
        return out
    xyz = world_points(z, calibration)[use]
    normalized = xyz[:, :2] / calibration.radii_m
    inside = (abs(normalized)**calibration.footprint_exponent).sum(1) <= 1
    xyz, normalized = xyz[inside], normalized[inside]
    if len(xyz) < 64:
        out['reasons'] = ['insufficient_calibrated_footprint_support']
        return out
    cells = np.floor((normalized+1)*5).astype(int).clip(0, 9)
    code = cells[:, 0]+10*cells[:, 1]
    anchors = np.array([np.median(xyz[code == c], axis=0) for c in np.unique(code)
                        if np.count_nonzero(code == c) >= 4])
    area = calibration.area_grid()
    area_cells = np.floor((area/np.asarray(calibration.radii_m)+1)*5).astype(int).clip(0, 9)
    possible = len(np.unique(area_cells[:, 0]+10*area_cells[:, 1]))
    out['occupied_cells'] = len(anchors)
    out['footprint_support_fraction'] = len(anchors)/possible
    if len(anchors) < 4 or out['footprint_support_fraction'] < .12:
        out['reasons'] = ['insufficient_spatial_support']
        return out
    distances = np.linalg.norm((area[:, None]-anchors[None, :, :2])/calibration.radii_m, axis=-1)
    ids = np.argsort(distances, axis=1)[:, :4]
    weights = 1/np.maximum(np.take_along_axis(distances, ids, 1), .01)**2
    heights = anchors[:, 2]-calibration.bottom_z_m
    field = (heights[ids]*weights).sum(1)/weights.sum(1)
    stats = field_statistics(field)
    out.update(statistics=stats, observed_height_mad_m=float(np.median(abs(heights-np.median(heights)))),
               extrapolated_area_fraction=1-out['footprint_support_fraction'])
    if stats['mean_m'] <= 0:
        out['reasons'] = ['nonpositive_liquid_depth']
        return out
    out.update(accepted=True, reasons=[])
    return out


FEATURE_NAMES = ('model_mean_over_bottom_distance', 'raw_mean_over_bottom_distance',
                 'raw_model_difference', 'raw_valid_fraction', 'model_spatial_support',
                 'raw_spatial_support', 'model_mad_over_distance', 'raw_mad_over_distance',
                 'model_range_over_distance', 'raw_range_over_distance',
                 'raw_estimate_available', 'log_bottom_distance')


def observable_features(model_result, raw_result, raw, mask, calibration):
    """Whitelisted input features only; no true surface, material or labels."""
    d = max(.05, float(calibration.camera_to_world_cv[2, 3]-calibration.bottom_z_m))
    def stat(result, name, default=0.):
        return (result.get('statistics') or {}).get(name, default)
    h = stat(model_result, 'mean_m')
    r = stat(raw_result, 'mean_m', h)
    raw_valid = np.isfinite(raw) & (raw > 0) & mask
    features = [h/d, r/d, (r-h)/d, raw_valid.sum()/max(mask.sum(), 1),
                model_result['footprint_support_fraction'], raw_result['footprint_support_fraction'],
                model_result.get('observed_height_mad_m', 0)/d,
                raw_result.get('observed_height_mad_m', 0)/d,
                (stat(model_result, 'p95_m')-stat(model_result, 'p05_m'))/d,
                (stat(raw_result, 'p95_m')-stat(raw_result, 'p05_m'))/d,
                float(raw_result['accepted']), np.log(d)]
    return np.nan_to_num(features).tolist(), d


def score_frame(prediction, reference):
    accepted = bool(prediction['accepted'])
    evaluable = accepted and reference is not None and reference['mean_m'] > 0
    out = dict(accepted=accepted, evaluable=evaluable, reasons=prediction['reasons'])
    if not evaluable:
        return out
    h, truth = prediction['statistics']['mean_m'], reference['mean_m']
    error = h-truth
    tolerance = max(.005, .02*truth)
    out.update(predicted_height_m=h, true_height_m=truth, signed_error_m=error,
               absolute_error_m=abs(error), abs_rel=abs(error)/truth,
               tolerance_m=tolerance, within_tolerance=bool(abs(error) <= tolerance),
               extrema_error_m={k:abs(prediction['statistics'][k]-reference[k])
                                for k in ('min_m', 'max_m', 'p05_m', 'p95_m')})
    return out


def summarize_frames(records):
    n, a = len(records), sum(r['accepted'] for r in records)
    valid = [r for r in records if r['evaluable']]
    errors = np.array([r['absolute_error_m'] for r in valid])
    good = sum(r['within_tolerance'] for r in valid)
    return dict(frames=n, accepted_frames=a, evaluable_frames=len(valid),
                output_coverage=a/max(n, 1), evaluable_output_rate=len(valid)/a if a else None,
                mae_mm=float(errors.mean()*1000) if len(valid) else None,
                p95_mm=float(np.percentile(errors, 95)*1000) if len(valid) else None,
                maximum_mm=float(errors.max()*1000) if len(valid) else None,
                signed_bias_mm=float(np.mean([r['signed_error_m'] for r in valid])*1000) if valid else None,
                abs_rel=float(np.mean([r['abs_rel'] for r in valid])) if valid else None,
                tolerance_pass_rate=good/len(valid) if valid else None,
                effective_pass_coverage=good/max(n, 1),
                min_mae_mm=float(np.mean([r['extrema_error_m']['min_m'] for r in valid])*1000) if valid else None,
                max_mae_mm=float(np.mean([r['extrema_error_m']['max_m'] for r in valid])*1000) if valid else None,
                rejection_reasons=dict(Counter(reason for r in records if not r['accepted'] for reason in r['reasons'])))
