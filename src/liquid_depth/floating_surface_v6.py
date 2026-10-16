"""Observation-supported local surfaces; inferred samples never count as measurements."""
from __future__ import annotations

import cv2
import joblib
import numpy as np
from scipy.spatial import cKDTree

from .empty_geometry_v41 import height_calibration
from .float_evidence_v3 import OptionalFloatMeterV3, sensor_sigma
from .floating_apex_v5 import candidates_v5, observed_peak_roi, select_v5
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .rgbd_evidence_model import observed_features


def basis(xy, degree):
    x, y = xy.T
    return np.column_stack([np.ones(len(x)), x, y]+([x*x, x*y, y*y] if degree == 2 else []))


def robust_fit(a, values, sigma, weights):
    if len(values) < 2*a.shape[1] or np.linalg.matrix_rank(a) < a.shape[1]: return None
    target = (values-np.median(values))/np.median(sigma)
    scaled_sigma = sigma/np.median(sigma)
    penalty = np.diag([1e-9]*3+([.2]*3 if a.shape[1] == 6 else []))
    w = weights/(scaled_sigma**2); coefficients = None
    for _ in range(5):
        lhs = a.T@(w[:, None]*a)+penalty
        if np.linalg.cond(lhs) > 1e8: return None
        coefficients = np.linalg.solve(lhs, a.T@(w*target))
        residual = (target-a@coefficients)/scaled_sigma
        scale = max(1., 1.4826*float(np.median(abs(residual-np.median(residual)))))
        u = abs(residual)/(4.685*scale)
        w = weights/(scaled_sigma**2)*np.where(u < 1, (1-u*u)**2, 0.)
        if np.count_nonzero(w > 0) < a.shape[1]+2: return None
    return coefficients*np.median(sigma), float(np.median(values)), residual, w


def fit_observed_surface(raw_values, pixels_xy, weights, cal, sensor):
    z = np.asarray(raw_values, float); pixels = np.asarray(pixels_xy, float); weights = np.asarray(weights, float)
    valid = np.isfinite(z) & (z > 0) & np.isfinite(pixels).all(1) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 12: return {'valid': False, 'reason': 'fewer_than_12_observed_points'}
    z, pixels, weights = z[valid], pixels[valid], weights[valid]
    if len(z) > 1024:
        sub = np.linspace(0, len(z)-1, 1024).astype(int); z, pixels, weights = z[sub], pixels[sub], weights[sub]
    span = np.ptp(pixels, axis=0)
    if span.min() < 2: return {'valid': False, 'reason': 'spatially_degenerate_support'}
    center = (pixels.min(0)+pixels.max(0))*.5; scale_xy = np.maximum(span*.5, 1.)
    xy = (pixels-center)/scale_xy
    sigma_z = sensor_sigma(z, {'sensor_family': sensor, 'calibration': {
        'intrinsics': cal.intrinsics.tolist(), 'camera_to_world_cv': cal.camera_to_world_cv.tolist(),
        'bottom_z_m': cal.bottom_z_m, 'radii_m': list(cal.radii_m), 'footprint_exponent': cal.footprint_exponent}})
    inverse = sensor != 'tof'; values = 1/z if inverse else z
    sigma = sigma_z/(z*z) if inverse else sigma_z
    cells = np.floor((xy+1)*1.499).astype(int); folds = (cells[:, 0]+cells[:, 1]) % 3
    models = []
    for degree in [1, 2]:
        if degree == 2 and len(z) < 30: continue
        a = basis(xy, degree); fit = robust_fit(a, values, sigma, weights)
        if fit is None: continue
        cv = []; fold_models = []
        for fold in range(3):
            train = folds != fold; test = ~train
            if not test.any(): continue
            subfit = robust_fit(a[train], values[train], sigma[train], weights[train])
            if subfit is None: continue
            c, offset, _, _ = subfit
            cv.extend((abs(a[test]@c+offset-values[test])/sigma[test]).tolist()); fold_models.append((c, offset))
        if len(fold_models) < 2: continue
        models.append({'degree': degree, 'fit': fit, 'cv': float(np.median(cv)), 'fold_models': fold_models})
    if not models: return {'valid': False, 'reason': 'rank_or_spatial_holdout_failure'}
    model = models[0]
    if len(models) > 1 and models[1]['cv'] < .85*model['cv']: model = models[1]
    c, offset, _residual, w = model['fit']; inlier = w > 0
    if inlier.sum() < 12: return {'valid': False, 'reason': 'insufficient_robust_inliers'}
    points = pixels[inlier]
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) < 3 or cv2.contourArea(hull) < 4: return {'valid': False, 'reason': 'inlier_hull_degenerate'}
    gx, gy = np.meshgrid(np.linspace(points[:, 0].min(), points[:, 0].max(), 24),
        np.linspace(points[:, 1].min(), points[:, 1].max(), 24))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.asarray([cv2.pointPolygonTest(hull, tuple(p), False) >= 0 for p in grid])
    distance, _ = cKDTree(points).query(grid)
    grid = grid[inside & (distance <= max(2., .15*np.sqrt(len(points))))]
    queries = np.vstack([points, grid]); a_query = basis((queries-center)/scale_xy, model['degree'])
    k, t = cal.intrinsics, cal.camera_to_world_cv
    ray_up = (queries[:, 0]-k[0, 2])/k[0, 0]*t[2, 0]+(queries[:, 1]-k[1, 2])/k[1, 1]*t[2, 1]+t[2, 2]

    def peak(coefficients, constant):
        predicted = a_query@coefficients+constant
        if not np.isfinite(predicted).all() or np.any(predicted <= 0): return None
        depth = 1/predicted if inverse else predicted
        return float(np.max(ray_up*depth+t[2, 3]-cal.bottom_z_m))

    height = peak(c, offset); fold_heights = [peak(*m) for m in model['fold_models']]
    if height is None or any(v is None for v in fold_heights): return {'valid': False, 'reason': 'nonphysical_model_depth'}
    effective = float(w.sum()**2/max((w*w).sum(), 1e-12))
    fold_std = float(np.std(fold_heights)); noise = float(np.median(sigma_z))
    occupied = len(np.unique(cells[inlier, 0]+3*cells[inlier, 1]))
    stats = {'valid': True, 'height_m': height, 'degree': model['degree'], 'observed_points': len(z),
        'robust_inliers': int(inlier.sum()), 'inlier_fraction': float(inlier.mean()), 'effective_weighted_points': effective,
        'occupied_cells': int(occupied), 'cv_residual_sigma': model['cv'], 'fold_peak_std_m': fold_std,
        'noise_m': noise, 'inferred_query_count': len(grid), 'inverse_depth_domain': inverse,
        'uncertainty_width_m': float(4*fold_std+2*noise/np.sqrt(max(effective, 1))),
        'rejected_observed_points': int((~inlier).sum()), 'curvature_coefficient_norm': float(np.linalg.norm(c[3:])) if len(c) > 3 else 0.}
    # Counts describe raw observations only. Dense queries never increase evidence size.
    return stats


def reconstruct_candidates(candidates, raw, prediction, base):
    cal = height_calibration(base.bundle); output = []
    for candidate in candidates:
        yy, xx = np.nonzero(candidate['mask'])
        if len(xx) < 12: continue
        weights = .05+prediction['origin'][2, yy, xx]*prediction['reliability'][0, yy, xx]
        fitted = fit_observed_surface(raw[yy, xx], np.column_stack([xx, yy]), weights, cal, base.bundle['sensor_family'])
        if not fitted['valid']: continue
        noise = max(candidate['noise_m'], .001)
        extra = [np.log1p(fitted['observed_points']), fitted['inlier_fraction'], np.log1p(fitted['effective_weighted_points']),
            fitted['occupied_cells']/9, fitted['cv_residual_sigma'], fitted['fold_peak_std_m']/noise,
            (fitted['height_m']-candidate['height_raw_m'])/noise, fitted['curvature_coefficient_norm']/noise,
            float(fitted['degree']), fitted['uncertainty_width_m']/noise,
            float(prediction['origin'][2, yy, xx].mean()), float(prediction['origin'][5, yy, xx].mean())]
        output.append({**candidate, 'surface': fitted, 'surface_features': np.concatenate([candidate['features'], extra]).astype(np.float32)})
    return output


def select_surface(candidates, policy, classifier=None):
    if not candidates: return None, []
    scores = classifier.predict_proba(np.stack([c['surface_features'] for c in candidates]))[:, 1] if classifier is not None else [c['object_score'] for c in candidates]
    scored = [{**c, 'object_score': float(score), 'height_raw_m': c['surface']['height_m'], 'correction_m': 0.,
        'interval_width_m': c['surface']['uncertainty_width_m']} for c, score in zip(candidates, scores, strict=True)]
    return select_v5(scored, {**policy, 'correction_fraction': 0., 'consensus_quantile': .5})


class FloatingSurfaceMeterV6:
    """Local geometric experiment; V4 gate remains mandatory, liquid is never overridden."""
    def __init__(self, base, dry, predictor, surface_bundle_path, previous_verifier_path, apex_bundle_path):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.surface_bundle = joblib.load(surface_bundle_path); self.previous = joblib.load(previous_verifier_path)
        self.apex = joblib.load(apex_bundle_path); self.frozen = FrozenSymmetricGeometryMeter(base)
        self.fallback = OptionalFloatMeterV3(base, dry, predictor, previous_verifier_path)
        self._last_reconstructed = []

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='auto'):
        if visual_mode not in {'off', 'auto'}: raise ValueError('peak_only_visual_mode_required')
        if visual_mode == 'off': return self.fallback.measure(depth_m=depth_m, visual_mode='off')
        old = self.frozen.measure(depth_m=depth_m); roi = observed_peak_roi(self.base, depth_m)
        prediction = self.predictor.predict(observed_features(rgb_bgr, depth_m, self.dry, roi, self.base.bundle))
        candidates = candidates_v5(self.base, self.dry, rgb_bgr, depth_m, prediction, old['liquid_depth_m'], roi)
        reconstructed = reconstruct_candidates(candidates, depth_m, prediction, self.base)
        self._last_reconstructed = reconstructed
        chosen, accepted = select_surface(reconstructed,
            self.surface_bundle['policy_by_sensor'][self.base.bundle['sensor_family']], self.surface_bundle['classifier'])
        if chosen is None:
            fallback = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            peak = fallback['highest_height_m']
        else: peak = max(v for v in [chosen['height_m'], old['liquid_depth_m']] if v is not None)
        return {**old, 'highest_height_m': peak, 'visual_module_executed': True,
            'visual_guidance_used': chosen is not None or fallback['visual_guidance_used'], 'liquid_override_used': False,
            'diagnostics': {**old['diagnostics'], 'apex_backend': 'observation_supported_surface_v6',
                'surface_used': chosen is not None, 'fallback_to_unchanged_v4': chosen is None,
                'modeled_candidates': len(reconstructed), 'observed_candidate_count': len(candidates),
                'inferred_samples_are_not_measurements': True,
                'verified_regions': [{k: v for k, v in c.items() if k not in {'mask', 'features', 'surface_features'}} for c in accepted]}}
