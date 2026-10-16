"""Mismatch-aware, observed-support-constrained and edge-preserving peak research route."""
from __future__ import annotations

import cv2
import joblib
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .empty_geometry_v41 import height_calibration
from .float_evidence_v3 import OptionalFloatMeterV3, sensor_sigma
from .floating_apex_v5 import observed_peak_roi, score_v5
from .floating_surface_v6 import basis, robust_fit
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .liquid_height_metrics import world_points
from .rgbd_evidence_model import observed_features


def fast_candidates(base, dry, rgb, raw, prediction, liquid, roi):
    """Same 43-D V5 feature convention; dilation/statistics operate only on local crops."""
    cal = height_calibration(base.bundle); valid = roi & np.isfinite(raw) & (raw > 0)
    heights = world_points(np.where(valid, raw, 0), cal)[..., 2]-cal.bottom_z_m
    sigma = sensor_sigma(np.where(valid, raw, 0), base.bundle)
    sem, origin, reliability = prediction['semantic'], prediction['origin'], prediction['reliability'][0]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(float)/255
    distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 3)
    smooth = median_filter(np.where(valid, heights, np.nan), size=3, mode='nearest')
    masks = [(sem[3] > t) & valid for t in [.2, .5, .75]]
    masks.append(masks[0] & (abs(heights-smooth) < np.maximum(.01, 2.5*sigma)))
    initial = []
    for variant, mask in enumerate(masks):
        count, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] >= 2: initial.append((cc == label, variant))
    layers = []
    for component, _ in initial:
        if component.sum() < 12 or np.median(sigma[component]) >= .01: continue
        z = heights[component]; noise = max(.001, float(np.median(sigma[component]))); step = max(.004, 2*noise)
        edges = np.arange(z.min()-step, z.max()+2*step, step)
        if len(edges) < 5 or len(edges) > 256: continue
        counts, _ = np.histogram(z, edges); peaks, _ = find_peaks(counts, height=max(3, .03*component.sum()), distance=2)
        yy, xx = np.nonzero(component); sy = slice(yy.min(), yy.max()+1); sx = slice(xx.min(), xx.max()+1)
        for index in sorted(peaks, key=lambda i: -counts[i])[:4]:
            mask = component[sy, sx] & (abs(heights[sy, sx]-.5*(edges[index]+edges[index+1])) <= max(.008, 3*noise))
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] < 3: continue
                full = np.zeros(raw.shape, bool); full[sy, sx] = cc == label; layers.append((full, 4))
    mass = sem[3]*origin[2]*reliability
    initial = sorted(initial+layers, key=lambda item: -float(mass[item[0]].sum()))
    candidates = []; seen = set()
    for component, variant in initial:
        indices = np.flatnonzero(component); signature = (len(indices), int(indices[0]), int(indices[-1]), int(indices.sum()))
        if signature in seen: continue
        seen.add(signature)
        if len(candidates) >= 80: break
        yy, xx = np.nonzero(component); x0, x1 = int(xx.min()), int(xx.max()+1); y0, y1 = int(yy.min()), int(yy.max()+1)
        sy = slice(max(0, y0-4), min(raw.shape[0], y1+4)); sx = slice(max(0, x0-4), min(raw.shape[1], x1+4))
        local = component[sy, sx]; ring = cv2.dilate(local.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~local & valid[sy, sx]
        if ring.sum() < 8: continue
        w, h = x1-x0, y1-y0; z = heights[yy, xx]; around = heights[sy, sx][ring]
        noise = max(.001, float(np.median(sigma[yy, xx]))); center = float(np.median(around)); mad = float(np.median(abs(around-center)))
        qs = np.percentile(z, [10, 50, 75, 90, 95, 100]); peak = float(qs[3]); prior = liquid if liquid is not None else center
        f = [np.log1p(len(z)), w/max(h, 1), len(z)/(w*h), float(distance[yy, xx].min())/max(raw.shape), noise,
            peak, center, peak-center, (peak-center)/noise, mad/noise, (peak-prior)/noise, prior,
            *(((qs-peak)/noise).tolist()), float(sem[3, yy, xx].mean()), float(sem[3, yy, xx].max()),
            float(origin[2, yy, xx].mean()), float(origin[5, yy, xx].mean()), float(origin[6, yy, xx].mean()),
            float(reliability[yy, xx].mean()), float(sem[3, sy, sx][ring].mean()), float(origin[2, sy, sx][ring].mean()),
            float(gray[yy, xx].mean()-gray[sy, sx][ring].mean()), float((gray[yy, xx] > .97).mean()), float((gray[yy, xx] < .03).mean()),
            float((abs(z-peak) <= 2*noise).mean()), float(np.median(np.asarray(dry)[yy, xx]-raw[yy, xx])),
            *[float(base.bundle['sensor_family'] == key) for key in ['tof', 'active_stereo', 'structured_light']]]
        local_smooth = smooth[yy, xx]; local_smooth = local_smooth[np.isfinite(local_smooth)]
        sq = np.percentile(local_smooth, [50, 90, 95]) if len(local_smooth) else np.full(3, peak)
        extra = [float(variant), *(((sq-peak)/noise).tolist()), float(np.median(abs(z-np.median(z)))/noise),
            float(base.ordinary_container_roi[yy, xx].mean()), float(sem[3, yy, xx].min()), float(reliability[yy, xx].min()), peak-float(base.bundle['rim_z_m'])]
        features = np.asarray(f+extra, np.float32)
        if np.isfinite(features).all(): candidates.append({'mask': component, 'indices': indices, 'features': features,
            'height_raw_m': peak, 'noise_m': noise, 'pixels': len(z), 'variant': variant, 'bbox_xyxy_px': [x0, y0, x1, y1]})
    return candidates


def fit_patch(raw_values, pixels, weights, cal, bundle, degree=1):
    """Small sensor-domain fit with explicit residual/misspecification diagnostics."""
    z = np.asarray(raw_values, float); pixels = np.asarray(pixels, float); weights = np.asarray(weights, float)
    valid = np.isfinite(z) & (z > 0) & np.isfinite(weights) & (weights > 0)
    z, pixels, weights = z[valid], pixels[valid], weights[valid]
    if len(z) < max(12, 5*(degree+1)): return {'valid': False, 'reason': 'observed_support'}
    if len(z) > 512:
        sub = np.linspace(0, len(z)-1, 512).astype(int); z, pixels, weights = z[sub], pixels[sub], weights[sub]
    span = np.ptp(pixels, axis=0)
    if span.min() < 2: return {'valid': False, 'reason': 'degenerate_support'}
    center = (pixels.max(0)+pixels.min(0))*.5; xy = (pixels-center)/np.maximum(span*.5, 1.)
    noise_z = sensor_sigma(z, bundle); inverse = bundle['sensor_family'] != 'tof'
    values = 1/z if inverse else z; sigma = noise_z/(z*z) if inverse else noise_z
    a = basis(xy, degree); result = robust_fit(a, values, sigma, weights)
    if result is None: return {'valid': False, 'reason': 'rank'}
    c, offset, residual, w = result; inlier = w > 0
    if inlier.sum() < 12: return {'valid': False, 'reason': 'inlier_support'}
    cells = np.floor((pixels-pixels.min(0))/3).astype(int); occupied = len(np.unique(cells, axis=0))
    spatial = np.floor((xy+1)*1.499).astype(int); fold_ids = (spatial[:, 0]+spatial[:, 1]) % 3
    cv = []; fold_peaks = []
    k, t = cal.intrinsics, cal.camera_to_world_cv
    ray = (pixels[:, 0]-k[0, 2])/k[0, 0]*t[2, 0]+(pixels[:, 1]-k[1, 2])/k[1, 1]*t[2, 1]+t[2, 2]

    def heights(coefficients, constant):
        pred = a@coefficients+constant
        if not np.isfinite(pred).all() or np.any(pred <= 0): return None
        depths = 1/pred if inverse else pred
        return ray*depths+t[2, 3]-cal.bottom_z_m

    modeled = heights(c, offset)
    if modeled is None: return {'valid': False, 'reason': 'nonphysical_depth'}
    for fold in range(3):
        use = fold_ids != fold; test = ~use
        if not test.any(): continue
        fitted = robust_fit(a[use], values[use], sigma[use], weights[use])
        if fitted is None: continue
        cc, oo, _, _ = fitted; hp = heights(cc, oo)
        if hp is None: continue
        cv.extend((abs(a[test]@cc+oo-values[test])/sigma[test]).tolist()); fold_peaks.append(float(hp[inlier].max()))
    if len(fold_peaks) < 2: return {'valid': False, 'reason': 'holdout_failure'}
    peak = float(modeled[inlier].max()); cv_sigma = float(np.median(cv)); noise = float(np.median(noise_z))
    original_height = ray*z+t[2, 3]-cal.bottom_z_m
    witness = inlier & (abs(original_height-peak) <= 3*noise_z)
    witness_cells = len(np.unique(cells[witness], axis=0)); fold_std = float(np.std(fold_peaks))
    # Residual/model mismatch is NOT allowed to disappear merely because folds agree.
    systematic = noise*max(0., cv_sigma-1.)
    uncertainty = 4*fold_std+2*noise/np.sqrt(max(occupied, 1))+2*systematic
    return {'valid': True, 'height_m': peak, 'degree': degree, 'observed_points': len(z),
        'inlier_fraction': float(inlier.mean()), 'occupied_3px_cells': occupied, 'witness_points': int(witness.sum()),
        'witness_cells': witness_cells, 'cv_sigma': cv_sigma, 'residual_sigma': float(np.median(abs(residual))),
        'fold_std_m': fold_std, 'noise_m': noise, 'uncertainty_width_m': float(uncertainty),
        'systematic_residual_m': float(systematic), 'pixel_span_min': float(span.min()), 'inferred_queries': 0}


def model_allowed(model, *, max_cv=3., min_cells=4):
    return bool(model.get('valid') and model['cv_sigma'] <= max_cv and model['residual_sigma'] <= max_cv and
        model['inlier_fraction'] >= .6 and model['occupied_3px_cells'] >= min_cells and model['witness_cells'] >= min_cells)


def fit_candidate_models(candidates, raw, prediction, base, *, piecewise=True):
    cal = height_calibration(base.bundle); output = []
    # Prioritize object evidence, never raw height. Bound expensive fits, not depth values.
    ranked = sorted(candidates, key=lambda c: -c['object_score']*np.sqrt(c['pixels']))
    for candidate in ranked[:28]:
        if candidate['pixels'] < 12 or candidate['object_score'] < .3: continue
        yy, xx = np.nonzero(candidate['mask']); pixels = np.column_stack([xx, yy]).astype(float)
        z = raw[yy, xx]; weights = .05+prediction['origin'][2, yy, xx]*prediction['reliability'][0, yy, xx]
        global_model = fit_patch(z, pixels, weights, cal, base.bundle)
        raw_height = candidate['corrected_raw_m']; models = []
        if global_model.get('valid'):
            sigma_z = sensor_sigma(z, base.bundle)
            k, t = cal.intrinsics, cal.camera_to_world_cv
            ray = (pixels[:, 0]-k[0, 2])/k[0, 0]*t[2, 0]+(pixels[:, 1]-k[1, 2])/k[1, 1]*t[2, 1]+t[2, 2]
            raw_witness = abs(ray*z+t[2, 3]-cal.bottom_z_m-raw_height) <= 3*sigma_z
            cells = np.floor((pixels-pixels.min(0))/3).astype(int)
            models.append(('guarded_raw', {**global_model, 'height_m': raw_height, 'witness_points': int(raw_witness.sum()),
                'witness_cells': len(np.unique(cells[raw_witness], axis=0))}))
            models.append(('global_surface', global_model))
        if piecewise and len(z) >= 36 and global_model.get('valid'):
            # Locate a change in observed slope; do not smooth through the resulting crease.
            inverse = base.bundle['sensor_family'] != 'tof'; values = 1/z if inverse else z
            sigma_z = sensor_sigma(z, base.bundle); domain_noise = np.median(sigma_z/(z*z) if inverse else sigma_z)
            bends = []
            for axis in [0, 1]:
                cuts = np.quantile(pixels[:, axis], [0., 1/3, 2/3, 1.]); locations = []; medians = []
                for i in range(3):
                    use = (pixels[:, axis] >= cuts[i]) & (pixels[:, axis] <= cuts[i+1])
                    locations.append(float(np.median(pixels[use, axis]))); medians.append(float(np.median(values[use])))
                separation = np.maximum(np.diff(locations), 1.)
                slopes = np.diff(medians)/separation
                bends.append(float(abs(slopes[1]-slopes[0])*np.ptp(pixels[:, axis])/max(domain_noise, 1e-9)))
            axis = int(np.argmax(bends)); pivot = float(np.median(pixels[:, axis]))
            subsets = [pixels[:, axis] <= pivot, pixels[:, axis] >= pivot] if max(bends) > 2.5 else []
            for subset in subsets:
                if subset.sum() < 12: continue
                model = fit_patch(z[subset], pixels[subset], weights[subset], cal, base.bundle)
                if model.get('valid'): models.append(('local_facet', model))
        for kind, model in models:
            noise = max(candidate['noise_m'], .001)
            extra = [float(kind != 'guarded_raw'), float(kind == 'local_facet'), np.log1p(model['observed_points']),
                model['inlier_fraction'], np.log1p(model['occupied_3px_cells']), np.log1p(model['witness_points']),
                np.log1p(model['witness_cells']), model['cv_sigma'], model['residual_sigma'], model['fold_std_m']/noise,
                model['uncertainty_width_m']/noise, (model['height_m']-raw_height)/noise, model['pixel_span_min'],
                model['systematic_residual_m']/noise, float(model['degree'])]
            output.append({**candidate, 'kind': kind, 'model': model,
                'quality_features': np.concatenate([candidate['features'], extra]).astype(np.float32),
                'height_m': model['height_m']})
    return output


def select_models(models, policy, classifier=None):
    if not models: return None, []
    scores = classifier.predict_proba(np.stack([m['quality_features'] for m in models]))[:, 1] if classifier is not None else [m.get('quality_score', m['object_score']) for m in models]
    accepted = []
    for m, score in zip(models, scores, strict=True):
        if policy['mode'] == 'guarded_raw' and m['kind'] != 'guarded_raw': continue
        if policy['mode'] == 'surface' and m['kind'] == 'guarded_raw': continue
        if not model_allowed(m['model'], max_cv=policy['max_cv'], min_cells=policy['min_cells']): continue
        if score < policy['threshold']: continue
        if m['model']['uncertainty_width_m'] > policy['interval_factor']*max(.05, .05*abs(m['height_m'])): continue
        accepted.append({**m, 'quality_score': float(score)})
    # A valid high facet is not averaged with a lower facet of the same object.
    return (max(accepted, key=lambda m: m['height_m']) if accepted else None), accepted


class FloatingSurfaceMeterV7:
    def __init__(self, base, dry, predictor, bundle_path, previous_path, apex_path):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.bundle = joblib.load(bundle_path); self.previous = joblib.load(previous_path); self.apex = joblib.load(apex_path)
        self.frozen = FrozenSymmetricGeometryMeter(base); self.fallback = OptionalFloatMeterV3(base, dry, predictor, previous_path)
        self._last_models = []; self._last_candidates = []

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='auto'):
        if visual_mode not in {'off', 'auto'}: raise ValueError('peak_only_visual_mode_required')
        if visual_mode == 'off': return self.fallback.measure(depth_m=depth_m, visual_mode='off')
        sensor = self.base.bundle['sensor_family']; policy = self.bundle['policy_by_sensor'][sensor]
        self._last_models = []; self._last_candidates = []
        if policy['threshold'] > 1:
            old = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            return {**old, 'diagnostics': {**old['diagnostics'], 'v7_used': False, 'policy_bypass': True, 'v7_models': []}}
        old = self.frozen.measure(depth_m=depth_m); roi = observed_peak_roi(self.base, depth_m)
        prediction = self.predictor.predict(observed_features(rgb_bgr, depth_m, self.dry, roi, self.base.bundle))
        candidates = score_v5(fast_candidates(self.base, self.dry, rgb_bgr, depth_m, prediction, old['liquid_depth_m'], roi), self.apex['classifier'], self.previous)
        fraction = self.apex['policy_by_sensor'][sensor]['correction_fraction']
        for c in candidates: c['corrected_raw_m'] = c['height_raw_m']+fraction*c['correction_m']
        models = fit_candidate_models(candidates, depth_m, prediction, self.base, piecewise=policy['mode'] != 'guarded_raw')
        self._last_candidates = candidates; self._last_models = models
        chosen, accepted = select_models(models, policy, self.bundle['classifier'])
        if chosen is None:
            result = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto'); peak = result['highest_height_m']; used = result['visual_guidance_used']
        else: peak = max(v for v in [chosen['height_m'], old['liquid_depth_m']] if v is not None); used = True
        return {**old, 'highest_height_m': peak, 'visual_module_executed': True, 'visual_guidance_used': used, 'liquid_override_used': False,
            'diagnostics': {**old['diagnostics'], 'apex_backend': 'mismatch_guarded_facets_v7', 'v7_used': chosen is not None,
                'policy_bypass': False, 'original_candidate_count': len(candidates), 'fitted_model_count': len(models),
                'model_rejections': sum(int(not model_allowed(m['model'], max_cv=policy['max_cv'], min_cells=policy['min_cells'])) for m in models),
                'inferred_points_are_not_measurements': True,
                'v7_models': [{k: v for k, v in m.items() if k not in {'mask', 'indices', 'features', 'quality_features'}} for m in accepted]}}
