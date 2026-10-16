"""Highest-instance ranking and continuous metric-risk research backend.

No simulator labels enter this module. Corrections are learned priors, not a
claim that common-mode hardware bias is observable from one smooth patch.
"""
from __future__ import annotations

import cv2
import joblib
import numpy as np

from .empty_geometry_v41 import height_calibration
from .float_evidence_v3 import OptionalFloatMeterV3, sensor_sigma
from .floating_apex_v5 import observed_peak_roi, score_v5, select_v5
from .floating_surface_v7 import fast_candidates
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .liquid_height_metrics import world_points
from .rgbd_evidence_model import observed_features
from .uneven_surface_meter import UnevenSurfaceMeter


def proposals(base, dry, rgb, raw, prediction, liquid, roi, previous, apex, *, include_relaxed=True):
    original = score_v5(fast_candidates(base, dry, rgb, raw, prediction, liquid, roi), apex['classifier'], previous)
    if not include_relaxed:
        return original, original
    # A second relaxed semantic pass adds small/missed instances. These remain
    # original metric pixels; object evidence and risk are checked downstream.
    sem = prediction['semantic'].copy()
    sem[3] = np.minimum(1., sem[3]/.4)
    relaxed = fast_candidates(base, dry, rgb, raw, {**prediction, 'semantic': sem}, liquid, roi)
    seen = {tuple(c['indices']) for c in original}
    added = []
    for candidate in relaxed:
        signature = tuple(candidate['indices'])
        if signature in seen:
            continue
        seen.add(signature)
        # Recompute original semantic features, never pass inflated scores to heads.
        yy, xx = np.nonzero(candidate['mask'])
        f = candidate['features'].copy()
        f[18] = prediction['semantic'][3, yy, xx].mean()
        f[19] = prediction['semantic'][3, yy, xx].max()
        f[40] = prediction['semantic'][3, yy, xx].min()
        sy = slice(max(0, yy.min()-4), min(raw.shape[0], yy.max()+5))
        sx = slice(max(0, xx.min()-4), min(raw.shape[1], xx.max()+5))
        component = candidate['mask'][sy, sx]
        ring = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        ring &= ~component & roi[sy, sx] & np.isfinite(raw[sy, sx]) & (raw[sy, sx] > 0)
        f[24] = prediction['semantic'][3, sy, sx][ring].mean()
        added.append({**candidate, 'features': f})
    added = score_v5(added, apex['classifier'], previous)
    return original, original+[{**c, 'relaxed': True} for c in added]


def candidate_features(candidates, base, dry, raw, prediction):
    """Metric and frame-context features independent of target liquid/float labels."""
    if not candidates:
        return np.empty((0, 67), np.float32)
    cal = height_calibration(base.bundle)
    valid = np.isfinite(raw) & (raw > 0)
    dry_valid = np.isfinite(dry) & (dry > 0)
    h_dry = world_points(np.where(dry_valid, dry, 0), cal)[..., 2]-cal.bottom_z_m
    sigma = sensor_sigma(np.where(valid, raw, 0), base.bundle)
    sem = prediction['semantic']
    stable = valid & dry_valid & (sem[1] > .8) & (sem[3] < .05)
    stable &= abs(raw-dry) <= np.maximum(.02, 3*sigma)
    delta = raw[stable]-dry[stable]
    static = [float(np.median(delta)), float(np.median(abs(delta-np.median(delta)))), np.log1p(len(delta))] if len(delta) >= 16 else [0., 0., 0.]
    credible = [c['height_raw_m'] for c in candidates if c['object_score'] >= .5]
    upper = float(np.quantile(credible, .9)) if credible else float(np.median([c['height_raw_m'] for c in candidates]))
    masses = np.asarray([c['pixels']*max(c['object_score'], .01) for c in candidates])
    features = []
    for c, mass in zip(candidates, masses, strict=True):
        yy, xx = np.nonzero(c['mask'])
        f = c['features']
        cells = np.floor(np.column_stack([xx, yy])/3).astype(int)
        point_noise = sigma[yy, xx]
        good = dry_valid[yy, xx]
        dry_h = h_dry[yy[good], xx[good]]
        depth = raw[yy, xx]
        extra = [c['object_score'], c['correction_m'], c['interval_width_m'], float(c.get('relaxed', False)),
            np.log1p(len(np.unique(cells, axis=0))), float(np.median(depth)),
            float(np.median(1/depth)), float(np.percentile(1/depth, 90)),
            float(np.median(dry_h)) if len(dry_h) else 0., float(np.std(dry_h)) if len(dry_h) else 0.,
            float(good.mean()), *static, upper-c['height_raw_m'],
            (upper-c['height_raw_m'])/max(c['noise_m'], .01), mass/max(masses.sum(), 1.),
            float(np.mean(np.asarray(credible) <= c['height_raw_m'])) if credible else .5,
            float(prediction['origin'][2, yy, xx].min()), float(np.median(point_noise)),
            float(np.std(depth)), float(np.percentile(depth, 10)), float(np.percentile(depth, 90)),
            float(min(xx.max()-xx.min()+1, yy.max()-yy.min()+1))]
        features.append(np.r_[f, extra])
    result = np.asarray(features, np.float32)
    assert result.shape[1] == 67 and np.isfinite(result).all()
    return result


def score_candidates(candidates, features, bundle, *, correction=True, ranking=True):
    if not candidates:
        return []
    scales = np.asarray([max(.01, c['noise_m']) for c in candidates])
    probability = bundle['object_classifier'].predict_proba(features)[:, 1]
    delta = bundle['correction'].predict(features) if correction else np.zeros(len(candidates))
    height = np.asarray([c['height_raw_m'] for c in candidates])+delta*scales
    risk_x = np.column_stack([features, delta, height])
    # Regression can numerically predict a negative nonnegative risk; mapping
    # risk to >=0 does not clip a measured height.
    risk = np.maximum(0., bundle['risk'].predict(risk_x))*scales
    gap = np.maximum(0., bundle['top_gap'].predict(risk_x))*scales if ranking else np.zeros(len(candidates))
    out = []
    for index, (c, h, d, p, r, g) in enumerate(zip(candidates, height, delta, probability, risk, gap, strict=True)):
        # The exact observed 3-pixel-cell count was already encoded in feature47.
        # This also permits compact offline caches without image-size masks.
        cells = c.get('cells', round(np.expm1(features[index, 47])))
        out.append({**c, 'height_m': float(h), 'learned_correction_m': float(d*max(.01, c['noise_m'])),
            'physical_score': float(p), 'risk_m': float(r), 'top_gap_m': float(g), 'cells': cells})
    return out


def select_ranked(scored, policy):
    eligible = [c for c in scored if c['pixels'] >= policy['min_pixels'] and c['cells'] >= policy['min_cells']
        and c['physical_score'] >= policy['physical_threshold'] and c['risk_m'] <= policy['max_risk_m']]
    # Deduplicate nested masks before selecting the highest instance. Never
    # maximize repeated noisy proposals for the same object independently.
    groups = []
    for c in sorted(eligible, key=lambda v: -v['physical_score']):
        for group in groups:
            ref = group[0]
            if len(np.intersect1d(c['indices'], ref['indices'], assume_unique=True))/max(min(c['pixels'], ref['pixels']), 1) >= .65:
                group.append(c)
                break
        else:
            groups.append([c])
    instances = []
    for group in groups:
        representative = min(group, key=lambda c: c['risk_m']+.1*c['top_gap_m'])
        h = float(np.median([c['height_m'] for c in group]))
        risk = float(np.median([c['risk_m'] for c in group]))
        gap = float(np.median([c['top_gap_m'] for c in group]))
        utility = h-policy['risk_weight']*risk-policy['gap_weight']*gap
        instances.append({**representative, 'height_m': h, 'risk_m': risk, 'top_gap_m': gap,
            'utility': utility, 'proposal_agreement': len(group)})
    chosen = max(instances, key=lambda c: c['utility']) if instances else None
    return chosen, instances


def fast_v5_peak(original, apex, sensor, liquid, geometric):
    chosen, _ = select_v5(original, apex['policy_by_sensor'][sensor])
    return geometric() if chosen is None else max(v for v in [liquid, chosen['height_m']] if v is not None)


class FloatingRankMeterV8:
    def __init__(self, base, dry, predictor, model_path, previous_path, apex_path, *, policy_override=None):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.bundle = joblib.load(model_path)
        self.previous, self.apex = joblib.load(previous_path), joblib.load(apex_path)
        self.frozen = FrozenSymmetricGeometryMeter(base)
        self.geometric = UnevenSurfaceMeter(base, dry)
        self.fallback = OptionalFloatMeterV3(base, dry, predictor, previous_path)
        self.policy_override = policy_override
        self._last_scored = []
        self._last_chosen = None

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='auto'):
        if visual_mode not in {'auto', 'off'}:
            raise ValueError('peak_only_visual_mode_required')
        if visual_mode == 'off':
            return self.fallback.measure(depth_m=depth_m, visual_mode='off')
        self._last_chosen = None
        self._last_scored = []
        policy = self.policy_override or self.bundle['policy_by_sensor'][self.base.bundle['sensor_family']]
        if policy.get('bypass') == 'v4':
            result = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            return {**result, 'diagnostics': {**result['diagnostics'], 'v8_used': False, 'v8_policy': policy, 'v8_instances': []}}
        old = self.frozen.measure(depth_m=depth_m)
        roi = observed_peak_roi(self.base, depth_m)
        prediction = self.predictor.predict(observed_features(rgb_bgr, depth_m, self.dry, roi, self.base.bundle))
        original, candidates = proposals(self.base, self.dry, rgb_bgr, depth_m, prediction, old['liquid_depth_m'], roi, self.previous, self.apex,
            include_relaxed=not policy.get('bypass'))
        chosen = None
        instances = []
        self._last_scored = []
        if not policy.get('bypass'):
            x = candidate_features(candidates, self.base, self.dry, depth_m, prediction)
            self._last_scored = score_candidates(candidates, x, self.bundle)
            chosen, instances = select_ranked(self._last_scored, policy)
        self._last_chosen = chosen
        if chosen is not None:
            peak = max(v for v in [chosen['height_m'], old['liquid_depth_m']] if v is not None)
        elif policy['fallback'] == 'v5':
            peak = fast_v5_peak(original, self.apex, self.base.bundle['sensor_family'], old['liquid_depth_m'],
                lambda: self.geometric.measure(depth_m)['highest_visible_height_m'])
        else:
            peak = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')['highest_height_m']
        return {**old, 'highest_height_m': peak, 'visual_module_executed': True, 'visual_guidance_used': True, 'liquid_override_used': False,
            'diagnostics': {**old['diagnostics'], 'apex_backend': 'instance_risk_rank_v8', 'v8_used': chosen is not None,
                'v8_policy': policy, 'original_candidates': len(original), 'all_candidates': len(candidates),
                'v8_instances': [{k: v for k, v in c.items() if k not in {'mask', 'indices', 'features'}} for c in instances],
                'chosen_bbox': chosen['bbox_xyxy_px'] if chosen else None, 'original_metric_pixels_only': True}}
