"""Peak-only research backend: observed-geometry proposals and validation-shrunk correction."""
from __future__ import annotations

import cv2
import joblib
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .empty_geometry_v41 import height_calibration
from .float_evidence_v3 import sensor_sigma
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .liquid_height_metrics import world_points
from .rgbd_evidence_model import observed_features
from .uneven_surface_meter import UnevenSurfaceMeter


def observed_peak_roi(base, raw):
    cal = height_calibration(base.bundle)
    valid = np.isfinite(raw) & (raw > 0)
    xyz = world_points(np.where(valid, raw, 0), cal)
    footprint = (abs(xyz[..., :2]/np.asarray(cal.radii_m))**cal.footprint_exponent).sum(-1)
    # Floating tops may project outside the DRY floor ROI or protrude above the rim.
    # No rim-height cap and no current true liquid/float geometry is used here.
    return base.ordinary_container_roi | (valid & (footprint <= 1.05**cal.footprint_exponent))


def candidates_v5(base, dry, rgb, raw, prediction, liquid_height, roi):
    cal = height_calibration(base.bundle); valid = roi & np.isfinite(raw) & (raw > 0)
    heights = world_points(np.where(valid, raw, 0), cal)[..., 2]-cal.bottom_z_m
    sigma = sensor_sigma(np.where(valid, raw, 0), base.bundle)
    semantic, origin, reliability = prediction['semantic'], prediction['origin'], prediction['reliability'][0]
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(float)/255
    distance = cv2.distanceTransform(roi.astype(np.uint8), cv2.DIST_L2, 3)
    local = median_filter(np.where(valid, heights, np.nan), size=3, mode='nearest')
    masks = [(semantic[3] > threshold) & valid for threshold in [.2, .5, .75]]
    masks.append(masks[0] & (abs(heights-local) < np.maximum(.01, 2.5*sigma)))
    initial = []
    for variant, mask in enumerate(masks):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        for label in range(1, n):
            if stats[label, cv2.CC_STAT_AREA] >= 2: initial.append((cc == label, variant))
    # Low-noise observations can separate connected wall/float returns by depth.
    # This proposal is disabled at high range noise to avoid a noise-selected plane.
    layer_proposals = []
    for component, _ in initial[:len(initial)]:
        if component.sum() < 12 or np.median(sigma[component]) >= .01: continue
        values = heights[component]; noise = max(.001, float(np.median(sigma[component])))
        step = max(.004, 2*noise); edges = np.arange(values.min()-step, values.max()+2*step, step)
        if len(edges) < 5 or len(edges) > 256: continue
        counts, _ = np.histogram(values, edges); peaks, _ = find_peaks(counts, height=max(3, .03*component.sum()), distance=2)
        for index in sorted(peaks, key=lambda k: -counts[k])[:4]:
            center = .5*(edges[index]+edges[index+1]); mask = component & (abs(heights-center) <= max(.008, 3*noise))
            n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] >= 3: layer_proposals.append((cc == label, 4))
    initial += layer_proposals
    ranked = sorted(initial, key=lambda item: -float((semantic[3]*origin[2]*reliability)[item[0]].sum()))
    candidates = []; seen = set()
    for component, variant in ranked:
        indices = np.flatnonzero(component)
        signature = (len(indices), int(indices[0]), int(indices[-1]), int(indices.sum()))
        if signature in seen: continue
        seen.add(signature)
        if len(candidates) >= 80: break
        ring = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~component & valid
        if ring.sum() < 8: continue
        yy, xx = np.nonzero(component); w, h = int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)
        z = heights[component]; around = heights[ring]; noise = max(.001, float(np.median(sigma[component])))
        center = float(np.median(around)); mad = float(np.median(abs(around-center)))
        qs = np.percentile(z, [10, 50, 75, 90, 95, 100]); peak = float(qs[3]); prior = liquid_height if liquid_height is not None else center
        f = [np.log1p(len(z)), w/max(h, 1), len(z)/(w*h), float(distance[component].min())/max(raw.shape), noise,
            peak, center, peak-center, (peak-center)/noise, mad/noise, (peak-prior)/noise, prior,
            *(((qs-peak)/noise).tolist()), float(semantic[3, component].mean()), float(semantic[3, component].max()),
            float(origin[2, component].mean()), float(origin[5, component].mean()), float(origin[6, component].mean()),
            float(reliability[component].mean()), float(semantic[3, ring].mean()), float(origin[2, ring].mean()),
            float(gray[component].mean()-gray[ring].mean()), float((gray[component] > .97).mean()),
            float((gray[component] < .03).mean()), float((abs(z-peak) <= 2*noise).mean()),
            float(np.median(np.asarray(dry)[component]-raw[component])),
            *[float(base.bundle['sensor_family'] == key) for key in ['tof', 'active_stereo', 'structured_light']]]
        smooth = local[component]; smooth = smooth[np.isfinite(smooth)]
        smoothqs = np.percentile(smooth, [50, 90, 95]) if len(smooth) else np.array([peak]*3)
        extra = [float(variant), *(((smoothqs-peak)/noise).tolist()),
            float(np.median(abs(z-np.median(z)))/noise), float((component & base.ordinary_container_roi).sum()/len(z)),
            float(semantic[3, component].min()), float(reliability[component].min()),
            float(peak-float(base.bundle['rim_z_m']))]
        features = np.asarray(f+extra, np.float32)
        if not np.isfinite(features).all(): continue
        candidates.append({'mask': component, 'features': features, 'height_raw_m': peak,
            'noise_m': noise, 'pixels': len(z), 'variant': variant,
            'bbox_xyxy_px': [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]})
    return candidates


def score_v5(candidates, classifier, previous):
    if not candidates: return []
    features = np.stack([c['features'] for c in candidates]); probabilities = classifier.predict_proba(features)[:, 1]
    original_features = features[:, :34]
    corrections = previous['regressor'].predict(original_features)
    lower = previous['lower'].predict(original_features); upper = previous['upper'].predict(original_features)
    return [{**c, 'object_score': float(p), 'correction_m': float(d*c['noise_m']),
        'interval_width_m': float((max(lo, hi, d)-min(lo, hi, d))*c['noise_m'])}
        for c, p, d, lo, hi in zip(candidates, probabilities, corrections, lower, upper, strict=True)]


def select_v5(scored, policy):
    accepted = []
    for c in scored:
        height = c['height_raw_m']+policy['correction_fraction']*c['correction_m']
        if c['object_score'] >= policy['threshold'] and c['interval_width_m'] <= policy['interval_factor']*max(.05, .05*abs(height)):
            accepted.append({**c, 'height_m': float(height)})
    # Nested masks describe the same object. Consensus avoids maximizing many correlated trials.
    groups = []
    for c in sorted(accepted, key=lambda c: -c['object_score']):
        for group in groups:
            reference = group[0]; overlap = (c['mask'] & reference['mask']).sum()
            if overlap/max(min(c['pixels'], reference['pixels']), 1) >= .65:
                group.append(c); break
        else: groups.append([c])
    peaks = []
    for group in groups:
        height = float(np.quantile([c['height_m'] for c in group], policy['consensus_quantile']))
        representative = max(group, key=lambda c: c['object_score'])
        peaks.append({**representative, 'height_m': height, 'proposal_agreement': len(group)})
    return (max(peaks, key=lambda c: c['height_m']) if peaks else None), peaks


class FloatingApexMeterV5:
    """Highest-only backend; use via FloatingPeakOnlyMeter's unchanged scene gate."""
    def __init__(self, base, dry, predictor, bundle_path, previous_verifier_path):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.bundle = joblib.load(bundle_path); self.previous = joblib.load(previous_verifier_path)
        self.frozen = FrozenSymmetricGeometryMeter(base); self.geometric = UnevenSurfaceMeter(base, dry)
        self._last_scored = []  # Offline evaluator may inspect masks AFTER measurement.

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='auto'):
        old = self.frozen.measure(depth_m=depth_m); peak = self.geometric.measure(depth_m)['highest_visible_height_m']
        if visual_mode not in {'off', 'auto'}: raise ValueError('peak_only_visual_mode_required')
        accepted = []; count = 0; roi = self.base.ordinary_container_roi
        self._last_scored = []
        if visual_mode == 'auto':
            roi = observed_peak_roi(self.base, depth_m)
            prediction = self.predictor.predict(observed_features(rgb_bgr, depth_m, self.dry, roi, self.base.bundle))
            candidates = candidates_v5(self.base, self.dry, rgb_bgr, depth_m, prediction, old['liquid_depth_m'], roi)
            scored = score_v5(candidates, self.bundle['classifier'], self.previous); count = len(scored)
            self._last_scored = scored
            chosen, accepted = select_v5(scored, self.bundle['policy_by_sensor'][self.base.bundle['sensor_family']])
            if chosen is not None: peak = max([v for v in [chosen['height_m'], old['liquid_depth_m']] if v is not None])
        diagnostics = {**old['diagnostics'], 'highest_source': 'observed_geometry_float_consensus' if accepted else 'legacy_geometric_peak',
            'apex_backend': 'floating_apex_v5', 'sensor_policy': self.bundle['policy_by_sensor'][self.base.bundle['sensor_family']],
            'verified_regions': [{k: v for k, v in c.items() if k not in {'mask', 'features'}} for c in accepted],
            'candidate_count': count, 'expanded_observed_roi_pixels': int((roi & ~self.base.ordinary_container_roi).sum()),
            'original_metric_depth_samples_only': True, 'liquid_visual_override_allowed': False}
        return {**old, 'highest_height_m': peak, 'diagnostics': diagnostics,
            'visual_module_executed': visual_mode == 'auto', 'visual_guidance_used': bool(accepted), 'liquid_override_used': False}
