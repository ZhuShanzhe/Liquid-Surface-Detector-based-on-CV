"""Optional original-pixel float evidence with multiscale features and region verification."""
import cv2
import joblib
import numpy as np
import torch

from .empty_geometry_v41 import height_calibration
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .liquid_height_metrics import world_points
from .rgbd_evidence_model import RGBDEvidencePredictor, observed_features
from .surface_refinement import StereoNoiseModel
from .uneven_surface_meter import UnevenSurfaceMeter


class MultiScaleEvidencePredictor(RGBDEvidencePredictor):
    @torch.inference_mode()
    def predict(self, features):
        h, w = features.shape[-2:]
        views = [(0, 0, w, h)]
        # Overlap avoids a blind seam; enlargement provides features, never new depth samples.
        cw, ch = int(w * .6), int(h * .6)
        views += [(x, y, x + cw, y + ch) for y in [0, h-ch] for x in [0, w-cw]]
        batch = np.stack([cv2.resize(features[:, y:y2, x:x2].transpose(1, 2, 0),
                         (w, h)).transpose(2, 0, 1) for x, y, x2, y2 in views])
        output = self.net(torch.from_numpy(batch.astype(np.float32)).to(self.device))
        values = {k: ((v / self.temperature.get(k, 1.)).sigmoid() if k == 'reliability'
                     else (v / self.temperature.get(k, 1.)).softmax(1)).cpu().numpy()
                  for k, v in output.items()}
        merged = {k: v[0].copy() for k, v in values.items()}
        for index, (x, y, x2, y2) in enumerate(views[1:], 1):
            crop = {k: cv2.resize(v[index].transpose(1, 2, 0), (x2-x, y2-y)).reshape(
                    y2-y, x2-x, v.shape[1]).transpose(2, 0, 1) for k, v in values.items()}
            old = {k: v[:, y:y2, x:x2] for k, v in merged.items()}
            score = crop['semantic'][3] * crop['origin'][2] * crop['reliability'][0]
            oldscore = old['semantic'][3] * old['origin'][2] * old['reliability'][0]
            better = score > oldscore
            for key in merged: old[key][:, better] = crop[key][:, better]
        return merged


def sensor_sigma(depth, bundle):
    z = np.asarray(depth)
    sensor = bundle['sensor_family']
    if sensor == 'tof': return .0015 + .0007 * z
    model = StereoNoiseModel.simulation_proxy(sensor)
    fx = height_calibration(bundle).intrinsics[0, 0]
    return np.sqrt((model.depth_noise_floor_m + model.depth_noise_quadratic*z*z)**2 +
                   (model.disparity_sigma_px*z*z/(fx*model.baseline_m))**2)


def region_candidates(base, dry, rgb, raw, prediction, liquid_height):
    cal = height_calibration(base.bundle)
    valid = base.ordinary_container_roi & np.isfinite(raw) & (raw > 0)
    heights = world_points(np.where(valid, raw, 0), cal)[..., 2] - cal.bottom_z_m
    sigma = sensor_sigma(np.where(valid, raw, 0), base.bundle)
    sem, origin, reliable = prediction['semantic'], prediction['origin'], prediction['reliability'][0]
    mask = valid & (sem[3] > .20) & (origin[2] > .08) & (reliable > .15)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    distance = cv2.distanceTransform(base.ordinary_container_roi.astype(np.uint8), cv2.DIST_L2, 3)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(float)/255
    candidates = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < 2: continue
        component = labels == label
        ring = cv2.dilate(component.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~component & valid
        if ring.sum() < 8: continue
        z = heights[component]; around = heights[ring]
        noise = max(.001, float(np.median(sigma[component])))
        center = float(np.median(around)); mad = float(np.median(abs(around-center)))
        quantiles = np.percentile(z, [10, 50, 75, 90, 95, 100])
        peak = float(quantiles[3])
        prior = liquid_height if liquid_height is not None else center
        features = [np.log1p(area), w/max(h, 1), area/(w*h),
            float(distance[component].min())/max(raw.shape), noise,
            peak, center, peak-center, (peak-center)/noise, mad/noise,
            (peak-prior)/noise, prior, *(((quantiles-peak)/noise).tolist()),
            float(sem[3, component].mean()), float(sem[3, component].max()),
            float(origin[2, component].mean()), float(origin[5, component].mean()),
            float(origin[6, component].mean()), float(reliable[component].mean()),
            float(sem[3, ring].mean()), float(origin[2, ring].mean()),
            float(gray[component].mean()-gray[ring].mean()),
            float((gray[component] > .97).mean()), float((gray[component] < .03).mean()),
            float((abs(z-peak) <= 2*noise).mean()),
            float(np.median(np.asarray(dry)[component]-raw[component])),
            *[float(base.bundle['sensor_family'] == key) for key in ['tof', 'active_stereo', 'structured_light']]]
        if not np.isfinite(features).all(): continue
        candidates.append({'mask': component, 'features': np.asarray(features, np.float32),
            'height_raw_m': peak, 'noise_m': noise, 'pixels': int(area),
            'bbox_xyxy_px': [int(x), int(y), int(x+w), int(y+h)]})
    return candidates


def score_regions(candidates, verifier):
    if not candidates: return []
    x = np.stack([c['features'] for c in candidates])
    probability = verifier['classifier'].predict_proba(x)[:, 1]
    correction = verifier['regressor'].predict(x)
    low, high = [verifier[key].predict(x) for key in ['lower', 'upper']]
    result = []
    for c, p, delta, lo, hi in zip(candidates, probability, correction, low, high, strict=True):
        result.append({**c, 'object_score': float(p),
            'height_m': float(c['height_raw_m'] + delta*c['noise_m']),
            'interval_width_m': float((max(lo, hi, delta)-min(lo, hi, delta))*c['noise_m'])})
    return result


def choose_peak(scored, policy):
    accepted = [c for c in scored if c['object_score'] >= policy['threshold']
        and c['interval_width_m'] <= policy['interval_factor'] * max(.05, .05*abs(c['height_m']))]
    return (max(accepted, key=lambda c: c['height_m']) if accepted else None), accepted


class OptionalFloatMeterV3:
    def __init__(self, base, dry, predictor=None, verifier_path=None):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.verifier = None if verifier_path is None else joblib.load(verifier_path)
        self.frozen = FrozenSymmetricGeometryMeter(base)
        self.geometric = UnevenSurfaceMeter(base, self.dry)

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='off'):
        if visual_mode not in {'off', 'auto'}: raise ValueError('visual_mode_off_or_auto_required')
        old = self.frozen.measure(depth_m=depth_m)
        peak = self.geometric.measure(depth_m)['highest_visible_height_m']
        diagnostic = {'highest_source': 'legacy_geometric_peak', 'verified_regions': [],
                      'depth_samples_are_original_pixels': True}
        if visual_mode == 'auto':
            if self.predictor is None or self.verifier is None: raise ValueError('optional_model_and_verifier_required')
            features = observed_features(rgb_bgr, depth_m, self.dry, self.base.ordinary_container_roi, self.base.bundle)
            prediction = self.predictor.predict(features)
            candidates = region_candidates(self.base, self.dry, rgb_bgr, np.asarray(depth_m), prediction, old['liquid_depth_m'])
            scored = score_regions(candidates, self.verifier)
            chosen, accepted = choose_peak(scored, self.verifier['policy'])
            diagnostic['candidate_count'] = len(scored)
            diagnostic['verified_regions'] = [{k: v for k, v in c.items() if k not in {'mask', 'features'}} for c in accepted]
            if chosen is not None:
                peak = chosen['height_m']; diagnostic['highest_source'] = 'verified_original_pixel_float'
        highest = peak
        if diagnostic['verified_regions']:
            values = [v for v in [old['liquid_depth_m'], peak] if v is not None]
            highest = max(values) if values else None
        return {**old, 'highest_height_m': highest,
            'route': 'optional_float_evidence_v3', 'visual_module_executed': visual_mode == 'auto',
            'visual_guidance_used': bool(diagnostic['verified_regions']), 'liquid_override_used': False,
            'diagnostics': {**old['diagnostics'], **diagnostic}}
