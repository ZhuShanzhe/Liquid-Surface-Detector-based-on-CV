"""Hard-supported and joint-risk-constrained highest-only experimental backend."""
from __future__ import annotations

import hashlib
import time
from collections import deque

import joblib
import numpy as np

from .empty_geometry_v41 import height_calibration
from .float_evidence_v3 import OptionalFloatMeterV3, sensor_sigma
from .floating_apex_v5 import observed_peak_roi
from .floating_rank_v8 import candidate_features, proposals, score_candidates, select_ranked
from .floating_surface_v7 import fit_patch, model_allowed
from .frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from .rgbd_evidence_model import observed_features


def joint_features(x, scored):
    scale = np.asarray([max(.01, c['noise_m']) for c in scored])
    return np.column_stack([x, [c['learned_correction_m']/s for c, s in zip(scored, scale, strict=True)],
        [c['height_m'] for c in scored], [c['risk_m']/s for c, s in zip(scored, scale, strict=True)],
        [c['top_gap_m']/s for c, s in zip(scored, scale, strict=True)]]).astype(np.float32)


def score_joint(candidates, x, v8, bundle):
    scored = score_candidates(candidates, x, v8)
    if not scored:
        return []
    features = joint_features(x, scored)
    probability = bundle['identity'].predict_proba(features)[:, 1]
    risk = np.maximum(0., bundle['joint_risk'].predict(features))
    for c, p, r in zip(scored, probability, risk, strict=True):
        c['conditional_risk_m'] = c['risk_m']
        c['v8_physical_score'] = c['physical_score']
        c['physical_score'] = float(p)
        c['risk_m'] = float(r*max(.01, c['noise_m']))
    return scored


def support_check(candidate, raw, cal, bundle, *, fit_cache=None):
    yy, xx = np.unravel_index(candidate['indices'], raw.shape)
    span = np.ptp(np.column_stack([xx, yy]), axis=0)
    if len(xx) < 12 or span.min() < 2 or candidate['cells'] < 3:
        return {'valid': False, 'reason': 'hard_spatial_support'}
    pixels = np.column_stack([xx, yy]).astype(float)
    key = tuple(candidate['indices'])
    model = fit_cache.get(key) if fit_cache is not None else None
    if model is None:
        model = fit_patch(raw[yy, xx], pixels, np.ones(len(xx)), cal, bundle)
        if fit_cache is not None:
            fit_cache[key] = model
    if not model_allowed(model, max_cv=3., min_cells=3):
        return {**model, 'valid': False, 'reason': 'hard_model_mismatch'}
    k, pose = cal.intrinsics, cal.camera_to_world_cv
    ray = (xx-k[0, 2])/k[0, 0]*pose[2, 0]+(yy-k[1, 2])/k[1, 1]*pose[2, 1]+pose[2, 2]
    heights = ray*raw[yy, xx]+pose[2, 3]-cal.bottom_z_m
    witness = abs(heights-candidate['height_m']) <= 3*sensor_sigma(raw[yy, xx], bundle)
    cells = np.floor(pixels/3).astype(int)
    witnessed_cells = len(np.unique(cells[witness], axis=0))
    return {**model, 'valid': witnessed_cells >= 3, 'own_height_witness_cells': witnessed_cells,
        'reason': 'supported' if witnessed_cells >= 3 else 'corrected_height_unwitnessed'}


def instance_candidates(scored, policy):
    selection = {'min_pixels': 4, 'min_cells': 2, 'physical_threshold': policy['identity_threshold'],
        'max_risk_m': policy['risk_cap_m'], 'risk_weight': policy['risk_weight'], 'gap_weight': 0.}
    _, instances = select_ranked(scored, selection)
    return sorted(instances, key=lambda c: -c['utility'])


class CausalSupport:
    """Confirmation only: no future frames, no held old height without current evidence."""
    def __init__(self):
        self.history = deque(maxlen=3)
        self.stream = None
        self.last_frame = None
        self.last_time = None
        self.last_digest = None

    def reset(self):
        self.history.clear()
        self.stream = self.last_frame = self.last_time = self.last_digest = None

    def begin(self, context, raw):
        if context is None:
            self.reset()
            return False, 'missing_frame_context'
        stream, frame, timestamp = context
        if (not isinstance(stream, str) or not stream or isinstance(frame, bool)
            or not isinstance(frame, (int, np.integer)) or not isinstance(timestamp, (int, float, np.floating))
            or not np.isfinite(timestamp)):
            self.reset()
            return False, 'invalid_frame_context'
        if self.stream != stream:
            self.reset()
            self.stream = stream
        if self.last_frame is not None and (frame <= self.last_frame or timestamp <= self.last_time or timestamp-self.last_time > .5):
            self.reset()
            self.stream = stream
            self.last_frame, self.last_time = frame, timestamp
            return False, 'duplicate_out_of_order_or_gap'
        digest = hashlib.blake2b(np.ascontiguousarray(raw).tobytes(), digest_size=12).digest()
        duplicate = digest == self.last_digest
        self.last_frame, self.last_time, self.last_digest = frame, timestamp, digest
        return not duplicate, 'duplicate_depth_buffer' if duplicate else 'valid_current_frame'

    def confirmation_count(self, candidate):
        count = 1
        for timestamp, frame in self.history:
            if self.last_time-timestamp > .5:
                continue
            for old in frame:
                overlap = len(np.intersect1d(candidate['indices'], old['indices'], assume_unique=True))/max(min(candidate['pixels'], old['pixels']), 1)
                if overlap >= .5 and abs(candidate['height_m']-old['height_m']) <= max(.015, .02*abs(candidate['height_m'])):
                    count += 1
                    break
        return count

    def finish(self, candidates, valid):
        if valid:
            self.history.append((self.last_time, [{k: c[k] for k in ['indices', 'pixels', 'height_m']} for c in candidates]))


def weak_eligible(c, policy):
    box = c['bbox_xyxy_px']
    return bool(4 <= c['pixels'] < 12 and c['cells'] >= 2 and min(box[2]-box[0], box[3]-box[1]) >= 2
        and c['physical_score'] >= max(.995, policy['identity_threshold'])
        and c['risk_m'] <= min(.08, max(.02, .02*abs(c['height_m'])))
        and c['features'][18] >= .5 and c['features'][20] >= .5 and c['features'][23] >= .5)


def choose_supported(instances, policy, support, memory=None, temporal_valid=False):
    accepted = []
    weak = []
    checked = []
    fits = 0
    for c in instances:
        if c['pixels'] < 12:
            if weak_eligible(c, policy):
                weak.append(c)
                confirmations = memory.confirmation_count(c) if temporal_valid and memory is not None else 1
                if temporal_valid and confirmations >= policy['confirm_frames']:
                    accepted.append({**c, 'support_kind': 'causal_small_object', 'confirmations': confirmations})
            continue
        if fits >= 8:
            continue
        fits += 1
        model = support(c)
        checked.append({'height_m': c['height_m'], 'pixels': c['pixels'], 'support': model})
        if model['valid']:
            accepted.append({**c, 'support_kind': 'hard_spatial', 'confirmations': 1})
    if memory is not None:
        memory.finish(weak, temporal_valid)
    chosen = max(accepted, key=lambda c: c['utility']) if accepted else None
    return chosen, accepted, checked


class FloatingJointMeterV9:
    def __init__(self, base, dry, predictor, model_path, v8_path, previous_path, apex_path, *, temporal=True, policy_override=None):
        self.base, self.dry, self.predictor = base, np.asarray(dry), predictor
        self.bundle, self.v8 = joblib.load(model_path), joblib.load(v8_path)
        self.previous, self.apex = joblib.load(previous_path), joblib.load(apex_path)
        self.frozen = FrozenSymmetricGeometryMeter(base)
        self.fallback = OptionalFloatMeterV3(base, dry, predictor, previous_path)
        self.temporal, self.policy_override = temporal, policy_override
        self.memory = CausalSupport()
        self.context = None
        self._last_scored = []
        self._last_chosen = None

    def set_frame_context(self, stream_id, frame_id, timestamp_s):
        self.context = stream_id, frame_id, timestamp_s

    def measure(self, *, depth_m, rgb_bgr=None, visual_mode='auto'):
        if visual_mode not in {'auto', 'off'}:
            raise ValueError('peak_only_visual_mode_required')
        self._last_scored = []
        self._last_chosen = None
        context, self.context = self.context, None
        if visual_mode == 'off':
            self.memory.reset()
            return self.fallback.measure(depth_m=depth_m, visual_mode='off')
        valid, reason = self.memory.begin(context if self.temporal else None, depth_m)
        policy = self.policy_override or self.bundle['policy_by_sensor'][self.base.bundle['sensor_family']]
        if policy.get('bypass'):
            result = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            return {**result, 'diagnostics': {**result['diagnostics'], 'v9_used': False, 'policy_bypass': True,
                'temporal_context': reason, 'v9_candidates': [], 'hard_checks': []}}
        stages = {}
        start = time.perf_counter()
        old = self.frozen.measure(depth_m=depth_m)
        roi = observed_peak_roi(self.base, depth_m)
        pred = self.predictor.predict(observed_features(rgb_bgr, depth_m, self.dry, roi, self.base.bundle))
        _, candidates = proposals(self.base, self.dry, rgb_bgr, depth_m, pred, old['liquid_depth_m'], roi, self.previous, self.apex)
        stages['visual_features_candidates_ms'] = 1000*(time.perf_counter()-start)
        start = time.perf_counter()
        x = candidate_features(candidates, self.base, self.dry, depth_m, pred)
        self._last_scored = score_joint(candidates, x, self.v8, self.bundle)
        instances = instance_candidates(self._last_scored, policy)
        stages['joint_features_score_group_ms'] = 1000*(time.perf_counter()-start)
        start = time.perf_counter()
        cal = height_calibration(self.base.bundle)
        chosen, accepted, checked = choose_supported(instances, policy,
            lambda c: support_check(c, depth_m, cal, self.base.bundle), self.memory, valid)
        self._last_chosen = chosen
        stages['hard_and_temporal_ms'] = 1000*(time.perf_counter()-start)
        start = time.perf_counter()
        if chosen is None:
            result = self.fallback.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            peak = result['highest_height_m']
            used = result['visual_guidance_used']
        else:
            peak = max(v for v in [chosen['height_m'], old['liquid_depth_m']] if v is not None)
            used = True
        stages['fallback_or_output_ms'] = 1000*(time.perf_counter()-start)
        return {**old, 'highest_height_m': peak, 'visual_module_executed': True, 'visual_guidance_used': used, 'liquid_override_used': False,
            'diagnostics': {**old['diagnostics'], 'apex_backend': 'hard_joint_causal_v9', 'v9_used': chosen is not None,
                'policy_bypass': False, 'temporal_context': reason, 'candidate_count': len(candidates),
                'joint_identity_pass': sum(c['physical_score'] >= policy['identity_threshold'] for c in self._last_scored),
                'joint_risk_pass': sum(c['risk_m'] <= policy['risk_cap_m'] for c in self._last_scored),
                'instances_before_hard': len(instances), 'weak_eligible_instances': sum(weak_eligible(c, policy) for c in instances),
                'stage_ms': stages,
                'chosen_support': chosen['support_kind'] if chosen else None, 'hard_checks': checked,
                'chosen_bbox': chosen['bbox_xyxy_px'] if chosen else None,
                'v9_candidates': [{k: v for k, v in c.items() if k not in {'mask', 'indices', 'features'}} for c in accepted]}}
