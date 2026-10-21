"""Optional highest-only visual evidence; never writes the liquid estimate.

The lightweight geometric/scene route still owns liquid measurement. A separate
presence probe is necessary because the scene classifier can miss tiny floats.
Inference receives no synthetic semantic, origin, instance or height labels.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .rgbd_evidence_model import RGBDEvidencePredictor
from .small_float_peak_core import float_features, score_candidates, choose_highest
from .native_float_components import dense_candidates

VERSION = 'small_float_highest_v1_20261020'


class SmallFloatPeakRefiner:
    def __init__(self, bundle, dry_depth_m, *, asset_dir=None, device='cpu'):
        import joblib
        import sklearn
        asset_dir = Path(asset_dir) if asset_dir else Path(__file__).resolve().parents[2] / 'assets/models/small_float_peak'
        manifest = json.loads((asset_dir / 'manifest.json').read_text())
        if sklearn.__version__ != manifest['sklearn_version']:
            raise ValueError('small_float_requires_scikit_learn_' + manifest['sklearn_version'])
        for name, expected in manifest['sha256'].items():
            if Path(name).name != name:
                raise ValueError('invalid_small_float_asset_path')
            if hashlib.sha256((asset_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError('small_float_asset_checksum_mismatch')
        self.policy = json.loads((asset_dir / 'policy.json').read_text())
        self.predictor = RGBDEvidencePredictor(asset_dir / 'pixel_best.pt', device=device)
        # Only load the trusted, hash-verified project asset, never user uploads.
        self.models = joblib.load(asset_dir / 'candidates.joblib')
        for model in self.models.values():
            model.n_jobs = 1
        self.bundle, self.dry = bundle, np.asarray(dry_depth_m)
        self.warmed = False

    def warm(self):
        if self.warmed:
            return 0.
        start = time.perf_counter()
        rgb = np.zeros((*self.dry.shape, 3), np.uint8)
        features, _ = float_features(rgb, self.dry, self.dry, self.bundle)
        self.predictor.predict(features)
        self.warmed = True
        return (time.perf_counter() - start) * 1000

    def refine(self, result, depth_m, rgb_bgr):
        start = time.perf_counter()
        if rgb_bgr is None:
            return result
        # Read the frozen liquid estimate as a feature only; never recompute it.
        features, _ = float_features(rgb_bgr, depth_m, self.dry, self.bundle)
        prediction = self.predictor.predict(features)
        candidates = dense_candidates(depth_m, prediction, self.bundle, result.get('liquid_depth_m'))
        chosen = choose_highest(score_candidates(candidates, self.models), self.policy)
        previous = result.get('highest_height_m')
        use = chosen is not None and (previous is None or chosen['height_m'] > previous)
        output = dict(result)
        if use:
            output['highest_height_m'] = chosen['height_m']
        selected = None if chosen is None else {key: chosen[key] for key in (
            'height_m', 'height_raw_m', 'pixels', 'bbox', 'noise_m',
            'object_probability', 'top_probability', 'correction_m')}
        output['diagnostics'] = {**result.get('diagnostics', {}), 'small_float_peak': dict(
            version=VERSION, candidate_count=len(candidates), accepted=chosen is not None,
            used=bool(use), selected=selected, previous_highest_m=previous,
            liquid_modified=False, visual_highest_only=True,
            auxiliary_ms=(time.perf_counter() - start) * 1000)}
        return output
