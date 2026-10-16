"""Lazy experimental highest-only factory; all liquid routes stay authoritative."""
import json
from pathlib import Path

from .floating_peak_only_v4 import FloatingPeakOnlyMeter
from .hybrid_scene_meter_v2 import HybridSceneMeterV2
from .supported_scene_meter import FormalSceneMeter


def from_empty_geometry(bundle, dry_depth_m, *, profile_path, policy_variant='selected', **base_kwargs):
    if policy_variant not in {'selected', 'no_rank', 'research'}:
        raise ValueError('unknown_experiment_policy')
    base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **base_kwargs)
    legacy = HybridSceneMeterV2(base=base, dry_depth_m=dry_depth_m)

    def factory():
        import torch

        from .float_evidence_v3 import MultiScaleEvidencePredictor
        from .floating_rank_v8 import FloatingRankMeterV8
        profile = json.loads(Path(profile_path).read_text())
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'], 'cuda' if torch.cuda.is_available() else 'cpu')
        meter = FloatingRankMeterV8(base, dry_depth_m, predictor, profile['model_checkpoint'], profile['previous_verifier_checkpoint'], profile['apex_checkpoint'])
        sensor = base.bundle['sensor_family']
        if policy_variant == 'research':
            meter.policy_override = meter.bundle['research_policy_by_sensor'][sensor]
        elif policy_variant == 'no_rank':
            meter.policy_override = {**meter.bundle['research_policy_by_sensor'][sensor], 'gap_weight': 0.}
        return meter
    return FloatingPeakOnlyMeter(legacy, factory)
