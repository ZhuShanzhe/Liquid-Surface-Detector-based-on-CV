"""Lazy V7 backend inside the unchanged scene and output gate."""
import json
from pathlib import Path

from .floating_peak_only_v4 import FloatingPeakOnlyMeter
from .hybrid_scene_meter_v2 import HybridSceneMeterV2
from .supported_scene_meter import FormalSceneMeter


def from_empty_geometry(bundle, dry_depth_m, *, profile_path, guards_only=False, **base_kwargs):
    base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **base_kwargs)
    legacy = HybridSceneMeterV2(base=base, dry_depth_m=dry_depth_m)

    def factory():
        import torch

        from .float_evidence_v3 import MultiScaleEvidencePredictor
        from .floating_surface_v7 import FloatingSurfaceMeterV7
        profile = json.loads(Path(profile_path).read_text())
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'], 'cuda' if torch.cuda.is_available() else 'cpu')
        meter = FloatingSurfaceMeterV7(base, dry_depth_m, predictor, profile['model_checkpoint'],
            profile['previous_verifier_checkpoint'], profile['apex_checkpoint'])
        if guards_only: meter.bundle = {**meter.bundle, 'policy_by_sensor': meter.bundle['guard_policy_by_sensor']}
        return meter

    return FloatingPeakOnlyMeter(legacy, factory)
