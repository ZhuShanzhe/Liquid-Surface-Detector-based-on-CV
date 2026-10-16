"""Profile-driven V5 backend inside the unchanged highest-only V4 application gate."""
import json
from pathlib import Path

from .floating_peak_only_v4 import FloatingPeakOnlyMeter
from .hybrid_scene_meter_v2 import HybridSceneMeterV2
from .supported_scene_meter import FormalSceneMeter


def from_empty_geometry(bundle, dry_depth_m, *, profile_path, **base_kwargs):
    base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **base_kwargs)
    legacy = HybridSceneMeterV2(base=base, dry_depth_m=dry_depth_m)

    def factory():
        import torch

        from .float_evidence_v3 import MultiScaleEvidencePredictor
        from .floating_apex_v5 import FloatingApexMeterV5

        profile = json.loads(Path(profile_path).read_text())
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'],
            'cuda' if torch.cuda.is_available() else 'cpu')
        return FloatingApexMeterV5(base, dry_depth_m, predictor,
            profile['apex_verifier_checkpoint'], profile['previous_verifier_checkpoint'])

    return FloatingPeakOnlyMeter(legacy, factory)
