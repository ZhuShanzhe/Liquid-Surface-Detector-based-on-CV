"""Optional observed-surface peak backend, under the unchanged V4 scene/output gate."""
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
        from .floating_surface_v6 import FloatingSurfaceMeterV6
        profile = json.loads(Path(profile_path).read_text())
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'], 'cuda' if torch.cuda.is_available() else 'cpu')
        return FloatingSurfaceMeterV6(base, dry_depth_m, predictor, profile['surface_checkpoint'],
            profile['previous_verifier_checkpoint'], profile['apex_checkpoint'])

    return FloatingPeakOnlyMeter(legacy, factory)
