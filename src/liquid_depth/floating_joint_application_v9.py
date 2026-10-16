"""Explicit causal-frame context around the unchanged highest-only scene gate."""
import json
from pathlib import Path

from .floating_peak_only_v4 import FloatingPeakOnlyMeter
from .hybrid_scene_meter_v2 import HybridSceneMeterV2
from .supported_scene_meter import FormalSceneMeter


class ContextBackend:
    def __init__(self, backend, context):
        self.backend, self.context = backend, context

    def measure(self, **kwargs):
        frame = self.context.get('frame')
        if frame is not None:
            self.backend.set_frame_context(*frame)
        return self.backend.measure(**kwargs)


class FloatingJointApplicationV9:
    def __init__(self, legacy, factory):
        self.context = {}
        self.route = FloatingPeakOnlyMeter(legacy, lambda: ContextBackend(factory(), self.context))

    def measure(self, scene, *, depth_m=None, rgb_bgr=None, stereo_images=None, visual_enabled=False,
                stream_id=None, frame_id=None, timestamp_s=None):
        self.context['frame'] = (stream_id, frame_id, timestamp_s) if all(v is not None for v in [stream_id, frame_id, timestamp_s]) else None
        if (scene != 'floating_objects' or not visual_enabled) and self.route._float_meter is not None:
            self.route._float_meter.backend.memory.reset()
        return self.route.measure(scene, depth_m=depth_m, rgb_bgr=rgb_bgr, stereo_images=stereo_images, visual_enabled=visual_enabled)


def from_empty_geometry(bundle, dry_depth_m, *, profile_path, temporal=True, **base_kwargs):
    base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **base_kwargs)
    legacy = HybridSceneMeterV2(base=base, dry_depth_m=dry_depth_m)

    def factory():
        import torch

        from .float_evidence_v3 import MultiScaleEvidencePredictor
        from .floating_joint_v9 import FloatingJointMeterV9
        profile = json.loads(Path(profile_path).read_text())
        predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'], 'cuda' if torch.cuda.is_available() else 'cpu')
        return FloatingJointMeterV9(base, dry_depth_m, predictor, profile['model_checkpoint'], profile['v8_checkpoint'],
            profile['previous_verifier_checkpoint'], profile['apex_checkpoint'], temporal=temporal)
    return FloatingJointApplicationV9(legacy, factory)
