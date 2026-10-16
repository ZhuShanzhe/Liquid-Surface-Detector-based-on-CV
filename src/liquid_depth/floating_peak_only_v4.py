"""Application gate: optional vision is confined to opaque floating-object peak height."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

APPLICATION_SCENES = frozenset({'ordinary_diffuse', 'partial_depth_loss', 'low_light', 'glare',
    'floating_objects', 'uneven_surface', 'homogeneous_transparent', 'homogeneous_translucent'})


class FloatingPeakOnlyMeter:
    """The legacy liquid output is authoritative, regardless of auxiliary output.

    ``floating_objects`` means the calibrated opaque/diffuse-liquid floating-object
    route, not arbitrary transparent, strongly dark, glare or mixed-scene overrides.
    Selection is explicit; no ground-truth label or autonomous scene classifier is used.
    """

    def __init__(self, legacy_meter: Any, float_meter_factory: Callable[[], Any] | None = None):
        self.legacy = legacy_meter
        self._factory = float_meter_factory
        self._float_meter = None

    @classmethod
    def from_empty_geometry(cls, bundle, dry_depth_m, *, profile_path=None, **base_kwargs):
        from .hybrid_scene_meter_v2 import HybridSceneMeterV2
        from .supported_scene_meter import FormalSceneMeter

        base = FormalSceneMeter.from_empty_geometry(bundle, dry_depth_m, **base_kwargs)
        legacy = HybridSceneMeterV2(base=base, dry_depth_m=dry_depth_m)

        def load_float_meter():
            import json

            import torch

            from .float_evidence_v3 import MultiScaleEvidencePredictor, OptionalFloatMeterV3

            if profile_path is None:
                raise ValueError('floating_peak_profile_required_when_explicitly_enabled')
            profile = json.loads(Path(profile_path).read_text())
            predictor = MultiScaleEvidencePredictor(profile['pixel_checkpoint'],
                'cuda' if torch.cuda.is_available() else 'cpu')
            return OptionalFloatMeterV3(base, dry_depth_m, predictor, profile['verifier_checkpoint'])

        return cls(legacy, load_float_meter)

    def measure(self, scene, *, depth_m=None, rgb_bgr=None, stereo_images=None, visual_enabled=False):
        if scene not in APPLICATION_SCENES:
            raise ValueError('explicit_supported_application_scene_required')
        # Never pass RGB or a visual mode through to legacy application routes.
        result = dict(self.legacy.measure(scene, depth_m=depth_m, stereo_images=stereo_images, visual_mode='off'))
        allowed = scene == 'floating_objects'
        execute = bool(visual_enabled and allowed)
        policy = {'requested': bool(visual_enabled), 'allowed': allowed,
            'allowed_scene': 'floating_objects', 'allowed_output': 'highest_height_m',
            'liquid_visual_guidance_used': False,
            'disabled_reason': None if execute else 'scene_not_floating_objects' if not allowed else 'operator_switch_off'}
        auxiliary = None
        peak_adopted = False
        if execute:
            if rgb_bgr is None:
                raise ValueError('aligned_RGB_required_for_floating_peak_only')
            if self._float_meter is None:
                if self._factory is None:
                    raise ValueError('floating_peak_model_factory_required')
                self._float_meter = self._factory()
            auxiliary = self._float_meter.measure(depth_m=depth_m, rgb_bgr=rgb_bgr, visual_mode='auto')
            highest = auxiliary.get('highest_height_m')
            if highest is not None and np.isfinite(highest):
                result['highest_height_m'] = float(highest)
                peak_adopted = True
        used = bool(peak_adopted and auxiliary is not None and auxiliary.get('visual_guidance_used', False))
        result.update({'scene': scene, 'application_route': 'floating_peak_only_v4',
            'visual_module_executed': execute, 'visual_guidance_used': used,
            'highest_visual_guidance_used': used, 'liquid_visual_guidance_used': False,
            'liquid_override_used': False, 'visual_application_policy': policy})
        diagnostic = dict(result.get('diagnostics', {}))
        diagnostic['floating_peak_policy'] = policy
        if auxiliary is not None:
            diagnostic['floating_peak_auxiliary'] = auxiliary['diagnostics']
        result['diagnostics'] = diagnostic
        return result
