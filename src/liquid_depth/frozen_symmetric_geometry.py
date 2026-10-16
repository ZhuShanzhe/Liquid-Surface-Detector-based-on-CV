"""Determined geometric baseline. Keep original symmetric_visual_meter.py unchanged."""
from .symmetric_visual_meter import SymmetricVisualMeter


class FrozenSymmetricGeometryMeter:
    def __init__(self,calibrated_meter):
        self._meter=SymmetricVisualMeter(calibrated_meter,predictor=None)

    def measure(self,*,depth_m):
        result=self._meter.measure(depth_m=depth_m)
        return {**result,'route':'determined_symmetric_geometry_v1',
                'visual_guidance_used':False,
                'baseline_manifest':'/root/autodl-tmp/liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json'}
