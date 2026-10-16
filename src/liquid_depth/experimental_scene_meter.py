"""New development dispatcher; old supported routes remain a frozen alternative."""

from .symmetric_visual_meter import SymmetricVisualMeter


class ExperimentalSceneMeter:
    def __init__(self, geometry_meter=None, *, region_predictor=None, refraction_meter=None):
        self.geometry = (
            None
            if geometry_meter is None
            else SymmetricVisualMeter(geometry_meter, predictor=region_predictor)
        )
        self.refraction = refraction_meter

    def measure(self, scene, *, rgb_bgr=None, depth_m=None, stereo_images=None):
        if scene in {"homogeneous_transparent", "homogeneous_translucent"}:
            if self.refraction is None:
                raise ValueError("five_prior_refraction_calibration_required")
            value = self.refraction.measure(scene, stereo_images=stereo_images)
            return {
                **value,
                "development_dispatcher": "visual_evidence_v1",
                "algorithm_unchanged_from_archive": True,
            }
        if scene not in {
            "ordinary_diffuse",
            "partial_depth_loss",
            "glare",
            "low_light",
            "floating_objects",
            "uneven_surface",
        }:
            raise ValueError("explicit_supported_scene_required")
        if self.geometry is None:
            raise ValueError("empty_geometry_required")
        value = self.geometry.measure(depth_m=depth_m, rgb_bgr=rgb_bgr)
        return {**value, "scene": scene, "development_dispatcher": "visual_evidence_v1"}
