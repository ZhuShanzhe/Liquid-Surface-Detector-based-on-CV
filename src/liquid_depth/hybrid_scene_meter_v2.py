"""Explicit application boundary and user-controlled auxiliary RGB-D evidence."""
from .adaptive_rgbd_meter import AdaptiveRGBDMeter


class HybridSceneMeterV2:
    def __init__(self,*,base=None,dry_depth_m=None,predictor=None,gate_checkpoint=None,refraction_meter=None):
        self.refraction=refraction_meter
        self.adaptive=None if base is None else AdaptiveRGBDMeter(base,dry_depth_m=dry_depth_m,predictor=predictor,gate_checkpoint=gate_checkpoint)

    def measure(self,scene,*,rgb_bgr=None,depth_m=None,stereo_images=None,visual_mode='off'):
        if scene in {'homogeneous_transparent','homogeneous_translucent'}:
            if self.refraction is None:raise ValueError('five_prior_refraction_calibration_required')
            result=self.refraction.measure(scene,stereo_images=stereo_images)
            return {**result,'visual_guidance_used':False,'visual_module_executed':False,'auxiliary_bypassed':'refraction_specific_route'}
        if scene not in {'ordinary_diffuse','partial_depth_loss','low_light','glare','floating_objects','uneven_surface'}:
            raise ValueError('unsupported_explicit_scene')
        if self.adaptive is None:raise ValueError('empty_geometry_and_dry_reference_required')
        if visual_mode!='off' and self.adaptive.predictor is None:raise ValueError('visual_predictor_required_unless_mode_off')
        result=self.adaptive.measure(rgb_bgr=rgb_bgr,depth_m=depth_m,mode=visual_mode)
        if visual_mode=='off':
            # OFF preserves the established geometric highest-envelope function;
            # the frozen symmetric version defines the liquid estimator, not a new peak detector.
            peak=self.adaptive.geometric_peak.measure(depth_m)['highest_visible_height_m']
            result={**result,'highest_height_m':peak,'diagnostics':{**result['diagnostics'],'highest_source':'legacy_geometric_peak'}}
        return {**result,'scene':scene}
