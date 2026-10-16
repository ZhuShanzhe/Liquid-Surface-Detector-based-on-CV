import numpy as np
import pytest

from liquid_depth.adaptive_rgbd_meter import AdaptiveRGBDMeter
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.symmetric_visual_meter import SymmetricVisualMeter


def base_meter():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=6.
    bundle={'source':'test_empty_geometry','sensor_family':'tof','image_size':[80,60],'rim_z_m':2.,
        'calibration':{'intrinsics':[[100.,0.,40.],[0.,100.,30.],[0.,0.,1.]],'camera_to_world_cv':pose.tolist(),
                       'bottom_z_m':0.,'radii_m':[1.,1.],'footprint_exponent':2.}}
    return FormalSceneMeter.from_empty_geometry(bundle,np.full((60,80),6.,np.float32))


class Prediction:
    def __init__(self,floating_origin=False):self.calls=0;self.floating_origin=floating_origin
    def predict(self,features):
        self.calls+=1;h,w=features.shape[-2:]
        sem=np.zeros((4,h,w),np.float32);sem[2]=1
        ori=np.zeros((7,h,w),np.float32);ori[1]=1
        if self.floating_origin is not None:
            sem[:,25:35,35:45]=0;sem[3,25:35,35:45]=1
            if self.floating_origin:ori[:,25:35,35:45]=0;ori[2,25:35,35:45]=1
        return {'semantic':sem,'origin':ori,'reliability':np.ones((1,h,w),np.float32)}


def test_frozen_baseline_is_exactly_rgb_free_symmetric_version():
    base=base_meter();raw=np.full((60,80),4.6,np.float32)
    a=FrozenSymmetricGeometryMeter(base).measure(depth_m=raw)
    b=SymmetricVisualMeter(base).measure(depth_m=raw)
    assert a['liquid_depth_m']==b['liquid_depth_m']
    assert not a['visual_guidance_used']


def test_off_never_executes_visual_predictor():
    base=base_meter();p=Prediction();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=p)
    out=meter.measure(rgb_bgr=None,depth_m=np.full((60,80),4.6),mode='off')
    assert p.calls==0
    assert not out['visual_module_executed']


def test_semantic_float_without_matching_return_evidence_cannot_raise_peak():
    base=base_meter();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=Prediction(False))
    raw=np.full((60,80),4.6,np.float32);raw[25:35,35:45]=4.35
    out=meter.measure(rgb_bgr=np.full((60,80,3),128,np.uint8),depth_m=raw,mode='auto')
    assert not out['diagnostics']['float_components']
    assert not out['liquid_override_used']


def test_coupled_float_evidence_changes_peak_not_liquid_without_gate():
    base=base_meter();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=Prediction(True))
    raw=np.full((60,80),4.6,np.float32);raw[25:35,35:45]=4.35;raw[30,40]=3.
    out=meter.measure(rgb_bgr=np.full((60,80,3),128,np.uint8),depth_m=raw,mode='auto')
    assert abs(out['highest_height_m']-1.65)<.02
    assert not out['liquid_override_used']
    assert out['visual_guidance_used']


def test_empty_depth_has_no_fabricated_numeric_output():
    base=base_meter();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=Prediction())
    out=meter.measure(rgb_bgr=np.zeros((60,80,3),np.uint8),depth_m=np.zeros((60,80)),mode='auto')
    assert out['liquid_depth_m'] is None
    assert out['highest_height_m'] is None


def test_features_use_only_observed_finite_arrays():
    base=base_meter();depth=np.full((60,80),5.);depth[0,0]=np.nan;dry=np.full_like(depth,6.);dry[1,1]=np.nan
    x=observed_features(np.zeros((60,80,3),np.uint8),depth,dry,base.ordinary_container_roi,base.bundle)
    assert x.shape==(12,60,80)
    assert np.isfinite(x).all()


def test_unknown_mode_is_rejected():
    base=base_meter();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=Prediction())
    with pytest.raises(ValueError,match='visual_mode'):
        meter.measure(rgb_bgr=None,depth_m=np.full((60,80),5.),mode='truth_best')


def test_return_source_does_not_bypass_metric_reliability():
    class BadMetric(Prediction):
        def predict(self,features):
            p=super().predict(features);p['reliability'][:]=0;return p
    base=base_meter();meter=AdaptiveRGBDMeter(base,dry_depth_m=np.full((60,80),6.),predictor=BadMetric(True))
    raw=np.full((60,80),4.6,np.float32);raw[25:35,35:45]=4.35
    out=meter.measure(rgb_bgr=np.full((60,80,3),128,np.uint8),depth_m=raw)
    assert not out['diagnostics']['float_components']


def test_transparent_route_bypasses_visual_auxiliary():
    from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
    class Refraction:
        def measure(self,scene,stereo_images):return {'liquid_depth_m':.4,'scene':scene}
    out=HybridSceneMeterV2(refraction_meter=Refraction()).measure('homogeneous_transparent',stereo_images={'left':'a','right':'b'})
    assert out['liquid_depth_m']==.4
    assert not out['visual_module_executed']


def test_determined_baseline_source_hashes_remain_unchanged():
    import hashlib
    import json
    from pathlib import Path
    path=Path('/root/autodl-tmp/liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json')
    if not path.exists():pytest.skip('server archive is not attached')
    repo=Path(__file__).resolve().parents[1]
    for name,digest in json.loads(path.read_text())['source_sha256'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest


def test_default_hybrid_is_visual_off_and_keeps_geometric_peak():
    from liquid_depth.hybrid_scene_meter_v2 import HybridSceneMeterV2
    from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter
    base=base_meter();dry=np.full((60,80),6.);raw=np.full((60,80),4.6)
    raw[25:35,35:45]=4.35
    predictor=Prediction(True)
    out=HybridSceneMeterV2(base=base,dry_depth_m=dry,predictor=predictor).measure('floating_objects',depth_m=raw)
    expected=UnevenSurfaceMeter(base,dry).measure(raw)['highest_visible_height_m']
    assert predictor.calls==0
    assert out['highest_height_m']==expected
    assert not out['visual_guidance_used']
