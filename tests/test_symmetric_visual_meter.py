import numpy as np

from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.symmetric_visual_meter import SymmetricVisualMeter, symmetric_candidates


def inputs():
    rng=np.random.default_rng(1510)
    h=np.r_[rng.normal(0,.005,240),rng.normal(1.4,.005,140)]
    xy=rng.uniform(0,100,(len(h),2));noise=np.full(len(h),.006)
    return h,xy,noise


def test_visual_wet_candidate_can_replace_bottom_dominance():
    h,xy,noise=inputs();p=np.zeros((4,len(h)))
    p[1,:240]=.05;p[2,:240]=.95;p[2,240:]=.95;p[1,240:]=.05
    candidates=symmetric_candidates(h,xy,noise,p)
    assert abs(candidates[0]['height_m']-1.4)<.02


def test_without_semantics_all_candidates_compete_by_support():
    h,xy,noise=inputs();c=symmetric_candidates(h,xy,noise)
    assert abs(c[0]['height_m'])<.02
    assert any(abs(x['height_m']-1.4)<.02 for x in c)


def test_height_translation_does_not_add_bottom_preference_without_semantics():
    h,xy,noise=inputs()
    a=symmetric_candidates(h,xy,noise);b=symmetric_candidates(h+2.7,xy,noise)
    assert abs((b[0]['height_m']-a[0]['height_m'])-2.7)<.002


def test_three_points_not_treated_as_sufficient_support():
    c=symmetric_candidates(np.array([1.,1.001,1.002]),np.array([[0,0],[1,0],[0,1]]),np.full(3,.005))
    assert not c or c[0]['support_returns']<24


def test_negative_measurements_are_not_zero_snapped():
    h,xy,noise=inputs();c=symmetric_candidates(h-2.,xy,noise)
    assert c[0]['height_m']<0


def meter():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=6.
    spec={'source':'test_empty_geometry','sensor_family':'tof','image_size':[80,60],'rim_z_m':2.,
          'calibration':{'intrinsics':[[100.,0.,40.],[0.,100.,30.],[0.,0.,1.]],
          'camera_to_world_cv':pose.tolist(),'bottom_z_m':0.,'radii_m':[1.,1.],'footprint_exponent':2.}}
    return FormalSceneMeter.from_empty_geometry(spec,np.full((60,80),6.,np.float32))


def test_full_chain_adopts_supported_wet_candidate_not_just_diagnostic():
    base=meter();raw=np.full((60,80),6.,np.float32)
    yy,xx=np.nonzero(base.ordinary_container_roi);raw[yy[::4],xx[::4]]=4.6
    p=np.zeros((4,60,80));p[0]=.01;p[1]=.03;p[2]=.95;p[3]=.01
    out=SymmetricVisualMeter(base).measure(depth_m=raw,probabilities=p)
    assert abs(out['liquid_depth_m']-1.4)<.01
    assert out['precision_eligible']


def test_no_current_return_has_no_fabricated_number():
    out=SymmetricVisualMeter(meter()).measure(depth_m=np.zeros((60,80)))
    assert out['liquid_depth_m'] is None
    assert not out['precision_eligible']


def test_float_supported_peak_rejects_isolated_high_outlier():
    base=meter();raw=np.full((60,80),4.6,np.float32)
    p=np.zeros((4,60,80));p[0]=.01;p[1]=.01;p[2]=.97;p[3]=.01
    raw[25:35,35:45]=4.35
    p[:,25:35,35:45]=np.array([.01,.01,.01,.97])[:,None,None]
    raw[30,40]=3.
    out=SymmetricVisualMeter(base).measure(depth_m=raw,probabilities=p)
    assert abs(out['liquid_depth_m']-1.4)<.01
    assert abs(out['highest_height_m']-1.65)<.02
    assert out['diagnostics']['raw_float_max_diagnostic_m']>2.
    assert out['diagnostics']['float_components']==1


def test_refraction_dispatcher_preserves_existing_calibrated_route():
    from liquid_depth.experimental_scene_meter import ExperimentalSceneMeter
    class Refraction:
        def measure(self,scene,stereo_images):return {'scene':scene,'liquid_depth_m':.42,'inputs':stereo_images}
    out=ExperimentalSceneMeter(refraction_meter=Refraction()).measure('homogeneous_transparent',stereo_images={'left':'a','right':'b'})
    assert out['liquid_depth_m']==.42
    assert out['algorithm_unchanged_from_archive']
