import numpy as np
import pytest
from liquid_depth.phase_refraction_v2 import decode,projector_delta,relative_phasor,PhasePriorMeter
from liquid_depth.dual_camera_scene_meter_v2 import DualCameraSceneMeter


def test_four_phase_decode():
    phase=np.ones((2,8,12))*.37
    images=np.stack([100+20*np.cos(phase+s) for s in np.arange(4)*np.pi/2],axis=1)
    assert np.allclose(np.angle(decode(images)),phase)


def test_snell_zero_and_index_one_no_contraction():
    q=np.array([[.03,.02],[-.04,.01]])
    assert np.max(abs(projector_delta(q,0,1.33,8,.035,470)))<1e-8
    assert np.max(abs(projector_delta(q,2,1.,8,.035,470)))<1e-8


def test_apparent_contraction_is_not_tof_optical_path():
    q=np.array([[0.,0.]])
    delta=projector_delta(q,.5,1.333,8,.035,470)
    assert delta[0]<0
    p={'frequencies_hz':[20e6,30e6,50e6]}
    tof=relative_phasor('tof',q,.5,[1.333,0,0,0,0],p)
    assert np.all(np.angle(tof[:,0])>0)


@pytest.mark.parametrize('sensor',['active_stereo','stereo_IR_image_proxy','unknown'])
def test_formal_dual_route_rejects_other_camera_families(sensor):
    with pytest.raises(ValueError,match='only_tof_or_structured_light'):
        DualCameraSceneMeter.from_empty_geometry({'sensor_family':sensor},np.ones((8,8)))


def test_prior_budget_and_shape_validation():
    with pytest.raises(ValueError,match='one_empty_plus_four'):
        PhasePriorMeter.fit('tof','homogeneous_transparent',{'container_max_height_m':1,'bottom_m':2},None,None,[],[0,.2,.5])


def test_depth_only_is_not_raw_phase_observation():
    with pytest.raises(ValueError,match='4phase'):
        decode(np.ones((48,64)))


@pytest.mark.parametrize('sensor',['tof','structured_light'])
@pytest.mark.parametrize('scene',['ordinary_diffuse','partial_depth_loss','glare','low_light','floating_objects','uneven_surface'])
def test_all_six_opaque_scenes_dispatch_without_liquid_vision(sensor,scene):
    class Backend:
        def measure(self,received,**kwargs):
            assert received==scene
            return {'liquid_depth_m':.123,'liquid_visual_guidance_used':False}
    meter=DualCameraSceneMeter(Backend(),sensor=sensor,kind='empty_geometry')
    value=meter.measure(scene,depth_m=np.ones((8,8)))
    assert value['sensor_family']==sensor and value['liquid_depth_m']==.123
    assert value['liquid_visual_guidance_used'] is False


@pytest.mark.parametrize('sensor',['tof','structured_light'])
@pytest.mark.parametrize('scene',['homogeneous_transparent','homogeneous_translucent'])
def test_refraction_cannot_use_empty_geometry_instead_of_raw_priors(sensor,scene):
    meter=DualCameraSceneMeter(None,sensor=sensor,kind='empty_geometry')
    with pytest.raises(ValueError,match='five_phase_priors'):
        meter.measure(scene,depth_m=np.ones((8,8)))


def test_sl_scatter_reference_is_between_bottom_and_surface_phase():
    q=np.array([[.02,.01]]);h=.8;n=1.333;b=2.;fx=470.;baseline=.035;ext=.08;scatter=.2
    profile={'bottom_m':b,'fx_px':fx,'baseline_m':baseline,'fringe_periods_px':[8.,13.]}
    k=2*np.pi/np.array([8.,13.]);delta=projector_delta(q,h,n,b,baseline,fx)[0]
    surface=-fx*baseline/(b-h)+fx*baseline/b
    got=relative_phasor('structured_light',q,h,[n,0.,ext,scatter,0.],profile)[:,0]
    expected=np.exp(-2*ext*h)*np.exp(1j*k*delta)+scatter*(1-np.exp(-2*ext*h))*np.exp(1j*k*(delta+surface)/2)
    assert np.allclose(got,expected)
