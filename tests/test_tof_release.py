import json
from pathlib import Path
import numpy as np
import pytest
from liquid_depth.tof_phase_refraction import decode,bottom_map,relative_phasor,PhasePriorMeter
from liquid_depth.tof_scene_meter import ToFSceneMeter,SCENES,OPAQUE_SCENES,REFRACTION_SCENES
from measure_tof import initialize

def test_four_phase_decode():
    phase=np.ones((3,8,12))*.37
    images=np.stack([100+20*np.cos(phase+s) for s in np.arange(4)*np.pi/2],axis=1)
    assert np.allclose(np.angle(decode(images)),phase)

def test_snell_no_medium_or_zero_depth():
    q=np.array([[.03,.02],[-.04,.01]])
    assert np.allclose(bottom_map(q,0,1.33,8),8*q)
    assert np.allclose(bottom_map(q,2,1.,8),8*q)

def test_tof_optical_path_positive():
    z=relative_phasor('tof',np.array([[0.,0.]]),.5,[1.333,0,0,0,0],{'frequencies_hz':[20e6,30e6,50e6]})
    assert np.all(np.angle(z[:,0])>0)

@pytest.mark.parametrize('sensor',['structured_light','active_stereo','stereo_IR_image_proxy','unknown'])
def test_retired_families_rejected_at_every_entry(sensor,tmp_path):
    with pytest.raises(ValueError,match='only_tof'):ToFSceneMeter(None,sensor=sensor,kind='empty_geometry')
    with pytest.raises(ValueError,match='only_tof'):ToFSceneMeter.from_empty_geometry({'sensor_family':sensor},np.ones((8,8)))
    with pytest.raises(ValueError,match='only_tof'):ToFSceneMeter.from_phase_priors(sensor=sensor,scene='homogeneous_transparent',profile={},ray_coordinates=None,roi=None,prior_correlations=[],known_heights_m=[])
    with pytest.raises(ValueError,match='only_tof'):PhasePriorMeter.fit(sensor,'homogeneous_transparent',{},None,None,[],[])
    with pytest.raises(ValueError,match='only_tof'):relative_phasor(sensor,None,0,None,None)
    init=tmp_path/'init.json';init.write_text(json.dumps({'sensor_family':sensor}))
    with pytest.raises(ValueError,match='only_tof'):initialize(init,'ordinary_diffuse')

@pytest.mark.parametrize('scene',sorted(OPAQUE_SCENES))
def test_six_opaque_scenes_unchanged_dispatch(scene):
    class Backend:
        def measure(self,received,**kwargs):
            assert received==scene
            return {'liquid_depth_m':.123,'liquid_visual_guidance_used':False}
    value=ToFSceneMeter(Backend(),sensor='tof',kind='empty_geometry').measure(scene,depth_m=np.ones((8,8)))
    assert value['liquid_depth_m']==.123 and value['sensor_family']=='tof' and not value['liquid_visual_guidance_used']

@pytest.mark.parametrize('scene',sorted(REFRACTION_SCENES))
def test_refraction_requires_five_raw_priors(scene):
    with pytest.raises(ValueError,match='five_phase_priors'):ToFSceneMeter(None,sensor='tof',kind='empty_geometry').measure(scene,depth_m=np.ones((8,8)))

def test_prior_budget_and_depth_only_validation():
    with pytest.raises(ValueError,match='one_empty_plus_four'):PhasePriorMeter.fit('tof','homogeneous_transparent',{'container_max_height_m':1,'bottom_m':2},None,None,[],[0,.2,.5])
    with pytest.raises(ValueError,match='4phase'):decode(np.ones((48,64)))

def test_default_registry_is_tof_only():
    root=Path(__file__).resolve().parents[1]
    route=json.loads((root/'configs/application_route.json').read_text())
    assert route['allowed_sensor_families']==['tof'] and route['selected_cli']=='scripts/measure_tof.py'
    manifest=json.loads((root/route['scene_manifest']).read_text())
    assert manifest['sensors']==['tof'] and set(manifest['scenes'])==SCENES

def test_simulator_rejects_retired_family():
    from tof_phase_simulator import generate
    with pytest.raises(ValueError,match='only_tof'):generate('structured_light',{}, {},0,1)
