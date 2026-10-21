import math
import pytest
from liquid_depth.scene_console_web import validate_controls

def test_old_defaults_are_compatible():
    c=validate_controls({})
    assert c['container_height_m']==1.5 and c['camera_to_bottom_m']==4.

@pytest.mark.parametrize('capacity,bottom,height',[(.1,.2,0),(.5,1.2,.3),(1.5,4,.6),(3,6,2.9),(9,10,8.9),(9.8,10,9.)])
def test_top_installation_valid(capacity,bottom,height):
    c=validate_controls(dict(container_height_m=capacity,camera_to_bottom_m=bottom,height_m=height,auto_frame=False))
    assert c['height_m']==height

@pytest.mark.parametrize('controls',[
 {'container_height_m':0}, {'container_height_m':.09}, {'container_height_m':9.9},
 {'camera_to_bottom_m':10.1}, {'container_height_m':4,'camera_to_bottom_m':3},
 {'container_height_m':1.5,'camera_to_bottom_m':1.59},
 {'container_height_m':.5,'height_m':.6}, {'container_height_m':float('nan')},
 {'camera_to_bottom_m':float('inf')}, {'container_height_m':True},
 {'container_height_m':1.,'height_m':.99,'wave_m':.01},
 {'material':'homogeneous_transparent','float_count':1}, {'loss':.51}])
def test_invalid_installation_rejected(controls):
    with pytest.raises(ValueError):validate_controls(dict(auto_frame=False,**controls))

def test_parameters_do_not_clip_requested_height():
    with pytest.raises(ValueError,match='liquid_exceeds_container_height'):
        validate_controls(dict(container_height_m=.5,height_m=.7))

def test_freeboard_uses_actual_container_height():
    c=validate_controls(dict(container_height_m=3.,camera_to_bottom_m=4.,height_m=2.,wave_m=.1))
    assert c['wave_m']==.1

def test_offscreen_boundary_uses_independent_dimensions_only():
    import numpy as np
    from liquid_depth.scene_console_web import simulated_dimension_geometry
    from liquid_depth.liquid_height_metrics import HeightCalibration
    k=np.array([[470.,0,240],[0,470.,135],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.2
    cal=HeightCalibration(k,pose,0.,(.55,.55),2.)
    bundle=simulated_dimension_geometry(np.full((270,480),1.2),cal,1.2,.6,.55,{'confirmed_empty':True})
    assert bundle['internal_dimensions_m']['height']==.6
    assert bundle['quality']['full_boundary_observed'] is False
    assert bundle['calibration']['radii_m']==[.55,.55]
    assert max(p[0] for p in bundle['projected_rim_outline_px'])>480

def test_independent_dimensions_do_not_override_wrong_empty_frame():
    import numpy as np
    from liquid_depth.scene_console_web import simulated_dimension_geometry
    from liquid_depth.liquid_height_metrics import HeightCalibration
    k=np.array([[470.,0,240],[0,470.,135],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1.2
    cal=HeightCalibration(k,pose,0.,(.55,.55),2.)
    with pytest.raises(ValueError,match='empty_bottom_disagrees'):
        simulated_dimension_geometry(np.full((270,480),2.),cal,1.2,.6,.55,{})

def test_auto_framing_moves_camera_and_changes_intrinsics():
    from liquid_depth.scene_console_web import framing_parameters
    bottom,fov=framing_parameters(1.5,2.,True)
    assert 2.<bottom<3. and 99.<fov<=100.
    assert validate_controls(dict(container_height_m=1.5,camera_to_bottom_m=2.))['camera_to_bottom_m']==bottom

def test_standard_installation_keeps_distance_and_fov():
    from liquid_depth.scene_console_web import framing_parameters
    assert framing_parameters(1.5,4.,True)==(4.,54.)

def test_unphysical_auto_framing_not_silently_clipped():
    with pytest.raises(ValueError,match='full_view_cannot_fit'):
        validate_controls(dict(container_height_m=9.8,camera_to_bottom_m=10.))

@pytest.mark.parametrize('flag',[1,'true',None])
def test_auto_frame_must_be_boolean(flag):
    with pytest.raises(ValueError,match='boolean_auto_frame'):
        validate_controls(dict(auto_frame=flag))
