import numpy as np
import pytest
from liquid_depth.refraction_inverse_v19 import ray_factors,bottom_mapping,calibrate_empty,estimate_layers

def test_snell_tof_factor_matches_explicit_trigonometry():
    angle=np.deg2rad(np.array([0.,15.,30.]))
    n=1.47
    expected=n*np.cos(angle)/np.cos(np.arcsin(np.sin(angle)/n))
    np.testing.assert_allclose(ray_factors(n,np.tan(angle)**2),expected,rtol=1e-12)

def test_unrefracted_mapping_returns_original_reference_ray():
    x=np.array([0.,.1,.3]);y=x/2
    a,b=bottom_mapping(x,y,.3,.025,.1,1.,1.,.4)
    np.testing.assert_allclose(a,x);np.testing.assert_allclose(b,y)

def test_background_warp_cannot_determine_layer_order_by_itself():
    x=np.array([.05,.2,.4]);y=x*.7
    a,b=bottom_mapping(x,y,.3,.03,.1,1.47,1.333,.4)
    c,d=bottom_mapping(x,y,.3,.07,.1,1.333,1.47,.4)
    np.testing.assert_allclose(a,c);np.testing.assert_allclose(b,d)

def test_equal_indices_hide_internal_interface_position():
    x=np.array([.1,.3]);y=x/3
    a,b=bottom_mapping(x,y,.3,.01,.1,1.333,1.333,.4)
    c,d=bottom_mapping(x,y,.3,.08,.1,1.333,1.333,.4)
    np.testing.assert_allclose(a,c);np.testing.assert_allclose(b,d)

def test_normal_incidence_tof_and_visual_depth_have_opposite_index_effect():
    n=1.47
    assert ray_factors(n,np.array([0.]))[0]==pytest.approx(n)
    eps=1e-7
    a,_=bottom_mapping(np.array([eps]),np.array([0.]),0.,.02,.02,n,n,.02)
    assert (a[0]/eps)==pytest.approx(1/n)

def test_empty_calibration_uses_measured_central_plane_and_excludes_surround():
    z=np.full((100,160),.515);z[20:80,30:130]=.5
    r=calibrate_empty(z)
    assert r['bottom_depth_m']==pytest.approx(.5)
    assert not r['roi'][0,0] and r['roi'][50,80]
    assert r['roi'].sum()<6000

def test_missing_empty_measurements_are_rejected():
    with pytest.raises(ValueError):calibrate_empty(np.zeros((80,120)))

def test_stereo_cannot_be_silently_interpreted_as_tof():
    im=np.zeros((80,120,3),np.float32);z=np.full((80,120),.4)
    r=estimate_layers(im,z,im,z,dict(sensor_model='active_stereo'))
    assert r['status']=='unsupported_sensor_model' and not r['accepted']

def test_empty_scene_does_not_create_internal_measurement():
    im=np.zeros((80,120,3),np.float32);z=np.full((80,120),.4)
    cam=dict(sensor_model='nondispersive_coaxial_tof_proxy',fx=100,fy=100,cx=60,cy=40)
    r=estimate_layers(im,z,im,z,cam)
    assert not r['accepted'] and r['thickness_m'] is None

def test_misaligned_reference_dimensions_are_rejected():
    im=np.zeros((80,120,3),np.float32);z=np.full((80,120),.4)
    cam=dict(sensor_model='nondispersive_coaxial_tof_proxy',fx=100,fy=100,cx=60,cy=40)
    with pytest.raises(ValueError):estimate_layers(im,z[:70],im,z,cam)


def test_competing_homogeneous_medium_prevents_confident_internal_output():
    from liquid_depth.refraction_evidence_v19 import qualify_profile
    r=dict(top_depth_m=.3,profile=[dict(n_upper=1.4,n_lower=1.4,thickness_m=.02,raw_loss=1.,photo_error=0.,chromatic_error=0.),dict(n_upper=1.5,n_lower=1.33,thickness_m=.021,raw_loss=2.,photo_error=0.,chromatic_error=0.)])
    q=qualify_profile(r)
    assert q['geometric_accepted'] and not q['accepted']
    assert q['thickness_m'] is None
    assert q['status']=='homogeneous_medium_not_excluded'

def test_separated_supported_profile_reports_candidate_but_not_proven_identity():
    from liquid_depth.refraction_evidence_v19 import qualify_profile
    r=dict(top_depth_m=.3,profile=[dict(n_upper=1.5,n_lower=1.33,thickness_m=.02,raw_loss=1.,photo_error=0.,chromatic_error=0.),dict(n_upper=1.49,n_lower=1.34,thickness_m=.021,raw_loss=2.,photo_error=0.,chromatic_error=0.)])
    q=qualify_profile(r)
    assert q['accepted'] and not q['liquid_identity_verified']
    assert .02<=q['thickness_m']<=.021

def test_large_thickness_ambiguity_is_not_hidden_by_reference_fit():
    from liquid_depth.refraction_evidence_v19 import qualify_profile
    r=dict(top_depth_m=.3,profile=[dict(n_upper=1.5,n_lower=1.33,thickness_m=.02,raw_loss=1.,photo_error=0.,chromatic_error=0.),dict(n_upper=1.49,n_lower=1.34,thickness_m=.05,raw_loss=2.,photo_error=0.,chromatic_error=0.)])
    q=qualify_profile(r)
    assert not q['accepted'] and q['thickness_m'] is None
