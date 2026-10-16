import json,threading,http.client
from pathlib import Path
import cv2
import numpy as np
import pytest
from liquid_depth.empty_geometry_v41 import build_geometry,height_calibration,refraction_geometry,make_refraction_meter
from liquid_depth.empty_geometry_runtime_v41 import create_empty_geometry_profile,EmptyGeometryLiquidDepthSystem
from liquid_depth.rail_runtime import make_product_system
from liquid_depth.product_web import PanelState,PanelHandler,ThreadingHTTPServer


def fixture(shape='circle',bottom=2.,tilt=0.):
    size=(240,320);k=np.array([[300.,0,159.5],[0,300.,119.5],[0,0,1.]])
    a=np.deg2rad(tilt);r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])@np.diag([1.,-1.,-1.])
    camera=np.array([.05,0,bottom]);rx=.5;ry=.5 if shape=='circle' else .4
    def project(world):
        p=(np.asarray(world)-camera)@r
        q=p@k.T;return q[:,:2]/q[:,2,None]
    if shape=='rectangle':xy=np.array([[-rx,-ry],[rx,-ry],[rx,ry],[-rx,ry]])
    else:
        aa=np.linspace(0,2*np.pi,12,endpoint=False);xy=np.column_stack((rx*np.cos(aa),ry*np.sin(aa)))
    outline=project(np.column_stack((xy,np.zeros(len(xy)))))
    anchor=project([[-.3,0,0],[.3,0,0]]);check=project([[0,-.2,0],[0,.2,0]])
    yy,xx=np.indices(size);rays=np.stack(((xx-k[0,2])/k[0,0],(yy-k[1,2])/k[1,1],np.ones(size)),axis=-1)@r.T
    c=-rays[...,2]
    spec=dict(shape=shape,bottom_outline_px=outline.tolist(),wall_height_m=.6,baseline_m=.1,
        metric_anchor=dict(kind='span',pixels=anchor.tolist(),length_m=.6),
        check_spans=[dict(pixels=check.tolist(),length_m=.4)],confirmed_empty=True,horizontal_flat_bottom=True,
        fixed_camera=True,rectified_aligned=True)
    return k,(bottom/c).astype(np.float32),spec,r,c


@pytest.mark.parametrize('shape',['circle','ellipse','rectangle'])
@pytest.mark.parametrize('tilt',[0.,8.])
def test_recovers_geometric_model(shape,tilt):
    k,dry,spec,r,c=fixture(shape,tilt=tilt);b=build_geometry(dry,k,spec);cal=height_calibration(b)
    assert b['bottom_plane_camera']['perpendicular_distance_m']==pytest.approx(2.,abs=1e-5)
    assert sorted(cal.radii_m)==pytest.approx(sorted([.5,.5 if shape=='circle' else .4]),abs=2e-5)
    assert b['quality']['camera_axis_tilt_deg']==pytest.approx(tilt,abs=.001)
    assert not b['production_qualified']
    assert len(b['quality']['independent_dimension_checks'])==1


def test_independent_distance_alternative():
    k,dry,spec,_,_=fixture();spec['metric_anchor']=dict(kind='perpendicular_distance',distance_m=2.)
    assert build_geometry(dry,k,spec)['bottom_plane_camera']['perpendicular_distance_m']==2.


def test_missing_metric_anchor_is_not_replaced_by_raw_depth():
    k,dry,spec,_,_=fixture();spec.pop('metric_anchor')
    with pytest.raises(ValueError,match='independent_metric_anchor'):build_geometry(dry,k,spec)


@pytest.mark.parametrize('flag',['confirmed_empty','horizontal_flat_bottom','fixed_camera','rectified_aligned'])
def test_explicit_installation_confirmation(flag):
    k,dry,spec,_,_=fixture();spec[flag]=False
    with pytest.raises(ValueError,match='confirmation'):build_geometry(dry,k,spec)


def test_bad_or_missing_bottom_support():
    k,dry,spec,_,_=fixture()
    with pytest.raises(ValueError,match='support_insufficient'):build_geometry(dry*0,k,spec)


def test_incomplete_circle_is_not_hallucinated():
    k,dry,spec,_,_=fixture();spec['bottom_outline_px']=spec['bottom_outline_px'][:6]
    spec['metric_anchor']=dict(kind='perpendicular_distance',distance_m=2.);spec['check_spans']=[]
    with pytest.raises(ValueError,match='incomplete_bottom'):build_geometry(dry,k,spec)


def test_independent_check_cannot_reuse_fit_span():
    k,dry,spec,_,_=fixture();spec['check_spans']=[dict(pixels=spec['metric_anchor']['pixels'],length_m=.6)]
    with pytest.raises(ValueError,match='independent_of_fit'):build_geometry(dry,k,spec)


def test_dimension_check_fails_instead_of_rescaling():
    k,dry,spec,_,_=fixture();spec['check_spans'][0]['length_m']=.5
    with pytest.raises(ValueError,match='dimension_check_failed'):build_geometry(dry,k,spec)


def test_near_ten_values_not_clipped():
    k,dry,spec,_,_=fixture(bottom=10.05)
    spec['metric_anchor']=dict(kind='perpendicular_distance',distance_m=10.05);spec['check_spans']=[]
    b=build_geometry(dry,k,spec)
    assert b['bottom_plane_camera']['perpendicular_distance_m']>10.
    assert any('numeric_value_retained' in x for x in b['warnings'])


def test_short_metric_span_has_explicit_failure():
    k,dry,spec,_,_=fixture(bottom=10.05)
    with pytest.raises(ValueError,match='metric_span_too_short'):build_geometry(dry,k,spec)


def test_refraction_adapter_requires_supported_pose():
    k,dry,spec,_,_=fixture();b=build_geometry(dry,k,spec)
    ref=refraction_geometry(b,k,left_intrinsics=k)
    assert ref['bottom_m']==pytest.approx(2.,abs=1e-5)
    assert ref['requires_one_empty_plus_four_known_liquid_levels']
    k,dry,spec,_,_=fixture(tilt=8.)
    with pytest.raises(ValueError,match='near_vertical'):refraction_geometry(build_geometry(dry,k,spec),k,left_intrinsics=k)


def test_independent_gravity_is_used_and_checked():
    k,dry,spec,r,c=fixture(tilt=4.);spec['up_camera']=r[2].tolist()
    b=build_geometry(dry,k,spec)
    assert b['quality']['normal_source']=='independent_gravity'
    assert 'no_independent_gravity_check' not in b['warnings']
    spec['up_camera']=[0.,.5,-1.]
    with pytest.raises(ValueError,match='disagrees_with_independent_gravity'):build_geometry(dry,k,spec)


def test_transparent_factory_passes_derived_geometry_and_requires_five_priors(monkeypatch):
    import liquid_depth.range_runtime_v33 as runtime
    k,dry,spec,_,_=fixture();b=build_geometry(dry,k,spec)
    masks={ch:np.ones(dry.shape,np.uint8) for ch in ('left','right')}
    heights=[0.,.1,.2,.3,.4];priors=[{'left':dry,'right':dry} for _ in heights]
    monkeypatch.setattr(runtime,'FivePriorImageMeter',lambda *a:a)
    meter=make_refraction_meter(b,k,masks,priors,heights,'transparent',left_intrinsics=k)
    assert meter[0]['bottom_m']==pytest.approx(2.,abs=1e-5)
    assert meter[0]['geometry_source']=='empty_geometry_v41'
    with pytest.raises(ValueError,match='one_empty'):
        make_refraction_meter(b,k,masks,priors,[.01,.1,.2,.3,.4],'transparent',left_intrinsics=k)
    with pytest.raises(ValueError,match='distinct'):
        make_refraction_meter(b,k,masks,priors,[0.,.1,.1,.3,.4],'transparent',left_intrinsics=k)
    left=k.copy();left[0,2]+=1
    with pytest.raises(ValueError,match='actual_rectified_left'):
        make_refraction_meter(b,k,masks,priors,heights,'transparent',left_intrinsics=left)


def write_frame(folder,k,raw):
    folder.mkdir()
    cv2.imwrite(str(folder/'rgb.png'),np.full((*raw.shape,3),100,np.uint8))
    np.save(folder/'depth.npy',raw)
    (folder/'depth_info.json').write_text(json.dumps(dict(K=k.ravel().tolist(),width=raw.shape[1],height=raw.shape[0])))
    return folder


def test_end_to_end_profile_factory_and_signed_measurement(tmp_path):
    k,dry,spec,r,c=fixture(tilt=4.)
    frame=write_frame(tmp_path/'empty',k,dry)
    profile,b=create_empty_geometry_profile(dict(frame_dir=str(frame),depth_scale_to_m=1.,geometry=spec,profile_path=str(tmp_path/'site.yaml')))
    meter=make_product_system(profile)
    assert isinstance(meter,EmptyGeometryLiquidDepthSystem)
    for h in [-.01,0.,.1,.3]:
        current=write_frame(tmp_path/('frame_'+str(h)),k,((2-h)/c).astype(np.float32))
        result=meter.measure(current,tmp_path/('output_'+str(h)))
        assert result['liquid_depth_candidate_m']==pytest.approx(h,abs=.0003)
        assert not result['accepted'] and result['liquid_depth_m'] is None
        assert (tmp_path/('output_'+str(h))/'depth_result.json').is_file()
    with pytest.raises(ValueError,match='profile_exists'):
        create_empty_geometry_profile(dict(frame_dir=str(frame),depth_scale_to_m=1.,geometry=spec,profile_path=str(profile)))


def test_http_preview_save_and_measure(tmp_path):
    k,dry,spec,r,c=fixture();root=tmp_path/'capture';root.mkdir();frame=write_frame(root/'empty',k,dry)
    state=PanelState(root,tmp_path/'out');handler=type('TestPanel',(PanelHandler,),{'state':state})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    def post(path,payload):
        conn=http.client.HTTPConnection('127.0.0.1',server.server_address[1]);conn.request('POST',path,json.dumps(payload),{'Content-Type':'application/json'});response=conn.getresponse();data=json.loads(response.read());conn.close();return response.status,data
    payload=dict(frame_dir=str(frame),depth_scale_to_m=1.,geometry=spec,profile_path=str(tmp_path/'http_site.yaml'))
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_address[1]);conn.request('GET','/empty_geometry.html')
        response=conn.getresponse();assert response.status==200
        assert b'/api/preview-empty-geometry' in response.read();conn.close()
        code,data=post('/api/preview-empty-geometry',payload);assert code==200 and data['geometry']['schema_version']==1
        assert not Path(payload['profile_path']).exists()
        code,data=post('/api/calibrate-empty-geometry',payload);assert code==200 and Path(data['profile_path']).exists()
        current=write_frame(root/'wet',k,((2-.2)/c).astype(np.float32))
        code,data=post('/api/measure',dict(frame_dir=str(current),profile_path=payload['profile_path']))
        assert code==200 and data['result']['liquid_depth_candidate_m']==pytest.approx(.2,abs=.0003)
        assert not data['result']['accepted']
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_changed_camera_intrinsics_require_new_calibration(tmp_path):
    k,dry,spec,_,_=fixture();frame=write_frame(tmp_path/'empty',k,dry)
    profile,_=create_empty_geometry_profile(dict(frame_dir=str(frame),depth_scale_to_m=1.,geometry=spec,profile_path=str(tmp_path/'site.yaml')))
    changed=k.copy();changed[0,0]+=1.
    current=write_frame(tmp_path/'changed',changed,dry)
    with pytest.raises(ValueError,match='recalibration_required'):EmptyGeometryLiquidDepthSystem(profile).measure(current)


def test_unusable_dry_reference_does_not_save_profile(tmp_path,monkeypatch):
    import liquid_depth.empty_geometry_runtime_v41 as runtime
    k,dry,spec,_,_=fixture();frame=write_frame(tmp_path/'empty',k,dry)
    def fail(*args):raise ValueError('insufficient independently observed dry depth')
    monkeypatch.setattr(runtime,'calibrate',fail)
    target=tmp_path/'bad.yaml'
    with pytest.raises(ValueError,match='independently observed dry'):
        create_empty_geometry_profile(dict(frame_dir=str(frame),depth_scale_to_m=1.,geometry=spec,profile_path=str(target)))
    assert not target.exists() and not (tmp_path/'bad_assets').exists()
