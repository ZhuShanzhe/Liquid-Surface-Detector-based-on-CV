"""Desktop adapters must preserve measurement isolation and calibration rules."""
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from liquid_depth.liquid_height_metrics import HeightCalibration
from liquid_depth.scene_console_web import (INSTALLATION_KEYS, depth_preview,
    framing_for_controls, simulation_request, validate_controls, validation_envelope)
from liquid_depth.simulation_calibration import (annotated_value, empty_calibration,
    phase_annotations,phase_modulation_qualification)


def calibration_fixture():
    k=np.array([[400.,0,239.5],[0,400.,134.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=4.
    return np.full((270,480),4.,np.float32),HeightCalibration(k,pose,0.,(.55,.55))


@pytest.mark.parametrize('shape', ['cylinder','ellipse','rectangle'])
@pytest.mark.parametrize('noise_mode', ['ideal','fixed_10mm'])
def test_empty_calibration_shapes_and_bounded_annotations(shape,noise_mode):
    dry,cal=calibration_fixture()
    bundle,audit=empty_calibration(dry,cal,np.ones(dry.shape,bool),shape=shape,
        radii_m=(.55,.55 if shape=='cylinder' else .7),capacity_m=1.5,bottom_m=4.,noise_mode=noise_mode)
    assert bundle['sensor_family']=='tof'
    assert audit['independent_of_wet_truth'] and not audit['hidden_bottom_outline_used']
    assert len(audit['metric_annotations'])==5
    assert all(abs(v['error_mm'])<=10 for v in audit['metric_annotations'].values())
    if noise_mode=='ideal':assert audit['bottom_distance_error_mm']==0


def test_hidden_bottom_uses_visible_rim_not_hidden_clicks():
    dry,cal=calibration_fixture()
    bundle,audit=empty_calibration(dry,cal,np.zeros(dry.shape,bool),shape='ellipse',
        radii_m=(.55,.7),capacity_m=1.5,bottom_m=4.,noise_mode='ideal')
    assert not audit['bottom_outline_visible']
    assert bundle['source']=='empty_rim_plus_independent_metric_and_gravity'


def test_fixed_noise_distribution_is_not_measurement_clipping():
    r=np.random.default_rng(211021)
    errors=np.array([annotated_value(2.,r,True)-2. for _ in range(20000)])
    assert abs(errors).max()<=.01
    assert abs(errors.mean())<.0001
    assert .0031<errors.std()<.0034
    # No output value is part of the noise function's interface.
    assert list(inspect.signature(annotated_value).parameters)==['value','rng','noisy']


def test_five_phase_priors_are_annotated_independently_and_keep_empty_zero():
    phase=dict(profile=dict(bottom_m=4.,container_max_height_m=1.5),
        known_heights_m=np.array([0,.1,.35,.65,.95])*1.5,prior_correlations='observed_only')
    inverse,audit=phase_annotations(phase,dict(metric_anchor_m=4.004,measured_capacity_m=1.499),noise_mode='fixed_10mm')
    assert inverse['known_heights_m'][0]==0
    assert inverse['profile']['bottom_m']==4.004
    assert all(abs(v['error_mm'])<=10 for v in audit['metric_annotations'].values())
    assert np.all(np.diff(inverse['known_heights_m'])>0)
    assert inverse['prior_correlations']=='observed_only'
    assert not audit['current_test_truth_used']


def test_default_and_parameter_change_boundaries():
    c=validate_controls({})
    assert c['calibration_noise_mode']=='fixed_10mm'
    assert 'height_m' not in INSTALLATION_KEYS and 'darkness' not in INSTALLATION_KEYS
    for key in ('material','container_shape','container_radius_x_m','camera_offset_y_m','calibration_noise_mode','refractive_index'):
        assert key in INSTALLATION_KEYS


@pytest.mark.parametrize('change', [dict(container_shape='triangle'),dict(container_radius_x_m=0),
    dict(camera_offset_x_m=.3),dict(container_radius_y_m=.7),dict(height_m=9.1),
    dict(loss=.51),dict(calibration_noise_mode='unbounded'),dict(unknown=1),
    dict(material='homogeneous_transparent',float_count=1)])
def test_invalid_controls_rejected(change):
    with pytest.raises(ValueError):validate_controls(change)


def test_automatic_framing_uses_shape_and_offset():
    c=validate_controls(dict(container_shape='ellipse',container_radius_x_m=.8,
        container_radius_y_m=1.1,camera_offset_x_m=.1,camera_offset_y_m=-.1,
        container_height_m=6.,camera_to_bottom_m=6.2))
    bottom,fov=framing_for_controls(c)
    assert 6.2<bottom<=10 and 54<=fov<=100
    assert c['container_radius_y_m']==1.1  # Never silently shrink the vessel.


def test_impossible_full_view_fails_without_geometry_shrink():
    with pytest.raises(ValueError,match='full_view_cannot_fit'):
        validate_controls(dict(container_height_m=9.8,camera_to_bottom_m=10.))


def test_exploratory_geometry_explicitly_marked():
    c=validate_controls(dict(container_shape='ellipse',container_radius_x_m=.2,
        container_radius_y_m=.4))
    result=validation_envelope(c)
    assert result['exploratory'] and not result['field_metrology_qualified']
    assert 'radius_outside_published_sampling' in result['reasons']
    assert 'aspect_outside_published_sampling' in result['reasons']


def test_empty_request_does_not_use_current_liquid_or_scene():
    c=validate_controls(dict(height_m=.8,glare=.6,wave_m=.05,float_count=3))
    request=simulation_request(c,seed=1,index=0,fov=60.,empty=True)
    assert request['height_m']==request['wave_m']==request['float_count']==request['glare']==0
    assert request['material']=='ordinary_diffuse'
    assert request['shape']=='cylindrical' and request['sensor_model']=='formal_diverse_v15'


def test_depth_preview_uses_observation_and_preserves_invalid_pixels():
    assert depth_preview(np.zeros((8,8)))==(None,None)
    d=np.full((8,8),3.);d[:2]=0
    image,limits=depth_preview(d)
    assert image.startswith('data:image/png;base64,') and limits==[3.,3.]


def test_historical_ast_evaluation_adapter_remains_compatible():
    path=Path(__file__).resolve().parents[1]/'scripts/render_scene_console.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    render=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render'))
    for fragment in ('gen.simulate_raw_depth(scene, labels)','gen.create_lighting(scene, rr)',
        'liquid_roughness=0.025 if glare else 0.22',
        "camera_position_m=(0.0, 0.0, b - h) if request.get('camera_mount') == 'top_center' else (0.025 * b, -0.025 * b, b - h)",
        'corruption_severity=max(0.18, dark * 0.7, glare * 0.8)',
        "np.savez_compressed(folder / 'offline_geometry.npz', true_depth_m=full, semantic=semantic)"):
        assert fragment in render


def test_blender_capture_paths_are_absolute_even_for_relative_work(tmp_path,monkeypatch):
    from liquid_depth import scene_console_web as module
    monkeypatch.chdir(tmp_path)
    console=object.__new__(module.Console)
    console.blender='blender';console.render_engine='cycles_cpu'
    def fake_run(command,**kwargs):
        out=Path(command[command.index('--out')+1])
        request=Path(command[command.index('--request')+1])
        assert out.is_absolute() and request.is_absolute()
        assert json.loads(request.read_text())['render_engine']=='cycles_cpu'
        out.mkdir()
        (out/'complete.json').write_text('{"complete":true}')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',fake_run)
    console.capture({'height_m':.6},Path('frame'))


def test_cpu_renderer_is_explicit_not_optix_auto_fallback():
    source=(Path(__file__).resolve().parents[1]/'scripts/render_scene_console.py').read_text()
    assert "render_scene.cycles.device='CPU'" in source
    assert "render_scene.cycles.samples=12;render_scene.cycles.use_denoising=False" in source
    assert "render_engine=request.get('render_engine','eevee')" in source


def test_phase_selected_fit_support_is_stricter_than_global_30_pixel_check():
    capture=np.zeros((3,4,10,10));capture[:,0]=20
    phase=dict(roi=np.ones((10,10),bool),prior_correlations=[capture.copy() for _ in range(5)])
    assert phase_modulation_qualification(phase,capture)['eligible']
    current=capture.copy();current[:,:,:3]=0;current[:,:,3,:7]=0
    check=phase_modulation_qualification(phase,current)
    assert check['phase_selected_current_support_pixels']==63
    assert not check['eligible'] and not check['uses_estimate_or_estimation_error']
    current[:,:,3,6]=capture[:,:,3,6]
    assert phase_modulation_qualification(phase,current)['eligible']


def test_phase_prior_common_support_is_checked_before_fitting():
    capture=np.zeros((3,4,10,10));capture[:,0]=20
    priors=[capture.copy() for _ in range(5)];priors[3][:,:,:4]=0
    check=phase_modulation_qualification(dict(roi=np.ones((10,10),bool),prior_correlations=priors))
    assert not check['eligible'] and check['phase_prior_modulation_support_pixels']==60
