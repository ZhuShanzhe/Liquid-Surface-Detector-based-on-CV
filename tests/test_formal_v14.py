import hashlib,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from liquid_depth.dynamic_wet_interior import DynamicInteriorRepair
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.compound_tof_repair import sanitize_capture

@pytest.fixture
def repair():
    b=5.798;shape=(270,480);f=201.41
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=b
    k=np.array([[f,0,239.5],[0,f,134.5],[0,0,1.]])
    bundle=dict(sensor_family='tof',rim_z_m=4.879,calibration=dict(intrinsics=k.tolist(),camera_to_world_cv=pose.tolist(),bottom_z_m=0.,radii_m=[.55,.55],footprint_exponent=2.))
    dry=np.full(shape,b,np.float32);xyz=world_points(dry,height_calibration(bundle))
    mask=(xyz[...,:2]**2).sum(-1)<=.55**2
    return DynamicInteriorRepair(SimpleNamespace(bundle=bundle,ordinary_container_roi=mask),dry)

def test_dynamic_geometry_rescues_wrong_wall_level(repair):
    d=np.full(repair.dry.shape,5.798-3.5,np.float32)
    value,peak,diag,_=repair.estimate_surface(d,4.76)
    assert abs(value-3.5)<1e-5 and abs(peak-3.5)<1e-5
    assert diag['dynamic_interior']['replaced']

def test_correct_plane_unchanged(repair):
    value,_,diag,_=repair.estimate_surface(np.full(repair.dry.shape,5.798-.75,np.float32),.75)
    assert abs(value-.75)<1e-5 and not diag['dynamic_interior']['replaced']

def test_half_missing_retains_distributed_support(repair):
    d=np.full(repair.dry.shape,5.798-3.5,np.float32)
    d[np.random.default_rng(21).random(d.shape)<.5]=0
    value,_,diag,_=repair.estimate_surface(d,4.76)
    assert abs(value-3.5)<1e-5
    assert diag['dynamic_interior']['coherent_physical_tiles']>=6

def test_no_negative_output_clipping(repair):
    value,_,_,_=repair.estimate_surface(np.full(repair.dry.shape,5.798+.01,np.float32),-.01)
    assert value==-.01

def test_no_ten_meter_input_clipping():
    d=np.array([[10.01,15.,.01]],np.float32);out,audit=sanitize_capture(d)
    assert out[0,0]==d[0,0] and out[0,1]==15.
    assert audit['no_maximum_depth_cutoff']

def test_public_tof_constructor_defaults_to_dynamic_repair(repair,monkeypatch):
    from liquid_depth.tof_scene_meter import ToFSceneMeter,DeterminedSceneMeter
    from liquid_depth.supported_scene_meter import FormalSceneMeter
    monkeypatch.setattr(DeterminedSceneMeter,'from_empty_geometry',lambda *a,**k:object())
    monkeypatch.setattr(FormalSceneMeter,'from_empty_geometry',lambda *a,**k:repair.base)
    meter=ToFSceneMeter.from_empty_geometry(repair.base.bundle,repair.dry)
    assert isinstance(meter.repair,DynamicInteriorRepair)
    assert meter._small_float_peak_inputs is not None

@pytest.mark.skipif(sys.platform!='linux' or shutil.which('g++') is None,reason='official compiled runtime is Linux/GNU profile')
def test_compiled_forest_matches_trusted_sklearn_predictions():
    from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
    from liquid_depth.compiled_native_forest import CompiledForest
    rng=np.random.default_rng(21);x=rng.normal(size=(80,7)).astype(np.float32)
    model=ExtraTreesClassifier(n_estimators=8,random_state=21).fit(x,(x[:,0]>0).astype(int))
    compiled=CompiledForest(model)
    np.testing.assert_allclose(compiled.predict_proba(x),model.predict_proba(x),atol=1e-14,rtol=0)
    model=ExtraTreesRegressor(n_estimators=8,random_state=21).fit(x,x[:,1]+x[:,3])
    np.testing.assert_allclose(CompiledForest(model).predict(x),model.predict(x),atol=1e-14,rtol=0)

def test_formal_auxiliary_asset_integrity():
    directory=Path(__file__).resolve().parents[1]/'assets/models/new_view_v14'
    m=json.loads((directory/'manifest.json').read_text())
    assert m['semantic_accuracy_is_not_release_gate'] and m['ordinary_only']
    for name,digest in m['sha256'].items():
        assert Path(name).name==name
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
