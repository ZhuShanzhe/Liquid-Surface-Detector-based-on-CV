from types import SimpleNamespace
import numpy as np
import pytest
from liquid_depth import compound_tof_repair as m

def fixture(monkeypatch):
    k=np.array([[100.,0,24.],[0,100.,24.],[0,0,1.]])
    t=np.diag([1.,-1.,-1.,1.]);t[2,3]=5.
    cal=SimpleNamespace(intrinsics=k,camera_to_world_cv=t,radii_m=(1.2,1.2),footprint_exponent=2.,bottom_z_m=0.)
    monkeypatch.setattr(m,'height_calibration',lambda b:cal)
    base=SimpleNamespace(bundle={'sensor_family':'tof','rim_z_m':2.},ordinary_container_roi=np.ones((48,48),bool))
    return m.CompoundToFRepair(base,np.full((48,48),5.,np.float32))

def test_native_validity_is_not_output_clip():
    raw=np.array([[0.,.001,.079,.08,.1,10.2,np.nan,np.inf]],np.float32)
    d,a=m.sanitize_capture(raw)
    assert np.array_equal(d[0,:6],[0,0,0,raw[0,3],raw[0,4],raw[0,5]])
    assert a['no_maximum_depth_cutoff'] and d[0,5]>10

def test_explicit_native_invalid():
    raw=np.ones((2,2),np.float32);mask=np.array([[1,0],[1,1]],bool)
    d,a=m.sanitize_capture(raw,validity=mask);assert d[0,1]==0 and a['rejected_native_invalid']==1

def test_bad_validity_shape():
    with pytest.raises(ValueError):m.sanitize_capture(np.ones((3,3)),validity=np.ones((2,2),bool))

def test_bad_validity_dtype():
    with pytest.raises(ValueError):m.sanitize_capture(np.ones((3,3)),validity=np.ones((3,3)))

def test_real_surface_beats_empty_echo(monkeypatch):
    r=fixture(monkeypatch);raw=np.full((48,48),4.2,np.float32);raw[::3,::3]=5
    h,peak,diag,_=r.estimate_surface(raw,.001)
    assert abs(h-.8)<1e-5 and diag['liquid_replaced'] and abs(peak-.8)<1e-5

def test_empty_is_not_forced_to_zero(monkeypatch):
    r=fixture(monkeypatch);h,_,diag,_=r.estimate_surface(np.full((48,48),5.),-.003)
    assert h==-.003 and not diag['liquid_replaced']

def test_small_residual_is_preserved(monkeypatch):
    r=fixture(monkeypatch);h,_,diag,_=r.estimate_surface(np.full((48,48),4.2),.81)
    assert h==.81 and not diag['liquid_replaced']

def test_small_float_patch_cannot_move_liquid(monkeypatch):
    r=fixture(monkeypatch);raw=np.full((48,48),5.);raw[20:24,20:24]=4.5
    h,_,diag,_=r.estimate_surface(raw,.002)
    assert h==.002 and not diag['liquid_replaced']

def test_isolated_high_spike_is_not_peak():
    h=np.ones((20,20));h[10,10]=4.;peak,n=m.supported_local_peak(h,np.ones(h.shape,bool),np.full(h.shape,.003))
    assert peak<1.01

def test_broad_coherent_top_is_retained():
    h=np.ones((20,20));h[8:13,8:13]=1.2;peak,n=m.supported_local_peak(h,np.ones(h.shape,bool),np.full(h.shape,.003))
    assert abs(peak-1.2)<1e-8

def test_visual_free_repair_does_not_need_rgb(monkeypatch):
    r=fixture(monkeypatch);old={'liquid_depth_m':.001,'highest_height_m':1.8,'route':'baseline'}
    out=r.repair(old,np.full((48,48),4.2))
    assert abs(out['liquid_depth_m']-.8)<1e-6 and not out['liquid_visual_guidance_used']

def test_empty_input_keeps_numeric_fallback(monkeypatch):
    r=fixture(monkeypatch);old={'liquid_depth_m':-.004,'highest_height_m':.01}
    out=r.repair(old,np.zeros((48,48)))
    assert out['liquid_depth_m']==-.004 and out['highest_height_m']==.01

def test_public_capture_validity_shape():
    from liquid_depth.unified_tof import ToFObservation
    x=ToFObservation('t',np.zeros((3,3,3),np.uint8),np.ones((3,3)),0.,capture_validity=np.ones((2,2),bool))
    with pytest.raises(ValueError):x.validate((3,3))

def test_public_capture_validity_accepted():
    from liquid_depth.unified_tof import ToFObservation
    x=ToFObservation('t',np.zeros((3,3,3),np.uint8),np.ones((3,3)),0.,capture_validity=np.ones((3,3),bool))
    x.validate((3,3))

def test_kernel_warmup_discards_predictions_and_is_idempotent(monkeypatch):
    from liquid_depth.unified_tof import UnifiedToFMeter
    import liquid_depth.rgbd_evidence_model as ev
    calls=[]
    monkeypatch.setattr(ev,'observed_features',lambda *args:np.zeros((16,4,4),np.float32))
    f=SimpleNamespace(predictor=SimpleNamespace(predict=lambda x:calls.append(x.shape)))
    meter=object.__new__(UnifiedToFMeter);meter._peak_warmed=False;meter.shape=(4,4);meter.roi=np.ones((4,4),bool)
    meter.installation=SimpleNamespace(dry_depth_m=np.ones((4,4)),geometry={})
    meter.geometric=SimpleNamespace(backend=SimpleNamespace(backend=SimpleNamespace(_float_meter=f)))
    meter.warm_optional_peak_models();meter.warm_optional_peak_models()
    assert calls==[(16,4,4)]
