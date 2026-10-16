import numpy as np
import pytest
from liquid_depth import transparent_relative_v29 as rel
from liquid_depth.relative_runtime_v29 import FixedCameraRelativeMeter
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom

def test_parallel_resident_wrapper_matches_geometry_and_copies_priors(monkeypatch):
    cal=dict(bottom_m=8.,fx=450.,cx=240.,cy=160.,container_max_height_m=.2)
    hs=[0.,.03,.08,.13,.18]
    def image(h):return {ch:np.full((8,8),h) for ch in ('left','right')}
    def fake_track(a,b,cal,ch):
        x,y=np.meshgrid(np.linspace(-.4,.4,10),np.linspace(-.25,.25,8));q=np.c_[x.ravel(),y.ravel()]
        other=inverse_bottom(bottom_map(q,float(a[0,0]),1.37,8.),float(b[0,0]),1.37,8.)
        return dict(anchor_q=q,current_q=other,weights=np.ones(len(q)),regions=len(q),channel=ch)
    monkeypatch.setattr(rel,'track_pair',fake_track)
    priors=[image(h) for h in hs]
    with FixedCameraRelativeMeter(priors,hs,cal,'transparent',workers=4) as meter:
        priors[0]['left'][:]=99.
        p=meter.measure(image(.047))
        assert abs(p['height_m']-.047)<1e-6
        assert p['qualified'] is False and p['production_enabled'] is False
        assert p['processing_ms']>0 and p['acquisition_included'] is False
        assert meter.images[0]['left'][0,0]==0.

def test_wrapper_rejects_non_five_budget():
    with pytest.raises(ValueError,match='exactly_five'):
        FixedCameraRelativeMeter([],[],{},'transparent')
