import numpy as np
import pytest
from test_ordinary_v39 import setup
from liquid_depth.ordinary_surface_v39 import calibrate_dry_reference
from liquid_depth.ordinary_surface_v39b import estimate_v39

@pytest.mark.parametrize('height',[-.05,0.,.025,.7])
@pytest.mark.parametrize('sigma',[.18,.36])
def test_edge_of_range_has_no_invisible_surface_mode(height,sigma):
    cal,m,dry,wet=setup(bottom=9.95,height=height,sigma=sigma,seed=3993)
    ref=calibrate_dry_reference(dry,cal,.8,m)
    result=estimate_v39(wet,cal,.8,m,ref)
    assert result['statistics'] is not None
    assert result['statistics']['mean_m']==pytest.approx(height,abs=.065)
    if height<0:assert result['statistics']['mean_m']<0

def test_above_capacity_retained():
    cal,m,dry,wet=setup(bottom=2.,height=.8,sigma=.06)
    ref=calibrate_dry_reference(dry,cal,.5,m)
    assert estimate_v39(wet,cal,.5,m,ref)['statistics']['mean_m']>.7

def test_missing_depth_no_output():
    cal,m,dry,wet=setup();ref=calibrate_dry_reference(dry,cal,.5,m)
    assert estimate_v39(wet*0,cal,.5,m,ref)['statistics'] is None
