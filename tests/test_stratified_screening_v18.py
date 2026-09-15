import importlib.util
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_stratified_seegroup_v18 import suppress_components,align_top_bottom

def test_suppression_compares_last_retained_peak():
    depth=np.array([.03,.041,.055,.1])[:,None,None]
    d,active=suppress_components(depth,np.ones_like(depth,dtype=bool))
    np.testing.assert_array_equal(active[:,0,0],[True,False,True,True])

def test_suppression_rejects_invalid_depth_without_hiding_valid_peak():
    depth=np.array([np.nan,.01,.1,.15])[:,None,None]
    d,active=suppress_components(depth,np.ones_like(depth,dtype=bool))
    np.testing.assert_allclose(d[active],[.1,.15])

def test_oracle_top_bottom_alignment_preserves_internal_fraction():
    depth=np.broadcast_to(np.array([2.,3.,5.,7.])[:,None,None],(4,2,2)).copy()
    aligned,valid,fit=align_top_bottom(depth,np.ones_like(depth,dtype=bool),np.ones((2,2),bool),.3,.4)
    np.testing.assert_allclose(aligned[:,0,0],[.3,.32,.36,.4])
    assert fit['support']==4 and valid.all()

def test_oracle_requires_distinct_supported_endpoints():
    depth=np.ones((4,2,2));active=np.ones_like(depth,bool)
    out,valid,fit=align_top_bottom(depth,active,np.ones((2,2),bool),.3,.4)
    assert fit is None and not valid.any()
