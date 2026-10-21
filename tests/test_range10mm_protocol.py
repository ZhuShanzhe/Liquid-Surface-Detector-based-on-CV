"""Protocol regression tests; do not import or inspect wet prediction errors."""
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
CODE=HERE if (HERE/'range_protocol.py').exists() else HERE.parent/'scripts/evaluate_range10mm'
sys.path.insert(0,str(CODE))
from range_protocol import band_of,installation,QUOTAS
from diverse_geometry import calibration_value

def test_depth_band_endpoints():
    assert band_of(.0999) is None
    assert band_of(.1)==0 and band_of(1.)==1 and band_of(9.)==8
    assert band_of(9.0001) is None
    assert sum(QUOTAS)==100

def test_fixed_budget_independent_of_distance():
    rng=np.random.default_rng(9238)
    for value in (.1,1.,5.,10.):
        delta=np.array([calibration_value(value,rng)-value for _ in range(20000)])
        assert max(abs(delta))<=.01000000001
        assert abs(delta.mean())<.0001
        assert .0031<delta.std()<.0035

def test_full_range_physically_feasible_geometry():
    for band in range(9):
        for attempt in range(25):
            g,h=installation(band,attempt)
            assert .1<=h<=9 and .9<=g['bottom_m']<=10
            assert h<g['capacity_m']<g['bottom_m']
            assert .3<=g['radius_m']<=1.5
            assert 45<=g['horizontal_fov_deg']<=100

def test_declared_upper_boundary():
    g,h=installation(8,0)
    assert h==9. and g['bottom_m']<=10
