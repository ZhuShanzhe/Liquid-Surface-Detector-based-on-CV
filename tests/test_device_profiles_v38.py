import numpy as np
import pytest
from liquid_depth.device_profiles_v38 import PROFILES,noisy_depth,metric_summary,BANDS,SCENES

def test_geometry_and_budget():
    assert [p.width for p in PROFILES]==[640,1280]
    assert PROFILES[1].baseline_m==.1
    assert 620<PROFILES[1].fx<640
    assert len(BANDS)*3*2*len(SCENES)==660
    assert 3/PROFILES[1].fps==.3

@pytest.mark.parametrize('profile',PROFILES)
@pytest.mark.parametrize('regime',['nominal','adverse'])
def test_repeatable_and_missing(profile,regime):
    z=np.full((96,96),4.);z[:5]=0
    a=noisy_depth(z,profile,123,0,regime)
    assert np.array_equal(a,noisy_depth(z,profile,123,0,regime))
    assert not np.array_equal(a,noisy_depth(z,profile,123,1,regime))
    assert not a[:5].any()
    assert np.isfinite(a).all()
    assert np.max(abs(a*1000-np.round(a*1000)))<.001

def test_tof_radial_range():
    z=np.full((32,32),8.)
    a=noisy_depth(z,PROFILES[0],3,0,ray_norm=np.full(z.shape,1.4))
    assert not a.any()

def test_signed_unclipped_metrics_and_denominators():
    rows=[dict(cluster='a',true_height_m=0.,predicted_height_m=-.003,empty=True),
          dict(cluster='b',true_height_m=.1,predicted_height_m=.12,empty=False),
          dict(cluster='c',true_height_m=1.,predicted_height_m=None,empty=False)]
    s=metric_summary(rows)
    assert s['frames']==3 and s['outputs']==2 and s['relative_outputs']==1
    assert s['pass_pct']==pytest.approx(100/3)
    assert sum(s['absolute']['counts'])==2 and sum(s['relative']['counts'])==1
    assert s['absolute']['pct_all'][0]==pytest.approx(100/3)
    assert s['relative_undefined_empty']==1

def test_bad_regime():
    with pytest.raises(ValueError):noisy_depth(np.ones((2,2)),PROFILES[0],1,0,'fake')

def test_empty_summary():
    assert metric_summary([])['outputs']==0
    assert metric_summary([])['absrel_positive_pct'] is None
