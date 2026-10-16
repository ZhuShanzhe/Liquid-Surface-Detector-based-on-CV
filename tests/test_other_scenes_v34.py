import importlib.util
from pathlib import Path
import numpy as np
from liquid_depth.liquid_height_metrics import HeightCalibration,estimate_height

spec=importlib.util.spec_from_file_location('summary34',Path(__file__).resolve().parents[1]/'scripts/summarize_other_scenes_v34.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)

def row(error=None,truth=.1):
    p={'numeric':error is not None,'reasons':['missing'] if error is None else [],'statistics':None}
    if error is not None:p.update(absolute_error_m=abs(error),signed_error_m=error,abs_rel=abs(error)/truth if truth else None,
        within_tolerance=abs(error)<=max(.005,.02*truth),extrema_error_m={k:abs(error) for k in ('min_m','max_m','p05_m','p95_m')})
    return dict(cluster='one',mask_iou=1.,predicted_mask_precision=1.,raw_surface_valid_fraction=.5,methods={'refined':p})

def test_missing_remains_in_histogram_denominator():
    s=summary.summarize([row(.003),row(.03),row(None)],'refined')
    assert s['numeric_coverage']==2/3
    assert s['missing_frames']==1
    assert sum(s['absolute_bins_counts'])==2
    assert abs(sum(s['absolute_bins_percent_all'])+100/3-100)<1e-8
    assert s['all_frame_pass_percent']==100/3

def test_empty_relative_error_not_evaluated():
    s=summary.summarize([row(-.003,0)],'refined')
    assert s['relative_evaluable_frames']==0
    assert s['abs_rel_percent'] is None
    assert s['signed_bias_mm']==-3

def test_height_bands_use_actual_height_not_camera_distance():
    assert summary.band(0,0)=='empty'
    assert summary.band(.05)=='0-0.1'
    assert summary.band(2.7)=='2-3'
    assert summary.band(8.99)=='8-9'

def test_negative_height_is_retained_without_zero_snap():
    k=np.array([[120,0,39.5],[0,120,29.5],[0,0,1.]])
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1
    cal=HeightCalibration(k,pose,0,(.25,.2))
    out=estimate_height(np.full((60,80),1.01),np.ones((60,80),bool),cal)
    assert out['statistics'] is not None
    assert abs(out['statistics']['mean_m']+.01)<1e-10
    assert not out['accepted']
