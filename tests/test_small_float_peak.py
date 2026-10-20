import inspect
from types import SimpleNamespace
import numpy as np
import pytest
from liquid_depth import small_float_peak as module
from liquid_depth.small_float_peak_core import choose_highest
from liquid_depth.tof_scene_meter import ToFSceneMeter, VERSION
from liquid_depth.unified_tof import UnifiedToFMeter


def candidate(height, probability=.9, correction=0.):
    return dict(height_m=height,height_raw_m=height-correction,object_probability=probability,
        top_probability=.8,correction_m=correction,noise_m=.005,pixels=4,bbox=[1,1,3,3])


def test_ranking_uses_height_not_larger_instance_or_confidence():
    low=candidate(.5,.99);low['pixels']=100
    high=candidate(.8,.8)
    assert choose_highest([low,high],dict(threshold=.6,top_threshold=0.)) is high


def test_large_correction_is_rejected_not_clipped():
    assert choose_highest([candidate(.8,correction=.1)],dict(threshold=.6,top_threshold=0.)) is None


@pytest.mark.parametrize('height,expected',[(.7,.7),(.3,.5)])
def test_visual_refinement_never_changes_liquid_or_input(monkeypatch,height,expected):
    m=object.__new__(module.SmallFloatPeakRefiner)
    m.bundle={};m.dry=np.ones((4,4));m.policy={};m.models={}
    m.predictor=SimpleNamespace(predict=lambda features:{})
    monkeypatch.setattr(module,'float_features',lambda *args:(None,None))
    monkeypatch.setattr(module,'dense_candidates',lambda *args:[candidate(height)])
    monkeypatch.setattr(module,'score_candidates',lambda candidates,models:candidates)
    monkeypatch.setattr(module,'choose_highest',lambda candidates,policy:candidates[0])
    before={'liquid_depth_m':-.003,'highest_height_m':.5,'diagnostics':{'sentinel':1},'reasons':['keep']}
    result=m.refine(before,np.ones((4,4)),np.zeros((4,4,3),np.uint8))
    assert result['liquid_depth_m']==-.003 and result['highest_height_m']==expected
    assert before=={'liquid_depth_m':-.003,'highest_height_m':.5,'diagnostics':{'sentinel':1},'reasons':['keep']}
    assert result['diagnostics']['small_float_peak']['liquid_modified'] is False
    assert result['reasons']==['keep']


def test_missing_rgb_keeps_frozen_output():
    m=object.__new__(module.SmallFloatPeakRefiner);result={'liquid_depth_m':.2}
    assert m.refine(result,None,None) is result


def test_v3_opaque_probe_default_and_phase_never_attaches_probe():
    assert VERSION=='tof_formal_v3_small_float_20261020'
    assert inspect.signature(UnifiedToFMeter).parameters['small_float_peak'].default is True
    m=ToFSceneMeter(None,sensor='tof',kind='five_phase_priors',scene='homogeneous_transparent')
    assert m._small_float_peak_inputs is None and m._small_float_peak is None
