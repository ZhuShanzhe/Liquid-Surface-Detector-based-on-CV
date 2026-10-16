"""No accepted outputs means absent conditional metrics, not a zero error."""
import importlib.util
import numpy as np
from pathlib import Path

path=Path(__file__).parents[1]/'scripts/calibrate_scenario_confidence.py'
spec=importlib.util.spec_from_file_location('height_confidence_calibration',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empty_acceptance_has_null_conditional_metrics():
    store=module._empty();store['frames']=10
    result=module._summarize(store)
    assert result['coverage']==0 and result['effective_pass_coverage']==0
    assert all(result[k] is None for k in ('mae_m','abs_rel','within_tolerance_rate','evaluable_acceptance_rate'))


def test_rejected_policy_counts_every_input_frame():
    store=module._empty();store.update(frames=10,accepted=8,evaluable=8,within=7,absolute=.08,relative=.08)
    result=module._apply_policy({'transparent':{'0.500000':module._summarize(store)}},
                               {'transparent':{'threshold':.5,'qualified':False}})
    assert result['overall']['frames']==10 and result['overall']['coverage']==0
    assert result['overall']['within_tolerance_rate'] is None


def test_no_evaluable_threshold_never_qualifies():
    metrics=module._summarize(module._empty())
    choice=module._choose({'0.500000':metrics},minimum_coverage=.8,minimum_evaluable_rate=.98,
                          maximum_abs_rel=.03,minimum_within_rate=.75)
    assert choice['qualified'] is False and choice['metrics'] is None


def test_failure_group_ratio_uses_prediction_only():
    raw=np.array([[True,False],[True,False]])
    predicted=np.array([[True,True],[False,False]])
    assert module._observed_valid_ratio(raw,predicted)==.5
    assert module._observed_valid_ratio(raw,np.zeros_like(predicted))==0
