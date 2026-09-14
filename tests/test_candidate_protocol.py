"""Research protocol guards; never use test feedback to choose a threshold."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('train_liquid_candidates_v11')


def report():
    row = dict(depth_mae_mm=15., depth_abs_rel=.025, valid_pixel_coverage=.6,
               pixel_tolerance_pass=.6, non_target_output_fraction=.05,
               within_tolerance_coverage=.36)
    return {'summary': {'all': {'selected': {str(t): deepcopy(row) for t in (.1,.3,.5,.7)}}}}


def test_threshold_uses_quality_and_coverage_not_mae_alone(protocol):
    base = report(); candidate = report()
    rows = candidate['summary']['all']['selected']
    rows['0.1'].update(depth_mae_mm=30., within_tolerance_coverage=.9)
    rows['0.3'].update(within_tolerance_coverage=.4)
    rows['0.5'].update(depth_mae_mm=2., valid_pixel_coverage=.1)
    rows['0.7'].update(depth_mae_mm=None)
    assert protocol.choose_operating_point(candidate, base) == .3


def test_no_threshold_is_a_valid_result(protocol):
    base = report(); candidate = report()
    for row in candidate['summary']['all']['selected'].values():
        row['valid_pixel_coverage'] = .01
    assert protocol.choose_operating_point(candidate, base) is None


@pytest.mark.parametrize('existing', ['baseline_test.json', 'frozen_test_selection.json'])
def test_selection_cannot_be_reopened_after_freeze_or_test(protocol, tmp_path, existing):
    (tmp_path / existing).write_text('{}')
    with pytest.raises(RuntimeError, match='already frozen'):
        protocol.select_for_test(SimpleNamespace(run_dir=tmp_path))
