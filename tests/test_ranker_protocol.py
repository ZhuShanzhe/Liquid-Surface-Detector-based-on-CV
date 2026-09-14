import importlib
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.models.liquid_candidate import LiquidCandidateNet


@pytest.fixture
def protocol(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('train_candidate_ranker_v12')


def test_feature_exposure_preserves_all_frozen_candidate_depths(protocol):
    b=UniversalLiquidSurfaceNet(base_channels=8,num_ray_layers=4,ray_layer_hidden_channels=8).eval()
    m=LiquidCandidateNet(b,False).eval().requires_grad_(False)
    x=torch.rand(1,5,16,24)
    with torch.no_grad():
        normal=m(x);featured=protocol.candidate_prediction(m,x,SimpleNamespace(feature_context=True))
    for key,value in normal.items():torch.testing.assert_close(value,featured[key],rtol=0,atol=0)
    assert not featured['rank_features'].requires_grad


@pytest.mark.parametrize('existing',['frozen_calibration_baseline.json','test_started.json'])
def test_calibration_baseline_cannot_be_refitted_after_freeze(protocol,tmp_path,existing):
    (tmp_path/existing).write_text('{}')
    with pytest.raises(FileExistsError):protocol.freeze_calibration_baseline(SimpleNamespace(run_dir=tmp_path))
