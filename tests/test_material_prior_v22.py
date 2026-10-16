import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('material_prior_cli', Path(__file__).parents[1]/'scripts/material_prior_v22.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_extract_requires_empty_roi_and_is_label_free():
    empty = np.ones((180, 320))
    raw = np.full_like(empty, .8)
    k = [[250, 0, 160], [0, 250, 90], [0, 0, 1]]
    obs, bottom = cli.extract(raw, empty, empty > 0, k, True)
    assert bottom == 1
    assert set(obs) == {'value_m', 'qx'}
    assert len(obs['value_m']) >= 16
    assert np.allclose(obs['value_m'], .2)


def test_sloping_empty_bottom_rejected():
    empty = np.tile(np.linspace(.8, 1.2, 320), (180, 1))
    with pytest.raises(ValueError, match='perpendicular'):
        cli.extract(empty, empty, empty > 0, [[250, 0, 160], [0, 250, 90], [0, 0, 1]], True)
