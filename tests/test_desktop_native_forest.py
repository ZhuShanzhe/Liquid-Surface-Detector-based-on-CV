"""Cross-platform numerical parity checks for the unchanged tree evaluator."""
import ctypes
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = (sys.platform == 'win32' or
             sys.platform == 'linux' and shutil.which('g++') is not None)
pytestmark = pytest.mark.skipif(not AVAILABLE, reason='native forest build platform required')


def test_native_tree_threshold_boundaries():
    """FP32 casts adjacent to an FP64 threshold must not alter branch decisions."""
    from liquid_depth.compiled_native_forest import build
    function = build()
    threshold = np.float64(.1)
    center = np.float32(threshold)
    x = np.array([[np.nextafter(center, -np.inf, dtype=np.float32)], [center],
                  [np.nextafter(center, np.inf, dtype=np.float32)]], np.float32)
    features = np.array([[0, 0, 0], [0, 0, 0]], np.int32)
    thresholds = np.array([[threshold, 0, 0], [threshold, 0, 0]], np.float64)
    left = np.array([[1, -1, -1], [1, -1, -1]], np.int32)
    right = np.array([[2, -1, -1], [2, -1, -1]], np.int32)
    leaf = np.array([[False, True, True], [False, True, True]], bool)
    value = np.array([[0, .125, .875], [0, .375, .625]], np.float64)
    result = np.empty(3, np.float64)
    function(3, 1, 2, 3, *[ctypes.c_void_p(a.ctypes.data) for a in
             (x, features, thresholds, left, right, leaf, value, result)])
    np.testing.assert_array_equal(result, np.where(x[:, 0].astype(np.float64) <= threshold, .25, .75))


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows packaged DLL integrity')
def test_windows_library_matches_source_and_manifest():
    from liquid_depth.compiled_native_forest import _windows_library
    source = ROOT/'src/liquid_depth/native_forest.cpp'
    library = _windows_library(source)
    manifest = json.loads(library.with_suffix('.json').read_text(encoding='utf-8'))
    assert manifest['library_sha256'] == hashlib.sha256(library.read_bytes()).hexdigest()
    assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows packaged DLL integrity')
def test_windows_missing_or_corrupt_assets_never_silently_fallback(tmp_path):
    from liquid_depth.compiled_native_forest import _windows_library
    source = tmp_path/'native_forest.cpp'
    shutil.copy2(ROOT/'src/liquid_depth/native_forest.cpp', source)
    with pytest.raises(RuntimeError, match='bundled_windows_forest_missing'):
        _windows_library(source)
    native = tmp_path/'native'
    native.mkdir()
    for name in ('forest-win_amd64.dll', 'forest-win_amd64.json'):
        shutil.copy2(ROOT/'src/liquid_depth/native'/name, native/name)
    record = json.loads((native/'forest-win_amd64.json').read_text(encoding='utf-8'))
    record['library_sha256'] = '0'*64
    (native/'forest-win_amd64.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(RuntimeError, match='bundled_windows_forest_integrity_mismatch'):
        _windows_library(source)


def test_native_matches_sklearn_estimators():
    pytest.importorskip('sklearn')
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from liquid_depth.compiled_native_forest import CompiledForest
    rng = np.random.default_rng(20261021)
    x = rng.normal(size=(512, 17)).astype(np.float32)
    classifiers = ExtraTreesClassifier(n_estimators=31, random_state=21).fit(x, x[:, 0] > 0)
    regressors = ExtraTreesRegressor(n_estimators=31, random_state=21).fit(x, x[:, 1]+x[:, 3])
    test = rng.normal(size=(256, 17)).astype(np.float32)
    np.testing.assert_allclose(CompiledForest(classifiers).predict_proba(test),
                               classifiers.predict_proba(test), atol=1e-14, rtol=0)
    np.testing.assert_allclose(CompiledForest(regressors).predict(test),
                               regressors.predict(test), atol=1e-14, rtol=0)


def test_native_matches_every_trusted_release_forest():
    sklearn = pytest.importorskip('sklearn')
    joblib = pytest.importorskip('joblib')
    from liquid_depth.compiled_native_forest import CompiledForest
    directories = [ROOT/'assets/models/new_view_v14', ROOT/'assets/models/small_float_peak']
    rng = np.random.default_rng(2026102101)
    compared = 0
    for directory in directories:
        manifest = json.loads((directory/'manifest.json').read_text())
        assert sklearn.__version__ == manifest['sklearn_version']
        for name, digest in manifest['sha256'].items():
            if not name.endswith('.joblib'):
                continue
            path = directory/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            state = joblib.load(path)
            models = state['models'] if 'models' in state else state.values()
            for model in models:
                if not hasattr(model, 'estimators_'):
                    continue
                x = rng.normal(size=(257, model.n_features_in_)).astype(np.float32)
                x[0] = 0
                compiled = CompiledForest(model)
                method = 'predict_proba' if hasattr(model, 'classes_') else 'predict'
                np.testing.assert_allclose(getattr(compiled, method)(x), getattr(model, method)(x),
                                           atol=1e-14, rtol=0)
                compared += 1
    assert compared >= 6
