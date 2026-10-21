"""Offline resource closure and observation/truth boundary acceptance checks.

These are source/bundle-contract tests, not a replacement for frozen-executable
end-to-end tests. Generated inputs and display truth are allowed in the producer;
they must not become hidden fields on the public measurement observation.
"""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RESOURCES = (
    'render_scene_console.py', 'generate_synthetic_liquid.py',
    'generate_fullframe_v16.py', 'generate_special_scene_strata_v1.py',
    'generate_other_scenes_v34.py', 'special_scene_protocol_v1.py',
    'measure_release.py', 'measure_determined.py', 'tof_phase_simulator.py',
)
RUNTIME_IMPORTS = ('numpy', 'cv2', 'torch', 'torchvision', 'scipy', 'sklearn',
                   'joblib', 'threadpoolctl', 'PySide6')


def parsed(relative):
    return ast.parse((ROOT/relative).read_text(encoding='utf-8'))


def test_producer_and_ui_resource_closure_is_present():
    for name in SCRIPT_RESOURCES:
        path = ROOT/'scripts'/name
        assert path.is_file() and path.stat().st_size > 0, name
        # Blender scripts use normal top-level imports from the adjacent scripts
        # directory. Discover any additional same-directory local imports too.
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom) and node.module:
                module = node.module.split('.')[0]
                local = ROOT/'scripts'/(module+'.py')
                if local.exists():
                    assert local.name in SCRIPT_RESOURCES, 'new producer dependency: '+local.name
    for name in ('simulation.py', 'native_forest.cpp',
                 'ui/scene_console.html', 'ui/scene_console.css', 'ui/scene_console.js'):
        assert (ROOT/'src/liquid_depth'/name).is_file(), name
    assert (ROOT/'configs/release_assets.json').is_file()


def test_every_frozen_model_has_a_matching_checksum():
    checked = set()
    legacy = json.loads((ROOT/'configs/release_assets.json').read_text())
    assert legacy['sklearn_version'] == '1.9.0'
    for item in legacy['models'].values():
        path = (ROOT/item['path']).resolve()
        assert path.is_relative_to((ROOT/'assets/models').resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        checked.add(path)
    router = ROOT/'assets/models/scene_router'
    card = json.loads((router/'model_card.json').read_text())
    path = router/'scene_head.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == card['sha256']
    checked.add(path.resolve())
    for name in ('new_view_v14', 'small_float_peak'):
        directory = ROOT/'assets/models'/name
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['sklearn_version'] == '1.9.0'
        for filename, digest in manifest['sha256'].items():
            assert Path(filename).name == filename
            path = directory/filename
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            checked.add(path.resolve())
    weights = {p.resolve() for p in (ROOT/'assets/models').rglob('*')
               if p.suffix.lower() in ('.pt', '.pth', '.joblib')}
    assert weights and weights <= checked, 'unverified release weight: '+str(weights-checked)


def test_desktop_runtime_dependencies_and_sklearn_version():
    for name in RUNTIME_IMPORTS:
        assert importlib.util.find_spec(name) is not None, name
    import sklearn
    assert sklearn.__version__ == '1.9.0'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows bundled native forest')
def test_desktop_native_dll_manifest_present_and_valid():
    from liquid_depth.compiled_native_forest import _windows_library
    library = _windows_library(ROOT/'src/liquid_depth/native_forest.cpp')
    assert library.stat().st_size > 0


def test_launcher_cli_cannot_inject_ground_truth_or_semantic_masks():
    tree = parsed('scripts/desktop_launcher.py')
    flags = {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
             for arg in node.args if isinstance(arg, ast.Constant)
             and isinstance(arg.value, str) and arg.value.startswith('--')}
    assert flags == {'--self-test', '--quick', '--output', '--work', '--blender',
                     '--no-auto-initialize', '--gui-self-test', '--gui-self-test-timeout'}
    assert not any(word in flag for flag in flags for word in ('truth', 'actual', 'semantic', 'true-height'))


def test_public_observation_contains_only_observable_measurement_inputs():
    tree = parsed('src/liquid_depth/unified_tof.py')
    contract = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ToFObservation')
    fields = {n.target.id for n in contract.body if isinstance(n, ast.AnnAssign)}
    assert fields == {'frame_id', 'rgb_bgr', 'depth_m', 'timestamp_s', 'correlations',
                      'sensor_family', 'installation_id', 'capture_validity'}
    tree = parsed('src/liquid_depth/scene_console_web.py')
    constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id == 'ToFObservation']
    assert len(constructors) == 1
    names = {n.id for n in ast.walk(constructors[0]) if isinstance(n, ast.Name)}
    assert not names & {'truth', 'semantic', 'physical', 'controls', 'actual'}
    console = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Console')
    run = next(n for n in console.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    measure = next(n for n in ast.walk(run) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'measure_if_eligible')
    truth = next(n for n in ast.walk(run) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'truth' for t in n.targets))
    assert truth.lineno > measure.lineno, 'current display/scoring truth loaded before measurement'


def test_desktop_routes_through_the_formal_factory():
    tree = parsed('src/liquid_depth/scene_console_web.py')
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert 'create_formal_meter' in names
    assert 'UnifiedToFMeter' not in names  # Never instantiate the unfrozen base facade instead.


def test_offline_license_snapshot_and_collection_are_present():
    directory = ROOT/'docs/third_party_licenses'
    records = json.loads((directory/'qt-chromium-credits.sources.json').read_text())
    content = (directory/'qt-chromium-credits.html').read_text(encoding='utf-8')
    assert records['pages'] == len(records['documents']) >= 100
    assert len({row['url'] for row in records['documents']}) == records['pages']
    assert all(len(row['sha256']) == 64 and row['url'] in content
               for row in records['documents'])
    assert '<script' not in content.lower()
    for name in ('GPL-2.0.txt', 'GPL-3.0.txt', 'LGPL-2.1.txt', 'LGPL-3.0.txt',
                 'GFDL-1.3.txt'):
        assert (directory/name).stat().st_size > 5000
    assert 'license_data(repo)' in (ROOT/'packaging/liquid_simulation.spec').read_text()
    notice = (ROOT/'docs/THIRD_PARTY_NOTICES.rst').read_text(encoding='utf-8')
    assert 'SBOM' in notice and '对应源码' in notice
