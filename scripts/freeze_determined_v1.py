"""Freeze selected runtime dependencies and evidence without deleting history."""
import ast
import hashlib
import json
import platform
import tarfile
from pathlib import Path

import cv2
import numpy as np
import scipy
import sklearn
import torch

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/determined-all-scenes-v1-20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2))


def dependencies():
    modules = {}
    for p in (REPO/'src/liquid_depth').rglob('*.py'):
        parts = list(p.relative_to(REPO/'src').with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        modules['.'.join(parts)] = p
    modules.update({p.stem: p for p in (REPO/'scripts').glob('*.py')})
    pending = ['liquid_depth', 'liquid_depth.determined_scene_meter', 'measure_determined',
               'evaluate_determined_v1', 'evaluate_determined_depth_grid', 'smoke_determined_v1',
               'freeze_determined_v1', 'index_determined_evidence_v1']
    found = set()
    while pending:
        name = pending.pop()
        if name in found or name not in modules:
            continue
        found.add(name)
        p = modules[name]
        parent = name.split('.') if p.name == '__init__.py' else name.split('.')[:-1]
        for node in ast.walk(ast.parse(p.read_text())):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = parent[:len(parent)-node.level+1] if node.level else []
                if node.module:
                    base += node.module.split('.')
                target = '.'.join(base)
                targets = [target]+['.'.join(base+[a.name]) for a in node.names]
            for target in targets:
                if target in modules:
                    pending.append(target)
                    pieces = target.split('.')
                    pending.extend('.'.join(pieces[:i]) for i in range(1, len(pieces)))
    return {n: modules[n] for n in sorted(found)}


def archive(path, entries):
    with tarfile.open(path, 'w:gz') as tar:
        for src, target in entries:
            tar.add(src, arcname=target)
    with tarfile.open(path, 'r:gz') as tar:
        for src, target in entries:
            assert hashlib.sha256(tar.extractfile(target).read()).hexdigest() == sha(src)


def main():
    config = json.loads((REPO/'configs/application_route.json').read_text())
    assert config['selected_cli'] == 'scripts/measure_determined.py'
    assert config['visual_backend_version'] == 'V8'
    assert config['future_development_scope'] == 'far_shallow_only'
    checks = {}
    paths = [BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json']
    paths += [BASE/'liquid-depth-artifacts'/n/'final-delivery-manifest.json' for n in [
        'float-tail-v3-20261015', 'floating-peak-only-v4-20261015', 'floating-apex-v5-20261016',
        'floating-surface-v6-20261016', 'floating-surface-v7-20261016', 'floating-rank-v8-20261016',
        'floating-joint-v9-20261016']]
    for p in paths:
        old = json.loads(p.read_text())
        checked = {name: sha(REPO/name) == h for name, h in old['source_sha256'].items()}
        unexpected = [n for n, ok in checked.items() if not ok and n != 'configs/application_route.json']
        assert not unexpected, unexpected
        checks[str(p)] = checked
    dump('preserved-algorithms.json', {'checks': checks,
         'intentional_metadata_change_only': 'configs/application_route.json',
         'no_old_algorithm_or_weight_replaced': True})
    reference = json.loads((BASE/'liquid-depth-artifacts/floating-joint-v9-20261016/independent-test/protocol.json').read_text())
    weight_checks = {p: sha(Path(p)) == h for p, h in reference['frozen_sha256'].items()
                     if p.endswith(('.pt', '.joblib'))}
    assert all(weight_checks.values()), weight_checks
    dump('preserved-weights.json', weight_checks)
    protocol = json.loads((OUT/'protocol.json').read_text())
    format_audit = {}
    for name, expected in protocol['source_sha256'].items():
        p = Path(name)
        if sha(p) == expected:
            continue
        assert p == REPO/'scripts/measure_determined.py', name
        original = p.read_bytes().replace(
            b'from liquid_depth.determined_scene_meter import SCENES, DeterminedSceneMeter',
            b'from liquid_depth.determined_scene_meter import DeterminedSceneMeter, SCENES')
        assert hashlib.sha256(original).hexdigest() == expected
        format_audit[name] = {'evaluated_sha256': expected, 'packaged_sha256': sha(p),
                              'only_change': 'imported symbol order; CLI outputs rechecked'}
    dump('post-evaluation-format-audit.json', format_audit)
    runtime = dependencies()
    dump('runtime-modules.json', {n: str(p.relative_to(REPO)) for n, p in runtime.items()})
    files = set(runtime.values())
    files.update(REPO/p for p in ['configs/application_route.json', 'configs/determined_scene_routes_v1.json',
        'configs/floating_rank_v8.json', 'tests/test_determined_scene_meter.py',
        'docs/determined_all_scene_baseline_v1_20261016.md', 'docs/far_shallow_history_and_current_v1_20261016.md'])
    files.update(p for pattern in ['requirements*.txt', 'pyproject.toml', 'setup.cfg', 'setup.py'] for p in REPO.glob(pattern))
    history = ['ordinary_range_noise_v36.md', '普通场景_多帧联合估计与空间相关噪声_2026-10-12.md',
        'ordinary_bias_echo_v39.md', 'ordinary_diffuse_denoising_v40.md', 'ordinary_empty_geometry_e2e_v41.md',
        'depth_failure_uniform_recovery_v49_2026-10-13.md', 'ordinary_far_shallow_partial_method_ablation_20261014.md',
        'partial_loss_quantized_disparity_v72_20261014.md', 'transparent_relative_v29.md',
        'transparent_range_v31.md', 'transparent_sr_v33.md', 'uneven_surface_merged_v11_20261015.md',
        'illumination_remaining_failure_repair_v2_20261015.md', 'floating_joint_risk_causal_v9_20261016.md']
    files.update(REPO/'docs'/p for p in history)
    assert all(p.exists() for p in files)
    archive(OUT/'determined-all-scenes-v1-source.tar.gz', [(p, str(p.relative_to(REPO))) for p in sorted(files)])
    profile = json.loads((REPO/'configs/floating_rank_v8.json').read_text())
    weights = {k: Path(v) for k, v in profile.items() if k.endswith('_checkpoint')}
    dump('model-assets.json', {k: {'path': str(p), 'sha256': sha(p)} for k, p in weights.items()})
    archive(OUT/'determined-v8-model-assets.tar.gz', [(p, str(p.relative_to(BASE))) for p in weights.values()])
    dump('environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
        'opencv': cv2.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'sklearn': sklearn.__version__, 'gpu': torch.cuda.get_device_name(0),
        'python_executable': '/root/autodl-tmp/envs/liquid-depth/bin/python'})
    artifact = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    dump('manifest.json', {'version': config['selected_version'],
        'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in sorted(files)},
        'artifact_sha256': artifact, 'future_scope': 'far_shallow_only',
        'other_defects_record_only': True, 'no_measurement_algorithm_changes': True,
        'raw_datasets_retained_at_original_paths': True, 'gui_and_usb_hardware_not_requalified': True})
    print('DETERMINED_BASELINE_FROZEN', len(files), len(runtime), len(artifact), flush=True)


if __name__ == '__main__':
    main()
