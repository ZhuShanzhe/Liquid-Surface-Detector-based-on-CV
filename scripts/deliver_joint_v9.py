"""Pin the optional V9 experiment without changing any previously frozen route."""
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
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {}
    roots = [('liquid-depth-archives/determined-symmetric-geometry-v1-20261015', 'manifest.json')]+[
        (f'liquid-depth-artifacts/{r}', 'final-delivery-manifest.json') for r in [
            'float-tail-v3-20261015', 'floating-peak-only-v4-20261015', 'floating-apex-v5-20261016',
            'floating-surface-v6-20261016', 'floating-surface-v7-20261016', 'floating-rank-v8-20261016']]
    for root, name in roots:
        manifest = json.loads((BASE/root/name).read_text())
        checks = {p: sha(REPO/p) == h for p, h in manifest['source_sha256'].items()}
        assert all(checks.values()), checks
        preserved[root] = checks
    protocol = json.loads((OUT/'independent-test/protocol.json').read_text())
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    (OUT/'preserved-routes.json').write_text(json.dumps(preserved, indent=2))
    (OUT/'environment.json').write_text(json.dumps({'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'torch': torch.__version__, 'opencv': cv2.__version__, 'sklearn': sklearn.__version__,
        'gpu': torch.cuda.get_device_name(0)}, indent=2))
    sources = ['src/liquid_depth/floating_joint_v9.py', 'src/liquid_depth/floating_joint_application_v9.py',
        'scripts/generate_joint_v9.py', 'scripts/capture_joint_v9.py', 'scripts/fit_joint_v9.py', 'scripts/evaluate_joint_v9.py',
        'scripts/analyze_joint_v9.py', 'scripts/probe_persistent_joint_v9.py', 'scripts/measure_joint_v9.py',
        'scripts/smoke_joint_v9.py', 'scripts/deliver_joint_v9.py', 'tests/test_floating_joint_v9.py',
        'configs/floating_joint_v9.json', 'docs/floating_joint_risk_causal_v9_20261016.md']
    catalogs = {}
    for split in ['validation', 'test']:
        root = BASE/f'liquid-depth-data/synthetic/floating-joint-v9-{split}-20261016'
        catalogs[split] = {'root': str(root), 'files_sha256': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    (OUT/'data-catalog.json').write_text(json.dumps(catalogs, indent=2))
    archive = OUT/'source-floating-joint-v9-20261016.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sources:
            tar.add(REPO/name, arcname=name)
    with tarfile.open(archive, 'r:gz') as tar:
        for name in sources:
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == sha(REPO/name)
    manifest = {'source_sha256': {p: sha(REPO/p) for p in sources}, 'archive_sha256': sha(archive),
        'artifact_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'final-delivery-manifest.json'},
        'old_application_default_unchanged': True, 'all_test_frames_retained': True,
        'visual_output_scope': 'floating_objects_highest_height_only', 'liquid_chain_unchanged': True}
    (OUT/'final-delivery-manifest.json').write_text(json.dumps(manifest, indent=2))
    print('V9_DELIVERY_VERIFIED', len(sources), len(manifest['artifact_sha256']), flush=True)


if __name__ == '__main__':
    main()
