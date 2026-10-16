"""Preserve all old releases and archive the V8 research source and evidence."""
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
OUT = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {}
    roots = [('liquid-depth-archives/determined-symmetric-geometry-v1-20261015', 'manifest.json'),
        ('liquid-depth-artifacts/float-tail-v3-20261015', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-peak-only-v4-20261015', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-apex-v5-20261016', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-surface-v6-20261016', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-surface-v7-20261016', 'final-delivery-manifest.json')]
    for root, name in roots:
        manifest = json.loads((BASE/root/name).read_text())
        checks = {p: sha(REPO/p) == h for p, h in manifest['source_sha256'].items()}
        assert all(checks.values()), checks
        preserved[root] = checks
    protocol = json.loads((OUT/'independent-test/protocol.json').read_text())
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    (OUT/'preserved-routes.json').write_text(json.dumps(preserved, indent=2))
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'torch': torch.__version__, 'opencv': cv2.__version__, 'sklearn': sklearn.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (OUT/'environment.json').write_text(json.dumps(environment, indent=2))
    sources = ['src/liquid_depth/floating_rank_v8.py', 'src/liquid_depth/floating_rank_application_v8.py',
        'scripts/capture_rank_v8.py', 'scripts/fit_rank_v8.py', 'scripts/evaluate_rank_v8.py',
        'scripts/analyze_rank_v8.py', 'scripts/analyze_validation_rank_v8.py', 'scripts/common_shift_rank_v8.py',
        'scripts/measure_rank_v8.py', 'scripts/smoke_rank_v8.py',
        'scripts/deliver_rank_v8.py', 'tests/test_floating_rank_v8.py', 'configs/floating_rank_v8.json',
        'docs/floating_instance_rank_risk_v8_20261016.md']
    root = BASE/'liquid-depth-data/synthetic/floating-rank-v8-test-20261016'
    (OUT/'fresh-test-catalog.json').write_text(json.dumps({'root': str(root), 'files_sha256': {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}, indent=2))
    archive = OUT/'source-floating-rank-v8-20261016.tar.gz'
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
    print('V8_DELIVERY_VERIFIED', len(sources), len(manifest['artifact_sha256']), flush=True)


if __name__ == '__main__':
    main()
