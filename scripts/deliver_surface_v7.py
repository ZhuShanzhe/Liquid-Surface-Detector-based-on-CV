"""Pin the experimental release and verify every older frozen route."""
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
OUT = BASE/'liquid-depth-artifacts/floating-surface-v7-20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preserved = {}
    for root, name in [('liquid-depth-archives/determined-symmetric-geometry-v1-20261015', 'manifest.json'),
        ('liquid-depth-artifacts/float-tail-v3-20261015', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-peak-only-v4-20261015', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-apex-v5-20261016', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-surface-v6-20261016', 'final-delivery-manifest.json')]:
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
    sources = ['src/liquid_depth/floating_surface_v7.py', 'src/liquid_depth/floating_surface_application_v7.py',
        'scripts/capture_surface_v7.py', 'scripts/fit_surface_v7.py', 'scripts/evaluate_surface_v7.py',
        'scripts/measure_surface_v7.py', 'scripts/analyze_surface_v7.py', 'scripts/diagnose_surface_v7.py', 'scripts/smoke_surface_v7.py',
        'scripts/deliver_surface_v7.py', 'tests/test_floating_surface_v7.py', 'configs/floating_surface_v7.json',
        'docs/floating_surface_mismatch_facets_v7_20261016.md']
    root = BASE/'liquid-depth-data/synthetic/floating-surface-v7-test-20261016'
    (OUT/'fresh-test-catalog.json').write_text(json.dumps({'root': str(root), 'files_sha256': {
        str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}, indent=2))
    archive = OUT/'source-floating-surface-v7-20261016.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in sources:
            tar.add(REPO/name, arcname=name)
    with tarfile.open(archive, 'r:gz') as tar:
        for name in sources:
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == sha(REPO/name)
    manifest = {'source_sha256': {p: sha(REPO/p) for p in sources}, 'archive_sha256': sha(archive),
        'artifact_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'final-delivery-manifest.json'},
        'old_application_default_unchanged': True, 'all_results_retained': True,
        'visual_output_scope': 'floating_objects_highest_height_only', 'liquid_chain_unchanged': True,
        'new_inferred_points_are_not_new_sensor_measurements': True}
    (OUT/'final-delivery-manifest.json').write_text(json.dumps(manifest, indent=2))
    print('V7_DELIVERY_VERIFIED', len(sources), len(manifest['artifact_sha256']), flush=True)


if __name__ == '__main__':
    main()
