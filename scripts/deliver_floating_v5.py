"""Verify preserved routes, pin inputs/models/results, and archive the new source overlay."""
import hashlib
import json
import platform
import tarfile
from pathlib import Path

import cv2
import numpy as np
import sklearn
import torch

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    preservation = {}
    for directory, name in [('liquid-depth-archives/determined-symmetric-geometry-v1-20261015', 'manifest.json'),
        ('liquid-depth-artifacts/float-tail-v3-20261015', 'final-delivery-manifest.json'),
        ('liquid-depth-artifacts/floating-peak-only-v4-20261015', 'final-delivery-manifest.json')]:
        manifest = json.loads((BASE/directory/name).read_text())
        sources = manifest['source_sha256']
        checks = {p: sha(REPO/p) == h for p, h in sources.items()}
        assert all(checks.values()), checks
        preservation[directory] = checks
    protocol = json.loads((OUT/'independent-test/protocol.json').read_text())
    assert all(sha(Path(p)) == h for p, h in protocol['frozen_sha256'].items())
    source_names = ['src/liquid_depth/floating_apex_v5.py', 'src/liquid_depth/floating_apex_application_v5.py',
        'scripts/generate_floating_v5.py', 'scripts/capture_floating_v5.py', 'scripts/fit_floating_v5.py',
        'scripts/evaluate_floating_v5.py', 'scripts/analyze_floating_v5.py', 'scripts/measure_floating_apex_v5.py',
        'scripts/smoke_floating_v5.py', 'scripts/deliver_floating_v5.py', 'tests/test_floating_apex_v5.py',
        'configs/floating_apex_v5.json', 'docs/floating_apex_peak_repair_v5_20261016.md']
    (OUT/'preserved-route-audit.json').write_text(json.dumps(preservation, indent=2))
    environment = {'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'numpy': np.__version__, 'sklearn': sklearn.__version__, 'device': torch.cuda.get_device_name(0)}
    (OUT/'environment.json').write_text(json.dumps(environment, indent=2))
    catalog = {}
    for split in ['train', 'validation', 'test']:
        root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{split}-20261016'
        catalog[split] = {'root': str(root), 'files_sha256': {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}}
    (OUT/'dataset-catalog.json').write_text(json.dumps(catalog, indent=2))
    archive = OUT/'source-floating-apex-v5-20261016.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in source_names: tar.add(REPO/name, arcname=name)
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(source_names)
        for name in source_names: assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == sha(REPO/name)
    manifest = {'source_sha256': {p: sha(REPO/p) for p in source_names}, 'archive_sha256': sha(archive),
        'artifact_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*'))
            if p.is_file() and p.name not in {'final-delivery-manifest.json', 'delivery.log'}},
        'application_default_kept_v4': True, 'new_optional_entry': 'scripts/measure_floating_apex_v5.py',
        'frozen_backups_unchanged': True, 'liquid_route_unchanged': True,
        'visual_scope': 'floating_objects_highest_height_only', 'remote_only_development': True}
    (OUT/'final-delivery-manifest.json').write_text(json.dumps(manifest, indent=2))
    print('DELIVERY_VERIFIED', len(source_names), len(manifest['artifact_sha256']), environment, flush=True)


if __name__ == '__main__': main()
