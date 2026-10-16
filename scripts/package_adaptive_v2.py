"""Package the optional route overlay without changing the determined baseline."""
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp')
REPO = ROOT / 'Liquid-Surface-Detector-based-on-CV'
OUT = ROOT / 'liquid-depth-artifacts/adaptive-route-v2-20261015'
SOURCE_FILES = [
    'src/liquid_depth/frozen_symmetric_geometry.py',
    'src/liquid_depth/rgbd_evidence_model.py',
    'src/liquid_depth/adaptive_rgbd_meter.py',
    'src/liquid_depth/hybrid_scene_meter_v2.py',
    'tests/test_adaptive_rgbd_v2.py',
    'configs/development_route_adaptive_v2.json',
    'docs/adaptive_rgbd_optional_evidence_v2_20261015.md',
] + [f'scripts/{name}.py' for name in [
    'freeze_symmetric_geometry_v1', 'acquire_water_priors_v2',
    'generate_adaptive_evidence_v2', 'train_rgbd_evidence_v2',
    'train_candidate_trust_v2', 'run_adaptive_pipeline_v2',
    'run_prior_refinement_v2', 'evaluate_adaptive_route_v2',
    'confirm_adaptive_v2', 'summarize_adaptive_v2', 'audit_adaptive_scope_v2',
    'measure_adaptive_v2', 'make_adaptive_demo_v2', 'package_adaptive_v2',
]]


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    audit = json.loads((OUT / 'final-audit-summary.json').read_text())
    assert all(audit['frozen_geometric_integrity'].values())
    assert all(audit['confirmation_source_integrity'].values())
    log = ROOT / 'adaptive-v2-final-tests.log'
    assert '600 passed' in log.read_text()
    shutil.copyfile(log, OUT / 'final-tests.log')
    archive = OUT / 'source-optional-rgbd-v2-final-20261015.tar.gz'
    # A source overlay: restore the immutable baseline first, then this archive.
    with tarfile.open(archive, 'x:gz') as handle:
        for relative in SOURCE_FILES:
            handle.add(REPO / relative, arcname=relative, recursive=False)
    retained = [
        'pixel-model/best.pt', 'candidate-model/gate.pt',
        'pixel-model-with-real/best.pt', 'candidate-model-with-real/gate.pt',
        'deployment-choice.json', 'confirmation-frozen-protocol.json',
        'final-audit-summary.json', 'final-tests.log',
        'pixel-model-with-real/real-priors-used.json',
    ]
    for name in ['confirmation-synthetic', 'confirmation-with-real']:
        retained += [f'{name}/{file}' for file in [
            'records.jsonl', 'summary.json', 'highest-summary.json',
            'pixel-and-reliability.json', 'protocol.json',
            'calibration-audit.json', 'partial-loss-scope.json',
        ]]
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'baseline_manifest': str(ROOT / 'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json'),
        'restore_order': ['baseline source.tar', archive.name],
        'source_archive_sha256': digest(archive),
        'source_sha256': {name: digest(REPO / name) for name in SOURCE_FILES},
        'retained_artifact_sha256': {name: digest(OUT / name) for name in retained},
        'all_tests': '600 passed, 14 warnings',
        'default_visual_mode': 'off',
        'all_confirmation_frames_retained': True,
        'liquid_auto_override_count': 0,
        'research_only_not_field_qualification': True,
    }
    target = OUT / 'final-delivery-manifest.json'
    with target.open('x') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({'source_archive': str(archive), 'sha256': manifest['source_archive_sha256'],
                      'source_files': len(SOURCE_FILES), 'report': str(REPO / SOURCE_FILES[6]),
                      'manifest': str(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
