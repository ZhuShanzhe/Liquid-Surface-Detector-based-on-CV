"""Verify boundary, replay CLI modes, and package the scope-only application overlay."""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-peak-only-v4-20261015'
OLD = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
DATA = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'
SOURCES = ['src/liquid_depth/floating_peak_only_v4.py', 'tests/test_floating_peak_only_v4.py',
    'configs/floating_peak_only_v4.json', 'configs/application_route.json',
    'scripts/measure_floating_peak_v4.py', 'scripts/evaluate_floating_scope_v4.py',
    'scripts/deliver_floating_scope_v4.py', 'docs/floating_peak_only_application_v4_20261015.md']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads((OUT/'scope-audit.json').read_text())
    assert all(value == 0 for value in audit['changes'].values())
    assert audit['model_calls_by_scene'] == {'floating_objects': 66}
    tests = BASE/'floating-peak-only-v4-tests.log'
    assert '627 passed' in tests.read_text(); shutil.copyfile(tests, OUT/'full-tests.log')
    previous = json.loads((OLD/'final-delivery-manifest.json').read_text())
    assert all(sha(REPO/name) == digest for name, digest in previous['source_sha256'].items())
    geometric = json.loads((BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json').read_text())
    assert all(sha(REPO/name) == digest for name, digest in geometric['source_sha256'].items())
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert all(sha(Path(name)) == digest for name, digest in protocol['sha256'].items())
    item = install(633, DATA, DATA/'dry', 202610155927, max_annotation_attempts=6, source_slot_stride=1)
    demo = OUT/'no-usb-example'; demo.mkdir(exist_ok=False)
    (demo/'calibration.json').write_text(json.dumps(item['bundle'], indent=2))
    frame = json.loads((DATA/'00633/frames.json').read_text())[0]
    results = {}
    for tag, scene, enabled in [('floating_off', 'floating_objects', False),
                                ('floating_on', 'floating_objects', True), ('glare_request_blocked', 'glare', True)]:
        cmd = [str(BASE/'envs/liquid-depth/bin/python'), 'scripts/measure_floating_peak_v4.py',
            '--scene', scene, '--bundle', str(demo/'calibration.json'),
            '--dry', str(DATA/'dry/0633/dry_depth_m.npy'), '--depth', frame['raw_path']]
        if enabled: cmd.append('--enable-floating-peak-vision')
        if scene == 'floating_objects' and enabled: cmd += ['--rgb', str(DATA/'00633/rgb.png')]
        if scene == 'glare': cmd += ['--profile', '/nonexistent-profile-to-prove-no-visual-load.json']
        output = subprocess.check_output(cmd, cwd=REPO, text=True, env={**os.environ,
            'PYTHONPATH': str(REPO/'src')+':'+str(REPO/'scripts'), 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'})
        results[tag] = json.loads(output); (demo/f'{tag}.json').write_text(output)
    assert len({r['liquid_depth_m'] for r in results.values()}) == 1
    assert results['floating_on']['visual_module_executed']
    assert not results['floating_off']['visual_module_executed']
    assert not results['glare_request_blocked']['visual_module_executed']
    archive = OUT/'source-floating-peak-only-v4.tar.gz'
    with tarfile.open(archive, 'x:gz') as handle:
        for name in SOURCES: handle.add(REPO/name, arcname=name, recursive=False)
    artifacts = ['summary.json', 'records.jsonl', 'scope-audit.json', 'depth-strata.json', 'protocol.json',
        'calibration-audit.json', 'full-tests.log', 'no-usb-example/calibration.json',
        'no-usb-example/floating_on.json', 'no-usb-example/floating_off.json', 'no-usb-example/glare_request_blocked.json']
    manifest = {'source_sha256': {name: sha(REPO/name) for name in SOURCES},
        'artifact_sha256': {name: sha(OUT/name) for name in artifacts}, 'source_archive_sha256': sha(archive),
        'parent_snapshot': str(OLD/'final-delivery-manifest.json'),
        'old_source_unchanged': True, 'new_training': False, 'default_visual_enabled': False,
        'vision_allowed_scene': 'floating_objects', 'vision_allowed_output': 'highest_height_m',
        'report_cohorts': {'actual_visible_floats': 60, 'empty_controls': 6, 'blocked_other_scenes': 330}}
    with (OUT/'final-delivery-manifest.json').open('x') as handle: json.dump(manifest, handle, indent=2)
    print('SCOPED_DELIVERY_VERIFIED', len(SOURCES), 'source files', len(artifacts), 'artifacts', flush=True)


if __name__ == '__main__': main()
