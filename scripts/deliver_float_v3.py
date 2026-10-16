"""No-USB smoke replay, verified source overlay, and retained-artifact manifest."""
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
DATA = BASE/'liquid-depth-data/synthetic/float-tail-v3-fresh-test-20261015'
SOURCES = ['src/liquid_depth/float_evidence_v3.py', 'tests/test_float_evidence_v3.py',
    'configs/optional_float_v3.json', 'docs/small_float_false_positive_tail_v3_20261015.md'] + [
    'scripts/'+name+'.py' for name in ['generate_float_tail_v3', 'capture_float_v3', 'train_float_pixels_v3',
    'fit_float_verifier_v3', 'run_float_v3', 'evaluate_float_v3', 'analyze_float_v3', 'measure_float_v3', 'deliver_float_v3',
    'repair_float_eval_summary_v3', 'audit_float_regions_v3']]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert json.loads((OUT/'status.json').read_text())['stage'] == 'complete'
    assert '613 passed' in (OUT/'full-tests.log').read_text()
    frozen = json.loads((OUT/'frozen-test-protocol.json').read_text())
    assert all(sha(Path(p)) == digest for p, digest in frozen['sha256'].items())
    old = BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
    previous = json.loads((old/'final-delivery-manifest.json').read_text())
    assert all(sha(REPO/p) == digest for p, digest in previous['source_sha256'].items())
    geometric_manifest = json.loads((BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json').read_text())
    assert all(sha(REPO/p) == digest for p, digest in geometric_manifest['source_sha256'].items())
    record = install(633, DATA, DATA/'dry', 202610155927, max_annotation_attempts=6, source_slot_stride=1)
    demo = OUT/'no-usb-example'; demo.mkdir(exist_ok=False)
    (demo/'calibration.json').write_text(json.dumps(record['bundle'], indent=2))
    frame = json.loads((DATA/'00633/frames.json').read_text())[0]
    for mode in ['off', 'auto']:
        command = [str(BASE/'envs/liquid-depth/bin/python'), 'scripts/measure_float_v3.py',
            '--bundle', str(demo/'calibration.json'), '--dry', str(DATA/'dry/0633/dry_depth_m.npy'),
            '--depth', frame['raw_path'], '--rgb', str(DATA/'00633/rgb.png'), '--visual', mode,
            '--profile', str(REPO/'configs/optional_float_v3.json')]
        text = subprocess.check_output(command, cwd=REPO,
            env={**os.environ, 'PYTHONPATH': str(REPO/'src')+':'+str(REPO/'scripts')}, text=True)
        (demo/f'{mode}.json').write_text(text)
    a, b = [json.loads((demo/f'{mode}.json').read_text()) for mode in ['off', 'auto']]
    assert a['liquid_depth_m'] == b['liquid_depth_m']
    archive = OUT/'source-float-tail-v3-final-20261015.tar.gz'
    with tarfile.open(archive, 'x:gz') as target:
        for source in SOURCES: target.add(REPO/source, arcname=source, recursive=False)
    artifacts = ['pixel-model/best.pt', 'pixel-model/training.json', 'pixel-model/real-priors-used.json',
        'region-model/verifier.joblib', 'region-model/validation-policy.json', 'frozen-test-protocol.json',
        'full-tests.log', 'fresh-evaluation/records.jsonl', 'fresh-evaluation/liquid-summary.json',
        'fresh-evaluation/highest-summary.json', 'fresh-evaluation/diagnostics-summary.json',
        'fresh-evaluation/paired-bootstrap.json', 'fresh-evaluation/tail-cases.json',
        'fresh-evaluation/loss-scope.json',
        'fresh-evaluation/accepted-object-comparison.json',
        'frozen-test-protocol-before-label-repair.json', 'frozen-test-protocol-before-region-report-repair.json',
        'no-usb-example/calibration.json', 'no-usb-example/off.json', 'no-usb-example/auto.json']
    manifest = {'source_sha256': {p: sha(REPO/p) for p in SOURCES},
        'artifact_sha256': {p: sha(OUT/p) for p in artifacts}, 'source_archive_sha256': sha(archive),
        'default_visual_mode': 'off', 'frozen_previous_sources_unchanged': True,
        'test_installations': 198, 'test_frames': 396, 'all_frames_retained': True,
        'restore_order': ['determined-symmetric-geometry-v1-20261015/source.tar into parent directory',
            'adaptive-route-v2-20261015/source-optional-rgbd-v2-final-20261015.tar.gz into project directory',
            archive.name+' into project directory'], 'research_only': True}
    with (OUT/'final-delivery-manifest.json').open('x') as handle: json.dump(manifest, handle, indent=2)
    print('DELIVERY_VERIFIED', len(SOURCES), 'source files;', len(artifacts), 'artifacts')


if __name__ == '__main__': main()
