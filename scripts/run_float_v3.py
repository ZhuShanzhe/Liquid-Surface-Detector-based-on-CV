"""Durable experiment orchestration, with a frozen protocol before fresh test access."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'
DATA = BASE/'liquid-depth-data/synthetic'
PY = BASE/'envs/liquid-depth/bin/python'


def status(stage, **values):
    (OUT/'status.json').write_text(json.dumps({'stage': stage, 'time_unix': time.time(), **values}, indent=2))


def run(script, arguments, logfile):
    env = {**os.environ, 'PYTHONPATH': str(REPO/'src')+':'+str(REPO/'scripts'),
           'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}
    with (OUT/logfile).open('w') as handle:
        subprocess.run([str(PY), 'scripts/'+script, *arguments], cwd=REPO, env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    root = DATA/'float-tail-v3-trainval-20261015'
    status('waiting_for_training_capture')
    while len(list(root.glob('*/supervision_offline_01.npz'))) < 594: time.sleep(10)
    status('small_object_and_hard_negative_finetuning')
    model = OUT/'pixel-model/best.pt'
    run('train_float_pixels_v3.py', ['--root', str(root), '--output', str(model.parent), '--initialize',
        str(BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015/pixel-model-with-real/best.pt')], 'pixel-training.log')
    status('training_region_verifier_and_validation_policy')
    run('fit_float_verifier_v3.py', ['--root', str(root), '--model', str(model), '--output', str(OUT/'region-model')], 'verifier-training.log')
    files = [REPO/'src/liquid_depth/float_evidence_v3.py', REPO/'scripts/evaluate_float_v3.py',
        REPO/'scripts/generate_float_tail_v3.py', model, OUT/'region-model/verifier.joblib']
    (OUT/'frozen-test-protocol.json').write_text(json.dumps({'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'test_seed': 202610155927, 'fresh_installations': 198, 'frames': 396,
        'test_used_for_training_or_thresholds': False, 'default_visual_mode': 'off'}, indent=2))
    status('waiting_for_fresh_capture')
    while json.loads((OUT/'capture-status.json').read_text()).get('stage') != 'complete': time.sleep(10)
    status('fresh_paired_evaluation')
    run('evaluate_float_v3.py', [], 'fresh-evaluation.log')
    status('complete')


if __name__ == '__main__':
    try: main()
    except Exception as exc: status('failed', error=repr(exc)); raise
