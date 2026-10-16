"""Independent 528-frame final test, never used for training or policy selection."""
import json
import os
import subprocess
from pathlib import Path

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-rank-v8-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/floating-rank-v8-test-20261016'
SEED = 202610163811


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not ROOT.exists(), 'new_test_root_required'
    jobs = []
    for worker in range(4):
        handle = (OUT/f'capture-worker-{worker}.log').open('w')
        jobs.append(subprocess.Popen([str(BASE/'tools/blender/blender-4.2.0-linux-x64/blender'), '-b', '-t', '3',
            '--python', 'scripts/generate_floating_v5.py', '--', '--root', str(ROOT), '--seed', str(SEED),
            '--installations', '264', '--worker', str(worker), '--workers', '4'], cwd=REPO,
            env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '3'}, stdout=handle, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in jobs]
    assert not any(codes), codes
    assert len(list(ROOT.glob('*/supervision_offline_01.npz'))) == 264
    (OUT/'capture-complete.json').write_text(json.dumps({'seed': SEED, 'installations': 264, 'frames': 528}))
    print('V8_CAPTURE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
