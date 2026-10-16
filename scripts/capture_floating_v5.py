"""Three independent seeds, fixed splits; no final-test access during optimization."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-apex-v5-20261016'
SPECS = [('train', 202610160317, 198), ('validation', 202610160419, 99), ('test', 202610160821, 198)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, seed, count in SPECS:
        root = BASE/f'liquid-depth-data/synthetic/floating-apex-v5-{name}-20261016'
        jobs = []; env = {**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '3'}
        for worker in range(4):
            log = (OUT/f'capture-{name}-{worker}.log').open('w')
            cmd = [str(BASE/'tools/blender/blender-4.2.0-linux-x64/blender'), '-b', '-t', '3',
                '--python', 'scripts/generate_floating_v5.py', '--', '--root', str(root), '--seed', str(seed),
                '--installations', str(count), '--worker', str(worker), '--workers', '4']
            jobs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in jobs):
            (OUT/'capture-status.json').write_text(json.dumps({'stage': name,
                'complete': len(list(root.glob('*/supervision_offline_01.npz'))), 'expected': count})); time.sleep(10)
        if any(job.returncode for job in jobs): raise RuntimeError('capture_failed:'+name)
        (OUT/f'{name}-capture-complete.json').write_text(json.dumps({'seed': seed, 'installations': count, 'frames': 2*count}))
    (OUT/'capture-status.json').write_text(json.dumps({'stage': 'complete'}))


if __name__ == '__main__': main()
