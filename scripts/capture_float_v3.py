"""Independent training/validation and final test capture without threshold tuning."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path('/root/autodl-tmp')
REPO = BASE / 'Liquid-Surface-Detector-based-on-CV'
OUT = BASE / 'liquid-depth-artifacts/float-tail-v3-20261015'
DATA = BASE / 'liquid-depth-data/synthetic'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    specs = [('trainval', 202610155361, 0, 594), ('fresh-test', 202610155927, 594, 792)]
    for tag, seed, start, count in specs:
        root = DATA / f'float-tail-v3-{tag}-20261015'
        jobs = []
        for worker in range(4):
            log = (OUT / f'capture-{tag}-{worker}.log').open('w')
            cmd = [str(BASE / 'tools/blender/blender-4.2.0-linux-x64/blender'), '-b', '-t', '3',
                   '--python', 'scripts/generate_float_tail_v3.py', '--', '--root', str(root),
                   '--seed', str(seed), '--start', str(start), '--count', str(count),
                   '--worker', str(worker), '--workers', '4']
            jobs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(job.poll() is None for job in jobs):
            (OUT / 'capture-status.json').write_text(json.dumps({'stage': tag,
                'complete': len(list(root.glob('*/supervision_offline_01.npz'))), 'expected': count-start}))
            time.sleep(10)
        if any(job.returncode for job in jobs): raise RuntimeError('capture_failed:' + tag)
    (OUT / 'capture-status.json').write_text(json.dumps({'stage': 'complete'}))


if __name__ == '__main__': main()
