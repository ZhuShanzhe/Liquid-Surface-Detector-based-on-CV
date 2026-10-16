"""Generate distinct validation and untouched final burst-test installations."""
import json
import os
import subprocess
from pathlib import Path

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/floating-joint-v9-20261016'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, seed in [('validation', 202610164119), ('test', 202610164913)]:
        root = BASE/f'liquid-depth-data/synthetic/floating-joint-v9-{split}-20261016'
        assert not root.exists(), 'new_output_root_required'
        jobs = []
        for worker in range(4):
            handle = (OUT/f'capture-{split}-worker-{worker}.log').open('w')
            jobs.append(subprocess.Popen([str(BASE/'tools/blender/blender-4.2.0-linux-x64/blender'), '-b', '-t', '3',
                '--python', 'scripts/generate_joint_v9.py', '--', '--root', str(root), '--seed', str(seed),
                '--installations', '99', '--worker', str(worker), '--workers', '4'], cwd=REPO,
                env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '3'}, stdout=handle, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in jobs]
        assert not any(codes), codes
        assert len(list(root.glob('*/supervision_offline_03.npz'))) == 99
        (OUT/f'capture-{split}-complete.json').write_text(json.dumps({'seed': seed, 'installations': 99, 'frames': 396,
            'four_noisy_depth_exposures_per_installation': True, 'same_static_RGB_per_burst': True}))
        print('V9_CAPTURE_COMPLETE', split, flush=True)


if __name__ == '__main__':
    main()
