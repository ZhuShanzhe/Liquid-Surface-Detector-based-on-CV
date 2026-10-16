"""Replay after an output-label type repair; no model, gate or test data changes."""
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

BASE = Path('/root/autodl-tmp'); REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/float-tail-v3-20261015'


def main():
    target = OUT/'frozen-test-protocol.json'; protocol = json.loads(target.read_text())
    changed = str(REPO/'scripts/evaluate_float_v3.py')
    for name, digest in protocol['sha256'].items():
        if name != changed: assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    shutil.copyfile(target, OUT/'frozen-test-protocol-before-label-repair.json')
    protocol['sha256'][changed] = hashlib.sha256(Path(changed).read_bytes()).hexdigest()
    protocol['summary_repair'] = 'integer depth stratum converted to a string; measurement, model, thresholds and test data unchanged'
    target.write_text(json.dumps(protocol, indent=2))
    shutil.move(str(OUT/'fresh-evaluation'), str(OUT/'first-summary-attempt'))
    shutil.copyfile(OUT/'fresh-evaluation.log', OUT/'first-summary-attempt.log')
    (OUT/'status.json').write_text(json.dumps({'stage': 'fresh_paired_replay_after_summary_label_fix'}))
    with (OUT/'fresh-evaluation.log').open('w') as log:
        subprocess.run([str(BASE/'envs/liquid-depth/bin/python'), 'scripts/evaluate_float_v3.py'], cwd=REPO,
            env={**os.environ, 'PYTHONPATH': str(REPO/'src')+':'+str(REPO/'scripts'),
                 'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1'}, stdout=log, stderr=subprocess.STDOUT, check=True)
    (OUT/'status.json').write_text(json.dumps({'stage': 'complete'}))


if __name__ == '__main__':
    try: main()
    except Exception as exc:
        (OUT/'status.json').write_text(json.dumps({'stage': 'failed', 'error': repr(exc)})); raise
