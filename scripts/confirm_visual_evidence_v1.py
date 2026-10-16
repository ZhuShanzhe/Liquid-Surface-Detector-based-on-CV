"""Freeze model/selector and confirm on new installations after unit-test bug fix."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
OUT=BASE/'liquid-depth-artifacts/visual-route-v1-20261015'
ROOT=BASE/'liquid-depth-data/synthetic/visual-evidence-v1-confirmation-20261015'
PY=BASE/'envs/liquid-depth/bin/python';SEED=202610153119


def status(stage,**kw):
    (OUT/'status.json').write_text(json.dumps({'stage':stage,'updated_unix':time.time(),**kw},indent=2))


def main():
    os.chdir(REPO);env={**os.environ,'PYTHONPATH':str(REPO/'src'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
    sources=[REPO/'src/liquid_depth/symmetric_visual_meter.py',REPO/'src/liquid_depth/visual_region_model.py',
             REPO/'scripts/generate_visual_evidence_v1.py',REPO/'scripts/evaluate_visual_route_v1.py',OUT/'model-v1/best.pt']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'confirmation-frozen-protocol.json').write_text(json.dumps({'seed':SEED,'source_sha256':hashes,
         'reason':'Histogram endpoint candidate omission discovered by a full-chain unit test, not by fitting test truth.',
         'epochs':60,'fresh_installations':198,'frames':396,'input_only_precision_cohort':True},indent=2))
    processes=[];logs=[];status('confirmation_generating')
    for worker in range(4):
        log=(OUT/f'confirmation-generate-{worker}.log').open('w');logs.append(log)
        processes.append(subprocess.Popen([str(BASE/'tools/blender/blender-4.2.0-linux-x64/blender'),'-b','-t','4',
            '--python','scripts/generate_visual_evidence_v1.py','--','--root',str(ROOT),'--seed',str(SEED),
            '--start-index','396','--count','594','--worker',str(worker),'--workers','4'],env=env,stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in processes):
        status('confirmation_generating',complete_installations=len(list(ROOT.glob('*/frames.json'))) if ROOT.exists() else 0)
        time.sleep(15)
    for log in logs:log.close()
    if any(p.returncode for p in processes) or len(list(ROOT.glob('*/frames.json')))!=198:raise RuntimeError('confirmation_capture_failed')
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):raise RuntimeError('frozen_source_changed')
    status('confirmation_evaluating')
    with (OUT/'confirmation-evaluation.log').open('w') as log:
        subprocess.run([str(PY),'scripts/evaluate_visual_route_v1.py','--root',str(ROOT),'--model',str(OUT/'model-v1/best.pt'),
            '--output',str(OUT/'confirmation'),'--seed',str(SEED)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status('confirmation_complete')


if __name__=='__main__':
    try:main()
    except Exception as exc:status('failed',error=repr(exc));raise
