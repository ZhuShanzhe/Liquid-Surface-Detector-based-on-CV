"""Complete synthetic ablation, then add real weak priors and revalidate optional-use gate."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'
ROOT=BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-20261015'
PY=BASE/'envs/liquid-depth/bin/python';SEED=202610154119


def main():
    status=OUT/'prior-status.json'
    status.write_text(json.dumps({'stage':'waiting_for_synthetic_control'}))
    while True:
        control=json.loads((OUT/'status.json').read_text())
        if control['stage']=='failed':raise RuntimeError('synthetic_control_failed:'+control.get('error',''))
        if control['stage']=='complete':break
        time.sleep(15)
    env={**os.environ,'PYTHONPATH':str(REPO/'src')+':'+str(REPO/'scripts'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1'}
    steps=[('real-prior-training',['scripts/train_rgbd_evidence_v2.py','--root',str(ROOT),'--output',str(OUT/'pixel-model-with-real'),
          '--seed',str(SEED),'--epochs','20','--initialize',str(OUT/'pixel-model/best.pt')]),
        ('real-prior-candidate',['scripts/train_candidate_trust_v2.py','--root',str(ROOT),'--model',str(OUT/'pixel-model-with-real/best.pt'),
          '--output',str(OUT/'candidate-model-with-real'),'--seed',str(SEED)]),
        ('real-prior-evaluation',['scripts/evaluate_adaptive_route_v2.py','--root',str(ROOT),'--model',str(OUT/'pixel-model-with-real/best.pt'),
          '--gate',str(OUT/'candidate-model-with-real/gate.pt'),'--output',str(OUT/'evaluation-with-real'),'--seed',str(SEED)])]
    for name,args in steps:
        status.write_text(json.dumps({'stage':name,'time_unix':time.time()}))
        with (OUT/f'{name}.log').open('w') as log:
            subprocess.run([str(PY),*args],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    status.write_text(json.dumps({'stage':'complete','time_unix':time.time()}))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (OUT/'prior-status.json').write_text(json.dumps({'stage':'failed','error':repr(exc)}));raise
