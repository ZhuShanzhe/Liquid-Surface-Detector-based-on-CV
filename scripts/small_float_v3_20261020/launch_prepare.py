import os,subprocess,json
from pathlib import Path
r=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');e=os.environ.copy();e['PYTHONPATH']='/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/src';e['OMP_NUM_THREADS']='1';e['OPENBLAS_NUM_THREADS']='1';pids=[]
for i in range(4):
    log=open(r/f'prepare-{i}.log','a');p=subprocess.Popen(['/root/autodl-tmp/envs/liquid-depth/bin/python','-u',str(r/'prepare_training.py'),'--worker',str(i)],cwd=r,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(p.pid)
(r/'prepare-pids.json').write_text(json.dumps(pids));print(pids)
