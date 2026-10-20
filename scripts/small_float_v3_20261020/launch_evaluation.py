import os,subprocess
from pathlib import Path
r=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');e=os.environ.copy();e['PYTHONPATH']='/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/src';e['OMP_NUM_THREADS']='2';e['OPENBLAS_NUM_THREADS']='1'
for i in range(4):
    log=open(r/f'evaluate-{i}.log','a');p=subprocess.Popen(['/root/autodl-tmp/envs/liquid-depth/bin/python','-u',str(r/'evaluate_small_float.py'),'--worker',str(i)],cwd=r,env=e,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);print(p.pid)
