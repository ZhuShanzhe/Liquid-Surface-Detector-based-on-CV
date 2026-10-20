import os,subprocess,json
from pathlib import Path
root=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='1'
pids=[]
for i in range(8):
    log=open(root/f'render-{i}.log','a')
    p=subprocess.Popen(['/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender','-b','-t','2','--python',str(root/'generate_small_floats.py'),'--','--out','/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020','--worker',str(i)],cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(p.pid)
(root/'render-pids.json').write_text(json.dumps(pids));print(pids)
