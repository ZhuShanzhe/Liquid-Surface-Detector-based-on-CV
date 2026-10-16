"""Bounded capture/evaluation workers, source lock and explicit completion state."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from special_scene_protocol_v1 import protocol

REPO=Path(__file__).resolve().parents[1]
BLENDER='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=66)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    sources=sorted(list((REPO/'src/liquid_depth').glob('*.py'))+
                   [p for p in (REPO/'configs').glob('*.json') if p.name!='current_scene_evaluation_v1.json']+
                   list((REPO/'scripts').glob('*special_scene*py'))+[REPO/'scripts/generate_visual_evidence_v1.py',
                   REPO/'scripts/generate_other_scenes_v34.py',REPO/'scripts/generate_synthetic_liquid.py',
                   REPO/'scripts/generate_fullframe_v16.py',REPO/'scripts/evaluate_shared_empty_v45.py'])
    lock={str(s):sha(s) for s in sources}
    frozen={**protocol(),'root':str(a.root),'out':str(a.out),'count_override':a.count,
            'source_sha256':lock,'python':sys.version,'worker_processes':a.workers,
            'report_only_current_scene_evaluation_registry_excluded_from_runtime_lock':True}
    (a.out/'protocol.lock.json').write_text(json.dumps(frozen,indent=2))
    env={**os.environ,'PYTHONPATH':str(REPO/'src'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'2'}
    def stage(name,commands):
        (a.out/'status.json').write_text(json.dumps({'stage':name,'state':'running'}))
        logs=[(a.out/f'{name}-{i}.log').open('w') for i in range(len(commands))]
        procs=[subprocess.Popen(c,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT) for c,f in zip(commands,logs)]
        codes=[x.wait() for x in procs]
        for f in logs:f.close()
        if any(codes):
            raise RuntimeError(f'{name} failed: {codes}')
        assert all(sha(Path(s))==h for s,h in lock.items()),'source changed during fixed evaluation'
    stage('capture',[[BLENDER,'-b','-t','2','--python','scripts/generate_special_scene_strata_v1.py','--',
                     '--root',str(a.root),'--count',str(a.count),'--worker',str(i),'--workers',str(a.workers)] for i in range(a.workers)])
    common=[sys.executable,'scripts/evaluate_special_scene_strata_v1.py','--root',str(a.root),'--out',str(a.out),
            '--count',str(a.count),'--workers',str(a.workers)]
    stage('calibrate',[common+['--phase','calibrate','--worker',str(i)] for i in range(a.workers)])
    stage('input-evidence',[[sys.executable,'scripts/prepare_special_scene_evidence_v1.py',
          '--root',str(a.root),'--out',str(a.out),'--count',str(a.count)]])
    stage('evaluate',[common+['--phase','evaluate','--worker',str(i)] for i in range(a.workers)])
    stage('summarize',[common+['--phase','summarize']])
    (a.out/'status.json').write_text(json.dumps({'stage':'complete','state':'complete',
        'frames':a.count*72,'source_lock_verified':True},indent=2))
    print('ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
