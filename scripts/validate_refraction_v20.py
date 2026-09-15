"""Run frozen V20 and V19 on two new nuisance-matched simulation sets."""
import os,json,subprocess,concurrent.futures,time,hashlib
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v20')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
PY='/root/autodl-tmp/envs/liquid-depth/bin/python'
BL='/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender'
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(args,log):
 with (O/log).open('w') as f:subprocess.run(args,cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
def evaluate(tag,d,seed):
 run([PY,'scripts/evaluate_refraction_v20.py','--data',str(d),'--output',str(O/(tag+'-v20')),'--burst-mode','persistent','--burst-tag','64','--frames','64'],tag+'-v20.log')
 run([PY,'scripts/stress_refraction_v20.py','--data',str(d),'--output',str(O/(tag+'-v20-stress.json')),'--count','24','--seed',str(seed),'--random-ghost-thickness'],tag+'-v20-stress.log')
 run([PY,'scripts/evaluate_refraction_v19.py','--data',str(d),'--output',str(O/(tag+'-v19-geometry')),'--burst-mode','persistent','--burst-tag','64','--frames','64'],tag+'-v19.log')
 run([PY,'scripts/evaluate_refraction_evidence_v19.py','--data',str(d),'--results',str(O/(tag+'-v19-geometry/results.json')),'--output',str(O/(tag+'-v19'))],tag+'-v19-evidence.log')
 run([PY,'scripts/stress_refraction_v19.py','--data',str(d),'--output',str(O/(tag+'-v19-stress-geometry.json')),'--count','24','--seed',str(seed),'--random-ghost-thickness'],tag+'-v19-stress.log')
 run([PY,'scripts/evaluate_refraction_evidence_v19.py','--data',str(d),'--results',str(O/(tag+'-v19-stress-geometry.json')),'--output',str(O/(tag+'-v19-stress')),'--stress'],tag+'-v19-stress-evidence.log')
 print('EVALUATION_FINISHED',tag,flush=True)
def main():
 start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  jobs=[]
  for tag,seed in [('heldoutA',205001),('heldoutB',206001)]:
   d=DATA/('stratified-v20-'+tag)
   run([BL,'-b','-t','6','--python','scripts/generate_stratified_refraction_v20.py','--','--output',str(d),'--count','96','--samples','2048','--device','GPU','--seed',str(seed),'--split',tag,'--exposure','-2','--auto-exposure'],tag+'-render.log')
   run([PY,'scripts/generate_refraction_burst_v19.py',str(d),'--frames','64','--tag','64','--modes','persistent'],tag+'-burst.log')
   jobs.append(pool.submit(evaluate,tag,d,seed+600000))
   print('ACQUISITION_FINISHED',tag,round(time.time()-start,1),flush=True)
  for job in jobs:job.result()
 print('FROZEN_VALIDATION_FINISHED',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
