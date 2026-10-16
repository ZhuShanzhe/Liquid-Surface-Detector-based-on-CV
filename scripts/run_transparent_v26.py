"""Freeze policy and source; compare 320 old frames and fresh target-only test."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches,decode_structured
from experiment_transparent_v26 import OUT,V25TEST,dump

repo=Path(__file__).resolve().parents[1]
data=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-refinement-v26-test-20261011')
started=time.time();cv2.setNumThreads(4)
if (OUT/'frozen-source.json').exists():raise FileExistsError('This frozen evaluation was already started')
files=[repo/p for p in ['src/liquid_depth/transparent_refinement_v26.py','src/liquid_depth/refraction_matching_v26.py',
    'src/liquid_depth/transparent_diagnostic_v25.py','src/liquid_depth/refraction_matching_v23.py',
    'src/liquid_depth/refraction_images_v23.py','scripts/experiment_transparent_v26.py','scripts/run_transparent_v26.py',
    'tests/test_transparent_refinement_v26.py']]+[OUT/'frozen-policy.json',OUT/'stage1-policy.json',OUT/'stage2-policy.json']
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};dump(OUT/'frozen-source.json',frozen)
env=dict(os.environ,PYTHONPATH='src:scripts')
def run(stage,command):
    dump(OUT/'progress.json',dict(stage=stage,status='running',elapsed_s=time.time()-started))
    with (OUT/(stage+'.log')).open('w') as log:res=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if res.returncode:
        dump(OUT/'progress.json',dict(stage=stage,status='failed',returncode=res.returncode));raise SystemExit(res.returncode)
run('regression',[sys.executable,'-m','pytest','-q'])
run('paired',[sys.executable,'scripts/experiment_transparent_v26.py','evaluate','--manifest',str(V25TEST),'--label','v25-paired'])
data.mkdir(exist_ok=False);configs=[]
for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
    for ci,(bottom,n) in enumerate([(b,n) for b in (.3,.8,2.,6.,9.8) for n in (1.315,1.425)]):
        seed=269117+ci*211+fi*10000
        for condition in ('transparent','translucent_mild'):
            folder=data/f'{family}_{ci:02d}_{condition}';folder.mkdir();captures=[]
            for j,(role,ratio) in enumerate([('prior',v) for v in (.10,.22,.35,.50,.65)]+
                [('test',v) for v in (.06,.12,.18,.31,.46,.60,.70,.76)]):
                sample=folder/f'{j:02d}';sample.mkdir()
                images,cal,truth=render_capture(seed,bottom*ratio,n,bottom,baseline,condition,family)
                for name,image in images.items():
                    if not cv2.imwrite(str(sample/(name+'.png')),image):raise IOError('write failed')
                dump(sample/'calibration.json',cal);np.savez_compressed(sample/'evaluator_truth.npz',**truth)
                obs=stereo_matches(images['left'],images['right'],cal,True) if family=='active_stereo' else decode_structured(images,cal,True)
                np.savez_compressed(sample/'matches_optimized.npz',**{k:obs[k] for k in ('left_q','right_q','weights','left_pixels','right_pixels')})
                capture=dict(folder=str(sample),role=role,evaluator_truth=str(sample/'evaluator_truth.npz'))
                if role=='prior':capture['known_liquid_height_m']=float(bottom*ratio+np.random.default_rng(seed+j*65537).normal(0,.0005+.001*bottom*ratio))
                captures.append(capture)
            configs.append(dict(id=folder.name,family=family,condition=condition,camera_to_bottom_m=bottom,split='test',captures=captures))
            dump(OUT/'progress.json',dict(stage='generate',configs=len(configs),total_configs=40,elapsed_s=time.time()-started))
            print('GENERATED',len(configs),40,flush=True)
dump(data/'manifest.json',dict(configs=configs,independent_seed_family='269117+211*i+10000*family',
    true_indices=[1.315,1.425],shared_forward_inverse_assumptions=True,test_used_for_selection=False))
run('holdout',[sys.executable,'scripts/experiment_transparent_v26.py','evaluate','--manifest',str(data/'manifest.json'),'--label','holdout'])
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
dump(OUT/'progress.json',dict(stage='complete',status='complete',elapsed_s=time.time()-started,source_and_policy_unchanged=unchanged))
if not unchanged:raise RuntimeError('Frozen source/policy changed during evaluation')
