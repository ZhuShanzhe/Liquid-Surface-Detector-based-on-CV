"""Fresh target-only, frozen-source test. No strong-scatter captures generated."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches,decode_structured
from experiment_transparent_v25 import OUT,dump

repo=Path(__file__).resolve().parents[1]
data=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-diagnostic-v25-20261011')
start=time.time();cv2.setNumThreads(4)
files=[repo/p for p in ['src/liquid_depth/transparent_diagnostic_v25.py',
    'src/liquid_depth/refraction_matching_v23.py','src/liquid_depth/refraction_images_v23.py',
    'scripts/experiment_transparent_v25.py','scripts/run_transparent_v25.py']]+[OUT/'frozen-selection.json']
if (OUT/'frozen-source.json').exists():raise FileExistsError('Test has already been started')
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
dump(OUT/'frozen-source.json',frozen)
data.mkdir(exist_ok=False);configs=[]
for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
    for ci,(bottom,n) in enumerate([(b,n) for b in (.3,.8,2.,6.,9.8) for n in (1.325,1.405)]):
        seed=257031+ci*193+fi*10000
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
            configs.append(dict(id=folder.name,family=family,condition=condition,split='test',camera_to_bottom_m=bottom,captures=captures))
            dump(OUT/'progress.json',dict(stage='generate',configs=len(configs),total=40,elapsed_s=time.time()-start))
            print('GENERATED',len(configs),40,flush=True)
dump(data/'manifest.json',dict(configs=configs,scope=['transparent','translucent_mild'],
    shared_forward_inverse_assumptions=True,independent_seed_family='257031+193*i+10000*family',
    test_used_for_selection=False))
env=dict(os.environ,PYTHONPATH='src:scripts')
for stage,cmd in [('test',[sys.executable,'scripts/experiment_transparent_v25.py','test','--manifest',str(data/'manifest.json')]),
                  ('regression',[sys.executable,'-m','pytest','-q'])]:
    dump(OUT/'progress.json',dict(stage=stage,status='running',elapsed_s=time.time()-start))
    with (OUT/(stage+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        dump(OUT/'progress.json',dict(stage=stage,status='failed',returncode=r.returncode));raise SystemExit(r.returncode)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items())
dump(OUT/'progress.json',dict(stage='complete',status='complete',elapsed_s=time.time()-start,source_unchanged=unchanged))
if not unchanged:raise RuntimeError('Source changed during holdout')
