import json,time,subprocess,os,hashlib
from pathlib import Path
import cv2
import numpy as np
from experiment_ordinary_v39 import configuration,clean_maps,PROFILE
from experiment_ordinary_v40 import observation
from liquid_depth.ordinary_surface_v40 import prepare,calibrate,estimate

REPO=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
BASE=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
cv2.setNumThreads(1)
verification={}
for name in ['test','outliers','rho05','rho0']:
    previous=BASE/f'ordinary-v40-{name}-20261012'
    final=BASE/f'ordinary-v40-{name}-20261012-final'
    old=[json.loads(x) for x in (previous/'records.jsonl').read_text().splitlines()]
    new=[json.loads(x) for x in (final/'records.jsonl').read_text().splitlines()]
    def stripped(row):return {k:v for k,v in row.items() if k!='elapsed_s'}
    hashes=json.loads((final/'protocol.json').read_text())['source_hashes']
    verification[name]=dict(records=len(new),identical_except_runtime=len(old)==len(new) and all(stripped(a)==stripped(b) for a,b in zip(old,new)),
        source_hashes_match=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()))
times=[];init=[]
for regime in ['nominal','adverse']:
    _,cal,rim,h,distance,_,drop=configuration(202610124207,0,'far_shallow')
    dry,wet,_,_=clean_maps(cal,rim,h)
    start=time.perf_counter();ctx=prepare(cal,rim,wet.shape,PROFILE.baseline_m)
    geom=time.perf_counter()-start
    empty,_=observation(dry,202610124207,99,regime,1.,'diffuse_clean',drop)
    raw,_=observation(wet,202610124207,0,regime,1.,'diffuse_clean',drop)
    start=time.perf_counter();ref=calibrate(empty,ctx);ref_time=time.perf_counter()-start
    estimate(raw,ctx,ref)
    for _ in range(20):
        start=time.perf_counter();estimate(raw,ctx,ref);times.append(time.perf_counter()-start)
    init.append(dict(regime=regime,geometry_prepare_s=geom,dry_reference_s=ref_time))
verification['runtime']=dict(device='server CPU single BLAS/OpenMP/OpenCV thread',native_shape=[800,1280],runs=len(times),
    p50_s=float(np.median(times)),p95_s=float(np.percentile(times,95)),max_s=max(times),
    excludes='capture, USB/network transport, RGB/IR matching, GPU model, GUI; NOT end-to-end latency',initialization=init)
tests=[REPO/'tests'/('test_ordinary_'+v+'.py') for v in ['v40','v39','v39b']]
run=subprocess.run(['/root/autodl-tmp/envs/liquid-depth/bin/python','-m','pytest','-q',*map(str,tests)],capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(REPO/'src')})
verification['tests']=dict(exit_code=run.returncode,output=run.stdout,stderr=run.stderr)
(BASE/'ordinary-v40-verification-20261012.json').write_text(json.dumps(verification,indent=2,allow_nan=False))
print(json.dumps(verification,indent=2))
