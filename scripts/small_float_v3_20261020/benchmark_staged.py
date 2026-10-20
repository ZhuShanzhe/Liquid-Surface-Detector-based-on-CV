"""Live sequential replay, not a sum of independently cached timings."""
import json,time,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');STAGE=ROOT/'staged-release'
sys.path.insert(0,str(STAGE/'scripts'))
from liquid_depth.unified_tof import UnifiedToFMeter,ToFInstallation,ToFObservation
from liquid_depth.scene_classifier import SceneClassifier
torch.set_num_threads(2);cv2.setNumThreads(1)
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020/test')
cached={r['frame']:r for i in range(4) for r in map(json.loads,(ROOT/'independent-test'/f'records-{i}.jsonl').read_text().splitlines())}
classifier=SceneClassifier(STAGE/'assets/models/scene_router/scene_head.pt');records=[];initialization=[];mismatches=[]
for g in range(100):
    bundle=json.loads((ROOT/'prepared-v2/test'/f'{g:04d}/bundle.json').read_text());dry=np.load(DATA/f'{g:04d}_00/dry_depth_m.npy')
    t=time.perf_counter();meter=UnifiedToFMeter(ToFInstallation(str(g),bundle,dry,'ordinary_diffuse'),classifier,profile_path=ROOT/'baseline/profile.json')
    warm=meter.warm_optional_peak_models();initialization.append(dict(group=g,initialization_ms=(time.perf_counter()-t)*1000,warm_ms=warm))
    for i,bits in enumerate((0,1,2,4,16,8,9,10,12,24)):
        frame=f'{g:04d}_{bits:02d}';folder=DATA/frame
        raw=np.load(folder/'depth_m.npy');rgb=cv2.imread(str(folder/'rgb.png'));valid=np.load(folder/'capture_valid.npy')
        start=time.perf_counter();r=meter.measure(ToFObservation(frame,rgb,raw,float(i),installation_id=str(g),capture_validity=valid));elapsed=(time.perf_counter()-start)*1000
        ref=cached[frame]
        for key,target in [('liquid_depth_m','new_liquid'),('highest_height_m','new_peak')]:
            a,b=r.get(key),ref[target]
            if (a is None)!=(b is None) or (a is not None and abs(a-b)>1e-7):mismatches.append(dict(frame=frame,key=key,actual=a,reference=b))
        records.append(dict(frame=frame,bits=bits,pipeline_ms=r['pipeline_ms'],wall_ms=elapsed,used=r.get('diagnostics',{}).get('small_float_peak',{}).get('used',False)))
    print('LIVE_REPLAY',g,flush=True)
def metric(xs):
    a=np.array(xs);return dict(n=len(a),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),max_ms=float(a.max()),over_500ms=int(np.sum(a>500)))
summary=dict(all=metric([r['wall_ms'] for r in records]),floats=metric([r['wall_ms'] for r in records if r['bits']&8]),controls=metric([r['wall_ms'] for r in records if not r['bits']&8]),initialization=metric([r['initialization_ms'] for r in initialization]),parity_mismatches=mismatches,
    measured_directly=True,sequential_cpu_threads=2,cold_model_load_in_initialization=True,scene_router_loaded_once=True,no_gpu_in_inference=True)
(ROOT/'live-benchmark.json').write_text(json.dumps(dict(summary=summary,records=records,initialization=initialization),indent=2));print(json.dumps(summary),flush=True)
