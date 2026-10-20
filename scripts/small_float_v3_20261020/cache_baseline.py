import sys,json,time,argparse
from pathlib import Path
import cv2,numpy as np,torch
# Replay the frozen formal v2, never silently rerun a newer default as baseline.
REPO=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020/frozen-v2')
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
from measure_release import validated_profile
from liquid_depth.unified_tof import UnifiedToFMeter,ToFInstallation,ToFObservation
from liquid_depth.scene_classifier import SceneClassifier
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020')
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist() if x.size<128 else {'shape':list(x.shape)}
    if isinstance(x,np.generic):return x.item()
    return x
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
out=ROOT/'baseline';out.mkdir(exist_ok=True);profile=out/'profile.json';profile.write_text(json.dumps(validated_profile()));classifier=SceneClassifier(REPO/'assets/models/scene_router/scene_head.pt')
for split,count in [('train',128),('validation',32),('test',100)]:
    dest=out/f'{split}-{a.worker}.jsonl';done=set()
    if dest.exists():done={json.loads(s)['frame'] for s in dest.read_text().splitlines()}
    with dest.open('a',buffering=1) as stream:
        for g in range(a.worker,count,2):
            prep=ROOT/'prepared-v2'/split/f'{g:04d}';deadline=time.monotonic()+10800
            while not (prep/'24.npz').exists():
                if time.monotonic()>deadline:raise TimeoutError(str(prep))
                time.sleep(2)
            bundle=json.loads((prep/'bundle.json').read_text());dry=np.load(DATA/split/f'{g:04d}_00/dry_depth_m.npy')
            meter=UnifiedToFMeter(ToFInstallation(str(g),bundle,dry,'ordinary_diffuse'),classifier,profile_path=profile);meter.warm_optional_peak_models()
            for i,bits in enumerate((0,1,2,4,16,8,9,10,12,24)):
                frame=f'{g:04d}_{bits:02d}'
                if frame in done:continue
                folder=DATA/split/frame;d=np.load(folder/'depth_m.npy');rgb=cv2.imread(str(folder/'rgb.png'));native=np.load(folder/'capture_valid.npy')
                result=meter.measure(ToFObservation(frame,rgb,d,float(i),installation_id=str(g),capture_validity=native))
                stream.write(json.dumps(clean(dict(frame=frame,group=g,bits=bits,split=split,result=result)))+'\n');done.add(frame)
            print('BASELINE',split,g,flush=True)
    (out/f'{split}-{a.worker}.complete').write_text('complete')
