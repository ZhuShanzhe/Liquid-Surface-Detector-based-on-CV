import argparse,json,time,sys
from pathlib import Path
import numpy as np,cv2,torch,joblib
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor,observed_features
from small_float_model import dense_candidates,score_candidates,choose_highest,float_features,peak_roi
sys.path.insert(0,'/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/scripts')
from special_scene_input_evidence_v1 import assess,assess_capture_origin
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020/test')
p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1)
while not (ROOT/'candidate-training/complete.json').exists():time.sleep(15)
while not all((ROOT/'baseline'/f'test-{i}.complete').exists() for i in (0,1)):time.sleep(10)
models=joblib.load(ROOT/'candidate-training/models.joblib')
for m in models.values():m.n_jobs=1
policy=json.loads((ROOT/'candidate-training/policy.json').read_text());predictor=RGBDEvidencePredictor(policy['pixel_checkpoint'],device='cpu')
rows=[json.loads(s) for i in (0,1) for s in (ROOT/'baseline'/f'test-{i}.jsonl').read_text().splitlines()]
out=ROOT/'independent-test';out.mkdir(exist_ok=True);path=out/f'records-{a.worker}.jsonl'
with path.open('w',buffering=1) as stream:
    for g in range(a.worker,100,4):
        prep=ROOT/'prepared-v2/test'/f'{g:04d}';bundle=json.loads((prep/'bundle.json').read_text())
        # Kernel warmup predicts only, never updates a parameter or uses a target.
        dry=np.load(DATA/f'{g:04d}_00/dry_depth_m.npy');x=np.load(prep/'00.npz')['features'].astype(np.float32);predictor.predict(x)
        for r in sorted([r for r in rows if r['group']==g],key=lambda r:r['bits']):
            f=np.load(prep/f"{r['bits']:02d}.npz");raw=f['raw'];old=r['result'];rgb=cv2.imread(str(DATA/r['frame']/'rgb.png'));start=time.perf_counter()
            if policy['feature_mode']=='legacy':features=observed_features(rgb,raw,dry,peak_roi(bundle,raw),bundle)
            else:features,_=float_features(rgb,raw,dry,bundle)
            prediction=predictor.predict(features);candidates=dense_candidates(raw,prediction,bundle,old['liquid_depth_m']);scored=score_candidates(candidates,models);chosen=choose_highest(scored,policy)
            value=old.get('highest_height_m');used=False
            if chosen is not None and (value is None or chosen['height_m']>value):value=chosen['height_m'];used=True
            extra_ms=(time.perf_counter()-start)*1000
            # Nothing above this point has read current truth, semantic,
            # reliability, instance or return-origin labels.
            folder=DATA/r['frame'];truth=json.loads((folder/'truth.json').read_text());truth.pop('meshes',None)
            geom=np.load(folder/'offline_geometry.npz');sem=geom['semantic'];error=f['offline_error'];roi=sem==2;valid=raw>0
            source=(sem==3) if r['bits']&8 else (sem==2);trueh=f['true_height'];direct=source&(error==0)&valid
            top=float(trueh[source].max()) if source.any() else None
            near=direct&(trueh>=top-.02) if top is not None else direct&False
            yy,xx=np.nonzero(near);observable=len(xx)>=3 and np.ptp(xx)>=1 and np.ptp(yy)>=1
            audit=dict(online=assess(raw,bundle),origin=assess_capture_origin(raw,sem,error,contains_liquid=True),liquid_missing_fraction=float(np.mean(~valid[roi])) if roi.any() else 1.,surface_pixels=int(roi.sum()),highest_near_direct_points=len(xx),highest_support_observable=bool(observable),visible_float_pixels=int((sem==3).sum()))
            selected=None if chosen is None else {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in chosen.items() if k not in {'mask','features'}}
            record=dict(frame=r['frame'],group=g,bits=r['bits'],truth=truth,old_liquid=old['liquid_depth_m'],new_liquid=old['liquid_depth_m'],old_peak=old.get('highest_height_m'),new_peak=value,used=used,
                baseline_visual=old['module_plan']['visual_peak'],new_activation=bool(used and not old['module_plan']['visual_peak']),candidate_count=len(candidates),selected=selected,
                baseline_ms=old['pipeline_ms'],extra_ms=extra_ms,total_estimated_ms=old['pipeline_ms']+extra_ms,input_audit=audit,request=json.loads((folder/'request.json').read_text()))
            stream.write(json.dumps(record)+'\n')
        print('INDEPENDENT_TEST',g,flush=True)
(out/f'complete-{a.worker}.json').write_text(json.dumps({'complete':True}))
