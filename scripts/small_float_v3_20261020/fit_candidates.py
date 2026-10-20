import json,time,sys,hashlib,argparse
from pathlib import Path
import cv2,numpy as np,torch,joblib
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor,observed_features
from small_float_model import dense_candidates,score_candidates,choose_highest,peak_roi
args=argparse.ArgumentParser();args.add_argument('--source',choices=['old','new'],default='new');source=args.parse_args().source
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020');OUT=ROOT/f'candidate-{source}';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4);cv2.setNumThreads(1)
while source=='new' and not (ROOT/'pixel-training/complete.json').exists():print('WAIT_PIXEL_TRAINING',flush=True);time.sleep(30)
for split in ('train','validation'):
    while not all((ROOT/'baseline'/f'{split}-{i}.complete').exists() for i in (0,1)):print('WAIT_BASELINE',split,flush=True);time.sleep(10)
checkpoint=ROOT/'pixel-training/best.pt' if source=='new' else Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/assets/models/v8/pixel_best.pt')
predictor=RGBDEvidencePredictor(checkpoint,device='cuda')
all_data={};train_x=[];train_y=[];train_top=[];train_delta=[];weights=[]
for split in ('train','validation'):
    records=[json.loads(s) for i in (0,1) for s in (ROOT/'baseline'/f'{split}-{i}.jsonl').read_text().splitlines()]
    annotated=[]
    for r in sorted(records,key=lambda x:x['frame']):
        prep=ROOT/'prepared-v2'/split/f"{r['group']:04d}";f=np.load(prep/f"{r['bits']:02d}.npz");bundle=json.loads((prep/'bundle.json').read_text())
        features=f['features'].astype(np.float32)
        if source=='old':
            folder=DATA/split/r['frame'];rgb=cv2.imread(str(folder/'rgb.png'));dry=np.load(folder/'dry_depth_m.npy')
            features=observed_features(rgb,f['raw'],dry,peak_roi(bundle,f['raw']),bundle)
        pred=predictor.predict(features);candidates=dense_candidates(f['raw'],pred,bundle,r['result']['liquid_depth_m'])
        sem=f['semantic'];rel=f['reliable'];inst=f['instances'];height=f['true_height'];highest=float(height[(sem==3)].max()) if (sem==3).any() else None
        for c in candidates:
            mask=c['mask'];purity=float(np.mean(sem[mask]==3));direct=float(rel[mask].mean());ids=inst[mask];ids=ids[ids>0]
            own=None
            if len(ids):
                k=int(np.bincount(ids).argmax());own=float(height[inst==k].max())
            y=bool(purity>=.60 and direct>=.50 and own is not None)
            top=bool(y and own>=highest-max(.005,c['noise_m']))
            c.update(target=int(y),top_target=int(top),reference=own,float_purity=purity,direct_purity=direct)
            if split=='train':
                train_x.append(c['features']);train_y.append(int(y));train_top.append(int(top));train_delta.append((own-c['height_raw_m'])/c['noise_m'] if y else 0.);weights.append(1/max(len(candidates),1))
        truth=json.loads((DATA/split/r['frame']/'truth.json').read_text());truth.pop('meshes',None)
        for c in candidates:c.pop('mask',None)
        annotated.append(dict(frame=r['frame'],group=r['group'],bits=r['bits'],old_peak=r['result'].get('highest_height_m'),liquid=r['result'].get('liquid_depth_m'),truth=truth,candidates=candidates))
    all_data[split]=annotated;print('CANDIDATES',split,len(annotated),sum(len(r['candidates']) for r in annotated),flush=True)
x=np.array(train_x,np.float32);y=np.array(train_y);t=np.array(train_top);w=np.array(weights)
kwargs=dict(n_estimators=160,max_depth=14,min_samples_leaf=3,n_jobs=8,random_state=202610204213%(2**32),class_weight='balanced')
classifier=ExtraTreesClassifier(**kwargs).fit(x,y,sample_weight=w)
top=ExtraTreesClassifier(**kwargs).fit(x,t,sample_weight=w)
reg=ExtraTreesRegressor(n_estimators=160,max_depth=12,min_samples_leaf=4,n_jobs=8,random_state=202610204213%(2**32)).fit(x[y==1],np.array(train_delta)[y==1],sample_weight=w[y==1])
bundle={'object':classifier,'top':top,'correction':reg};joblib.dump(bundle,OUT/'models.joblib')
val=all_data['validation']
for r in val:r['scored']=score_candidates(r['candidates'],bundle)
policies=[]
for threshold in (.60,.75,.85,.90,.95,.98,.995):
    for top_threshold in (0.,.15,.30,.50):
        p=dict(threshold=threshold,top_threshold=top_threshold);errors=[];control_regress=0;updates=0
        for r in val:
            baseline=r['old_peak'];value=baseline;c=choose_highest(r['scored'],p)
            if c is not None and (value is None or c['height_m']>value):value=c['height_m'];updates+=1
            truth=r['truth']['highest_height_m'];e=abs(value-truth) if value is not None else 10.
            if r['bits']&8:errors.append(e)
            elif value is not None and baseline is not None and abs(baseline-truth)<=max(.05,.05*truth) and e>max(.05,.05*truth):control_regress+=1
        a=np.array(errors);policies.append({**p,'mae_m':float(a.mean()),'p95_m':float(np.percentile(a,95)),'max_m':float(a.max()),'control_new_failures':control_regress,'updates':updates,'objective':float(a.mean()+.1*np.percentile(a,95)+.1*a.max()+10*control_regress)})
selected=min(policies,key=lambda p:p['objective']);selected.update(pixel_checkpoint=str(checkpoint),feature_mode='tof_linear' if source=='new' else 'legacy',source=source)
(OUT/'policy.json').write_text(json.dumps(selected,indent=2));(OUT/'validation-policies.json').write_text(json.dumps(policies,indent=2))
for r in all_data['validation']:
    for k in ('candidates','scored'):r.pop(k,None)
summary=dict(train_frames=len(all_data['train']),validation_frames=len(val),training_candidates=len(x),positive_candidates=int(y.sum()),top_candidates=int(t.sum()),selected=selected,test_frames_used=0)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
protocol=dict(base_commit='182bbd73aacd5bd1731e5321faf2c427421b4259',test_seed=202610202351,test_installations=100,test_frames=1000,
    liquid_must_be_exactly_unchanged=True,visual_highest_only=True,higher_supported_candidate_only=True,no_error_based_exclusion=True,
    acceptance='No new tolerance failure on no-float controls; per float condition MAE<=baseline+2mm, P95<=baseline+5mm, max<=baseline+5mm, pass>=baseline-2 percentage points; some material aggregate improvement; latency p95<=500ms target, report cold and warm separately',
    top_support_audit='offline visible highest source has >=3 direct-return points in a 2-D spatial patch; separate strata, not a numerical-output gate',
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'small_float_model.py',checkpoint,OUT/'models.joblib',OUT/'policy.json']})
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'complete.json').write_text(json.dumps({'complete':True}));print('CANDIDATE_TRAINING_COMPLETE',flush=True)
