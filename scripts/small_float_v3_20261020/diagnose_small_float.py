"""Post-test failure diagnosis only. Must not change model, policy or test set."""
import json
from pathlib import Path
from collections import Counter
import cv2,numpy as np,torch,joblib
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor
from small_float_model import float_features,dense_candidates,score_candidates
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020/test')
torch.set_num_threads(2);cv2.setNumThreads(1)
rows=[json.loads(s) for i in range(4) for s in (ROOT/'independent-test'/f'records-{i}.jsonl').read_text().splitlines()]
policy=json.loads((ROOT/'candidate-training/policy.json').read_text());pred=RGBDEvidencePredictor(policy['pixel_checkpoint']);models=joblib.load(ROOT/'candidate-training/models.joblib')
for m in models.values():m.n_jobs=1
results=[]
for r in rows:
    a=r['input_audit'];t=r['truth']['highest_height_m'];v=r['new_peak']
    if not r['bits']&8 or not a['online']['eligible'] or not a['origin']['eligible']:continue
    if r['bits']&16 and a['liquid_missing_fraction']>.5+1/max(a['surface_pixels'],1):continue
    if v is not None and abs(v-t)<=max(.05,.05*abs(t)):continue
    g=r['group'];f=np.load(ROOT/'prepared-v2/test'/f"{g:04d}/{r['bits']:02d}.npz");bundle=json.loads((ROOT/'prepared-v2/test'/f'{g:04d}/bundle.json').read_text())
    raw=f['raw'];dry=np.load(DATA/f'{g:04d}_00/dry_depth_m.npy');rgb=cv2.imread(str(DATA/r['frame']/'rgb.png'));features,_=float_features(rgb,raw,dry,bundle);p=pred.predict(features)
    sem=f['semantic'];trueh=f['true_height'];source=sem==3
    top=float(trueh[source].max()) if source.any() else -np.inf
    near=source&(trueh>=top-.02)&(f['offline_error']==0)&(raw>0)
    visible=near&(p['semantic'][3]>=.25)&(p['reliability'][0]>=.2)
    candidates=score_candidates(dense_candidates(raw,p,bundle,r['old_liquid']),models)
    relevant=[c for c in candidates if np.sum(c['mask']&near)>=3]
    accepted=[c for c in relevant if c['object_probability']>=policy['threshold'] and c['top_probability']>=policy['top_threshold'] and abs(c['correction_m'])<=3*c['noise_m']]
    if v is not None and v>t:reason='inherited_high_outlier_not_demoted'
    elif not a['highest_support_observable']:reason='highest_direct_support_insufficient'
    elif np.sum(visible)<3:reason='highest_pixel_presence_or_reliability_missed'
    elif not relevant:reason='highest_component_spatial_or_coherence_filter'
    elif not accepted:reason='highest_candidate_confidence_or_correction_rejected'
    else:reason='accepted_candidate_underestimates_true_apex'
    results.append(dict(frame=r['frame'],bits=r['bits'],error_mm=abs(v-t)*1000 if v is not None else None,reason=reason,
        highest_direct_points=int(near.sum()),highest_detected_direct_points=int(visible.sum()),relevant_candidate_count=len(relevant),accepted_relevant_count=len(accepted),
        candidates=[{k:c[k] for k in ('height_m','height_raw_m','pixels','bbox','object_probability','top_probability','correction_m','noise_m')} for c in sorted(relevant,key=lambda c:c['height_m'],reverse=True)[:3]]))
out=dict(note='post-test diagnosis only; no model, policy or test-set modifications',counts=dict(Counter(r['reason'] for r in results)),records=results)
(ROOT/'failure-diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['counts']),flush=True)
