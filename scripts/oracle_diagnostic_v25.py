"""Evaluator-only cause isolation; NEVER deployment or claimed model accuracy."""
import json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from liquid_depth import transparent_diagnostic_v25 as v25
from experiment_transparent_v25 import OUT,dump,metrics,observation

data=Path('/root/autodl-tmp/liquid-depth-data/synthetic/transparent-diagnostic-v25-20261011')
configs=json.loads((data/'manifest.json').read_text())['configs']
selected=json.loads((OUT/'frozen-selection.json').read_text())['methods']
records=[]
for c in configs:
    captures=c['captures'];cal=json.loads((Path(captures[0]['folder'])/'calibration.json').read_text())
    measured=[observation(cap) for cap in captures];ideal=[];truth_h=[]
    for cap,obs in zip(captures,measured):
        ps=obs['left_pixels'];prefix='right' if c['family']=='active_stereo' else 'projector'
        with np.load(cap['evaluator_truth']) as truth:
            qs=np.column_stack([cv2.remap(truth[prefix+'_'+axis],ps[:,0].astype(np.float32).reshape(-1,1),
                ps[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR).ravel() for axis in ['u','v']])
            truth_h.append(float(truth['height_m']))
        ideal.append(dict(obs,right_q=(qs-np.array([cal['reference_cx'],cal['reference_cy']]))/cal['fx']))
    mode=selected[c['family']+'/'+c['condition']].split('_')[1]
    for arm,observations,hs in [
        ('oracle_matches_noisy_labels',ideal,[cap['known_liquid_height_m'] for cap in captures[:5]]),
        ('measured_matches_exact_labels',measured,truth_h[:5]),
        ('oracle_matches_exact_labels',ideal,truth_h[:5])]:
        p=v25.fit_prior(observations[:5],hs,cal,c['condition'],mode)
        for j in range(5,len(captures)):
            pred=v25.infer(observations[j],p,cal)['height_m'];h=truth_h[j]
            records.append(dict(config_id=c['id'],sample_id=j,family=c['family'],condition=c['condition'],
                arm=arm,true_height_m=h,predicted_height_m=pred,absolute_error_m=abs(pred-h) if pred is not None else None,
                evaluator_oracle=True,deployable=False))
groups=defaultdict(list)
for r in records:groups[r['family'],r['condition'],r['arm']].append(r)
summary=[dict(family=f,condition=c,arm=a,**metrics(rs)) for (f,c,a),rs in groups.items()]
dump(OUT/'oracle-ablation.json',dict(evaluator_only=True,not_achievable_accuracy=True,records=records,summary=summary))
for r in summary:print(r,flush=True)
