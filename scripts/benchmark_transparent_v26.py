"""Resident image-processing latency, excluding exposures and prior fitting."""
import json,time
from pathlib import Path
import cv2,numpy as np
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth import transparent_refinement_v26 as v26
from liquid_depth.refraction_matching_v23 import stereo_matches,decode_structured as old_decode
from liquid_depth.refraction_matching_v26 import decode_structured
from experiment_transparent_v26 import OUT,V25TEST,setup,BASE_POLICY,dump

cv2.setNumThreads(4)
policy=json.loads((OUT/'frozen-policy.json').read_text())['selection']
configs=[c for c in json.loads(V25TEST.read_text())['configs'] if c['condition'] in v25.TARGET_CONDITIONS and c['camera_to_bottom_m'] in (.3,2.,9.8)]
records=[]
for c in configs:
    key=c['family']+'/'+c['condition'];setting=policy[key]
    old_obs,cal,known=setup(c)
    old_prior=v25.fit_prior(old_obs[:5],known,cal,c['condition'],BASE_POLICY[key].split('_')[1])
    if c['family']=='active_stereo':new_prior=v26.fit_prior(old_obs[:5],known,cal,c['condition'],setting['harmonics'],setting['ridge'])
    else:
        new_obs,_,_=setup(c,setting['matching'])
        new_prior=v25.fit_prior(new_obs[:5],known,cal,c['condition'],setting['baseline_mode'])
    folder=Path(c['captures'][8]['folder'])
    images={p.stem:cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in folder.glob('*.png')}
    for version in ('v25','v26'):
        times=[]
        for iteration in range(6):
            start=time.perf_counter()
            if c['family']=='active_stereo':
                obs=stereo_matches(images['left'],images['right'],cal,True)
                pred=v25.infer(obs,old_prior,cal) if version=='v25' else v26.infer(obs,new_prior,cal,setting['correction'])
            else:
                obs=old_decode(images,cal,True) if version=='v25' else decode_structured(images,cal,0.,24)
                pred=v25.infer(obs,old_prior if version=='v25' else new_prior,cal)
            elapsed=(time.perf_counter()-start)*1000
            if iteration:times.append(elapsed)
        records.append(dict(config_id=c['id'],family=c['family'],condition=c['condition'],version=version,
            mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),times_ms=times))
summary=[]
for family in ('active_stereo','structured_light'):
    for version in ('v25','v26'):
        times=[t for r in records if r['family']==family and r['version']==version for t in r['times_ms']]
        summary.append(dict(family=family,version=version,samples=len(times),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),max_ms=float(max(times))))
result=dict(scope='resident image matching/phase decode plus liquid-height inversion',
    excludes=['exposure acquisition','disk/network transfer','known-level calibration','GUI'],
    structured_light_exposures_per_group=30,records=records,summary=summary)
dump(OUT/'processing-latency.json',result)
print(json.dumps(summary,indent=2))
