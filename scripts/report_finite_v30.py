"""Post-freeze all-frame analysis, no optimization or policy writes."""
import json,time
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
from experiment_finite_v30 import OUT,DATA,images,METHODS,ensure_cache
from experiment_transparent_v27 import summary,dump
from liquid_depth import finite_relative_v30 as new
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from liquid_depth.finite_runtime_v30 import FiniteBottomRelativeMeter


def analyze():
    obj=json.loads((OUT/'test-raw.json').read_text());rs=obj['records']
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection']
    selected=[r for r in rs if r['stage']==policy[r['condition']]]
    base=[r for r in rs if r['stage']=='v29_finite']
    result=dict(policy=policy,methods={m:summary([r for r in rs if r['stage']==m]) for m in METHODS})
    def both(predicate):
        return dict(baseline=summary([r for r in base if predicate(r)]),selected=summary([r for r in selected if predicate(r)]),
            uncalibrated_direct=summary([r for r in rs if r['stage']=='direct' and predicate(r)]))
    result['strata']={'all':both(lambda r:True),'far_small':both(lambda r:r['bottom_m']>=6 and r['empty_width_px']<80)}
    for lo,hi in [(0,.02),(.02,.05),(.05,.1),(.1,1.)]:
        result['strata'][f'H_{lo}_{hi}']=both(lambda r:lo<=r['true_height_m']<hi)
    for condition in policy:result['strata'][condition]=both(lambda r:r['condition']==condition)
    for lo,hi in [(0,32),(32,64),(64,96),(96,1e9)]:
        result['strata'][f'px_{lo}_{hi:g}']=both(lambda r:lo<=r['empty_width_px']<hi)
        for condition in policy:
            result['strata'][condition+f'_px_{lo}_{hi:g}']=both(lambda r:lo<=r['empty_width_px']<hi and r['condition']==condition)
    for b in (2.6,6.6,9.6):
        result['strata'][f'B_{b}']=both(lambda r:r['bottom_m']==b)
        for d in (.4,.8,1.6):
            result['strata'][f'B_{b}_D_{d}']=both(lambda r:r['bottom_m']==b and r['container_width_m']==d)
            for condition in policy:result['strata'][f'B_{b}_D_{d}_{condition}']=both(lambda r:r['bottom_m']==b and r['container_width_m']==d and r['condition']==condition)
    for c in (.12,.32):result['strata'][f'C_{c}']=both(lambda r:r['container_max_height_m']==c)
    failures=[r for r in selected if r['predicted_height_m'] is None or r['absolute_error_m']>r['tolerance_m']]
    result['failures']=dict(frames=len(failures),missing=sum(r['predicted_height_m'] is None for r in selected),
        reasons=dict(Counter(x for r in selected for x in r['diagnostics'].get('reasons',[]))),
        errors_by_mm={f'{lo}_{hi:g}':sum(r['absolute_error_m'] is not None and lo<=r['absolute_error_m']*1000<hi for r in failures)
            for lo,hi in [(5,10),(10,20),(20,50),(50,100),(100,1e9)]},
        strict_1pct=sum(r['absolute_error_m'] is not None and r['absolute_error_m']<=.01*r['true_height_m'] for r in selected))
    # Paired bootstrap clusters by texture seed shared across the two conditions.
    def key(r):return (r['config_id'],r['sample_id'])
    bm={key(r):r for r in base};groups={}
    for r in selected:
        b=bm[key(r)];g='_'.join(r['config_id'].split('_')[:2]);v=groups.setdefault(g,[])
        v.append(int(r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m'])-
            int(b['absolute_error_m'] is not None and b['absolute_error_m']<=b['tolerance_m']))
    values=np.array([np.mean(x) for x in groups.values()]);rng=np.random.default_rng(981113)
    boots=np.mean(rng.choice(values,(4000,len(values)),replace=True),axis=1)*100
    result['paired_all_frame_pass_delta_pct_ci95']=np.percentile(boots,[2.5,97.5]).tolist()
    result['independent_texture_clusters']=len(values)
    direct={key(r):r for r in rs if r['stage']=='direct'};pairs=[(r,direct[key(r)]) for r in selected]
    result['selected_vs_direct']=dict(all_frames=len(pairs),
        common_outputs=sum(a['predicted_height_m'] is not None and b['predicted_height_m'] is not None for a,b in pairs),
        common_mae_delta_mm=float(np.mean([(a['absolute_error_m']-b['absolute_error_m'])*1000 for a,b in pairs
            if a['predicted_height_m'] is not None and b['predicted_height_m'] is not None])),
        improved=sum(a['absolute_error_m'] is not None and b['absolute_error_m'] is not None and a['absolute_error_m']<b['absolute_error_m'] for a,b in pairs))
    paired_groups={}
    for a,b in pairs:
        g='_'.join(a['config_id'].split('_')[:2]);v=paired_groups.setdefault(g,[])
        v.append((int(a['absolute_error_m'] is not None and a['absolute_error_m']<=a['tolerance_m'])-
            int(b['absolute_error_m'] is not None and b['absolute_error_m']<=b['tolerance_m']),
            (a['absolute_error_m']-b['absolute_error_m'])*1000 if a['absolute_error_m'] is not None and b['absolute_error_m'] is not None else np.nan))
    group_means=np.array([np.nanmean(x,axis=0) for x in paired_groups.values()])
    idx=rng.integers(0,len(group_means),size=(4000,len(group_means)))
    boot=np.nanmean(group_means[idx],axis=1)
    result['selected_vs_direct']['pass_delta_pct_ci95']=np.percentile(boot[:,0]*100,[2.5,97.5]).tolist()
    result['selected_vs_direct']['common_mae_delta_mm_ci95']=np.percentile(boot[:,1],[2.5,97.5]).tolist()
    for condition in policy:
        result[condition+'_fit_errors']=dict(Counter(x['prior'].get('error','ok') for f in obj['fits'] for x in f['fits']
            if x['stage']==policy[condition] and f['config_id'].endswith('_'+condition)))
    # Evaluator-only analytic sensitivity. Never enters prediction or selection.
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs'];sensitivity=[]
    for c in cs:
        with np.load(c['tests'][5]['evaluator_truth']) as f:h=float(f['height_m']);index=float(f['index'])
        b=c['camera_to_bottom_m'];radius=c['empty_width_px']/2
        q=np.array([[radius/450.,0.]])
        world=bottom_map(q,h,index,b);qt=inverse_bottom(world,h+.005,index,b)
        pixels=float(np.linalg.norm(qt-q)*450)
        sensitivity.append(dict(config_id=c['id'],bottom_m=b,width_m=c['container_width_m'],
            empty_width_px=c['empty_width_px'],five_mm_edge_motion_px=pixels,
            one_millipixel_height_error_mm=.005/pixels,
            one_centipixel_height_error_mm=.05/pixels))
    result['analytic_sensitivity_evaluator_only']=sensitivity
    dump(OUT/'analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)


def benchmark():
    cv2.setNumThreads(1)
    cs=json.loads((DATA/'test/manifest.json').read_text())['configs'];obj=json.loads((OUT/'test-raw.json').read_text())
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection'];fitmap={f['config_id']:f for f in obj['fits']}
    records=[]
    for condition in policy:
        c=next(c for c in cs if c['condition']==condition and c['camera_to_bottom_m']==9.6 and c['container_width_m']==.4)
        ims,masks,cal,hs=images(c);stage=policy[condition]
        prior=next(x['prior'] for x in fitmap[c['id']]['fits'] if x['stage']==stage)
        kind='direct' if stage.startswith('direct') else 'small_lk'
        fn=new.direct_pair if kind=='direct' else new.small_lk
        if 'error' in prior:continue
        meter=FiniteBottomRelativeMeter([{ch:ims[a,ch] for ch in ('left','right')} for a in range(5)],hs,cal,condition,masks)
        for j in (0,5,11):
            t=time.perf_counter()
            pred=meter.measure({ch:ims[j+5,ch] for ch in ('left','right')})
            elapsed_ms=(time.perf_counter()-t)*1000
            expected=next(r['predicted_height_m'] for r in obj['records'] if r['config_id']==c['id'] and r['stage']==stage and r['sample_id']==j)
            delta=abs(pred['height_m']-expected) if pred['height_m'] is not None and expected is not None else None
            if delta is None or delta>1e-9:raise RuntimeError('runtime_disagrees_with_frozen_test')
            records.append(dict(config_id=c['id'],sample=j,elapsed_ms=elapsed_ms,
                output=pred['height_m'] is not None,empty_width_px=c['empty_width_px'],parity_delta_m=delta))
    dump(OUT/'uncached-processing-smoke.json',dict(records=records,frames=len(records),
        mean_ms=float(np.mean([x['elapsed_ms'] for x in records])) if records else None,
        p95_ms=float(np.percentile([x['elapsed_ms'] for x in records],95)) if records else None,
        acquisition_included=False,display_included=False,prior_fit_offline=True,threads=1,certified=False))


if __name__=='__main__':analyze();benchmark()
