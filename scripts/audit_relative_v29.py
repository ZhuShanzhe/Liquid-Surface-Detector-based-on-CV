"""Evaluator-only counterfactuals. Never imported by normal inference."""
import json
from pathlib import Path
import numpy as np
from liquid_depth import transparent_relative_v29 as rel
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from experiment_relative_v29 import OUT,DATA,read_cache,cache_path,load,rows,dump,summary

MODES={'exact_labels':(False,True),'ideal_tracks':(True,False),'ideal_tracks_and_labels':(True,True)}

def truth(cap):
    with np.load(cap['evaluator_truth']) as f:return float(f['height_m']),float(f['index'])

def ideal_pair(p,ha,hb,index,cal):
    q=p['anchor_q'];x=bottom_map(q,ha,index,cal['bottom_m'])
    return dict(p,current_q=inverse_bottom(x,hb,index,cal['bottom_m']))

def main():
    configs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection'];records=[];fits=[]
    normal=json.loads((OUT/'test-raw.json').read_text())['records']
    actual={(r['config_id'],r['sample_id']):r for r in normal if r['stage']==policy[r['condition']]}
    replay=True
    for i,c in enumerate(configs):
        stage=policy[c['condition']]
        if stage=='wet_uniform' or '/relative_' not in stage:
            continue  # No invented relative oracle for a different selected route.
        layout,method=stage.split('/');_,cal,labels=load(c,layout);cache=read_cache(cache_path(c,layout))
        caps=c['priors'][layout]+c['tests'];gt=[truth(p) for p in caps]
        for mode,(ideal,exact) in dict(normal_replay=(False,False),**MODES).items():
            hs=[x[0] for x in gt[:5]] if exact else labels
            cc={key:ideal_pair(p,gt[key[0]][0],gt[key[1]][0],gt[key[1]][1],cal) if ideal else p for key,p in cache.items()}
            ps=[(a,b,p) for (a,b,ch),p in cc.items() if b<5]
            try:
                prior=rel.fit_relative(ps,hs,cal,c['condition'],'empty' if method=='relative_empty' else 'multi')
                corr=rel.empirical_correction(ps,prior,cal) if method=='relative_corrected' else None
                preds=[rel.infer_relative([(a,p) for (a,b,ch),p in cc.items() if b==j+5],prior,cal,corr) for j in range(len(c['tests']))]
            except ValueError as e:
                prior=dict(error=str(e));preds=[dict(height_m=None,reasons=[str(e)],qualified=False) for _ in c['tests']]
            rs=rows(c,preds,mode,layout)
            if mode=='normal_replay':
                replay &= all(r['predicted_height_m']==actual[r['config_id'],r['sample_id']]['predicted_height_m'] for r in rs)
            else:records+=rs
            fits.append(dict(config_id=c['id'],mode=mode,prior=prior))
        print('AUDIT',i+1,len(configs),flush=True)
    if not replay:raise RuntimeError('normal_replay_mismatch')
    dump(OUT/'oracle-raw.json',dict(records=records,fits=fits))
    result=dict(normal_replay_exact=replay,all_evaluated_frames=len(actual),modes={})
    for mode in MODES:
        rs=[r for r in records if r['stage']==mode]
        result['modes'][mode]=dict(overall=summary(rs),far=summary([r for r in rs if r['bottom_m']>=6]),
            fixes=sum(actual[r['config_id'],r['sample_id']]['absolute_error_m'] is not None and actual[r['config_id'],r['sample_id']]['absolute_error_m']>r['tolerance_m'] and r['absolute_error_m'] is not None and r['absolute_error_m']<=r['tolerance_m'] for r in rs))
    dump(OUT/'oracle-summary.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
