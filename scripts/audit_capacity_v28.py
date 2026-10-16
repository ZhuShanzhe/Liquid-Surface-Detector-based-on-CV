"""Evaluator-only oracle counterfactuals. Never imported by normal prediction."""
import json,copy
from pathlib import Path
import numpy as np
from liquid_depth import transparent_refinement_v26 as v26
from liquid_depth import transparent_diagnostic_v25 as v25
from liquid_depth.refraction_images_v23 import bottom_map,inverse_bottom
from capacity_priors_v28 import OUT,DATA,load,POLICY26,result_rows,summary,dump,progress

def truth(cap):
    with np.load(cap['evaluator_truth']) as f:return float(f['height_m']),float(f['index'])

def oracle_observation(obs,height,index,cal):
    out=copy.deepcopy(obs)
    right=inverse_bottom(bottom_map(obs['left_q'],height,index,cal['bottom_m']),height,index,cal['bottom_m'],(cal['baseline_m'],0.))
    out['right_q']=right
    out['right_pixels']=right*cal['fx']+[cal['reference_cx'],cal['reference_cy']]
    return out

def predictions(c,layout,prior_matches=False,current_matches=False,exact_labels=False):
    obs,cal,labels=load(c,layout);caps=c['priors'][layout]+c['tests']
    if exact_labels:labels=[truth(cap)[0] for cap in caps[:5]]
    for i,cap in enumerate(caps):
        if (i<5 and prior_matches) or (i>=5 and current_matches):
            h,n=truth(cap);obs[i]=oracle_observation(obs[i],h,n,cal)
    setting=POLICY26[c['family']+'/'+c['condition']]
    try:
        if c['family']=='active_stereo':
            prior=v26.fit_prior(obs[:5],labels,cal,c['condition'],setting['harmonics'],setting['ridge'])
            preds=[v26.infer(o,prior,cal,setting['correction']) for o in obs[5:]]
        else:
            prior=v25.fit_prior(obs[:5],labels,cal,c['condition'],setting['baseline_mode'])
            preds=[v25.infer(o,prior,cal) for o in obs[5:]]
        return preds,prior,obs,cal
    except ValueError as exc:
        return [dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False)]*len(c['tests']),dict(error=str(exc)),obs,cal

def main():
    configs=json.loads((DATA/'test/manifest.json').read_text())['configs']
    policy=json.loads((OUT/'frozen-policy.json').read_text())['selection']
    rows=[];fits=[];sensitivity=[]
    for ci,c in enumerate(configs):
        layout=policy[c['family']+'/'+c['condition']]
        for pm,cm,el in [(False,False,False),(False,False,True),(True,False,False),(False,True,False),
                         (True,True,False),(True,False,True),(False,True,True),(True,True,True)]:
            mode=f'prior{int(pm)}_current{int(cm)}_labels{int(el)}'
            preds,p,obs,cal=predictions(c,layout,pm,cm,el)
            rows+=result_rows(c,preds,mode,layout);fits.append(dict(config_id=c['id'],mode=mode,prior=p))
            if not(pm or cm or el):
                for j,cap in enumerate(c['tests']):
                    h,n=truth(cap);o=obs[j+5];exact=oracle_observation(o,h,n,cal)
                    delta=(exact['right_q']-o['right_q'])*cal['fx']
                    step=1e-5
                    derivative=(v25.pixel_residual(h+step,n,0.,exact,cal)-v25.pixel_residual(h-step,n,0.,exact,cal))/(2*step)
                    sens=float(np.sqrt(np.mean(np.sum(derivative**2,axis=1)))) if len(delta) else None
                    sensitivity.append(dict(config_id=c['id'],sample_id=j,bottom_m=c['camera_to_bottom_m'],capacity_m=c['container_max_height_m'],
                        family=c['family'],condition=c['condition'],true_height_m=h,
                        pixel_bias_median_x=float(np.median(delta[:,0])) if len(delta) else None,
                        pixel_error_mad_x=float(np.median(abs(delta[:,0]-np.median(delta[:,0])))) if len(delta) else None,
                        model_sensitivity_px_per_m=sens,height_mm_per_0_001px=1/max(sens,1e-12) if sens is not None else None,
                        true_index=n,effective_index=p.get('index'),prior_fit_error=p.get('error'),
                        prior_label_noise_mm=[(cap_['known_liquid_height_m']-truth(cap_)[0])*1000 for cap_ in c['priors'][layout]]))
        progress('audit',configs=ci+1,total_configs=len(configs),status='running');print('AUDIT',ci+1,len(configs),flush=True)
    modes=sorted(set(r['stage'] for r in rows))
    metrics=[dict(mode=m,overall=summary([r for r in rows if r['stage']==m]),far=summary([r for r in rows if r['stage']==m and r['bottom_m']>=6])) for m in modes]
    dump(OUT/'oracle-raw.json',dict(records=rows,fits=fits));dump(OUT/'oracle-summary.json',metrics)
    dump(OUT/'sensitivity.json',sensitivity)
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__':main()
