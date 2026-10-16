"""Select residual-bias models on development only, after choosing five levels."""
import json
import cv2
from liquid_depth import transparent_refinement_v27 as v27
from experiment_transparent_v27 import OUT,DATA,POLICY26,load,predict26,rows,summary,score,dump,progress

def candidates(family,condition):
    yield dict(name='v26',baseline=True)
    old=POLICY26[family+'/'+condition]
    if family=='active_stereo':
        yield dict(name='periodic3',harmonics=3,period_px=1.,ridge=.1,spatial=False,fit_y=False)
        for ridge in (.02,.2):
            yield dict(name='spatial_'+str(ridge),harmonics=old['harmonics'],period_px=1.,ridge=ridge,spatial=True,fit_y=False)
    else:
        for ridge in (.02,.2):
            yield dict(name='phase_'+str(ridge),harmonics=1,period_px=16.,ridge=ridge,spatial=False,fit_y=True)
            yield dict(name='field_'+str(ridge),harmonics=0,period_px=16.,ridge=ridge,spatial=True,fit_y=True)
        yield dict(name='phase_field',harmonics=1,period_px=16.,ridge=.1,spatial=True,fit_y=True)

def predict(c,layout,setting):
    if setting.get('baseline'):return predict26(c,layout)
    obs,cal,hs=load(c,layout);old=POLICY26[c['family']+'/'+c['condition']]
    correction=old['correction'] if c['family']=='active_stereo' else dict(kind=old['baseline_mode'])
    try:
        prior=v27.fit_prior(obs[:5],hs,cal,c['condition'],setting)
        return [v27.infer(o,prior,cal,correction) for o in obs[5:]],prior
    except ValueError as exc:
        return [dict(height_m=None,reasons=['prior_failed',str(exc)],qualified=False)]*len(c['tests']),dict(error=str(exc))

def main():
    configs=json.loads((DATA/'development/manifest.json').read_text())['configs']
    layouts=json.loads((OUT/'layout-policy.json').read_text())['selection']
    choices=[];selection={};all_records=[]
    for family in ('active_stereo','structured_light'):
        for condition in ('transparent','translucent_mild'):
            key=family+'/'+condition;cs=[c for c in configs if c['family']==family and c['condition']==condition]
            metrics=[]
            for setting in candidates(family,condition):
                rs=[]
                for c in cs:
                    preds,_=predict(c,layouts[key],setting);rs+=rows(c,preds,setting['name'],layouts[key])
                m=dict(setting=setting,score=score(rs),overall=summary(rs),
                    far_shallow=summary([r for r in rs if r['bottom_m']>=6 and r['true_height_m']/r['bottom_m']<.18]))
                metrics.append(m);all_records+=rs;print('RESIDUAL',family,condition,m,flush=True)
                progress('select_bias',family=family,condition=condition,candidate=setting['name'])
            best=min(metrics,key=lambda m:m['score']);selection[key]=best['setting']
            choices.append(dict(family=family,condition=condition,selected=best,candidates=metrics))
    dump(OUT/'bias-candidates.json',choices);dump(OUT/'bias-development-raw.json',all_records)
    dump(OUT/'frozen-policy.json',dict(layouts=layouts,bias=selection,prior_budget=5,test_used_for_selection=False,
        confidence_gate=False,coverage_gate=False,strong_scatter_included=False))

if __name__=='__main__':
    cv2.setNumThreads(4);main()
