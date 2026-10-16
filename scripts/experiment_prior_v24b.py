"""Calibration development and untouched-seed validation with matched inputs."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from liquid_depth.refraction_images_v23 import render_capture
from liquid_depth.refraction_matching_v23 import stereo_matches,decode_structured
from liquid_depth import prior_reliability_v24 as new
from liquid_depth import refraction_inverse_v23 as old
from experiment_image_refraction_v23 import summarize,gate

BASE=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/prior-reliability-v24b-20261011')
OLD=Path('/root/autodl-tmp/liquid-depth-data/synthetic/image-refraction-v23-final-20261011')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/prior-reliability-v24b-20261011')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def observation(c):
    with np.load(Path(c['folder'])/'matches_optimized.npz') as f:obs={k:f[k] for k in f.files}
    obs['regions']=len(obs['weights']);obs['visibility']=0.
    return obs

def evaluate(configs,methods,output):
    records=[];fits=[];started=time.time()
    for ci,c in enumerate(configs):
        captures=c['captures'];obs=[observation(x) for x in captures]
        cal=json.loads((Path(captures[0]['folder'])/'calibration.json').read_text())
        for method in methods:
            kind,count=method.rsplit('_',1);count=int(count);ids=[0,2,4] if count==3 else list(range(5))
            hs=[captures[i]['known_liquid_height_m'] for i in ids]
            try:
                p=(old.fit_prior([obs[i] for i in ids],hs,cal) if kind=='v23'
                   else new.fit_prior([obs[i] for i in ids],hs,cal,mode=kind))
                fits.append(dict(config=c['id'],method=method,accepted=True,parameters=p))
            except ValueError as exc:
                p=None;fits.append(dict(config=c['id'],method=method,accepted=False,reason=str(exc)))
            for j,capture in enumerate(captures[5:],5):
                prediction=((old.infer(obs[j],p,cal) if kind=='v23' else new.infer(obs[j],p,cal)) if p is not None
                            else dict(height_m=None,observable=False,uncertainty_m=None,reasons=['material_prior_failed']))
                # The unknown liquid height is read only AFTER fitting/prediction.
                with np.load(capture['evaluator_truth']) as truth:h=float(truth['height_m'])
                predicted=prediction['height_m']
                records.append(dict(config_id=c['id'],sample_id=j,family=c['family'],condition=c['condition'],
                    bottom_m=cal['bottom_m'],method=method,true_height_m=h,predicted_height_m=predicted,
                    observable=prediction['observable'],accepted=prediction['observable'],diagnostics=prediction,
                    absolute_error_m=abs(predicted-h) if predicted is not None else None,
                    tolerance_m=max(.005,.02*h),predicted_tolerance_m=max(.005,.02*max(predicted or 0,0))))
        print('EVALUATED',ci+1,len(configs),round(time.time()-started,1),flush=True)
    dump(output,dict(records=records,fits=fits))
    return records

def develop():
    BASE.mkdir(exist_ok=True)
    configs=[c for c in json.loads((OLD/'manifest.json').read_text())['configs'] if c['split'] in ('development','test')]
    methods=['v23_3','v23_5','affine_3','affine_5','piecewise_3','piecewise_5','pchip_3','pchip_5']
    records=evaluate(configs,methods,BASE/'development.json')
    settings={};summary=[]
    for family in ('active_stereo','structured_light'):
        available=[]
        for method in methods:
            rs=[r for r in records if r['family']==family and r['method']==method]
            summary.append(dict(family=family,method=method,gate='none',**summarize(rs)))
            candidates=[]
            for multiplier in (.25,.5,1,2,4,8,16,1e6):
                metrics=summarize([dict(r,accepted=gate(r,multiplier)) for r in rs])
                candidates.append(dict(multiplier=multiplier,**metrics))
            feasible=[r for r in candidates if r['outputs'] and r['pass_pct']>=90 and r['absrel_pct']<=3 and r['inaccurate_acceptance_pct']<=5]
            chosen=max(feasible,key=lambda r:(r['effective_pass_pct'],r['coverage_pct'])) if feasible else None
            settings[family+'/'+method]=dict(multiplier=chosen['multiplier'] if chosen else None,development_metrics=chosen,candidates=candidates)
            if chosen and not method.startswith('v23'):available.append(dict(method=method,**chosen))
        selected=max(available,key=lambda r:(r['effective_pass_pct'],r['coverage_pct'],-r['absrel_pct'])) if available else None
        settings[family+'/selected']=selected
    # Preserve the primary baseline and select extra output risk separately.
    for family in ('active_stereo','structured_light'):
        settings[family+'/routes']={}
        primary_multiplier=2 if family=='active_stereo' else 16
        for count in (3,5):
            options=['v23_3']  # Preserve the previous accepted three-frame path.
            def primary_score(method):
                metric=summarize([dict(r,accepted=gate(r,primary_multiplier)) for r in records if r['family']==family and r['method']==method])
                return metric['effective_pass_pct'],metric['coverage_pct']
            primary_method=max(options,key=primary_score)
            primary={(r['config_id'],r['sample_id']):r for r in records if r['family']==family and r['method']==primary_method}
            candidates=[]
            for mode in ('affine','piecewise','pchip','v23'):
                method=f'{mode}_{count}'
                for multiplier in (.25,.5,1,2,4,8,16):
                    rows=[];extra=[]
                    for r in records:
                        if r['family']!=family or r['method']!=method:continue
                        old=primary[r['config_id'],r['sample_id']]
                        selected=new.route(old['diagnostics'],r['diagnostics'],primary_multiplier,multiplier)
                        row=dict(old if selected['source']=='v23_preserved' else r,accepted=selected['accepted'])
                        rows.append(row)
                        if selected['source']!='v23_preserved':extra.append(row)
                    metrics=summarize(rows);added=summarize(extra)
                    valid=(metrics['outputs'] and metrics['pass_pct']>=90 and metrics['absrel_pct']<=3
                        and metrics['inaccurate_acceptance_pct']<=5 and added['outputs']>=3
                        and added['pass_pct']>=90 and added['absrel_pct']<=3)
                    candidates.append(dict(method=method,multiplier=multiplier,valid=bool(valid),overall=metrics,added=added))
            feasible=[c for c in candidates if c['valid']]
            chosen=max(feasible,key=lambda c:(c['overall']['effective_pass_pct'],c['overall']['coverage_pct'],-c['overall']['absrel_pct'])) if feasible else None
            settings[family+'/routes'][str(count)]=dict(selected=chosen,candidates=candidates,primary_multiplier=primary_multiplier,primary_method=primary_method)
    dump(BASE/'development-summary.json',summary);dump(BASE/'frozen-calibration-policy.json',settings)
    for r in summary:print(r['family'],r['method'],round(r['absrel_pct'] or 0,3),round(r['pass_pct'] or 0,2),round(r['coverage_pct'],2),flush=True)
    for family in ('active_stereo','structured_light'):print('SELECTED',family,settings[family+'/selected'],flush=True)
    for family in ('active_stereo','structured_light'):print('ROUTES',family,{k:v['selected'] for k,v in settings[family+'/routes'].items()},flush=True)

def generate():
    DATA.mkdir(exist_ok=False);configs=[]
    pairs=[(b,n) for b in (.3,.8,2.,6.,9.8) for n in (1.31,1.43)]
    start=time.time()
    for fi,(family,baseline) in enumerate([('active_stereo',.055),('structured_light',.035)]):
        for ci,(bottom,n) in enumerate(pairs):
            seed=245801+ci*173+fi*10000
            for condition in ('transparent','translucent_mild','translucent_strong'):
                folder=DATA/f'{family}_{ci:02d}_{condition}';folder.mkdir();captures=[]
                for j,(role,ratio) in enumerate([('prior',v) for v in (.10,.22,.35,.50,.65)]+[('test',v) for v in (.08,.15,.28,.42,.58,.72)]):
                    sample=folder/f'{j:02d}';sample.mkdir()
                    images,cal,truth=render_capture(seed,bottom*ratio,n,bottom,baseline,condition,family)
                    for name,image in images.items():
                        if not cv2.imwrite(str(sample/(name+'.png')),image):raise IOError('write failed')
                    dump(sample/'calibration.json',cal);np.savez_compressed(sample/'evaluator_truth.npz',**truth)
                    obs=stereo_matches(images['left'],images['right'],cal,True) if family=='active_stereo' else decode_structured(images,cal,True)
                    np.savez_compressed(sample/'matches_optimized.npz',**{k:obs[k] for k in ('left_q','right_q','weights','left_pixels','right_pixels')})
                    capture=dict(folder=str(sample),role=role,evaluator_truth=str(sample/'evaluator_truth.npz'))
                    if role=='prior':
                        sigma=.0005+.001*bottom*ratio
                        capture['known_liquid_height_m']=float(bottom*ratio+np.random.default_rng(seed+j*65537).normal(0,sigma))
                    captures.append(capture)
                configs.append(dict(id=folder.name,family=family,condition=condition,split='test',camera_to_bottom_m=bottom,captures=captures))
                print('GENERATED',len(configs),60,round(time.time()-start,1),flush=True)
    dump(DATA/'manifest.json',dict(configs=configs,test_seed_family='245801+173*i+10000*family',shared_forward_inverse_assumptions=True))

def test():
    settings=json.loads((BASE/'frozen-calibration-policy.json').read_text())
    configs=json.loads((DATA/'manifest.json').read_text())['configs']
    methods=['v23_3','v23_5','affine_3','affine_5','piecewise_3','piecewise_5','pchip_3','pchip_5']
    records=evaluate(configs,methods,BASE/'test-raw.json')
    routes=[]
    for family in ('active_stereo','structured_light'):
        for count in (3,5):
            policy=settings[family+'/routes'][str(count)];chosen=policy['selected']
            lookup={(r['config_id'],r['sample_id'],r['method']):r for r in records if r['family']==family}
            for r in records:
                if r['family']!=family or r['method']!=policy['primary_method']:continue
                fallback=lookup[(r['config_id'],r['sample_id'],chosen['method'])] if chosen else r
                selected=new.route(r['diagnostics'],fallback['diagnostics'],policy['primary_multiplier'],chosen['multiplier'] if chosen else None)
                row=dict(r if selected['source']=='v23_preserved' else fallback,method=f'route_{count}',accepted=selected['accepted'],source=selected['source'])
                routes.append(row)
    dump(BASE/'routed-test-raw.json',routes)
    groups=defaultdict(list);ranges=defaultdict(list)
    old_gates=json.loads(Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/image-refraction-v23-20261011/final/evaluation/frozen-gates.json').read_text())
    for r in records:
        multiplier=(old_gates[r['family']]['selected_multiplier'] if r['method'].startswith('v23')
                    else settings[r['family']+'/'+r['method']]['multiplier'])
        for name,row in [('observable',r),('gated',dict(r,accepted=gate(r,multiplier)))]:
            key=(r['family'],r['condition'],r['method'],name)
            groups[key].append(row);ranges[key+(r['bottom_m'],)].append(row)
    for r in routes:
        key=(r['family'],r['condition'],r['method'],'gated')
        groups[key].append(r);ranges[key+(r['bottom_m'],)].append(r)
    dump(BASE/'test-summary.json',dict(summary=[dict(family=f,condition=c,method=m,gate=g,**summarize(rs)) for (f,c,m,g),rs in groups.items()],
        by_range=[dict(family=f,condition=c,method=m,gate=g,bottom_m=b,**summarize(rs)) for (f,c,m,g,b),rs in ranges.items()],
        test_used_for_selection=False))
    print('TEST COMPLETE',flush=True)

if __name__=='__main__':
    cv2.setNumThreads(4)
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['develop','generate','test']);args=ap.parse_args()
    dict(develop=develop,generate=generate,test=test)[args.command]()
