import json,math
from pathlib import Path
import numpy as np

ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020')
assert all((ROOT/'independent-test'/f'complete-{i}.json').exists() for i in range(4))
rows=[json.loads(s) for i in range(4) for s in (ROOT/'independent-test'/f'records-{i}.jsonl').read_text().splitlines()]
assert len(rows)==1000 and len({r['frame'] for r in rows})==1000
names={0:'普通无漂浮物对照',1:'低照无漂浮物对照',2:'眩光无漂浮物对照',4:'不平整无漂浮物对照',16:'深度丢失无漂浮物对照',8:'小漂浮物',9:'低照+小漂浮物',10:'眩光+小漂浮物',12:'不平整+小漂浮物',24:'深度丢失+小漂浮物'}

def eligible(r):
    a=r['input_audit']
    return a['online']['eligible'] and a['origin']['eligible'] and (not r['bits']&16 or a['liquid_missing_fraction']<=.5+1/max(a['surface_pixels'],1))

def metrics(rs,key):
    errors=[];relative=[];passed=0
    for r in rs:
        v=r[key];t=r['truth']['highest_height_m']
        if v is None or not math.isfinite(v):continue
        e=abs(v-t);errors.append(e);relative.append(e/max(abs(t),1e-9));passed+=e<=max(.05,.05*abs(t))
    a=np.array(errors);rel=np.array(relative)
    return dict(n=len(rs),outputs=len(a),coverage=len(a)/len(rs) if rs else None,mae_mm=float(a.mean()*1000) if len(a) else None,
        p95_mm=float(np.percentile(a,95)*1000) if len(a) else None,max_mm=float(a.max()*1000) if len(a) else None,
        pass_rate=passed/len(rs) if rs else None,abs_bins={label:int(np.sum((a>lo)&(a<=hi))) for label,lo,hi in [('0-5mm',-1,.005),('5-20mm',.005,.02),('20-50mm',.02,.05),('50-100mm',.05,.1),('100-500mm',.1,.5),('>500mm',.5,np.inf)]},
        relative_bins={label:int(np.sum((rel>lo)&(rel<=hi))) for label,lo,hi in [('0-1%',-1,.01),('1-2%',.01,.02),('2-5%',.02,.05),('5-10%',.05,.1),('>10%',.1,np.inf)]})

def pair(rs):return dict(old=metrics(rs,'old_peak'),new=metrics(rs,'new_peak'),updates=sum(r['used'] for r in rs),new_activations=sum(r['new_activation'] for r in rs))
primary=[r for r in rows if eligible(r)];floats=[r for r in primary if r['bits']&8];controls=[r for r in primary if not r['bits']&8]
by={str(bits):dict(name=name,generated=sum(r['bits']==bits for r in rows),**pair([r for r in primary if r['bits']==bits])) for bits,name in names.items()}
new_fail=[]
for r in controls:
    t=r['truth']['highest_height_m'];tol=max(.05,.05*abs(t))
    if r['old_peak'] is not None and abs(r['old_peak']-t)<=tol and (r['new_peak'] is None or abs(r['new_peak']-t)>tol):new_fail.append(r['frame'])
checks={}
for bits in (8,9,10,12,24):
    p=by[str(bits)];a=p['old'];b=p['new']
    checks[str(bits)]=bool(a['n'] and b['mae_mm']<=a['mae_mm']+2 and b['p95_mm']<=a['p95_mm']+5 and b['max_mm']<=a['max_mm']+5 and b['pass_rate']>=a['pass_rate']-.02)
fa=pair(floats);material=bool(fa['new']['mae_mm']<=fa['old']['mae_mm']-2 or fa['new']['pass_rate']>=fa['old']['pass_rate']+.02)
liquid_unchanged=all(r['old_liquid']==r['new_liquid'] for r in rows)
latency={}
for k in ('baseline_ms','extra_ms','total_estimated_ms'):
    a=np.array([r[k] for r in rows]);latency[k]=dict(median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))
worst=sorted(primary,key=lambda r:abs(r['new_peak']-r['truth']['highest_height_m']) if r['new_peak'] is not None else np.inf,reverse=True)[:20]
paired=[]
for g in sorted({r['group'] for r in floats}):
    rs=[r for r in floats if r['group']==g and r['old_peak'] is not None and r['new_peak'] is not None]
    if rs:
        old=np.array([abs(r['old_peak']-r['truth']['highest_height_m']) for r in rs]);new=np.array([abs(r['new_peak']-r['truth']['highest_height_m']) for r in rs]);tol=np.array([max(.05,.05*abs(r['truth']['highest_height_m'])) for r in rs])
        paired.append([float(np.sum(new-old)),float(np.sum((new<=tol).astype(int)-(old<=tol).astype(int))),len(rs)])
rng=np.random.default_rng(20261020);boot=[]
if paired:
    p=np.array(paired)
    for _ in range(3000):
        v=p[rng.integers(0,len(p),len(p))].sum(axis=0);boot.append([v[0]/v[2]*1000,v[1]/v[2]*100])
ci=dict(mae_change_mm_95ci=np.percentile(boot,[2.5,97.5],axis=0)[:,0].tolist(),pass_change_percentage_points_95ci=np.percentile(boot,[2.5,97.5],axis=0)[:,1].tolist(),resampling_unit='physical installation',replicates=3000) if boot else None
failure_counts={}
for label,rs in [('all_float',floats),('no_visual_activation',[r for r in floats if not r['baseline_visual']]),('highest_observable',[r for r in floats if r['input_audit']['highest_support_observable']])]:
    failed=[r for r in rs if r['new_peak'] is None or abs(r['new_peak']-r['truth']['highest_height_m'])>max(.05,.05*abs(r['truth']['highest_height_m']))]
    failure_counts[label]=dict(n=len(rs),failures=len(failed),no_dense_candidates=sum(r['candidate_count']==0 for r in failed),no_accepted_candidate=sum(r['selected'] is None for r in failed),underestimates=sum(r['new_peak'] is not None and r['new_peak']<r['truth']['highest_height_m'] for r in failed),overestimates=sum(r['new_peak'] is not None and r['new_peak']>r['truth']['highest_height_m'] for r in failed),highest_not_observable=sum(not r['input_audit']['highest_support_observable'] for r in failed))
out=dict(base_commit='182bbd73aacd5bd1731e5321faf2c427421b4259',selected=json.loads((ROOT/'frozen-test-protocol.json').read_text()),
    generated=len(rows),eligible=len(primary),all_frames=pair(rows),primary_float=fa,primary_controls=pair(controls),by_condition=by,
    highest_observable=pair([r for r in floats if r['input_audit']['highest_support_observable']]),highest_not_observable=pair([r for r in floats if not r['input_audit']['highest_support_observable']]),
    float_visible_area={name:pair([r for r in floats if lo<=r['input_audit']['visible_float_pixels']<hi]) for name,lo,hi in [('0-9px',0,10),('10-49px',10,50),('50-199px',50,200),('>=200px',200,np.inf)]},
    per_float_condition_nonregression=checks,new_control_failures=new_fail,liquid_exactly_unchanged=liquid_unchanged,material_aggregate_improvement=material,latency=latency,installation_clustered_bootstrap=ci,failure_counts=failure_counts,
    numerical_gate_pass=all(checks.values()) and not new_fail and liquid_unchanged and material,
    latency_is_cached_baseline_plus_current_auxiliary_not_direct_e2e=True,
    worst=[{k:r[k] for k in ('frame','bits','old_peak','new_peak','truth','used','baseline_visual','new_activation','candidate_count','selected','input_audit')} for r in worst])
(ROOT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in out.items() if k not in ('worst','selected')},ensure_ascii=False,indent=2))
