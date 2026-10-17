"""Recount dual-only scene cells and far-shallow diagnostics; no depth-bin tables."""
import gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from dual_camera_protocol_v2 import SENSORS,SCENES,protocol,SEED

BASE=Path('/root/autodl-tmp/liquid-depth-artifacts')
OUT=BASE/'dual-camera-v2-20261017'
REPO=Path(__file__).resolve().parents[1]


def tolerance(row,peak=False):
    h=row['truth']['highest_height_m' if peak else 'liquid_height_m']
    return max(.05,.05*h) if peak or row['scene']=='uneven_surface' else max(.02,.02*h)


def metrics(rows,peak=False):
    key='highest_height_m' if peak else 'liquid_depth_m';truth='highest_height_m' if peak else 'liquid_height_m'
    n=len(rows);good=[r for r in rows if r.get(key) is not None and np.isfinite(r[key]) and truth in r['truth']]
    e=np.array([abs(r[key]-r['truth'][truth]) for r in good]);rel=np.array([abs(r[key]-r['truth'][truth])/r['truth'][truth] for r in good if r['truth'][truth]>0])
    passed=sum(abs(r[key]-r['truth'][truth])<=tolerance(r,peak) for r in good)
    m=dict(n=n,installations=len({r['cluster'] for r in rows}),numeric=len(good),no_output=n-len(good),
        numeric_coverage_pct=100*len(good)/n if n else None,pass_pct=100*passed/n if n else None,
        mae_mm=float(e.mean()*1000) if len(e) else None,p95_mm=float(np.percentile(e,95)*1000) if len(e) else None,
        max_mm=float(e.max()*1000) if len(e) else None,absrel_pct=float(rel.mean()*100) if len(rel) else None,
        relative_n=len(rel),absolute_bins_pct=(100*np.histogram(e,[0,.005,.02,.05,.1,np.inf])[0]/n).tolist() if n else [],
        relative_bins_pct=(100*np.histogram(rel,[0,.01,.02,.05,.1,np.inf])[0]/len(rel)).tolist() if len(rel) else [],
        signed_bias_mm=float(np.mean([r[key]-r['truth'][truth] for r in good])*1000) if good else None,
        no_output_reasons=dict(Counter(r.get('inference_failure') or 'no_numeric_evidence' for r in rows
            if r.get(key) is None or not np.isfinite(r[key]) or truth not in r['truth'])))
    # Cluster bootstrap within cohort to preserve the old/new acquisition mixture.
    draws=np.random.default_rng(SEED);sums=np.zeros((1000,4))
    cohorts=defaultdict(lambda:defaultdict(list))
    for r in rows:cohorts[r['cohort']][r['cluster']].append(r)
    for clusters in cohorts.values():
        matrix=[]
        for q in clusters.values():
            g=[r for r in q if r.get(key) is not None and np.isfinite(r[key]) and truth in r['truth']]
            matrix.append([len(q),len(g),sum(abs(r[key]-r['truth'][truth]) for r in g),sum(abs(r[key]-r['truth'][truth])<=tolerance(r,peak) for r in g)])
        a=np.asarray(matrix,float);sums+=a[draws.integers(0,len(a),(1000,len(a)))].sum(axis=1)
    if n:
        m['cluster_bootstrap_95ci']={'mae_mm':np.percentile(1000*sums[:,2]/np.maximum(sums[:,1],1),[2.5,97.5]).tolist(),
            'pass_pct':np.percentile(100*sums[:,3]/sums[:,0],[2.5,97.5]).tolist()}
    times=[r['algorithm_ms'] for r in rows if r.get('algorithm_ms') is not None]
    m['latency_ms']={'timed_frames':len(times),'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':max(times)} if times else None
    return m


def main():
    rows=[];old=REPO/'reports/evidence/current_predictions_20261016.jsonl.gz'
    with gzip.open(old,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['sensor'] not in SENSORS or r['scene'].startswith('homogeneous_'):continue
            r['cluster']=r['cohort']+':'+str(r.get('case',r.get('config_id')))
            if r['scene']=='partial_depth_loss':r['level']=r['loss_bin']
            rows.append(r)
    old_n=len(rows)
    for p in sorted((OUT/'far').glob('records-worker-*.jsonl')):rows += [json.loads(s) for s in p.read_text().splitlines()]
    far_n=len(rows)-old_n
    phase=OUT/'phase-final/records.jsonl';rows += [json.loads(s) for s in phase.read_text().splitlines()]
    assert all(r['sensor'] in SENSORS for r in rows)
    assert not any(r.get('liquid_visual_guidance_used',False) for r in rows)
    groups={};quota={};failures={}
    for scene in SCENES:
        q=[r for r in rows if r['scene']==scene]
        if scene in ('low_light','glare'):levels=['mild','moderate','severe']
        elif scene=='partial_depth_loss':levels=['0-10%','10-20%','20-30%','30-40%','40-50%']
        elif scene=='homogeneous_translucent':levels=['low_extinction','medium_extinction']
        else:levels=[]
        for sensor in ('all',*SENSORS):
            ss=[r for r in q if sensor=='all' or r['sensor']==sensor]
            for level in ('all',*levels):
                selected=[r for r in ss if level=='all' or r['level']==level]
                key='|'.join((scene,sensor,level,'all'))
                groups[key]={'liquid':metrics(selected)}
                if scene in ('floating_objects','uneven_surface'):groups[key]['highest']=metrics(selected,True)
                if sensor!='all' and (level!='all' or not levels):quota[key]=len(selected)
            far=[r for r in ss if r['camera_to_bottom_m']>=6 and 0<r['truth']['liquid_height_m']<=1]
            key='|'.join((scene,sensor,'all','far_shallow'))
            groups[key]={'liquid':metrics(far)}
            if scene in ('floating_objects','uneven_surface'):groups[key]['highest']=metrics(far,True)
            if sensor!='all':
                assert len(far)>=100,(key,len(far))
                failures[scene+'|'+sensor]=sorted([r for r in far if r['liquid_depth_m'] is not None],key=lambda r:abs(r['liquid_depth_m']-r['truth']['liquid_height_m']),reverse=True)[:3]
        assert groups[f'{scene}|tof|all|all']['liquid']['n']>=100
        assert groups[f'{scene}|structured_light|all|all']['liquid']['n']>=100
    assert min(quota.values())>=100,quota
    with gzip.open(OUT/'current-dual-predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    (OUT/'metrics.json').write_text(json.dumps(groups,indent=2,allow_nan=False))
    (OUT/'far-failure-cases.json').write_text(json.dumps(failures,indent=2,allow_nan=False))
    attempts=[]
    for p in (OUT/'far').glob('capture-attempts-worker-*.json'):attempts+=json.loads(p.read_text())
    phase_cals=[]
    for p in (OUT/'phase-final').glob('phase_*.json'):phase_cals+=json.loads(p.read_text())['calibrations']
    frozen=json.loads((OUT/'final-holdout-source-lock.json').read_text())
    assert all(hashlib.sha256((REPO/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
    audit=dict(protocol=protocol(),old_retained_frames=old_n,new_far_frames=far_n,new_phase_frames=len(rows)-old_n-far_n,total_frames=len(rows),
        report_cells=quota,all_report_cells_at_least_100=True,minimum_cell=min(quota.values()),
        cameras=sorted({r['sensor'] for r in rows}),far_input_attempted_pairs=len(attempts),far_selected_pairs=sum(r['paired_eligible'] for r in attempts),
        far_rejected_pairs=sum(not r['paired_eligible'] for r in attempts),
        phase_calibrations=len(phase_cals),phase_calibration_failures=sum(not c['success'] for c in phase_cals),
        phase_calibration_median_ms=float(np.median([c['calibration_ms'] for c in phase_cals])),
        phase_search_boundary_hits=sum(r['diagnostics'].get('constraint_boundary_reached',False) for r in rows if r['cohort']=='new_dual_raw_phase_holdout_v2'),
        holdout_source_lock_verified=True,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old,phase,*(OUT/'far').glob('records-worker-*.jsonl')]})
    audit['pre_final_development_note']='initial phase-holdout used to find a structured-light scattering phase-definition defect; archived as development evidence, excluded; final phase-final uses seed+904327 with corrected physical phase midpoint'
    (OUT/'evaluation-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('protocol','report_cells','source_sha256')},indent=2))
    for scene in SCENES:
        for sensor in SENSORS:
            m=groups[f'{scene}|{sensor}|all|all']['liquid'];far=groups[f'{scene}|{sensor}|all|far_shallow']['liquid']
            print(scene,sensor,'n',m['n'],'mae',round(m['mae_mm'],2),'pass',round(m['pass_pct'],2),'far',far['n'],round(far['mae_mm'],2),round(far['pass_pct'],2))


if __name__=='__main__':main()
