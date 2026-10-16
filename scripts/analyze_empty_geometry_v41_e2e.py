"""Audit E2E records without modifying estimators, gating or measurements."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from liquid_depth.device_profiles_v38 import metric_summary


def cluster_ci(rows,seed=41001):
    groups=defaultdict(list)
    for r in rows:groups[r['cluster']].append(r)
    aggregates=[]
    for rr in groups.values():
        ee=[abs(r['predicted_height_m']-r['true_height_m']) for r in rr if r['predicted_height_m'] is not None]
        passed=sum(r['predicted_height_m'] is not None and abs(r['predicted_height_m']-r['true_height_m'])<=max(.005,.02*r['true_height_m']) for r in rr)
        aggregates.append([sum(ee)*1000,len(ee),passed,len(rr)])
    a=np.asarray(aggregates,float);rng=np.random.default_rng(seed)
    if len(a)<2:return None
    sample=a[rng.integers(0,len(a),(2000,len(a)))].sum(1)
    good=sample[:,1]>0
    return dict(unit='installation_cluster',bootstrap_replicates=2000,
        mae_mm_95ci=np.percentile(sample[good,0]/sample[good,1],[2.5,97.5]).tolist(),
        pass_pct_95ci=np.percentile(100*sample[:,2]/sample[:,3],[2.5,97.5]).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    protocol=json.loads((a.run/'protocol.json').read_text())
    hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in protocol['source_hashes'].items()}
    if not all(hashes.values()):raise AssertionError('evaluation_source_changed_since_run')
    rows=[json.loads(x) for x in (a.run/'records.jsonl').read_text().splitlines()]
    summary=json.loads((a.run/'summary.json').read_text());audit=json.loads((a.run/'calibration-audit.json').read_text())
    report=dict(source_hash_match=hashes,configurations=protocol['configurations'],total_method_records=len(rows),summary=summary,
        primary_groups={},paired_comparison={},same_container={},calibration_failures={},tail_examples=[])
    for regime in ('nominal','adverse'):
      for method in ('ideal_geometry_v40','empty_span','empty_distance'):
        key=regime+'|'+method
        rr=[r for r in rows if r['regime']==regime and r['method']==method]
        report['primary_groups'][key]={}
        for group in ('uniform','far_shallow','near_zero','empty','fixed_sweep'):
            subset=[r for r in rr if r['population']==group]
            good=[r for r in subset if r['predicted_height_m'] is not None]
            positive=[r for r in good if r['true_height_m']>0]
            rel=np.asarray([abs(r['predicted_height_m']-r['true_height_m'])/r['true_height_m']*100 for r in positive])
            report['primary_groups'][key][group]=dict(
                relative_cumulative_pct_of_positive_outputs={str(t):float(np.mean(rel<=t)*100) if len(rel) else None for t in (.5,1,2,3,5,10)},
                ci=cluster_ci(subset),positive_output_count=len(positive))
        matched=defaultdict(dict)
        for r in [r for r in rows if r['regime']==regime and r['population']=='uniform']:
            matched[(r['case'],r['level_index'],r['repeat'])][r['method']]=r
        both=[x for x in matched.values() if all(x[m]['predicted_height_m'] is not None for m in ('empty_span','empty_distance'))]
        if method=='empty_distance':
            errs={m:np.asarray([abs(x[m]['predicted_height_m']-x[m]['true_height_m'])*1000 for x in both]) for m in ('empty_span','empty_distance')}
            report['paired_comparison'][regime]=dict(joint_output_frames=len(both),
                span_mae_mm=float(errs['empty_span'].mean()) if both else None,
                distance_mae_mm=float(errs['empty_distance'].mean()) if both else None,
                distance_lower_error_pct=float(np.mean(errs['empty_distance']<errs['empty_span'])*100) if both else None)
        failed=[x for x in audit if x['method']==method and x['regime']==regime and not x['initialized']]
        report['calibration_failures'][key]=dict(failed_installations=len(failed),reasons=dict(Counter(x['error'] for x in failed)))
        if method=='ideal_geometry_v40':continue
        ar={x['case']:x for x in audit if x['method']==method and x['regime']==regime and x['initialized']}
        groups=defaultdict(list)
        for r in rr:
            if r['population']=='fixed_sweep' and r['predicted_height_m'] is not None:groups[(r['case'],r['level_index'])].append(r)
        observed=[];proxy=[];details=[]
        for (case,level),values in groups.items():
            h=values[0]['true_height_m'];bottom=values[0]['camera_to_surface_m']+h
            delta=ar[case]['bottom_distance_error_mm']
            # First-order scalar geometry error, ignoring tilt/ROI changes.
            geometric_proxy=(2*h/bottom-(h/bottom)**2)*delta
            err=float(np.mean([v['predicted_height_m']-h for v in values])*1000)
            observed.append(err);proxy.append(geometric_proxy)
            details.append(dict(case=case,height_m=h,signed_mean_error_mm=err,bottom_anchor_error_mm=delta,
                normal_error_deg=ar[case]['normal_error_deg'],first_order_anchor_error_proxy_mm=geometric_proxy))
        report['same_container'][key]=dict(level_groups=len(details),
            anchor_proxy_error_correlation=float(np.corrcoef(observed,proxy)[0,1]) if len(details)>2 else None,
            first_order_proxy_ignores_normal_and_roi_errors=True,levels=details)
    tails=[r for r in rows if r['method']=='empty_distance' and r['predicted_height_m'] is not None]
    tails.sort(key=lambda r:abs(r['predicted_height_m']-r['true_height_m']),reverse=True)
    report['tail_examples']=[dict(**r,absolute_error_mm=abs(r['predicted_height_m']-r['true_height_m'])*1000) for r in tails[:30]]
    (a.run/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('AUDIT_OK',len(rows),'source hashes verified')
    for key,ss in summary.items():
        if 'ideal_geometry' in key:continue
        for g in ('uniform','far_shallow','near_zero','empty','fixed_sweep'):
            v=ss[g]
            print(key,g,{k:round(v[k],4) if isinstance(v[k],float) else v[k] for k in ('frames','outputs','mae_mm','p95_mm','pass_pct','numeric_output_pct','absrel_positive_pct')})
    print('PAIRED',report['paired_comparison'])
    print('INIT_FAILURES',report['calibration_failures'])

if __name__=='__main__':main()
