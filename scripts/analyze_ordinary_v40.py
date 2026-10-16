import json,hashlib
from pathlib import Path
import numpy as np
from liquid_depth.device_profiles_v38 import metric_summary

BASE=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation')
NAMES=['test','outliers','rho05','rho0']
SUFFIX='20261012-final'


def collect(name):
    folder=BASE/f'ordinary-v40-{name}-{SUFFIX}'
    rows=[json.loads(x) for x in (folder/'records.jsonl').read_text().splitlines()]
    summary=json.loads((folder/'summary.json').read_text())
    result={'complete':json.loads((folder/'complete.json').read_text()),'groups':summary}
    result['diagnostics']={}
    for regime in ('nominal','adverse'):
        rr=[r for r in rows if r['method']=='paired' and r['regime']==regime]
        qq=[r for r in rows if r['method']=='dense' and r['regime']==regime]
        def percentiles(values):return dict(zip(['min','p05','median','p95','max'],map(float,np.percentile(values,[0,5,50,95,100]))))
        result['diagnostics'][regime]=dict(
            true_surface_points=percentiles([r['true_surface_points'] for r in rr]),
            true_surface_fraction=percentiles([r['true_surface_fraction'] for r in rr]),
            true_surface_blocks16=percentiles([r['true_surface_blocks16'] for r in rr]),
            raw_within5mm_pct=percentiles([100*r['points_within_5mm_true_plane']/r['true_surface_points'] for r in rr]),
            raw_within20mm_pct=percentiles([100*r['points_within_20mm_true_plane']/r['true_surface_points'] for r in rr]),
            raw_residual_p50_mm=percentiles([r['genuine_raw_plane_residual_p50_mm'] for r in rr]),
            denoised_residual_p50_mm=percentiles([r['genuine_denoised_plane_residual_p50_mm'] for r in rr]),
            frames_with_over10_points=sum(r['raw_over10_points']>0 for r in rr),
            spatial_transfer=percentiles([r['spatial_transfer'] for r in rr]),
            raw_free_angle_deg=percentiles([r['free_plane_angle_deg'] for r in qq]),
            corrected_free_angle_deg=percentiles([r['free_plane_angle_deg'] for r in rr]),
            raw_free_height_mae_mm=float(np.mean([abs(r['free_plane_height_m']-r['true_height_m'])*1000 for r in qq])),
            corrected_free_height_mae_mm=float(np.mean([abs(r['free_plane_height_m']-r['true_height_m'])*1000 for r in rr])),
            negative_empty_outputs=sum(r['empty'] and r['predicted_height_m']<0 for r in rr),
            outputs_over5mm=sum(abs(r['predicted_height_m']-r['true_height_m'])>.005 for r in rr),
            outputs_over10mm=sum(abs(r['predicted_height_m']-r['true_height_m'])>.01 for r in rr),
            outputs_over20mm=sum(abs(r['predicted_height_m']-r['true_height_m'])>.02 for r in rr),
            max_error_mm=max(abs(r['predicted_height_m']-r['true_height_m'])*1000 for r in rr))
        result.setdefault('distance_groups',{})[regime]={}
        for lo,hi in [(0,5.5),(5.5,8),(8,9),(9,20)]:
            result['distance_groups'][regime][f'{lo}-{hi}']={m:metric_summary([r for r in rows if r['regime']==regime and r['method']==m and lo<=r['camera_to_surface_m']<hi]) for m in ['v39b_relaxed_range','dense','denoise','scalar','paired']}
    # Cluster bootstrap: four temporal observations never count as four installs.
    result['paired_cluster_bootstrap']={}
    for regime in ('nominal','adverse'):
        for group in ('far_shallow','uniform'):
            subset=[r for r in rows if r['regime']==regime and (r['population']==group or r['group']==group)]
            clusters=sorted({r['cluster'] for r in subset});diff=[]
            for cl in clusters:
                x=[r for r in subset if r['cluster']==cl]
                e={m:np.mean([abs(r['predicted_height_m']-r['true_height_m'])*1000 for r in x if r['method']==m]) for m in ['v39b_relaxed_range','paired']}
                diff.append(e['paired']-e['v39b_relaxed_range'])
            rng=np.random.default_rng(4007);v=np.array(diff)
            boot=v[rng.integers(0,len(v),(2000,len(v)))].mean(1)
            result['paired_cluster_bootstrap'][regime+'|'+group]=dict(configurations=len(v),mae_change_mm=float(v.mean()),ci95_mm=np.percentile(boot,[2.5,97.5]).tolist())
    hashes=json.loads((folder/'protocol.json').read_text())['source_hashes']
    result['source_hashes_verified']=all(Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (folder/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    result={name:collect(name) for name in NAMES}
    (BASE/f'ordinary-v40-analysis-{SUFFIX}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    for name,value in result.items():
        print(name,value['complete'],value['source_hashes_verified'])
        for key,groups in value['groups'].items():
            if key.endswith('paired') or key.endswith('v39b_relaxed_range'):
                print(key, {g:{k:round(v[k],3) if isinstance(v[k],float) else v[k] for k in ['mae_mm','p95_mm','pass_pct','numeric_output_pct']} for g,v in groups.items() if g in ['far_shallow','uniform','near_zero','empty']})
