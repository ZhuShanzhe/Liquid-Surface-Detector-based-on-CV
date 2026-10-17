"""Select every ToF row from the frozen comparison release, without error filtering."""
import argparse,gzip,hashlib,json
from collections import Counter
from pathlib import Path
from recount_tof import independent_metrics

ROOT=Path(__file__).resolve().parents[1]
SCENES=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent','partial_depth_loss','low_light','glare','uneven_surface','floating_objects')
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    source=a.source/'current-dual-predictions.jsonl.gz'
    with gzip.open(source,'rt',encoding='utf-8') as f:rows=[r for line in f if (r:=json.loads(line))['sensor']=='tof']
    old=json.loads((a.source/'metrics.json').read_text());metrics={k:v for k,v in old.items() if k.split('|')[1]=='tof' and k.split('|')[-1]=='all'}
    for key,quantities in metrics.items():
        sc,_,level,_=key.split('|');group=[r for r in rows if r['scene']==sc and (level=='all' or r['level']==level)]
        for name,fields in quantities.items():
            got=independent_metrics(group,name=='highest')
            for field,value in got.items():
                import numpy as np
                assert (value is None and fields[field] is None) or (value is not None and fields[field] is not None and np.allclose(value,fields[field],atol=1e-10,rtol=1e-12)),(key,name,field)
    quotas={k:v['liquid']['n'] for k,v in metrics.items() if k.split('|')[2]!='all' or k.split('|')[0] not in ('partial_depth_loss','glare','low_light','homogeneous_translucent')}
    assert min(quotas.values())>=100
    with gzip.open(a.out/'current-tof-predictions.jsonl.gz','wt',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    cohorts=Counter(r['cohort'] for r in rows)
    audit=dict(version='tof_formal_v1_20261017',source_commit='15bfe99c507af2755e32f99a2ba472c67093d352',
        total_frames=len(rows),retained_opaque_frames=sum(n for c,n in cohorts.items() if not c.startswith('new_')),
        additional_geometry_frames=cohorts['new_dual_far_blender_v2'],phase_frames=cohorts['new_dual_raw_phase_holdout_v2'],
        sensor_families=['tof'],cohorts=dict(cohorts),minimum_cell=min(quotas.values()),report_cells=quotas,
        all_tof_rows_retained=True,predictions_unchanged=True,no_error_based_filter=True,new_training_or_new_test_set=False,
        source_prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        provenance='Original cohort IDs, physical observations and inclusion decisions retained; source release in Git history; current artifact is ToF-only subset, not new collection.')
    (a.out/'evaluation-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
