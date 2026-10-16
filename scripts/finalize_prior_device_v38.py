"""Replace the predeclared development texture with a fresh independent seed."""
import json,hashlib
from pathlib import Path
from refraction_device_v38 import job
from liquid_depth.device_profiles_v38 import metric_summary
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/evaluation/device-v38-test-20261012')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/device-v38-test-20261012/five-prior-independent')
def main():
    plan=json.loads((ROOT/'independent-prior-plan.json').read_text());out=ROOT/'five-prior-independent'
    out.mkdir(exist_ok=False);DATA.mkdir(exist_ok=False)
    original=json.loads((ROOT/'five-prior/test-raw.json').read_text())['records']
    kept=[r for r in original if r['cluster']!=str(plan['excluded_development_seed'])]
    assert len(original)-len(kept)==40 and len(kept)==920
    spec=(plan['replacement_independent_seed'],*plan['replacement_geometry'])
    replacement=job(spec,DATA,out);rows=kept+replacement
    assert len(rows)==960 and len({r['cluster'] for r in rows})==24
    assert not any(r['cluster']==str(plan['excluded_development_seed']) for r in rows)
    summary={}
    for sc in ['transparent','translucent_mild']:
        rr=[r for r in rows if r['scenario']==sc];summary[sc]={'all':metric_summary(rr)}
        for pop in ['empty','uniform','near_zero_enrichment']:summary[sc][pop]=metric_summary([r for r in rr if r['population']==pop])
        for b in range(11):summary[sc]['band_'+str(b)]=metric_summary([r for r in rr if r['band']==b])
    (out/'test-raw.json').write_text(json.dumps(dict(records=rows),allow_nan=False))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'audit.json').write_text(json.dumps(dict(plan=plan,original_preserved=True,
       source_sha256=hashlib.sha256(Path('scripts/refraction_device_v38.py').read_bytes()).hexdigest(),
       final_frames=960,final_configs=24),indent=2))
    print('INDEPENDENT_PRIOR_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
