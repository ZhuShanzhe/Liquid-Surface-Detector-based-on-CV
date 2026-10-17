"""Replay all phase observations and eight ToF scene CLIs after route retirement."""
import argparse,gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from measure_tof import initialize
from recount_tof import independent_metrics

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.evidence/'current-tof-predictions.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    assert len(rows)==5744 and {r['sensor'] for r in rows}=={'tof'}
    metrics=json.loads((a.evidence/'metrics.json').read_text());groups=0
    for key,quantities in metrics.items():
        sc,_,lev,_=key.split('|');q=[r for r in rows if r['scene']==sc and (lev=='all' or r['level']==lev)]
        for name,want in quantities.items():
            got=independent_metrics(q,name=='highest')
            for k,v in got.items():assert (v is None and want[k] is None) or (v is not None and want[k] is not None and np.allclose(v,want[k],rtol=1e-12,atol=1e-10)),(key,name,k)
            groups+=1
    phase=[r for r in rows if r['scene'].startswith('homogeneous_')];cache={};maxdiff=0.
    for r in phase:
        path=r['initialization_path']
        if path not in cache:cache[path]=initialize(path,r['scene'])
        with np.load(r['observation_path'],allow_pickle=False) as f:value=cache[path].measure(r['scene'],correlations=f['correlations'])
        delta=abs(value['liquid_depth_m']-r['liquid_depth_m']);maxdiff=max(maxdiff,delta);assert delta<1e-9
    checks=[];env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for sc in dict.fromkeys(r['scene'] for r in rows):
        r=next(r for r in rows if r['scene']==sc and r['cohort'].startswith('new_'));folder=a.out/sc;folder.mkdir(exist_ok=True)
        args=['--scene',sc]
        if sc.startswith('homogeneous_'):args+=['--initialization',r['initialization_path'],'--phase-capture',r['observation_path']]
        else:
            case=int(Path(r['geometry_path']).stem)
            init=dict(kind='empty_geometry',sensor_family='tof',bundle=r['geometry_path'],dry_depth=f'/root/autodl-tmp/liquid-depth-data/synthetic/dual-camera-v2-20261017/far/dry/{case:04d}/dry_depth_m.npy')
            path=folder/'init.json';path.write_text(json.dumps(init));args+=['--initialization',str(path),'--depth',r['input_path'],'--rgb',r['source_rgb_path']]
            if sc=='floating_objects':args+=['--enable-floating-peak-vision']
        value=json.loads(subprocess.run([sys.executable,'scripts/measure_tof.py',*args],cwd=ROOT,env=env,capture_output=True,text=True,check=True).stdout)
        assert value['sensor_family']=='tof' and value['release_version']=='tof_formal_v1_20261017'
        assert not value['liquid_visual_guidance_used'] and value['visual_module_executed']==(sc=='floating_objects')
        assert abs(value['liquid_depth_m']-r['liquid_depth_m'])<1e-6
        if sc in ('floating_objects','uneven_surface'):assert abs(value['highest_height_m']-r['highest_height_m'])<1e-6
        checks.append(dict(scene=sc,match=True));(folder/'result.json').write_text(json.dumps(value,indent=2))
    report=dict(frames=len(rows),cli_replays=len(checks),scene_checks=checks,phase_rows_replayed=len(phase),phase_calibrations_replayed=len(cache),
        phase_max_numerical_difference_m=maxdiff,metric_quantity_groups=groups,all_match=True)
    log=a.evidence/'pytest.log'
    if log.exists():report['pytest_result']=log.read_text().splitlines()[-1]
    report['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('src/liquid_depth/tof_scene_meter.py','src/liquid_depth/tof_phase_refraction.py','scripts/measure_tof.py','configs/application_route.json')}
    (a.evidence/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
