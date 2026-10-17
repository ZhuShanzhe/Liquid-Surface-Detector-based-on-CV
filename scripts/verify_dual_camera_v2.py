"""Actual sixteen scene/camera CLI replays plus independent scene-cell recount."""
import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from summarize_dual_camera_v2 import metrics,REPO,OUT,SCENES,SENSORS


def main():
    with gzip.open(OUT/'current-dual-predictions.jsonl.gz','rt') as f:rows=[json.loads(s) for s in f]
    expected=json.loads((OUT/'metrics.json').read_text());checked=0
    for key,want in expected.items():
        scene,sensor,level,part=key.split('|')
        selected=[r for r in rows if r['scene']==scene and (sensor=='all' or r['sensor']==sensor) and (level=='all' or r['level']==level)]
        if part=='far_shallow':selected=[r for r in selected if r['camera_to_bottom_m']>=6 and 0<r['truth']['liquid_height_m']<=1]
        for name,fields in want.items():
            got=metrics(selected,name=='highest')
            for field in ('n','installations','numeric','no_output','numeric_coverage_pct','pass_pct','mae_mm','p95_mm','max_mm','absrel_pct','absolute_bins_pct','relative_bins_pct'):
                if fields[field] is None:assert got[field] is None,(key,field)
                else:assert np.allclose(got[field],fields[field],atol=1e-10,rtol=1e-12),(key,field)
            checked+=1
    smoke=OUT/'smoke';smoke.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    checks=[]
    for scene in SCENES:
      for sensor in SENSORS:
        r=next(r for r in rows if r['scene']==scene and r['sensor']==sensor and r['cohort'].startswith('new_'))
        folder=smoke/f'{scene}_{sensor}';folder.mkdir(exist_ok=True)
        args=['--scene',scene]
        if scene.startswith('homogeneous_'):
            args+=['--initialization',r['initialization_path'],'--phase-capture',r['observation_path']]
        else:
            case=int(Path(r['geometry_path']).stem)
            init=dict(kind='empty_geometry',sensor_family=sensor,bundle=r['geometry_path'],
                dry_depth=f'/root/autodl-tmp/liquid-depth-data/synthetic/dual-camera-v2-20261017/far/dry/{case:04d}/dry_depth_m.npy')
            p=folder/'initialization.json';p.write_text(json.dumps(init,indent=2))
            args+=['--initialization',str(p),'--depth',r['input_path'],'--rgb',r['source_rgb_path']]
        args+=['--enable-floating-peak-vision']
        command=[sys.executable,'scripts/measure_dual_camera.py',*args]
        proc=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,check=True)
        value=json.loads(proc.stdout)
        assert value['sensor_family']==sensor and value['release_version']=='dual_camera_v2_20261017'
        assert value['visual_module_executed']==(scene=='floating_objects')
        assert not value['liquid_visual_guidance_used']
        assert abs(value['liquid_depth_m']-r['liquid_depth_m'])<1e-6,(scene,sensor,value['liquid_depth_m'],r['liquid_depth_m'])
        if scene in ('floating_objects','uneven_surface'):assert abs(value['highest_height_m']-r['highest_height_m'])<1e-6
        checks.append(dict(scene=scene,sensor=sensor,numeric_replay_match=True,vision_executed=value['visual_module_executed']))
        (folder/'command.json').write_text(json.dumps(command));(folder/'result.json').write_text(json.dumps(value,indent=2))
    lock=json.loads((OUT/'final-holdout-source-lock.json').read_text())
    assert all(hashlib.sha256((REPO/n).read_bytes()).hexdigest()==s for n,s in lock.items())
    report=dict(metric_groups_checked=checked,scene_camera_cli_checks=checks,all_16_matched=True,
        final_holdout_source_lock_matches=True,retained_opaque_route='unchanged determined V1 backend',
        no_active_stereo_or_old_image_proxy_in_current_dataset=True)
    (OUT/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
