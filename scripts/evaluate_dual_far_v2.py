"""Input-audited new far-shallow scenes through the dual-only public facade."""
import argparse,json,time
from collections import Counter
from dataclasses import fields,replace
from pathlib import Path
import cv2,numpy as np,torch
from evaluate_shared_empty_v45 import install
from evaluate_uniform_loss_v48 import delete_to_actual_loss
from special_scene_input_evidence_v1 import assess,assess_capture_origin
from dual_camera_protocol_v2 import *
from measure_release import validated_profile
from liquid_depth.simulation import SyntheticScene,simulate_raw_depth
from liquid_depth.dual_camera_scene_meter_v2 import DualCameraSceneMeter


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'calibrations').mkdir(exist_ok=True)
    profile_path=a.out/'portable-model-profile.json'
    profile_path.write_text(json.dumps(validated_profile()))
    torch.set_num_threads(2);cv2.setNumThreads(1)
    attempts=[];rows=[];cals=[];scene_keys={f.name for f in fields(SyntheticScene)}
    with (a.out/f'records-worker-{a.worker}.jsonl').open('w') as stream:
      for gid in range(a.worker,FAR_INSTALLATIONS,a.workers):
        meters=[];bundles=[]
        for si,sensor in enumerate(SENSORS):
            case=gid*2+si
            item=install(case,a.root/'installations',a.root/'dry',SEED-case*8071+gid*8071,max_annotation_attempts=6,source_slot_stride=1)
            cals.append(item['audit']);bundles.append(item['bundle'])
            (a.out/'calibrations'/f'{case:05d}.json').write_text(json.dumps(item['bundle']))
            meters.append(DualCameraSceneMeter.from_empty_geometry(item['bundle'],item['dry'],profile_path=profile_path))
        for scene,level in CONDITIONS:
            folder=a.root/'observations'/f'{gid:04d}'/f'{scene}_{level}'
            meta=json.loads((folder/'metadata.json').read_text());rgb=cv2.imread(str(folder/'rgb.png'))
            with np.load(folder/'labels_offline.npz') as d:semantic=d['semantic'];geometric=d['true_geometric_full']
            with np.load(folder/'sensor_labels.npz') as d:labels={k:d[k] for k in d.files}
            base=SyntheticScene(**{k:meta[k] for k in scene_keys});accepted=0
            for rep in range(64):
                values=[];evidence=[];errors=[]
                for si,sensor in enumerate(SENSORS):
                    rawfile=folder/f'{sensor}_raw_{rep:02d}.npy';errfile=folder/f'{sensor}_error_offline_{rep:02d}.npy'
                    if not rawfile.exists():
                        rr=np.random.default_rng(SEED+gid*130363+rep*79907)
                        simulated=simulate_raw_depth(replace(base,sensor_family=sensor,seed=base.seed+rep*79907),labels)
                        visible=(semantic==2)|(semantic==3)
                        raw=np.where(visible,simulated['raw_depth_m'],geometric+rr.normal(0,.001+.0007*geometric))
                        raw[(rr.random(raw.shape)<.04)|(geometric<=0)]=0
                        raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
                        np.save(rawfile,raw);np.save(errfile,simulated['simulated_error_type'])
                    raw=np.load(rawfile);err=np.load(errfile);ev=assess(raw,bundles[si])
                    cap=assess_capture_origin(raw,semantic,err,contains_liquid=True)
                    ev['capture_origin']=cap;ev['eligible']&=cap['eligible'];ev['reasons']+=cap['reasons']
                    values.append(raw);errors.append(err);evidence.append(ev)
                good=all(x['eligible'] for x in evidence)
                attempts.append(dict(gid=gid,scene=scene,level=level,draw=rep,paired_eligible=good,evidence=dict(zip(SENSORS,evidence))))
                if not good:continue
                for si,sensor in enumerate(SENSORS):
                    raw=values[si];current=[(scene,level,raw,None)]
                    loss=100*np.mean(raw[semantic==2]<=0)
                    if scene=='ordinary_diffuse' and loss<=50:
                        current.append(('partial_depth_loss',f'{min(int(loss//10)*10,40)}-{min(int(loss//10)*10,40)+10}%',raw,loss))
                        for target in (5,15,25,35,45,50):
                            if target>loss+.001:
                                missing=delete_to_actual_loss(raw,semantic==2,target,SEED+gid*10007+rep*509)
                                current.append(('partial_depth_loss',f'{min(int(target//10)*10,40)}-{min(int(target//10)*10,40)+10}%',missing,target))
                    for output_scene,output_level,frame,actual_loss in current:
                        # Controlled deletion may reduce evidence; still audit only input support.
                        ev=assess(frame,bundles[si]);cap=assess_capture_origin(frame,semantic,errors[si],contains_liquid=True)
                        if not (ev['eligible'] and cap['eligible']):
                            raise RuntimeError('controlled_0_50pct_frame_has_insufficient_capture_evidence')
                        start=time.perf_counter()
                        result=meters[si].measure(output_scene,depth_m=frame,rgb_bgr=rgb,visual_enabled=output_scene=='floating_objects')
                        row=dict(scene=output_scene,sensor=sensor,level=output_level,cluster=f'dual_far_{gid}',physical_id=f'dual_far_{gid}',
                            repeat=rep,camera_to_bottom_m=meta['camera_to_bottom_m'],truth=meta['truth'],
                            liquid_depth_m=result.get('liquid_depth_m'),highest_height_m=result.get('highest_height_m'),
                            algorithm_ms=(time.perf_counter()-start)*1000,cohort='new_dual_far_blender_v2',
                            actual_loss_pct=actual_loss,input_evidence=ev,scope_included=True,diagnostics=result.get('diagnostics',{}),
                            visual_module_executed=result.get('visual_module_executed',False),liquid_visual_guidance_used=result.get('liquid_visual_guidance_used',False),
                            input_path=str(folder/f'{sensor}_raw_{rep:02d}.npy'),geometry_path=str(a.out/'calibrations'/f'{gid*2+si:05d}.json'),
                            source_rgb_path=str(folder/'rgb.png'))
                        assert not row['liquid_visual_guidance_used']
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
                accepted+=1
                if accepted==REPEATS:break
            if accepted!=REPEATS:raise RuntimeError(f'input_only_quota_not_filled:{gid}:{scene}:{level}')
        print('DUAL_FAR_EVALUATED',gid,'rows',len(rows),flush=True)
    (a.out/f'capture-attempts-worker-{a.worker}.json').write_text(json.dumps(attempts))
    (a.out/f'calibration-audit-worker-{a.worker}.json').write_text(json.dumps(cals))
    print('DUAL_FAR_COMPLETE',a.worker,len(rows),flush=True)


if __name__=='__main__':main()
