"""New five-prior phase holdout. Development and holdout seeds are disjoint."""
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import cv2,numpy as np
from dual_camera_protocol_v2 import SEED
from phase_capture_simulator_v2 import generate,rays
from liquid_depth.dual_camera_scene_meter_v2 import DualCameraSceneMeter


def spec(group,regime,rep,dev=False,seed_base=None):
    seed=((SEED if seed_base is None else seed_base) if not dev else 202610171111)+1000000+group*100003+regime*10007+rep*197
    r=np.random.default_rng(seed)
    if regime:
        b=float(r.uniform(6.05,9.95));cap=float(r.uniform(.25,1.))
    else:
        cap=float(np.linspace(.15,9.,25)[rep]);b=float(r.uniform(cap+.21,min(10.,cap+2.)))
    physical=dict(seed=seed,index=float(r.uniform(1.28,1.5)),extinction=0. if group==0 else float(r.uniform(.012,.05) if group==1 else r.uniform(.055,.12)),
        surface_ratio=float(r.uniform(.025,.13)),scatter_strength=0. if group==0 else float(r.uniform(.08,.35)))
    profile=dict(bottom_m=b,container_max_height_m=cap,fx_px=470.94,baseline_m=.035,
        center_q=[float(r.uniform(-.06,.06)),float(r.uniform(-.03,.03))],
        frequencies_hz=[20e6,30e6,50e6],fringe_periods_px=[8.,13.],
        sensor_profile='tof_phase3_or_sl_35mm480px_v2',fixed_camera=True)
    return physical,profile


def job(args,group,regime,rep):
    cv2.setNumThreads(1)
    physical,profile=spec(group,regime,rep,args.dev,args.seed)
    scene='homogeneous_transparent' if group==0 else 'homogeneous_translucent'
    level=('clear','low_extinction','medium_extinction')[group]
    cluster=f'phase_{physical["seed"]}'
    folder=args.data/cluster;folder.mkdir(parents=True,exist_ok=True)
    r=np.random.default_rng(physical['seed']+997)
    known=np.array([0,.1,.35,.65,.95])*profile['container_max_height_m']
    labels=known+np.r_[0,r.normal(0,.0003+.001*known[1:])]
    if regime:tests=np.array([.02,.12,.5,.98])*profile['container_max_height_m']
    else:tests=np.array([.04,.31,.67,.99])*profile['container_max_height_m']
    allrows=[];calibrations=[]
    for sensor in ('tof','structured_light'):
        priors=[]
        for j,h in enumerate(known):
            capture,roi=generate(sensor,profile,physical,float(h),j)
            priors.append(capture);np.savez_compressed(folder/f'{sensor}_prior_{j}.npz',correlations=capture)
        np.savez_compressed(folder/'rays_roi.npz',ray_coordinates=rays(profile),roi=roi)
        init=dict(kind='five_phase_priors',sensor_family=sensor,scene=scene,profile=profile,
            rays_roi='rays_roi.npz',prior_captures=[f'{sensor}_prior_{j}.npz' for j in range(5)],known_heights_m=labels.tolist())
        (folder/f'{sensor}_initialization.json').write_text(json.dumps(init,indent=2))
        start=time.perf_counter();failure=None
        try:
            meter=DualCameraSceneMeter.from_phase_priors(sensor=sensor,scene=scene,profile=profile,ray_coordinates=rays(profile),roi=roi,prior_correlations=priors,known_heights_m=labels)
        except ValueError as exc:meter=None;failure=str(exc)
        calibrations.append(dict(cluster=cluster,sensor=sensor,scene=scene,level=level,regime='far_shallow' if regime else 'general',
            calibration_ms=(time.perf_counter()-start)*1000,success=meter is not None,error=failure,
            known_heights_m=labels.tolist(),fitted_parameters=meter.backend.parameters.tolist() if meter else None,
            residual=meter.backend.calibration_residual if meter else None))
        for j,h in enumerate(tests):
            capture,_=generate(sensor,profile,physical,float(h),1000+j)
            np.savez_compressed(folder/f'{sensor}_test_{j}.npz',correlations=capture)
            start=time.perf_counter();value=meter.measure(scene,correlations=capture) if meter else {'liquid_depth_m':None,'diagnostics':{'prior_error':failure}}
            allrows.append(dict(scene=scene,sensor=sensor,level=level,cluster=cluster,physical_id=cluster,
                camera_to_bottom_m=profile['bottom_m'],truth={'liquid_height_m':float(h)},repeat=j,
                liquid_depth_m=value['liquid_depth_m'],algorithm_ms=(time.perf_counter()-start)*1000,
                diagnostics=value['diagnostics'],inference_failure=failure,
                cohort='new_dual_raw_phase_holdout_v2',regime='far_shallow' if regime else 'general',
                observation_path=str(folder/f'{sensor}_test_{j}.npz'),initialization_path=str(folder/f'{sensor}_initialization.json'),
                scope_included=True,known_heights_budget=5,hardware_qualified=False,
                simulation_scope='4phase sensor observations; analytic Snell/optical-transport approximation with noise and forward/inverse mismatch'))
    (folder/'truth_offline.json').write_text(json.dumps(dict(physical=physical,profile=profile,true_prior_heights=known.tolist(),test_heights=tests.tolist()),indent=2))
    (args.out/f'{cluster}.json').write_text(json.dumps(dict(records=allrows,calibrations=calibrations),allow_nan=False))
    return allrows


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dev',action='store_true');p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=SEED+904327);a=p.parse_args()
    a.data.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    jobs=[(g,b,r) for g in range(3) for b in range(2) for r in range(25 if not a.dev else 1)]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(job,a,*s) for s in jobs]
        for f in as_completed(futures):
            rows+=f.result();print('PHASE_HOLDOUT',len(rows),len(jobs)*8,flush=True)
    (a.out/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    print('PHASE_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
