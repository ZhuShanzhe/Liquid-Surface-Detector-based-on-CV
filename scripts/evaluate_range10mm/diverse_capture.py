"""Blender: diverse installation holdout, input-only reservations, frozen runtime."""
import ast,argparse,fcntl,importlib.util,json,math,random,sys,time
from pathlib import Path
import numpy as np
REPO=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from range_protocol import ROOT,BANDS,QUOTAS,band_of,installation
sys.path[:0]=[str(REPO/'scripts')]
import render_scene_console as producer
# Change only producer geometry/photometric parameters; the estimator is frozen.
src=Path(producer.__file__).read_text()
tree=ast.parse(src);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='render')
text=ast.unparse(fn)
old='gen.simulate_raw_depth(scene, labels)';assert old in text
text=text.replace(old,"gen.simulate_raw_depth(replace(scene,scenario='ordinary'), labels)")
old='gen.create_lighting(scene, rr)';assert old in text
text=text.replace(old,"gen.create_lighting(replace(scene,scenario_profile='calibration',corruption_severity=glare) if glare else scene,rr)")
old='liquid_roughness=0.025 if glare else 0.22';assert old in text
text=text.replace(old,'liquid_roughness=max(.008,.22*(1-glare)) if glare else .22')
old="camera_position_m=(0.0, 0.0, b - h) if request.get('camera_mount') == 'top_center' else (0.025 * b, -0.025 * b, b - h)"
assert old in text;text=text.replace(old,"camera_position_m=(*request.get('camera_xy_m',[0.,0.]), b-h)")
old='corruption_severity=max(0.18, dark * 0.7, glare * 0.8)';assert old in text
text=text.replace(old,"corruption_severity=float(request.get('sensor_severity',.18))")
old="np.savez_compressed(folder / 'offline_geometry.npz', true_depth_m=full, semantic=semantic)";assert old in text
text=text.replace(old,"highest_obj=max(floats,key=lambda o:max((o.matrix_world@v.co).z for v in o.data.vertices)) if floats else None; highest_d=cast(scene,tree_for([highest_obj])) if highest_obj else np.zeros_like(full); np.savez_compressed(folder/'offline_geometry.npz',true_depth_m=full,semantic=semantic,highest_instance_visible=(highest_d>0)&(abs(highest_d-full)<.001))")
namespace=dict(producer.__dict__);exec(compile(text,'<diverse-physical-producer>','exec'),namespace);render=namespace['render']
spec=importlib.util.spec_from_file_location('q',REPO/'src/liquid_depth/sim_input_qualification.py');qmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(qmod)
defs=[('ordinary',0,None),('low_light_low',1,(.15,.4)),('low_light_medium',1,(.4,.7)),('low_light_strong',1,(.7,.98)),
 ('glare_low',2,(.1,.35)),('glare_medium',2,(.35,.65)),('glare_strong',2,(.65,1.)),('uneven_surface',4,None),('floating_objects',8,None),
 ('loss_low',0,(0.,1/6)),('loss_medium',0,(1/6,1/3)),('loss_high',0,(1/3,.5)),('transparent',0,None),('translucent',0,None)]
defs += [(f'compound_{b:02d}',b,None) for b in (3,5,6,7,9,10,11,12,13,14,15)]
p=argparse.ArgumentParser();p.add_argument('--category',type=int,required=True);p.add_argument('--limit',type=int,default=100);p.add_argument('--smoke-high',action='store_true');a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);ci=a.category
name,bits,interval=defs[ci];statepath=ROOT/f'capture-{ci:02d}.json';dest=ROOT/'data';dest.mkdir(exist_ok=True)
state=json.loads(statepath.read_text()) if statepath.exists() else dict(category=name,category_id=ci,bits=bits,cohort='single' if ci<14 else 'composite',attempts=0,qualified=[],skipped=[])
def save_state():
    tmp=statepath.with_suffix('.tmp');tmp.write_text(json.dumps(state));tmp.replace(statepath)
for _ in range(5000):
    counts=[sum(x.get('depth_band')==j for x in state['qualified']) for j in range(9)]
    todo=[8] if a.smoke_high and counts[8]<1 else ([] if a.smoke_high else [j for j in range(9) if counts[j]<QUOTAS[j]])
    if not todo:break
    band=todo[0];trial=state['next_attempt_by_band'][band];state['next_attempt_by_band'][band]+=1
    assert trial<900,'input-only sampling failed to achieve quota'
    inst,h=installation(band,trial);gid=inst['installation_id'];rep=0;attempt=band*1000+trial
    r=np.random.default_rng(20261021190011+ci*10000019+attempt*99991)
    cap=inst['capacity_m'];b=inst['bottom_m']
    if bits&4 and h in (.1,9.):h+=.005 if h==.1 else -.005
    material='homogeneous_'+name if ci in (12,13) else 'ordinary_diffuse'
    req=dict(inst,seed=2026102180039+gid*104729+rep*300007,index=ci+70,height_m=h,material=material,
        darkness=float(r.uniform(*(interval if ci in (1,2,3) else (.35,.98)))) if bits&1 else 0.,
        glare=float(r.uniform(*(interval if ci in (4,5,6) else (.35,1.)))) if bits&2 else 0.,
        wave_m=min(float(r.uniform(.01,.15)),h*.25,(cap-h)*.35) if bits&4 else 0.,
        float_count=int(r.choice([1,2,3,6,12,18])) if bits&8 else 0,loss=0.,sensor_severity=float(r.uniform(.05,.65)))
    if name.startswith('loss_'):req['loss']=float(r.uniform(interval[0]+.01,interval[1]-.01))
    req['noise_seed']=int(r.integers(1,2**31));req['repeat']=rep
    folder=dest/f'new_{ci:02d}_{band:02d}_{trial:04d}';start=time.time();render(req,folder,corpus=False)
    state['attempts']+=1;state['new_attempts']+=1
    raw=np.load(folder/'depth_m.npy');sem=np.load(folder/'offline_geometry.npz')['semantic'];valid=raw>0
    # Correlated range disturbances persist under point averaging. These are
    # predeclared sensitivity parameters, not asserted commercial-camera specs.
    nr=np.random.default_rng(req['noise_seed']);yy,xx=np.indices(raw.shape)
    distance=b-h
    common=float(nr.normal(0,.001+.0005*distance)) if inst['noise_family'] in (2,3) else 0.
    spatial=np.zeros_like(raw)
    if inst['noise_family'] in (1,3):
        for _ in range(4):
            spatial += nr.normal(0,.0006+.00015*distance)*np.sin(xx/raw.shape[1]*nr.uniform(2,8)+nr.uniform(0,6.28))*np.cos(yy/raw.shape[0]*nr.uniform(2,8)+nr.uniform(0,6.28))
    raw[valid]+=common+spatial[valid]
    # Local IR saturation occupies a controlled part of the visible surface;
    # optical darkness is not automatically converted to global ToF dropout.
    if bits&2:
        y,x=np.nonzero((sem==2)|(sem==3))
        if len(x):
            k=int(nr.integers(len(x)));rx=max(2,float(np.ptp(x))*nr.uniform(.08,.30));ry=max(2,float(np.ptp(y))*nr.uniform(.08,.30))
            spot=((xx-x[k])/rx)**2+((yy-y[k])/ry)**2<1
            raw[spot&valid&(nr.random(raw.shape)<(.25+.65*req['glare']))]=0
    raw=np.where(np.isfinite(raw)&(raw>.08),raw,0).astype(np.float32)
    np.save(folder/'depth_m.npy',raw);np.save(folder/'capture_valid.npy',raw>0)
    req['injected_common_bias_m']=common;req['spatial_rms_m']=float(np.sqrt(np.mean(spatial**2)))
    (folder/'request.json').write_text(json.dumps(req))
    q=qmod.qualify_simulated_input(raw,sem,material='ordinary_diffuse',has_floats=bool(bits&8),minimum_depth_m=.08)
    origin=np.load(folder/'offline_return_origin.npz')['error_type'];evidence=(sem==2)&(raw>0)&(origin==0)
    ey,ex=np.nonzero(evidence);span=min(np.ptp(ey)+1,np.ptp(ex)+1) if len(ex) else 0
    q.update(genuine_liquid_return_pixels=int(evidence.sum()),genuine_liquid_span_px=int(span))
    if ci not in (12,13) and (evidence.sum()<30 or span<3):q['reasons'].append('insufficient_genuine_liquid_support')
    if bits&8:
        highest=np.load(folder/'offline_geometry.npz')['highest_instance_visible']
        n=int((highest&(raw>0)&(origin==0)).sum());q['highest_instance_genuine_return_pixels']=n
        if n<3:q['reasons'].append('highest_instance_not_observable')
    q['eligible']=not q['reasons']
    if ci in (12,13):
        small=(sem==2)[111:159,208:272]
        q=dict(eligible=int(small.sum())>=30,reasons=[] if small.sum()>=30 else ['insufficient_projected_phase_support'],phase_qualification_pending=True,projected_phase_pixels=int(small.sum()))
    if name.startswith('loss_') and q.get('eligible') and not interval[0]<=q['liquid_invalid_fraction']<=interval[1]:q=dict(q,eligible=False,reasons=['outside_predeclared_actual_loss_bin'])
    true_h=json.loads((folder/'truth.json').read_text())['liquid_height_m']
    if band_of(true_h)!=band:q=dict(q,eligible=False,reasons=q['reasons']+['outside_prespecified_true_height_band'])
    if q.get('eligible',ci in (12,13)):
        empty=ROOT/'empty'/f'{gid:04d}';empty.parent.mkdir(exist_ok=True)
        lock=(ROOT/'empty'/f'{gid:04d}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        if not (empty/'complete.json').exists():render(dict(inst,seed=2026102180039+gid*104729,index=0,height_m=0.,material='ordinary_diffuse'),empty,corpus=False)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        if ci in (12,13):
            drysem=np.load(empty/'offline_geometry.npz')['semantic'][111:159,208:272]
            common_visible=small&(drysem==2);q['prior_and_current_visible_phase_pixels']=int(common_visible.sum())
            if common_visible.sum()<30:
                q.update(eligible=False,reasons=['insufficient_empty_and_wet_phase_support'])
                state['skipped'].append(dict(frame=str(folder),request=req,qualification=q,measurement_calls=0,source='new_input',depth_band=band))
                save_state();continue
        state['qualified'].append(dict(frame=str(folder),qualification=q,seed=req['seed'],attempt=attempt,installation_id=gid,source='new_input',depth_band=band))
    else:state['skipped'].append(dict(frame=str(folder),request=req,qualification=q,measurement_calls=0,source='new_input',depth_band=band))
    save_state()
    print('CAPTURE',name,attempt,len(state['qualified']),round(time.time()-start,3),flush=True)
if not a.smoke_high:
    assert len(state['qualified'])==100,(name,len(state['qualified']))
    assert [sum(x['depth_band']==j for x in state['qualified']) for j in range(9)]==QUOTAS
    (ROOT/f'capture-{ci:02d}.complete').write_text('true')
