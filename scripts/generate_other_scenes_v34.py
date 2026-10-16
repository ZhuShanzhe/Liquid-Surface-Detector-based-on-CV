"""Blender entry point: fresh, depth-stratified, top-view RGB-D surface tests."""
import argparse,json,math,random,sys,time
from dataclasses import replace
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_fullframe_v16 import context_proxy

SCENES=('ordinary','low_light','glare','floating_objects','uneven_surface','depth_failure')
SENSORS=('active_stereo','structured_light','tof')
BANDS=((0,0),(.01,.1),(.1,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9))

def make_scene(slot, seed):
    case=slot//len(SCENES);scenario=SCENES[slot%len(SCENES)]
    band=case//9;sensor=SENSORS[(case//3)%3];rep=case%3
    rng=np.random.default_rng(seed+case*100003)
    low,high=BANDS[band];h=float(rng.uniform(low+.02*(high-low),high-.02*(high-low))) if high else 0.
    z=float(min((.55,2.8,8.8)[rep]*rng.uniform(.9,1.06),(10-h)*(.65,.78,.92)[rep]))
    angle=float(rng.uniform(0,2*np.pi));offset=z*.07
    rx=z*math.tan(math.radians(54/2))*.38
    ry=rx*float(rng.uniform(.75,1.05));shape=('cylindrical','elliptical','rectangular_tank')[case%3]
    if shape=='cylindrical':ry=rx
    base=gen.sample_scene(slot,seed=seed,width=320,height=180,scenario_profile='calibration',camera_profile='near_vertical')
    severity=(.22,.48,.7)[rep]
    amp=min(rx*.035,h*.12) if scenario=='uneven_surface' else 0.
    scene=replace(base,seed=seed+case*100003,index=slot,scenario=scenario,split='test',
        sequence_id=f'other34_{seed}_{case:04d}_{scenario}',sensor_family=sensor,
        horizontal_fov_deg=54.,camera_position_m=(offset*math.cos(angle),offset*math.sin(angle),z),
        camera_target_m=(0.,0.,0.),surface_radius_x_m=rx,surface_radius_y_m=ry,
        container_shape=shape,container_exponent=10. if shape=='rectangular_tank' else 2.,
        container_taper_ratio=1.,container_bottom_z_m=-h-.003,wall_thickness_m=.003,
        container_rim_z_m=min(.12,z*.15),tilt_x=0.,tilt_y=0.,wave_amplitude_m=amp,
        liquid_turbidity=.95,liquid_color_rgb=tuple(rng.uniform(.15,.75,3)),
        liquid_roughness=.025 if scenario=='glare' else .28,
        light_level=(.08 if scenario=='low_light' else 1.),exposure=(-2.5 if scenario=='low_light' else .2),
        corruption_severity=severity,floating_object_count=(12 if scenario=='floating_objects' and h>0 else 0))
    return scene,dict(case=case,band=band,replicate=rep,interior_bottom_z_m=-h,nominal_liquid_height_m=h,
        cluster=f'case_{case:04d}',camera_to_bottom_m=float(np.linalg.norm(np.array(scene.camera_position_m)-[0,0,-h])))

def generate(root,slot,seed,repeats,scene_override=None):
    scene,extra=make_scene(slot,seed) if scene_override is None else scene_override
    folder=root/f'{slot:05d}';folder.mkdir(exist_ok=False)
    rng=random.Random(seed+slot*104729)
    gen.clear_scene();gen.create_environment(scene,rng);gen.create_container(scene)
    if extra['nominal_liquid_height_m']>0:
        obj=gen.create_liquid(scene)
        bsdf=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        gen._set_input(bsdf,'Transmission Weight',0.);gen._set_input(bsdf,'Transmission',0.)
        gen._set_input(bsdf,'Alpha',1.)
    gen.create_floating_objects(scene,rng);gen.create_camera(scene);gen.create_lighting(scene,rng)
    gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=16))
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(scene)
    sensor=gen.simulate_raw_depth(scene,labels)
    full,target,visible,opaque,wall=context_proxy(scene,sensor['raw_depth_m'],labels['target_depth_m'],labels['mask'])
    # The full-frame background and physical occlusion are retained for every repeat.
    clean_surface=visible.astype(bool)&np.isclose(full,sensor['raw_depth_m'],rtol=0,atol=.00051)
    np.savez_compressed(folder/'geometry.npz',target=target,mask=visible,analytic_mask=labels['mask'])
    meta=gen.scene_metadata(scene,folder);meta.update(extra,slot=slot,empty=extra['nominal_liquid_height_m']==0,
        opaque_surface=True,render_engine='eevee',background_depth_present=True,
        sensor_limitations='parametric proxy; not real device qualification; repeated RGB is static')
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    records=[]
    for repeat in range(repeats):
        simulated=gen.simulate_raw_depth(replace(scene,seed=scene.seed+repeat*7907),labels)
        raw=np.where(clean_surface,simulated['raw_depth_m'],full).astype(np.float32)
        for variant in (('base','drop75','drop90','drop97') if scene.scenario=='depth_failure' else ('base',)):
            output=raw.copy()
            if variant!='base':
                # Controlled contiguous loss within evaluator-known surface support.
                yy,xx=np.nonzero(visible)
                phase=(xx+repeat*7)%scene.width
                remove=phase<=np.quantile(phase,int(variant[4:])/100)
                output[yy[remove],xx[remove]]=0
            filename=f'raw_{repeat:02d}_{variant}.npy';np.save(folder/filename,output)
            records.append(dict(folder=str(folder),raw_path=str(folder/filename),repeat=repeat,variant=variant,slot=slot))
    (folder/'frames.json').write_text(json.dumps(records))
    return len(records)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int,default=594);p.add_argument('--seed',type=int,default=202610120341)
    p.add_argument('--repeats',type=int,default=4)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(parents=True,exist_ok=True)
    started=time.time();done=0
    for slot in range(a.worker,a.limit,a.workers):
        done+=generate(a.root,slot,a.seed,a.repeats)
        print(json.dumps(dict(worker=a.worker,slot=slot,frames=done,elapsed_s=time.time()-started)),flush=True)
