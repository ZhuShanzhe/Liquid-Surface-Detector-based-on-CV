"""Native-resolution full-frame device-profile study, Blender entry point."""
import argparse,json,random,sys,time,math
from dataclasses import replace
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/liquid_depth'))
import generate_synthetic_liquid as gen
from generate_ordinary_v35 import add_floor
from generate_fullframe_v16 import tree_for,cast
from device_profiles_v38 import PROFILES,SCENES,BANDS,noisy_depth

def make_scene(slot,seed):
    scenario=SCENES[slot%len(SCENES)];j=slot//len(SCENES)
    profile=PROFILES[j%2];case=j//2;band=case//3;rep=case%3
    rng=np.random.default_rng(seed+case*100003)
    low,high=BANDS[band];h=float(rng.uniform(low+.03*(high-low),high-.03*(high-low))) if high else 0.
    z=min((.65,3.,9.)[rep]*float(rng.uniform(.95,1.03)),9.95-h)
    off=z*.025;angle=rng.uniform(0,2*np.pi)
    # Same physical vessel for both profiles; not adjusted to their FOV.
    rx=z*math.tan(math.radians(54/2))*.38;ry=rx*float(rng.uniform(.8,1.1))
    shape=('cylindrical','elliptical','rectangular_tank')[(band+rep)%3]
    if shape=='cylindrical':ry=rx
    base=gen.sample_scene(slot,seed=seed,width=profile.width,height=profile.height,
                          scenario_profile='calibration',camera_profile='near_vertical')
    sc=replace(base,seed=seed+case*100003,index=slot,scenario=scenario,split='test',
        sequence_id=f'device38_{seed}_{case}_{scenario}_{profile.name}',sensor_family=profile.sensor,
        horizontal_fov_deg=profile.hfov_deg,camera_position_m=(off*np.cos(angle),off*np.sin(angle),z),camera_target_m=(0.,0.,0.),
        surface_radius_x_m=rx,surface_radius_y_m=ry,container_shape=shape,
        container_exponent=10. if shape=='rectangular_tank' else 2.,container_taper_ratio=1.,
        container_bottom_z_m=-h-.003,wall_thickness_m=.003,container_rim_z_m=min(.12,z*.15),
        tilt_x=0.,tilt_y=0.,wave_amplitude_m=min(rx*.035,h*.12) if scenario=='uneven_surface' else 0.,
        liquid_turbidity=.05 if scenario=='transparent' else (.3 if scenario=='translucent' else .95),
        liquid_color_rgb=tuple(rng.uniform(.15,.75,3)),liquid_roughness=.025 if scenario=='glare' else .28,
        light_level=.08 if scenario in ('low_light','compound') else 1.,exposure=-2.5 if scenario in ('low_light','compound') else .2,
        corruption_severity=(.22,.48,.7)[(case+1)%3],floating_object_count=12 if scenario in ('floating_objects','compound') and h>0 else 0)
    extra=dict(case=case,band=band,replicate=rep,profile=profile.name,interior_bottom_z_m=-h,
        nominal_liquid_height_m=h,cluster=f'{seed}_{case}_{scenario}',empty=h==0,
        population='empty' if band==0 else ('near_zero_enrichment' if band==1 else 'uniform'),
        camera_to_bottom_m=float(np.linalg.norm(np.array(sc.camera_position_m)-[0,0,-h])),
        external_rgb_ideal_registered=profile.sensor=='tof',native_resolution=[profile.width,profile.height],
        protocol='device_v38',profile_device_qualified=False)
    assert extra['camera_to_bottom_m']<=10
    return sc,extra,profile

def generate(root,slot,seed):
    sc,extra,profile=make_scene(slot,seed);folder=root/f'{slot:05d}';folder.mkdir(exist_ok=False)
    rng=random.Random(seed+slot*9173)
    gen.clear_scene();gen.create_environment(sc,rng);gen.create_container(sc);floor=add_floor(sc)
    if not extra['empty']:
        obj=gen.create_liquid(sc)
        if sc.scenario not in ('transparent','translucent','multilayer'):
            bs=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
            gen._set_input(bs,'Transmission Weight',0.);gen._set_input(bs,'Transmission',0.);gen._set_input(bs,'Alpha',1.)
    gen.create_floating_objects(sc,rng);gen.create_camera(sc);gen.create_lighting(sc,rng)
    gen.configure_render(sc,argparse.Namespace(engine='eevee',render_samples=16))
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(sc);target=labels['target_depth_m'];mask=labels['mask'].astype(bool)
    obs=[o for o in bpy.context.scene.objects if o.type=='MESH']
    opaque=cast(sc,tree_for([o for o in obs if o.name not in ('liquid_volume','transparent_container')]))
    wall=cast(sc,tree_for([o for o in obs if o.name=='transparent_container']))
    visible=mask&~((opaque>0)&(opaque<target-np.maximum(.003,target*.002)))
    if extra['empty']:
        fd=cast(sc,tree_for([floor]));valid=mask&(fd>0)
        assert valid.sum()>.98*mask.sum() and np.max(abs(fd[valid]-target[valid]))<1e-4
    records=[]
    for repeat in range(4):
        varied=replace(sc,seed=sc.seed+repeat*7907)
        # Darkness of visible light alone must not imply loss of active IR depth.
        sensor_scene=replace(varied,scenario='ordinary') if sc.scenario=='low_light' or extra['empty'] else varied
        fault=gen.simulate_raw_depth(sensor_scene,labels,components=dict(depth_noise=False,disparity_noise=False,quantization=False,range_cutoff=False))['raw_depth_m']
        r=np.random.default_rng(sc.seed+repeat*13007+383)
        backing=opaque.copy();firstwall=(wall>0)&((opaque==0)|(wall<opaque))
        hit=firstwall&(r.random(target.shape)<.3);backing[hit]=wall[hit]
        backing[(backing<=0)|(r.random(target.shape)<.15)]=0
        full=np.where(visible,fault,backing)
        front=visible&firstwall&(wall<target-.002)&(r.random(target.shape)<.2);full[front]=backing[front]
        for regime in ('nominal','adverse'):
            yy,xx=np.indices(full.shape);k=gen._simulation.camera_intrinsics(sc)
            ray_norm=np.sqrt(1+((xx-k[0,2])/k[0,0])**2+((yy-k[1,2])/k[1,1])**2)
            raw=noisy_depth(full,profile,seed+slot*137,repeat,regime,ray_norm)
            variants=('base','drop75','drop90','drop97') if sc.scenario=='depth_failure' else ('base',)
            for variant in variants:
                output=raw.copy()
                if variant!='base':
                    yy,xx=np.nonzero(visible);phase=(xx+repeat*7)%sc.width
                    remove=phase<=np.quantile(phase,int(variant[4:])/100);output[yy[remove],xx[remove]]=0
                name=f'raw_{repeat:02d}_{regime}_{variant}.npy';np.save(folder/name,output)
                records.append(dict(folder=str(folder),raw_path=str(folder/name),repeat=repeat,variant=variant,slot=slot,regime=regime))
    np.savez_compressed(folder/'geometry.npz',target=np.where(visible,target,0),mask=visible)
    meta=gen.scene_metadata(sc,folder);meta.update(extra)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2));(folder/'frames.json').write_text(json.dumps(records))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=660)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(parents=True,exist_ok=True)
    for slot in range(a.worker,a.limit,a.workers):
        generate(a.root,slot,a.seed);print('DONE',slot,flush=True)
