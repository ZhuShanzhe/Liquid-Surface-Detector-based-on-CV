"""Fresh Blender RGB, physical floor/float geometry and dual-mechanism holdout."""
import argparse,json,random,sys
from dataclasses import replace
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_fullframe_v16 import cast,tree_for
from generate_other_scenes_v34 import make_scene
from generate_special_scene_strata_v1 import floor_for
from dual_camera_protocol_v2 import *


def capture(root,gid,scenario,level):
    d=far_design(gid);folder=root/'observations'/f'{gid:04d}'/f'{scenario}_{level}'
    if (folder/'complete.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    base,_=make_scene(0,SEED+gid*701);z=d['z'];angle=d['offset_angle']
    scene=replace(base,index=gid,seed=SEED+gid*100003,width=480,height=270,
        scenario='ordinary' if scenario=='ordinary_diffuse' else scenario,sensor_family='tof',
        container_shape=d['shape'],container_exponent=10. if d['shape']=='rectangular_tank' else 2.,
        surface_radius_x_m=d['radius_x'],surface_radius_y_m=d['radius_y'],
        camera_position_m=(z*.07*np.cos(angle),z*.07*np.sin(angle),z),
        container_bottom_z_m=-d['nominal_liquid_height_m']-.003,container_rim_z_m=.4,
        floating_object_count=12 if scenario=='floating_objects' else 0,
        wave_frequency_x=d['frequency_x'],wave_frequency_y=d['frequency_y'],wave_phase=d['phase'],
        **condition_settings(scenario,level,d))
    rng=np.random.default_rng(SEED+gid*104729);py_rng=random.Random(SEED+gid*104729)
    gen.clear_scene();gen.create_environment(scene,py_rng);gen.create_container(scene);floor=floor_for(scene,rng)
    gen.create_camera(scene);gen.create_lighting(scene,py_rng)
    gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=8))
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    dry_geom=cast(scene,tree_for(objects))
    if scenario=='ordinary_diffuse':
        dry=dry_geom+rng.normal(0,.001+.0005*dry_geom**2)
        dry[(dry_geom<=0)|(rng.random(dry.shape)<.04)]=0
        dry=np.where((dry>0)&(dry<=10.5),dry,0).astype(np.float32)
        for si,sensor in enumerate(SENSORS):
            case=gid*2+si;f=root/'dry'/f'{case:04d}';f.mkdir(parents=True,exist_ok=True);np.save(f/'dry_depth_m.npy',dry)
            f=root/'installations'/f'{case:05d}';f.mkdir(parents=True,exist_ok=True)
            meta=gen.scene_metadata(replace(scene,sensor_family=sensor),f)
            meta.update(d,case=case,scenario='ordinary',interior_bottom_z_m=-d['nominal_liquid_height_m'])
            (f/'metadata.json').write_text(json.dumps(meta,indent=2))
    liquid=gen.create_liquid(scene);bsdf=liquid.data.materials[0].node_tree.nodes.get('Principled BSDF')
    for key,value in [('Transmission Weight',0.),('Transmission',0.),('Alpha',1.)]:gen._set_input(bsdf,key,value)
    gen.create_floating_objects(scene,py_rng)
    floats=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('floating_object_')]
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(scene);water=labels['target_depth_m'].copy()
    floor_depth=cast(scene,tree_for([floor]));other=cast(scene,tree_for([o for o in objects if o!=floor]))
    float_depth=cast(scene,tree_for(floats)) if floats else np.zeros_like(water)
    candidates=np.stack([other,floor_depth,water,float_depth]);any_hit=(candidates>0).any(axis=0)
    nearest=np.argmin(np.where(candidates>0,candidates,np.inf),axis=0)
    semantic=np.where(any_hit,nearest,0).astype(np.uint8)
    visible=(semantic==2)|(semantic==3)
    labels['target_depth_m']=np.where(semantic==3,float_depth,water).astype(np.float32)
    labels['mask']=((labels['mask']>0)|(semantic==3)).astype(np.uint8)
    gx,gy=np.meshgrid(np.linspace(-d['radius_x'],d['radius_x'],256),np.linspace(-d['radius_y'],d['radius_y'],256))
    inside=abs(gx/d['radius_x'])**scene.container_exponent+abs(gy/d['radius_y'])**scene.container_exponent<=1
    h=gen.surface_height_and_gradient(scene,gx,gy)[0][inside]+d['nominal_liquid_height_m']
    peak=max([float(h.max())]+[float((o.matrix_world@v.co).z+d['nominal_liquid_height_m']) for o in floats for v in o.data.vertices])
    truth=dict(liquid_height_m=float(h.mean()),highest_height_m=peak,lowest_height_m=float(h.min()),wave_peak_to_peak_m=float(np.ptp(h)))
    geometric=np.where(any_hit,np.min(np.where(candidates>0,candidates,np.inf),axis=0),0)
    np.savez_compressed(folder/'labels_offline.npz',semantic=semantic,water_depth_m=water,true_geometric_full=geometric)
    # Extra draws for input-only qualification are generated outside Blender from these frozen labels.
    np.savez_compressed(folder/'sensor_labels.npz',**labels)
    for sensor in SENSORS:
        for rep in range(REPEATS):
            rr=np.random.default_rng(SEED+gid*130363+rep*79907)
            sim=gen.simulate_raw_depth(replace(scene,sensor_family=sensor,seed=scene.seed+rep*79907),labels)
            raw=np.where(visible,sim['raw_depth_m'],geometric+rr.normal(0,.001+.0007*geometric))
            raw[(rr.random(raw.shape)<.04)|~any_hit]=0
            raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
            np.save(folder/f'{sensor}_raw_{rep:02d}.npy',raw);np.save(folder/f'{sensor}_error_offline_{rep:02d}.npy',sim['simulated_error_type'])
    meta=gen.scene_metadata(scene,folder);meta.update(d,scenario=scenario,level=level,truth=truth,
        interior_bottom_z_m=-d['nominal_liquid_height_m'],physical_peak_method='analytic area mean plus Blender float mesh vertices')
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2));(folder/'complete.json').write_text(json.dumps({'frames':8}))
    print('DUAL_FAR_CAPTURE',gid,scenario,level,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(parents=True,exist_ok=True)
    for gid in range(a.worker,FAR_INSTALLATIONS,a.workers):
        for scene,level in CONDITIONS:capture(a.root,gid,scene,level)
