"""New fixed-protocol test-only RGB-D capture. Never changes frozen generators."""
import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_fullframe_v16 import cast, tree_for
from generate_other_scenes_v34 import make_scene
from special_scene_protocol_v1 import SEED, REPEATS, SCENES, SENSORS, LEVELS, design, settings, validate


def scene_for(gid, scenario, level):
    d = design(gid)
    base, _ = make_scene((d['band']*9+d['replicate']%3)*6, SEED+gid*701)
    z, rx = d['z'], d['radius_x']
    angle = d['offset_angle']
    scene = replace(base, index=gid, seed=SEED+gid*100003, width=480, height=270,
        scenario=scenario, sensor_family='tof', container_shape=d['shape'],
        container_exponent=10. if d['shape']=='rectangular_tank' else 2.,
        surface_radius_x_m=rx, surface_radius_y_m=d['radius_y'],
        camera_position_m=(z*.07*np.cos(angle), z*.07*np.sin(angle), z),
        container_bottom_z_m=-d['nominal_liquid_height_m']-.003,
        container_rim_z_m=min(.12,z*.15), floating_object_count=0,
        wave_frequency_x=d['frequency_x'], wave_frequency_y=d['frequency_y'],
        wave_phase=d['phase'], **settings(scenario,level,d))
    return scene, d


def floor_for(scene, rng):
    bottom=scene.container_bottom_z_m+scene.wall_thickness_m
    vertices=[(*gen.container_perimeter_point(scene,2*np.pi*i/128,bottom)[:2],bottom) for i in range(128)]
    mesh=bpy.data.meshes.new('interior_floor_mesh')
    mesh.from_pydata(vertices,[],[tuple(range(128))]);mesh.update()
    obj=bpy.data.objects.new('interior_floor',mesh);bpy.context.collection.objects.link(obj)
    mat=gen.principled_material('randomized_floor',tuple(rng.uniform(.1,.8,3)),roughness=.7)
    tex=mat.node_tree.nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=8.
    ramp=mat.node_tree.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(*rng.uniform(.03,.4,3),1.)
    ramp.color_ramp.elements[1].color=(*rng.uniform(.4,.9,3),1.)
    mat.node_tree.links.new(tex.outputs['Fac'],ramp.inputs['Fac'])
    mat.node_tree.links.new(ramp.outputs['Color'],mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    obj.data.materials.append(mat)
    return obj


def render_condition(root,gid,scenario,level):
    scene,d=scene_for(gid,scenario,level)
    condition=SCENES.index(scenario)*3+LEVELS.index(level)
    folder=root/'observations'/f'{gid:04d}'/f'{scenario}_{level}'
    if (folder/'complete.json').exists():
        return
    folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED+gid*104729)
    py_rng=random.Random(SEED+gid*104729)
    gen.clear_scene();gen.create_environment(scene,py_rng);gen.create_container(scene)
    floor=floor_for(scene,rng)
    gen.create_camera(scene);gen.create_lighting(scene,py_rng)
    gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=8))
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    dry_geometric=cast(scene,tree_for(objects))
    if condition==0:
        dry=dry_geometric+rng.normal(0,.001+.0005*dry_geometric**2)
        dry[(dry_geometric<=0)|(rng.random(dry.shape)<.04)]=0
        dry=np.where((dry>0)&(dry<=10.5),dry,0).astype(np.float32)
        for si,sensor in enumerate(SENSORS):
            case=gid*2+si
            f=root/'dry'/f'{case:04d}';f.mkdir(parents=True,exist_ok=True)
            np.save(f/'dry_depth_m.npy',dry)
            f=root/'installations'/f'{case:05d}';f.mkdir(parents=True,exist_ok=True)
            meta=gen.scene_metadata(replace(scene,sensor_family=sensor),f)
            meta.update(d,case=case,scenario='ordinary',interior_bottom_z_m=-d['nominal_liquid_height_m'],
                        camera_to_bottom_m=scene.camera_position_m[2]+d['nominal_liquid_height_m'])
            (f/'metadata.json').write_text(json.dumps(meta,indent=2))
    if d['nominal_liquid_height_m']>0:
        obj=gen.create_liquid(scene)
        bsdf=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        for key,value in [('Transmission Weight',0.),('Transmission',0.),('Alpha',1.)]:
            gen._set_input(bsdf,key,value)
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(scene)
    floor_depth=cast(scene,tree_for([floor]))
    other=cast(scene,tree_for([o for o in objects if o!=floor]))
    water=labels['target_depth_m'] if d['nominal_liquid_height_m']>0 else np.zeros_like(floor_depth)
    candidates=np.stack([other,floor_depth,water])
    nearest=np.argmin(np.where(candidates>0,candidates,np.inf),axis=0)
    any_hit=(candidates>0).any(axis=0)
    semantic=np.where(any_hit,nearest,0).astype(np.uint8)
    visible=semantic==2
    gx,gy=np.meshgrid(np.linspace(-d['radius_x'],d['radius_x'],256),np.linspace(-d['radius_y'],d['radius_y'],256))
    area=(abs(gx/d['radius_x'])**scene.container_exponent+abs(gy/d['radius_y'])**scene.container_exponent)<=1
    height=gen.surface_height_and_gradient(scene,gx,gy)[0][area]+d['nominal_liquid_height_m']
    wet=d['nominal_liquid_height_m']>0
    truth=dict(liquid_height_m=float(height.mean()) if wet else 0.,
               highest_height_m=float(height.max()) if wet else 0.,
               lowest_height_m=float(height.min()) if wet else 0.,
               wave_peak_to_peak_m=float(np.ptp(height)) if wet else 0.)
    if wet and (height.min()<0 or height.max()>scene.container_rim_z_m+d['nominal_liquid_height_m']):
        raise ValueError('invalid_generated_floor_intersection_or_overflow')
    np.savez_compressed(folder/'labels_offline.npz',semantic=semantic,water_depth_m=water,
                        true_geometric_full=np.where(any_hit,np.min(np.where(candidates>0,candidates,np.inf),axis=0),0))
    for si,sensor in enumerate(SENSORS):
        for rep in range(REPEATS):
            # Same seed across severities and sensors is deliberate paired common-random-number control.
            repeat_rng=np.random.default_rng(SEED+gid*130363+rep*79907)
            ss=replace(scene,sensor_family=sensor,seed=scene.seed+rep*79907)
            sim=gen.simulate_raw_depth(ss,labels)
            raw=np.where(visible,sim['raw_depth_m'],dry_geometric+repeat_rng.normal(0,.001+.0007*dry_geometric))
            raw[(repeat_rng.random(raw.shape)<.04)|~any_hit]=0
            raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
            np.save(folder/f'{sensor}_raw_{rep:02d}.npy',raw)
            np.save(folder/f'{sensor}_error_offline_{rep:02d}.npy',sim['simulated_error_type'])
    meta=gen.scene_metadata(scene,folder)
    meta.update(d,scenario=scenario,level=level,condition=condition,truth=truth,
                interior_bottom_z_m=-d['nominal_liquid_height_m'],
                camera_to_bottom_m=scene.camera_position_m[2]+d['nominal_liquid_height_m'])
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    (folder/'complete.json').write_text(json.dumps({'physical_id':gid,'condition':condition,'frames':8}))
    print('CAPTURE_COMPLETE',gid,scenario,level,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--count',type=int,default=66)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);validate()
    for gid in range(a.worker,a.count,a.workers):
        for scenario in SCENES:
            for level in LEVELS:
                render_condition(a.root,gid,scenario,level)
