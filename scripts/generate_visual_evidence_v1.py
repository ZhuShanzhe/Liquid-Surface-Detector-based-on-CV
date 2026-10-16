"""Blender RGB-D pilot with exact visible semantic labels, not height pseudo-labels."""
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

SCENES = ('ordinary', 'low_light', 'glare', 'floating_objects', 'uneven_surface', 'depth_failure')


def render_one(root, index, seed, repeats):
    rng = np.random.default_rng(seed + index*100003)
    scenario = SCENES[index % len(SCENES)]
    # Independently randomized installations; balanced sensor and depth schedules.
    case = ((index//18) % 11)*9 + ((index//6) % 3)*3 + int(rng.integers(0,3))
    base, extra = make_scene(case*6 + SCENES.index(scenario), seed+index*701)
    severity = float(rng.choice([.12, .32, .52, .72]))
    scene = replace(base, index=index, seed=seed+index*100003,
                    corruption_severity=severity,
                    floating_object_count=int(rng.integers(2, 10)) if scenario=='floating_objects' and extra['nominal_liquid_height_m']>0 else 0,
                    light_level=float(rng.uniform(.12, .5)) if scenario=='low_light' else float(rng.uniform(.6, 1.3)),
                    exposure=float(rng.uniform(-2., -.3)) if scenario=='low_light' else float(rng.uniform(-1.3, -.3)),
                    liquid_color_rgb=tuple(rng.uniform(.05,.85,3)))
    folder=root/f'{index:05d}'
    folder.mkdir(parents=True, exist_ok=False)
    py_rng=random.Random(seed+index*104729)
    gen.clear_scene(); gen.create_environment(scene,py_rng); gen.create_container(scene)
    # A real dry interior floor, shared by the empty and wet observations.
    count=128
    bottom=scene.container_bottom_z_m+scene.wall_thickness_m
    vertices=[(*gen.container_perimeter_point(scene,2*np.pi*i/count,bottom)[:2],bottom) for i in range(count)]
    mesh=bpy.data.meshes.new('interior_floor_mesh'); mesh.from_pydata(vertices,[],[tuple(range(count))]);mesh.update()
    floor=bpy.data.objects.new('interior_floor',mesh);bpy.context.collection.objects.link(floor)
    mat=gen.principled_material('randomized_floor',tuple(rng.uniform(.1,.8,3)),roughness=float(rng.uniform(.25,.9)))
    tex=mat.node_tree.nodes.new('ShaderNodeTexNoise');tex.inputs['Scale'].default_value=float(rng.uniform(4,18))
    ramp=mat.node_tree.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color=(*rng.uniform(.03,.4,3),1.)
    ramp.color_ramp.elements[1].color=(*rng.uniform(.4,.9,3),1.)
    mat.node_tree.links.new(tex.outputs['Fac'],ramp.inputs['Fac'])
    mat.node_tree.links.new(ramp.outputs['Color'],mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    floor.data.materials.append(mat)
    gen.create_camera(scene);gen.create_lighting(scene,py_rng)
    gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=8))
    # Labels are geometric first-hit classes. They never enter inference.
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    dry_geometric=cast(scene,tree_for(objects))
    noise=.001+.0005*dry_geometric**2
    dry=dry_geometric+rng.normal(0,noise)
    dry[(dry_geometric<=0)|(rng.random(dry.shape)<.04)]=0
    dry=np.where((dry>0)&(dry<=10.5),dry,0).astype(np.float32)
    dryfolder=root/'dry'/f'{index:04d}';dryfolder.mkdir(parents=True)
    np.save(dryfolder/'dry_depth_m.npy',dry)
    if extra['nominal_liquid_height_m']>0:
        obj=gen.create_liquid(scene)
        bsdf=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        gen._set_input(bsdf,'Transmission Weight',0.);gen._set_input(bsdf,'Transmission',0.);gen._set_input(bsdf,'Alpha',1.)
    gen.create_floating_objects(scene,py_rng)
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(scene)
    floating=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('floating')]
    float_depth=cast(scene,tree_for(floating))
    floor_depth=cast(scene,tree_for([floor]))
    other=cast(scene,tree_for([o for o in objects if o!=floor]))
    water=labels['target_depth_m'] if extra['nominal_liquid_height_m']>0 else np.zeros_like(floor_depth)
    candidates=np.stack([other,floor_depth,water,float_depth])
    nearest=np.argmin(np.where(candidates>0,candidates,np.inf),axis=0)
    any_hit=(candidates>0).any(axis=0)
    semantic=np.where(any_hit,nearest,0).astype(np.uint8)
    # Class 0 background/wall, 1 dry bottom, 2 liquid, 3 floating solid.
    visible_water=(semantic==2)
    visible_float=(semantic==3)
    _,directions,_=gen._simulation._camera_rays(scene)
    origin=np.asarray(scene.camera_position_m)
    transform=gen.camera_to_world(scene)
    axial=-(directions@transform[:3,:3])[...,2]
    xyz=origin+directions*(np.where(float_depth>0,float_depth,0)/np.maximum(axial,1e-6))[...,None]
    float_height=xyz[...,2]-bottom
    # Physical area mean and mesh apex, not a visibility-dependent or pixel maximum truth.
    gx,gy=np.meshgrid(np.linspace(-scene.surface_radius_x_m,scene.surface_radius_x_m,192),
                      np.linspace(-scene.surface_radius_y_m,scene.surface_radius_y_m,192))
    area=(abs(gx/scene.surface_radius_x_m)**scene.container_exponent+abs(gy/scene.surface_radius_y_m)**scene.container_exponent)<=1
    physical=gen.surface_height_and_gradient(scene,gx,gy)[0][area]-bottom
    truth={'liquid_height_m':float(np.mean(physical)) if extra['nominal_liquid_height_m']>0 else 0.,
           'highest_height_m':float(np.max(physical)) if extra['nominal_liquid_height_m']>0 else 0.,
           'definition':'physical area-mean liquid height and geometric top mesh height above interior bottom'}
    for obj in floating:
        truth['highest_height_m']=max(truth['highest_height_m'],max(float((obj.matrix_world@v.co).z-bottom) for v in obj.data.vertices))
    np.savez_compressed(folder/'labels_offline.npz',semantic=semantic,water_depth_m=water,float_depth_m=float_depth)
    records=[]
    for repeat in range(repeats):
        components={'scenario_dropout':False} if scenario=='depth_failure' else None
        sim=gen.simulate_raw_depth(replace(scene,seed=scene.seed+repeat*79907),labels,components=components)
        raw=np.where(visible_water,sim['raw_depth_m'],dry_geometric+rng.normal(0,.001+.0007*dry_geometric))
        raw[visible_float]=float_depth[visible_float]+rng.normal(0,.001+.0007*float_depth[visible_float])
        origin_label=np.where(visible_water,1,np.where(visible_float,2,np.where(semantic==1,3,4))).astype(np.uint8)
        origin_label[visible_water & (sim['simulated_error_type']==2)]=5
        origin_label[visible_water & (sim['simulated_error_type']==3)]=6
        if scenario=='depth_failure':
            loss=float(rng.uniform(0,.5));u=rng.random(raw.shape)
            raw[visible_water & (u<loss)]=0
        raw[(rng.random(raw.shape)<.04)|~any_hit]=0
        raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
        origin_label[raw<=0]=0
        np.save(folder/f'raw_{repeat:02d}.npy',raw)
        np.save(folder/f'origin_offline_{repeat:02d}.npy',origin_label)
        records.append({'raw_path':str(folder/f'raw_{repeat:02d}.npy'),'repeat':repeat})
    meta=gen.scene_metadata(scene,folder)
    meta.update(extra,case=index,scenario=scenario,interior_bottom_z_m=bottom,
                camera_to_bottom_m=float(scene.camera_position_m[2]-bottom),
                truth=truth,seed=scene.seed,nominal_liquid_height_m=extra['nominal_liquid_height_m'],
                limitation='Parameterized sensor, opaque diffuse liquid pilot; not refractive image formation or real camera qualification.')
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))
    (folder/'frames.json').write_text(json.dumps(records))
    print('GENERATED',index,scenario,scene.sensor_family,truth,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--count',type=int,default=396)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--repeats',type=int,default=2)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(parents=True,exist_ok=True)
    for i in range(a.start_index+a.worker,a.count,a.workers): render_one(a.root,i,a.seed,a.repeats)
