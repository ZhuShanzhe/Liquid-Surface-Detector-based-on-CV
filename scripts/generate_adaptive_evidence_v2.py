"""Expanded RGB-D floating-object diversity with exact offline origin supervision."""
import argparse
import json
import math
import sys
from dataclasses import fields, replace
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_visual_evidence_v1 as previous
from generate_synthetic_liquid import principled_material

gen=previous.gen
original_scene=previous.make_scene
RANDOM_HEADROOM=False


def expanded_scene(slot,seed):
    scene,extra=original_scene(slot,seed)
    if RANDOM_HEADROOM:
        rng=np.random.default_rng(seed+slot*317)
        scene=replace(scene,container_rim_z_m=float(rng.uniform(.02,max(.025,scene.camera_position_m[2]*.25))))
    return replace(scene,width=480,height=270),extra


def diverse_floats(scene,rng):
    if scene.container_bottom_z_m+scene.wall_thickness_m>=-1e-6:return
    count=scene.floating_object_count
    if not count and scene.scenario in {'glare','low_light','depth_failure'} and rng.random()<.35:
        count=rng.randint(1,5)
    if not count:return
    fx=scene.width/(2*math.tan(math.radians(scene.horizontal_fov_deg/2)))
    for i in range(count):
        angle=rng.uniform(0,2*math.pi);rad=math.sqrt(rng.uniform(0,.65))
        x=scene.surface_radius_x_m*rad*math.cos(angle);y=scene.surface_radius_y_m*rad*math.sin(angle)
        surface=float(gen.surface_height_and_gradient(scene,np.array(x),np.array(y))[0])
        pixels=rng.choice([3,5,8,12,20,32,48])
        radius=min(scene.surface_radius_x_m*.22,pixels*scene.camera_position_m[2]/(2*fx))
        top=rng.uniform(.03,.3)
        shape=rng.choice(['block','ellipsoid','bottle','foam'])
        center=(x,y,surface+top*.5)
        if shape in {'block','foam'}:
            bpy.ops.mesh.primitive_cube_add(size=1,location=center)
        elif shape=='bottle':
            bpy.ops.mesh.primitive_cylinder_add(vertices=16,radius=.5,depth=1,location=center)
        else:bpy.ops.mesh.primitive_uv_sphere_add(segments=16,ring_count=8,radius=.5,location=center)
        obj=bpy.context.object;obj.name=f'floating_object_{i:02d}_{shape}'
        obj.scale=(radius*2,radius*rng.uniform(1.,2.8),top)
        obj.rotation_euler.z=rng.uniform(0,math.pi)
        if shape=='foam':obj.scale.z*=.3
        color=tuple(rng.uniform(.04,.9) for _ in range(3))
        obj.data.materials.append(principled_material('float_diverse_'+str(i),color,
            roughness=rng.uniform(.1,.95),metallic=rng.choice([0.,0.,.4])))


def augment_observations(root,index,seed,consistent_float_sensor=False):
    folder=root/f'{index:05d}'
    meta=json.loads((folder/'metadata.json').read_text())
    names={f.name for f in fields(gen._simulation.SyntheticScene)}
    scene=gen._simulation.SyntheticScene(**{k:v for k,v in meta.items() if k in names})
    labels=np.load(folder/'labels_offline.npz');sem=labels['semantic'];water=labels['water_depth_m'];floating=labels['float_depth_m']
    origin,directions,transform=gen._simulation._camera_rays(scene)
    axial=-(directions@transform[:3,:3])[...,2]
    bottom=meta['interior_bottom_z_m']
    distance=(bottom-origin[2])/np.where(abs(directions[...,2])>1e-8,directions[...,2],np.nan)
    bottom_depth=distance*axial
    first=np.where(sem==2,water,np.where(sem==3,floating,np.where(sem==1,bottom_depth,0))).astype(np.float32)
    frames=json.loads((folder/'frames.json').read_text())
    for frame in frames:
        raw=np.load(frame['raw_path']);source=np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
        rng=np.random.default_rng(seed+index*79907+frame['repeat']*71)
        # Near-field coherent ghost patches are hard negatives, not floating objects.
        if index%5==0 and (sem==2).any():
            y,x=np.indices(raw.shape);cy,cx=np.argwhere(sem==2)[rng.integers((sem==2).sum())]
            patch=((x-cx)**2/(scene.width*.025)**2+(y-cy)**2/(scene.height*.04)**2)<1
            patch&=(sem==2)&(raw>0)
            raw[patch]=np.maximum(.05,first[patch]-rng.uniform(.03,.3))
            source[patch]=5
        # Float measurements also have holes, random errors and range-dependent noise.
        fm=(sem==3)&(raw>0)
        sigma=(.0015+.0007*first) if scene.sensor_family=='tof' else (.0008+.0015*first**2)
        raw[fm]+=rng.normal(0,sigma[fm])
        if consistent_float_sensor and scene.sensor_family!='tof' and fm.any():
            z=first[fm]
            floor,quadratic,baseline,disparity_sigma=(.0006,.0011,.055,.18) if scene.sensor_family=='active_stereo' else (.0008,.0015,.035,.26)
            z=z+rng.normal(0,floor+quadratic*z*z)
            fx=scene.width/(2*np.tan(np.deg2rad(scene.horizontal_fov_deg/2)))
            disparity=fx*baseline/np.maximum(z,.001)+rng.normal(0,disparity_sigma,len(z))
            disparity=np.round(disparity*16)/16
            raw[fm]=fx*baseline/np.maximum(disparity,.001)
            sigma=np.sqrt(sigma**2+(disparity_sigma*first**2/(fx*baseline))**2)
        miss=fm&(rng.random(raw.shape)<.08);raw[miss]=0;source[miss]=0
        raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
        source[raw<=0]=0
        reliable=((source>=1)&(source<=3)&(first>0)&(abs(raw-first)<=np.maximum(.02,2*sigma))).astype(np.uint8)
        np.save(frame['raw_path'],raw);np.save(folder/f"origin_offline_{frame['repeat']:02d}.npy",source)
        np.savez_compressed(folder/f"supervision_offline_{frame['repeat']:02d}.npz",origin=source,reliable=reliable,first_depth_m=first)
    meta['pilot_generation']='adaptive_v2_multiscale_floats_rgbd';meta['float_semantic_pixels']=int((sem==3).sum())
    meta['float_noise_profile']='consistent_disparity_and_depth_proxy' if consistent_float_sensor else 'initial_depth_noise_only_float_proxy'
    meta['supervision_limit']='origin and metric reliability labels are training/offline only; they never enter runtime features'
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--count',type=int,default=792);p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--consistent-float-sensor',action='store_true')
    p.add_argument('--random-headroom',action='store_true')
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(parents=True,exist_ok=True)
    RANDOM_HEADROOM=a.random_headroom
    previous.make_scene=expanded_scene;gen.create_floating_objects=diverse_floats
    for index in range(a.start_index+a.worker,a.count,a.workers):
        previous.render_one(a.root,index,a.seed,2);augment_observations(a.root,index,a.seed,a.consistent_float_sensor)
