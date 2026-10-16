#!/usr/bin/env python3
"""Blender-only RGB-matched full-frame sensor proxy; geometry stays offline."""
import argparse,csv,json,random,sys,time
from dataclasses import fields,replace
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen

def tree_for(objects):
    graph=bpy.context.evaluated_depsgraph_get()
    vertices=[]; faces=[]
    for obj in objects:
        evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh()
        start=len(vertices);mat=obj.matrix_world
        vertices.extend(tuple(mat@v.co) for v in mesh.vertices)
        faces.extend(tuple(start+i for i in p.vertices) for p in mesh.polygons)
        evaluated.to_mesh_clear()
    return BVHTree.FromPolygons(vertices,faces,all_triangles=False) if faces else None

def cast(scene,tree):
    origin,directions,transform=gen._simulation._camera_rays(scene)
    flat=directions.reshape(-1,3); origin=Vector(origin)
    axial=-(flat@transform[:3,:3])[:,2]
    values=np.zeros(len(flat),np.float32)
    for i,direction in enumerate(flat):
        hit=tree.ray_cast(origin,Vector(direction),30.) if tree else (None,)*4
        if hit[0] is not None: values[i]=hit[3]*axial[i]
    return values.reshape(scene.height,scene.width)

def context_proxy(scene,raw,target,mask,*,front_wall_returns=True):
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    opaque=cast(scene,tree_for([o for o in objects if o.name not in ('liquid_volume','transparent_container')]))
    wall=cast(scene,tree_for([o for o in objects if o.name=='transparent_container']))
    rng=np.random.default_rng(scene.seed+scene.index*70001+116)
    mask=mask.astype(bool)
    occluded=mask&(opaque>0)&(opaque<target-np.maximum(.003,target*.002))
    valid_mask=mask&~occluded
    firstwall=(wall>0)&((opaque==0)|(wall<opaque))
    backing=opaque.copy()
    # Glass can return from the shell or transmit to a background surface.
    wall_return=firstwall&(rng.random(raw.shape)<rng.uniform(.15,.55))
    backing[wall_return]=wall[wall_return]
    sigma=.0008+.0012*backing**2
    noisy=backing+rng.normal(0,sigma)
    drop=rng.random(raw.shape)<rng.uniform(.08,.30)
    drop |= firstwall&(rng.random(raw.shape)<rng.uniform(.10,.35))
    noisy[(backing<=0)|drop]=0
    full=np.where(valid_mask,raw,noisy)
    # Front glass returns can also contaminate rays through the liquid.
    front=valid_mask&firstwall&(wall<target-.002)&(rng.random(raw.shape)<.20) if front_wall_returns else np.zeros_like(mask)
    full[front]=noisy[front]
    full=np.round(full/.001)*.001
    full=np.where(np.isfinite(full)&(full>=.1)&(full<=10),full,0).astype(np.float32)
    return full,np.where(valid_mask,target,0).astype(np.float32),valid_mask.astype(np.uint8),opaque,wall

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--manifest',type=Path);ap.add_argument('--fresh',action='store_true')
    ap.add_argument('--count',type=int,default=128);ap.add_argument('--seed',type=int,default=20260916031)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.csv').exists():raise FileExistsError(args.output)
    rows=list(csv.DictReader(args.manifest.open())) if args.manifest else [None]*args.count
    if args.limit:rows=rows[:args.limit]
    records=[];audit=[];start=time.time()
    names={f.name for f in fields(gen._simulation.SyntheticScene)}
    for i,row in enumerate(rows):
        folder=args.output/'samples'/f'{i:06d}';folder.mkdir(parents=True,exist_ok=True)
        if row is None:
            material=('transparent','translucent')[i%2];band=(i//2)%4
            low,high=((.1,.3),(.3,1),(1,3),(3,10))[band]
            schedule=gen._simulation.SCENARIO_PROFILES['calibration']
            index=schedule.index(material)+len(schedule)*((i//8)%8)
            scene=gen.sample_scene(index,seed=args.seed+i*1009,width=320,height=180,min_distance_m=low,max_distance_m=high,scenario_profile='calibration',camera_profile='industrial_top')
            scene=replace(scene,split='test',sequence_id=f'v16_{args.seed}_{i:04d}')
            render_seed=args.seed+i*104729
        else:
            m=json.loads(Path(row['metadata_path']).read_text())
            scene=gen._simulation.SyntheticScene(**{k:m[k] for k in names})
            render_seed=scene.seed+scene.index*104729
        rng=random.Random(render_seed)
        gen.clear_scene();gen.create_environment(scene,rng);gen.create_container(scene)
        gen.create_liquid(scene);gen.create_floating_objects(scene,rng)
        gen.create_camera(scene);gen.create_lighting(scene,rng)
        if row is None:
            gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=32))
            bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
            labels=gen.render_geometric_labels(scene);sensor=gen.simulate_raw_depth(scene,labels)
            rgb_path=str(folder/'rgb.png');raw=sensor['raw_depth_m']
            target=labels['target_depth_m'];mask=labels['mask']
        else:
            rgb_path=row['rgb_path'];raw=np.load(row['raw_depth_path'])
            target=np.load(row['target_depth_path']);mask=np.load(row['mask_path'])
        full,target,mask,opaque,wall=context_proxy(scene,raw,target,mask)
        for key,value in [('raw',full),('target',target),('mask',mask),('roi_raw',raw)]:
            np.save(folder/f'{key}.npy',value)
        if i<4:
            np.save(folder/'opaque_geometry.npy',opaque);np.save(folder/'wall_geometry.npy',wall)
        record=dict(rgb_path=rgb_path,raw_depth_path=str(folder/'raw.npy'),target_depth_path=str(folder/'target.npy'),mask_path=str(folder/'mask.npy'),roi_raw_path=str(folder/'roi_raw.npy'),split=scene.split,sequence_id=scene.sequence_id,scenario=scene.scenario)
        records.append(record)
        outside=mask==0
        audit.append(dict(sequence_id=scene.sequence_id,scenario=scene.scenario,outside_pixels=int(outside.sum()),outside_valid=int(((full>0)&outside).sum()),inside_pixels=int(mask.sum()),inside_valid=int(((full>0)&~outside).sum())))
        (folder/'provenance.json').write_text(json.dumps(dict(scene=gen.scene_metadata(scene,folder),render_seed=render_seed,source=row,sensor_model='v16_fullframe_ray_cast_proxy'),indent=2))
        if (i+1)%16==0:print('CONTEXT',i+1,len(rows),round(time.time()-start,1),flush=True)
    with (args.output/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    print('COMPLETE',len(records),time.time()-start,flush=True)
if __name__=='__main__':main()
