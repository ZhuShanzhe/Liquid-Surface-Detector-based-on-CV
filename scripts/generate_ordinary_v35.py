"""Correct dry floor and independent per-repeat full-frame sensor composition."""
import argparse,json,random,sys,time,math
from dataclasses import replace
from pathlib import Path
import bpy,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_other_scenes_v34 import make_scene
from generate_fullframe_v16 import tree_for,cast


def add_floor(scene):
    z=scene.container_bottom_z_m+scene.wall_thickness_m
    angles=np.linspace(0,2*np.pi,256,endpoint=False)
    e=scene.container_exponent
    verts=[(scene.surface_radius_x_m*np.sign(np.cos(t))*abs(np.cos(t))**(2/e),
            scene.surface_radius_y_m*np.sign(np.sin(t))*abs(np.sin(t))**(2/e),z) for t in angles]
    mesh=bpy.data.meshes.new('interior_floor_mesh');mesh.from_pydata(verts,[],[list(range(len(verts)))]);mesh.update()
    obj=bpy.data.objects.new('interior_floor',mesh);bpy.context.collection.objects.link(obj)
    mat=bpy.data.materials.new('opaque_floor');mat.use_nodes=True
    bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=(.24,.28,.3,1)
    bs.inputs['Roughness'].default_value=.5;obj.data.materials.append(mat)
    return obj


def compose(scene,raw,target,mask,opaque,wall):
    rng=np.random.default_rng(scene.seed+scene.index*70001+351)
    mask=mask.astype(bool)
    visible=mask&~((opaque>0)&(opaque<target-np.maximum(.003,target*.002)))
    firstwall=(wall>0)&((opaque==0)|(wall<opaque));backing=opaque.copy()
    wr=firstwall&(rng.random(raw.shape)<.3);backing[wr]=wall[wr]
    noisy=backing+rng.normal(0,.0008+.0012*backing**2)
    noisy[(backing<=0)|(rng.random(raw.shape)<.15)]=0
    full=np.where(visible,raw,noisy)
    front=visible&firstwall&(wall<target-.002)&(rng.random(raw.shape)<.2)
    full[front]=noisy[front];full=np.round(full/.001)*.001
    low=.008 if scene.sensor_family=='tof' else .12
    full=np.where(np.isfinite(full)&(full>=low)&(full<=10.5),full,0).astype(np.float32)
    return full,visible


def generate(root,slot,seed,repeats,replicates):
    band=slot//(3*replicates);sensor=(slot//replicates)%3;rep=slot%replicates
    oldcase=band*9+sensor*3+rep%3
    scene,extra=make_scene(oldcase*6,seed+rep//3*170003)
    rng=np.random.default_rng(seed+slot*39019)
    h=extra['nominal_liquid_height_m'];z=scene.camera_position_m[2]
    # Container and camera range are installation quantities, never estimator labels.
    capacity=h+min(.3,z*.25)*float(rng.uniform(.5,1))
    scene=replace(scene,index=slot,seed=seed+slot*39019,sequence_id=f'ordinary35_{seed}_{slot}',
        container_rim_z_m=capacity-h,container_material='glass',corruption_severity=float(rng.uniform(.1,.55)))
    extra.update(case=slot,cluster=f'{seed}_{slot}',slot=slot,replicate=rep,
        dry_floor_present=True,capacity_m=capacity,empty=h==0,protocol='ordinary_v35')
    folder=root/f'{slot:05d}';folder.mkdir(exist_ok=False)
    rr=random.Random(seed+slot*9173)
    gen.clear_scene();gen.create_environment(scene,rr);shell=gen.create_container(scene);floor=add_floor(scene)
    if h>0:
        obj=gen.create_liquid(scene);bs=obj.data.materials[0].node_tree.nodes.get('Principled BSDF')
        gen._set_input(bs,'Transmission Weight',0.);gen._set_input(bs,'Transmission',0.);gen._set_input(bs,'Alpha',1.)
    gen.create_camera(scene);gen.create_lighting(scene,rr)
    gen.configure_render(scene,argparse.Namespace(engine='eevee',render_samples=16))
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    labels=gen.render_geometric_labels(scene)
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    opaque=cast(scene,tree_for([o for o in objects if o.name not in ('liquid_volume','transparent_container')]))
    wall=cast(scene,tree_for([o for o in objects if o.name=='transparent_container']))
    fd=cast(scene,tree_for([floor]));mask=labels['mask'].astype(bool)
    # This audit verifies actual Blender floor hits, not just an analytic zero target.
    floor_error=None
    if h==0:
        valid=mask&(fd>0);floor_error=float(np.max(abs(fd[valid]-labels['target_depth_m'][valid])))
        assert valid.sum()>.98*mask.sum() and floor_error<1e-4,(valid.sum(),mask.sum(),floor_error)
    records=[]
    for repeat in range(repeats):
        varied=replace(scene,seed=scene.seed+repeat*7907)
        sensor_depth=gen.simulate_raw_depth(varied,labels)['raw_depth_m']
        raw,visible=compose(varied,sensor_depth,labels['target_depth_m'],mask,opaque,wall)
        name=f'raw_{repeat:02d}_base.npy';np.save(folder/name,raw)
        records.append(dict(folder=str(folder),raw_path=str(folder/name),repeat=repeat,variant='base',slot=slot))
    np.savez_compressed(folder/'geometry.npz',target=np.where(visible,labels['target_depth_m'],0),mask=visible,analytic_mask=mask)
    meta=gen.scene_metadata(scene,folder);meta.update(extra,dry_floor_max_ray_error_m=floor_error)
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2));(folder/'frames.json').write_text(json.dumps(records))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--replicates',type=int,default=3);p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(exist_ok=True,parents=True)
    for slot in range(a.worker,11*3*a.replicates,a.workers):
        generate(a.root,slot,a.seed,a.repeats,a.replicates);print('DONE',slot,flush=True)
