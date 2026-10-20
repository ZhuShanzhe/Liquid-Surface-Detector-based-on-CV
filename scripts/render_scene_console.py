"""Blender data producer for the module-router experiment and interactive console.

Ground truth is written separately. Runtime gets only RGB, raw depth, empty
calibration and (for uniform refractive liquids) independent five-frame priors.
"""
import argparse, json, random, sys
from dataclasses import replace
from pathlib import Path
import bpy, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_liquid as gen
from generate_fullframe_v16 import cast, tree_for
from generate_special_scene_strata_v1 import floor_for
from generate_other_scenes_v34 import make_scene


def render(request, folder, *, corpus=False):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(request['seed']); rr = random.Random(request['seed'])
    h = float(request['height_m']); b = float(request['bottom_m']); cap = float(request['capacity_m'])
    material = request.get('material', 'ordinary_diffuse')
    dark, glare = float(request.get('darkness', 0)), float(request.get('glare', 0))
    wave, count = float(request.get('wave_m', 0)), int(request.get('float_count', 0))
    radius = float(request.get('radius_m', min(.9, (b-cap)*.32)))
    rx, ry = radius, radius * request.get('aspect', 1.)
    base, _ = make_scene(0, request['seed'])
    scene = replace(base, seed=request['seed'], index=request.get('index', 0),
        width=384 if corpus else 480, height=216 if corpus else 270, horizontal_fov_deg=54.,
        # Composition is described by independent controls. `compound` in the
        # legacy sensor proxy ALSO enables refractive bias/multipath, and must
        # not be selected just because an opaque liquid has glare.
        sensor_family='tof', scenario='glare' if glare else 'low_light' if dark else 'ordinary',
        container_shape=request.get('shape', 'cylindrical'),
        container_exponent=10. if request.get('shape') == 'rectangular_tank' else 2.,
        surface_radius_x_m=rx, surface_radius_y_m=ry,
        # Source mesh origin follows nominal surface. After shifting to bottom=0,
        # camera and look-at remain fixed when the liquid-height slider changes.
        camera_position_m=(.025*b, -.025*b, b-h),
        camera_target_m=(0., 0., -h), container_bottom_z_m=-h-.003,
        wall_thickness_m=.003, container_rim_z_m=cap-h, floating_object_count=count,
        wave_amplitude_m=wave, wave_frequency_x=float(rng.uniform(3, 9)),
        wave_frequency_y=float(rng.uniform(3, 9)), wave_phase=float(rng.uniform(0, 6.28)),
        liquid_turbidity=.97 if material == 'ordinary_diffuse' else .03 if material.endswith('transparent') else .35,
        liquid_color_rgb=tuple(rng.uniform(.15, .7, 3)),
        liquid_roughness=.025 if glare else .22,
        light_level=max(.015, 1-dark*.985), exposure=-.2-3*dark,
        corruption_severity=max(.18, dark*.7, glare*.8))
    gen.clear_scene(); gen.create_environment(scene, rr); gen.create_container(scene)
    floor = floor_for(scene, rng)
    gen.create_camera(scene); gen.create_lighting(scene, rr)
    gen.configure_render(scene, argparse.Namespace(engine='eevee', render_samples=4 if corpus else 12))
    # Raster floor texture makes refractive appearance nontrivial, varied per install.
    mat = floor.data.materials[0]; mat.use_nodes=True
    nodes=mat.node_tree.nodes; links=mat.node_tree.links
    checker=nodes.new('ShaderNodeTexChecker'); checker.inputs['Scale'].default_value=8+rng.uniform(0,16)
    checker.inputs['Color1'].default_value=(*rng.uniform(.05,.5,3),1)
    checker.inputs['Color2'].default_value=(*rng.uniform(.55,.95,3),1)
    links.new(checker.outputs['Color'], nodes.get('Principled BSDF').inputs['Base Color'])
    before=[o for o in bpy.context.scene.objects if o.type=='MESH']
    dry=cast(scene, tree_for(before))
    dry=np.where(dry>0,np.maximum(dry+rng.normal(0,.001+.00015*b,dry.shape),0),0).astype(np.float32)
    np.save(folder/'dry_depth_m.npy',dry)
    empty_floor=cast(scene,tree_for([floor]))
    empty_roi=(empty_floor>0)&(abs(empty_floor-dry)<.02)
    liquid=gen.create_liquid(scene) if h>0 else None
    if liquid and material=='ordinary_diffuse':
        bsdf=liquid.data.materials[0].node_tree.nodes.get('Principled BSDF')
        gen._set_input(bsdf,'Transmission Weight',0.);gen._set_input(bsdf,'Transmission',0.);gen._set_input(bsdf,'Alpha',1.)
    gen.create_floating_objects(scene,rr)
    floats=[o for o in bpy.context.scene.objects if o.name.startswith('floating_object_') and o.type=='MESH']
    bpy.context.scene.render.filepath=str(folder/'rgb.png');bpy.ops.render.render(write_still=True)
    if True:
        allmesh=[o for o in bpy.context.scene.objects if o.type=='MESH']
        full=cast(scene,tree_for(allmesh)); water=cast(scene,tree_for([liquid])) if liquid else cast(scene,tree_for([floor]))
        fdepth=cast(scene,tree_for(floats)) if floats else np.zeros_like(full)
        visible=(water>0)&(abs(full-water)<.001)
        semantic=np.zeros_like(full,dtype=np.uint8);semantic[visible]=2
        semantic[(fdepth>0)&(abs(full-fdepth)<.001)]=3
        labels=gen.render_geometric_labels(scene) if liquid else {'mask':visible.astype(np.uint8),'target_depth_m':water}
        labels['target_depth_m']=np.where(semantic==3,fdepth,water).astype(np.float32)
        labels['mask']=((semantic==2)|(semantic==3)).astype(np.uint8)
        sim=gen.simulate_raw_depth(scene,labels) if liquid else {'raw_depth_m':full}
        # Raycast miss remains invalid. Never turn a missing background into a
        # tiny positive depth that can back-project next to the camera.
        background=np.where(full>0,full+rng.normal(0,.001,full.shape),0)
        raw=np.where((semantic==2)|(semantic==3),sim['raw_depth_m'],background)
        # The established proxy has non-normalized flying weights. Replace only
        # flying samples in THIS demo producer by a declared convex mixture.
        if liquid and 'simulated_error_type' in sim:
            flying=sim['simulated_error_type']==3; w=rng.uniform(.2,.8,full.shape)
            raw[flying]=(w*labels['target_depth_m']+(1-w)*np.maximum(labels['target_depth_m'],dry))[flying]
        roi=(semantic==2)|(semantic==3)
        loss=float(request.get('loss',0));ids=np.flatnonzero(roi & (raw>0))
        desired=int(loss*roi.sum())-int((roi & (raw<=0)).sum())
        if desired>0 and len(ids):raw.flat[rng.choice(ids,min(desired,len(ids)),replace=False)]=0
        raw=np.where(np.isfinite(raw)&(raw>0),raw,0).astype(np.float32)
        origin=np.array(sim.get('simulated_error_type',np.zeros_like(raw)),dtype=np.uint8)
        origin[(raw<=0)&(origin==0)]=1
        np.savez_compressed(folder/'offline_return_origin.npz',error_type=origin)
        np.save(folder/'capture_valid.npy',raw>0)
        np.save(folder/'depth_m.npy',raw)
        np.savez_compressed(folder/'routing_prior.npz',dry_depth_m=dry,roi=empty_roi)
    if not corpus:
        np.savez_compressed(folder/'offline_geometry.npz',true_depth_m=full,semantic=semantic)
        # Include world vertices for the truthful 3D viewer, never pass them to meter.
        meshes=[]
        for obj in ([liquid] if liquid else [])+floats:
            verts=np.array([tuple(obj.matrix_world@v.co) for v in obj.data.vertices]);verts[:,2]+=h
            meshes.append(dict(kind='liquid' if obj==liquid else 'float',vertices=verts.tolist(),
                               triangles=[list(t.vertices) for t in triangulate(obj)]))
        gx,gy=np.meshgrid(np.linspace(-rx,rx,96),np.linspace(-ry,ry,96))
        inside=(abs(gx/rx)**scene.container_exponent+abs(gy/ry)**scene.container_exponent)<=1
        heights=gen.surface_height_and_gradient(scene,gx,gy)[0][inside]+h if liquid else np.zeros(inside.sum())
        highest=max([float(heights.max())]+[v[2] for m in meshes if m['kind']=='float' for v in m['vertices']])
        truth=dict(liquid_height_m=float(heights.mean()),highest_height_m=highest,
                   lowest_height_m=float(heights.min()),meshes=meshes)
        (folder/'truth.json').write_text(json.dumps(truth))
        metadata=gen.scene_metadata(scene,folder);metadata['interior_bottom_z_m']=-h
        metadata['nominal_liquid_height_m']=h;metadata['camera_to_bottom_m']=b
        (folder/'metadata.json').write_text(json.dumps(metadata))
    (folder/'request.json').write_text(json.dumps(request))
    (folder/'complete.json').write_text(json.dumps({'complete':True,'corpus':corpus}))


def triangulate(obj):
    obj.data.calc_loop_triangles()
    return obj.data.loop_triangles


def corpus_request(group, condition, seed):
    rng=np.random.default_rng(seed+group*104729)
    b=float(rng.uniform(1.7,9.7));cap=float(rng.uniform(.25,min(3,b*.55)))
    h=float(rng.uniform(.02,.92)*cap)
    material=('ordinary_diffuse','homogeneous_transparent','homogeneous_translucent')[condition%3]
    # Factor-balanced combinations, not an exclusive scene label.
    bits=condition//3
    return dict(seed=seed+group*104729,index=group,bottom_m=b,capacity_m=cap,height_m=h,
        radius_m=(b-cap)*float(rng.uniform(.17,.29)),aspect=1.,
        shape=('cylindrical','elliptical','rectangular_tank')[group%3], material=material,
        darkness=float((.3,.62,.93)[group%3]) if bits&1 else 0.,
        glare=float((.3,.62,.93)[group%3]) if bits&2 else 0.,
        wave_m=min(.12,h*.15) if bits&4 else 0.,float_count=12 if bits&8 else 0.,
        loss=float((.1,.3,.5)[group%3]) if bits&16 else 0.)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--corpus',action='store_true');p.add_argument('--groups',type=int,default=36)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=20261019073)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.corpus:
        for g in range(args.worker,args.groups,args.workers):
            # All single factors plus three non-conflicting compound examples.
            for bits in (0,1,2,4,8,3,9,12):
                for material in range(3):
                    c=bits*3+material;folder=args.out/f'{g:04d}_{c:02d}'
                    if not (folder/'complete.json').exists():render(corpus_request(g,c,args.seed),folder,corpus=True)
            print('CORPUS_GROUP_COMPLETE',g,flush=True)
    else:render(json.loads(args.request.read_text()),args.out)
