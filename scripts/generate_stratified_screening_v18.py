#!/usr/bin/env python3
"""Controlled two-fluid dielectric vessel. Cycles required; no Eevee fallback."""
import argparse,json,math,sys,time
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector

def glass(name,eta):
    m=bpy.data.materials.new(name);m.use_nodes=True;nt=m.node_tree;nt.nodes.clear()
    out=nt.nodes.new('ShaderNodeOutputMaterial');g=nt.nodes.new('ShaderNodeBsdfGlass')
    g.inputs['Color'].default_value=(1,1,1,1);g.inputs['Roughness'].default_value=0
    g.inputs['IOR'].default_value=eta
    if abs(g.inputs['IOR'].default_value-eta)>1e-6:raise RuntimeError('Relative IOR was clamped')
    nt.links.new(g.outputs[0],out.inputs['Surface'])
    return m
def quad(name,vertices,material):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],[(0,1,2,3)]);mesh.update()
    ob=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(ob);ob.data.materials.append(material)
    return ob
def slab_sides(x,y,z0,z1,material,prefix):
    quad(prefix+'front',[(-x,-y,z1),(x,-y,z1),(x,-y,z0),(-x,-y,z0)],material)
    quad(prefix+'right',[(x,-y,z1),(x,y,z1),(x,y,z0),(x,-y,z0)],material)
    quad(prefix+'back',[(x,y,z1),(-x,y,z1),(-x,y,z0),(x,y,z0)],material)
    quad(prefix+'left',[(-x,y,z1),(-x,-y,z1),(-x,-y,z0),(-x,y,z0)],material)
def diffuse(name,color):
    m=bpy.data.materials.new(name);m.diffuse_color=(*color,1);m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF');p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=.6
    return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=64);p.add_argument('--samples',type=int,default=32);p.add_argument('--device',choices=['CPU','GPU'],default='CPU');a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    a.output.mkdir(parents=True,exist_ok=False);rows=[];start=time.time()
    for i in range(a.count):
        group=('single','index_matched','weak','strong')[i//16];j=i%16
        t=(.002,.010,.030,.050)[j%4];distance=(.3,.6)[(j//4)%2];pattern=bool(j//8)
        n1={'single':1.333,'index_matched':1.333,'weak':1.343,'strong':1.47}[group];n2=1.333
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device=a.device
        if a.device=='GPU':
            pref=bpy.context.preferences.addons['cycles'].preferences;pref.compute_device_type='OPTIX';pref.get_devices()
            devices=[d for d in pref.devices if d.type=='OPTIX']
            if not devices:raise RuntimeError('No OptiX device; explicit CPU run required')
            for d in pref.devices:d.use=d in devices
        scene.cycles.samples=a.samples;scene.cycles.use_denoising=False;scene.cycles.use_adaptive_sampling=False
        scene.cycles.max_bounces=16;scene.cycles.transmission_bounces=12;scene.cycles.glossy_bounces=8;scene.cycles.seed=18100+j
        scene.render.resolution_x=320;scene.render.resolution_y=180;scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.render.image_settings.color_depth='16'
        scene.view_settings.view_transform='Standard';scene.view_settings.look='None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
        scene.world.use_nodes=True;scene.world.node_tree.nodes.get('Background').inputs[0].default_value=(.5,.55,.6,1)
        scene.world.node_tree.nodes.get('Background').inputs[1].default_value=.5
        x,y=.14,.09
        top=glass('air_upper',n1);lower=glass('air_lower',n2)
        quad('free_surface',[(-x,-y,0),(x,-y,0),(x,y,0),(-x,y,0)],top)
        if group!='single':
            quad('internal_interface',[(-x,-y,-t),(x,-y,-t),(x,y,-t),(-x,y,-t)],glass('upper_lower_relative_index',n2/n1))
            slab_sides(x,y,-t,0,top,'upper_');slab_sides(x,y,-.1,-t,lower,'lower_')
        else:slab_sides(x,y,-.1,0,top,'single_')
        base=diffuse('bottom',(.55,.55,.55));nt=base.node_tree
        if pattern:
            tex=nt.nodes.new('ShaderNodeTexChecker');tex.inputs['Color1'].default_value=(.12,.18,.3,1);tex.inputs['Color2'].default_value=(.8,.75,.55,1);tex.inputs['Scale'].default_value=45
            coord=nt.nodes.new('ShaderNodeTexCoord');nt.links.new(coord.outputs['Object'],tex.inputs['Vector']);nt.links.new(tex.outputs['Color'],nt.nodes.get('Principled BSDF').inputs['Base Color'])
        quad('opaque_bottom',[(-x,-y,-.1001),(x,-y,-.1001),(x,y,-.1001),(-x,y,-.1001)],base)
        table=diffuse('surround',(.2,.25,.28))
        quad('table',[(-1,-1,-.105),(1,-1,-.105),(1,1,-.105),(-1,1,-.105)],table)
        rim=diffuse('thin_vessel_rim',(.2,.25,.3))
        for xx,yy,scale in [(0,-y,(x+.003,.002,.002)),(0,y,(x+.003,.002,.002)),(-x,0,(.002,y,.002)),(x,0,(.002,y,.002))]:
            bpy.ops.mesh.primitive_cube_add(size=2,location=(xx,yy,.002));ob=bpy.context.object;ob.scale=scale;ob.data.materials.append(rim)
        bpy.ops.object.light_add(type='AREA',location=(.15,-.2,.55));light=bpy.context.object;light.data.energy=20;light.data.shape='DISK';light.data.size=.15
        light.rotation_euler=(Vector((0,0,0))-light.location).to_track_quat('-Z','Y').to_euler()
        bpy.ops.object.camera_add(location=(0,0,distance));camera=bpy.context.object;camera.rotation_euler=(0,0,0)
        camera.data.type='PERSP';camera.data.lens=31.1769145;camera.data.sensor_width=36;camera.data.sensor_fit='HORIZONTAL';camera.data.clip_start=.01;scene.camera=camera
        # Blender camera local -Z points vertically down with zero rotation.
        folder=a.output/f'{i:03d}';folder.mkdir();scene.render.filepath=str(folder/'rgb.png')
        bpy.ops.render.render(write_still=True)
        f=320/(2*math.tan(math.radians(60)/2))
        u,v=np.meshgrid((np.arange(320)+.5-160)/f,(np.arange(180)+.5-90)/f)
        norm=np.sqrt(1+u*u+v*v);cos0=1/norm;sin0=np.sqrt(1-cos0*cos0)
        # Geometric amodal targets; use common central support away from vessel edges.
        roi=(abs(u*(distance+.1))<x*.80)&(abs(v*(distance+.1))<y*.80)
        topmask=(abs(u*distance)<x-.003)&(abs(v*distance)<y-.003)
        dtop=np.full((180,320),distance,np.float32);dint=dtop+t
        # Nondispersive, coaxial ToF-like single-path proxy (not calibrated to a device).
        cos1=np.sqrt(1-(sin0/n1)**2);cos2=np.sqrt(1-(sin0/n2)**2)
        r0=((n1-1)/(n1+1))**2;ri=0 if group=='single' else ((n2-n1)/(n2+n1))**2
        weights=np.array([r0,(1-r0)**2*ri,(1-r0)**2*(1-ri)**2*.15]);weights/=weights.sum()
        rng=np.random.default_rng(20260918001+j);draw=rng.random((180,320));event=np.searchsorted(np.cumsum(weights),draw)
        optical_internal=distance+n1*t*cos0/cos1
        optical_bottom=distance+n1*t*cos0/cos1+n2*(.1-t)*cos0/cos2
        if group=='single':optical_bottom=distance+n1*.1*cos0/cos1
        raw=np.choose(event,[dtop,optical_internal,optical_bottom])
        raw=np.where(topmask,raw,distance+.105)
        raw=raw+rng.normal(0,.001+.001*raw**2,raw.shape);raw[rng.random(raw.shape)<.25]=0
        raw=np.maximum(np.round(raw*1000)/1000,0).astype(np.float32)
        np.savez_compressed(folder/'arrays.npz',raw=raw,roi=roi,top_mask=topmask,top=dtop,internal=dint,event=event)
        meta=dict(id=i,pair=j,group=group,thickness_m=t if group!='single' else None,camera_distance_m=distance,pattern=pattern,n_upper=n1,n_lower=n2,true_liquid_interfaces=1 if group=='single' else 2,render_engine=scene.render.engine,render_device=a.device,samples=a.samples,adaptive_sampling=False,denoising=False,transmission_bounces=scene.cycles.transmission_bounces,shared_interface_count=0 if group=='single' else 1,shared_interface_ior=None if group=='single' else n2/n1,raw_proxy='nondispersive_coaxial_single_path_tof_like_not_device_calibrated',raw_event_weights=weights.tolist(),camera=dict(width=320,height=180,fx=f,fy=f,cx=160,cy=90,depth_scale_to_m=1,depth_alignment='color'),assumptions='flat lossless media; side boundary air; opaque base; no glass shell; opaque rim; no dispersion/absorption/multipath sensor response')
        (folder/'scene.json').write_text(json.dumps(meta,indent=2));rows.append(meta)
        print('RENDER',i+1,a.count,group,round(time.time()-start,1),flush=True)
    (a.output/'manifest.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
