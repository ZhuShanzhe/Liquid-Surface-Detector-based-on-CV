"""New rendered containers with matched iid/spatial/persistent noise sequences."""
import argparse,json,sys
from pathlib import Path
from dataclasses import fields as dataclass_fields,replace
import numpy as np,bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
import generate_ordinary_v35 as base
from generate_fullframe_v16 import cast,tree_for
SyntheticScene=base.gen._simulation.SyntheticScene
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'liquid_depth'))
from correlated_noise_v37 import fields

def generate(root,slot,seed,replicates,frames):
    base.generate(root,slot,seed,1,replicates)
    folder=root/f'{slot:05d}';meta=json.loads((folder/'metadata.json').read_text())
    scene=SyntheticScene(**{f.name:meta[f.name] for f in dataclass_fields(SyntheticScene)})
    labels=base.gen.render_geometric_labels(scene);target=labels['target_depth_m'];mask=labels['mask'].astype(bool)
    objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
    opaque=cast(scene,tree_for([o for o in objects if o.name not in ('liquid_volume','transparent_container')]))
    wall=cast(scene,tree_for([o for o in objects if o.name=='transparent_container']))
    zr=fields(target.shape,frames,seed+slot*797+351);dr=fields(target.shape,frames,seed+slot*797+357)
    records=[];audit=[];previous={}
    for repeat,(zf,df) in enumerate(zip(zr,dr)):
        varied=replace(scene,seed=scene.seed+repeat*7907)
        # Freeze fault/dropout sampling but construct noise BEFORE range selection.
        clean=base.gen.simulate_raw_depth(varied,labels,components={k:False for k in ('depth_noise','disparity_noise','quantization','range_cutoff')})
        inlier=mask&(clean['simulated_error_type']==0)
        cos=clean['incidence_cosine'];family=scene.sensor_family
        floor,quad={'active_stereo':(.0006,.0011),'structured_light':(.0008,.0015),'tof':(.0015,.0007)}[family]
        sigma=(floor+quad*(target if family=='tof' else target**2))*(1+3*(1-cos)**2)
        for mode in ('iid','spatial','persistent'):
            noisy=target+sigma*zf[mode]
            if family!='tof':
                fb=meta['camera_intrinsics'][0][0]*(.055 if family=='active_stereo' else .035)
                sd=.18 if family=='active_stereo' else .26
                disparity=fb/np.maximum(noisy,1e-6)+sd*df[mode]
                noisy=fb/np.maximum(np.round(disparity*16)/16,1e-3)
            else:noisy=np.round(noisy/.001)*.001
            raw=clean['raw_depth_m'].copy();raw[inlier]=noisy[inlier]
            low=.008 if family=='tof' else .12
            raw[(raw<low)|(raw>10.5)]=0
            full,_=base.compose(varied,raw,target,mask,opaque,wall)
            name=f'raw_{repeat:02d}_{mode}.npy';np.save(folder/name,full)
            records.append(dict(folder=str(folder),raw_path=str(folder/name),repeat=repeat,variant=mode,slot=slot,timestamp=repeat/30))
            # Noise audits are simulator diagnostics only, never model inputs.
            a=df[mode]
            audit.append(dict(frame=repeat,mode=mode,mean=float(a.mean()),variance=float(a.var()),
                horizontal_adjacent_correlation=float(np.corrcoef(a[:,:-1].ravel(),a[:,1:].ravel())[0,1]),
                temporal_correlation=float(np.corrcoef(a.ravel(),previous[mode].ravel())[0,1]) if mode in previous else None))
            previous[mode]=a
    meta.update(protocol='ordinary_v37',sequence_frames=frames,noise_block_px=8,spatial_variance_fraction=.5,temporal_field_rho=.9,
        scene_static=True,fps=30,normalization='ensemble unit variance; no per-frame centering or renormalization')
    (folder/'metadata.json').write_text(json.dumps(meta,indent=2));(folder/'frames.json').write_text(json.dumps(records))
    (folder/'noise-audit.json').write_text(json.dumps(audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--replicates',type=int,default=9);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.root.mkdir(exist_ok=True,parents=True)
    for slot in range(a.worker,33*a.replicates,a.workers):
        generate(a.root,slot,a.seed,a.replicates,a.frames);print('DONE',slot,flush=True)
