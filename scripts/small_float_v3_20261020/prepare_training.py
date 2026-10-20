import json,time,sys,argparse
from pathlib import Path
import cv2,numpy as np
REPO=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV');sys.path.insert(0,str(REPO/'scripts'))
from evaluate_shared_empty_v45 import install
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.empty_geometry_v41 import height_calibration
from liquid_depth.liquid_height_metrics import world_points
from liquid_depth.rgbd_evidence_model import observed_features
from liquid_depth.compound_tof_repair import sanitize_capture
from small_float_model import float_features,peak_roi
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/small-float-20261020')

def run(worker,workers):
    cv2.setNumThreads(1)
    for split,count in [('train',128),('validation',32),('test',100)]:
        for group in range(worker,count,workers):
            base=DATA/split/f'{group:04d}_00';deadline=time.monotonic()+21600
            while not (base/'complete.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(str(base))
                time.sleep(3)
            out=ROOT/'prepared-v2'/split/f'{group:04d}';out.mkdir(parents=True,exist_ok=True)
            init=out/'init';(init/'wet/00000').mkdir(parents=True,exist_ok=True);(init/'dry/0000').mkdir(parents=True,exist_ok=True)
            (init/'wet/00000/metadata.json').write_text((base/'metadata.json').read_text());dry=np.load(base/'dry_depth_m.npy');np.save(init/'dry/0000/dry_depth_m.npy',dry)
            item=install(0,init/'wet',init/'dry',202610203701+group+(0 if split=='train' else 10000 if split=='validation' else 20000),max_annotation_attempts=24,source_slot_stride=1)
            bundle=item['bundle'];(out/'bundle.json').write_text(json.dumps(bundle));cal=height_calibration(bundle)
            meter=FormalSceneMeter.from_empty_geometry(bundle,dry)
            for bits in (0,1,2,4,16,8,9,10,12,24):
                dest=out/f'{bits:02d}.npz'
                if dest.exists():continue
                folder=DATA/split/f'{group:04d}_{bits:02d}'
                while not (folder/'complete.json').exists():
                    if time.monotonic()>deadline:raise TimeoutError(str(folder))
                    time.sleep(3)
                raw,_=sanitize_capture(np.load(folder/'depth_m.npy'));rgb=cv2.imread(str(folder/'rgb.png'))
                x,roi=float_features(rgb,raw,dry,bundle)
                geo=np.load(folder/'offline_geometry.npz');s=geo['semantic'];err=np.load(folder/'offline_return_origin.npz')['error_type'];full=geo['true_depth_m']
                label=np.ones(s.shape,np.uint8);label[(full<=0)|(~roi)]=0;label[(s==2)&roi]=2;label[(s==3)&roi]=3
                reliable=((s==3)&(err==0)&(raw>0)&roi).astype(np.float32)
                n,cc,stats,_=cv2.connectedComponentsWithStats((s==3).astype(np.uint8))
                trueh=world_points(full,cal)[...,2]-cal.bottom_z_m
                weight=np.ones(s.shape,np.float32);top_id=0;top=-np.inf
                for k in range(1,n):
                    mask=cc==k;valid=mask&(raw>0)
                    # Equal-instance and small-component emphasis; no image
                    # resizing that invents extra independent metric samples.
                    weight[mask]=float(np.clip(80/max(mask.sum(),1),2,20))
                    if valid.any() and trueh[mask].max()>top:top=float(trueh[mask].max());top_id=k
                if top_id:weight[cc==top_id]*=2
                np.savez_compressed(dest,features=x.astype(np.float16),semantic=label,reliable=reliable.astype(np.float16),weight=weight.astype(np.float16),instances=cc.astype(np.uint16),true_height=trueh.astype(np.float32),raw=raw,offline_error=err)
            print('PREPARED',split,group,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args();run(a.worker,a.workers)
