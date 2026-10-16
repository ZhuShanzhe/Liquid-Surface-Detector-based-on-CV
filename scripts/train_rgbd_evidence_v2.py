"""RGB-D multi-head training with real RGB weak priors and grouped synthetic splits."""
import argparse
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from liquid_depth.rgbd_evidence_model import RGBDEvidenceNet, observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter


def balanced_ce(logits,target):
    losses=F.cross_entropy(logits,target,ignore_index=255,reduction='none');values=[]
    for k in range(logits.shape[1]):
        mask=target==k
        if mask.any():values.append(losses[mask].mean())
    return torch.stack(values).mean() if values else logits.sum()*0


def dice(logits,target):
    valid=target!=255;safe=torch.where(valid,target,0)
    gt=F.one_hot(safe,logits.shape[1]).permute(0,3,1,2)*valid[:,None]
    p=logits.softmax(1)*valid[:,None]
    return 1-((2*(p*gt).sum((0,2,3))+1)/(p.sum((0,2,3))+gt.sum((0,2,3))+1)).mean()


class EvidenceData(Dataset):
    def __init__(self,paths,seed,augment=False):
        self.items=[];self.augment=augment
        for folder in paths:
            item=install(int(folder.name),folder.parent,folder.parent/'dry',seed,max_annotation_attempts=6,source_slot_stride=1)
            base=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.5)
            roi=base.ordinary_container_roi
            rgb=cv2.imread(str(folder/'rgb.png'));semantic=np.load(folder/'labels_offline.npz')['semantic'].astype(np.int64)
            for frame in json.loads((folder/'frames.json').read_text()):
                raw=np.load(frame['raw_path']);x=observed_features(rgb,raw,item['dry'],roi,item['bundle'])
                target=np.load(folder/f"supervision_offline_{frame['repeat']:02d}.npz")
                origin=target['origin'].astype(np.int64);origin[~roi]=255
                reliable=target['reliable'].astype(np.float32);reliable[~roi]=-1
                self.items.append((x.astype(np.float16),semantic,origin,reliable))

    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        x,s,o,r=self.items[i];x=x.astype(np.float32);s=s.copy();o=o.copy();r=r.copy()
        if self.augment:
            if (s==3).any() and np.random.random()<.5:
                yy,xx=np.argwhere(s==3)[np.random.randint((s==3).sum())]
                size=int(np.random.choice([96,144,192]));sy=max(0,min(yy-size//2,s.shape[0]-size));sx=max(0,min(xx-size//2,s.shape[1]-size))
                dest=(s.shape[1],s.shape[0]);x=cv2.resize(x[:,sy:sy+size,sx:sx+size].transpose(1,2,0),dest).transpose(2,0,1)
                s=cv2.resize(s[sy:sy+size,sx:sx+size].astype(np.uint8),dest,interpolation=cv2.INTER_NEAREST).astype(np.int64)
                o=cv2.resize(o[sy:sy+size,sx:sx+size].astype(np.uint8),dest,interpolation=cv2.INTER_NEAREST).astype(np.int64)
                r=cv2.resize(r[sy:sy+size,sx:sx+size],dest,interpolation=cv2.INTER_NEAREST)
            if np.random.random()<.5:x=x[:,:,::-1].copy();s=s[:,::-1].copy();o=o[:,::-1].copy();r=r[:,::-1].copy()
            x[:3]=np.clip(x[:3]*np.random.uniform(.75,1.25),0,1)
            if np.random.random()<.1:x[4:]=0 # RGB-only missing-modality augmentation.
        return tuple(torch.from_numpy(v) for v in (x,s,o,r))


def weak_real_patches(root,limit=160):
    if not (root/'provenance.json').exists():return [],{'status':'not_downloaded','used':0}
    result=[];names=[]
    with zipfile.ZipFile(root/'images.zip') as images,zipfile.ZipFile(root/'masks.zip') as masks:
        im={Path(n).stem:n for n in images.namelist() if n.lower().endswith(('.jpg','.png'))}
        # Official MaSTr masks use 0001m.png while RGB uses 0001.jpg.
        ma={Path(n).stem.removesuffix('m'):n for n in masks.namelist() if n.lower().endswith('.png')}
        for name in sorted(set(im)&set(ma)):
            rgb=np.asarray(Image.open(BytesIO(images.read(im[name]))).convert('RGB'))
            mask=np.asarray(Image.open(BytesIO(masks.read(ma[name]))))
            if mask.ndim!=2:continue
            water=mask==1;objects=(mask==0).astype(np.uint8)
            n,comp,stats,_=cv2.connectedComponentsWithStats(objects)
            target=np.full(mask.shape,255,np.uint8);target[water]=2
            usable=[]
            for k in range(1,n):
                x,y,w,h,area=stats[k]
                if area<9 or area>.025*mask.size or min(x,y)<2 or x+w>=mask.shape[1]-2 or y+h>=mask.shape[0]-2:continue
                component=comp==k
                ring=cv2.dilate(component.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)&~component
                known_ring=ring&(mask!=4)
                if known_ring.sum()<12 or water[known_ring].mean()<.8:continue
                target[component]=3;usable.append((x,y,w,h))
            if not usable:continue
            x,y,w,h=max(usable,key=lambda t:t[2]*t[3]);cx=x+w//2;cy=y+h//2
            size=max(96,min(max(w,h)*4,min(mask.shape)));sy=max(0,min(cy-size//2,mask.shape[0]-size));sx=max(0,min(cx-size//2,mask.shape[1]-size))
            img=cv2.resize(rgb[sy:sy+size,sx:sx+size],(480,270));label=cv2.resize(target[sy:sy+size,sx:sx+size],(480,270),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            features=np.zeros((12,270,480),np.float32);features[:3]=img.transpose(2,0,1)/255.;features[3]=1
            result.append((torch.from_numpy(features),torch.from_numpy(label)));names.append(name)
            if len(result)>=limit:break
    return result,{'status':'weak_isolated_obstacle_prior','used':len(result),'source_images':names,
        'not_metric_depth_supervision':True,'not_all_environment_pixels_are_float':True,
        'provenance_sha256':hashlib.sha256((root/'provenance.json').read_bytes()).hexdigest()}


def validation(model,loader,device):
    confusion=np.zeros((4,4),np.int64);origin_correct=0;origin_n=0
    model.eval()
    with torch.inference_mode():
        for x,s,o,_ in loader:
            output=model(x.to(device));p=output['semantic'].argmax(1).cpu().numpy();truth=s.numpy()
            confusion+=np.bincount((truth*4+p).ravel(),minlength=16).reshape(4,4)
            po=output['origin'].argmax(1).cpu();valid=o!=255
            origin_correct+=int(((po==o)&valid).sum());origin_n+=int(valid.sum())
    union=confusion.sum(0)+confusion.sum(1)-confusion.diagonal();iou=confusion.diagonal()/np.maximum(union,1)
    return {'iou':iou.tolist(),'score':float(.5*iou[3]+.3*iou[2]+.2*iou[1]),'origin_accuracy':origin_correct/max(origin_n,1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--initialize',type=Path)
    p.add_argument('--real-priors',type=Path,default=Path('/root/autodl-tmp/liquid-depth-data/external/water-obstacle-priors-v2'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);np.random.seed(a.seed%(2**32));torch.set_num_threads(4);cv2.setNumThreads(1)
    folders=sorted(p for p in a.root.iterdir() if p.name.isdigit() and (p/'frames.json').exists())
    splits={'train':[str(p) for p in folders if int(p.name)<396],
            'validation':[str(p) for p in folders if 396<=int(p.name)<594],
            'test':[str(p) for p in folders if int(p.name)>=594]}
    if [len(splits[k]) for k in ['train','validation','test']]!=[396,198,198]:raise ValueError('complete_792_installation_dataset_required')
    (a.output/'split.json').write_text(json.dumps(splits,indent=2))
    train=EvidenceData([Path(p) for p in splits['train']],a.seed,True);val=EvidenceData([Path(p) for p in splits['validation']],a.seed)
    sampler=WeightedRandomSampler([3. if (s==3).any() else 1. for _,s,_,_ in train.items],len(train),replacement=True)
    loader=DataLoader(train,batch_size=6,sampler=sampler,num_workers=0);valloader=DataLoader(val,batch_size=6)
    real,real_info=weak_real_patches(a.real_priors);(a.output/'real-priors-used.json').write_text(json.dumps(real_info,indent=2))
    device='cuda' if torch.cuda.is_available() else 'cpu';model=RGBDEvidenceNet().to(device)
    if a.initialize:model.load_state_dict(torch.load(a.initialize,map_location=device,weights_only=False)['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    if real:
        real_loader=DataLoader(real,batch_size=6,shuffle=True)
        for epoch in range(12):
            model.train();losses=[]
            for x,y in real_loader:
                opt.zero_grad();logits=model(x.to(device))['semantic'];loss=balanced_ce(logits,y.to(device))+.5*dice(logits,y.to(device))
                loss.backward();opt.step();losses.append(float(loss.detach()))
            print('REAL_WARMUP',epoch+1,float(np.mean(losses)),flush=True)
    best=-1.;history=[]
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,s,o,r in loader:
            x,s,o,r=[t.to(device) for t in [x,s,o,r]];opt.zero_grad();output=model(x)
            semantic=balanced_ce(output['semantic'],s)+.7*dice(output['semantic'],s)
            origin=balanced_ce(output['origin'],o)
            rl=F.binary_cross_entropy_with_logits(output['reliability'][:,0],r.clamp(0,1),reduction='none')
            parts=[rl[(r==k)].mean() for k in [0,1] if (r==k).any()]
            reliability=torch.stack(parts).mean()
            loss=semantic+.5*origin+.3*reliability;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**validation(model,valloader,device)};history.append(row)
        if row['score']>best:
            best=row['score'];torch.save({'model':model.state_dict(),'epoch':epoch+1,'validation':row,
                'temperature':{'semantic':1.,'origin':1.,'reliability':1.},'observed_features_only':True},a.output/'best.pt')
        print(json.dumps(row),flush=True)
    (a.output/'training.json').write_text(json.dumps({'history':history,'real_prior':real_info,'device':device},indent=2))
    print('RGBD_TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
