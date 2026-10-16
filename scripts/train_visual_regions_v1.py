"""Train/calibrate on disjoint installations; leave test images unopened until eval."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

from liquid_depth.visual_region_model import VisualRegionNet
from liquid_depth.supported_scene_meter import FormalSceneMeter
from evaluate_shared_empty_v45 import install


class Samples(Dataset):
    def __init__(self,paths,seed,augment=False,balanced=False):
        self.augment=augment;self.balanced=balanced;self.items=[]
        for p in paths:
            rgb=cv2.cvtColor(cv2.imread(str(p/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            labels=np.load(p/'labels_offline.npz')['semantic'].astype(np.int64)
            item=install(int(p.name),p.parent,p.parent/'dry',seed,max_annotation_attempts=6,source_slot_stride=1)
            meter=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.)
            roi=meter.ordinary_container_roi.astype(np.float32)
            self.items.append((np.concatenate([rgb,roi[...,None]],-1).transpose(2,0,1),labels))

    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        x,y=self.items[i];x=x.copy();y=y.copy()
        if self.augment:
            if self.balanced and (y==3).any() and np.random.random()<.65:
                # Train-only crop and resize; no synthetic depth points are created.
                positions=np.argwhere(y==3);cy,cx=positions[np.random.randint(len(positions))]
                size=int(np.random.choice([48,72,96]));sy=max(0,min(int(cy)-size//2,y.shape[0]-size));sx=max(0,min(int(cx)-size//2,y.shape[1]-size))
                x=cv2.resize(x[:,sy:sy+size,sx:sx+size].transpose(1,2,0),(y.shape[1],y.shape[0])).transpose(2,0,1)
                y=cv2.resize(y[sy:sy+size,sx:sx+size].astype(np.uint8),(y.shape[1],y.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int64)
            if np.random.random()<.5:x=x[:,:,::-1].copy();y=y[:,::-1].copy()
            gain=np.random.uniform(.65,1.35);x[:3]=np.clip(x[:3]*gain+np.random.normal(0,.008,x[:3].shape),0,1)
            if np.random.random()<.15:x[3]=1 # Avoid learning a simulator-specific aperture.
        return torch.from_numpy(x),torch.from_numpy(y)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=2026101501)
    p.add_argument('--balanced',action='store_true');p.add_argument('--initialize',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed);np.random.seed(a.seed % (2**32));torch.set_num_threads(4)
    folders=sorted(p for p in a.root.iterdir() if p.name.isdigit() and (p/'frames.json').exists())
    # Three complete blocks, each has all 11 depth bins / 3 sensors / 6 scenes.
    train=[p for p in folders if int(p.name)//198==0]
    validation=[p for p in folders if int(p.name)//198==1]
    test=[p for p in folders if int(p.name)//198>=2]
    if not train or not validation or not test:raise ValueError('need_594_independent_installations')
    split={k:[str(p) for p in v] for k,v in [('train',train),('validation',validation),('test',test)]}
    (a.output/'split.json').write_text(json.dumps(split,indent=2))
    trainset=Samples(train,a.seed,True,a.balanced);valset=Samples(validation,a.seed)
    sampler=torch.utils.data.WeightedRandomSampler([5. if (y==3).any() else 1. for _,y in trainset.items],len(trainset),replacement=True) if a.balanced else None
    loader=DataLoader(trainset,batch_size=12,shuffle=sampler is None,sampler=sampler,num_workers=0)
    valloader=DataLoader(valset,batch_size=12)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=VisualRegionNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    if a.initialize:
        model.load_state_dict(torch.load(a.initialize,map_location=device,weights_only=False)['model'])
        for group in opt.param_groups:group['lr']=.0005
    counts=np.bincount(np.concatenate([y.ravel() for _,y in trainset.items]),minlength=4)
    weight=np.sqrt(counts.sum()/np.maximum(counts,1));weight/=weight.mean()
    lossfn=nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=device))
    best=float('inf');history=[]
    for epoch in range(a.epochs):
        model.train();losses=[]
        for x,y in loader:
            x=x.to(device);y=y.to(device);opt.zero_grad();logits=model(x)
            if a.balanced:
                cross=nn.functional.cross_entropy(logits,y,weight=lossfn.weight,reduction='none')
                pt=logits.softmax(1).gather(1,y[:,None]).squeeze(1)
                ce=(cross*(1-pt).pow(2)).mean()
            else:ce=lossfn(logits,y)
            probs=logits.softmax(1);target=nn.functional.one_hot(y,4).permute(0,3,1,2)
            dice=1-((2*(probs*target).sum((0,2,3))+1)/(probs.sum((0,2,3))+target.sum((0,2,3))+1)).mean()
            loss=ce+.5*dice;loss.backward();opt.step();losses.append(float(loss.detach()))
        model.eval();vl=[]
        with torch.no_grad():
            for x,y in valloader:vl.append(float(lossfn(model(x.to(device)),y.to(device))))
        row={'epoch':epoch+1,'train_loss':float(np.mean(losses)),'validation_loss':float(np.mean(vl))};history.append(row)
        if row['validation_loss']<best:
            best=row['validation_loss'];torch.save({'model':model.state_dict(),'temperature':1.,'classes':['background_wall','dry_bottom','liquid','floating_object'],'input':'RGB + installed ROI; no depth or truth'},a.output/'best.pt')
        print(json.dumps(row),flush=True)
    state=torch.load(a.output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(state['model']);model.eval()
    # Temperature chosen only on validation; ECE on test is reported separately.
    logits=[];targets=[]
    with torch.no_grad():
        for x,y in valloader:
            logits.append(model(x.to(device)).permute(0,2,3,1).reshape(-1,4)[::31].cpu())
            targets.append(y.reshape(-1)[::31])
    logits=torch.cat(logits);targets=torch.cat(targets)
    values=[(float(nn.functional.cross_entropy(logits/t,targets)),float(t)) for t in np.geomspace(.5,4,31)]
    state['temperature']=min(values)[1];state['seed']=a.seed
    state['split_sha256']=hashlib.sha256((a.output/'split.json').read_bytes()).hexdigest()
    torch.save(state,a.output/'best.pt')
    (a.output/'training.json').write_text(json.dumps({'epochs':history,'device':device,'temperature':state['temperature'],'parameters':sum(p.numel() for p in model.parameters()),'class_counts':counts.tolist(),'balanced_crops_and_focal':a.balanced},indent=2))
    print('TRAINING_COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
