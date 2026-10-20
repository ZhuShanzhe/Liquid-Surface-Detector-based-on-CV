import json,time,random
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from liquid_depth.rgbd_evidence_model import RGBDEvidenceNet
ROOT=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');PREP=ROOT/'prepared-v2';OUT=ROOT/'pixel-training';OUT.mkdir(exist_ok=True)
torch.manual_seed(202610204011);np.random.seed(202610204011%(2**32));random.seed(202610204011);torch.set_num_threads(4);cv2.setNumThreads(1)

class Patches(Dataset):
    def __init__(self,paths,augment):
        self.items=[];self.augment=augment
        for p in paths:
            with np.load(p) as f:self.items.append(tuple(f[k].copy() for k in ('features','semantic','reliable','weight','instances','true_height')))
    def __len__(self):return len(self.items)*(2 if self.augment else 1)
    def __getitem__(self,i):
        x,y,r,w,cc,h=self.items[i%len(self.items)];x=x.astype(np.float32);y=y.copy();r=r.astype(np.float32);w=w.astype(np.float32)
        if self.augment:
            ys,xs=np.nonzero(y==3)
            if len(xs) and np.random.random()<.75:
                ids=np.unique(cc[y==3]);selected=int(np.random.choice(ids))
                if np.random.random()<.5:selected=max(ids,key=lambda k:float(h[cc==k].max()))
                yy,xx=np.nonzero(cc==selected);j=np.random.randint(len(xx));cy,cx=yy[j],xx[j]
            else:
                yy,xx=np.nonzero(x[3]>0);j=np.random.randint(len(xx)) if len(xx) else 0
                cy,cx=(yy[j],xx[j]) if len(xx) else (x.shape[1]//2,x.shape[2]//2)
            # GroupNorm statistics depend on spatial context. Keep the native
            # full frame exactly as inference; focus small instances through
            # balanced losses/weights rather than crop/resize distribution shift.
            if np.random.random()<.5:x=x[:,:,::-1].copy();y=y[:,::-1].copy();r=r[:,::-1].copy();w=w[:,::-1].copy()
            x[:3]=np.clip(x[:3]*np.random.uniform(.7,1.3),0,1)
        return tuple(torch.from_numpy(v) for v in (x,y.astype(np.int64),r,w))

def validation(net,loader,device):
    net.eval();tp=fp=fn=0;image_fp=neg=0;valid_float_frames=hit_frames=0
    with torch.inference_mode():
        for x,y,r,w in loader:
            out=net(x.to(device));p=out['semantic'].softmax(1)[:,3].cpu();re=out['reliability'][:,0].sigmoid().cpu()
            mask=(p>=.5)&(re>=.5)&(x[:,3]>.5);gt=y==3
            tp+=int((mask&gt).sum());fp+=int((mask&~gt).sum());fn+=int((~mask&gt).sum())
            for a,b in zip(mask.numpy(),gt.numpy()):
                if b.sum()>=3:valid_float_frames+=1;hit_frames+=bool((a&b).sum()>=3)
                else:neg+=1;image_fp+=bool(a.sum()>=3)
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);f1=2*precision*recall/max(precision+recall,1e-9)
    return dict(precision=precision,recall=recall,f1=f1,frame_recall=hit_frames/max(valid_float_frames,1),negative_image_fp=image_fp/max(neg,1),score=.7*f1+.3*precision)

while True:
    train=sorted((PREP/'train').glob('*/*.npz'));val=sorted((PREP/'validation').glob('*/*.npz'))
    if len(train)==1280 and len(val)==320:break
    print('WAIT_PREPARED',len(train),len(val),flush=True);time.sleep(30)
data=Patches(train,True);validation_data=Patches(val,False)
loader=DataLoader(data,batch_size=8,shuffle=True,num_workers=0);valloader=DataLoader(validation_data,batch_size=8)
device='cuda';model=RGBDEvidenceNet().to(device)
old=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/assets/models/v8/pixel_best.pt')
model.load_state_dict(torch.load(old,map_location='cpu',weights_only=False)['model'])
for p in model.origin.parameters():p.requires_grad=False
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4)
best=-1;history=[];start=time.monotonic()
for epoch in range(35):
    model.train();losses=[]
    for x,y,r,w in loader:
        x,y,r,w=[v.to(device) for v in (x,y,r,w)];opt.zero_grad()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            out=model(x);ce=F.cross_entropy(out['semantic'],y,reduction='none');parts=[]
            for k in range(4):
                m=y==k
                if m.any():parts.append((ce[m]*w[m]).sum()/w[m].sum())
            sem=torch.stack(parts).mean();p=out['semantic'].softmax(1)[:,3];gt=(y==3).float();dice=1-(2*(p*gt).sum()+1)/(p.sum()+gt.sum()+1)
            bce=F.binary_cross_entropy_with_logits(out['reliability'][:,0],r,reduction='none');rp=[]
            for m in (r>.5,r<=.5):
                if m.any():rp.append(bce[m].mean())
            loss=sem+.7*dice+.5*torch.stack(rp).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();losses.append(float(loss.detach()))
    v=validation(model,valloader,device);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),**v);history.append(row)
    if v['score']>best:
        best=v['score'];torch.save(dict(model={k:v.cpu() for k,v in model.state_dict().items()},epoch=epoch+1,validation=v,temperature={'semantic':1.,'origin':1.,'reliability':1.},scope='float_presence_and_metric_return_only_never_liquid',schema='small_float_v1',features='tof_linear_noise'),OUT/'best.pt')
    print(json.dumps(row),flush=True)
    (OUT/'training.json').write_text(json.dumps(dict(history=history,elapsed_s=time.monotonic()-start,seed=202610204011,train_frames=1280,validation_frames=320,test_frames_used=0),indent=2))
(OUT/'complete.json').write_text(json.dumps({'complete':True,'epochs':35,'best_score':best}));print('SMALL_FLOAT_TRAINING_COMPLETE',flush=True)
