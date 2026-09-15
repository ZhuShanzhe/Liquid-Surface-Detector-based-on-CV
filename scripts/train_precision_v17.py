#!/usr/bin/env python3
import argparse,json,time,random
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
import train_observable_v16 as v16
from liquid_depth.models.precision_v17 import PrecisionPipeline,precision_loss,fit_surface
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916')

def save(p,value):v16.save(p,value)
class SurfaceWrapper(nn.Module):
    def __init__(self,pipe):super().__init__();self.pipe=pipe
    def forward(self,rgb,raw):
        p=self.pipe(rgb,raw);p['depth_m']=fit_surface(p['depth_m'],p['mask_probability']);return p

def make(radius=0,checkpoint=None,post=False):
    initial=v16.make_pipeline('gated',checkpoint or v16.OUT/'gated/best.pth')
    pipe=PrecisionPipeline(initial.locator,initial.depth_model,radius).cuda()
    return SurfaceWrapper(pipe) if post else pipe

def evaluate(pipe,mode='clean',part=0,manifest=None,split='val'):
    return v16.evaluate(pipe,manifest or v16.DEV,split,part,mode)

def diagnostic():
    if (OUT/'routing-diagnostic.json').exists():raise FileExistsError(OUT/'routing-diagnostic.json')
    records={}
    for radius in (4,0,-2):
        pipe=make(radius).eval()
        for post in (False,True):
            model=SurfaceWrapper(pipe) if post else pipe
            r=evaluate(model);name=f'r{radius}'+('_plane' if post else '')
            records[name]=r;print('DIAGNOSTIC',name,{s:(v['all']['mae_mm'],v['all']['success']) for s,v in r['summary'].items()},flush=True)
        del pipe;torch.cuda.empty_cache()
    save(OUT/'routing-diagnostic.json',records)


def train(args):
    radius=args.radius;out=OUT/f'r{radius}';out.mkdir(exist_ok=False)
    pipe=make(radius);model=pipe.depth_model
    base=evaluate(pipe);reference=evaluate(make(4))
    save(out/'baseline.json',base);save(out/'v16-reference.json',reference)
    def score(report):
        groups=report['summary']
        if any(groups[s]['all']['success']<reference['summary'][s]['all']['success']-.02 or groups[s]['all']['coverage']<reference['summary'][s]['all']['coverage']-.01 for s in ('transparent','translucent')):return float('inf')
        return np.mean([groups[s]['all']['mae_mm'] for s in ('transparent','translucent')])
    best=score(base);epoch_best=0;history=[]
    torch.save({'depth':model.state_dict(),'epoch':0,'radius':radius},out/'best.pth')
    opt=torch.optim.AdamW([{'params':model.network.parameters(),'lr':3e-5},{'params':model.correction.parameters(),'lr':1e-4}],weight_decay=1e-4)
    ds=v16.Observations(v16.DEV,'train',mode='clean')
    # Parent augmentation is deliberately disabled; this wrapper alone augments.
    for epoch in range(1,7):
        model.train();v16.frozen_bn(model);losses=[];start=time.time()
        for batch in v16.loader(ds,True):
            b=v16.gpu(batch)
            rgb=b['rgb'];raw=b['raw'];r=np.random.random()
            # 70% clean, 10% global, 10% local, 10% dropout; shared policy across arms.
            if r<.1:raw=torch.where(raw>0,raw*(1+np.random.uniform(-.04,.04)),raw)
            elif r<.2:raw=torch.where(torch.rand_like(raw)<np.random.uniform(.1,.6),0,raw)
            elif r<.3:
                raw=raw.clone();h,w=raw.shape[-2:];y=np.random.randint(h//4,3*h//4);x=np.random.randint(w//4,3*w//4)
                raw[:,:,y-h//4:y+h//4,x-w//4:x+w//4]*=1+np.random.uniform(-.1,.1)
            if np.random.random()<.5:
                rgb=rgb.flip(-1);raw=raw.flip(-1);b['truth']=b['truth'].flip(-1);b['valid']=b['valid'].flip(-1)
            p=pipe(rgb,raw);valid=b['valid']*p['has_depth_anchor'][:,None,None,None]
            if not valid.any():continue
            loss=precision_loss(p['unclipped_depth_m'],b['truth'],valid)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss))
        report=evaluate(pipe);value=score(report)
        if value<best:
            best=value;epoch_best=epoch;torch.save({'depth':model.state_dict(),'epoch':epoch,'radius':radius},out/'best.pth')
        record=dict(epoch=epoch,loss=float(np.mean(losses)),score=None if not np.isfinite(value) else float(value),best_epoch=epoch_best,seconds=time.time()-start,summary=report['summary'])
        history.append(record);save(out/'history.json',history)
        print('EPOCH',radius,epoch,record['score'],epoch_best,round(record['seconds'],1),flush=True)
    save(out/'frozen.json',dict(radius=radius,epoch=epoch_best,checkpoint_sha256=v16.digest(out/'best.pth'),score=None if not np.isfinite(best) else float(best),locator_sha256=v16.digest(v16.OUT/'locator.pth'),qualified_on_selection_fold=bool(np.isfinite(best))))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['diagnostic','train'],required=True);ap.add_argument('--radius',type=int,default=0);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260917031);np.random.seed(20260917031%(2**32));random.seed(20260917031)
    if args.stage=='diagnostic':diagnostic()
    else:train(args)
if __name__=='__main__':main()
