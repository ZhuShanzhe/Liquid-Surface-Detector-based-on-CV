#!/usr/bin/env python3
"""V16 controlled input-route experiment; labels are only read by losses/metrics."""
import argparse,csv,hashlib,inspect,json,random,time
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,RandomSampler
from evaluate_layer_voting import load_model
from train_liquid_adaptation_v15 import make_depth,corrupt
from screen_depth_methods_v14 import BASE,ART,stats,summary,digest
from liquid_depth.models.observable_v16 import ObservableLiquidPipeline,encode_rgbd
from liquid_depth.models.liquid_adaptation_v15 import depth_loss
OUT=ART/'training/observable-v16-20260916'
DEV=Path('/root/autodl-tmp/liquid-depth-data/synthetic/observable-v16-development/manifest.csv')
V15=ART/'training/liquid-adaptation-v15-20260915'
MODES=('clean','bias8','bias-8','local10','drop80')

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def fold(row):
    return int(hashlib.sha256(row['sequence_id'].encode()).hexdigest()[:8],16)%3

class Observations(torch.utils.data.Dataset):
    def __init__(self,manifest,split,part=None,mode='clean'):
        self.rows=[r for r in csv.DictReader(Path(manifest).open()) if r['split']==split and (part is None or fold(r)==part)]
        self.mode=mode
        if not self.rows:raise ValueError('Empty partition')
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        cv2.setNumThreads(0);r=self.rows[i]
        rgb=cv2.cvtColor(cv2.imread(r['rgb_path']),cv2.COLOR_BGR2RGB).astype(np.float32)/255
        raw=np.load(r['raw_depth_path']).astype(np.float32)
        truth=np.load(r['target_depth_path']).astype(np.float32)
        mask=np.load(r['mask_path'])>0
        rng=np.random if self.mode=='random' else np.random.RandomState((20260916031+i)%(2**32))
        if self.mode=='drop80':
            raw=raw.copy();raw[rng.random_sample(raw.shape)<.8]=0
        else:raw=corrupt(raw,self.mode,rng)
        if self.mode=='random':
            rgb=np.clip(rgb*rng.uniform(.85,1.15)+rng.uniform(-.04,.04),0,1)
            if rng.random_sample()<.5:rgb,raw,truth,mask=[np.flip(a,1).copy() for a in (rgb,raw,truth,mask)]
        valid=mask&np.isfinite(truth)&(truth>=.1)&(truth<=10)
        return {'rgb':torch.from_numpy(rgb.transpose(2,0,1).copy()),'raw':torch.from_numpy(raw[None].copy()),'truth':torch.from_numpy(truth[None].copy()),'valid':torch.from_numpy(valid[None].astype(np.float32)),'mask':torch.from_numpy(mask[None].astype(np.float32)),'index':i}

def loader(d,training=False,samples=1536):
    sampler=RandomSampler(d,replacement=True,num_samples=samples) if training else None
    return DataLoader(d,batch_size=8,num_workers=4,pin_memory=True,sampler=sampler)
def gpu(b):return {k:v.cuda(non_blocking=True) for k,v in b.items()}
def frozen_bn(m):
    for mod in m.modules():
        if isinstance(mod,nn.BatchNorm2d):mod.eval()
def construct_locator(path=None):
    model,state=load_model(BASE)
    if path:model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    return model,state

@torch.inference_mode()
def locator_score(model,manifest):
    model.eval();results={}
    for mode in ('clean','bias5'):
        ds=Observations(manifest,'val',0,mode);inter=union=0
        for batch in loader(ds):
            b=gpu(batch);mask=model(encode_rgbd(b['rgb'],b['raw']))['mask_logits']>=0;valid=b['mask']>0
            inter+=int((mask&valid).sum());union+=int((mask|valid).sum())
        results[mode]=inter/max(union,1)
    return results

def train_locator(args):
    model,state=construct_locator()
    best=-1.;history=[]
    initial=locator_score(model,args.manifest);save(OUT/'locator-initial.json',initial)
    # All score thresholds are selected only on development fold zero.
    best=min(initial.values());best_epoch=0
    torch.save({'model':model.state_dict(),'base_sha256':digest(BASE),'epoch':0},OUT/'locator.pth')
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    ds=Observations(args.manifest,'train',mode='random')
    for epoch in range(1,7):
        model.train();frozen_bn(model);losses=[];start=time.time()
        for batch in loader(ds,True):
            b=gpu(batch);p=model(encode_rgbd(b['rgb'],b['raw']))
            prob=p['mask_logits'].sigmoid()
            bce=F.binary_cross_entropy_with_logits(p['mask_logits'],b['mask'])
            dice=1-(2*(prob*b['mask']).sum((1,2,3))+1)/(prob.sum((1,2,3))+b['mask'].sum((1,2,3))+1)
            loss=bce+dice.mean()+.1*depth_loss(p['depth_m'],b['truth'],b['valid'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite locator loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss))
        scores=locator_score(model,args.manifest);score=min(scores.values())
        if score>best:
            best=score;best_epoch=epoch
            torch.save({'model':model.state_dict(),'base_sha256':digest(BASE),'epoch':epoch},OUT/'locator.pth')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),scores=scores,best_epoch=best_epoch,seconds=time.time()-start))
        save(OUT/'locator-history.json',history);print('LOCATOR',history[-1],flush=True)
    save(OUT/'locator-frozen.json',dict(checkpoint_sha256=digest(OUT/'locator.pth'),best_epoch=best_epoch,best_iou=best))

def make_pipeline(route,checkpoint=None,adapted=True):
    loc,_=construct_locator(OUT/'locator.pth' if adapted else None);loc.eval().requires_grad_(False)
    depth=make_depth()
    state=torch.load(checkpoint or V15/'robust/best.pth',map_location='cpu',weights_only=False)
    depth.load_state_dict(state['depth'])
    return ObservableLiquidPipeline(loc,depth,route).cuda()

@torch.inference_mode()
def evaluate(pipe,manifest,split,part,mode):
    pipe.eval();ds=Observations(manifest,split,part,mode);frames=[]
    for batch in loader(ds):
        b=gpu(batch);p=pipe(b['rgb'],b['raw']);depth=p['depth_m'];mask=p['mask_probability']>=.5
        for j,i in enumerate(batch['index'].tolist()):
            valid=b['valid'][j]>0;target_mask=b['mask'][j]>0
            predicted=mask[j]&(depth[j]>0)
            selected=stats(torch.where(predicted,depth[j],0),b['truth'][j],valid)
            selected['emitted']=int(predicted.sum());selected['non_target']=int((predicted&~valid).sum())
            frames.append(dict(sequence_id=ds.rows[i]['sequence_id'],scenario=ds.rows[i]['scenario'],all=stats(depth[j],b['truth'][j],valid),selected=selected,intersection=int((mask[j]&target_mask).sum()),union=int((mask[j]|target_mask).sum()),target=int(target_mask.sum()),predicted=int(mask[j].sum()),anchor_count=int(p['anchor_count'][j]),has_anchor=bool(p['has_depth_anchor'][j])))
    groups={}
    for scene in ('all','transparent','translucent'):
        rows=frames if scene=='all' else [r for r in frames if r['scenario']==scene]
        if not rows:continue
        selected=summary([r['selected'] for r in rows]);emitted=sum(r['selected']['emitted'] for r in rows)
        selected['non_target_fraction']=sum(r['selected']['non_target'] for r in rows)/max(emitted,1)
        inter=sum(r['intersection'] for r in rows)
        groups[scene]=dict(all=summary([r['all'] for r in rows]),selected=selected,mask_iou=inter/max(sum(r['union'] for r in rows),1),mask_recall=inter/max(sum(r['target'] for r in rows),1),mask_precision=inter/max(sum(r['predicted'] for r in rows),1),anchorless_frames=sum(not r['has_anchor'] for r in rows))
    return dict(mode=mode,summary=groups,frames=frames)

def train_depth(args):
    pipe=make_pipeline(args.route);depth=pipe.depth_model
    out=OUT/args.route;out.mkdir(exist_ok=False)
    def validation():
        return {m:evaluate(pipe,args.manifest,'val',0,m) for m in ('clean','bias5')}
    def score(rs):
        return float(np.mean([min(rs[m]['summary'][s]['all']['success'] for m in rs) for s in ('transparent','translucent')]))
    baseline=validation();save(out/'epoch0.json',baseline);best=score(baseline);best_epoch=0
    torch.save({'depth':depth.state_dict(),'epoch':0},out/'best.pth')
    opt=torch.optim.AdamW([{'params':depth.network.parameters(),'lr':1e-4},{'params':depth.correction.parameters(),'lr':3e-4}],weight_decay=1e-4)
    ds=Observations(args.manifest,'train',mode='random');history=[]
    for epoch in range(1,5):
        depth.train();frozen_bn(depth);losses=[];start=time.time()
        for batch in loader(ds,True):
            b=gpu(batch);p=pipe(b['rgb'],b['raw']);valid=b['valid']*p['has_depth_anchor'][:,None,None,None]
            if not valid.any():continue
            loss=depth_loss(p['depth_m'],b['truth'],valid)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite depth loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(depth.parameters(),1);opt.step();losses.append(float(loss))
        reports=validation();current=score(reports)
        if current>best:
            best=current;best_epoch=epoch;torch.save({'depth':depth.state_dict(),'epoch':epoch},out/'best.pth')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),score=current,best_epoch=best_epoch,seconds=time.time()-start,summary={m:r['summary'] for m,r in reports.items()}))
        save(out/'history.json',history);print('DEPTH',args.route,epoch,current,best_epoch,round(time.time()-start,1),flush=True)
    save(out/'frozen.json',dict(checkpoint_sha256=digest(out/'best.pth'),locator_sha256=digest(OUT/'locator.pth'),best_epoch=best_epoch,score=best,input_keys=['rgb','raw_depth_m'],route=args.route))
    
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['locator','depth','test'],required=True);ap.add_argument('--route',choices=['full','gated'],default='gated')
    ap.add_argument('--manifest',default=str(DEV));args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260916031);np.random.seed(20260916031%(2**32));random.seed(20260916031)
    if args.stage=='locator':train_locator(args)
    elif args.stage=='depth':train_depth(args)
    else:
        runs=[('v15_robust_full','full',V15/'robust/best.pth',False),('v15_strong_full','full',V15/'strong/robust/best.pth',False),('adapted_locator_full','full',V15/'robust/best.pth',True),('predicted_gate_only','gated',V15/'robust/best.pth',True),('full_tuned','full',OUT/'full/best.pth',True),('gated_tuned','gated',OUT/'gated/best.pth',True)]
        for name,route,checkpoint,adapted in runs:
            out=OUT/'test'/name;out.mkdir(parents=True,exist_ok=False)
            pipe=make_pipeline(route,checkpoint,adapted)
            for mode in MODES:
                report=evaluate(pipe,args.manifest,'test',None,mode);save(out/f'{mode}.json',report)
                print('TEST',name,mode,{s:r['all']['success'] for s,r in report['summary'].items()},flush=True)
            del pipe;torch.cuda.empty_cache()
if __name__=='__main__':main()
