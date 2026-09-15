#!/usr/bin/env python3
"""Train depth then quality; disjoint stopping, calibration and operating folds."""
import argparse,json,math,time,random,hashlib
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch import nn
from torch.nn import functional as F
import train_candidate_ranker_v12 as v12
from train_liquid_candidates_v11 import weights_for
from screen_depth_methods_v14 import ROOT,ART,DATA,BASE,CAND,RANK,stats,summary,fit_plane,digest
from liquid_depth.refinement import TransCGDFNetRefiner
from liquid_depth.models.candidate_ranker import CandidateRanker,MonotoneCalibrator
from liquid_depth.models.liquid_adaptation_v15 import LiquidDFNet,QualityNet,prepare_depth,depth_loss,encode_depth
OUT=ART/'training/liquid-adaptation-v15-20260915'
MANIFEST=DATA/'synthetic/liquid-sim-v3-calibration-top-12000/manifest_multilayer_v4_refined.csv'
THRESHOLDS=(.5,.55,.6,.65,.7,.75,.8,.85,.9,.95)

def save(p,d):
 Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))

def corrupt(raw,mode,rng):
 raw=raw.copy();valid=raw>0
 if mode=='random':mode=rng.choice(['clean','clean','scale','local','drop'])
 if mode.startswith('bias'):
  factor=1+float(mode[4:])/100;raw[valid]=np.clip(raw[valid]*factor,.1,10)
 elif mode=='scale':raw[valid]=np.clip(raw[valid]*(1+rng.choice([-1,1])*rng.uniform(.01,.06)),.1,10)
 elif mode in ('local','local10'):
  h,w=raw.shape;cy,cx=(h//2,w//2) if mode=='local10' else (rng.randint(h//4,3*h//4),rng.randint(w//4,3*w//4))
  y0,y1=max(0,cy-h//4),min(h,cy+h//4);x0,x1=max(0,cx-w//4),min(w,cx+w//4)
  patch=raw[y0:y1,x0:x1];ok=patch>0
  factor=1.1 if mode=='local10' else 1+rng.choice([-1,1])*rng.uniform(.03,.12)
  patch[ok]=np.clip(patch[ok]*factor,.1,10)
 elif mode=='drop':raw[rng.random_sample(raw.shape)<rng.uniform(.2,.6)]=0
 elif mode!='clean':raise ValueError(mode)
 return raw

class Prepared(torch.utils.data.Dataset):
 def __init__(self,d,mode='clean',seed=20260915201):self.d=d;self.rows=d.rows;self.mode=mode;self.seed=seed
 def __len__(self):return len(self.d)
 def __getitem__(self,i):
  cv2.setNumThreads(0)
  x,y,rays=self.d[i];raw=.1*np.exp(x[3].numpy()*math.log(100));raw[x[4].numpy()<.5]=0
  rng=np.random if self.mode=='random' else np.random.RandomState((self.seed+i) % (2**32))
  raw=corrupt(raw,self.mode,rng)
  rgb=(x[:3].numpy()*np.array([.229,.224,.225],np.float32)[:,None,None]+np.array([.485,.456,.406],np.float32)[:,None,None]).clip(0,1)
  rgb240,dn,low,span,anchor=prepare_depth(rgb.transpose(1,2,0),raw)
  x=x.clone();x[3]=torch.from_numpy(np.where(raw>0,np.log(np.maximum(raw,.1)/.1)/math.log(100),0));x[4]=torch.from_numpy((raw>0).astype(np.float32))
  return dict(x=x,truth=y['depth_m'],valid=y['valid'],rays=rays,rgb=torch.from_numpy(rgb),raw=torch.from_numpy(raw[None]),rgb240=torch.from_numpy(rgb240),dn=torch.from_numpy(dn),low=low,span=span,anchor=anchor,index=i)

def batches(d,mode,args,training=False,samples=None):
 ds=Prepared(d,mode,args.seed)
 sampler=None
 if training:
  w,_=weights_for(d);w=[v*({'transparent':2.,'translucent':1.,'multilayer':.25,'compound':.25}[r['scenario']]) for v,r in zip(w,d.rows)]
  sampler=WeightedRandomSampler(w,samples or args.samples,replacement=True)
 return DataLoader(ds,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=True)

def device(b):return {k:v.cuda(non_blocking=True) for k,v in b.items()}

def load_frozen():
 opts=argparse.Namespace(base=BASE,candidates=CAND,feature_context=True)
 cand=v12.load_candidates(opts)
 rank=CandidateRanker(False,24).cuda().eval().requires_grad_(False)
 rank.load_state_dict(torch.load(RANK,map_location='cpu',weights_only=False)['model'])
 return cand,rank,opts

def make_depth():
 ref=TransCGDFNetRefiner(DATA/'research/sources/TransCG',DATA/'research/transcg/checkpoints/checkpoint.tar',[320,240],.1,10,3,True)
 return LiquidDFNet(ref.model).cuda()

def prediction(model,arm,b,cand,rank,opts):
 with torch.no_grad():p=v12.candidate_prediction(cand,b['x'],opts)
 prior=p['depth_m'];mask=p['mask_logits'].sigmoid()
 if arm=='plane':
  with torch.no_grad():
   selected=rank(b['x'],p,b['rays'])['depth_m']
   depth=torch.stack([fit_plane(selected[i],mask[i]>=.5,b['rays'][i]) for i in range(len(selected))])
 else:depth=model(b['rgb240'],b['dn'],b['low'],b['span'],b['rgb'],b['raw'],prior,mask,b['anchor'])
 return depth,prior,mask

@torch.inference_mode()
def evaluate(model,quality,arm,d,mode,args,cand,rank,opts,cal=None):
 if model is not None:model.eval()
 if quality is not None:quality.eval()
 frames=[]
 for batch in batches(d,mode,args):
  b=device(batch);depth,prior,mask=prediction(model,arm,b,cand,rank,opts)
  depth=torch.where(torch.isfinite(depth)&(depth>=.1)&(depth<=10),depth,0)
  conf=quality(b['rgb'],b['raw'],depth,prior,mask).sigmoid() if quality is not None else torch.ones_like(depth)
  if cal is not None:conf=cal(conf)
  for j,idx in enumerate(batch['index'].tolist()):
   valid=b['valid'][j]>0;row=d.rows[idx]
   rec=dict(sequence_id=row['sequence_id'],scenario=row['scenario'],all=stats(depth[j],b['truth'][j],valid),selected={})
   eligible=(mask[j]>=.5)&(depth[j]>0)
   score=conf[j][eligible]
   correct=(((depth[j]-b['truth'][j]).abs()<=(b['truth'][j]*.02).clamp_min(.005))&valid)[eligible].float()
   bins=(score*10).long().clamp(0,9)
   rec['cal_count']=torch.bincount(bins,minlength=10).cpu().tolist()
   rec['cal_good']=torch.bincount(bins,weights=correct,minlength=10).cpu().tolist()
   rec['cal_score']=torch.bincount(bins,weights=score,minlength=10).cpu().tolist()
   rec['brier_sum']=float(((score-correct)**2).sum())
   for t in THRESHOLDS:
    emit=(mask[j]>=.5)&(conf[j]>=t)&(depth[j]>0)
    z=torch.where(emit,depth[j],0)
    st=stats(z,b['truth'][j],valid);st['non_target']=int((emit&~valid).sum());st['emitted']=int(emit.sum())
    rec['selected'][str(t)]=st
   frames.append(rec)
 groups={'all':frames,**{s:[f for f in frames if f['scenario']==s] for s in ('transparent','translucent','multilayer','compound')}}
 report={}
 for scene,fs in groups.items():
  report[scene]={'all':summary([f['all'] for f in fs]),'selected':{str(t):summary([f['selected'][str(t)] for f in fs]) for t in THRESHOLDS}}
  count=np.sum([f['cal_count'] for f in fs],0);positive=np.sum([f['cal_good'] for f in fs],0);prob=np.sum([f['cal_score'] for f in fs],0)
  report[scene]['ece']=float(np.abs(positive-prob).sum()/max(count.sum(),1))
  report[scene]['brier']=sum(f['brier_sum'] for f in fs)/max(float(count.sum()),1)
  for t in THRESHOLDS:
   n=sum(f['selected'][str(t)]['non_target'] for f in fs);e=sum(f['selected'][str(t)]['emitted'] for f in fs)
   report[scene]['selected'][str(t)]['non_target_fraction']=n/max(e,1)
 return {'arm':arm,'mode':mode,'summary':report,'frames':frames}

@torch.inference_mode()
def calibrate(model,quality,arm,d,args,cand,rank,opts):
 count=np.zeros(32);good=np.zeros(32)
 for mode in ('clean','bias4','bias-4','local10'):
  for batch in batches(d,mode,args):
   b=device(batch);depth,prior,mask=prediction(model,arm,b,cand,rank,opts)
   safe=torch.where(torch.isfinite(depth)&(depth>=.1)&(depth<=10),depth,0)
   score=quality(b['rgb'],b['raw'],safe,prior,mask).sigmoid()
   target=((safe-b['truth']).abs()<=(b['truth']*.02).clamp_min(.005))&(b['valid']>0)
   eligible=(mask>=.5)&(safe>0)
   bins=(score[eligible]*32).long().clamp(0,31)
   count+=torch.bincount(bins,minlength=32).cpu().numpy();good+=torch.bincount(bins,weights=target[eligible].float(),minlength=32).cpu().numpy()
 cal=MonotoneCalibrator().fit_histogram(count,good)
 return cal,{'values':cal.values.tolist(),'count':count.tolist(),'good':good.tolist(),'modes':['clean','bias4','bias-4','local10']}

def choose_threshold(reports,scene):
 choices=[]
 for t in THRESHOLDS:
  vals=[r['summary'][scene]['selected'][str(t)] for r in reports]
  if all(v['precision'] is not None and v['precision']>=.8 and v['coverage']>=.1 and v['non_target_fraction']<=.1 for v in vals):choices.append((min(v['success'] for v in vals),t))
 return max(choices)[1] if choices else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['pretrained','clean','robust','plane'],required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--samples',type=int,default=3072);ap.add_argument('--quality-epochs',type=int,default=3);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--seed',type=int,default=20260915201);ap.add_argument('--stage',choices=['train','test'],default='train');ap.add_argument('--manifest');args=ap.parse_args()
 torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed % (2**32));random.seed(args.seed)
 out=OUT/args.arm;out.mkdir(parents=True,exist_ok=True)
 args.manifest=args.manifest or str(MANIFEST)
 cand,rank,opts=load_frozen();model=make_depth() if args.arm!='plane' else None
 quality=QualityNet().cuda()
 if args.stage=='test':
  frozen=json.loads((out/'frozen.json').read_text());state=torch.load(out/'best.pth',map_location='cpu',weights_only=False)
  assert digest(out/'best.pth')==frozen['checkpoint_sha256']
  if model is not None:model.load_state_dict(state['depth'])
  quality.load_state_dict(state['quality']);quality.eval();cal=MonotoneCalibrator();cal.values=np.asarray(frozen['calibration']['values'])
  d=v12.get_data(args,'test')
  for mode in ('clean','bias8','bias-8','local10'):
   p=out/f'fresh-{mode}.json'
   if p.exists():raise FileExistsError(p)
   save(p,evaluate(model,quality,args.arm,d,mode,args,cand,rank,opts,cal));print('TEST_DONE',args.arm,mode,flush=True)
  return
 if (out/'frozen.json').exists():raise FileExistsError(out/'frozen.json')
 train=v12.get_data(args,'train');stop=v12.get_data(args,'val',0)
 baseline={m:evaluate(model,None,args.arm,stop,m,args,cand,rank,opts) for m in ('clean','bias5')}
 for m,r in baseline.items():save(out/f'epoch0-{m}.json',r)
 def score(rs):
  a=rs['clean']['summary']['transparent']['all']['success'];b=rs['bias5']['summary']['transparent']['all']['success']
  if a<baseline['clean']['summary']['transparent']['all']['success']-.03:return -10+a
  return .6*min(a,b)+.2*(a+b)
 best=score(baseline);best_epoch=0
 if model is not None:torch.save(model.state_dict(),out/'depth-best.pth')
 history=[]
 if args.arm in ('clean','robust'):
  opt=torch.optim.AdamW([{'params':model.network.parameters(),'lr':1e-4},{'params':model.correction.parameters(),'lr':3e-4}],weight_decay=1e-4)
  for epoch in range(1,args.epochs+1):
   model.train()
   # Preserve stable pretrained normalization statistics across small material batches.
   for mod in model.modules():
    if isinstance(mod,nn.BatchNorm2d):mod.eval()
   losses=[];start=time.time()
   for step,batch in enumerate(batches(train,'random' if args.arm=='robust' else 'clean',args,True)):
    b=device(batch);depth,prior,mask=prediction(model,args.arm,b,cand,rank,opts)
    valid=b['valid']*b['anchor'][:,None,None,None]
    if not valid.any():continue
    loss=depth_loss(depth,b['truth'],valid)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss))
    if (step+1)%64==0:print('TRAIN',args.arm,epoch,step+1,float(np.mean(losses)),flush=True)
   reports={m:evaluate(model,None,args.arm,stop,m,args,cand,rank,opts) for m in ('clean','bias5')}
   current=score(reports)
   item=dict(epoch=epoch,loss=float(np.mean(losses)),score=current,seconds=time.time()-start,summary={m:r['summary'] for m,r in reports.items()});history.append(item);save(out/'history.json',history)
   if current>best:
    best=current;best_epoch=epoch;torch.save(model.state_dict(),out/'depth-best.pth')
   print('EPOCH_DONE',args.arm,epoch,current,'best',best_epoch,flush=True)
  model.load_state_dict(torch.load(out/'depth-best.pth',map_location='cpu',weights_only=True))
 if model is not None:model.eval().requires_grad_(False)
 # Train correctness only after depth is frozen. Labels outside GT mask are negatives.
 opt=torch.optim.AdamW(quality.parameters(),lr=3e-4,weight_decay=1e-4)
 for epoch in range(1,args.quality_epochs+1):
  quality.train();losses=[]
  for batch in batches(train,'random',args,True,samples=2048):
   b=device(batch)
   with torch.no_grad():
    depth,prior,mask=prediction(model,args.arm,b,cand,rank,opts);depth=torch.where(torch.isfinite(depth)&(depth>=.1)&(depth<=10),depth,0)
    correct=((depth-b['truth']).abs()<=(b['truth']*.02).clamp_min(.005))&(b['valid']>0)
    eligible=(mask>=.5)&(depth>0)
   if not eligible.any():continue
   logits=quality(b['rgb'],b['raw'],depth,prior,mask)
   loss=F.binary_cross_entropy_with_logits(logits[eligible],correct[eligible].float())
   opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(quality.parameters(),1.);opt.step();losses.append(float(loss))
  print('QUALITY_DONE',args.arm,epoch,float(np.mean(losses)),flush=True)
 quality.eval();caldata=v12.get_data(args,'val',1);opdata=v12.get_data(args,'val',2)
 cal,calinfo=calibrate(model,quality,args.arm,caldata,args,cand,rank,opts)
 reports=[]
 for mode in ('clean','bias4','bias-4','local10'):
  r=evaluate(model,quality,args.arm,opdata,mode,args,cand,rank,opts,cal);reports.append(r);save(out/f'operating-{mode}.json',r)
 threshold=choose_threshold(reports,'translucent' if args.arm=='plane' else 'transparent')
 torch.save({'depth':model.state_dict() if model is not None else None,'quality':quality.state_dict(),'arm':args.arm,'epoch':best_epoch,'seed':args.seed},out/'best.pth')
 frozen={'arm':args.arm,'best_epoch':best_epoch,'score':best,'threshold':threshold,'qualified_on_operating_fold':threshold is not None,'diagnostic_threshold':.5,'calibration':calinfo,'checkpoint_sha256':digest(out/'best.pth'),'source_sha256':digest(__file__),'module_sha256':digest(Path(__file__).parents[1]/'src/liquid_depth/models/liquid_adaptation_v15.py'),'seed':args.seed,'scope':'axial camera depth, confidence trained then calibrated on disjoint sequences'}
 save(out/'frozen.json',frozen);print('FROZEN',args.arm,best_epoch,threshold,flush=True)
if __name__=='__main__':main()
