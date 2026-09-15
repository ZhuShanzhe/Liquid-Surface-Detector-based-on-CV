#!/usr/bin/env python3
"""Frozen method screening. GT is used only in metrics, never model inputs."""
import argparse,json,math,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
import train_candidate_ranker_v12 as v12
from liquid_depth.models.candidate_ranker import CandidateRanker
from liquid_depth.refinement import TransCGDFNetRefiner
from liquid_depth.baseline_refiners import DREDSSwinDRNetRefiner
ROOT=Path('/root/autodl-tmp')
ART=ROOT/'liquid-depth-artifacts'
DATA=ROOT/'liquid-depth-data'
BASE=ART/'training/universal-liquid-v9-1-transparent-multilayer/best.pth'
CAND=ART/'training/liquid-candidate-v11-factorial-20260915/control/best.pth'
RANK=ART/'training/candidate-ranker-v12-context-20260915/rank_only/best.pth'
METHODS=('raw','direct','v12','dfnet_native','dfnet_wide','swin_native','swin_wide','lingbot','plane_direct','plane_v12')
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def stats(pred,truth,valid):
 finite=torch.isfinite(pred)&(pred>0)
 accepted=valid&finite
 err=(pred-truth).abs()
 good=accepted&(err<=torch.clamp(truth*.02,min=.005))
 n=int(accepted.sum()); possible=int(valid.sum())
 return dict(n=n,possible=possible,good=int(good.sum()),abs_sum=float(err[accepted].sum()),rel_sum=float((err/truth.clamp_min(.001))[accepted].sum()))
def summary(rows):
 s={k:sum(r[k] for r in rows) for k in ('n','possible','good','abs_sum','rel_sum')}
 n=s['n'];p=s['possible']
 return dict(frames=len(rows),**s,mae_mm=1000*s['abs_sum']/n if n else None,absrel=s['rel_sum']/n if n else None,precision=s['good']/n if n else None,coverage=n/p if p else None,success=s['good']/p if p else None)
def fit_plane(depth,mask,rays):
 # Reciprocal axial depth is linear in camera-ray coordinates on a plane.
 # Fixed IRLS fit to predicted-mask points only; no truth, pose, or level input.
 d=depth.flatten();r=rays.permute(1,2,0).reshape(-1,3).clone();r[:,2]=1
 ok=mask.flatten()&torch.isfinite(d)&(d>=.1)&(d<=10)
 if int(ok.sum())<64:return torch.full_like(depth,float('nan'))
 ids=torch.where(ok)[0][::4];A=r[ids];b=1/d[ids];w=torch.ones_like(b)
 for _ in range(4):
  coef=torch.linalg.lstsq(A*w[:,None].sqrt(),b*w.sqrt()).solution
  error=(A@coef-b).abs();scale=error.median().clamp_min(1e-5)*1.4826
  w=(1.345*scale/error.clamp_min(1e-6)).clamp_max(1)
 inv=r@coef;pred=torch.where(inv>0,1/inv,float('nan'))
 return pred.reshape_as(depth)
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest',required=True);a.add_argument('--split',default='test');a.add_argument('--fold',type=int);a.add_argument('--method',choices=METHODS,required=True);a.add_argument('--out',required=True);a.add_argument('--limit',type=int,default=0);a.add_argument('--stress',action='store_true');args=a.parse_args()
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 opts=argparse.Namespace(manifest=args.manifest,base=BASE,candidates=CAND,feature_context=True)
 d=v12.get_data(opts,args.split,args.fold)
 if args.limit:d.d.rows=d.d.rows[:args.limit];d.rows=d.d.rows
 torch.set_num_threads(4);torch.manual_seed(20260915107)
 cand=v12.load_candidates(opts)
 rank=CandidateRanker(False,24).cuda().eval();rank.load_state_dict(torch.load(RANK,map_location='cpu',weights_only=False)['model'])
 model=None; sources={'base':str(BASE),'candidates':str(CAND),'ranker':str(RANK)}
 if args.method.startswith('dfnet'):
  wide=args.method.endswith('wide');source=DATA/'research/sources/TransCG';weight=DATA/'research/transcg/checkpoints/checkpoint.tar'
  model=TransCGDFNetRefiner(source,weight,[320,240],.1 if wide else .3,10. if wide else 1.,3.,True)
  sources.update(external=str(weight))
 elif args.method.startswith('swin'):
  source=DATA/'research/sources/DREDS/SwinDRNet';weight=DATA/'research/dreds_std/archives/model.pth'
  model=DREDSSwinDRNetRefiner(source,weight,[224,224],10. if args.method.endswith('wide') else 3.)
  sources.update(external=str(weight))
 elif args.method=='lingbot':
  sys.path.insert(0,str(ROOT/'tools/lingbot-depth'))
  from mdm.model.v2 import MDMModel
  weight=ART/'external/lingbot-depth/model.pt';model=MDMModel.from_pretrained(str(weight)).cuda().eval()
  from types import MethodType
  from mdm.model.dinov2_rgbd.layers.block import Block
  # Eval-only equivalent to block-diagonal attention, per independent sample.
  def forward_independent(self,x):
   if self.training:raise RuntimeError('Inference fallback only')
   return [Block.forward(self,t) for t in x] if isinstance(x,list) else Block.forward(self,x)
  model.enable_pytorch_native_sdpa()
  for block in model.encoder.backbone.blocks:block.forward=MethodType(forward_independent,block)
  sources.update(external=str(weight))
 records=[];latency=[]
 with torch.inference_mode():
  for i in range(len(d)):
   x,y,rays=d[i];x=x[None].cuda();rays=rays[None].cuda();y={k:v.cuda() for k,v in y.items()}
   if args.stress:x=v12.biased(x)
   raw=.1*torch.exp(x[:,3]*math.log(100));raw=torch.where(x[:,4]>.5,raw,0.)
   rgb=(x[:,:3]*x.new_tensor([.229,.224,.225])[None,:,None,None]+x.new_tensor([.485,.456,.406])[None,:,None,None]).clamp(0,1)
   p=v12.candidate_prediction(cand,x,opts)
   mask=p['mask_logits'][0].sigmoid()>=.5
   bgr=(rgb[0].permute(1,2,0).cpu().numpy()[...,::-1]*255).round().astype(np.uint8)
   raw_np=raw[0].cpu().numpy()
   torch.cuda.synchronize();start=time.perf_counter()
   if args.method=='raw':pred=raw[0,None]
   elif args.method=='direct':pred=cand.backbone(x)['depth_m'][0]
   elif args.method=='v12':pred=rank(x,v12.candidate_prediction(cand,x,opts),rays)['depth_m'][0]
   elif args.method.startswith('plane_'):
    base=cand.backbone(x)['depth_m'][0] if args.method=='plane_direct' else rank(x,v12.candidate_prediction(cand,x,opts),rays)['depth_m'][0]
    pred=fit_plane(base,mask,rays[0])
   elif args.method=='lingbot':
    pred=model.infer(rgb,depth_in=raw,resolution_level=9,apply_mask=False)['depth'][0,None]
   elif not np.any(raw_np>0):pred=torch.zeros_like(raw[0,None])
   else:pred=torch.from_numpy(model.predict(bgr,raw_np).depth_m).cuda()[None]
   torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
   if i>=3:latency.append(elapsed)
   valid=y['valid']>0;truth=y['depth_m']
   row=d.rows[i];med=float(truth[valid].median()) if valid.any() else 0
   rec=dict(sequence_id=row['sequence_id'],scenario=row['scenario'],median_depth_m=med,band='0.1-0.3' if med<.3 else '0.3-1' if med<1 else '1-3' if med<3 else '3-10',all=stats(pred,truth,valid),common_mask=stats(pred,truth,valid&mask),raw_missing=stats(pred,truth,valid&(x[0,4:5]<.5)))
   records.append(rec)
   if (i+1)%16==0:print(json.dumps(dict(method=args.method,completed=i+1,total=len(d))),flush=True)
 groups={'all':records}
 for s in sorted({r['scenario'] for r in records}):groups[s]=[r for r in records if r['scenario']==s]
 for b in ('0.1-0.3','0.3-1','1-3','3-10'):groups[b]=[r for r in records if r['band']==b]
 report=dict(method=args.method,manifest=args.manifest,manifest_sha256=digest(args.manifest),script_sha256=digest(__file__),stress=args.stress,sources={k:dict(path=v,sha256=digest(v)) for k,v in sources.items()},scope='axial camera depth; GT ROI evaluation, not deployable liquid height; no GT scale alignment; latency excludes common feature extraction and file IO',latency_ms=dict(median=float(np.median(latency)) if latency else None,p95=float(np.percentile(latency,95)) if latency else None),summary={g:{view:summary([r[view] for r in rs]) for view in ('all','common_mask','raw_missing')} for g,rs in groups.items()},frames=records)
 save(out,report);print(json.dumps(report['summary']['all']),flush=True)
if __name__=='__main__':main()
