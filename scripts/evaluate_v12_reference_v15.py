#!/usr/bin/env python3
"""Unchanged V12 ranking and old calibration on the V15 fresh conditions."""
import argparse,json,torch
import train_liquid_adaptation_v15 as t
from liquid_depth.models.candidate_ranker import MonotoneCalibrator

class OriginalQuality(torch.nn.Module):
 def forward(self,*args):return torch.logit(self.score.clamp(1e-6,1-1e-6))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);a=ap.parse_args()
 args=argparse.Namespace(manifest=a.manifest,seed=20260915201,batch_size=8,workers=4)
 torch.set_num_threads(4);cand,rank,opts=t.load_frozen();quality=OriginalQuality().cuda()
 def predict(model,arm,b,cand,rank,opts):
  p=t.v12.candidate_prediction(cand,b['x'],opts);q=rank(b['x'],p,b['rays']);quality.score=q['raw_confidence']
  return q['depth_m'],p['depth_m'],p['mask_logits'].sigmoid()
 t.prediction=predict;t.THRESHOLDS=(.4,*t.THRESHOLDS)
 calinfo=json.loads((t.RANK.parent/'calibration.json').read_text());cal=MonotoneCalibrator(calinfo['values'])
 d=t.v12.get_data(args,'test');out=t.OUT/'v12_reference';out.mkdir(exist_ok=True)
 for mode in ('clean','bias8','bias-8','local10'):
  p=out/f'fresh-{mode}.json'
  if p.exists():raise FileExistsError(p)
  t.save(p,t.evaluate(None,quality,'v12_reference',d,mode,args,cand,rank,opts,cal));print('REFERENCE_DONE',mode,flush=True)
if __name__=='__main__':main()
