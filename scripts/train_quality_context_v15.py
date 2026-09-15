#!/usr/bin/env python3
"""Train global-context quality after freezing the selected depth model."""
import argparse,sys,json,hashlib
from pathlib import Path
import torch
import train_liquid_adaptation_v15 as t
from liquid_depth.models.liquid_adaptation_v15 import ContextQualityNet

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['plane','strong'],required=True)
 extra,remaining=ap.parse_known_args();root=t.OUT;arm='plane' if extra.kind=='plane' else 'robust'
 parent=root/'strong/robust/best.pth' if extra.kind=='strong' else None
 if parent is not None:
  info=json.loads(parent.with_name('frozen.json').read_text());assert t.digest(parent)==info['checkpoint_sha256']
 original_load=t.load_frozen;original_make=t.make_depth;holder={}
 def load():
  cand,rank,opts=original_load();holder['cand']=cand;return cand,rank,opts
 def provider(x):return holder['cand'].frozen_features(x)[0]
 def make():
  model=original_make();state=torch.load(parent,map_location='cpu',weights_only=False);model.load_state_dict(state['depth']);return model
 t.OUT=root/('context-'+extra.kind);t.load_frozen=load;t.QualityNet=lambda:ContextQualityNet(provider)
 if parent is not None:
  t.make_depth=make
  from train_liquid_adaptation_strong_v15 import stronger_corrupt
  t.corrupt=stronger_corrupt
 sys.argv=[sys.argv[0],'--arm',arm,'--epochs','0','--quality-epochs','4',*remaining]
 t.main()
 p=t.OUT/arm/'frozen.json';d=json.loads(p.read_text());d['variant']='context-'+extra.kind;d['parent_best_epoch']=info['best_epoch'] if parent is not None else None;d['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();d['parent_checkpoint_sha256']=t.digest(parent) if parent is not None else t.digest(t.RANK);p.write_text(json.dumps(d,indent=2))
if __name__=='__main__':main()
