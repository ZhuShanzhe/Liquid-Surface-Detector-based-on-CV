import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from liquid_depth.models.liquid_adaptation_v15 import prepare_depth,LiquidDFNet,depth_loss
from train_liquid_adaptation_v15 import corrupt,choose_threshold,THRESHOLDS

def test_empty_depth_has_no_scale_anchor():
 _,_,_,_,anchor=prepare_depth(np.zeros((180,320,3),np.float32),np.zeros((180,320),np.float32))
 assert anchor==0

def test_bias_preserves_missing_values():
 raw=np.array([[0,.2],[1,9.9]],np.float32)
 out=corrupt(raw,'bias8',np.random.RandomState(0))
 assert out[0,0]==0 and np.isclose(out[1,0],1.08) and out[1,1]<=10
 assert raw[1,0]==1

def test_zero_correction_preserves_pretrained_depth():
 class Identity(torch.nn.Module):
  def forward(self,rgb,depth):return depth
 model=LiquidDFNet(Identity()).eval()
 b=2;rgb=torch.zeros(b,3,16,16);raw=torch.ones(b,1,16,16)
 out=model(rgb,torch.ones(b,16,16)*.5,torch.ones(b)*.2,torch.ones(b)*1.6,rgb,raw,raw,raw,torch.tensor([1.,0.]))
 assert torch.allclose(out[0],raw[0]) and (out[1]==0).all()

def test_masked_loss_does_not_learn_background_depth():
 d=torch.tensor([[[[1.,8.]]]],requires_grad=True);t=torch.ones_like(d);v=torch.tensor([[[[1.,0.]]]])
 loss=depth_loss(d,t,v);loss.backward();assert loss==0 and d.grad[0,0,0,1]==0

def test_guard_requires_all_validation_conditions():
 good={str(t):dict(precision=.9,coverage=.3,non_target_fraction=.01,success=.27) for t in THRESHOLDS}
 bad={str(t):dict(precision=.7,coverage=.3,non_target_fraction=.01,success=.21) for t in THRESHOLDS}
 assert choose_threshold([{'summary':{'transparent':{'selected':good}}},{'summary':{'transparent':{'selected':bad}}}],'transparent') is None

def test_context_quality_uses_frozen_features_and_preserves_shape():
 from liquid_depth.models.liquid_adaptation_v15 import ContextQualityNet
 calls=[]
 def provider(x):
  calls.append(x.detach().clone())
  return torch.ones(x.shape[0],24,*x.shape[-2:])
 model=ContextQualityNet(provider)
 rgb=torch.zeros(2,3,8,8);raw=torch.ones(2,1,8,8)
 logits=model(rgb,raw,raw,raw,raw)
 assert logits.shape==raw.shape and len(calls)==1
 assert calls[0].shape==(2,5,8,8)
 logits.mean().backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
 assert not any('feature_provider' in k for k in model.state_dict())

def test_context_audit_detects_roi_only_depth(tmp_path):
 import csv
 from audit_depth_context_v15 import audit
 raw=np.array([[1.,0.],[0.,0.]],np.float32);mask=np.array([[1,0],[0,0]],np.uint8)
 np.save(tmp_path/'raw.npy',raw);np.save(tmp_path/'mask.npy',mask)
 p=tmp_path/'manifest.csv'
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['raw_depth_path','mask_path','scenario']);w.writeheader();w.writerow(dict(raw_depth_path='raw.npy',mask_path='mask.npy',scenario='transparent'))
 result=audit(p)['transparent']
 assert result['inside_valid_fraction']==1 and result['outside_valid_fraction']==0
 raw[1,1]=2;np.save(tmp_path/'raw.npy',raw)
 assert audit(p)['transparent']['outside_measured']==1
