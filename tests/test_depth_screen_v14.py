"""Metric and geometry-contract tests for the independent screening evaluator."""
import importlib.util
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from screen_depth_methods_v14 import stats,fit_plane

def test_missing_outputs_are_not_counted_as_success():
 truth=torch.tensor([1.,1.,1.,1.]);pred=torch.tensor([1.,float('nan'),0.,1.1])
 result=stats(pred,truth,torch.ones(4,dtype=torch.bool))
 assert result['possible']==4 and result['n']==2 and result['good']==1

def test_axial_tolerance_boundary():
 truth=torch.tensor([.1,1.,2.]);pred=truth+torch.tensor([.0049,.0199,.0401])
 assert stats(pred,truth,torch.ones(3,dtype=torch.bool))['good']==2

def test_plane_fit_recovers_tilt_without_using_outside_mask():
 v,u=torch.meshgrid(torch.linspace(-.4,.4,32),torch.linspace(-.6,.6,48),indexing='ij')
 rays=torch.stack([u,v,-torch.ones_like(u)])
 truth=(1/(1+u*.1-v*.15))[None]
 mask=torch.zeros_like(truth,dtype=torch.bool);mask[:,4:-4,4:-4]=True
 observed=truth.clone();observed[~mask]=4
 recovered=fit_plane(observed,mask,rays)
 assert (recovered-truth).abs().max()<1e-4

def test_plane_fit_refuses_insufficient_support():
 d=torch.ones(1,4,4);r=torch.ones(3,4,4)
 assert torch.isnan(fit_plane(d,d.bool(),r)).all()
