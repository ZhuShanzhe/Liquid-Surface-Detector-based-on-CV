import numpy as np
import torch
import pytest
from liquid_depth.liquid_height_metrics import HeightCalibration,estimate_height
from liquid_depth.models.height_objective import differentiable_height,liquid_height_loss


def cal():
    pose=np.diag([1.,-1.,-1.,1.]);pose[2,3]=1
    return HeightCalibration(np.array([[100.,0,39.5],[0,100.,39.5],[0,0,1]]),pose,-.2,(.35,.35))


def test_numpy_torch_estimator_match_and_gradient():
    depth=torch.ones((80,80),requires_grad=True);prob=torch.ones_like(depth)
    result=differentiable_height(depth,prob,cal())
    reference=estimate_height(depth.detach().numpy(),prob.detach().numpy()>0,cal())
    assert float(result)==pytest.approx(reference['statistics']['mean_m'],abs=1e-6)
    result.backward();assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum()>0


def test_height_loss_uses_liquid_tolerance_and_has_nonzero_gradient():
    depth=torch.full((1,1,80,80),1.01,requires_grad=True)
    loss=liquid_height_loss(depth,torch.ones_like(depth),torch.ones_like(depth),torch.ones_like(depth),[cal()],[.2])
    assert float(loss)>1
    loss.backward();assert torch.isfinite(depth.grad).all() and depth.grad.abs().sum()>0
