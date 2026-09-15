import inspect
import torch
from liquid_depth.models.precision_v17 import route_measurements,fit_surface,precision_loss,PrecisionPipeline

def test_eroded_gate_keeps_only_predicted_interior():
    raw=torch.ones(1,1,12,12);prob=torch.zeros_like(raw);prob[:,:,2:10,2:10]=1
    selected,count=route_measurements(raw,prob,-2,min_points=1)
    assert count.item()==16 and selected.sum()==16
def test_sparse_route_does_not_fall_back_to_background():
    raw=torch.ones(1,1,12,12);prob=torch.zeros_like(raw);prob[:,:,4:6,4:6]=1
    selected,count=route_measurements(raw,prob,0)
    assert count.item()==4 and selected.sum()==0
def test_inverse_depth_fit_recovers_slanted_plane_with_outliers():
    y,x=torch.meshgrid(torch.linspace(-1,1,40),torch.linspace(-1,1,48),indexing='ij')
    truth=(1/(.08*x+.04*y+.7))[None,None];observed=truth.clone()
    observed[:,:,::5,:]*=1.5
    fitted=fit_surface(observed,torch.ones_like(observed))
    assert (fitted-truth).abs().max()<.001
def test_fit_preserves_anchorless_or_degenerate_prediction():
    d=torch.zeros(1,1,40,48);assert torch.equal(fit_surface(d,torch.ones_like(d)),d)
    d=torch.ones_like(d);mask=torch.zeros_like(d);mask[:,:,20,:]=1
    assert torch.equal(fit_surface(d,mask),d)
def test_large_error_loss_retains_gradient():
    pred=torch.tensor([[[[1.1,2.]]]],requires_grad=True);truth=torch.ones_like(pred)
    precision_loss(pred,truth,torch.ones_like(pred)).backward()
    assert pred.grad[0,0,0,1]>=pred.grad[0,0,0,0]*.99
def test_public_pipeline_needs_only_observations():
    assert list(inspect.signature(PrecisionPipeline.forward).parameters)==['self','rgb','raw']
    assert list(inspect.signature(fit_surface).parameters)==['depth','probability','min_points']

class FakeDepth(torch.nn.Module):
    def __init__(self,z,mask=None):super().__init__();self.z=z;self.mask=mask
    def forward(self,rgb,raw):
        return dict(depth_m=self.z.clone(),mask_probability=torch.ones_like(self.z) if self.mask is None else self.mask,has_depth_anchor=torch.ones(self.z.shape[0],dtype=torch.bool))
def test_blend_uses_predicted_distance_and_preserves_missing():
    from liquid_depth.models.precision_v17 import DistanceBlendPipeline
    near=torch.tensor([.4,1.5,3.,0.])[:,None,None,None].expand(-1,1,4,4).clone()
    far=near+1
    pipe=DistanceBlendPipeline(FakeDepth(near),FakeDepth(far),1.,2.)
    out=pipe(torch.zeros(4,3,4,4),torch.full_like(near,8.))
    assert torch.allclose(out['depth_m'][:,0,0,0],torch.tensor([.4,2.,4.,0.]))
    assert torch.allclose(out['far_weight'],torch.tensor([0.,.5,1.,0.]))
    assert list(inspect.signature(pipe.forward).parameters)==['rgb','raw']
def test_blend_distance_ignores_background_and_falls_back_on_missing_far():
    from liquid_depth.models.precision_v17 import DistanceBlendPipeline
    near=torch.full((1,1,4,4),8.);near[:,:,1:3,1:3]=.2
    mask=torch.zeros_like(near);mask[:,:,1:3,1:3]=1
    pipe=DistanceBlendPipeline(FakeDepth(near,mask),FakeDepth(torch.zeros_like(near)))
    out=pipe(torch.empty(0),torch.empty(0))
    assert out['estimated_distance_m'].item()<.201
    assert torch.equal(out['depth_m'],near)
