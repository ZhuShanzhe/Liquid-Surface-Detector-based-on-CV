import pytest
import torch
import numpy as np
from liquid_depth.models.candidate_ranker import CandidateRanker,plane_support,ranking_loss,MonotoneCalibrator


def fixture():
    torch.manual_seed(19)
    b,h,w=2,16,20
    d=.2+torch.rand(b,4,h,w)
    p={'layer_depths_m':d,'depth_m':d[:,:1],
       'normal':torch.nn.functional.normalize(torch.rand(b,3,h,w),dim=1),
       'mask_logits':torch.ones(b,1,h,w),'layer_scales_m':torch.ones_like(d)*.01,
       'layer_presence_logits':torch.rand_like(d),'liquid_interface_probability':torch.rand_like(d),
       'layer_confidence':torch.rand_like(d)}
    rays=torch.zeros(b,3,h,w);rays[:,2]=-1
    x=torch.rand(b,5,h,w)
    return x,p,rays


def test_permutation_equivariant_quality_and_invariant_output():
    x,p,r=fixture();m=CandidateRanker().eval();q=m(x,p,r)
    order=torch.tensor([2,0,3,1]);pp={k:(v[:,order] if v.shape[1]==4 else v) for k,v in p.items()}
    qq=m(x,pp,r)
    torch.testing.assert_close(q['quality_logits'][:,order],qq['quality_logits'],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(q['none_logits'],qq['none_logits'],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(q['depth_m'],qq['depth_m'])


def test_geometry_is_zero_for_flat_consistent_planes():
    d=torch.tensor([.2,.4,.6,.8])[None,:,None,None].expand(1,4,16,20)
    normal=torch.zeros(1,3,16,20);normal[:,2]=1
    rays=normal*-1
    assert float(plane_support(d,normal,rays).abs().max())==0


def test_geometry_supports_tilted_planes_in_axial_depth_coordinates():
    v,u=torch.meshgrid(torch.linspace(-.3,.3,16),torch.linspace(-.4,.4,20),indexing='ij')
    rays=torch.stack([u,-v,-torch.ones_like(u)])[None]
    n=torch.tensor([.2,.1,1.]);n=n/n.norm();normal=n[None,:,None,None].expand_as(rays)
    projection=(normal*rays).sum(1,keepdim=True)
    d=-torch.tensor([.2,.4,.6,.8])[None,:,None,None]/projection
    assert float(plane_support(d,normal,rays).abs().max())<1e-4


@pytest.mark.parametrize('all_invalid',[False,True])
def test_no_correct_candidate_has_no_forced_positive(all_invalid):
    x,p,r=fixture();m=CandidateRanker(False);p['layer_depths_m'].requires_grad_(True)
    q=m(x,p,r);y={'depth_m':torch.full_like(p['depth_m'],5.),'valid':torch.ones_like(p['depth_m'])}
    if all_invalid:y['valid'].zero_()
    loss=ranking_loss(q,p['layer_depths_m'].detach(),y);assert torch.isfinite(loss)
    loss.backward();assert p['layer_depths_m'].grad is None
    assert m.none_head[-1].bias.grad.item()<0 # Gradient descent increases none probability.


def test_calibrator_monotone_empty_bins_and_clamped_endpoints():
    c=MonotoneCalibrator(bins=4).fit_histogram([10,0,10,10],[8,0,2,9])
    assert np.all(np.diff(c.values)>=0)
    assert c.values[0]==pytest.approx(.5)
    out=c(torch.tensor([-1.,0.,.5,1.,2.]))
    assert torch.isfinite(out).all();assert out[0]==out[1];assert out[-1]==out[-2]
    with pytest.raises(ValueError):MonotoneCalibrator(bins=4).fit_histogram([0]*4,[0]*4)


def test_ranker_only_selects_existing_depths():
    x,p,r=fixture();q=CandidateRanker()(x,p,r)
    assert ((q['depth_m']-p['layer_depths_m']).abs().min(1).values==0).all()
    assert ((q['raw_confidence']>=0)&(q['raw_confidence']<=1)).all()


def test_feature_context_does_not_train_frozen_features():
    x,p,r=fixture();p['rank_features']=torch.rand(2,24,16,20,requires_grad=True)
    m=CandidateRanker(True,24);q=m(x,p,r)
    (q['quality_logits'].sum()+q['none_logits'].sum()).backward()
    assert p['rank_features'].grad is None
    assert m.global_context.weight.grad is not None
