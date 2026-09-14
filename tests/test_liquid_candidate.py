import torch
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.models.liquid_candidate import LiquidCandidateNet, LiquidCandidateLoss, corrupt_depth_prior


def build(dedicated=True):
    torch.manual_seed(31)
    base=UniversalLiquidSurfaceNet(base_channels=8,num_ray_layers=4,ray_layer_hidden_channels=8).eval()
    return base,LiquidCandidateNet(base,dedicated=dedicated)


def test_four_candidates_and_frozen_base_outputs():
    base,m=build(); x=torch.randn(2,5,16,24)
    with torch.no_grad():a=base(x); b=m(x)
    assert b['layer_depths_m'].shape==(2,4,16,24)
    assert torch.equal(a['depth_m'],b['depth_m'])
    assert torch.equal(a['mask_logits'],b['mask_logits'])
    m.train()
    assert not m.backbone.training
    assert all(not p.requires_grad for p in m.backbone.parameters())


def test_control_is_initially_identical_to_original():
    base,m=build(False);x=torch.randn(1,5,16,24)
    with torch.no_grad():a=base(x); b=m(x)
    assert torch.equal(a['layer_depths_m'],b['layer_depths_m'])


def test_corruption_preserves_rgb_validity_and_missing_depth():
    x=torch.rand(3,5,16,24);x[:,4:5]=(x[:,4:5]>.3).float()
    x[:,3:4]*=x[:,4:5]
    a=corrupt_depth_prior(x,torch.Generator().manual_seed(19),probability=1.)
    b=corrupt_depth_prior(x,torch.Generator().manual_seed(19),probability=1.)
    assert torch.equal(a,b) and torch.equal(a[:,:3],x[:,:3])
    assert torch.equal(a[:,4:5],x[:,4:5])
    assert torch.equal(a[:,3:4][x[:,4:5]==0],x[:,3:4][x[:,4:5]==0])
    assert (a[:,3:4]>=0).all() and (a[:,3:4]<=1).all()


def test_direct_depth_is_not_bounded_to_old_prior_factor():
    _,m=build();x=torch.zeros(1,5,16,24)
    with torch.no_grad():
        m.direct_depth.weight.zero_();m.direct_depth.bias.fill_(-20)
        p=m(x)
    assert p['direct_depth_m'].max()<.101
    assert torch.isfinite(p['layer_depths_m']).all()


def test_loss_backpropagates_only_through_new_heads():
    _,m=build();m.train();x=torch.randn(1,5,16,24)
    y={'depth_m':torch.ones(1,1,16,24),'valid':torch.ones(1,1,16,24),
       'layer_depths_m':torch.cat([torch.ones(1,1,16,24),torch.ones(1,1,16,24)*1.2,
                                 torch.zeros(1,2,16,24)],1),
       'layer_valid':torch.cat([torch.ones(1,2,16,24),torch.zeros(1,2,16,24)],1)}
    loss=LiquidCandidateLoss()(m(x),y)['total'];loss.backward()
    assert torch.isfinite(loss)
    assert m.direct_depth.weight.grad is not None
    assert all(p.grad is None for p in m.backbone.parameters())


def test_empty_supervision_is_finite():
    _,m=build();x=torch.zeros(1,5,16,24)
    y={'depth_m':torch.zeros(1,1,16,24),'valid':torch.zeros(1,1,16,24),
       'layer_depths_m':torch.zeros(1,4,16,24),'layer_valid':torch.zeros(1,4,16,24)}
    assert torch.isfinite(LiquidCandidateLoss()(m(x),y)['total'])
