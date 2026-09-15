from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
import torch
from liquid_depth.models.candidate_ranker_v13 import ResidualSetRanker, ranking_loss_v13, pairwise_ranking_loss
from liquid_depth.models.candidate_protocol_v13 import operating_point, THRESHOLDS, SCENARIOS

_spec = importlib.util.spec_from_file_location('v12_test_fixture', Path(__file__).with_name('test_candidate_ranker.py'))
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def fixture():
    x, p, rays = _fixture.fixture()
    p['rank_features'] = torch.rand(2,24,16,20)
    return x, p, rays


def test_zero_residual_preserves_v12_and_existing_candidates():
    x,p,r = fixture()
    m = ResidualSetRanker().eval()
    old, q = m.teacher(x,p,r), m(x,p,r)
    torch.testing.assert_close(old['quality_logits'],q['quality_logits'])
    torch.testing.assert_close(old['depth_m'],q['depth_m'])
    torch.testing.assert_close(old['raw_confidence'].clamp(1e-5,1-1e-5),q['raw_confidence'])
    assert ((q['depth_m']-p['layer_depths_m']).abs().amin(1)==0).all()


def test_trained_set_heads_remain_permutation_equivariant():
    x,p,r = fixture()
    m = ResidualSetRanker().eval()
    for head in (m.residual,m.none_delta,m.selected_delta):
        torch.nn.init.normal_(head[-1].weight,std=.1)
    q = m(x,p,r)
    order = torch.tensor([3,1,0,2])
    pp = {k:(v[:,order] if v.shape[1]==4 else v) for k,v in p.items()}
    qq = m(x,pp,r)
    torch.testing.assert_close(q['quality_logits'][:,order],qq['quality_logits'])
    for key in ('depth_m','none_logits','raw_confidence'):
        torch.testing.assert_close(q[key],qq[key])


@pytest.mark.parametrize('has_labels',[False,True])
def test_no_good_candidate_has_finite_loss_and_no_teacher_gradient(has_labels):
    x,p,r = fixture()
    p['layer_depths_m'].requires_grad_(True)
    p['rank_features'].requires_grad_(True)
    m = ResidualSetRanker().train()
    y = {'depth_m':torch.ones_like(p['depth_m'])*5,'valid':torch.ones_like(p['depth_m'])*has_labels}
    q = m(x,p,r)
    loss = ranking_loss_v13(q,p['layer_depths_m'],y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(t.grad is None for t in m.teacher.parameters())
    assert p['layer_depths_m'].grad is None
    assert p['rank_features'].grad is None
    assert m.selected_delta[-1].bias.grad.item()>0


def test_pairwise_does_not_pull_strong_positive_down_to_equal_other_positive():
    logits = torch.tensor([6.,1.,-2.,-3.]).reshape(1,4,1,1).requires_grad_()
    good = torch.tensor([True,True,False,False]).reshape(1,4,1,1)
    loss = pairwise_ranking_loss(logits,good,torch.ones(1,1,1,1,dtype=torch.bool))
    loss.backward()
    assert (logits.grad[:,:2]<0).all()
    assert (logits.grad[:,2:]>0).all()


def test_pairwise_all_good_and_all_bad_have_no_forced_competition():
    for positive in (False,True):
        logits = torch.randn(1,4,2,2,requires_grad=True)
        loss = pairwise_ranking_loss(logits,torch.full_like(logits,positive,dtype=torch.bool),
                                     torch.ones(1,1,2,2,dtype=torch.bool))
        assert loss.item()==0
        loss.backward()
        assert logits.grad.abs().sum()==0


def report():
    row = dict(depth_mae_mm=15.,depth_abs_rel=.025,pixel_tolerance_pass=.6,
               valid_pixel_coverage=.6,non_target_output_fraction=.02,within_tolerance_coverage=.36)
    return {'summary':{s:{'selected':{str(t):deepcopy(row) for t in THRESHOLDS}} for s in ('all',)+SCENARIOS}}


def test_global_success_cannot_hide_transparent_regression():
    base = report()
    result = report()
    for row in result['summary']['transparent']['selected'].values():
        row['pixel_tolerance_pass']=.4
    out = operating_point(result,base)
    assert out['threshold'] is None
    assert out['rejected_by_group']['0.4']['transparent']==['precision']


def test_missing_scene_and_zero_coverage_cannot_qualify():
    base = report()
    result = report()
    del result['summary']['compound']
    assert operating_point(result,base)['threshold'] is None
    result = report()
    for row in result['summary']['translucent']['selected'].values():
        row['valid_pixel_coverage']=.1
    assert operating_point(result,base)['threshold'] is None


def test_qualifying_threshold_maximizes_macro_correct_coverage():
    base = report()
    result = report()
    for s in SCENARIOS:
        result['summary'][s]['selected']['0.6']['within_tolerance_coverage']=.4
    assert operating_point(result,base)['threshold']==.6


def test_evidence_mixture_is_permutation_equivariant_and_frozen():
    from liquid_depth.models.candidate_ranker_v13 import EvidenceMixtureRanker
    x,p,r = fixture()
    p['rank_features'].requires_grad_(True)
    model = EvidenceMixtureRanker().eval()
    q = model(x,p,r)
    order = torch.tensor([2,0,3,1])
    pp = {k:(v[:,order] if v.shape[1]==4 else v) for k,v in p.items()}
    qq = model(x,pp,r)
    torch.testing.assert_close(q['quality_logits'][:,order],qq['quality_logits'])
    torch.testing.assert_close(q['depth_m'],qq['depth_m'])
    torch.testing.assert_close(q['raw_confidence'],qq['raw_confidence'])
    y = {'depth_m':p['depth_m'],'valid':torch.ones_like(p['depth_m'])}
    ranking_loss_v13(q,p['layer_depths_m'],y).backward()
    assert p['rank_features'].grad is None
