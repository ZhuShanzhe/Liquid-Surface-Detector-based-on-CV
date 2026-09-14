import pytest
import torch
from liquid_depth.models.layer_voting import fuse_ballots, predict_liquid_vote, sample_prediction, make_ballot


def ballots(depths, accepted=None):
    return [{'depth_m': torch.tensor([[[[d]]]]),
             'confidence': torch.tensor([[[[.8]]]]),
             'accepted': torch.full((1,1,1,1), True if accepted is None else accepted[i]),
             'layer_index': torch.tensor([[[[i % 4]]]])}
            for i, d in enumerate(depths)]


def test_metric_votes_ignore_layer_indices_and_reject_outlier():
    r = fuse_ballots(ballots([1., 1.002, .999, 1.3, 1.001]), torch.ones(1,1,1,1))
    assert r['accepted'].item() and r['support'].item() == 4
    assert abs(r['depth_m'].item() - 1.) < .002


def test_permutation_invariance():
    b = ballots([1.1, 1., 1.002, 1.001, 1.2])
    a = fuse_ballots(b, torch.ones(1,1,1,1), weighted=True)
    c = fuse_ballots(b[::-1], torch.ones(1,1,1,1), weighted=True)
    assert torch.equal(a['depth_m'], c['depth_m'])
    assert torch.equal(a['accepted'], c['accepted'])


def test_abstentions_do_not_create_majority():
    r = fuse_ballots(ballots([1.]*5, [True,False,False,False,False]), torch.ones(1,1,1,1))
    assert not r['accepted'].item()


def test_tied_modes_reject_without_averaging_layers():
    r = fuse_ballots(ballots([1.,1.,1.1,1.1]), torch.ones(1,1,1,1), min_fraction=0)
    assert not r['accepted'].item()
    assert r['depth_m'].item() == 1.


def test_all_invalid_finite_rejection():
    r = fuse_ballots(ballots([float('nan'),float('inf'),0.]), torch.ones(1,1,1,1))
    assert not r['accepted'].item() and torch.isfinite(r['depth_m']).all()


def test_consistent_wrong_layer_is_not_solved_by_votes():
    # Agreement does not establish truth; this negative control is intentional.
    r = fuse_ballots(ballots([1.04]*5), torch.ones(1,1,1,1))
    assert r['accepted'].item() and r['agreement'].item() == 1.
    assert abs(r['depth_m'].item() - 1.) > .03


def test_prior_disagreement_rejects():
    r = fuse_ballots(ballots([2.]*5), torch.ones(1,1,1,1))
    assert not r['accepted'].item() and r['rejection_code'].item() == 3


def test_public_api_requires_eval_and_supported_sample_count():
    m = torch.nn.Identity()
    with pytest.raises(ValueError):
        predict_liquid_vote(m, torch.zeros(1,5,4,4))
    m.eval()
    with pytest.raises(ValueError):
        predict_liquid_vote(m, torch.zeros(1,5,4,4), samples=2)


class ToyModel(torch.nn.Module):
    def forward(self, x):
        z = 1 + x[:, :1] * .01
        depths = torch.cat([z, z + .2], 1)
        return {'depth_m': z, 'mask_logits': torch.ones_like(z)*9,
                'layer_depths_m': depths, 'layer_confidence': torch.ones_like(depths)*.9,
                'liquid_interface_probability': torch.cat([torch.ones_like(z)*.9, torch.ones_like(z)*.1],1),
                'layer_presence_logits': torch.ones_like(depths)*9}


def test_flip_alignment_is_exact_for_scalar_depth():
    x = torch.arange(5*4*6).reshape(1,5,4,6).float()/100
    model = ToyModel().eval()
    base = sample_prediction(model,x,0)
    for index in (1,2,5):
        result = sample_prediction(model,x,index)
        assert torch.equal(result['layer_depths_m'],base['layer_depths_m'])


def test_component_permutation_preserves_selected_depth():
    pred = ToyModel()(torch.zeros(1,5,4,4))
    original = make_ballot(pred)
    for key in ('layer_depths_m','layer_confidence','liquid_interface_probability','layer_presence_logits'):
        pred[key] = pred[key].flip(1)
    reordered = make_ballot(pred)
    assert torch.equal(original['depth_m'],reordered['depth_m'])
    assert torch.equal(original['accepted'],reordered['accepted'])


def test_repeated_same_input_is_identical_not_independent_evidence():
    model = ToyModel().eval(); x = torch.zeros(1,5,4,4)
    b = make_ballot(sample_prediction(model,x,0))
    r = fuse_ballots([b]*5, torch.ones(1,1,4,4))
    assert torch.equal(r['depth_m'],b['depth_m'])
    assert (r['agreement'] == 1).all()
