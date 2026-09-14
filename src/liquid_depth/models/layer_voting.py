"""Opt-in same-frame TTA consensus. Not an independent metric witness.

Votes refer to aligned metric depths, never unordered component indices.
Each augmentation contributes at most one vote per pixel. Returned consensus
is a stability score, NOT a calibrated probability of correct liquid height.
"""
from __future__ import annotations

import math
import torch

from .layered import select_liquid_interface

SAMPLE_NAMES = ('identity', 'horizontal', 'vertical', 'dark95', 'bright105',
                'both', 'drop05')


def sample_prediction(model, inputs, sample_index):
    """Aligned scalar predictions; reflected normal channels are not consumed."""
    if sample_index not in range(len(SAMPLE_NAMES)):
        raise ValueError('sample_index must be in [0, 6]')
    x = inputs.clone()
    dims = {1: (-1,), 2: (-2,), 5: (-2, -1)}.get(sample_index, ())
    if dims:
        x = x.flip(dims)
    if sample_index in (3, 4):
        mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
        std = x.new_tensor([.229, .224, .225])[None, :, None, None]
        rgb = ((x[:, :3] * std + mean) * (.95 if sample_index == 3 else 1.05)).clamp(0, 1)
        x[:, :3] = (rgb - mean) / std
    if sample_index == 6:
        # Fixed seed and shared spatial pattern make results batch-size invariant.
        g = torch.Generator(device=x.device).manual_seed(20260914)
        drop = torch.rand((1, 1, *x.shape[-2:]), generator=g, device=x.device) < .05
        x[:, 3:5] = torch.where(drop, torch.zeros_like(x[:, 3:5]), x[:, 3:5])
    prediction = model(x)
    keys = ('depth_m', 'mask_logits', 'layer_depths_m', 'layer_confidence',
            'liquid_interface_probability', 'layer_presence_logits')
    return {k: prediction[k].flip(dims) if dims else prediction[k] for k in keys}


def make_ballot(prediction, confidence_threshold=.5):
    result = select_liquid_interface(prediction, prediction['depth_m'],
                                     relative_tolerance=.008,
                                     confidence_threshold=confidence_threshold)
    finite = torch.isfinite(result['depth_m']) & torch.isfinite(result['confidence'])
    result['accepted'] &= (prediction['mask_logits'].sigmoid() >= .5) & finite
    return result


def fuse_ballots(ballots, metric_prior_m, *, weighted=False, min_fraction=.6,
                 absolute_cluster_m=.005, relative_cluster=.01, min_margin=1):
    """Choose a depth mode, use its median, and reject ties/disagreement.

    A strict majority counts against ALL requested samples, including abstentions.
    A cluster is supported only if its elected medoid is itself a valid ballot.
    Priors are model-derived (correlated), not calibrated external measurements.
    """
    if not ballots or not 0 <= min_fraction <= 1:
        raise ValueError('nonempty ballots and fraction in [0,1] required')
    if absolute_cluster_m <= 0 or relative_cluster < 0 or min_margin < 0:
        raise ValueError('invalid clustering thresholds')
    d = torch.cat([b['depth_m'] for b in ballots], dim=1)
    c = torch.cat([b['confidence'] for b in ballots], dim=1)
    valid = torch.cat([b['accepted'] for b in ballots], dim=1).bool()
    valid &= torch.isfinite(d) & torch.isfinite(c) & (d > 0)
    d = torch.where(valid, d, torch.zeros_like(d))
    c = torch.where(valid, c.clamp(0, 1), torch.zeros_like(c))
    tolerance = torch.maximum(torch.minimum(d[:, :, None], d[:, None, :]) * relative_cluster,
                              torch.full_like(d[:, :, None] - d[:, None, :], absolute_cluster_m))
    neighbors = ((d[:, :, None] - d[:, None, :]).abs() <= tolerance)
    neighbors &= valid[:, :, None] & valid[:, None, :]
    counts = neighbors.sum(2)
    weights = c if weighted else torch.ones_like(c)
    scores = (neighbors * weights[:, None]).sum(2)
    # Deterministic depth-based tie breaker keeps component/sample order immaterial.
    best_score = scores.max(1, keepdim=True).values
    candidates = (scores >= best_score - 1e-7) & valid
    winner = torch.where(candidates, d, torch.full_like(d, torch.inf)).argmin(1, keepdim=True)
    members = neighbors.gather(1, winner[:, :, None].expand(-1, -1, d.shape[1], -1, -1)).squeeze(1)
    support = members.sum(1, keepdim=True)
    runner = counts.masked_fill(members, 0).max(1, keepdim=True).values
    mode_values = torch.where(members, d, torch.full_like(d, torch.nan))
    center = mode_values.nanmedian(1, keepdim=True).values
    center = torch.nan_to_num(center)
    agreement = support.float() / len(ballots)
    conf = (c * members).sum(1, keepdim=True) / support.clamp_min(1)
    prior_tolerance = torch.maximum(metric_prior_m.abs() * .008,
                                    torch.full_like(metric_prior_m, .02))
    prior_ok = torch.isfinite(metric_prior_m) & (metric_prior_m > 0)
    prior_ok &= (center - metric_prior_m).abs() <= 3 * prior_tolerance
    enough = support >= max(1, math.ceil(len(ballots) * min_fraction))
    margin_ok = support - runner >= min_margin
    accepted = enough & margin_ok & prior_ok
    reason = torch.zeros_like(support)
    reason = torch.where(~enough, 1, reason)
    reason = torch.where(enough & ~margin_ok, 2, reason)
    reason = torch.where(enough & margin_ok & ~prior_ok, 3, reason)
    return {'depth_m': center, 'accepted': accepted,
            'confidence': conf * agreement * accepted,
            'support': support, 'agreement': agreement, 'runner_support': runner,
            'rejection_code': reason}


@torch.inference_mode()
def predict_liquid_vote(model, inputs, *, samples=5, weighted=True,
                        min_fraction=.6, confidence_threshold=.5):
    """Research-only API for normalized Bx5xHxW input; model must be in eval mode.

    Does not change the default runtime. Does not perform pose alignment across
    video frames and must not be used to pool moving-camera frames directly.
    """
    if model.training:
        raise ValueError('Voting requires model.eval(); do not change training state implicitly')
    if samples not in (3, 5, 7):
        raise ValueError('samples must be 3, 5, or 7')
    ballots = []
    for index in range(samples):
        prediction = sample_prediction(model, inputs, index)
        if index == 0:
            prior = prediction['depth_m']
        ballots.append(make_ballot(prediction, confidence_threshold))
    result = fuse_ballots(ballots, prior, weighted=weighted, min_fraction=min_fraction)
    result['research_only'] = True
    return result
