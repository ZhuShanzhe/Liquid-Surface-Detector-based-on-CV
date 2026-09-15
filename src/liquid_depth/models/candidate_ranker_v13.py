"""Opt-in V13: residual set interaction and explicit selected-point correctness.

Candidates and the V12 teacher stay frozen. No geometry or production routing.
Probabilities still require independent calibration; they are not OOD guarantees.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from .candidate_ranker import CandidateRanker


def frame_mean(value, mask):
    mask = mask.expand_as(value).to(value.dtype)
    count = mask.flatten(1).sum(1)
    means = (value * mask).flatten(1).sum(1) / count.clamp_min(1)
    active = count > 0
    return (means * active).sum() / active.sum().clamp_min(1)


def pairwise_ranking_loss(logits, good, valid):
    """Only good/bad pairs compete; multiple correct depths need not tie."""
    pairs = good[:, :, None] & (~good[:, None, :]) & valid[:, None]
    difference = logits[:, :, None] - logits[:, None, :]
    return frame_mean(F.softplus(1. - difference), pairs)


class ResidualSetRanker(nn.Module):
    def __init__(self, context_features=24):
        super().__init__()
        self.geometry = False
        self.context_features = context_features
        self.teacher = CandidateRanker(False, context_features).requires_grad_(False).eval()
        self.image_context = nn.Sequential(nn.Conv2d(5, 8, 3, padding=1), nn.SiLU())
        self.embed = nn.Sequential(nn.Conv2d(17, 16, 1), nn.SiLU())
        self.residual = nn.Sequential(nn.Conv2d(48, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        self.none_delta = nn.Sequential(nn.Conv2d(33, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        self.selected_delta = nn.Sequential(nn.Conv2d(52, 16, 1), nn.SiLU(), nn.Conv2d(16, 1, 1))
        for head in (self.residual, self.none_delta, self.selected_delta):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, x, p, rays):
        with torch.no_grad():
            old = self.teacher(x, p, rays)
        d = p['layer_depths_m'].detach().float()
        prior = p['depth_m'].detach().float().clamp_min(.1)
        raw = .1 * torch.exp(x[:, 3:4].float().clamp(0, 1) * math.log(100))
        separation = (d[:, :, None] - d[:, None, :]).abs()
        eye = torch.eye(4, device=d.device, dtype=torch.bool)[None, :, :, None, None]
        separation = separation.masked_fill(eye, float('inf')).amin(2) / d.clamp_min(.1)
        f = torch.stack([
            old['quality_logits'].float().clamp(-12, 12) / 4,
            torch.log(d.clamp_min(.1) / .1) / math.log(100),
            torch.log(d.clamp_min(.1) / prior).clamp(-4, 4),
            (p['layer_scales_m'].detach().float() / d.clamp_min(.1)).clamp(0, 10),
            p['layer_presence_logits'].detach().float().sigmoid(),
            p['liquid_interface_probability'].detach().float(),
            p['layer_confidence'].detach().float(),
            separation.clamp_max(10),
            ((d - raw) / d.clamp_min(.1)).clamp(-10, 10) * x[:, 4:5],
        ], 2)
        b, k, _, h, w = f.shape
        context = self.image_context(x.float())[:, None].expand(-1, k, -1, -1, -1)
        e = self.embed(torch.cat([f, context], 2).reshape(b*k, 17, h, w)).reshape(b, k, 16, h, w)
        mean, maximum = e.mean(1), e.amax(1)
        pooled = torch.cat([mean, maximum], 1)
        shared = pooled[:, None].expand(-1, k, -1, -1, -1)
        delta = self.residual(torch.cat([e, shared], 2).reshape(b*k, 48, h, w)).reshape(b, k, h, w)
        logits = old['quality_logits'].float() + 4 * torch.tanh(delta.float())
        index = logits.argmax(1, keepdim=True)
        none = old['none_logits'].float() + self.none_delta(torch.cat([pooled, old['none_logits'].float()], 1)).float()
        chosen = e.gather(1, index[:, :, None].expand(-1, -1, 16, -1, -1)).squeeze(1)
        top = logits.topk(2, dim=1).values
        # Base score follows the NEW selection, never the old argmax.
        initial = logits.sigmoid().gather(1, index) * (1 - none.sigmoid())
        selected_features = torch.cat([chosen, pooled, top[:, :1], top[:, :1]-top[:, 1:2],
                                       none, x[:, 4:5].float()], 1)
        selected = torch.logit(initial.clamp(1e-5, 1-1e-5)) + self.selected_delta(selected_features).float()
        return {'quality_logits': logits, 'none_logits': none, 'index': index,
                'depth_m': d.gather(1, index), 'selected_logits': selected,
                'raw_confidence': selected.sigmoid()}


def ranking_loss_v13(result, candidates, y):
    valid = y['valid'] > 0
    good = ((candidates.detach() - y['depth_m']).abs() <= (y['depth_m']*.02).clamp_min(.005)) & valid
    exists = good.any(1, keepdim=True)
    quality_map = F.binary_cross_entropy_with_logits(result['quality_logits'].float(), good.float(), reduction='none')
    quality = frame_mean(quality_map, valid) + .1 * frame_mean(quality_map, ~valid)
    none_map = F.binary_cross_entropy_with_logits(result['none_logits'].float(), (~exists).float(), reduction='none')
    # Equal frame contribution; explicitly include no-correct-candidate targets.
    none = .5 * frame_mean(none_map, exists) + .5 * frame_mean(none_map, valid & ~exists)
    none = none + .1 * frame_mean(none_map, ~valid)
    rank = pairwise_ranking_loss(result['quality_logits'].float(), good, valid)
    total = quality + .5 * none + .5 * rank
    if 'selected_logits' in result:
        selected_good = good.gather(1, result['index']).float()
        score_map = F.binary_cross_entropy_with_logits(result['selected_logits'].float(), selected_good, reduction='none')
        total = total + frame_mean(score_map, valid) + .1 * frame_mean(score_map, ~valid)
    return total


class EvidenceMixtureRanker(nn.Module):
    """Learn which observable selector evidence to trust; no scene labels at runtime."""
    def __init__(self, context_features=24):
        super().__init__()
        self.geometry = False
        self.context_features = context_features
        self.context = nn.Sequential(nn.Conv2d(5+context_features,16,3,padding=1),nn.SiLU(),
                                     nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.global_context = nn.Conv2d(context_features,16,1)
        self.gates = nn.Conv2d(16,5,1)
        nn.init.zeros_(self.gates.weight)
        nn.init.zeros_(self.gates.bias)
        self.bias = nn.Conv2d(16,1,1)
        self.none_head = nn.Sequential(nn.Conv2d(18,16,1),nn.SiLU(),nn.Conv2d(16,1,1))
        self.selected_head = nn.Sequential(nn.Conv2d(19,16,1),nn.SiLU(),nn.Conv2d(16,1,1))

    def forward(self, x, p, rays):
        d = p['layer_depths_m'].detach().float()
        prior = p['depth_m'].detach().float().clamp_min(.1)
        frozen = p['rank_features'].detach().float()
        context = self.context(torch.cat([x.float(),frozen],1))
        context = context + self.global_context(frozen.mean((2,3),keepdim=True))
        raw = .1*torch.exp(x[:,3:4].float().clamp(0,1)*math.log(100))
        raw_score = -(d-raw).abs()/(raw*.02).clamp_min(.005)
        scores = torch.stack([
            -(d-prior).abs()/(prior*.02).clamp_min(.005),
            raw_score,
            p['layer_presence_logits'].detach().float(),
            torch.log(p['layer_confidence'].detach().float().clamp_min(1e-7)),
            torch.log(p['liquid_interface_probability'].detach().float().clamp_min(1e-7)),
        ],2)
        # Standardize within each unordered candidate set, not across scenes.
        mean = scores.mean(1,keepdim=True)
        scores = (scores-mean)/scores.std(1,keepdim=True,unbiased=False).clamp_min(.05)
        scores = scores.clamp(-4,4)
        availability = torch.ones_like(scores[:,:1])
        availability[:,:,1] = x[:,4:5].float()
        scores = scores*availability
        mixture = self.gates(context).float().softmax(1)
        logits = 2*(scores*mixture[:,None]).sum(2) + self.bias(context).float()
        top = logits.topk(2,dim=1).values
        index = logits.argmax(1,keepdim=True)
        none = self.none_head(torch.cat([context,logits.mean(1,keepdim=True),logits.amax(1,keepdim=True)],1)).float()
        selected = self.selected_head(torch.cat([context,top[:,:1],top[:,:1]-top[:,1:2],x[:,4:5].float()],1)).float()
        return {'quality_logits':logits,'none_logits':none,'selected_logits':selected,'index':index,
                'depth_m':d.gather(1,index),'raw_confidence':selected.sigmoid()}
