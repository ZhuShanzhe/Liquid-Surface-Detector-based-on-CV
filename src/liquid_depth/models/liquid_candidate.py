"""Research-only fixed-budget liquid candidate refinement (V11).

The backbone and default runtime are unchanged. The optional liquid branch is
an absolute log-depth regressor, not a bounded offset around the old prior.
It is still RGB-D conditioned, NOT an independent physical depth witness.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .multitask import ConvBlock
from .layered import PermutationInvariantLayerLoss


class LiquidCandidateNet(nn.Module):
    def __init__(self, backbone, dedicated=True):
        super().__init__()
        if backbone.ray_layer_head is None or backbone.ray_layer_head.num_layers != 4:
            raise ValueError('V11 requires exactly four baseline ray components')
        if backbone.level_calibration_enabled or backbone.robust_depth_anchor_enabled:
            raise ValueError('Use the unanchored V9.1 checkpoint; other priors need a separate adapter')
        self.ray_head = copy.deepcopy(backbone.ray_layer_head)
        self.ray_head.requires_grad_(True)
        self.backbone = backbone.requires_grad_(False).eval()
        self.dedicated = bool(dedicated)
        self.min_depth_m = backbone.min_depth_m
        self.max_depth_m = backbone.max_depth_m
        self.log_span = math.log(self.max_depth_m/self.min_depth_m)
        if self.dedicated:
            c = backbone.depth_head.in_channels
            self.context = ConvBlock(c+5, c)
            # Trainable absolute log-depth readout; old readout is initialization only.
            self.direct_depth = copy.deepcopy(backbone.depth_head).requires_grad_(True)
            self.context_depth = nn.Conv2d(c,1,1)
            nn.init.zeros_(self.context_depth.weight); nn.init.zeros_(self.context_depth.bias)
            self.direct_scale = nn.Conv2d(c,1,1)
            nn.init.zeros_(self.direct_scale.weight); nn.init.constant_(self.direct_scale.bias,-4.)
            self.direct_presence = nn.Conv2d(c,1,1)
            self.direct_identity = nn.Conv2d(c,1,1)
            self.direct_quality = nn.Conv2d(c,1,1)
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def frozen_features(self, inputs):
        b = self.backbone
        with torch.no_grad():
            e1 = b.enc1(inputs)
            if b.rgb_prior is not None:
                missing = 1-inputs[:,4:5].clamp(0,1)
                e1 = e1 + torch.sigmoid(b.rgb_prior_scale)*(.2+.8*missing)*b.rgb_prior(inputs[:,:3])
            e2 = b.enc2(b.pool(e1)); e3 = b.enc3(b.pool(e2))
            f = b.bottleneck(b.pool(e3))
            d3 = b.dec3(torch.cat([b._up(f,e3),e3],1))
            d2 = b.dec2(torch.cat([b._up(d3,e2),e2],1))
            features = b.dec1(torch.cat([b._up(d2,e1),e1],1))
            prior = self.min_depth_m*torch.exp(torch.sigmoid(b.depth_head(features))*self.log_span)
            result = {'depth_m':prior,'mask_logits':b.mask_head(features),
                      'normal':F.normalize(b.normal_head(features),dim=1,eps=1e-6)}
        return features, result

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != 5:
            raise ValueError('Expected normalized Bx5xHxW RGB/log-depth/validity input')
        features, result = self.frozen_features(inputs)
        ray = self.ray_head(features, result['depth_m'])
        if not self.dedicated:
            return {**result, **ray}
        context = self.context(torch.cat([features,inputs],1))
        depth_logits = self.direct_depth(features)+self.context_depth(context)
        direct = self.min_depth_m*torch.exp(depth_logits.sigmoid()*self.log_span)
        scale = F.softplus(self.direct_scale(context))+.002
        presence = self.direct_presence(context)
        identity = self.direct_identity(context)
        quality = self.direct_quality(context)
        # Replace the old most-liquid-like component, never consult ground truth.
        remove = ray['liquid_interface_probability'].detach().argmax(1,keepdim=True)
        indices = torch.arange(4,device=inputs.device)[None,:,None,None].expand_as(ray['layer_depths_m'])
        keep = indices.masked_fill(indices==remove,4).sort(1).values[:,:3]
        new = {}
        for key, value in [('layer_depths_m',direct),('layer_scales_m',scale),
                           ('layer_presence_logits',presence),('layer_interface_logits',identity)]:
            new[key] = torch.cat([value,ray[key].gather(1,keep)],1)
        pres = new['layer_presence_logits'].sigmoid()
        confidence = pres*torch.exp(-new['layer_scales_m']/new['layer_depths_m'].clamp_min(.1))
        # Quality is explicitly supervised for the dedicated candidate's tolerance.
        new['layer_confidence'] = torch.cat([confidence[:,:1]*quality.sigmoid(),confidence[:,1:]],1)
        new['liquid_interface_probability'] = new['layer_interface_logits'].softmax(1)*pres
        new['layer_depths_sorted_m'],new['layer_sort_order'] = new['layer_depths_m'].sort(1)
        return {**result,**new,'direct_depth_m':direct,'direct_quality_logits':quality,
                'direct_presence_logits':presence,'replaced_component':remove}


def corrupt_depth_prior(inputs, generator, probability=.5):
    """Training-only wrong-return proxy, derived ONLY from input depth, no labels.

    Leaves RGB and invalid samples untouched. Adds signed coherent bias or
    local farther returns. Does not train recovery of total depth failure.
    """
    x = inputs.clone()
    b,_,h,w=x.shape
    choose=torch.rand((b,1,1,1),generator=generator,device=x.device)<probability
    local=torch.rand((b,1,1,1),generator=generator,device=x.device)<.5
    fraction=.03+.17*torch.rand((b,1,1,1),generator=generator,device=x.device)
    sign=torch.where(torch.rand((b,1,1,1),generator=generator,device=x.device)<.5,-1.,1.)
    raw=.1*torch.exp(x[:,3:4].clamp(0,1)*math.log(100.))
    coarse=torch.rand((b,1,4,4),generator=generator,device=x.device)>.5
    region=F.interpolate(coarse.float(),size=(h,w),mode='nearest')>.5
    modify=choose & (x[:,4:5]>.5) & (~local | region)
    delta=sign*fraction*raw
    biased=(raw+delta).clamp(.1,10.)
    encoded=torch.log(biased/.1)/math.log(100.)
    x[:,3:4]=torch.where(modify,encoded,x[:,3:4])
    return x


def frame_mean(value, valid):
    counts=valid.flatten(1).sum(1)
    per=(value*valid).flatten(1).sum(1)/counts.clamp_min(1)
    return per[counts>0].mean() if bool((counts>0).any()) else value.sum()*0


class LiquidCandidateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=PermutationInvariantLayerLoss(multilayer_pixel_boost=4.)

    def forward(self,p,y):
        valid=y['valid']>0; target=y['depth_m']
        tolerance=torch.maximum(target*.02,torch.full_like(target,.005))
        distance=(p['layer_depths_m']-target).abs()
        oracle=frame_mean(torch.log1p(distance.min(1,keepdim=True).values/tolerance),valid)
        # Shared loss stride reduces matching memory, not model/output resolution.
        keys=('layer_depths_m','layer_scales_m','layer_presence_logits','layer_interface_logits')
        small={k:p[k][...,::4,::4] for k in keys}
        set_loss=self.layers(small,y['layer_depths_m'][...,::4,::4],
                             y['layer_valid'][...,::4,::4],target[...,::4,::4],valid[...,::4,::4])['total']
        direct=oracle*0; calibration=oracle*0; presence=oracle*0
        if 'direct_depth_m' in p:
            error=(p['direct_depth_m']-target).abs()
            direct=frame_mean(torch.log1p(error/tolerance),valid)
            reliable=(error.detach()<=tolerance).float()
            calibration=frame_mean(F.binary_cross_entropy_with_logits(p['direct_quality_logits'],reliable,reduction='none'),valid)
            presence_map=F.binary_cross_entropy_with_logits(p['direct_presence_logits'],valid.float(),reduction='none')
            presence=.5*frame_mean(presence_map,valid)+.5*frame_mean(presence_map,~valid)
        total=.25*set_loss+oracle+direct+.2*calibration+.2*presence
        return {'total':total,'set':set_loss,'oracle':oracle,'direct':direct,
                'quality':calibration,'presence':presence}
