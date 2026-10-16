"""Fine tune small objects and hard negatives; all current-height labels are training only."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from train_rgbd_evidence_v2 import EvidenceData, balanced_ce, dice, validation, weak_real_patches

from liquid_depth.rgbd_evidence_model import RGBDEvidenceNet


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--initialize', type=Path, required=True)
    p.add_argument('--seed', type=int, default=202610155361); p.add_argument('--epochs', type=int, default=18)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); np.random.seed(a.seed % (2**32)); torch.set_num_threads(4); cv2.setNumThreads(1)
    train = EvidenceData([a.root/f'{i:05d}' for i in range(396)], a.seed, True)
    val = EvidenceData([a.root/f'{i:05d}' for i in range(396, 594)], a.seed)
    sampler = WeightedRandomSampler([2.5 if (s == 3).any() else 1.5 for _, s, _, _ in train.items], len(train), True)
    loader = DataLoader(train, batch_size=6, sampler=sampler); vl = DataLoader(val, batch_size=6)
    real, info = weak_real_patches(Path('/root/autodl-tmp/liquid-depth-data/external/water-obstacle-priors-v2'))
    (a.output/'real-priors-used.json').write_text(json.dumps(info))
    model = RGBDEvidenceNet().cuda(); model.load_state_dict(torch.load(a.initialize, weights_only=False)['model'])
    opt = torch.optim.AdamW(model.parameters(), lr=.0002, weight_decay=.0001)
    best = validation(model, vl, 'cuda')['score']; history = []
    torch.save({'model': model.state_dict(), 'validation': {'score': best}, 'epoch': 0}, a.output/'best.pt')
    for epoch in range(a.epochs):
        model.train(); losses = []
        for step, batch in enumerate(loader):
            x, s, o, r = [v.cuda() for v in batch]; opt.zero_grad(); out = model(x)
            ce = F.cross_entropy(out['semantic'], s, reduction='none')
            neg = ce[(s != 3) & (out['semantic'].argmax(1) == 3)]
            hard = neg.mean() if neg.numel() else ce.sum()*0
            reliability = F.binary_cross_entropy_with_logits(out['reliability'][:, 0], r.clamp(0, 1), reduction='none')
            reliability = torch.stack([reliability[r == k].mean() for k in [0, 1] if (r == k).any()]).mean()
            loss = balanced_ce(out['semantic'], s)+.7*dice(out['semantic'], s)+.35*balanced_ce(out['origin'], o)+.25*reliability+.12*hard
            if real and step % 10 == 0:
                rx, ry = real[np.random.randint(len(real))]
                rl = model(rx[None].cuda())['semantic']
                loss = loss+.1*(balanced_ce(rl, ry[None].cuda())+.5*dice(rl, ry[None].cuda()))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); losses.append(float(loss.detach()))
        row = {'epoch': epoch+1, 'loss': float(np.mean(losses)), **validation(model, vl, 'cuda')}; history.append(row)
        if row['score'] > best:
            best = row['score']; torch.save({'model': model.state_dict(), 'validation': row, 'epoch': epoch+1}, a.output/'best.pt')
        print(json.dumps(row), flush=True)
    (a.output/'training.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__': main()
