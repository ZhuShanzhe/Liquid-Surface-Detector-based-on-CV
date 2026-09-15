#!/usr/bin/env python3
"""V13 research experiment. Fresh test is sealed until all selections are frozen."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import train_candidate_ranker_v12 as v12
from train_liquid_candidates_v11 import sha, save_json, weights_for
from liquid_depth.models.candidate_ranker import CandidateRanker, MonotoneCalibrator
from liquid_depth.models.candidate_ranker_v13 import ResidualSetRanker, EvidenceMixtureRanker, ranking_loss_v13
from liquid_depth.models.candidate_protocol_v13 import operating_point, checkpoint_score
from liquid_depth.models.liquid_candidate import corrupt_depth_prior

ARMS = ('pairwise', 'set', 'set_repeat')


def make_ranker(arm, teacher_state):
    if arm.startswith('evidence'):
        return EvidenceMixtureRanker(24).cuda()
    if arm == 'pairwise':
        model = CandidateRanker(False, 24)
        model.load_state_dict(teacher_state)
    else:
        model = ResidualSetRanker(24)
        model.teacher.load_state_dict(teacher_state)
    return model.cuda()


def data_loader(d, args, training=False):
    sampler = None
    if training:
        weights, _ = weights_for(d)
        sampler = WeightedRandomSampler(weights, args.samples, replacement=True,
                    generator=torch.Generator().manual_seed(args.seed))
    return DataLoader(d, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=training and args.workers > 0)


@torch.inference_mode()
def calibrate_mixture(candidate, model, data, args):
    """Clean and fixed +5% shifted inputs on calibration sequences ONLY."""
    count = np.zeros(32)
    positive = np.zeros(32)
    for stress in (False, True):
        for x, y, rays in data_loader(data, args):
            x, rays = x.cuda(), rays.cuda()
            y = {k: v.cuda() for k, v in y.items()}
            if stress:
                x = v12.biased(x)
            p = v12.candidate_prediction(candidate, x, args)
            q = model(x, p, rays)
            mask = p['mask_logits'].sigmoid() >= .5
            good = ((q['depth_m']-y['depth_m']).abs() <= (y['depth_m']*.02).clamp_min(.005)) & (y['valid'] > 0)
            bins = (q['raw_confidence'][mask]*32).long().clamp(0, 31)
            count += torch.bincount(bins, minlength=32).cpu().numpy()
            positive += torch.bincount(bins, weights=good[mask].float(), minlength=32).cpu().numpy()
    c = MonotoneCalibrator().fit_histogram(count, positive)
    return c, {'values': c.values.tolist(), 'count': count.tolist(), 'positive': positive.tolist(),
               'scope': 'same calibration sequences at clean and +5%; not independent samples or an OOD guarantee'}


def source_hashes(args):
    return {k: sha(v) for k, v in {'base': args.base, 'candidates': args.candidates,
            'teacher': args.teacher, 'manifest': args.manifest}.items()}


def train(args):
    marker = args.run_dir/'protocol.json'
    if marker.exists():
        raise FileExistsError('Refusing to overwrite a started experiment')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    data, stop, cal, selection = [v12.get_data(args, split, fold) for split, fold in
                                   [('train', None), ('val', 0), ('val', 1), ('val', 2)]]
    sets = [{r['sequence_id'] for r in d.rows} for d in (data, stop, cal, selection)]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
        raise ValueError('Sequence leakage')
    protocol = {'source_sha256': source_hashes(args), 'epochs': args.epochs,
                'samples': args.samples, 'seed': args.seed, 'arms': args.arms,
                'sequence_ids': {k: sorted(s) for k, s in zip(('train','stop','calibration','selection'), sets)},
                'scope': 'camera axial depth; max(5mm,2%); not calibrated liquid height',
                'historical_development_exposure': 'V11 used historical validation; V12 checkpoint was selected earlier',
                'new_holdout': str(args.fresh_manifest), 'production_promoted': False,
                'training_bias_probability': .5, 'checkpoint_rule': 'worst-scene delta + 0.25 macro delta vs frozen V12',
                'calibration': 'clean and +5% on calibration fold', 'threshold_grid': v12.THRESHOLDS}
    save_json(marker, protocol)
    candidate = v12.load_candidates(args)
    teacher_state = torch.load(args.teacher, map_location='cpu', weights_only=False)
    if teacher_state.get('geometry') or teacher_state.get('context_features') != 24:
        raise ValueError('Expected context V12 without geometry')
    teacher = make_ranker('pairwise', teacher_state['model']).eval()
    teacher_stop = v12.evaluate(candidate, teacher, stop, args)
    save_json(args.run_dir/'teacher_stop.json', teacher_stop)
    base, _ = v12.load_model(args.base)
    baseline = v12.evaluate(base, None, selection, args)
    save_json(args.run_dir/'baseline_selection.json', baseline)
    del base
    decisions = {}
    for arm in args.arms:
        folder = args.run_dir/arm
        folder.mkdir()
        seed = args.seed + int(arm.endswith('_repeat'))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = make_ranker(arm, teacher_state['model'])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        corruption = torch.Generator(device='cuda').manual_seed(seed+17)
        dl = data_loader(data, args, True)
        # Epoch zero remains eligible, so a training regression need not replace V12.
        initial = v12.evaluate(candidate, model, stop, args) if arm.startswith('evidence') else teacher_stop
        best = checkpoint_score(initial, teacher_stop)
        torch.save({'model': model.state_dict(), 'arm': arm, 'epoch': 0}, folder/'best.pth')
        save_json(folder/'best_stop.json', initial)
        start = time.monotonic()
        for epoch in range(1, args.epochs+1):
            model.train()
            total = 0.
            for batch, (x, y, rays) in enumerate(dl, 1):
                x, rays = x.cuda(), rays.cuda()
                y = {k: v.cuda() for k, v in y.items()}
                x = corrupt_depth_prior(x, corruption, probability=.5)
                with torch.no_grad():
                    p = v12.candidate_prediction(candidate, x, args)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    q = model(x, p, rays)
                    loss = ranking_loss_v13(q, p['layer_depths_m'], y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach())
            scheduler.step()
            report = v12.evaluate(candidate, model, stop, args)
            score = checkpoint_score(report, teacher_stop)
            record = {'epoch': epoch, 'score': score, 'loss': total/batch,
                      'summary': report['summary'], 'elapsed_s': time.monotonic()-start}
            with (folder/'history.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False)+'\n')
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'arm': arm, 'epoch': epoch}, folder/'best.pth')
                save_json(folder/'best_stop.json', report)
            print(json.dumps({'arm': arm, 'epoch': epoch, 'score': score, 'best': best,
                              'elapsed_s': record['elapsed_s']}), flush=True)
        checkpoint = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        c, info = calibrate_mixture(candidate, model, cal, args)
        save_json(folder/'calibration.json', info)
        selected = v12.evaluate(candidate, model, selection, args, c)
        save_json(folder/'selection.json', selected)
        stress = v12.evaluate(candidate, model, selection, args, c, stress=True)
        save_json(folder/'selection_stress.json', stress)
        decision = operating_point(selected, baseline)
        decision.update(checkpoint_sha256=sha(folder/'best.pth'),
                        calibration_sha256=sha(folder/'calibration.json'),
                        best_epoch=checkpoint['epoch'])
        decisions[arm] = decision
        save_json(folder/'complete.json', decision)
        print(json.dumps({'arm': arm, 'threshold': decision['threshold'], 'best_epoch': checkpoint['epoch']}), flush=True)
        del model, optimizer, dl
        torch.cuda.empty_cache()
    save_json(args.run_dir/'trained_selection.json', {'protocol': protocol, 'arms': decisions})
    print('TRAINING_COMPLETE; freeze after fresh generation completes, before testing.', flush=True)


def freeze(args):
    path = args.run_dir/'frozen_selection.json'
    if path.exists() or (args.run_dir/'test_started.json').exists():
        raise FileExistsError('Already frozen or tested')
    selected = json.loads((args.run_dir/'trained_selection.json').read_text())
    if selected['protocol']['source_sha256'] != source_hashes(args):
        raise ValueError('Source changed')
    fresh = v12.get_data(args, 'test', manifest=args.fresh_manifest)
    seen = set().union(*(set(v) for v in selected['protocol']['sequence_ids'].values()))
    old = v12.get_data(args, 'test')
    seen.update(r['sequence_id'] for r in old.rows)
    if seen & {r['sequence_id'] for r in fresh.rows}:
        raise ValueError('Fresh test overlaps historical sequences')
    if len(fresh) != 256 or len({r['sequence_id'] for r in fresh.rows}) != 256:
        raise ValueError('Expected all 256 independent new sequences')
    selected['fresh_sha256'] = sha(args.fresh_manifest)
    selected['code_sha256'] = {str(p): sha(p) for p in [
        Path(__file__), Path('scripts/train_candidate_ranker_v12.py'),
        Path('src/liquid_depth/models/candidate_ranker_v13.py'),
        Path('src/liquid_depth/models/candidate_protocol_v13.py')]}
    selected['diagnostic_threshold_for_unqualified'] = .4
    selected['teacher_calibration_sha256'] = sha(args.teacher.with_name('calibration.json'))
    selected['training_code_sha256'] = json.loads((args.run_dir/'training_code_sha256.json').read_text())
    selected['production_promoted'] = False
    save_json(path, selected)
    print('FROZEN', flush=True)


def test(args):
    frozen = json.loads((args.run_dir/'frozen_selection.json').read_text())
    if source_hashes(args) != frozen['protocol']['source_sha256'] or sha(args.fresh_manifest) != frozen['fresh_sha256']:
        raise ValueError('Frozen inputs changed')
    if sha(args.teacher.with_name('calibration.json')) != frozen['teacher_calibration_sha256']:
        raise ValueError('Frozen V12 calibration changed')
    if any(sha(Path(p)) != value for p, value in frozen['code_sha256'].items()):
        raise ValueError('Frozen code changed')
    for arm, info in frozen['arms'].items():
        folder = args.run_dir/arm
        if sha(folder/'best.pth') != info['checkpoint_sha256'] or sha(folder/'calibration.json') != info['calibration_sha256']:
            raise ValueError('Frozen arm changed')
    with (args.run_dir/'test_started.json').open('x') as f:
        json.dump({'frozen_sha256': sha(args.run_dir/'frozen_selection.json')}, f)
    candidate = v12.load_candidates(args)
    base, _ = v12.load_model(args.base)
    teacher_state = torch.load(args.teacher, map_location='cpu', weights_only=False)
    teacher = make_ranker('pairwise', teacher_state['model']).eval()
    teacher_cal = MonotoneCalibrator(json.loads(args.teacher.with_name('calibration.json').read_text())['values'])
    for label, manifest in [('legacy', args.manifest), ('fresh', args.fresh_manifest)]:
        data = v12.get_data(args, 'test', manifest=manifest)
        for name, generator, ranker, c in [('baseline', base, None, None),
                                         ('v12', candidate, teacher, teacher_cal)]:
            for stress in (False, True):
                report = v12.evaluate(generator, ranker, data, args, c, stress)
                save_json(args.run_dir/f'{label}_{name}_{stress}.json', report)
        for arm, info in frozen['arms'].items():
            folder = args.run_dir/arm
            state = torch.load(folder/'best.pth', map_location='cpu', weights_only=False)
            model = make_ranker(arm, teacher_state['model'])
            model.load_state_dict(state['model'])
            model.eval()
            c = MonotoneCalibrator(json.loads((folder/'calibration.json').read_text())['values'])
            for stress in (False, True):
                report = v12.evaluate(candidate, model, data, args, c, stress)
                report['frozen_threshold'] = info['threshold']
                report['diagnostic_only'] = info['threshold'] is None
                save_json(folder/f'{label}_{stress}.json', report)
            del model
            print(json.dumps({'tested': arm, 'dataset': label}), flush=True)
    save_json(args.run_dir/'test_complete.json', {'complete': True, 'production_promoted': False})


def main():
    p = argparse.ArgumentParser()
    for name in ('base', 'candidates', 'teacher', 'manifest', 'fresh-manifest', 'run-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--phase', choices=['train', 'freeze', 'test'], required=True)
    p.add_argument('--arms', nargs='+', choices=list(ARMS)+['evidence','evidence_repeat'], default=list(ARMS))
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--samples', type=int, default=4096)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=20260917)
    args = p.parse_args()
    args.feature_context = True
    torch.set_num_threads(4)
    v12.loader = data_loader
    {'train': train, 'freeze': freeze, 'test': test}[args.phase](args)


if __name__ == '__main__':
    main()
