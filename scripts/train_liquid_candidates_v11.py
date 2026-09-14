#!/usr/bin/env python3
"""Four-arm frozen-backbone candidate experiment. No default-route promotion."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from evaluate_layer_voting import load_model, pixel_stats, summarize
from liquid_depth.models.layered import select_liquid_interface
from liquid_depth.models.liquid_candidate import LiquidCandidateNet, LiquidCandidateLoss, corrupt_depth_prior
from liquid_depth.training.universal_dataset import UniversalMultiTaskDataset

SCENARIOS=('transparent','translucent','multilayer','compound')
ARMS={'control':(False,False),'direct':(True,False),'perturb':(False,True),'combined':(True,True)}
THRESHOLDS=(.1,.3,.5,.7)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(args,split):
    d=UniversalMultiTaskDataset(args.manifest,split,(320,180),augment=False,min_depth_m=.1,max_depth_m=10.)
    d.rows=[r for r in d.rows if r['scenario'] in SCENARIOS and r.get('layer_depths_path')]
    return d


def weights_for(d):
    buckets=[]
    for row in d.rows:
        meta=json.loads((d.root/row['metadata_path']).read_text())
        # Training-only range stratification; no metadata enters model inference.
        distance=float(np.linalg.norm(meta['camera_position_m']))
        band=int(np.searchsorted([.3,1.,3.],distance))
        buckets.append((row['scenario'],band))
    counts=Counter(buckets)
    return [1/counts[b] for b in buckets],dict((str(k),v) for k,v in counts.items())


def scalar_summary(frames):
    n=sum(f['n'] for f in frames)
    layers=sum(f['layer_n'] for f in frames)
    return {
        'frames':len(frames),'valid_pixels':n,
        'oracle_recall':sum(f['oracle_good'] for f in frames)/max(n,1),
        'oracle_mae_mm':1000*sum(f['oracle_abs'] for f in frames)/max(n,1),
        'active_oracle_recall':sum(f['active_oracle_good'] for f in frames)/max(n,1),
        'mask_recall':sum(f['masked'] for f in frames)/max(n,1),
        'accurate_candidate_tile_coverage':float(np.mean([f['tile_coverage'] for f in frames])),
        'layer_tolerance_recall':sum(f['layer_good'] for f in frames)/max(layers,1),
        'hard_layer_count_mae':sum(f['count_error'] for f in frames)/max(sum(f['layer_pixels'] for f in frames),1),
        'selected':{str(t):summarize([f['selected'][str(t)] for f in frames]) for t in THRESHOLDS},
    }


@torch.inference_mode()
def evaluate(model,d,args,stress=False):
    model.eval(); frames=[]; offset=0
    loader=DataLoader(d,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    for x,y in loader:
        x=x.cuda(); y={k:v.cuda() for k,v in y.items()}
        if stress:
            # Frozen deterministic +5% range bias; same input transformation per arm.
            x=x.clone(); raw=.1*torch.exp(x[:,3:4]*math.log(100.))
            encoded=torch.log((raw*1.05).clamp(.1,10.)/.1)/math.log(100.)
            x[:,3:4]=torch.where(x[:,4:5]>.5,encoded,x[:,3:4])
        p=model(x); valid=y['valid']>0; truth=y['depth_m']
        tol=torch.maximum(truth*.02,torch.full_like(truth,.005))
        err=(p['layer_depths_m']-truth).abs()
        oracle=err.min(1,keepdim=True).values
        good=(oracle<=tol)&valid
        active=((err<=tol)&(p['layer_presence_logits'].sigmoid()>=.5)).any(1,keepdim=True)&valid
        mask=p['mask_logits'].sigmoid()>=.5
        chosen=select_liquid_interface(p,p['depth_m'],relative_tolerance=.008,confidence_threshold=0.)
        layer_valid=y['layer_valid']>0
        ld=(p['layer_depths_m'][:,:,None]-y['layer_depths_m'][:,None]).abs().min(1).values
        lt=torch.maximum(y['layer_depths_m']*.02,torch.full_like(y['layer_depths_m'],.005))
        lc=layer_valid.sum(1); layer_pixels=lc>0
        count_error=((p['layer_presence_logits'].sigmoid()>=.5).sum(1)-lc).abs()
        target_tiles=F.adaptive_avg_pool2d(valid.float(),(8,8))
        good_tiles=F.adaptive_avg_pool2d(good.float(),(8,8))
        tiled=(good_tiles>=.1*target_tiles)&(target_tiles>0)
        for i in range(x.shape[0]):
            row=d.rows[offset+i]
            item={'sequence_id':row['sequence_id'],'rgb_path':row['rgb_path'],'scenario':row['scenario'],
                  'n':int(valid[i].sum()),'oracle_good':int(good[i].sum()),
                  'oracle_abs':float(oracle[i][valid[i]].sum()),'active_oracle_good':int(active[i].sum()),
                  'masked':int((mask[i]&valid[i]).sum()),
                  'tile_coverage':float(tiled[i].sum()/((target_tiles[i]>0).sum().clamp_min(1))),
                  'layer_n':int(layer_valid[i].sum()),'layer_good':int(((ld[i]<=lt[i])&layer_valid[i]).sum()),
                  'layer_pixels':int(layer_pixels[i].sum()),'count_error':int(count_error[i][layer_pixels[i]].sum()),
                  'selected':{}}
            for threshold in THRESHOLDS:
                emitted=chosen['accepted'][i]&mask[i]&(chosen['confidence'][i]>=threshold)
                item['selected'][str(threshold)]=pixel_stats(chosen['depth_m'][i],emitted,truth[i],valid[i])
            frames.append(item)
        offset+=x.shape[0]
    summary={'all':scalar_summary(frames)}
    summary.update({s:scalar_summary([f for f in frames if f['scenario']==s]) for s in SCENARIOS
                    if any(f['scenario']==s for f in frames)})
    return {'summary':summary,'frames':frames,'stress':stress,
            'scope':'camera-depth candidate accuracy, not bottom-calibrated liquid height'}


def score(report):
    groups=[report['summary'][s] for s in SCENARIOS if s in report['summary']]
    return float(np.mean([g['oracle_recall']+.1*g['accurate_candidate_tile_coverage']
                          +.05*g['selected']['0.5']['within_tolerance_coverage'] for g in groups]))


def build(args,arm):
    base,state=load_model(args.checkpoint)
    return LiquidCandidateNet(base,dedicated=ARMS[arm][0]).cuda(),state


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def train_one(args,arm,train,val):
    directory=args.run_dir/arm
    if directory.exists():
        raise FileExistsError(f'Run directory already exists: {directory}')
    directory.mkdir(parents=True)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model,state=build(args,arm)
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    criterion=LiquidCandidateLoss().cuda()
    weights,buckets=weights_for(train)
    sampling=torch.Generator().manual_seed(args.seed)
    corruption=torch.Generator(device='cuda').manual_seed(args.seed+101)
    loader=DataLoader(train,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True,
                      persistent_workers=args.workers>0,
                      sampler=WeightedRandomSampler(weights,args.samples_per_epoch,replacement=True,generator=sampling))
    protocol={'arm':arm,'dedicated':ARMS[arm][0],'perturb':ARMS[arm][1],
              'checkpoint_sha256':sha(args.checkpoint),'manifest_sha256':sha(args.manifest),
              'seed':args.seed,'epochs':args.epochs,'samples_per_epoch':args.samples_per_epoch,
              'training_records':len(train),'validation_records':len(val),'train_buckets':buckets,
              'candidate_budget':4,'backbone_frozen':True,'trainable_parameters':sum(p.numel() for p in trainable),
              'image_size':[320,180],'set_loss_stride':4,'candidate_tolerance':'max(.005m,.02*camera_depth)',
              'learning_rate':args.learning_rate,'production_promoted':False}
    save_json(directory/'protocol.json',protocol)
    best=-float('inf'); start=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train(); totals=Counter(); batches=0
        for x,y in loader:
            x=x.cuda(non_blocking=True); y={k:v.cuda(non_blocking=True) for k,v in y.items()}
            if ARMS[arm][1]:
                x=corrupt_depth_prior(x,corruption,probability=.5)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model(x)
            # Matching and small metric errors remain float32.
            p={k:v.float() if v.is_floating_point() else v for k,v in p.items()}
            loss=criterion(p,y)
            if not torch.isfinite(loss['total']):
                raise RuntimeError('Nonfinite loss; refusing to checkpoint')
            loss['total'].backward(); torch.nn.utils.clip_grad_norm_(trainable,1.)
            optimizer.step(); batches+=1
            totals.update({k:float(v.detach()) for k,v in loss.items()})
            if batches%100==0:
                print(json.dumps({'arm':arm,'epoch':epoch,'batch':batches,'loss':totals['total']/batches,
                                  'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
        scheduler.step()
        report=evaluate(model,val,args); value=score(report)
        record={'epoch':epoch,'score':value,'train_loss':{k:v/batches for k,v in totals.items()},
                'summary':report['summary'],'elapsed_s':time.monotonic()-start}
        with (directory/'history.jsonl').open('a') as f:
            f.write(json.dumps(record,allow_nan=False)+'\n')
        if value>best:
            best=value
            torch.save({'model':model.state_dict(),'arm':arm,'epoch':epoch,'score':value,
                        'base_checkpoint':str(args.checkpoint),'protocol':protocol},directory/'best.pth')
            save_json(directory/'best_validation.json',report)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'epoch':epoch,'arm':arm,'base_checkpoint':str(args.checkpoint),'protocol':protocol},directory/'last.pth')
        print(json.dumps({'arm':arm,'epoch':epoch,'score':value,'best':best,
                          'oracle':report['summary']['all']['oracle_recall'],
                          'multilayer_oracle':report['summary']['multilayer']['oracle_recall'],
                          'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    save_json(directory/'complete.json',{'best_score':best,'elapsed_s':time.monotonic()-start})
    del model,optimizer; torch.cuda.empty_cache()


def choose_operating_point(report, baseline):
    """Validation-only operating threshold; not probability calibration."""
    reference=baseline['summary']['all']['selected']['0.5']
    qualified=[]
    for threshold in THRESHOLDS:
        row=report['summary']['all']['selected'][str(threshold)]
        if row['depth_mae_mm'] is None:
            continue
        if (row['depth_mae_mm']<=reference['depth_mae_mm']*1.05
            and row['depth_abs_rel']<=max(.03,reference['depth_abs_rel'])
            and row['valid_pixel_coverage']>=reference['valid_pixel_coverage']-.02
            and row['pixel_tolerance_pass']>=reference['pixel_tolerance_pass']-.01
            and row['non_target_output_fraction']<=reference['non_target_output_fraction']+.02):
            qualified.append((row['within_tolerance_coverage'],threshold))
    return max(qualified)[1] if qualified else None


def select_for_test(args):
    if (args.run_dir/'baseline_test.json').exists() or (args.run_dir/'frozen_test_selection.json').exists():
        raise RuntimeError('Selection is already frozen or test evaluation has begun; use a new experiment directory')
    baseline=json.loads((args.run_dir/'baseline_validation.json').read_text())
    decisions={}
    for arm in ARMS:
        report=json.loads((args.run_dir/arm/'best_validation.json').read_text())
        a=report['summary']['all']; b=baseline['summary']['all']
        m=report['summary']['multilayer']; bm=baseline['summary']['multilayer']
        gates={'oracle_gain_at_least_1pp':a['oracle_recall']>=b['oracle_recall']+.01,
               'multilayer_oracle_improved':m['oracle_recall']>bm['oracle_recall'],
               'all_layer_recall_loss_at_most_5pp':a['layer_tolerance_recall']>=b['layer_tolerance_recall']-.05}
        # This gate authorizes research testing, not deployment or final selection.
        decisions[arm]={'gates':gates,'eligible_for_test':all(gates.values()),
                        'checkpoint_sha256':sha(args.run_dir/arm/'best.pth'),
                        'score':score(report),
                        'validation_operating_threshold':choose_operating_point(report,baseline)}
    save_json(args.run_dir/'frozen_test_selection.json',{'arms':decisions,'rule':'frozen before test inference',
                                                        'baseline_operating_threshold':choose_operating_point(baseline,baseline),
                                                        'operating_rule':'maximize valid-target tolerance coverage subject to baseline MAE*1.05, AbsRel<=max(.03,baseline), coverage>=baseline-.02, pass>=baseline-.01 and off-target output<=baseline+.02',
                                                        'production_promoted':False})
    print(json.dumps(decisions),flush=True)


@torch.inference_mode()
def benchmark(model,args,d):
    x=d[0][0][None].cuda(); durations=[]; model.eval()
    for i in range(35):
        torch.cuda.synchronize(); start=time.perf_counter()
        p=model(x)
        select_liquid_interface(p,p['depth_m'],relative_tolerance=.008,confidence_threshold=.5)
        torch.cuda.synchronize()
        if i>=5:durations.append((time.perf_counter()-start)*1000)
    return {'p50_ms':float(np.median(durations)),'p95_ms':float(np.percentile(durations,95)),
            'max_ms':max(durations),'warmup':5,'repetitions':30,
            'scope':'resident GPU model and selector; excludes acquisition, host preprocessing, geometry and UI'}


def run_test(args):
    frozen=json.loads((args.run_dir/'frozen_test_selection.json').read_text())
    d=dataset(args,'test')
    base,_=load_model(args.checkpoint)
    path=args.run_dir/'baseline_test.json'
    if path.exists():raise FileExistsError(path)
    baseline=evaluate(base,d,args); baseline['latency']=benchmark(base,args,d)
    save_json(path,baseline)
    save_json(args.run_dir/'baseline_stress_test.json',evaluate(base,d,args,stress=True))
    del base; torch.cuda.empty_cache()
    for arm,decision in frozen['arms'].items():
        if not decision['eligible_for_test']:continue
        checkpoint=args.run_dir/arm/'best.pth'
        if sha(checkpoint)!=decision['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint changed')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['protocol']['checkpoint_sha256']!=sha(args.checkpoint) or state['protocol']['manifest_sha256']!=sha(args.manifest):
            raise RuntimeError('Base checkpoint or manifest differs from frozen training protocol')
        m,_=build(args,arm)
        m.load_state_dict(state['model'],strict=True)
        report=evaluate(m,d,args); report['latency']=benchmark(m,args,d)
        save_json(args.run_dir/arm/'test.json',report)
        save_json(args.run_dir/arm/'stress_test.json',evaluate(m,d,args,stress=True))
        print(json.dumps({'tested':arm,'oracle':report['summary']['all']['oracle_recall'],
                          'multilayer_oracle':report['summary']['multilayer']['oracle_recall']}),flush=True)
        del m; torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--phase',choices=['train','select','test'],default='train')
    p.add_argument('--arms',default=','.join(ARMS))
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--samples-per-epoch',type=int,default=4096)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--learning-rate',type=float,default=2e-4)
    p.add_argument('--seed',type=int,default=20260915)
    args=p.parse_args()
    torch.set_num_threads(4)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.phase=='train':
        train,val=dataset(args,'train'),dataset(args,'val')
        if not (args.run_dir/'baseline_validation.json').exists():
            base,_=load_model(args.checkpoint)
            save_json(args.run_dir/'baseline_validation.json',evaluate(base,val,args))
            del base; torch.cuda.empty_cache()
        for arm in args.arms.split(','):
            if arm not in ARMS:raise ValueError(arm)
            train_one(args,arm,train,val)
    elif args.phase=='select':select_for_test(args)
    else:run_test(args)


if __name__=='__main__':main()
