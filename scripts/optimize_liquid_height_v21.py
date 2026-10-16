#!/usr/bin/env python3
"""First height-aware experiment: bounded learned residual head on frozen V17.

Train, epoch-selection and audit folds are separated by sequence ID. Test labels
are never used for model/threshold selection. Head does not change acceptance.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from liquid_depth.liquid_height_metrics import FEATURE_NAMES, summarize_frames
from liquid_depth.models.liquid_height_head import HeightResidualHead


def load(path):return json.loads(Path(path).read_text())
def fold(r):return int(hashlib.sha256(r['sequence_id'].encode()).hexdigest()[:8],16)%3
def eligible(rows):return [r for r in rows if r['accepted'] and r['evaluable']]
def tensors(rows,mean,scale):
    x=torch.tensor(np.array([r['features'] for r in rows]),dtype=torch.float32)
    x=(x-mean)/scale
    h=torch.tensor([r['predicted_height_m'] for r in rows])
    truth=torch.tensor([r['true_height_m'] for r in rows])
    d=torch.tensor([r['bottom_distance_m'] for r in rows])
    return x,h,truth,d


def loss_values(head,data):
    x,h,truth,d=data
    error=(h+head(x)*d-truth).abs()
    tolerance=torch.clamp(.02*truth,min=.005)
    normalized=error/tolerance
    # Equal-frame normalized height error and worst-20% tail, no z denominator.
    huber=torch.where(normalized<1,.5*normalized.square(),normalized-.5)
    return huber.mean()+.2*torch.topk(huber,max(1,len(huber)//5)).values.mean()+.01*head(x).square().mean()


def train(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    training=load(args.train);validation=load(args.val)
    assert training['feature_names']==validation['feature_names']==list(FEATURE_NAMES)
    tr=eligible(training['records'])
    selection_rows=[r for r in validation['records'] if fold(r)==0]
    audit=[r for r in validation['records'] if fold(r)!=0]
    va=eligible(selection_rows)
    tids={r['sequence_id'] for r in training['records']}
    assert not tids.intersection(r['sequence_id'] for r in validation['records'])
    if len(tr)<64 or len(va)<16:raise ValueError('Insufficient independent train/validation support')
    torch.set_num_threads(2);torch.manual_seed(20261011)
    features=np.array([r['features'] for r in tr],np.float32)
    mean=torch.tensor(features.mean(0));scale=torch.tensor(features.std(0).clip(.02))
    td=tensors(tr,mean,scale);vd=tensors(va,mean,scale)
    head=HeightResidualHead();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    with torch.no_grad():best=float(loss_values(head,vd))
    initial=best;best_state=copy.deepcopy(head.state_dict());best_epoch=0;history=[]
    for epoch in range(1,301):
        head.train();opt.zero_grad();loss=loss_values(head,td)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
        head.eval()
        with torch.no_grad():value=float(loss_values(head,vd))
        if value<best:best=value;best_epoch=epoch;best_state=copy.deepcopy(head.state_dict())
        if epoch%25==0:history.append(dict(epoch=epoch,train=float(loss),validation=value,best_epoch=best_epoch))
    head.load_state_dict(best_state)
    frozen=dict(feature_names=list(FEATURE_NAMES),selection_fold=0,audit_folds=[1,2],
        best_epoch=best_epoch,initial_validation_loss=initial,best_validation_loss=best,
        training_frames=len(tr),selection_frames=len(va),audit_frames=len(audit),
        source_checkpoint_sha256=training['checkpoint_sha256'],
        source_manifest_sha256=training['manifest_sha256'],validation_manifest_sha256=validation['manifest_sha256'],
        input_contract='observable feature whitelist only; no material/true-height/GT-mask input',
        acceptance_policy='unchanged from baseline; no test-derived threshold',
        deployment_enabled=False,history=history)
    torch.save(dict(model=head.state_dict(),mean=mean,scale=scale,**frozen),out/'height-head.pth')
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    for name,rows in [('selection',selection_rows),('audit',audit)]:
        result=apply(head,mean,scale,rows)
        (out/f'{name}.json').write_text(json.dumps(dict(baseline=summarize_frames(rows),candidate=summarize_frames(result)),indent=2))
    print('FROZEN',json.dumps(frozen),flush=True)


@torch.inference_mode()
def apply(head,mean,scale,rows):
    records=copy.deepcopy(rows);accepted=eligible(records)
    if not accepted:return records
    x,h,truth,d=tensors(accepted,mean,scale);corrected=h+head(x)*d
    for r,v in zip(accepted,corrected.tolist()):
        delta=v-r['predicted_height_m']
        r['height_head_delta_m']=delta
        if not np.isfinite(v) or v<=0:
            r.update(accepted=False,evaluable=False,reasons=['nonpositive_calibrated_height'])
            continue
        error=v-r['true_height_m'];r.update(predicted_height_m=v,signed_error_m=error,
            absolute_error_m=abs(error),abs_rel=abs(error)/r['true_height_m'],
            within_tolerance=bool(abs(error)<=r['tolerance_m']))
        # A scalar head cannot improve surface shape: translated extrema only.
        r['extrema_error_m']={k:abs(r['prediction']['statistics'][k]+delta-r['reference'][k])
                              for k in ('min_m','max_m','p05_m','p95_m')}
    return records


def evaluate(args):
    from evaluate_liquid_height_v21 import groups
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    data=load(args.input);state=torch.load(args.head,map_location='cpu',weights_only=False)
    if state['feature_names']!=data['feature_names']:raise ValueError('Feature mismatch')
    if state['source_checkpoint_sha256']!=data['checkpoint_sha256']:raise ValueError('Base model mismatch')
    head=HeightResidualHead().eval();head.load_state_dict(state['model'])
    records=apply(head,state['mean'],state['scale'],data['records'])
    result={k:v for k,v in data.items() if k not in ('records','overall','by_scenario','by_camera_range','by_liquid_depth','by_sensor')}
    result.update(backend='v21_height_residual_head',head_sha256=hashlib.sha256(Path(args.head).read_bytes()).hexdigest(),
        baseline_overall=data['overall'],overall=summarize_frames(records),by_scenario=groups(records,'scenario'),
        by_camera_range=groups(records,'range_band'),by_liquid_depth=groups(records,'height_band'),
        by_sensor=groups(records,'sensor'),records=records)
    out.write_text(json.dumps(result,indent=2,allow_nan=False));print('COMPLETE_HEAD',json.dumps(result['overall']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate'])
    p.add_argument('--train');p.add_argument('--val');p.add_argument('--input');p.add_argument('--head')
    p.add_argument('--output',required=True);args=p.parse_args()
    train(args) if args.stage=='train' else evaluate(args)
