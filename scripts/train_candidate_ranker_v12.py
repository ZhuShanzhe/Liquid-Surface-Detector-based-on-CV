#!/usr/bin/env python3
"""Frozen V11 candidates -> V12 ranking, disjoint calibration, locked evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from evaluate_layer_voting import load_model,pixel_stats,summarize
from train_liquid_candidates_v11 import dataset,weights_for,sha,save_json
from liquid_depth.models.liquid_candidate import LiquidCandidateNet,corrupt_depth_prior
from liquid_depth.models.candidate_ranker import CandidateRanker,ranking_loss,MonotoneCalibrator
from liquid_depth.models.layered import select_liquid_interface

THRESHOLDS=(.3,.4,.5,.6,.7,.8,.9,.95)
ARMS=('rank_only','geometry')


class RayDataset(torch.utils.data.Dataset):
    def __init__(self,d):
        self.d=d; self.rows=d.rows; self.root=d.root; self.rays=[]
        v,u=np.mgrid[:180,:320]
        for row in self.rows:
            m=json.loads((d.root/row['metadata_path']).read_text())
            k=np.array(m['camera_intrinsics'],dtype=np.float32)
            sx=320/m['width']; sy=180/m['height']
            # Pixel-center resize convention. No camera extrinsic or level is read.
            fx=k[0,0]*sx; fy=k[1,1]*sy; cx=(k[0,2]+.5)*sx-.5; cy=(k[1,2]+.5)*sy-.5
            ray=np.stack([(u-cx)/fx,-(v-cy)/fy,-np.ones_like(u)],0).astype(np.float32)
            self.rays.append(torch.from_numpy(ray))
    def __len__(self):return len(self.d)
    def __getitem__(self,i):
        x,y=self.d[i]
        return x,y,self.rays[i]


def get_data(args,split,fold=None,manifest=None):
    opts=argparse.Namespace(manifest=manifest or args.manifest)
    d=dataset(opts,split)
    if fold is not None:
        d.rows=[r for r in d.rows if int(hashlib.sha256(r['sequence_id'].encode()).hexdigest()[:8],16)%3==fold]
    if not d.rows:raise ValueError('Empty requested dataset')
    return RayDataset(d)


def load_candidates(args):
    b,_=load_model(args.base)
    m=LiquidCandidateNet(b,dedicated=False).cuda().eval().requires_grad_(False)
    state=torch.load(args.candidates,map_location='cpu',weights_only=False)
    if state['arm']!='control' or state['protocol']['checkpoint_sha256']!=sha(args.base):
        raise ValueError('Expected the frozen V11 control with its exact base checkpoint')
    m.load_state_dict(state['model'],strict=True)
    return m


def candidate_prediction(candidate,x,args):
    if args.feature_context and isinstance(candidate,LiquidCandidateNet):
        features,p=candidate.frozen_features(x)
        p.update(candidate.ray_head(features,p['depth_m']))
        p['rank_features']=features
        return p
    return candidate(x)


def loader(d,args,training=False):
    if training:
        weights,_=weights_for(d)
        sampler=WeightedRandomSampler(weights,args.samples,replacement=True,
                                      generator=torch.Generator().manual_seed(args.seed))
        return DataLoader(d,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,pin_memory=True,persistent_workers=True)
    return DataLoader(d,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)


def biased(x):
    x=x.clone(); raw=.1*torch.exp(x[:,3:4]*math.log(100))
    x[:,3:4]=torch.where(x[:,4:5]>.5,torch.log((raw*1.05).clamp(.1,10)/.1)/math.log(100),x[:,3:4])
    return x


def summary(frames):
    n=sum(f['n'] for f in frames)
    count=np.sum([f['cal_count'] for f in frames],0)
    correct=np.sum([f['cal_good'] for f in frames],0)
    prob=np.sum([f['cal_prob'] for f in frames],0)
    occupied=count>0
    return {'frames':len(frames),'valid_pixels':n,
            'oracle_recall':sum(f['oracle_good'] for f in frames)/max(n,1),
            'top1_recall_all_valid':sum(f['top1_good'] for f in frames)/max(n,1),
            'masked_top1_recall_all_valid':sum(f['masked_top1_good'] for f in frames)/max(n,1),
            'mask_recall':sum(f['mask_good'] for f in frames)/max(n,1),
            'confidence_ece':float(np.abs(correct[occupied]-prob[occupied]).sum()/max(count.sum(),1)),
            'confidence_brier':sum(f['brier_sum'] for f in frames)/max(count.sum(),1),
            'none_detection_recall':sum(f['none_detected'] for f in frames)/max(sum(f['none_n'] for f in frames),1),
            'none_false_reject_rate':sum(f['none_false'] for f in frames)/max(sum(f['has_n'] for f in frames),1),
            'selected':{str(t):summarize([f['selected'][str(t)] for f in frames]) for t in THRESHOLDS},
            'accepted_when_no_good_candidate':{str(t):sum(f['none_accepted'][str(t)] for f in frames)/max(sum(f['none_n'] for f in frames),1) for t in THRESHOLDS}}


@torch.inference_mode()
def evaluate(candidate,ranker,d,args,calibrator=None,stress=False):
    candidate.eval()
    if ranker is not None:ranker.eval()
    frames=[]; offset=0
    for x,y,rays in loader(d,args):
        x=x.cuda(); rays=rays.cuda(); y={k:v.cuda() for k,v in y.items()}
        if stress:x=biased(x)
        p=candidate_prediction(candidate,x,args)
        if ranker is not None:
            q=ranker(x,p,rays); depth=q['depth_m']; raw=q['raw_confidence']
            confidence=calibrator(raw) if calibrator else raw
            support=torch.ones_like(raw,dtype=torch.bool); none=q['none_logits'].sigmoid()>=.5
        else:
            q=select_liquid_interface(p,p['depth_m'],relative_tolerance=.008,confidence_threshold=0.)
            depth=q['depth_m']; support=q['accepted']; none=~support
            confidence=calibrator(q['confidence'])*support if calibrator else q['confidence']
        valid=y['valid']>0; mask=p['mask_logits'].sigmoid()>=.5
        tol=(y['depth_m']*.02).clamp_min(.005)
        top=((depth-y['depth_m']).abs()<=tol)&valid
        oracle=((p['layer_depths_m']-y['depth_m']).abs()<=tol).any(1,keepdim=True)&valid
        no_good=valid&~oracle
        for i in range(x.shape[0]):
            row=d.rows[offset+i]; emitted_mask=mask[i]
            c=confidence[i][emitted_mask]; label=top[i][emitted_mask].float()
            bins=(c*10).long().clamp(0,9)
            item={'sequence_id':row['sequence_id'],'rgb_path':row['rgb_path'],'scenario':row['scenario'],
                  'n':int(valid[i].sum()),'oracle_good':int(oracle[i].sum()),
                  'top1_good':int(top[i].sum()),'masked_top1_good':int((top[i]&mask[i]).sum()),
                  'mask_good':int((valid[i]&mask[i]).sum()),
                  'none_n':int(no_good[i].sum()),'has_n':int(oracle[i].sum()),
                  'none_detected':int((no_good[i]&none[i]).sum()),'none_false':int((oracle[i]&none[i]).sum()),
                  'cal_count':torch.bincount(bins,minlength=10).cpu().tolist(),
                  'cal_good':torch.bincount(bins,weights=label,minlength=10).cpu().tolist(),
                  'cal_prob':torch.bincount(bins,weights=c,minlength=10).cpu().tolist(),
                  'brier_sum':float(((c-label)**2).sum()),'selected':{},'none_accepted':{}}
            for t in THRESHOLDS:
                emitted=mask[i]&support[i]&(confidence[i]>=t)
                item['selected'][str(t)]=pixel_stats(depth[i],emitted,y['depth_m'][i],valid[i])
                item['none_accepted'][str(t)]=int((no_good[i]&emitted).sum())
            frames.append(item)
        offset+=x.shape[0]
    return {'summary':{'all':summary(frames),**{s:summary([f for f in frames if f['scenario']==s]) for s in sorted({f['scenario'] for f in frames})}},
            'frames':frames,'stress':stress,'scope':'camera axial depth, not bottom-referenced liquid height'}


@torch.inference_mode()
def calibrate(candidate,ranker,d,args):
    if ranker is not None:ranker.eval()
    count=np.zeros(32); positive=np.zeros(32)
    for x,y,rays in loader(d,args):
        x=x.cuda(); rays=rays.cuda(); y={k:v.cuda() for k,v in y.items()}
        p=candidate_prediction(candidate,x,args)
        q=ranker(x,p,rays) if ranker is not None else select_liquid_interface(p,p['depth_m'],relative_tolerance=.008)
        mask=p['mask_logits'].sigmoid()>=.5
        if ranker is None:mask=mask&q['accepted']
        good=((q['depth_m']-y['depth_m']).abs()<=(y['depth_m']*.02).clamp_min(.005))&(y['valid']>0)
        score=q['raw_confidence'] if ranker is not None else q['confidence']
        bins=(score[mask]*32).long().clamp(0,31)
        count+=torch.bincount(bins,minlength=32).cpu().numpy()
        positive+=torch.bincount(bins,weights=good[mask].float(),minlength=32).cpu().numpy()
    c=MonotoneCalibrator().fit_histogram(count,positive)
    return c,{'values':c.values.tolist(),'count':count.tolist(),'positive':positive.tolist(),
              'scope':'selected-point correctness within predicted liquid mask; clean calibration fold only'}


def operating_point(report,baseline):
    ref=baseline['summary']['all']['selected']['0.5']; choices=[]
    for t in THRESHOLDS:
        a=report['summary']['all']['selected'][str(t)]
        if a['depth_mae_mm'] is None:continue
        if (a['depth_mae_mm']<=ref['depth_mae_mm']*1.05 and a['depth_abs_rel']<=max(.03,ref['depth_abs_rel'])
            and a['pixel_tolerance_pass']>=max(.5,ref['pixel_tolerance_pass']-.01)
            and a['valid_pixel_coverage']>=ref['valid_pixel_coverage']-.02
            and a['non_target_output_fraction']<=ref['non_target_output_fraction']+.02):
            choices.append((a['within_tolerance_coverage'],t))
    return max(choices)[1] if choices else None


def freeze_calibration_baseline(args):
    path=args.run_dir/'frozen_calibration_baseline.json'
    if path.exists() or (args.run_dir/'test_started.json').exists():raise FileExistsError('Baseline calibration already frozen or tested')
    frozen=json.loads((args.run_dir/'frozen_selection.json').read_text())
    for k,p in [('base',args.base),('candidates',args.candidates),('manifest',args.manifest)]:
        if sha(p)!=frozen['protocol'][k+'_sha256']:raise RuntimeError('Frozen source changed')
    m=load_candidates(args);cal=get_data(args,'val',1);selection=get_data(args,'val',2)
    c,info=calibrate(m,None,cal,args)
    report=evaluate(m,None,selection,args,c)
    save_json(args.run_dir/'calibration_only_selection.json',report)
    baseline=json.loads((args.run_dir/'baseline_selection.json').read_text())
    save_json(path,{'calibration':info,'threshold':operating_point(report,baseline),
                    'frozen_selection_sha256':sha(args.run_dir/'frozen_selection.json'),
                    'scope':'old V11 selection and prior support retained; only confidence mapping changes'})
    print(json.dumps({'calibration_only_threshold':operating_point(report,baseline)}),flush=True)


def train(args):
    if args.run_dir.exists():raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    d=get_data(args,'train'); stop=get_data(args,'val',0); cal=get_data(args,'val',1); select=get_data(args,'val',2)
    sets=[{r['sequence_id'] for r in x.rows} for x in (d,stop,cal,select)]
    if any(sets[i]&sets[j] for i in range(4) for j in range(i)):raise ValueError('Sequence leakage')
    protocol={'seed':args.seed,'epochs':args.epochs,'samples_per_epoch':args.samples,
              'base_sha256':sha(args.base),'candidates_sha256':sha(args.candidates),'manifest_sha256':sha(args.manifest),
              'train_frames':len(d),'stop_frames':len(stop),'calibration_frames':len(cal),'selection_frames':len(select),
              'sequence_ids':{name:sorted(ids) for name,ids in zip(('train','stop','calibration','selection'),sets)},
              'candidate_depths_frozen':True,'training_depth_bias_probability':.25,'feature_context':args.feature_context,
              'tolerance':'max(.005m,.02*axial camera depth)','threshold_grid':THRESHOLDS,
              'fresh_test_manifest':str(args.fresh_manifest),'fresh_test_sha256':sha(args.fresh_manifest),
              'production_promoted':False}
    save_json(args.run_dir/'protocol.json',protocol)
    candidate=load_candidates(args)
    base,_=load_model(args.base)
    baseline=evaluate(base,None,select,args); save_json(args.run_dir/'baseline_selection.json',baseline)
    del base
    decisions={}
    for arm in ARMS:
        folder=args.run_dir/arm; folder.mkdir()
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        model=CandidateRanker(geometry=arm=='geometry',context_features=24 if args.feature_context else 0).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs)
        corruption=torch.Generator(device='cuda').manual_seed(args.seed+17)
        data=loader(d,args,True); best=-1.; start=time.monotonic()
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; batches=0
            for x,y,rays in data:
                x=x.cuda(); rays=rays.cuda(); y={k:v.cuda() for k,v in y.items()}
                x=corrupt_depth_prior(x,corruption,probability=.25)
                with torch.no_grad():p=candidate_prediction(candidate,x,args)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):q=model(x,p,rays)
                q={k:v.float() if v.is_floating_point() else v for k,v in q.items()}
                loss=ranking_loss(q,p['layer_depths_m'],y)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
                total+=float(loss.detach());batches+=1
                if batches%128==0:print(json.dumps({'arm':arm,'epoch':epoch,'batch':batches,'loss':total/batches}),flush=True)
            schedule.step(); report=evaluate(candidate,model,stop,args)
            value=float(np.mean([a['top1_recall_all_valid'] for s,a in report['summary'].items() if s!='all']))
            record={'epoch':epoch,'score':value,'loss':total/batches,'summary':report['summary'],'elapsed_s':time.monotonic()-start}
            with (folder/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if value>best:
                best=value;torch.save({'model':model.state_dict(),'geometry':model.geometry,'context_features':model.context_features,'epoch':epoch,'protocol':protocol},folder/'best.pth')
                save_json(folder/'best_stop.json',report)
            print(json.dumps({'arm':arm,'epoch':epoch,'score':value,'best':best,'elapsed_s':time.monotonic()-start}),flush=True)
        state=torch.load(folder/'best.pth',map_location='cpu',weights_only=False);model.load_state_dict(state['model'])
        c,cal_info=calibrate(candidate,model,cal,args);save_json(folder/'calibration.json',cal_info)
        selected=evaluate(candidate,model,select,args,c);save_json(folder/'selection.json',selected)
        raw=evaluate(candidate,model,select,args);save_json(folder/'selection_uncalibrated.json',raw)
        decisions[arm]={'threshold':operating_point(selected,baseline),'checkpoint_sha256':sha(folder/'best.pth'),
                        'calibration_sha256':sha(folder/'calibration.json'),'best_epoch':state['epoch']}
        save_json(folder/'complete.json',{'elapsed_s':time.monotonic()-start,**decisions[arm]})
        del model,opt;torch.cuda.empty_cache()
    save_json(args.run_dir/'frozen_selection.json',{'arms':decisions,'protocol':protocol,
                                                  'rule':'validation-only operating points; test does not authorize production promotion',
                                                  'production_promoted':False})
    print(json.dumps({'frozen':decisions}),flush=True)


@torch.inference_mode()
def benchmark(candidate,ranker,d,args,c=None):
    x,y,rays=d[0]; x=x[None].cuda();rays=rays[None].cuda();times=[]
    for i in range(35):
        torch.cuda.synchronize();start=time.perf_counter();p=candidate_prediction(candidate,x,args)
        if ranker is None:select_liquid_interface(p,p['depth_m'],relative_tolerance=.008)
        else:
            q=ranker(x,p,rays)
            if c is not None:c(q['raw_confidence'])
        torch.cuda.synchronize()
        if i>=5:times.append((time.perf_counter()-start)*1000)
    return {'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95)),
            'scope':'GPU resident model, candidate ranking and calibration only; not end-to-end latency'}


def test(args):
    frozen=json.loads((args.run_dir/'frozen_selection.json').read_text()); protocol=frozen['protocol']
    if args.feature_context!=protocol.get('feature_context',False):raise ValueError('Feature context differs from frozen protocol')
    for k,p in [('base',args.base),('candidates',args.candidates),('manifest',args.manifest),('fresh_test',args.fresh_manifest)]:
        if sha(p)!=protocol[k+'_sha256']:raise RuntimeError('Frozen source changed: '+k)
    if (args.run_dir/'test_started.json').exists():raise FileExistsError('Test already started; do not retune this run')
    cbpath=args.run_dir/'frozen_calibration_baseline.json';cb=json.loads(cbpath.read_text())
    if cb['frozen_selection_sha256']!=sha(args.run_dir/'frozen_selection.json'):raise RuntimeError('Calibration baseline linked to another freeze')
    save_json(args.run_dir/'test_started.json',{'frozen_sha256':sha(args.run_dir/'frozen_selection.json'),'calibration_baseline_sha256':sha(cbpath)})
    candidate=load_candidates(args); base,_=load_model(args.base)
    for label,manifest in [('legacy',args.manifest),('fresh',args.fresh_manifest)]:
        d=get_data(args,'test',manifest=manifest)
        if label=='fresh':
            seen=set().union(*(set(v) for v in protocol['sequence_ids'].values()))
            if seen&{r['sequence_id'] for r in d.rows}:raise ValueError('Fresh holdout overlaps fitting sequences')
        for name,m in [('baseline',base),('control',candidate)]:
            for stress in (False,True):
                report=evaluate(m,None,d,args,stress=stress)
                if not stress:report['latency']=benchmark(m,None,d,args)
                save_json(args.run_dir/f'{label}_{name}_{"stress" if stress else "clean"}.json',report)
        calibrator=MonotoneCalibrator(cb['calibration']['values'])
        for stress in (False,True):
            report=evaluate(candidate,None,d,args,calibrator,stress)
            save_json(args.run_dir/f'{label}_calibration_only_{"stress" if stress else "clean"}.json',report)
        for arm,decision in frozen['arms'].items():
            folder=args.run_dir/arm
            if sha(folder/'best.pth')!=decision['checkpoint_sha256'] or sha(folder/'calibration.json')!=decision['calibration_sha256']:
                raise RuntimeError('Frozen arm changed')
            state=torch.load(folder/'best.pth',map_location='cpu',weights_only=False)
            ranker=CandidateRanker(state['geometry'],state.get('context_features',0)).cuda().eval();ranker.load_state_dict(state['model'])
            c=MonotoneCalibrator(json.loads((folder/'calibration.json').read_text())['values'])
            for stress in (False,True):
                report=evaluate(candidate,ranker,d,args,c,stress)
                if not stress:report['latency']=benchmark(candidate,ranker,d,args,c)
                save_json(folder/f'{label}_{"stress" if stress else "clean"}.json',report)
            print(json.dumps({'tested':arm,'dataset':label}),flush=True)
            del ranker
    save_json(args.run_dir/'test_complete.json',{'complete':True,'production_promoted':False})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True);p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--fresh-manifest',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--phase',choices=['train','calibrate-baseline','test'],default='train')
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--samples',type=int,default=4096)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--feature-context',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    if args.phase=='train':train(args)
    elif args.phase=='calibrate-baseline':freeze_calibration_baseline(args)
    else:test(args)


if __name__=='__main__':main()
