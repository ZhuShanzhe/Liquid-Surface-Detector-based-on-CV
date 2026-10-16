#!/usr/bin/env python3
"""Height-aware fine-tuning, frozen validation selection, no test-set selection."""
import argparse,csv,hashlib,json,random,subprocess,sys,time
from collections import Counter
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,WeightedRandomSampler
import train_precision_v17 as precision
import train_observable_v16 as v16
from evaluate_liquid_height_v21 import metadata_from
from liquid_depth.liquid_height_metrics import HeightCalibration,reference_statistics
from liquid_depth.models.height_objective import liquid_height_loss
from liquid_depth.models.precision_v17 import fit_surface

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=6)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.manual_seed(20261011031);np.random.seed(20261011);random.seed(20261011)
base=precision.OUT/'r4/best.pth';pipe=precision.make(4,base,False);depth_model=pipe.depth_model
dataset=v16.Observations(v16.DEV,'train',mode='clean')
calibrations=[];references=[];strata=[]
for row in dataset.rows:
    metadata=metadata_from(row);cal=HeightCalibration.from_simulation(metadata)
    truth=reference_statistics(metadata,cal)['mean_m']
    calibrations.append(cal);references.append(truth)
    distance=cal.camera_to_world_cv[2,3]-cal.bottom_z_m-truth
    strata.append((row['scenario'],int(np.searchsorted([.3,1,3],distance)),int(np.searchsorted([.1,.3,1],truth))))
counts=Counter(strata);weights=[1/counts[s] for s in strata]
sampler=WeightedRandomSampler(weights,768,replacement=True,generator=torch.Generator().manual_seed(20261011031))
loader=DataLoader(dataset,batch_size=8,num_workers=4,pin_memory=True,sampler=sampler)
opt=torch.optim.AdamW([{'params':depth_model.network.parameters(),'lr':3e-6},
                      {'params':depth_model.correction.parameters(),'lr':1e-5}],weight_decay=1e-4)
protocol=dict(seed=20261011031,epochs=a.epochs,samples_per_epoch=768,
    sampling='material x camera-range x true-liquid-depth balanced training only',
    loss='0.5 height-tolerance-normalized vertical pixel Huber + 0.5 differentiable area-mean-height Huber + 0.2 worst20% frame loss',
    tolerance='max(0.005m,0.02*true liquid depth)',source_checkpoint_sha256=v16.digest(base),
    selection='development validation fold0 only; no test use; per-material coverage and effective-pass guards',
    unmodified_locator_sha256=v16.digest(v16.OUT/'locator.pth'),
    train_strata={str(k):v for k,v in counts.items()},production_enabled=False)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))

def evaluate(epoch,path):
    dest=out/f'val-epoch{epoch}.json'
    with (out/f'val-epoch{epoch}.stdout').open('w') as stdout,(out/f'val-epoch{epoch}.stderr').open('w') as stderr:
        subprocess.run([sys.executable,'scripts/evaluate_liquid_height_v21.py','--backend','v17',
            '--checkpoint',str(path),'--manifest',str(v16.DEV),'--split','val','--fold','0',
            '--output',str(dest),'--label','height_aware_v21'],stdout=stdout,stderr=stderr,check=True)
    r=json.loads(dest.read_text());errors=np.array([x['absolute_error_m']/x['tolerance_m'] if x['evaluable'] else 20 for x in r['records']])
    score=float(errors.mean()+.2*np.sort(errors)[-max(1,len(errors)//5):].mean())
    return score,r

best,baseline=evaluate(0,base);best_epoch=0;history=[]
torch.save({'depth':depth_model.state_dict(),'epoch':0,'metric_contract_version':3},out/'best.pth')
started=time.time()
for epoch in range(1,a.epochs+1):
    depth_model.train();v16.frozen_bn(depth_model);losses=[]
    for batch in loader:
        b=v16.gpu(batch);indices=batch['index'].tolist()
        rgb,raw=b['rgb'],b['raw']
        # Same observation-noise family as V17, labels unchanged; no test-derived corruption.
        u=np.random.random()
        if u<.1:raw=torch.where(raw>0,raw*(1+np.random.uniform(-.03,.03)),raw)
        elif u<.2:raw=torch.where(torch.rand_like(raw)<np.random.uniform(.1,.6),0,raw)
        prediction=pipe(rgb,raw)
        restored=fit_surface(prediction['unclipped_depth_m'],prediction['mask_probability'])
        valid=b['valid']*prediction['has_depth_anchor'][:,None,None,None]
        if not valid.any():continue
        loss=liquid_height_loss(restored,b['truth'],valid,prediction['mask_probability'],
                               [calibrations[i] for i in indices],[references[i] for i in indices])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite height loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(depth_model.parameters(),1);opt.step()
        losses.append(float(loss.detach()))
    checkpoint=out/f'epoch{epoch}.pth';torch.save({'depth':depth_model.state_dict(),'epoch':epoch,'metric_contract_version':3},checkpoint)
    score,report=evaluate(epoch,checkpoint)
    qualified=all(report['by_scenario'][s]['output_coverage']>=baseline['by_scenario'][s]['output_coverage']-.02 and
        report['by_scenario'][s]['effective_pass_coverage']>=baseline['by_scenario'][s]['effective_pass_coverage']-.02
        for s in ('transparent','translucent'))
    if qualified and score<best:
        best=score;best_epoch=epoch
        torch.save({'depth':depth_model.state_dict(),'epoch':epoch,'metric_contract_version':3},out/'best.pth')
    rec=dict(epoch=epoch,training_loss=float(np.mean(losses)),validation_score=score,
             guard_pass=qualified,best_epoch=best_epoch,elapsed_s=time.time()-started,by_scenario=report['by_scenario'])
    history.append(rec);(out/'history.json').write_text(json.dumps(history,indent=2));print('HEIGHT_EPOCH',json.dumps(rec),flush=True)
frozen=dict(**protocol,selected_epoch=best_epoch,checkpoint_sha256=v16.digest(out/'best.pth'),
            validation_score=best,selected_before_test=True,elapsed_s=time.time()-started)
(out/'frozen.json').write_text(json.dumps(frozen,indent=2));print('HEIGHT_FROZEN',json.dumps(frozen),flush=True)
