"""Learn candidate trust on train installations and choose optional-use gate on validation."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from torch import nn

from liquid_depth.adaptive_rgbd_meter import CandidateTrustNet, build_observed_candidates
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor, observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter


def collect(paths,root,seed,predictor):
    rows=[]
    for folder in paths:
        folder=root/Path(folder).name;case=int(folder.name)
        item=install(case,root,root/'dry',seed,max_annotation_attempts=6,source_slot_stride=1)
        base=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.5)
        rgb=cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw=np.load(frame['raw_path']);x=observed_features(rgb,raw,item['dry'],base.ordinary_container_roi,item['bundle'])
            output=predictor.predict(x);candidates,_=build_observed_candidates(base,raw,output)
            old=FrozenSymmetricGeometryMeter(base).measure(depth_m=raw)['liquid_depth_m']
            # Truth is used only for supervised candidate labels, after observed proposals.
            truth=item['meta']['truth']['liquid_height_m'];tolerance=max(.02,.02*truth)
            if old is None:continue
            for c in candidates:
                err=abs(c['height_m']-truth);old_err=abs(old-truth)
                rows.append({'case':case,'repeat':frame['repeat'],'scene':item['meta']['scenario'],
                    'features':c['features'].tolist(),'height_m':c['height_m'],'baseline_m':old,'true_m':truth,
                    'target':[float(err+.001<old_err),float(err<=tolerance)],'tolerance_m':tolerance,'source':c['source']})
        if case%30==0:print('CANDIDATE_FEATURES',case,len(rows),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.manual_seed(a.seed);np.random.seed(a.seed%(2**32))
    predictor=RGBDEvidencePredictor(a.model,'cuda' if torch.cuda.is_available() else 'cpu')
    splits=json.loads((a.model.parent/'split.json').read_text())
    train=collect(splits['train'],a.root,a.seed,predictor);val=collect(splits['validation'],a.root,a.seed,predictor)
    for name,rows in [('train',train),('validation',val)]:
        (a.output/f'{name}-candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    x=np.asarray([r['features'] for r in train],np.float32);y=np.asarray([r['target'] for r in train],np.float32)
    mean=x.mean(0);scale=np.maximum(x.std(0),.01);xt=torch.from_numpy(np.clip((x-mean)/scale,-20,20));yt=torch.from_numpy(y)
    model=CandidateTrustNet();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    for _ in range(200):
        opt.zero_grad();logits=model(xt);loss=nn.functional.binary_cross_entropy_with_logits(logits,yt);loss.backward();opt.step()
    model.eval();vx=np.asarray([r['features'] for r in val],np.float32)
    with torch.inference_mode():probs=model(torch.from_numpy(np.clip((vx-mean)/scale,-20,20))).sigmoid().numpy()
    grouped={}
    for r,t in zip(val,probs,strict=True):
        key=(r['case'],r['repeat']);value=(float(t[0]*t[1]),r,t)
        if key not in grouped or value[0]>grouped[key][0]:grouped[key]=value
    choices=[];threshold=1.01
    for t in [.5,.6,.7,.8,.85,.9,.93,.95,.97,.99]:
        used=[(r,pr) for _,r,pr in grouped.values() if pr[0]>=t and pr[1]>=.7]
        if not used:
            choices.append({'threshold':t,'used':0});continue
        delta=np.array([abs(r['height_m']-r['true_m'])-abs(r['baseline_m']-r['true_m']) for r,_ in used])
        harmed=(delta>.005)
        choice={'threshold':t,'used':len(used),'installations':len({r['case'] for r,_ in used}),
            'harm_over_5mm':int(harmed.sum()),'mean_error_change_mm':float(1000*delta.mean()),'max_error_change_mm':float(1000*delta.max())}
        choices.append(choice)
        if threshold>1 and choice['installations']>=10 and not harmed.any() and delta.mean()<0:threshold=t
    # If validation cannot establish a safe-use region, auto retains geometric liquid values.
    torch.save({'model':model.state_dict(),'mean':mean.tolist(),'scale':scale.tolist(),'threshold':threshold,
                'validation_policy':'at least ten installations; no >5mm observed harm; mean improvement; not a field guarantee'},a.output/'gate.pt')
    (a.output/'validation-policy.json').write_text(json.dumps({'threshold':threshold,'choices':choices,
        'gate_uses_test_truth':False,'retains_geometric_numeric_output_when_disabled':True},indent=2))
    print('CANDIDATE_TRAINING_COMPLETE',threshold,flush=True)


if __name__=='__main__':main()
