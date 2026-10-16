#!/usr/bin/env python3
"""Re-evaluate real liquid height; calibration and inference are GT-independent."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from collections import defaultdict
import cv2
import numpy as np
import torch
from liquid_depth.liquid_height_metrics import (HeightCalibration, estimate_height,
    reference_statistics, score_frame, summarize_frames, observable_features, FEATURE_NAMES)

ART = Path('/root/autodl-tmp/liquid-depth-artifacts')
WEIGHTS = {
    'v6': ART/'training/universal-liquid-v6-v3-balanced-full/best.pth',
    'v8': ART/'training/universal-liquid-v8-ordinary-sensor-anchor/best.pth',
    'v17': ART/'training/precision-v17-20260916/r4/best.pth',
    'v17_raw': ART/'training/precision-v17-20260916/r4/best.pth',
    'v16_plane': ART/'training/precision-v17-20260916/reference/best.pth',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_from(manifest, split, scenarios=None):
    root = Path(manifest).parent
    rows=[]
    for r in csv.DictReader(Path(manifest).open()):
        if r['split']!=split or (scenarios and r['scenario'] not in scenarios):continue
        r=dict(r)
        for k,v in r.items():
            if k.endswith('_path') and v:
                r[k]=str(Path(v) if Path(v).is_absolute() else root/v)
        rows.append(r)
    return rows


def metadata_from(row):
    path=Path(row.get('metadata_path') or Path(row['raw_depth_path']).parent/'provenance.json')
    obj=json.loads(path.read_text());return obj.get('scene',obj)


class Predictor:
    def __init__(self,backend):
        self.backend=backend
        if backend in ('v17','v16_plane','v17_raw'):
            import train_precision_v17 as train
            self.model=train.make(4, WEIGHTS[backend], True).eval()
        else:
            from evaluate_layer_voting import load_model
            self.model,self.state=load_model(WEIGHTS['v8' if backend=='raw' else backend])

    @torch.inference_mode()
    def __call__(self,rgb,raw):
        if self.backend in ('v17','v16_plane','v17_raw'):
            p=self.model(rgb,raw);return (raw if self.backend=='v17_raw' else p['depth_m']),p['mask_probability'],None
        color=(rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
        valid=torch.isfinite(raw)&(raw>0)
        log=torch.log(raw.clamp(.1,10)/.1)/np.log(100)
        inputs=torch.cat((color,torch.where(valid,log,0),valid.float()),1)
        p=self.model(inputs)
        return (raw if self.backend=='raw' else p['depth_m']),p['mask_logits'].sigmoid(),p['confidence']


def groups(records,key):
    grouped=defaultdict(list)
    for r in records:grouped[r[key]].append(r)
    return {k:summarize_frames(v) for k,v in sorted(grouped.items())}


def run(args):
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);cv2.setNumThreads(0)
    torch.manual_seed(20261011)
    rows=rows_from(args.manifest,args.split,args.scenarios)
    if args.fold is not None:
        rows=[r for r in rows if int(hashlib.sha256(r['sequence_id'].encode()).hexdigest()[:8],16)%3==args.fold]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('Empty evaluation')
    model=Predictor(args.backend);records=[];started=time.time()
    for offset in range(0,len(rows),args.batch_size):
        batch=rows[offset:offset+args.batch_size]
        rgbs=[cv2.cvtColor(cv2.imread(r['rgb_path']),cv2.COLOR_BGR2RGB).astype(np.float32)/255 for r in batch]
        raws=[np.load(r['raw_depth_path']).astype(np.float32) for r in batch]
        rgb=torch.from_numpy(np.stack(rgbs).transpose(0,3,1,2)).cuda()
        raw=torch.from_numpy(np.stack(raws)[:,None]).cuda()
        torch.cuda.synchronize();t=time.perf_counter()
        depth,mask,confidence=model(rgb,raw)
        torch.cuda.synchronize();model_ms=1000*(time.perf_counter()-t)/len(batch)
        depth,mask=depth[:,0].cpu().numpy(),mask[:,0].cpu().numpy()
        confidence=confidence[:,0].cpu().numpy() if confidence is not None else [None]*len(batch)
        for j,r in enumerate(batch):
            metadata=metadata_from(r);cal=HeightCalibration.from_simulation(metadata)
            # Only the prediction and declared calibration enter the estimator.
            prediction=estimate_height(depth[j],mask[j]>=.5,cal,confidence[j],args.confidence)
            raw_prediction=estimate_height(raws[j],mask[j]>=.5,cal)
            features,d=observable_features(prediction,raw_prediction,raws[j],mask[j]>=.5,cal)
            # Ground truth is accessed only after prediction has been computed.
            reference=reference_statistics(metadata,cal)
            result=score_frame(prediction,reference)
            truth_depth=np.load(r['target_depth_path']);truth_mask=np.load(r['mask_path'])>0
            ok=truth_mask&np.isfinite(truth_depth)&(truth_depth>0)
            # Offline geometry sanity check only; NEVER an input to the predictor/head.
            oracle=score_frame(estimate_height(truth_depth,truth_mask,cal),reference)
            z=float(np.median(truth_depth[ok])) if ok.any() else None
            range_band='unknown' if z is None else ('0.1-0.3' if z<.3 else '0.3-1' if z<1 else '1-3' if z<3 else '3-10')
            h=reference['mean_m'];height_band='0-0.1' if h<.1 else '0.1-0.3' if h<.3 else '0.3-1' if h<1 else '1-3' if h<3 else '3-10+'
            result.update(sequence_id=r['sequence_id'],scenario=r['scenario'],range_band=range_band,
                height_band=height_band,container=metadata['container_shape'],sensor=metadata['sensor_family'],
                model_ms_per_frame_batch_average=model_ms,reference=reference,prediction=prediction,
                raw_prediction=raw_prediction,features=features,bottom_distance_m=d,
                offline_truth_geometry_sanity=oracle,
                raw_valid_fraction_on_target=float(((raws[j]>0)&ok).sum()/max(ok.sum(),1)))
            records.append(result)
        if offset%80==0:print(json.dumps(dict(backend=args.backend,done=len(records),total=len(rows),elapsed_s=round(time.time()-started,1))),flush=True)
    report=dict(schema_version=1,metric_contract_version=3,measurand='area_mean_vertical_liquid_depth_above_container_bottom_m',
        tolerance=dict(absolute_floor_m=.005,relative_to='true_liquid_depth',relative_fraction=.02),
        calibration='ideal_simulation_bottom_pose_intrinsics_fixed_footprint',
        estimator='fixed_XY_cell_medians_four_neighbor_IDW_uniform_area_grid',
        gate='64 predicted points; >=4 spatial cells; >=12% footprint cells; positive mean depth',
        acceptance_uses_ground_truth=False,synthetic_only=True,hardware_qualified=False,
        backend=args.backend,experiment_label=args.label,checkpoint=str(WEIGHTS.get(args.backend,WEIGHTS['v8'])),
        checkpoint_sha256=digest(WEIGHTS.get(args.backend,WEIGHTS['v8'])),
        manifest=str(args.manifest),manifest_sha256=digest(args.manifest),split=args.split,
        confidence_threshold=args.confidence,feature_names=FEATURE_NAMES,
        model_time_is_batched_not_end_to_end=True,
        elapsed_s=time.time()-started,overall=summarize_frames(records),
        offline_truth_geometry_sanity=summarize_frames([r['offline_truth_geometry_sanity'] for r in records]),
        by_scenario=groups(records,'scenario'),by_camera_range=groups(records,'range_band'),
        by_liquid_depth=groups(records,'height_band'),by_sensor=groups(records,'sensor'),records=records)
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('COMPLETE',args.backend,json.dumps(report['overall']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
    p.add_argument('--split',default='test',choices=['train','val','test'])
    p.add_argument('--backend',choices=['v6','v8','v17','v16_plane','raw','v17_raw'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--scenarios',nargs='+')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--confidence',type=float)
    p.add_argument('--fold',type=int,choices=[0,1,2]);p.add_argument('--checkpoint')
    p.add_argument('--label',default='frozen_baseline')
    args=p.parse_args()
    if args.checkpoint:
        if args.backend not in ('v17','v16_plane'):raise ValueError('Only precision checkpoints supported')
        WEIGHTS[args.backend]=Path(args.checkpoint)
    run(args)
