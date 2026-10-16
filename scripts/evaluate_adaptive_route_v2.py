"""Paired OFF/ON/AUTO tests, per-scene errors, small-object recall and return reliability."""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from evaluate_shared_empty_v45 import install
from evaluate_visual_route_v1 import metrics

from liquid_depth.adaptive_rgbd_meter import AdaptiveRGBDMeter
from liquid_depth.rgbd_evidence_model import RGBDEvidencePredictor, observed_features
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter


def cm(truth,pred,n):
    return np.bincount((truth*n+pred).ravel(),minlength=n*n).reshape(n,n)


def describe(confusion):
    diag=confusion.diagonal();union=confusion.sum(0)+confusion.sum(1)-diag
    return {'confusion':confusion.tolist(),'iou':(diag/np.maximum(union,1)).tolist(),
       'precision':(diag/np.maximum(confusion.sum(0),1)).tolist(),'recall':(diag/np.maximum(confusion.sum(1),1)).tolist()}


def object_counts(target,pred):
    n,gt,stats,_=cv2.connectedComponentsWithStats(target.astype(np.uint8));pn,pm=cv2.connectedComponents(pred.astype(np.uint8))
    counts=defaultdict(lambda:[0,0]);matched_pred=set()
    for i in range(1,n):
        area=stats[i,cv2.CC_STAT_AREA];diameter=2*np.sqrt(area/np.pi)
        key='lt4px' if diameter<4 else '4-8px' if diameter<8 else '8-16px' if diameter<16 else 'ge16px'
        labels,support=np.unique(pm[gt==i],return_counts=True);hit=0
        for label,intersection in zip(labels,support,strict=True):
            if label==0:continue
            union=int(area)+int((pm==label).sum())-int(intersection)
            if intersection/max(union,1)>=.1:hit=1;matched_pred.add(int(label));break
        counts[key][0]+=1;counts[key][1]+=hit
    return dict(counts),pn-1,len(matched_pred)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1)
    predictor=RGBDEvidencePredictor(a.model,'cuda' if torch.cuda.is_available() else 'cpu')
    paths=json.loads((a.model.parent/'split.json').read_text())['test']
    installs={};audits=[]
    for path in paths:
        case=int(Path(path).name);item=install(case,a.root,a.root/'dry',a.seed,max_annotation_attempts=6,source_slot_stride=1)
        installs[case]=item;audits.append(item['audit'])
    (a.output/'calibration-audit.json').write_text(json.dumps(audits,indent=2))
    records=[];semantic=np.zeros((4,4),np.int64);rgb_sem=np.zeros_like(semantic);origin_cm=np.zeros((7,7),np.int64)
    reliability=[];objects=defaultdict(lambda:[0,0]);rgb_objects=defaultdict(lambda:[0,0])
    object_precision={'coupled':[0,0],'rgb_only':[0,0]}
    for path in paths:
        folder=a.root/Path(path).name;case=int(folder.name);item=installs[case]
        base=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.5)
        meter=AdaptiveRGBDMeter(base,dry_depth_m=item['dry'],predictor=predictor,gate_checkpoint=a.gate)
        peak_control=UnevenSurfaceMeter(base,item['dry'])
        rgb=cv2.imread(str(folder/'rgb.png'))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw=np.load(frame['raw_path']);results={}
            for mode in ['off','on','auto']:
                start=time.perf_counter();result=meter.measure(rgb_bgr=rgb,depth_m=raw,mode=mode)
                results[mode]=(result,1000*(time.perf_counter()-start))
            start=time.perf_counter();legacy=peak_control.measure(raw)
            results['geometric_peak_control']=({**results['off'][0],
                'highest_height_m':legacy['highest_visible_height_m'],
                'diagnostics':{**results['off'][0]['diagnostics'],'legacy_peak_diagnostics':legacy}},
                1000*(time.perf_counter()-start))
            features=observed_features(rgb,raw,item['dry'],base.ordinary_container_roi,item['bundle'])
            coupled=predictor.predict(features);features[4:]=0;rgb_only=predictor.predict(features)
            # Labels enter strictly after all measurement calls and candidate decisions.
            label=np.load(folder/'labels_offline.npz')['semantic'];offline=np.load(folder/f"supervision_offline_{frame['repeat']:02d}.npz")
            semantic+=cm(label,coupled['semantic'].argmax(0),4);rgb_sem+=cm(label,rgb_only['semantic'].argmax(0),4)
            valid=base.ordinary_container_roi&(raw>0);origin_cm+=cm(offline['origin'][valid],coupled['origin'].argmax(0)[valid],7)
            reliability.extend(zip(coupled['reliability'][0,valid][::7].tolist(),offline['reliable'][valid][::7].tolist()))
            for prediction,table,name in [(coupled,objects,'coupled'),(rgb_only,rgb_objects,'rgb_only')]:
                counts,n_pred,matched=object_counts(label==3,prediction['semantic'].argmax(0)==3)
                object_precision[name][0]+=n_pred;object_precision[name][1]+=matched
                for key,(n,hit) in counts.items():table[key][0]+=n;table[key][1]+=hit
            water=label==2;missing=1-float((raw[water]>0).mean()) if water.any() else None
            truth=item['meta']['truth'];far=item['meta']['camera_to_bottom_m']>=6 and truth['liquid_height_m']<=1
            for mode,(r,elapsed) in results.items():
                records.append({'case':case,'repeat':frame['repeat'],'mode':mode,'scene':item['meta']['scenario'],
                    'sensor':item['meta']['sensor_family'],'band':item['meta']['band'],
                    'camera_to_bottom_m':item['meta']['camera_to_bottom_m'],'truth':truth,
                    'liquid_depth_m':r['liquid_depth_m'],'highest_height_m':r['highest_height_m'],
                    'numeric_output':r['numeric_output'],'visual_guidance_used':r['visual_guidance_used'],
                    'liquid_override_used':r.get('liquid_override_used',False),'algorithm_ms':elapsed,
                    'far_shallow_deferred':far,'common_precision_eligible':results['off'][0]['precision_eligible'],
                    'actual_liquid_missing_fraction_offline':missing,'floating_pixels_offline':int((label==3).sum()),
                    'diagnostics':r['diagnostics']})
        if case%20==0:print('EVALUATED',case,flush=True)
    groups=defaultdict(list)
    for r in records:
        for subset in ['all_frames','far_shallow' if r['far_shallow_deferred'] else 'remaining',
                       'evidence_eligible' if r['common_precision_eligible'] else 'evidence_insufficient']:
            for stratum in ['all',f"sensor={r['sensor']}",f"band={r['band']}"]:
                groups[f"{r['scene']}|{stratum}|{subset}|{r['mode']}"].append(r)
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    (a.output/'summary.json').write_text(json.dumps({k:metrics(v) for k,v in groups.items()},indent=2))
    (a.output/'highest-summary.json').write_text(json.dumps({k:metrics(v,'highest_height_m',True) for k,v in groups.items()},indent=2))
    conf,target=np.asarray(reliability).T
    reliability_metrics={'brier':float(np.mean((conf-target)**2)),'samples':len(target),'mean_prediction':float(conf.mean()),'positive_fraction':float(target.mean())}
    for k in ['on','auto']:
        rows=[r for r in records if r['mode']==k];times=np.array([r['algorithm_ms'] for r in rows])
        reliability_metrics[k]={'liquid_overrides':sum(r['liquid_override_used'] for r in rows),
               'visual_used':sum(r['visual_guidance_used'] for r in rows),
               'latency_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':float(times.max())}}
    obj=lambda table:{k:{'objects':v[0],'hits_iou_ge_0_1':v[1],'recall':v[1]/max(v[0],1)} for k,v in table.items()}
    pixel={'coupled_semantics':describe(semantic),'rgb_only_input_ablation':describe(rgb_sem),'origin':describe(origin_cm),
           'object_precision':{k:{'predicted_components':v[0],'matched_components':v[1],'precision':v[1]/max(v[0],1)} for k,v in object_precision.items()},
           'coupled_object_recall_by_size':obj(objects),'rgb_only_object_recall_by_size':obj(rgb_objects),
           'ablation_limit':'Same jointly trained model with depth channels removed, not a separately optimized RGB-only network.',
           'reliability':reliability_metrics}
    (a.output/'pixel-and-reliability.json').write_text(json.dumps(pixel,indent=2))
    source=['src/liquid_depth/adaptive_rgbd_meter.py','src/liquid_depth/rgbd_evidence_model.py','src/liquid_depth/frozen_symmetric_geometry.py',str(Path(__file__))]
    (a.output/'protocol.json').write_text(json.dumps({'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source},
        'checkpoint_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'gate_sha256':hashlib.sha256(a.gate.read_bytes()).hexdigest(),
        'test_truth_used_for_switch':False,'all_frames_retained':True,'sensor_proxy_not_field_qualification':True},indent=2))
    print('ADAPTIVE_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
