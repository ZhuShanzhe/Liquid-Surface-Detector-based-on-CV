"""Paired frozen-baseline/symmetric/learned-visual evaluation, truth after inference."""
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
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.symmetric_visual_meter import SymmetricVisualMeter
from liquid_depth.visual_region_model import RegionPredictor
from liquid_depth.illumination_scene_meter import IlluminationSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter


def metrics(rows,key='liquid_depth_m',peak=False):
    present=[r for r in rows if r[key] is not None]
    ae=np.array([abs(r[key]-r['truth']['highest_height_m' if peak else 'liquid_height_m']) for r in present])
    depths=np.array([r['truth']['highest_height_m' if peak else 'liquid_height_m'] for r in present])
    tolerance=np.maximum(.05 if peak else .02,(.05 if peak else .02)*depths)
    wet=depths>.001;relative=ae[wet]/depths[wet]
    return {'frames':len(rows),'numeric_frames':len(present),'numeric_coverage_pct':100*len(present)/max(len(rows),1),
            'pass_pct':100*int((ae<=tolerance).sum())/max(len(rows),1),
            'mae_mm':float(1000*ae.mean()) if len(ae) else None,'p95_mm':float(1000*np.percentile(ae,95)) if len(ae) else None,
            'absolute_error_bins_pct':{label:100*int(((ae>=lo)&(ae<hi)).sum())/max(len(rows),1) for label,lo,hi in [('0-5mm',0,.005),('5-20mm',.005,.02),('20-50mm',.02,.05),('50-100mm',.05,.1),('100mm+',.1,np.inf)]},
            'relative_error_bins_pct_nonzero':{label:100*int(((relative>=lo)&(relative<hi)).sum())/max(len(relative),1) for label,lo,hi in [('0-1%',0,.01),('1-2%',.01,.02),('2-5%',.02,.05),('5-10%',.05,.1),('10%+',.1,np.inf)]},
            'relative_denominator_nonzero_frames':int(wet.sum())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--split',choices=['test','validation'],default='test')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1);torch.set_num_threads(4)
    splits=json.loads((a.model.parent/'split.json').read_text());predictor=RegionPredictor(a.model,'cuda' if torch.cuda.is_available() else 'cpu')
    records=[];audits=[];confusion=np.zeros((4,4),np.int64);calibration=[]
    # Preflight every test installation before collecting accuracy.
    installations={}
    for folder in splits[a.split]:
        case=int(Path(folder).name)
        try:
            item=install(case,a.root,a.root/'dry',a.seed,max_annotation_attempts=6,source_slot_stride=1)
            installations[case]=item;audits.append(item['audit'])
        except ValueError as exc:
            audits.append({'case':case,'invalid_prior':str(exc)})
    (a.output/'installation-audit.json').write_text(json.dumps(audits,indent=2))
    if len(installations)!=len(splits[a.split]):
        raise ValueError('repair_invalid_calibration_inputs_before_accuracy_evaluation')
    for folder in splits[a.split]:
        folder=a.root/Path(folder).name;case=int(folder.name)
        if case not in installations:continue
        item=installations[case];meta=item['meta'];rgb=cv2.imread(str(folder/'rgb.png'))
        base=FormalSceneMeter.from_empty_geometry(item['bundle'],item['dry'],max_return_depth_m=10.,
                  quantized_disparity_profile={'disparity_step_px':1/16,'disparity_noise_px':.26} if meta['sensor_family']=='structured_light' else None)
        symmetric=SymmetricVisualMeter(base)
        visual=SymmetricVisualMeter(base,predictor=predictor)
        illumination=IlluminationSceneMeter(base,dry_depth_m=item['dry'],refinement_enabled=True)
        uneven=UnevenSurfaceMeter(base,item['dry'])
        start=time.perf_counter();probs=predictor(rgb,base.ordinary_container_roi)
        semantic_ms=1000*(time.perf_counter()-start)
        # No labels loaded until predicted probabilities have been produced.
        target=np.load(folder/'labels_offline.npz')['semantic']
        pred=probs.argmax(0);confusion+=np.bincount((target*4+pred).ravel(),minlength=16).reshape(4,4)
        confidence=probs.max(0).ravel()[::31];correct=(pred==target).ravel()[::31]
        calibration.extend(zip(confidence.tolist(),correct.tolist()))
        for frame in json.loads((folder/'frames.json').read_text()):
            raw=np.load(frame['raw_path'])
            start=time.perf_counter();old=base.measure('ordinary_diffuse',depth_m=raw);old_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter();sym=symmetric.measure(depth_m=raw,rgb_bgr=rgb);sym_ms=1000*(time.perf_counter()-start)
            start=time.perf_counter();new=visual.measure(depth_m=raw,rgb_bgr=rgb,probabilities=probs);new_ms=1000*(time.perf_counter()-start)+semantic_ms
            start=time.perf_counter()
            if meta['scenario'] in {'low_light','glare'}:
                scene_old=illumination.measure(meta['scenario'],rgb_bgr=rgb,depth_m=raw)
            elif meta['scenario'] in {'floating_objects','uneven_surface'}:
                u=uneven.measure(raw)
                scene_old={'liquid_depth_m':u['liquid_surface_height_m'],'highest_height_m':u['highest_visible_height_m'],'diagnostics':u}
            else:scene_old=old
            scene_old_ms=1000*(time.perf_counter()-start)
            # One input-only precision cohort shared by all methods. No true-height selection.
            grey=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY);roi=base.ordinary_container_roi
            unavailable_rgb=float(np.mean((grey[roi]<=2)|(grey[roi]>=253))) if roi.any() else 1.
            common=sym['precision_eligible'] and unavailable_rgb<.95
            exclusion=list(sym['diagnostics']['flags'])+(['rgb_uninformative'] if unavailable_rgb>=.95 else [])
            # Physical return-origin bookkeeping is offline and never enters selectors.
            origins=np.load(folder/f"origin_offline_{frame['repeat']:02d}.npy")
            origin_counts={str(i):int(((origins==i)&roi).sum()) for i in range(7)}
            for method,result,ms in [('archived_baseline',old,old_ms),('archived_scene_route',scene_old,scene_old_ms),('symmetric_depth',sym,sym_ms),('visual_guided',new,new_ms)]:
                records.append({'case':case,'repeat':frame['repeat'],'method':method,'scene':meta['scenario'],
                    'sensor':meta['sensor_family'],'band':meta['band'],'camera_to_bottom_m':meta['camera_to_bottom_m'],
                    'liquid_depth_m':result['liquid_depth_m'],'highest_height_m':result.get('highest_height_m'),
                    'common_precision_eligible':bool(common),'exclusion_reasons':exclusion,
                    'truth':meta['truth'],'algorithm_ms':ms,'diagnostics':result['diagnostics'],
                    'physical_origin_counts_offline_only':origin_counts})
        print('EVALUATED',case,meta['scenario'],flush=True)
    groups=defaultdict(list)
    for r in records:
        for cohort in ['all_frames']+(['precision_eligible'] if r['common_precision_eligible'] else ['excluded_evidence']):
            for stratum in ['all',f"sensor={r['sensor']}",f"band={r['band']}"]:
                groups[f"{r['scene']}|{stratum}|{cohort}|{r['method']}"].append(r)
    summary={k:metrics(v) for k,v in groups.items()}
    peaks={k:metrics(v,'highest_height_m',True) for k,v in groups.items() if k.endswith(('|visual_guided','|archived_scene_route'))}
    union=confusion.sum(0)+confusion.sum(1)-confusion.diagonal()
    iou=np.divide(confusion.diagonal(),union,out=np.zeros(4,float),where=union>0)
    confidence,correct=np.asarray(calibration).T
    ece=sum(float(np.mean((confidence>=lo)&(confidence<hi)))*abs(float(np.mean(confidence[(confidence>=lo)&(confidence<hi)]))-float(np.mean(correct[(confidence>=lo)&(confidence<hi)]))) for lo,hi in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]) if ((confidence>=lo)&(confidence<hi)).any())
    (a.output/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'highest-summary.json').write_text(json.dumps(peaks,indent=2))
    precision=np.divide(confusion.diagonal(),confusion.sum(0),out=np.zeros(4,float),where=confusion.sum(0)>0)
    recall=np.divide(confusion.diagonal(),confusion.sum(1),out=np.zeros(4,float),where=confusion.sum(1)>0)
    (a.output/'segmentation.json').write_text(json.dumps({'confusion':confusion.tolist(),'iou':iou.tolist(),'precision':precision.tolist(),'recall':recall.tolist(),'ece_15bins':ece},indent=2))
    sources=[Path('src/liquid_depth/symmetric_visual_meter.py'),Path('src/liquid_depth/visual_region_model.py'),Path(__file__)]
    protocol={'checkpoint_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'limitations':['synthetic pilot, no real sensor qualification','no transparent/refraction image experiment in this dataset','same parametric sensor family as training simulator'],
              'old_route_unchanged':True,'truth_used_for_candidate_selection':False,'shared_input_only_precision_cohort':True,
              'liquid_tolerance':'max(20mm,2% actual bottom-relative liquid depth)',
              'highest_tolerance':'max(50mm,5% actual highest bottom-relative height)',
              'all_frames_and_exclusions_retained':True,'zero_snap_or_clipping':False}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('EVALUATION_COMPLETE',json.dumps({k:v for k,v in summary.items() if '|all|all_frames|' in k}),flush=True)


if __name__=='__main__':main()
