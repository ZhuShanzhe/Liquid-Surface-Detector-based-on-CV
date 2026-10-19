"""Routing validation by measured height regret, then untouched grouped holdout.

Rendered labels are only used AFTER the two candidate predictions are computed.
The head threshold is selected on validation installations, never test rows.
"""
import argparse,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_shared_empty_v45 import install
from measure_release import validated_profile
from liquid_depth.scene_classifier import SceneClassifier
from liquid_depth.tof_scene_meter import ToFSceneMeter
from liquid_depth.unified_tof import ToFInstallation,ToFObservation,UnifiedToFMeter


def summarize(rows,key):
    values=[r for r in rows if r[key] is not None];a=np.array([abs(r[key]-r['true_'+key]) for r in values])
    rel=np.array([abs(r[key]-r['true_'+key])/r['true_'+key] for r in values if r['true_'+key]>0])
    passes=sum(abs(r[key]-r['true_'+key])<=max(.05 if key=='highest_height_m' or r['uneven'] else .02,
                (.05 if key=='highest_height_m' or r['uneven'] else .02)*r['true_'+key]) for r in values)
    return dict(frames=len(rows),outputs=len(values),coverage_pct=100*len(values)/max(len(rows),1),
        mae_mm=float(a.mean()*1000) if len(a) else None,p95_mm=float(np.percentile(a,95)*1000) if len(a) else None,
        max_mm=float(a.max()*1000) if len(a) else None,pass_all_pct=100*passes/max(len(rows),1),
        absrel_pct=float(rel.mean()*100) if len(rel) else None,
        absolute_bins_pct=(100*np.histogram(a,[0,.005,.02,.05,.1,.2,1,np.inf])[0]/max(len(rows),1)).tolist())


def run(args):
    torch.set_num_threads(2);cv2.setNumThreads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    classifier=SceneClassifier(args.checkpoint)
    profile=args.out/'portable-profile.json';profile.write_text(json.dumps(validated_profile()))
    manifest=json.loads(args.manifest.read_text());val=set(manifest['validation_groups']);test=set(manifest['test_groups'])
    rows=[];errors=[]
    for group in sorted(val|test):
        folders=sorted(args.data.glob(f'{group:04d}_*'))
        if not folders:raise RuntimeError('missing_group')
        # All conditions share the same independent empty installation.
        folder=folders[0];init=args.out/f'init-{group:04d}'
        (init/'installation/00000').mkdir(parents=True,exist_ok=True);(init/'dry/0000').mkdir(parents=True,exist_ok=True)
        (init/'installation/00000/metadata.json').write_text((folder/'metadata.json').read_text())
        dry=np.load(folder/'dry_depth_m.npy',allow_pickle=False);np.save(init/'dry/0000/dry_depth_m.npy',dry)
        item=install(0,init/'installation',init/'dry',202610192003+group,max_annotation_attempts=24,source_slot_stride=1)
        installation=ToFInstallation(f'g{group}',item['bundle'],dry,'ordinary_diffuse')
        unified=UnifiedToFMeter(installation,classifier,profile_path=profile)
        manual=ToFSceneMeter.from_empty_geometry(item['bundle'],dry,profile_path=profile)
        # Warm actual optional visual backend, not included in warm measurement timings.
        warm=folders[0];depth=np.load(warm/'depth_m.npy');rgb=cv2.imread(str(warm/'rgb.png'))
        manual.measure('floating_objects',depth_m=depth,rgb_bgr=rgb,visual_enabled=True)
        for folder in folders:
            req=json.loads((folder/'request.json').read_text())
            depth=np.load(folder/'depth_m.npy',allow_pickle=False);rgb=cv2.imread(str(folder/'rgb.png'))
            start=time.perf_counter();off=manual.measure('ordinary_diffuse',depth_m=depth,rgb_bgr=rgb,visual_enabled=False)
            offms=(time.perf_counter()-start)*1000
            start=time.perf_counter();on=manual.measure('floating_objects',depth_m=depth,rgb_bgr=rgb,visual_enabled=True)
            onms=(time.perf_counter()-start)*1000
            prediction=classifier.predict(rgb,depth_m=depth,dry_depth_m=dry,roi=unified.roi)
            # No hidden true type or height was used by either runtime candidate.
            truth=json.loads((folder/'truth.json').read_text())
            rows.append(dict(group=group,split='validation' if group in val else 'test',folder=str(folder),
                label_tags=[k for k,x in [('low_light',req['darkness']),('glare',req['glare']),('uneven_surface',req['wave_m']),('floating_objects',req['float_count']),('partial_depth_loss',req.get('loss',0))] if x],
                uneven=bool(req['wave_m']),true_liquid_depth_m=truth['liquid_height_m'],true_highest_height_m=truth['highest_height_m'],
                liquid_depth_m=off['liquid_depth_m'],on_liquid=on['liquid_depth_m'],
                off_highest=off.get('highest_height_m'),on_highest=on.get('highest_height_m'),
                float_score=prediction['tag_scores']['floating_objects'],classification=prediction,off_ms=offms,on_ms=onms))
        print('E2E_GROUP',group,len(rows),flush=True)
    validation=[r for r in rows if r['split']=='validation'];choices=[]
    for threshold in np.arange(.1,1.01,.025):
        e=[]
        for r in validation:
            h=r['on_highest'] if r['float_score']>=threshold else r['off_highest']
            e.append(abs(h-r['true_highest_height_m']) if h is not None else 10.)
        # Continuous height regret (rather than only a label or through-rate).
        choices.append(dict(threshold=float(threshold),mae_m=float(np.mean(e)),p95_m=float(np.percentile(e,95)),max_m=float(max(e))))
    selected=min(choices,key=lambda c:(c['mae_m'],c['p95_m']))
    # Freeze outcome-derived activation threshold before calculating test metrics.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    state['activation_thresholds']={'float_highest':selected['threshold']};state['activation_calibration']='validation_highest_height_MAE'
    state['model_id']=state['model_id']+'_outcome_calibrated'
    torch.save(state,args.out/'scene_head_outcome.pt')
    records=[]
    for r in rows:
        use=r['float_score']>=selected['threshold']
        for route,h in [('automatic',r['on_highest'] if use else r['off_highest']),
                        ('manual',r['on_highest'] if 'floating_objects' in r['label_tags'] else r['off_highest'])]:
            records.append({**r,'route':route,'highest_height_m':h,'visual_executed':use if route=='automatic' else 'floating_objects' in r['label_tags']})
    metrics={}
    for split in ('validation','test'):
        for route in ('automatic','manual'):
            subset=[r for r in records if r['split']==split and r['route']==route]
            metrics[f'{split}|{route}|all']=dict(liquid=summarize(subset,'liquid_depth_m'),highest=summarize(subset,'highest_height_m'))
            for kind in ('ordinary','single','compound'):
                q=[r for r in subset if (not r['label_tags'] if kind=='ordinary' else len(r['label_tags'])==1 if kind=='single' else len(r['label_tags'])>1)]
                metrics[f'{split}|{route}|{kind}']=dict(liquid=summarize(q,'liquid_depth_m'),highest=summarize(q,'highest_height_m'))
    summary=dict(validation_threshold=selected,threshold_candidates=choices,metrics=metrics,
        liquid_exact_equal=all(r['liquid_depth_m']==r['on_liquid'] for r in rows),
        test_installations=len(test),validation_installations=len(val),excluded_by_error=0,
        scope='opaque_uniform_liquid_only_known_RGB_quality_combinations; optical_compounds_not_qualified')
    (args.out/'routing_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (args.out/'routing_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
