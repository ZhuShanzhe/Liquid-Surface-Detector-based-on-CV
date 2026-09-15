#!/usr/bin/env python3
"""Frozen original SeeGroup and project V9.1 on controlled stratified scenes."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from liquid_depth.models.universal import UniversalLiquidSurfaceNet
from liquid_depth.models.layered import select_liquid_interface
from liquid_depth.models.observable_v16 import encode_rgbd
from liquid_depth.stratified_diagnostics import match_interfaces
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic/stratified-v18-cycles')
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/stratified-seegroup-v18')
SOURCE=Path('/root/autodl-tmp/liquid-depth-data/research/sources/SeeGroup')
OFFICIAL=Path('/root/autodl-tmp/liquid-depth-data/research/checkpoints/seegroup/seegroup.pth')
V9=Path('/root/autodl-tmp/liquid-depth-artifacts/training/universal-liquid-v9-1-transparent-multilayer/best.pth')
def load_original():
    sys.path.insert(0,str(SOURCE))
    from model.seegroup import SeeGroup
    model=SeeGroup(encoder='vitl',multi_depth_module=dict(type='SeeGroupMultiDepthHead',max_depth=20,num_heads=4,min_beta=1.0,depth_head_module=dict(type='SeeGroupDepthHead',in_channels=1024,features=256,out_channels=[256,512,1024,1024])))
    state=torch.load(OFFICIAL,map_location='cpu',weights_only=True)
    weights=state['model'] if 'model' in state else state
    weights={k.removeprefix('module.'):v for k,v in weights.items()}
    model.load_state_dict(weights,strict=True)
    return model.cuda().eval()
def load_v9():
    state=torch.load(V9,map_location='cpu',weights_only=False);keys=state['model']
    model=UniversalLiquidSurfaceNet(base_channels=int(state.get('base_channels',32)),min_depth_m=float(state.get('min_depth_m',.1)),max_depth_m=float(state.get('max_depth_m',10)),rgb_prior_enabled=bool(state.get('rgb_prior_enabled',any(k.startswith('rgb_prior.') for k in keys))),separate_confidence_head=bool(state.get('separate_confidence_head',True)),level_calibration_enabled=bool(state.get('level_calibration_enabled',any(k.startswith('level_calibration_head.') for k in keys))),calibration_scale_limit=float(state.get('calibration_scale_limit',.05)),calibration_bias_limit_m=float(state.get('calibration_bias_limit_m',.02)),robust_depth_anchor_enabled=bool(state.get('robust_depth_anchor_enabled',False)),num_ray_layers=int(state.get('num_ray_layers',4)),ray_layer_hidden_channels=int(state.get('ray_layer_hidden_channels',32)))
    model.load_state_dict(keys,strict=True);return model.cuda().eval()
def read(folder):
    rgb=cv2.cvtColor(cv2.imread(str(folder/'rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
    arrays=dict(np.load(folder/'arrays.npz'));meta=json.loads((folder/'scene.json').read_text())
    return torch.from_numpy(rgb.transpose(2,0,1))[None].cuda(),arrays,meta
def report_frame(depth,active,arrays,meta):
    ids=np.flatnonzero(arrays['roi']);ids=ids[np.linspace(0,len(ids)-1,min(512,len(ids))).astype(int)]
    truth=[meta['camera_distance_m']]
    if meta['group']!='single':truth.append(meta['camera_distance_m']+meta['thickness_m'])
    truth.append(meta['camera_distance_m']+.1001)
    result=dict(id=meta['id'],pair=meta['pair'],group=meta['group'],sampled_pixels=len(ids),candidate_count=[],tolerances={},heuristic_thickness_errors_mm=[],oracle_pair_thickness_errors_mm=[],inner_nearest_errors_mm=[],top_nearest_errors_mm=[])
    p=depth.reshape(4,-1);valid=active.reshape(4,-1)&np.isfinite(p)&(p>0)
    for tolname,rel in [('fixed5mm',0.),('relative2pct',.02)]:
        sums=dict(matched=0,truth=0,predicted=0,complete=0,all_truth=0,legacy_hits=0,inner_hits=0,top_hits=0,both_liquid_hits=0)
        for i in ids:
            pred=p[:,i][valid[:,i]]
            m=match_interfaces(pred,truth,relative_tolerance=rel)
            sums['matched']+=m['matched_count'];sums['truth']+=len(truth);sums['predicted']+=len(pred);sums['complete']+=m['complete_set_correct'];sums['all_truth']+=m['all_truth_matched'];sums['legacy_hits']+=round(m['legacy_recall']*len(truth))
            matched={x[0] for x in m['pairs']};sums['top_hits']+=0 in matched
            if meta['group']!='single':
                sums['inner_hits']+=1 in matched
                sums['both_liquid_hits']+=(0 in matched and 1 in matched)
        result['tolerances'][tolname]=sums
    for i in ids:
        pred=np.sort(p[:,i][valid[:,i]]);result['candidate_count'].append(len(pred))
        if len(pred):result['top_nearest_errors_mm'].append(float(abs(pred-truth[0]).min()*1000))
        if meta['group']!='single':
            if len(pred):result['inner_nearest_errors_mm'].append(float(abs(pred-truth[1]).min()*1000))
            if len(pred)>=2:
                result['heuristic_thickness_errors_mm'].append(float(abs(pred[1]-pred[0]-meta['thickness_m'])*1000))
                # Diagnostic only: distinct two-candidate assignment minimizes depth error using truth.
                options=[(abs(float(a)-truth[0])+abs(float(b)-truth[1]),abs(float(b-a)-meta['thickness_m'])*1000) for a_i,a in enumerate(pred) for b in pred[a_i+1:]]
                result['oracle_pair_thickness_errors_mm'].append(min(options)[1])
    return result
def main():
    global DATA, OUT
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--data',type=Path,default=DATA);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
    DATA,OUT=a.data,a.output;OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(18001)
    if a.smoke:
        model=load_original()
        x=torch.rand(1,3,180,320,device='cuda')
        with torch.inference_mode():r=model.forward_unscale({'image':(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]})
        print('ORIGINAL_SMOKE', {k:list(v.shape) for k,v in r.items()},flush=True);return
    assert (DATA/'manifest.json').exists()
    records={};times={};folder=OUT/'predictions';folder.mkdir(exist_ok=False)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OFFICIAL,V9]}
    (OUT/'checkpoint-hashes.json').write_text(json.dumps(hashes,indent=2))
    for kind in ('official','v9'):
        model=load_original() if kind=='official' else load_v9()
        methods=('official_native','official_sensor_scale') if kind=='official' else ('v9_1_set','v9_1_zero_raw')
        for name in methods:records[name]=[];times[name]=[]
        for i in range(64):
            rgb,arrays,meta=read(DATA/f'{i:03d}');raw=torch.from_numpy(arrays['raw'][None,None]).cuda()
            with torch.inference_mode():
                if kind=='official':
                    x=(rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
                    torch.cuda.synchronize();start=time.perf_counter();out=model.forward_unscale({'image':x});torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                    depth=out['depth'][0].cpu().numpy();active=out['valid_mask'][0].cpu().numpy()>0
                    first=np.where(active&np.isfinite(depth)&(depth>0),depth,np.inf).min(0)
                    support=np.isfinite(first)&(arrays['raw']>0)
                    scale=float(np.median(arrays['raw'][support])/np.median(first[support])) if support.any() else None
                    variants=[('official_native',depth,1),('official_sensor_scale',depth*scale if scale is not None else depth*0,scale)]
                    for name,d,s in variants:
                        r=report_frame(d,active,arrays,meta);r['sensor_scale']=s;records[name].append(r);times[name].append(elapsed)
                        np.savez_compressed(folder/f'{name}-{i:03d}.npz',depth=d,active=active)
                else:
                    for name in methods:
                        inp=encode_rgbd(rgb,raw if name=='v9_1_set' else torch.zeros_like(raw))
                        torch.cuda.synchronize();start=time.perf_counter();out=model(inp);torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
                        d=out['layer_depths_m'][0].cpu().numpy();active=out['layer_presence_logits'][0].sigmoid().cpu().numpy()>=.5
                        r=report_frame(d,active,arrays,meta)
                        selected=select_liquid_interface(out,out['depth_m'],relative_tolerance=.008,absolute_floor_m=.02,rejection_multiplier=3.,confidence_threshold=.5)
                        goodmask=out['mask_logits'][0,0].sigmoid().cpu().numpy()>=.5
                        accepted=selected['accepted'][0,0].cpu().numpy()&goodmask&arrays['roi']
                        selecteddepth=selected['depth_m'][0,0].cpu().numpy()
                        r['original_selector_top']=dict(coverage=float(accepted.sum()/arrays['roi'].sum()),mae_mm=float(abs(selecteddepth[accepted]-meta['camera_distance_m']).mean()*1000) if accepted.any() else None)
                        records[name].append(r);times[name].append(elapsed)
                        np.savez_compressed(folder/f'{name}-{i:03d}.npz',depth=d,active=active,mask=goodmask)
            if i%8==7:print('EVAL',kind,i+1,flush=True)
        for name in methods:
            (OUT/(name+'.json')).write_text(json.dumps(records[name],indent=2))
        del model;torch.cuda.empty_cache()
    (OUT/'timings.json').write_text(json.dumps({k:dict(first_ms=v[0],median_ms=float(np.median(v[1:])),p95_ms=float(np.percentile(v[1:],95)),note='batch1 model forward, official includes resize; excludes file IO, host transfer and metrics; no separate repeated latency benchmark') for k,v in times.items()},indent=2))
    print('SCREENING_FINISHED',flush=True)
if __name__=='__main__':main()
