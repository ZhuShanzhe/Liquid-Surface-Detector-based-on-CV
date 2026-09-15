#!/usr/bin/env python3
"""V17 phase B: development-selected distance blend; new untouched test seed."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
import train_precision_v17 as train
import train_observable_v16 as v16
from liquid_depth.models.precision_v17 import DistanceBlendPipeline
OUT=train.OUT/'phase-b'
WINDOWS=[(.5,1.),(1.,1.5),(1.,2.)]
MANIFEST=Path('/root/autodl-tmp/liquid-depth-data/synthetic/precision-v17b-fresh/manifest.csv')
def make(name,window=None):
    if name=='v16':return train.make(4,v16.OUT/'gated/best.pth',False).eval()
    if name=='v16_plane':return train.make(4,v16.OUT/'gated/best.pth',True).eval()
    if name=='v17':return train.make(4,train.OUT/'r4/best.pth',True).eval()
    return DistanceBlendPipeline(make('v16_plane'),make('v17'),*window).eval()
def bands(manifest):
    result={}
    for r in csv.DictReader(Path(manifest).open()):
        z=np.load(r['target_depth_path']);m=(np.load(r['mask_path'])>0)&(z>=.1)&(z<=10)
        result[r['sequence_id']]=int(np.searchsorted([.3,1,3],np.median(z[m]))) if m.any() else -1
    return result
def ranges(r,band_map):
    result={}
    for s in ('transparent','translucent'):
        for band in range(4):
            a=[f['all'] for f in r['frames'] if f['scenario']==s and band_map[f['sequence_id']]==band]
            n=sum(f['n'] for f in a);p=sum(f['possible'] for f in a)
            result[f'{s}:{band}']=dict(frames=len(a),mae_mm=1000*sum(f['abs_sum'] for f in a)/max(n,1),success=sum(f['good'] for f in a)/max(p,1),coverage=n/max(p,1))
    return result
def select():
    OUT.mkdir(exist_ok=False)
    train.save(OUT/'protocol.json',dict(windows_m=WINDOWS,near='v16_plane',far='v17',distance_input='median predicted near depth inside predicted mask',selection='development fold1 only',near_guards='Each material in bands 0 and 1: MAE <= v16_plane + 0.5 mm; success >= v16_plane - 2 pp; coverage >= v16_plane - 1 pp',global_guards='Each material MAE <= v16 * 1.05; success >= v16 - 2 pp; coverage >= v16 - 1 pp',score='macro material MAE',test_seed=20260917081,phase_a_test_role='diagnostic motivating phase B; never untouched for phase B'))
    bm=bands(v16.DEV);reports={};metrics={};eligible=[]
    for name,w in [('v16',None),('v16_plane',None)]+[(f'blend{i}',w) for i,w in enumerate(WINDOWS)]:
        pipe=make(name,w);r=train.evaluate(pipe,part=1);reports[name]=r;metrics[name]=ranges(r,bm)
        ref=reports['v16']['summary'];s=r['summary']
        ok=all(s[k]['all']['mae_mm']<=ref[k]['all']['mae_mm']*1.05 and s[k]['all']['success']>=ref[k]['all']['success']-.02 and s[k]['all']['coverage']>=ref[k]['all']['coverage']-.01 for k in ('transparent','translucent'))
        if name.startswith('blend'):
            for k,m in metrics[name].items():
                if int(k.split(':')[1])>1:continue
                b=metrics['v16_plane'][k]
                ok=ok and m['frames']>0 and m['mae_mm']<=b['mae_mm']+.5 and m['success']>=b['success']-.02 and m['coverage']>=b['coverage']-.01
        score=float(np.mean([s[k]['all']['mae_mm'] for k in ('transparent','translucent')]))
        if ok:eligible.append((score,name,w))
        print('SELECT_B',name,score,bool(ok),flush=True)
        del pipe;torch.cuda.empty_cache()
    winner=min(eligible,key=lambda x:x[0])
    frozen=dict(selected=winner[1],window_m=winner[2],eligible=eligible,locator_sha256=v16.digest(v16.OUT/'locator.pth'),near_sha256=v16.digest(v16.OUT/'gated/best.pth'),far_sha256=v16.digest(train.OUT/'r4/best.pth'))
    diagnostic=min((sum(reports[f'blend{i}']['summary'][k]['all']['mae_mm'] for k in ('transparent','translucent'))/2,f'blend{i}',w) for i,w in enumerate(WINDOWS))
    frozen['diagnostic']=dict(name=diagnostic[1],window_m=diagnostic[2],promotion_allowed=False,reason='Lowest development macro MAE, retained as diagnostic even when guards unverified')
    train.save(OUT/'selection.json',dict(reports=reports,ranges=metrics))
    train.save(OUT/'frozen.json',frozen)
    print('FROZEN_B',frozen,flush=True)
def test():
    f=json.loads((OUT/'frozen.json').read_text())
    assert f['locator_sha256']==v16.digest(v16.OUT/'locator.pth')
    assert f['near_sha256']==v16.digest(v16.OUT/'gated/best.pth')
    assert f['far_sha256']==v16.digest(train.OUT/'r4/best.pth')
    fresh=list(csv.DictReader(MANIFEST.open()));ids={r['sequence_id'] for r in fresh}
    old=[v16.DEV,'/root/autodl-tmp/liquid-depth-data/synthetic/precision-v17-fresh/manifest.csv','/root/autodl-tmp/liquid-depth-data/synthetic/observable-v16-fresh/manifest.csv','/root/autodl-tmp/liquid-depth-data/synthetic/liquid-adaptation-v15-fresh-20260915/manifest.csv','/root/autodl-tmp/liquid-depth-data/synthetic/liquid-sim-v3-calibration-top-12000/manifest_multilayer_v4_refined.csv']
    assert len(ids)==len(fresh)==256
    for p in old:assert not ids.intersection(r['sequence_id'] for r in csv.DictReader(Path(p).open()))
    train.save(OUT/'data-audit.json',dict(scenes=len(ids),manifest_sha256=v16.digest(MANIFEST),no_overlap_with=[str(p) for p in old]))
    for name in ('v16','v16_plane','v17','selected','diagnostic'):
        pipe=make(f['diagnostic']['name'],f['diagnostic']['window_m']) if name=='diagnostic' else (make(f['selected'],f['window_m']) if name=='selected' else make(name))
        dest=OUT/'test'/name;dest.mkdir(parents=True,exist_ok=False)
        for mode in ('clean','bias8','bias-8','local10','drop80'):
            r=train.evaluate(pipe,mode=mode,part=None,manifest=MANIFEST,split='test')
            train.save(dest/f'{mode}.json',r)
            print('TEST_B',name,mode,{s:(v['all']['mae_mm'],v['all']['success']) for s,v in r['summary'].items() if s!='all'},flush=True)
        del pipe;torch.cuda.empty_cache()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['select','test'],required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260917081)
    select() if a.stage=='select' else test()
