#!/usr/bin/env python3
"""Freeze route/postprocess on development fold1; audit fold2; then test fresh scenes."""
import argparse,json
from pathlib import Path
import numpy as np,torch
import train_precision_v17 as train
import train_observable_v16 as v16
OUT=train.OUT
def candidates():
    return [
        ('v16',4,v16.OUT/'gated/best.pth',False),
        ('v16_plane',4,v16.OUT/'gated/best.pth',True),
        ('mae_r4',4,OUT/'r4/best.pth',False),
        ('mae_r4_plane',4,OUT/'r4/best.pth',True),
        ('mae_r0',0,OUT/'r0/best.pth',False),
        ('mae_r0_plane',0,OUT/'r0/best.pth',True)]
def select():
    if (OUT/'frozen-experiment.json').exists():raise FileExistsError(OUT/'frozen-experiment.json')
    report={};baseline=None;options=[]
    for name,radius,checkpoint,post in candidates():
        pipe=train.make(radius,checkpoint,post).eval()
        r=train.evaluate(pipe,part=1);report[name]=r
        if name=='v16':baseline=r
        groups=r['summary'];reference=baseline['summary']
        qualified=all(groups[s]['all']['success']>=reference[s]['all']['success']-.02 and groups[s]['all']['coverage']>=reference[s]['all']['coverage']-.01 and groups[s]['all']['mae_mm']<=reference[s]['all']['mae_mm']*1.05 for s in ('transparent','translucent'))
        if name.startswith('mae_'):
            qualified &= json.loads((checkpoint.parent/'frozen.json').read_text())['qualified_on_selection_fold']
        score=float(np.mean([groups[s]['all']['mae_mm'] for s in ('transparent','translucent')]))
        if qualified:options.append((score,name))
        print('SELECT',name,score,qualified,{s:(groups[s]['all']['mae_mm'],groups[s]['all']['success']) for s in ('transparent','translucent')},flush=True)
        del pipe;torch.cuda.empty_cache()
    winner=min(options)[1]
    train.save(OUT/'selection.json',dict(selected=winner,eligible=options,reports=report))
    frozen={'selected':winner,'input_keys':['rgb','raw_depth_m'],'confidence_qualified':False,'locator_sha256':v16.digest(v16.OUT/'locator.pth'),'candidates':[{ 'name':name,'radius':radius,'checkpoint':str(checkpoint),'checkpoint_sha256':v16.digest(checkpoint),'surface_fit':post} for name,radius,checkpoint,post in candidates()]}
    train.save(OUT/'frozen-experiment.json',frozen)
    profile={'selected':winner,'variants':{c['name']:{'checkpoint':('reference/best.pth' if c['name'].startswith('v16') else f"r{c['radius']}/best.pth"),'radius':c['radius'],'surface_fit':c['surface_fit']} for c in frozen['candidates']}}
    train.save(OUT/'runtime-profile.json',profile)
    for name,radius,checkpoint,post in candidates():
        if name not in ('v16',winner):continue
        pipe=train.make(radius,checkpoint,post)
        results={mode:train.evaluate(pipe,mode=mode,part=2) for mode in ('clean','bias8','bias-8','drop80')}
        train.save(OUT/f'audit-{name}.json',results);del pipe;torch.cuda.empty_cache()
    print('FROZEN_WINNER',winner,flush=True)

def test(manifest):
    frozen=json.loads((OUT/'frozen-experiment.json').read_text())
    assert v16.digest(v16.OUT/'locator.pth')==frozen['locator_sha256']
    for c in frozen['candidates']:
        assert v16.digest(c['checkpoint'])==c['checkpoint_sha256']
        out=OUT/'test'/c['name'];out.mkdir(parents=True,exist_ok=False)
        pipe=train.make(c['radius'],Path(c['checkpoint']),c['surface_fit']).eval()
        for mode in ('clean','bias8','bias-8','local10','drop80'):
            r=train.evaluate(pipe,mode=mode,part=None,manifest=manifest,split='test')
            train.save(out/f'{mode}.json',r)
            print('TEST',c['name'],mode,{s:(v['all']['mae_mm'],v['all']['success']) for s,v in r['summary'].items() if s!='all'},flush=True)
        del pipe;torch.cuda.empty_cache()
def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['select','test'],required=True);p.add_argument('--manifest');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260917031)
    select() if a.stage=='select' else test(a.manifest)
if __name__=='__main__':main()
