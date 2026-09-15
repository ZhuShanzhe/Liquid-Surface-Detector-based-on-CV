#!/usr/bin/env python3
"""Summarize phase B without changing its frozen selection."""
import csv,json
from pathlib import Path
import numpy as np
import evaluate_precision_v17b as exp
OUT=exp.OUT
def main():
    f=json.loads((OUT/'frozen.json').read_text());bm=exp.bands(exp.MANIFEST)
    reports={};rows=[];ranges=[]
    for path in sorted((OUT/'test').glob('*/*.json')):
        name=path.parent.name;mode=path.stem;r=json.loads(path.read_text());reports[name,mode]=r
        for material in ('transparent','translucent'):
            a=r['summary'][material]['all'];s=r['summary'][material]['selected']
            frames=[x['all'] for x in r['frames'] if x['scenario']==material]
            rows.append(dict(method=name,mode=mode,material=material,**a,selected_coverage=s['coverage'],selected_precision=s['precision'],frame_p90_mae_mm=float(np.percentile([1000*x['abs_sum']/max(x['n'],1) for x in frames],90))))
        if mode=='clean':
            ranges.extend(dict(method=name,material=k.split(':')[0],band=int(k.split(':')[1]),**v) for k,v in exp.ranges(r,bm).items())
    assert len(reports)==25
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2));(OUT/'range-results.json').write_text(json.dumps(ranges,indent=2))
    with (OUT/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    comparisons=[]
    for method,reference in [(m,r) for m in ('selected','diagnostic') for r in ('v16','v17','v16_plane')]:
        for mode in ('clean','bias8','bias-8','local10','drop80'):
            for material in ('transparent','translucent'):
                aa={x['sequence_id']:x['all'] for x in reports[method,mode]['frames'] if x['scenario']==material}
                bb={x['sequence_id']:x['all'] for x in reports[reference,mode]['frames'] if x['scenario']==material}
                assert aa.keys()==bb.keys();ids=sorted(aa)
                a=np.array([[aa[i][k] for k in ('abs_sum','n','good','possible')] for i in ids])
                b=np.array([[bb[i][k] for k in ('abs_sum','n','good','possible')] for i in ids])
                def metric(v):return np.array([1000*v[0]/max(v[1],1),100*v[2]/max(v[3],1)])
                delta=metric(a.sum(0))-metric(b.sum(0));rng=np.random.default_rng(17081);samples=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(ids),len(ids));samples.append(metric(a[ix].sum(0))-metric(b[ix].sum(0)))
                lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
                comparisons.append(dict(method=method,reference=reference,mode=mode,material=material,scenes=len(ids),mae_delta_mm=float(delta[0]),mae_ci_mm=[float(lo[0]),float(hi[0])],success_delta_pp=float(delta[1]),success_ci_pp=[float(lo[1]),float(hi[1])]))
    (OUT/'paired-bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,material in zip(axes,('transparent','translucent')):
        methods=['v16','v16_plane','v17','diagnostic'];values=[next(x['mae_mm'] for x in rows if x['method']==m and x['mode']=='clean' and x['material']==material) for m in methods]
        bars=ax.bar(['V16','V16 + fit','V17 phase A','Blend (research)'],values,color=['#46678c']*3+['#e87924'])
        ax.bar_label(bars,fmt='%.2f');ax.set_title(material);ax.set_ylabel('Axial depth MAE (mm)');ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Independent phase B test | 128 scenes per material');fig.tight_layout();fig.savefig(OUT/'V17B-comparison.png',dpi=160)
    print('SUMMARY_B',f['selected'],[x for x in rows if x['mode']=='clean'])
if __name__=='__main__':main()
