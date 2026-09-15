#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/training/precision-v17-20260916')
def main():
    frozen=json.loads((OUT/'frozen-experiment.json').read_text());selected=frozen['selected']
    reports={};rows=[]
    for p in sorted((OUT/'test').glob('*/*.json')):
        r=json.loads(p.read_text());name=p.parent.name;mode=p.stem;reports[name,mode]=r
        for material in ('transparent','translucent'):
            s=r['summary'][material];a=s['all'];ss=s['selected']
            frames=[f for f in r['frames'] if f['scenario']==material]
            frame_mae=[1000*f['all']['abs_sum']/f['all']['n'] for f in frames if f['all']['n']]
            rows.append(dict(method=name,mode=mode,material=material,mae_mm=a['mae_mm'],absrel=a['absrel'],success=a['success'],coverage=a['coverage'],selected_success=ss['success'],selected_precision=ss['precision'],selected_coverage=ss['coverage'],non_target=ss['non_target_fraction'],frame_median_mae_mm=float(np.median(frame_mae)),frame_p90_mae_mm=float(np.percentile(frame_mae,90)),anchorless=s['anchorless_frames']))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
    pairs=[(n,'v16') for n in ('v16_plane','mae_r4','mae_r4_plane','mae_r0','mae_r0_plane')]
    pairs.extend([('mae_r4_plane','mae_r4'),('mae_r0_plane','mae_r0')])
    comparisons=[]
    for method,reference in pairs:
        for mode in ('clean','bias8','bias-8','local10','drop80'):
            for material in ('transparent','translucent'):
                aa={f['sequence_id']:f['all'] for f in reports[method,mode]['frames'] if f['scenario']==material}
                bb={f['sequence_id']:f['all'] for f in reports[reference,mode]['frames'] if f['scenario']==material};assert set(aa)==set(bb)
                ids=sorted(aa);a=np.array([[aa[i][k] for k in ('abs_sum','n','good','possible')] for i in ids]);b=np.array([[bb[i][k] for k in ('abs_sum','n','good','possible')] for i in ids])
                def metric(v):return np.array([1000*v[0]/max(v[1],1),100*v[2]/max(v[3],1)])
                delta=metric(a.sum(0))-metric(b.sum(0));rng=np.random.default_rng(17071);samples=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(ids),len(ids));samples.append(metric(a[ix].sum(0))-metric(b[ix].sum(0)))
                lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
                comparisons.append(dict(method=method,reference=reference,mode=mode,material=material,scenes=len(ids),mae_delta_mm=float(delta[0]),mae_ci_mm=[float(lo[0]),float(hi[0])],success_delta_pp=float(delta[1]),success_ci_pp=[float(lo[1]),float(hi[1])]))
    (OUT/'paired-bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    manifest=Path('/root/autodl-tmp/liquid-depth-data/synthetic/precision-v17-fresh/manifest.csv')
    scene_rows={r['sequence_id']:r for r in csv.DictReader(manifest.open())}
    bands={}
    for seq,r in scene_rows.items():
        z=np.load(r['target_depth_path']);mask=(np.load(r['mask_path'])>0)&(z>=.1)&(z<=10)
        bands[seq]=int(np.searchsorted([.3,1,3],np.median(z[mask]))) if mask.any() else -1
    range_reports=[]
    for method in ('v16',selected):
        for material in ('transparent','translucent'):
            for band in range(4):
                frames=[f for f in reports[method,'clean']['frames'] if f['scenario']==material and bands[f['sequence_id']]==band]
                a=[f['all'] for f in frames];n=sum(f['n'] for f in a);possible=sum(f['possible'] for f in a)
                range_reports.append(dict(method=method,material=material,band=['0.1-0.3m','0.3-1m','1-3m','3-10m'][band],frames=len(frames),mae_mm=1000*sum(f['abs_sum'] for f in a)/max(n,1),success=sum(f['good'] for f in a)/max(possible,1),coverage=n/max(possible,1)))
    (OUT/'range-results.json').write_text(json.dumps(range_reports,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['v16','v16_plane','mae_r4','mae_r4_plane','mae_r0','mae_r0_plane']
    names=['V16','V16 + fit','MAE r4','MAE r4 + fit','MAE r0','MAE r0 + fit']
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for ax,material in zip(axs,('transparent','translucent')):
        vals=[next(r['mae_mm'] for r in rows if r['method']==m and r['mode']=='clean' and r['material']==material) for m in methods]
        bars=ax.bar(np.arange(len(vals)),vals,color=['#e87924' if m==selected else '#2463eb' for m in methods])
        ax.bar_label(bars,fmt='%.1f');ax.set_xticks(np.arange(len(vals)),names,rotation=25,ha='right');ax.set_title(material);ax.set_ylabel('Axial depth MAE (mm), lower is better');ax.set_ylim(0,max(vals)*1.2);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('V17 | 128 new scenes per material | orange: selected before test');fig.tight_layout();fig.savefig(OUT/'V17-comparison.png',dpi=160)
    print('SUMMARIZED',selected,len(rows),len(comparisons))
if __name__=='__main__':main()
