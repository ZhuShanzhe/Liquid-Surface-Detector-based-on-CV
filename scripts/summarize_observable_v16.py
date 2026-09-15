#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
from screen_depth_methods_v14 import summary
OUT=Path('/root/autodl-tmp/liquid-depth-artifacts/training/observable-v16-20260916')
def main():
    reports={};rows=[]
    for p in sorted((OUT/'test').glob('*/*.json')):
        r=json.loads(p.read_text());name=p.parent.name;mode=p.stem;reports[name,mode]=r
        for material in ('transparent','translucent'):
            s=r['summary'][material];a=s['all'];selected=s['selected']
            rows.append(dict(method=name,mode=mode,material=material,mae_mm=a['mae_mm'],success=a['success'],coverage=a['coverage'],mask_iou=s['mask_iou'],mask_precision=s['mask_precision'],mask_recall=s['mask_recall'],selected_precision=selected['precision'],selected_coverage=selected['coverage'],selected_success=selected['success'],non_target=selected['non_target_fraction'],anchorless=s['anchorless_frames']))
    with (OUT/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
    comparisons=[]
    for method,reference in [('gated_tuned','v15_robust_full'),('full_tuned','v15_robust_full'),('predicted_gate_only','adapted_locator_full'),('gated_tuned','predicted_gate_only'),('full_tuned','adapted_locator_full'),('gated_tuned','full_tuned')]:
        for mode in ('clean','bias8','bias-8','local10','drop80'):
            for material in ('transparent','translucent'):
                aa={f['sequence_id']:f['all'] for f in reports[method,mode]['frames'] if f['scenario']==material}
                bb={f['sequence_id']:f['all'] for f in reports[reference,mode]['frames'] if f['scenario']==material}
                assert set(aa)==set(bb)
                ids=sorted(aa);a=np.array([[aa[i][k] for k in ('good','possible','abs_sum','n')] for i in ids]);b=np.array([[bb[i][k] for k in ('good','possible','abs_sum','n')] for i in ids])
                def metric(v):return np.array([100*v[0]/max(v[1],1),1000*v[2]/max(v[3],1)])
                diff=metric(a.sum(0))-metric(b.sum(0));rng=np.random.default_rng(16031);samples=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(ids),len(ids));samples.append(metric(a[ix].sum(0))-metric(b[ix].sum(0)))
                lo,hi=np.percentile(samples,[2.5,97.5],axis=0)
                comparisons.append(dict(method=method,reference=reference,mode=mode,material=material,scenes=len(ids),success_delta_pp=float(diff[0]),success_ci_pp=[float(lo[0]),float(hi[0])],mae_delta_mm=float(diff[1]),mae_ci_mm=[float(lo[1]),float(hi[1])]))
    (OUT/'paired-bootstrap.json').write_text(json.dumps(comparisons,indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['v15_robust_full','adapted_locator_full','predicted_gate_only','full_tuned','gated_tuned']
    names=['V15 full input','Locator adapted','Predicted gate','Full finetuned','Gated finetuned']
    fig,axs=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
    for ax,material in zip(axs,('transparent','translucent')):
        for shift,mode,label,color in [(-.18,'clean','Normal','#2463eb'),(.18,'bias8','Depth +8%','#ea580c')]:
            vals=[100*next(r['success'] for r in rows if r['method']==m and r['mode']==mode and r['material']==material) for m in methods]
            bars=ax.bar(np.arange(5)+shift,vals,.34,label=label,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=8)
        ax.set_xticks(np.arange(5),names,rotation=20,ha='right');ax.set_title(material);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axs[0].set_ylabel('Correct / all valid liquid pixels (%)');axs[1].legend()
    fig.suptitle('V16 | full-frame sensor proxy | 64 new scenes per material');fig.tight_layout();fig.savefig(OUT/'V16-comparison.png',dpi=160)
    print('SUMMARIZED',len(rows),len(comparisons))
if __name__=='__main__':main()
