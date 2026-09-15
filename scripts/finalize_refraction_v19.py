#!/usr/bin/env python3
"""Finalize all frozen runs, emphasizing the final acquisition-qualified validation."""
from pathlib import Path
import csv,json,shutil
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from liquid_depth.refraction_calibration_v19 import assess_reference

O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v19')
ROOT=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
stages=['heldout-single','heldout-16','heldout-resampled16','heldout-64','heldout-baselines','heldout-evidence','heldout-stress-evidence']
stages += [prefix+suffix for prefix in ['validationB','validationC'] for suffix in ['-64','-baselines','-evidence','-stress-evidence']]
rows=[]
for stage in stages:
 for row in json.loads((O/stage/'summary.json').read_text()):rows.append(dict(run=stage,**row))
def record(run,method,group='strong'):
 return next(r for r in rows if r['run']==run and r['method']==method and r['group']==group)
def fmt(v):
 return '—' if v is None else f'{v:.2f}'
selected=[record('validationC-baselines','seegroup_observed_calibration'),record('validationC-baselines','v9_1_temporal'),record('validationC-64','refraction_rgbd'),record('validationC-evidence','chromatic_refraction_evidence')]
names=['SeeGroup＋可观测标定','V9.1＋64帧深度中位数','新折射模型：几何门限','新折射模型：默认完整检查']
table='| 方法 | 输出场景 | 覆盖率 | 层厚 MAE | P95 | 双界面通过率 |\n|---|---:|---:|---:|---:|---:|\n'
for name,r in zip(names,selected):
 table+=f"| {name} | {r['accepted']}/{r['scenes']} | {r['coverage_pct']:.2f}% | {fmt(r['thickness_mae_mm'])} mm | {fmt(r['thickness_p95_mm'])} mm | {r['both5mm_all_scenes_pct']:.2f}% |\n"
batches='| 批次 | 几何输出 / 24 | 几何层厚 MAE | 证据输出 / 24 | 证据层厚 MAE |\n|---|---:|---:|---:|---:|\n'
for prefix,label in [('heldout','A：原始曝光'),('validationB','B：固定降低两档'),('validationC','C：标定自动曝光')]:
 g=record(prefix+'-64','refraction_rgbd');e=record(prefix+'-evidence','chromatic_refraction_evidence')
 batches+=f"| {label} | {g['accepted']}/24 | {fmt(g['thickness_mae_mm'])} mm | {e['accepted']}/24 | {fmt(e['thickness_mae_mm'])} mm |\n"
quality=[];qtable='| 批次 | 参考饱和比例均值 | 合格参考 / 96 | 曝光档位 |\n|---|---:|---:|---|\n'
for dataset,label in [('stratified-v19-heldout','A'),('stratified-v19-validationB','B'),('stratified-v19-validationC','C')]:
 D=ROOT/dataset;measurements=[];exposures=[]
 for i in range(96):
  folder=D/f'{i:03d}';cam=json.loads((folder/'camera.json').read_text());im=cv2.cvtColor(cv2.imread(str(folder/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  q=assess_reference(im,np.load(folder/'observation.npz')['empty_raw'],cam);measurements.append(dict(id=i,**q))
  exposures.append(json.loads((folder/'scene.json').read_text()).get('exposure',0.))
 mean=float(np.mean([q['saturated_fraction'] for q in measurements]));good=sum(q['usable'] for q in measurements)
 summary=dict(dataset=dataset,label=label,mean_saturation=mean,usable=good,exposures=sorted(set(exposures)),frames=measurements)
 quality.append(summary);qtable+=f"| {label} | {mean*100:.2f}% | {good}/96 | {sorted(set(exposures))} EV |\n"
(O/'reference-quality-audit.json').write_text(json.dumps(quality,indent=2))
ghost=json.loads((O/'validationC-stress.json').read_text());ghost_count=sum(r['accepted'] for r in ghost if r['case']=='stable_ghost_in_single_liquid')
ghost_guard=record('validationC-stress-evidence','chromatic_refraction_evidence','stable_ghost_in_single_liquid')['accepted']
weak=record('validationC-evidence','chromatic_refraction_evidence','weak')['accepted']
template=(O/'final-report-template.md').read_text()
for key,value in dict(FINAL_TABLE=table,EXPOSURE_TABLE=qtable,BATCH_TABLE=batches,COVERAGE=f"{selected[-1]['coverage_pct']:.2f}%",WEAK_COUNT=weak,GHOST_GEOMETRY=ghost_count,GHOST_EVIDENCE=ghost_guard,QUALITY_OK=quality[-1]['usable']).items():
 template=template.replace('{{'+key+'}}',str(value))
assert '{{' not in template
(O/'V19-report.md').write_text(template)
Path('docs/refraction_v19.md').write_text(template.replace('](V19-final-comparison.png)','](refraction_v19_comparison.png)').replace('](V19-final-reference-examples.png)','](refraction_v19_references.png)'))
(O/'V19-summary.json').write_text(json.dumps(rows,indent=2))
keys=sorted(set().union(*(r.keys() for r in rows)))
with (O/'V19-results.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1.3,1,1]})
labels=['SeeGroup + calibration','V9.1 + temporal median','Inverse optics: geometry','Inverse optics: full checks'];colors=['#9aabb8','#bdc8cf','#327ca8','#298767'];y=np.arange(4)
maes=[r['thickness_mae_mm'] if r['thickness_mae_mm'] is not None else 0 for r in selected]
axes[0].barh(y,maes,color=colors);axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Conditional thickness MAE (mm)');axes[0].set_xlim(0,max(maes)*1.2+1)
for i,(r,v) in enumerate(zip(selected,maes)):axes[0].text(v+.4,i,'N/A' if r['thickness_mae_mm'] is None else f'{v:.2f}',va='center')
axes[1].barh(y,[r['coverage_pct'] for r in selected],color=colors);axes[1].set_yticks(y,[]);axes[1].invert_yaxis();axes[1].set_xlabel('Output coverage (%)');axes[1].set_xlim(0,119)
for i,r in enumerate(selected):axes[1].text(r['coverage_pct']+1,i,f"{r['accepted']}/24",va='center')
axes[2].bar(['Geometry','Full checks'],[ghost_count,ghost_guard],color=colors[-2:]);axes[2].set_ylim(0,27);axes[2].set_ylabel('False accepts / 24 ghost controls')
for i,n in enumerate([ghost_count,ghost_guard]):axes[2].text(i,n+.7,f'{n}/24',ha='center')
fig.suptitle('V19 final validation C: fixed camera, acquired reference, exposure quality checks',fontsize=13)
fig.text(.5,.02,'24 strong-index cases. Error is conditional on output. Simplified ToF proxy; no real-camera accuracy claim.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(O/'V19-final-comparison.png',dpi=150);plt.close(fig)
D=ROOT/'stratified-v19-validationC';results=json.loads((O/'validationC-evidence/results.json').read_text())
accepted=[r['id'] for r in results if r['accepted'] and r['group']=='strong'];uncertain=[r['id'] for r in results if not r['accepted'] and r['group']=='strong']
examples=[(accepted[0],'Evidence supported') if accepted else (3,'Strong / no confirmed output'),(uncertain[0],'Strong / insufficient evidence'),(14,'Weak interface')]
sheet=Image.new('RGB',(960,430),'white');draw=ImageDraw.Draw(sheet)
for col,(i,label) in enumerate(examples):
 for row,name in enumerate(['empty_rgb.png','rgb.png']):
  sheet.paste(Image.open(D/f'{i:03d}'/name).convert('RGB'),(col*320,row*210+25))
  draw.text((col*320+5,row*210+7),label+': '+('reference' if row==0 else 'liquid'),fill='black')
sheet.save(O/'V19-final-reference-examples.png')
shutil.copy2(O/'V19-final-comparison.png','docs/refraction_v19_comparison.png');shutil.copy2(O/'V19-final-reference-examples.png','docs/refraction_v19_references.png')
(O/'final-headline.json').write_text(json.dumps(dict(selected=selected,ghost_geometry=ghost_count,ghost_evidence=ghost_guard,weak_accepted=weak,quality_usable=quality[-1]['usable'],example_ids=[i for i,_ in examples]),indent=2))
print('FINAL_REPORT_READY',json.dumps(dict(strong=selected[-1],ghost_geometry=ghost_count,ghost_evidence=ghost_guard,quality=quality[-1]['usable'])),flush=True)
