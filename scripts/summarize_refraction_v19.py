#!/usr/bin/env python3
"""Assemble transparent comparisons with conditional error and rejection coverage."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v19')
D=Path('/root/autodl-tmp/liquid-depth-data/synthetic/stratified-v19-heldout')
runs=['heldout-single','heldout-16','heldout-resampled16','heldout-64','heldout-baselines','heldout-evidence','heldout-stress-evidence']
rows=[]
for run in runs:
 for row in json.loads((O/run/'summary.json').read_text()):rows.append(dict(run=run,**row))
(O/'V19-summary.json').write_text(json.dumps(rows,indent=2))
keys=sorted(set().union(*(r.keys() for r in rows)))
with (O/'V19-results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
selected=[]
for run,method in [('heldout-baselines','seegroup_observed_calibration'),('heldout-baselines','v9_1_temporal'),('heldout-64','refraction_rgbd'),('heldout-evidence','chromatic_refraction_evidence')]:
 selected.append(next(r for r in rows if r['run']==run and r['method']==method and r['group']=='strong'))
labels=['SeeGroup + calibration','V9.1 + temporal median','Inverse optics: geometry','Inverse optics: evidence']
colors=['#9aabb8','#bdc8cf','#327ca8','#298767']
fig,ax=plt.subplots(1,3,figsize=(15,4.8),gridspec_kw={'width_ratios':[1.3,1,1]})
y=np.arange(4)
ax[0].barh(y,[r['thickness_mae_mm'] for r in selected],color=colors);ax[0].set_yticks(y,labels);ax[0].invert_yaxis();ax[0].set_xlabel('Conditional thickness MAE (mm)');ax[0].set_xlim(0,29)
for i,r in enumerate(selected):ax[0].text(r['thickness_mae_mm']+.4,i,f"{r['thickness_mae_mm']:.2f}",va='center')
ax[1].barh(y,[r['coverage_pct'] for r in selected],color=colors);ax[1].set_yticks(y,[]);ax[1].invert_yaxis();ax[1].set_xlabel('Output coverage (%)');ax[1].set_xlim(0,119)
for i,r in enumerate(selected):ax[1].text(r['coverage_pct']+1,i,f"{r['accepted']}/24",va='center')
ax[2].bar(['Geometry','Evidence'],[21,0],color=['#327ca8','#298767']);ax[2].set_ylim(0,26);ax[2].set_ylabel('False accepts / 24 ghost controls')
ax[2].text(0,21.7,'21/24',ha='center');ax[2].text(1,.7,'0/24',ha='center')
fig.suptitle('V19 held-out screening: lower error does not guarantee interface identity',fontsize=13)
fig.text(.5,.02,'Fixed overhead camera + acquired empty reference. Strong-index subset: 24 cases. Simplified ToF proxy; no real-camera accuracy claim.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.07,1,.94]);fig.savefig(O/'V19-comparison.png',dpi=150);plt.close(fig)
sheet=Image.new('RGB',(960,430),'white');draw=ImageDraw.Draw(sheet)
for col,(i,label) in enumerate([(95,'Evidence supported'),(15,'Strong / ambiguous example'),(14,'Weak interface')]):
 for row,name in enumerate(['empty_rgb.png','rgb.png']):
  im=Image.open(D/f'{i:03d}'/name).convert('RGB');sheet.paste(im,(col*320,row*210+25))
  draw.text((col*320+5,row*210+7),f'{label}: '+('empty reference' if row==0 else 'liquid observation'),fill='black')
sheet.save(O/'V19-reference-examples.png')
print('REPORT_ASSETS_FINISHED')
