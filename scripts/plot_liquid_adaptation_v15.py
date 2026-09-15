import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();r=Path(a.run_dir)
models=[('pretrained','DFNet baseline'),('robust','DFNet robust'),('strong/robust','DFNet strong'),('v12_reference','V12 reference'),('plane','V12 + plane')]
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharey=True)
for ax,scene in zip(axes,['transparent','translucent']):
 x=np.arange(len(models));width=.36
 for off,mode,label,color in [(-.5,'clean','Normal input','#2563eb'),(.5,'bias8','Input depth +8%','#ea580c')]:
  vals=[100*json.loads((r/p/f'fresh-{mode}.json').read_text())['summary'][scene]['all']['success'] for p,_ in models]
  bars=ax.bar(x+off*width,vals,width,label=label,color=color)
  for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}',ha='center',fontsize=8)
 ax.set_xticks(x,[n for _,n in models],rotation=25,ha='right');ax.set_ylim(0,85);ax.set_title(scene.title()+' liquid');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Correct / all valid liquid pixels (%)');axes[1].legend(loc='upper right')
fig.suptitle('V15 | 64 independent scenes per material | ROI-only synthetic depth')
fig.savefig(r/'V15-comparison.png',dpi=180)
