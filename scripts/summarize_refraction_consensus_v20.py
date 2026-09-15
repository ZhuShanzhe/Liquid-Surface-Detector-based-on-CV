"""Summarize frozen held-out experiments; all labels remain evaluation-only."""
import json,csv,math,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest
O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v20')
D=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
def load(p):return json.loads(p.read_text())
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n
 mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [100*(mid-half),100*(mid+half)]
def summarize(rows):
 acc=[r for r in rows if r['accepted']]
 e=[r['thickness_error_mm'] for r in acc if r.get('thickness_error_mm') is not None]
 return dict(scenes=len(rows),accepted=len(acc),coverage_pct=100*len(acc)/len(rows),coverage_wilson95_pct=wilson(len(acc),len(rows)),
  thickness_mae_mm=float(np.mean(e)) if e else None,thickness_p95_mm=float(np.percentile(e,95)) if e else None,
  thickness_max_mm=max(e) if e else None,top_mae_mm=float(np.mean([r['top_error_mm'] for r in acc])) if acc else None,
  both5mm_all_scenes_pct=100*sum(r.get('both5mm',False) for r in rows)/len(rows),
  statuses=dict(Counter(r['status'] for r in rows)))
def main():
 rows=[]
 for tag in ['heldoutC','heldoutD']:
  for version in ['v19','v20']:
   rows += [dict(r,split=tag,version=version) for r in load(O/(tag+'-'+version)/'results.json')]
 summaries=[]
 for split in ['heldoutC','heldoutD','pooled']:
  for v in ['v19','v20']:
   for g in ['strong','weak','single','index_matched']:
    rs=[r for r in rows if r['version']==v and r['group']==g and (split=='pooled' or r['split']==split)]
    summaries.append(dict(split=split,version=v,group=g,**summarize(rs)))
 old={(r['split'],r['id']):r for r in rows if r['version']=='v19' and r['group']=='strong'}
 new={(r['split'],r['id']):r for r in rows if r['version']=='v20' and r['group']=='strong'}
 gain=[list(k) for k in old if new[k]['accepted'] and not old[k]['accepted']]
 loss=[list(k) for k in old if old[k]['accepted'] and not new[k]['accepted']]
 common=[k for k in old if old[k]['accepted'] and new[k]['accepted']]
 paired=dict(gained=gain,lost=loss,common=len(common),
  exact_mcnemar_p=float(binomtest(len(gain),len(gain)+len(loss),.5).pvalue) if gain or loss else 1.,
  common_v19_mae_mm=float(np.mean([old[k]['thickness_error_mm'] for k in common])) if common else None,
  common_v20_mae_mm=float(np.mean([new[k]['thickness_error_mm'] for k in common])) if common else None)
 ghost=[]
 for tag in ['heldoutC','heldoutD']:
  for v in ['v19','v20']:
   rs=load(O/(tag+'-v19-stress')/'results.json') if v=='v19' else [r['result'] for r in load(O/(tag+'-v20-stress.json')) if r['case']=='stable_ghost_in_single_liquid']
   ghost.append(dict(split=tag,version=v,scenes=len(rs),false_accepts=sum(r['accepted'] for r in rs)))
 breakdown=[]
 for pattern in [False,True]:
  rs=[r for r in rows if r['version']=='v20' and r['group']=='strong' and r['pattern']==pattern]
  breakdown.append(dict(pattern=pattern,**summarize(rs)))
 for r in rows:r.pop('profile',None)
 false_sets={v:set() for v in ['v19','v20']}
 for tag in ['heldoutC','heldoutD']:
  for r in load(O/(tag+'-v19-stress')/'results.json'):
   if r['accepted']:false_sets['v19'].add((tag,r['id']))
  for r in load(O/(tag+'-v20-stress.json')):
   if r['case']=='stable_ghost_in_single_liquid' and r['accepted']:false_sets['v20'].add((tag,r['id']))
 false_changes=dict(new=sorted(false_sets['v20']-false_sets['v19']),retained=sorted(false_sets['v20']&false_sets['v19']),removed=sorted(false_sets['v19']-false_sets['v20']))
 selection_counts=dict(Counter(r.get('selection','legacy') for r in rows if r['version']=='v20' and r['accepted']))
 payload=dict(false_acceptance_changes=false_changes,selection_counts=selection_counts,summaries=summaries,paired=paired,ghost=ghost,pattern_breakdown=breakdown,
  frozen=load(O/'frozen-consensus-before-heldout.json'),scope='two transparent liquid layers under a single-path ToF-like simulation proxy')
 (O/'V20-summary.json').write_text(json.dumps(payload,indent=2))
 (O/'V20-case-results.json').write_text(json.dumps(rows,indent=2))
 fields=['split','version','group','id','accepted','status','pattern','top_error_mm','thickness_error_mm','internal_error_mm','both5mm','candidate_thickness_m','profile_width_m','contrast_lower_bound']
 with (O/'V20-results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
 # Scientific figure; accepted-only errors and all-scene denominators are separate.
 fig,axs=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
 labels=['Strong contrast','Weak contrast'];colors=['#8c9aa8','#117a70']
 for j,v in enumerate(['v19','v20']):
  ss=[next(s for s in summaries if s['split']=='pooled' and s['version']==v and s['group']==g) for g in ['strong','weak']]
  xx=np.arange(2)+(j-.5)*.32
  axs[0].bar(xx,[s['coverage_pct'] for s in ss],width=.30,color=colors[j],label='V19' if v=='v19' else 'V20.1')
  for x,s in zip(xx,ss):axs[0].text(x,s['coverage_pct']+2,f"{s['accepted']}/{s['scenes']}",ha='center',fontsize=10)
 axs[0].set(xticks=np.arange(2),xticklabels=labels,ylim=(0,100),ylabel='Evidence-qualified output (%)',title='Coverage on 192 new scenes');axs[0].legend()
 for j,v in enumerate(['v19','v20']):
  s=next(s for s in summaries if s['split']=='pooled' and s['version']==v and s['group']=='strong')
  vals=[s['thickness_mae_mm'] or 0,s['thickness_p95_mm'] or 0]
  xx=np.arange(2)+(j-.5)*.32;axs[1].bar(xx,vals,width=.30,color=colors[j],label='V19' if v=='v19' else 'V20.1')
  for x,y in zip(xx,vals):axs[1].text(x,y+.06,f'{y:.2f}',ha='center',fontsize=10)
 axs[1].set(xticks=np.arange(2),xticklabels=['MAE','P95'],ylabel='Upper-layer thickness error (mm)',title='Accepted strong-contrast cases')
 axs[1].set_ylim(0,max(3,axs[1].get_ylim()[1]*1.2))
 for j,v in enumerate(['v19','v20']):
  gs=[g for g in ghost if g['version']==v];k=sum(g['false_accepts'] for g in gs);n=sum(g['scenes'] for g in gs)
  axs[2].bar(j,100*k/n,color=colors[j],width=.5);axs[2].text(j,100*k/n+1,f'{k}/{n}',ha='center')
 axs[2].set(xticks=[0,1],xticklabels=['V19','V20.1'],ylim=(0,20),ylabel='False acceptance (%)',title='Stable delayed-return counterexamples')
 fig.suptitle('Fixed camera + one empty reference + 64 depth frames | simulation only',fontsize=13)
 fig.savefig(O/'V20-comparison.png',dpi=180);plt.close(fig)
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
