#!/usr/bin/env python3
"""Summarize locked V15 experiments without tuning on fresh test labels."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from summarize_depth_screen_v14 import compare

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);a=ap.parse_args();root=Path(a.run_dir)
 result={};rows=[];paired={}
 folders={'pretrained':root/'pretrained','clean':root/'clean','robust':root/'robust','plane':root/'plane','strong':root/'strong/robust','context-strong':root/'context-strong/robust','context-plane':root/'context-plane/plane'}
 for arm,folder in folders.items():
  if not (folder/'frozen.json').exists():continue
  frozen=json.loads((folder/'frozen.json').read_text());result[arm]={'frozen':frozen,'conditions':{}}
  for p in folder.glob('fresh-*.json'):
   r=json.loads(p.read_text());mode=r['mode'];result[arm]['conditions'][mode]=r['summary']
   for scene,v in r['summary'].items():
    base=v['all'];t=frozen['threshold'];gate=v['selected'][str(t)] if t is not None else None
    row=dict(arm=arm,condition=mode,scene=scene,epoch=frozen['best_epoch'],threshold=t,qualified=frozen['qualified_on_operating_fold'],full_mae_mm=base['mae_mm'],full_success=base['success'],full_coverage=base['coverage'],ece=v['ece'],brier=v['brier'],selected_precision=gate['precision'] if gate else None,selected_coverage=gate['coverage'] if gate else None,selected_success=gate['success'] if gate else None,selected_non_target=gate['non_target_fraction'] if gate else None)
    diag=v['selected']['0.5'];row.update(diagnostic_precision_at_05=diag['precision'],diagnostic_coverage_at_05=diag['coverage'],diagnostic_success_at_05=diag['success'])
    rows.append(row)
   if arm in ('clean','robust','strong','context-strong'):
    baseline=root/'pretrained'/p.name
    if baseline.exists():paired.setdefault(arm,{})[mode]={s:compare(r,json.loads(baseline.read_text()),s) for s in ('transparent','translucent')}
 with (root/'results.csv').open('w',newline='') as f:
  if rows:
   w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (root/'summary.json').write_text(json.dumps(result,indent=2));(root/'paired-bootstrap.json').write_text(json.dumps(paired,indent=2))
 for row in rows:
  if row['scene'] in ('transparent','translucent'):print(json.dumps(row))
if __name__=='__main__':main()
