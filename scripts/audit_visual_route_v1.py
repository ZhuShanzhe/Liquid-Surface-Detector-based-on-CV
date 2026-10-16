"""Summarize frozen results without changing inference or original records."""
import hashlib
import json
import tarfile
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
from evaluate_visual_route_v1 import metrics

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
ROOT=BASE/'liquid-depth-artifacts/visual-route-v1-20261015'


def main():
    result={}
    for name in ['final-model-v1','final-balanced']:
        path=ROOT/name
        rows=[json.loads(s) for s in (path/'records.jsonl').read_text().splitlines()]
        groups=defaultdict(list)
        for r in rows:
            far=r['camera_to_bottom_m']>=6 and r['truth']['liquid_height_m']<=1
            for subset in ['all','far_shallow' if far else 'remaining']:
                for eligibility in ['all_frames','precision_eligible' if r['common_precision_eligible'] else 'excluded_evidence']:
                    groups[f"{r['scene']}|{subset}|{eligibility}|{r['method']}"].append(r)
        scores={k:metrics(v) for k,v in groups.items()}
        (path/'remaining-far-shallow-summary.json').write_text(json.dumps(scores,indent=2))
        visual=[r for r in rows if r['method']=='visual_guided']
        times=np.array([r['algorithm_ms'] for r in visual])
        reasons=Counter(reason for r in visual if not r['common_precision_eligible'] for reason in r['exclusion_reasons'])
        result[name]={'unique_frames':len(visual),'precision_eligible':sum(r['common_precision_eligible'] for r in visual),
                     'exclusions':dict(reasons),'numeric':sum(r['liquid_depth_m'] is not None for r in visual),
                     'algorithm_latency_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'max':float(times.max())},
                     'mean_and_highest_output_are_not_float_detection_accuracy':True,
                     'float_components_detected_total':sum(r['diagnostics'].get('float_components',0) for r in visual)}
    archive=BASE/'liquid-depth-archives/pre-visual-route-20261015/repository.tar'
    checked=0;mismatches=[]
    with tarfile.open(archive) as tar:
        for member in tar:
            relative=Path(member.name).relative_to(REPO.name)
            if not member.isfile() or not (str(relative).startswith(('src/','scripts/','tests/','docs/','configs/')) or str(relative)=='README.md'):
                continue
            if relative.suffix not in {'.py','.md','.json','.yaml','.yml','.toml','.sh'}:continue
            current=REPO/relative
            old=tar.extractfile(member).read()
            checked+=1
            if not current.exists() or hashlib.sha256(old).digest()!=hashlib.sha256(current.read_bytes()).digest():mismatches.append(str(relative))
    result['archived_source_integrity']={'checked_files':checked,'changed_original_files':mismatches}
    frozen=json.loads((ROOT/'final-frozen-protocol.json').read_text())['source_sha256']
    result['final_frozen_source_matches_current']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen.items()}
    (ROOT/'audit-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    for name in ['final-model-v1']:
        s=json.loads((ROOT/name/'remaining-far-shallow-summary.json').read_text())
        for k,v in s.items():
            if '|remaining|all_frames|' in k:print(k,v['frames'],round(v['mae_mm'],2),round(v['p95_mm'],2),round(v['pass_pct'],2))


if __name__=='__main__':main()
