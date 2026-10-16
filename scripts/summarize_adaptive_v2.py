"""Deployment selection and paired diagnostic summary, without test-based policy tuning."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'


def main():
    frozen=json.loads((OUT/'confirmation-frozen-protocol.json').read_text())
    selected=frozen['validation_selected_default'];gate='candidate-model-with-real' if selected.endswith('with-real') else 'candidate-model'
    choice={'selection_basis':'validation semantic score, not test pass rate','selected_pixel_model':selected,
        'pixel_checkpoint':str(OUT/selected/'best.pt'),'candidate_checkpoint':str(OUT/gate/'gate.pt'),
        'default_visual_mode':'off','research_visual_mode':'auto',
        'recommended_trial_context':'known floating-object monitoring only; small-target recall and false alarms still need work',
        'recommendation_status':'exploratory observations, not a separately qualified scene-selection policy',
        'geometric_backup_manifest':str(BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json'),
        'scene_boundary':{'transparent_translucent':'preserved five-prior refraction; auxiliary bypassed','other_scenes':'operator-selectable off/on/auto'},
        'research_only':True}
    (OUT/'deployment-choice.json').write_text(json.dumps(choice,indent=2))
    total={}
    for name in ['confirmation-synthetic','confirmation-with-real']:
        p=OUT/name;rows=[json.loads(s) for s in (p/'records.jsonl').read_text().splitlines()]
        lookup={(r['case'],r['repeat'],r['mode']):r for r in rows};summary={}
        for mode in ['on','auto']:
            groups=defaultdict(list)
            for r in rows:
                if r['mode']==mode:groups[r['scene']].append(r)
            for scene,rs in groups.items():
                delta=[];switches=0;improvements=0;harm=0;peak_delta=[];false_float=0;negative=0
                for r in rs:
                    old=lookup[(r['case'],r['repeat'],'off')];peak=lookup[(r['case'],r['repeat'],'geometric_peak_control')]
                    h=r['truth']['liquid_height_m'];top=r['truth']['highest_height_m']
                    if r['liquid_depth_m'] is not None and old['liquid_depth_m'] is not None:
                        d=abs(r['liquid_depth_m']-h)-abs(old['liquid_depth_m']-h);delta.append(d)
                        switches+=r['liquid_override_used'];improvements+=d<-.005;harm+=d>.005
                    if r['highest_height_m'] is not None and peak['highest_height_m'] is not None:
                        peak_delta.append(abs(r['highest_height_m']-top)-abs(peak['highest_height_m']-top))
                    if r['floating_pixels_offline']==0:
                        negative+=1;false_float+=bool(r['diagnostics'].get('float_components'))
                summary[f'{scene}|{mode}']={'frames':len(rs),'liquid_overrides':switches,
                    'liquid_error_improved_over_5mm':int(improvements),'liquid_error_worsened_over_5mm':int(harm),
                    'mean_liquid_error_change_mm':float(1000*np.mean(delta)) if delta else None,
                    'mean_peak_error_change_vs_legacy_geometry_mm':float(1000*np.mean(peak_delta)) if peak_delta else None,
                    'no_visible_float_frames':negative,'false_float_use_frames':false_float}
        total[name]={'paired_effects':summary,'pixel':json.loads((p/'pixel-and-reliability.json').read_text())}
    manifest=json.loads((BASE/'liquid-depth-archives/determined-symmetric-geometry-v1-20261015/manifest.json').read_text())
    total['frozen_geometric_integrity']={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()==h for p,h in manifest['source_sha256'].items()}
    total['confirmation_source_integrity']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items()}
    (OUT/'final-audit-summary.json').write_text(json.dumps(total,indent=2))
    print('CHOICE',json.dumps(choice))
    for name in ['confirmation-synthetic','confirmation-with-real']:
        data=total[name]['pixel'];print(name,'IOU',data['coupled_semantics']['iou'],'RGB_ONLY',data['rgb_only_input_ablation']['iou'],
            'OBJECTS',data.get('object_precision'),'RECALL',data['coupled_object_recall_by_size'],'RELIABILITY',data['reliability'])
        s=json.loads((OUT/name/'summary.json').read_text());peak=json.loads((OUT/name/'highest-summary.json').read_text())
        for k,v in s.items():
            if '|all|remaining|' in k and k.endswith(('|off','|auto')):
                print('LIQUID',k,v['frames'],v['mae_mm'],v['p95_mm'],v['pass_pct'])
        for k,v in peak.items():
            if '|all|remaining|' in k and k.endswith(('|geometric_peak_control','|auto')):
                print('PEAK',k,v['frames'],v['mae_mm'],v['p95_mm'],v['pass_pct'])


if __name__=='__main__':main()
