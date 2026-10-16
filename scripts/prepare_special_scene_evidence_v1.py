"""Fill fixed paired quotas by input evidence only, never by prediction error."""
import argparse
import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path
import numpy as np
from liquid_depth.simulation import SyntheticScene, render_geometric_labels, simulate_raw_depth
from special_scene_input_evidence_v1 import assess, assess_capture_origin, POLICY
from special_scene_protocol_v1 import SEED, REPEATS, SCENES, LEVELS, SENSORS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--count',type=int,default=66)
    a=p.parse_args()
    lock={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          str(Path(__file__).with_name('special_scene_input_evidence_v1.py')):hashlib.sha256(Path(__file__).with_name('special_scene_input_evidence_v1.py').read_bytes()).hexdigest()}
    (a.out/'eligibility-policy-before-selection.json').write_text(json.dumps({'policy':POLICY,'max_pair_attempts_per_condition':64,
        'additional_offline_capture_audit':'64 direct liquid-origin returns (floor for empty controls),6of16 visible surface tiles,25% two-axis span; no numeric accuracy criterion',
        'target_accepted_pairs_per_condition':4,'no_prediction_or_truth_error_available_to_selector':True,
        'source_sha256':lock},indent=2))
    selections={};attempts=[]
    scene_keys={f.name for f in fields(SyntheticScene)}
    for gid in range(a.count):
        cals=[json.loads((a.out/'calibrations'/f'{gid*2+si:05d}.json').read_text()) for si in range(2)]
        if not all(c['success'] for c in cals):
            raise ValueError('fix_invalid_empty_calibration_before_capture_selection')
        for scenario in SCENES:
            for level in LEVELS:
                folder=a.root/'observations'/f'{gid:04d}'/f'{scenario}_{level}'
                meta=json.loads((folder/'metadata.json').read_text())
                scene=SyntheticScene(**{k:meta[k] for k in scene_keys})
                with np.load(folder/'labels_offline.npz') as data:
                    semantic=data['semantic']
                accepted=[];labels=None;geometric=None;visible=None
                for rep in range(64):
                    if not (folder/f'{SENSORS[1]}_raw_{rep:02d}.npy').exists():
                        if labels is None:
                            labels=render_geometric_labels(scene)
                            with np.load(folder/'labels_offline.npz') as f:
                                geometric=f['true_geometric_full'];visible=f['semantic']==2
                        for sensor in SENSORS:
                            rng=np.random.default_rng(SEED+gid*130363+rep*79907)
                            ss=replace(scene,sensor_family=sensor,seed=scene.seed+rep*79907)
                            sim=simulate_raw_depth(ss,labels)
                            raw=np.where(visible,sim['raw_depth_m'],geometric+rng.normal(0,.001+.0007*geometric))
                            raw[(rng.random(raw.shape)<.04)|(geometric<=0)]=0
                            raw=np.where(np.isfinite(raw)&(raw>0)&(raw<=10.5),raw,0).astype(np.float32)
                            np.save(folder/f'{sensor}_raw_{rep:02d}.npy',raw)
                            np.save(folder/f'{sensor}_error_offline_{rep:02d}.npy',sim['simulated_error_type'])
                    evidence={sensor:assess(np.load(folder/f'{sensor}_raw_{rep:02d}.npy'),cal['bundle'])
                              for sensor,cal in zip(SENSORS,cals)}
                    for sensor in SENSORS:
                        capture=assess_capture_origin(np.load(folder/f'{sensor}_raw_{rep:02d}.npy'),semantic,
                            np.load(folder/f'{sensor}_error_offline_{rep:02d}.npy'),
                            contains_liquid=meta['nominal_liquid_height_m']>0)
                        evidence[sensor]['capture_origin']=capture
                        evidence[sensor]['eligible']&=capture['eligible']
                        evidence[sensor]['reasons']+=capture['reasons']
                    eligible=all(e['eligible'] for e in evidence.values())
                    attempts.append(dict(physical_id=gid,scenario=scenario,level=level,repeat=rep,
                                         paired_eligible=eligible,evidence=evidence))
                    if eligible:accepted.append(rep)
                    if len(accepted)==REPEATS:break
                if len(accepted)!=REPEATS:
                    raise ValueError(f'quota_not_filled_without_weakening_input_policy:{gid}:{scenario}:{level}')
                selections[f'{gid}|{scenario}|{level}']=accepted
        print('EVIDENCE_QUOTAS',gid,flush=True)
    (a.out/'selected-input-repeats.json').write_text(json.dumps(selections,indent=2))
    (a.out/'capture-eligibility-attempts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in attempts))
    audit={'attempted_pairs':len(attempts),'selected_pairs':sum(len(v) for v in selections.values()),
           'rejected_pairs':sum(not r['paired_eligible'] for r in attempts),
           'paired_input_acquisition_success_pct':100*sum(r['paired_eligible'] for r in attempts)/len(attempts),
           'individual_input_evidence':{sensor:{'qualified_frames':sum(r['evidence'][sensor]['eligible'] for r in attempts),
             'excluded_frames':sum(not r['evidence'][sensor]['eligible'] for r in attempts)} for sensor in SENSORS},
           'selection_uses_online_geometry_support_plus_offline_capture_source_audit':True,
           'selection_uses_no_numeric_error_or_estimate':True,
           'stratum_quota_uniform_after_input_qualification':True}
    (a.out/'capture-eligibility-summary.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
