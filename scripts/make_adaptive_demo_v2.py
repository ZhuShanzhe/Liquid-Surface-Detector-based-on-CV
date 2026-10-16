"""Build reproducible installed input artifacts for a fixed no-USB smoke case."""
import json
import subprocess
import os
from pathlib import Path

from evaluate_shared_empty_v45 import install

BASE=Path('/root/autodl-tmp');REPO=BASE/'Liquid-Surface-Detector-based-on-CV'
ROOT=BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-confirmation-20261015'
OUT=BASE/'liquid-depth-artifacts/adaptive-route-v2-20261015'


def main():
    folder=ROOT/'00633';item=install(633,ROOT,ROOT/'dry',202610154229,max_annotation_attempts=6,source_slot_stride=1)
    demo=OUT/'no-usb-example';demo.mkdir(exist_ok=False)
    (demo/'calibration.json').write_text(json.dumps(item['bundle'],indent=2))
    frame=json.loads((folder/'frames.json').read_text())[0]
    env={**os.environ,'PYTHONPATH':str(REPO/'src')+':'+str(REPO/'scripts')}
    for mode in ['off','auto']:
        cmd=[str(BASE/'envs/liquid-depth/bin/python'),'scripts/measure_adaptive_v2.py',
            '--bundle',str(demo/'calibration.json'),'--dry',str(ROOT/'dry/0633/dry_depth_m.npy'),
            '--depth',frame['raw_path'],'--rgb',str(folder/'rgb.png'),'--visual',mode,
            '--profile',str(OUT/'deployment-choice.json'),'--scene','floating_objects']
        output=subprocess.check_output(cmd,cwd=REPO,env=env,text=True)
        record=json.loads(output);(demo/f'{mode}.json').write_text(json.dumps(record,indent=2))
        print(mode,record['liquid_depth_m'],record['highest_height_m'],record['visual_guidance_used'])


if __name__=='__main__':main()
