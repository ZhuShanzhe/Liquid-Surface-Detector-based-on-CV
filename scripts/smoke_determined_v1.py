"""Exercise all eight public scene entries without a physical USB camera."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from evaluate_shared_empty_v45 import install

BASE = Path('/root/autodl-tmp')
REPO = BASE/'Liquid-Surface-Detector-based-on-CV'
OUT = BASE/'liquid-depth-artifacts/determined-all-scenes-v1-20261016'
ROOT = BASE/'liquid-depth-data/synthetic/adaptive-evidence-v2-confirmation-20261015'


def call(folder, args):
    command = [sys.executable, 'scripts/measure_determined.py']+args
    proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
    result = json.loads(proc.stdout)
    (folder/'command.json').write_text(json.dumps(command, indent=2))
    (folder/'output.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    rows = [json.loads(line) for line in (OUT/'opaque-records.jsonl').read_text().splitlines()]
    output = OUT/'no-usb-example'
    output.mkdir(exist_ok=False)
    checks = []
    for scene in sorted({r['scene'] for r in rows}):
        row = next(r for r in rows if r['scene'] == scene)
        case = row['case']
        folder = output/scene
        folder.mkdir()
        item = install(case, ROOT, ROOT/'dry', 202610154229, max_annotation_attempts=6, source_slot_stride=1)
        (folder/'bundle.json').write_text(json.dumps(item['bundle'], indent=2))
        np.save(folder/'dry.npy', item['dry'])
        (folder/'initialization.json').write_text(json.dumps({'kind': 'empty_geometry', 'bundle': 'bundle.json',
                                                           'dry_depth': 'dry.npy', 'max_return_depth_m': 10.5}, indent=2))
        raw_folder = ROOT/f'{case:05d}'
        frame = next(f for f in json.loads((raw_folder/'frames.json').read_text()) if f['repeat'] == row['repeat'])
        common = ['--scene', scene, '--initialization', str(folder/'initialization.json'), '--depth', frame['raw_path']]
        # Passing opt-in to all scenes also verifies that non-floating routes never execute vision.
        args = common+['--enable-floating-peak-vision']
        if scene == 'floating_objects':
            args += ['--rgb', str(raw_folder/'rgb.png')]
        result = call(folder, args)
        assert result['liquid_depth_m'] == row['liquid_depth_m']
        assert abs(result['highest_height_m']-row['highest_height_m']) < 1e-7
        assert result['visual_module_executed'] == (scene == 'floating_objects')
        assert not result['liquid_visual_guidance_used']
        if scene == 'floating_objects':
            off = folder/'off'
            off.mkdir()
            value = call(off, common)
            assert not value['visual_module_executed']
            assert value['liquid_depth_m'] == result['liquid_depth_m']
        checks.append({'scene': scene, 'matched_replay': True, 'liquid_visual_override': False})
    frozen = json.loads((OUT/'refraction-frozen-records.json').read_text())
    for entry in json.loads((OUT/'refraction-example-configs.json').read_text()):
        scene, config = entry['scene'], entry['config']
        folder = output/scene
        folder.mkdir()
        masks = {ch: cv2.imread(str(Path(config['folder'])/('roi_'+ch+'.png')), 0) for ch in ('left', 'right')}
        np.savez_compressed(folder/'masks.npz', **masks)
        prior = [{ch: str(Path(p['folder'])/(ch+'.png')) for ch in ('left', 'right')} for p in config['priors']]
        spec = {'kind': 'five_priors', 'calibration': str(Path(config['priors'][0]['folder'])/'calibration.json'),
                'masks': 'masks.npz', 'prior_images': prior,
                'known_heights_m': [p['known_liquid_height_m'] for p in config['priors']]}
        (folder/'initialization.json').write_text(json.dumps(spec, indent=2))
        current = Path(config['tests'][0]['folder'])
        result = call(folder, ['--scene', scene, '--initialization', str(folder/'initialization.json'),
                               '--left', str(current/'left.png'), '--right', str(current/'right.png'),
                               '--enable-floating-peak-vision'])
        old = next(r for r in frozen if r['config_id'] == config['id'] and r['sample_id'] == 0)
        assert abs(result['liquid_depth_m']-old['liquid_depth_m']) < 1e-7
        assert not result['visual_module_executed']
        checks.append({'scene': scene, 'matched_frozen_refraction': True, 'visual_executed': False})
    (output/'scope-audit.json').write_text(json.dumps(checks, indent=2))
    print('DETERMINED_PUBLIC_CLI_SMOKE_COMPLETE', len(checks), flush=True)


if __name__ == '__main__':
    main()
