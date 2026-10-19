"""Independent grouped render cohort: RGB-D validation for activation, untouched test."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_scene_console import render,corpus_request

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out.mkdir(parents=True,exist_ok=True)
    # Four validation and eight final-test physical installations, 8 factor configurations.
    # New geometries/appearances, not noise repeats of any training group.
    for g in range(a.worker,12,a.workers):
        for bits in (0,1,2,4,8,3,9,12,16,17,18,24):
            req=corpus_request(100+g,bits*3,202610192127)
            # Keep complete empty/rim visibility in this calibrated-demo cohort.
            req['radius_m']=(req['bottom_m']-req['capacity_m'])*.18
            folder=a.out/f'{g:04d}_{bits:02d}'
            if not (folder/'complete.json').exists():render(req,folder)
        print('HOLDOUT_GROUP',g,flush=True)
    if a.worker==0:
        (a.out/'protocol.json').write_text(json.dumps(dict(seed=202610192127,validation_groups=list(range(4)),test_groups=list(range(4,12)),
            cases_per_group=12,no_truth_in_runtime=True,scope='opaque_known_factor_combinations_only'),indent=2))
