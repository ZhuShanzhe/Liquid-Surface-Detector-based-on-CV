"""Development-only timing and serial/parallel prediction equivalence check."""
import time,json,argparse
import cv2,numpy as np
from experiment_relative_v29 import OUT,DATA,load,read_cache,cache_path,read_image,dump
from liquid_depth import transparent_relative_v29 as rel
from liquid_depth.relative_runtime_v29 import FixedCameraRelativeMeter

def main(backend='threads',phase='busy'):
    cv2.setNumThreads(1);configs=json.loads((DATA/'development/manifest.json').read_text())['configs'];results=[]
    for condition in ('transparent','translucent_mild'):
        c=next(x for x in configs if x['condition']==condition and x['camera_to_bottom_m']==8.4 and x['container_max_height_m']==.18)
        _,cal,hs=load(c,'empty_four');cache=read_cache(cache_path(c,'empty_four'))
        pairs=[(a,b,p) for (a,b,ch),p in cache.items() if b<5]
        prior=rel.fit_relative(pairs,hs,cal,condition,'multi');corr=rel.empirical_correction(pairs,prior,cal)
        refs=[{ch:read_image(p['folder'],ch) for ch in ('left','right')} for p in c['priors']['empty_four']]
        images=[{ch:read_image(c['tests'][j]['folder'],ch) for ch in ('left','right')} for j in (0,4,9)]
        with FixedCameraRelativeMeter(refs,hs,cal,condition,workers=4,backend=backend) as meter:
            assert abs(meter.prior['index']-prior['index'])<1e-12
            meter.measure(images[0])  # pool/runtime warm-up, not included in timing.
            for j,image in zip((0,4,9),images):
                serial=rel.infer_relative([(a,p) for (a,b,ch),p in cache.items() if b==j+5],prior,cal,corr)
                measured=meter.measure(image)
                assert measured['height_m']==serial['height_m'], 'parallel matching changed prediction'
                results.append(dict(condition=condition,sample_id=j,prediction_equal=True,calibration_ms=meter.calibration_ms,
                    matching_ms=measured['matching_ms'],solve_ms=measured['solve_ms'],total_ms=measured['processing_ms']))
    report=dict(frames=results,mean_ms=float(np.mean([r['total_ms'] for r in results])),p95_ms=float(np.percentile([r['total_ms'] for r in results],95)),
        camera_acquisition_included=False,source=f'six development frames, resident images, four {backend} workers, OpenCV one thread, {phase}',
        all_predictions_exactly_equal=True,production_qualified=False)
    dump(OUT/f'processing-{backend}-{phase}.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['threads','processes'],default='threads');ap.add_argument('--phase',choices=['busy','idle'],default='busy');args=ap.parse_args()
    main(args.backend,args.phase)
