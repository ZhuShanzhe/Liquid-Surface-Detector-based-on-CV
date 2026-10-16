"""Resident research wrapper; parallel matching does not change V29 equations.

Input is five observed left/right dictionaries and five known heights. No
evaluation manifests, hidden test heights, or sensor-certified output contract.
"""
import time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from . import transparent_relative_v29 as rel


def _initialize_worker():
    import cv2
    cv2.setNumThreads(1)


def _track_worker(job):
    anchor,current,cal,channel=job
    return rel.track_pair(anchor,current,cal,channel)


class FixedCameraRelativeMeter:
    def __init__(self, prior_images, known_heights, calibration, condition,
                 method='relative_corrected', workers=4,backend='threads'):
        if method not in ('relative_empty','relative_multi','relative_corrected'):
            raise ValueError('unsupported_relative_method')
        if len(prior_images)!=5:raise ValueError('exactly_five_prior_groups_required')
        if backend not in ('threads','processes'):raise ValueError('unsupported_parallel_backend')
        rel.validate_heights(known_heights,calibration)
        self.cal=dict(calibration);self.images=[{ch:np.array(p[ch],copy=True) for ch in ('left','right')} for p in prior_images]
        self.pool=(ThreadPoolExecutor(max_workers=max(1,int(workers))) if backend=='threads' else
            ProcessPoolExecutor(max_workers=max(1,int(workers)),mp_context=get_context('spawn'),initializer=_initialize_worker))
        start=time.perf_counter()
        try:
            jobs=[(a,b,ch) for a in range(5) for b in range(a+1,5) for ch in ('left','right')]
            tasks=[(self.images[a][ch],self.images[b][ch],self.cal,ch) for a,b,ch in jobs]
            pairs=[(a,b,p) for (a,b,ch),p in zip(jobs,self.pool.map(_track_worker,tasks))]
            self.prior=rel.fit_relative(pairs,known_heights,self.cal,condition,'empty' if method=='relative_empty' else 'multi')
            self.correction=rel.empirical_correction(pairs,self.prior,self.cal) if method=='relative_corrected' else None
        except Exception:
            self.pool.shutdown();raise
        self.calibration_ms=(time.perf_counter()-start)*1000

    def measure(self, images):
        start=time.perf_counter();jobs=[(a,ch) for a in range(5) for ch in ('left','right')]
        tasks=[(self.images[a][ch],images[ch],self.cal,ch) for a,ch in jobs]
        groups=[(a,p) for (a,ch),p in zip(jobs,self.pool.map(_track_worker,tasks))];matched=time.perf_counter()
        result=rel.infer_relative(groups,self.prior,self.cal,self.correction)
        result.update(matching_ms=(matched-start)*1000,solve_ms=(time.perf_counter()-matched)*1000,
            processing_ms=(time.perf_counter()-start)*1000,acquisition_included=False,
            production_enabled=False,fixed_camera_required=True)
        return result

    def close(self):self.pool.shutdown()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
