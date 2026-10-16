"""Opt-in fixed-camera finite-bottom research interface for the frozen V30 recipe.

Five prior capture groups each contain left/right rectified monochrome images.
The known heights, camera geometry and fixed empty-bottom ROIs come from setup.
No confidence-qualified industrial output is claimed by this interface.
"""
import time
import numpy as np
from . import finite_relative_v30 as core


class FiniteBottomRelativeMeter:
    def __init__(self,prior_groups,known_heights,calibration,condition,rois):
        core.old.v25.require_target(condition)
        if len(prior_groups)!=5:raise ValueError('five_prior_capture_groups_required')
        self.cal=dict(calibration);self.condition=condition
        core.old.validate_heights(known_heights,self.cal)
        self.priors=[self._images(group) for group in prior_groups]
        shape=self.priors[0]['left'].shape
        if any(im.shape!=shape for group in self.priors for im in group.values()):
            raise ValueError('inconsistent_rectified_image_shape')
        self.rois={ch:np.array(rois[ch],copy=True) for ch in ('left','right')}
        if any(mask.shape!=shape for mask in self.rois.values()):raise ValueError('roi_shape_mismatch')
        if shape!=(self.cal['height'],self.cal['width']):raise ValueError('calibration_crop_shape_mismatch')
        pairs=[(a,b,core.direct_pair(self.priors[a][ch],self.priors[b][ch],self.cal,self.rois[ch],ch))
            for a in range(5) for b in range(a+1,5) for ch in ('left','right')]
        # Frozen development recipe: regularize weak mild-translucent responses.
        floor=.005 if condition=='transparent' else .05
        self.prior=core.fit_response(pairs,known_heights,self.cal,condition,floor)

    @staticmethod
    def _images(group):
        result={}
        for ch in ('left','right'):
            im=np.asarray(group[ch])
            if im.dtype!=np.uint8 or im.ndim!=2:raise ValueError('rectified_uint8_monochrome_required')
            result[ch]=im.copy()
        return result

    def measure(self,current_group):
        start=time.perf_counter();images=self._images(current_group)
        if any(images[ch].shape!=self.priors[0][ch].shape for ch in images):
            raise ValueError('current_crop_shape_changed_recalibration_required')
        groups=[(a,core.direct_pair(self.priors[a][ch],images[ch],self.cal,self.rois[ch],ch))
            for a in range(5) for ch in ('left','right')]
        result=core.infer_response(groups,self.prior,self.cal)
        result.update(processing_ms=(time.perf_counter()-start)*1000,prior_capture_budget=5,
            fixed_camera_required=True,qualified=False,production_enabled=False,method='finite_bottom_v30')
        return result
