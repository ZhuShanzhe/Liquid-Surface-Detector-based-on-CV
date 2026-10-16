"""Opt-in research wrapper. Five calibration groups, no test-frame memory.

This is not enabled by the production GUI or scenario router automatically.
"""
import cv2
import numpy as np
from . import range_registration_v33 as reg
from . import range_relative_v32 as model

class FivePriorImageMeter:
    def __init__(self,cal,masks,prior_images,known_heights,condition,method='baseline_v32',sr_backend=None):
        if method not in reg.OPTIONS:raise ValueError('unknown_registration_method')
        if len(prior_images)!=5 or len(known_heights)!=5:raise ValueError('requires_exactly_five_prior_groups')
        self.cal=dict(cal);self.native_masks={ch:np.asarray(masks[ch]).copy() for ch in ('left','right')}
        if condition not in ('transparent','translucent_mild'):raise ValueError('outside_transparent_route')
        self.method=method;self.factor=1 if method=='baseline_v32' else reg.OPTIONS[method]['factor']
        self.sr=sr_backend
        if 'swinir' in method and self.sr is None:self.sr=reg.SwinIRX2()
        self.masks={ch:cv2.resize(m,None,fx=self.factor,fy=self.factor,interpolation=cv2.INTER_NEAREST)
                    if self.factor!=1 else m for ch,m in self.native_masks.items()}
        self.prior_images=[self._prepare(images) for images in prior_images]
        pairs=[]
        for a in range(5):
            for b in range(a+1,5):
                for ch in ('left','right'):
                    pairs.append((a,b,reg.pair(self.prior_images[a][ch],self.prior_images[b][ch],self.cal,self.masks[ch],ch,self.method)))
        self.prior=model.fit_local(pairs,known_heights,self.cal,condition,'pchip')

    def _prepare(self,images):
        if set(images)!= {'left','right'}:raise ValueError('requires_left_and_right_images')
        for ch in ('left','right'):
            a=np.asarray(images[ch])
            if a.ndim!=2 or a.shape!=self.native_masks[ch].shape or not np.isfinite(a).all():
                raise ValueError('invalid_native_grayscale_image')
        if 'swinir' in self.method:return dict(zip(('left','right'),self.sr([images['left'],images['right']])))
        if self.factor==2:return {ch:cv2.resize(np.asarray(im,dtype=np.float32),None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC) for ch,im in images.items()}
        return {ch:np.asarray(im).copy() for ch,im in images.items()}

    def measure(self,images):
        current=self._prepare(images);groups=[]
        for a,anchor in enumerate(self.prior_images):
            for ch in ('left','right'):
                groups.append((a,reg.pair(anchor[ch],current[ch],self.cal,self.masks[ch],ch,self.method)))
        result=model.infer(groups,self.prior,self.cal,'robust',None)
        result.update(registration_method=self.method,research_only=True,production_enabled=False,
                      measurement_frame_memory=False,prior_groups=5)
        return result
