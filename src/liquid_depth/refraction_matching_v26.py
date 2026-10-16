"""Block-aggregated phase decoding. Does not consume liquid height or truth."""
import cv2
import numpy as np
from .refraction_matching_v23 import package


def decode_structured(images,cal,sigma=.8,block_size=16):
    black=images['black'].astype(float);white=images['white'].astype(float)
    contrast=white-black;h,w=black.shape
    valid=(contrast>12)&(white<250)&(black<245)
    quality=np.ones_like(black);coordinates=[]
    for axis,size in [('x',512),('y',384)]:
        gray=np.zeros((h,w),np.int32);certainty=np.ones((h,w))
        for bit in range(5):
            diff=images[f'{axis}_g{bit}'].astype(float)-images[f'{axis}_i{bit}'].astype(float)
            gray|=(diff>0).astype(np.int32)<<bit
            certainty=np.minimum(certainty,np.abs(diff)/np.maximum(contrast,1))
        binary=gray.copy()
        for shift in (1,2,4):binary^=binary>>shift
        p=[images[f'{axis}_p{k}'].astype(float) for k in range(4)]
        cosine=p[0]-p[2];sine=p[3]-p[1]
        # Raw phasors avoid dividing each noisy sample by a noisy contrast.
        if sigma>0:
            cosine=cv2.GaussianBlur(cosine,(0,0),sigma)
            sine=cv2.GaussianBlur(sine,(0,0),sigma)
        fraction=np.mod(np.arctan2(sine,cosine),2*np.pi)/(2*np.pi)*16
        coordinate=binary*16+fraction
        amplitude=np.hypot(cosine,sine)/2
        modulation=amplitude/np.maximum(contrast,1)
        valid&=(modulation>.18)&(certainty>.1)&(fraction>.8)&(fraction<15.2)
        valid&=(coordinate>=0)&(coordinate<size)
        quality=np.minimum(quality,modulation*certainty)
        coordinates.append(coordinate)
    yy,xx=np.indices((h,w));valid&=(xx>32)&(xx<w-32)&(yy>32)&(yy<h-32)
    points=[];other=[];weights=[]
    for y0 in range(32,h-32,block_size):
        for x0 in range(32,w-32,block_size):
            sl=np.s_[y0:y0+block_size,x0:x0+block_size]
            keep=valid[sl]
            if keep.sum()<6:continue
            # Symmetric aggregation rather than selecting a noise-favoured maximum.
            q=np.maximum(quality[sl][keep],.01)
            points.append((np.average(xx[sl][keep],weights=q),np.average(yy[sl][keep],weights=q)))
            other.append(tuple(np.average(c[sl][keep],weights=q) for c in coordinates))
            weights.append(float(np.mean(q)))
    return package(np.asarray(points).reshape(-1,2),np.asarray(other).reshape(-1,2),np.asarray(weights),cal,
        method='block_phase_v26',visibility=float(np.median(np.maximum(contrast,0))/255))
