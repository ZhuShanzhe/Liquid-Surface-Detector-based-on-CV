"""GT-free image matching and structured-light decoding."""
import cv2
import numpy as np


def normalized(image):
    f=image.astype(np.float32)
    mean=cv2.GaussianBlur(f,(0,0),7)
    sd=np.sqrt(np.maximum(cv2.GaussianBlur(f*f,(0,0),7)-mean*mean,1))
    return np.uint8(np.clip((f-mean)/sd*35+128,0,255))


def stereo_matches(left,right,cal,optimized=True):
    l,r=(normalized(left),normalized(right)) if optimized else (left,right)
    h,w=l.shape
    sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=5,
        P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=8,speckleWindowSize=30,
        speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity=sgbm.compute(l,r).astype(np.float32)/16
    roi=np.zeros_like(l);roi[32:h-32,140:w-32]=255
    points=cv2.goodFeaturesToTrack(l,600,.015,8,mask=roi,blockSize=5)
    if points is None:
        return empty_observation()
    points=points.reshape(-1,2)
    ix=np.round(points[:,0]).astype(int);iy=np.round(points[:,1]).astype(int)
    dp=disparity[iy,ix]
    good=(dp>0)&(dp<126)
    points,dp=points[good],dp[good]
    if not len(points):return empty_observation()
    other=points-np.column_stack((dp,np.zeros(len(dp)))).astype(np.float32)
    fb=np.zeros(len(points))
    if optimized:
        other,status,error=cv2.calcOpticalFlowPyrLK(l,r,points[:,None].astype(np.float32),
            other[:,None].astype(np.float32),winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back,bst,bec=cv2.calcOpticalFlowPyrLK(r,l,other,points[:,None].astype(np.float32).copy(),
            winSize=(21,21),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
        other=other.reshape(-1,2);back=back.reshape(-1,2)
        fb=np.linalg.norm(back-points,axis=1)
        good=(status.ravel()>0)&(bst.ravel()>0)&(fb<.7)
        good&=(other[:,0]>=12)&(other[:,0]<w-12)&(other[:,1]>=12)&(other[:,1]<h-12)
        points,other,fb=points[good],other[good],fb[good]
    quality=[];keep=[]
    for i,(p,t) in enumerate(zip(points,other)):
        a=cv2.getRectSubPix(l,(13,13),tuple(p)).astype(float).ravel()
        b=cv2.getRectSubPix(r,(13,13),tuple(t)).astype(float).ravel()
        a-=a.mean();b-=b.mean()
        corr=float(a@b/np.sqrt(max(a@a*(b@b),1e-9)))
        if not optimized or corr>.65:
            keep.append(i);quality.append(max(.05,corr)/(1+fb[i]))
    points,other=points[keep],other[keep]
    return package(points,other,np.asarray(quality),cal,
                   visibility=float(np.std(left)/128),method='LK2D' if optimized else 'SGBM')


def decode_structured(images,cal,optimized=True):
    black=images['black'].astype(float);white=images['white'].astype(float)
    contrast=white-black
    h,w=black.shape
    # Dense candidates are reduced to spatial blocks below, never to truth ROI.
    valid=(contrast>(12 if optimized else 5))&(white<250)&(black<245)
    coordinates=[];quality=np.ones_like(black)
    for axis,size in [('x',512),('y',384)]:
        gray=np.zeros((h,w),np.int32)
        certainty=np.ones((h,w))
        for bit in range(5):
            diff=images[f'{axis}_g{bit}'].astype(float)-images[f'{axis}_i{bit}'].astype(float)
            gray|=(diff>0).astype(np.int32)<<bit
            certainty=np.minimum(certainty,np.abs(diff)/np.maximum(contrast,1))
        binary=gray.copy()
        for shift in (1,2,4):binary^=binary>>shift
        p=[images[f'{axis}_p{k}'].astype(float) for k in range(4)]
        cosine=p[0]-p[2];sine=p[3]-p[1]
        if optimized:
            # Normalize albedo/veil before phasor smoothing. Averaging wrapped
            # phase angles themselves would introduce discontinuity errors.
            cosine=cv2.GaussianBlur(cosine/np.maximum(contrast,4),(0,0),.8)
            sine=cv2.GaussianBlur(sine/np.maximum(contrast,4),(0,0),.8)
        phase=np.mod(np.arctan2(sine,cosine),2*np.pi)
        fraction=phase/(2*np.pi)*16
        coordinate=binary*16+fraction
        amplitude=np.hypot(p[3]-p[1],p[0]-p[2])/2
        modulation=amplitude/np.maximum(contrast,1)
        if optimized:
            valid&=(modulation>.18)&(certainty>.10)&(fraction>.8)&(fraction<15.2)
        valid&=(coordinate>=0)&(coordinate<size)
        quality=np.minimum(quality,modulation*certainty)
        coordinates.append(coordinate)
    yy,xx=np.indices((h,w))
    valid&=(xx>32)&(xx<w-32)&(yy>32)&(yy<h-32)
    ps=[];qs=[];weights=[]
    # Choose one well-modulated pixel per fixed region to limit correlation.
    for y0 in range(32,h-32,16):
        for x0 in range(32,w-32,16):
            region=np.s_[y0:y0+16,x0:x0+16]
            scores=np.where(valid[region],quality[region],-1)
            if scores.max()<=0:continue
            dy,dx=np.unravel_index(np.argmax(scores),scores.shape);y,x=y0+dy,x0+dx
            ps.append((x,y));qs.append((coordinates[0][y,x],coordinates[1][y,x]));weights.append(quality[y,x])
    return package(np.asarray(ps).reshape(-1,2),np.asarray(qs).reshape(-1,2),np.asarray(weights),cal,
                   visibility=float(np.median(np.maximum(contrast,0))/255),method='GrayPhase2D')


def empty_observation():
    return dict(left_q=np.empty((0,2)),right_q=np.empty((0,2)),weights=np.empty(0),
                left_pixels=np.empty((0,2)),right_pixels=np.empty((0,2)),regions=0,visibility=0.)


def package(points,other,weights,cal,**extra):
    if not len(points):return empty_observation()
    ql=(points-np.array([cal['cx'],cal['cy']]))/cal['fx']
    qr=(other-np.array([cal['reference_cx'],cal['reference_cy']]))/cal['fx']
    finite=np.isfinite(ql).all(1)&np.isfinite(qr).all(1)&np.isfinite(weights)
    ql,qr,points,other,weights=ql[finite],qr[finite],points[finite],other[finite],weights[finite]
    # Same cap for every method; deterministic spatial representative selection.
    selected=[]
    cells=np.floor(points/24).astype(int)
    for cell in np.unique(cells,axis=0):
        ids=np.where((cells==cell).all(1))[0]
        selected.append(ids[np.argmax(weights[ids])])
    s=np.asarray(selected,dtype=int)
    return dict(left_q=ql[s],right_q=qr[s],weights=weights[s],left_pixels=points[s],
                right_pixels=other[s],regions=len(s),**extra)
