"""Geometry-preserving RGB-D input adapter; frozen inference otherwise identical."""
import time
import cv2,numpy as np,torch
import evaluate_device_v38 as core

def infer_letterbox(model,rgb,raw):
    # Both channels share precisely the same crop, scaling and pad convention.
    ow,oh=320,256;hh,ww=raw.shape;scale=min(ow/ww,oh/hh)
    w,h=int(round(ww*scale)),int(round(hh*scale));x0=(ow-w)//2;y0=(oh-h)//2
    image=np.zeros((oh,ow,3),np.uint8);depth=np.zeros((oh,ow),np.float32)
    image[y0:y0+h,x0:x0+w]=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    depth[y0:y0+h,x0:x0+w]=cv2.resize(raw,(w,h),interpolation=cv2.INTER_NEAREST)
    c=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)[None]/255).cuda()
    d=torch.from_numpy(depth[None,None]).cuda()
    t=time.perf_counter();z,m,_=model(c,d);torch.cuda.synchronize();elapsed=time.perf_counter()-t
    z=z[0,0,y0:y0+h,x0:x0+w].cpu().numpy();m=m[0,0,y0:y0+h,x0:x0+w].cpu().numpy()
    return dict(depth=cv2.resize(z,(ww,hh),interpolation=cv2.INTER_LINEAR),
                mask=cv2.resize(m,(ww,hh),interpolation=cv2.INTER_LINEAR)>=.5,neural_seconds=elapsed)

if __name__=='__main__':
    core.infer=infer_letterbox;core.main()
