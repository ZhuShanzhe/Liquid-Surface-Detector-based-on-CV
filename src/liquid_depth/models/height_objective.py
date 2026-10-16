"""Differentiable bottom-referenced spatial estimator for supervised training."""
import numpy as np
import torch


def differentiable_height(depth,probability,calibration):
    """Same cell medians and area interpolation as evaluation, without GT masks."""
    h,w=depth.shape
    y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    k=torch.as_tensor(calibration.intrinsics,dtype=depth.dtype,device=depth.device)
    pose=torch.as_tensor(calibration.camera_to_world_cv,dtype=depth.dtype,device=depth.device)
    camera=torch.stack(((x-k[0,2])/k[0,0]*depth,(y-k[1,2])/k[1,1]*depth,depth),-1)
    xyz=camera@pose[:3,:3].T+pose[:3,3]
    radii=depth.new_tensor(calibration.radii_m)
    normalized=xyz[...,:2]/radii
    use=(probability.detach()>=.5)&torch.isfinite(depth)&(depth>0)
    use &= normalized.detach().abs().pow(calibration.footprint_exponent).sum(-1)<=1
    if int(use.sum())<64:return None
    points=xyz[use];coords=normalized[use]
    index=torch.floor((coords.detach()+1)*5).long().clamp(0,9)
    code=index[:,0]+10*index[:,1]
    anchors=[]
    for c in torch.unique(code):
        q=points[code==c]
        if len(q)>=4:anchors.append(torch.quantile(q,.5,dim=0))
    if len(anchors)<4:return None
    anchors=torch.stack(anchors)
    area=depth.new_tensor(calibration.area_grid())
    cells=torch.floor((area/radii+1)*5).long().clamp(0,9)
    possible=len(torch.unique(cells[:,0]+10*cells[:,1]))
    if len(anchors)/possible<.12:return None
    distances=torch.linalg.vector_norm((area[:,None]-anchors[None,:,:2])/radii,dim=-1)
    selected=distances.detach().argsort(dim=1)[:,:4]
    weights=distances.gather(1,selected).clamp_min(.01).pow(-2)
    heights=anchors[:,2]-calibration.bottom_z_m
    return ((heights[selected]*weights).sum(1)/weights.sum(1)).mean()


def liquid_height_loss(depth,truth,valid,predicted_probability,calibrations,references):
    """Relative scale is true liquid height, not camera distance; labels for loss only."""
    frame_losses=[];height_losses=[]
    for j,(cal,reference) in enumerate(zip(calibrations,references)):
        support=valid[j,0]>0
        if not support.any():continue
        h,w=depth.shape[-2:]
        y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
        k=depth.new_tensor(cal.intrinsics);rotation=depth.new_tensor(cal.camera_to_world_cv[:3,:3])
        ray=torch.stack(((x-k[0,2])/k[0,0],(y-k[1,2])/k[1,1],torch.ones_like(x)),-1)
        vertical_factor=(ray@rotation.T)[...,2]
        tolerance=max(.005,.02*reference)
        error=((depth[j,0]-truth[j,0])*vertical_factor).abs()[support]/tolerance
        pixel=torch.where(error<1,.5*error.square(),error-.5).mean()
        estimate=differentiable_height(depth[j,0],predicted_probability[j,0],cal)
        if estimate is None:
            global_loss=pixel
        else:
            global_error=(estimate-reference).abs()/tolerance
            global_loss=torch.where(global_error<1,.5*global_error.square(),global_error-.5)
            height_losses.append(global_loss)
        frame_losses.append(.5*pixel+.5*global_loss)
    if not frame_losses:raise ValueError('No supervised frame')
    losses=torch.stack(frame_losses)
    return losses.mean()+.2*torch.topk(losses,max(1,len(losses)//5)).values.mean()
