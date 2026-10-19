"""Grouped train/validation/test split for a multi-label visual routing head.

Thresholds are selected on validation only. Optical-family ambiguity requires
operator confirmation rather than an unverified numerical fallback.
"""
import argparse, hashlib, json, random, time
from pathlib import Path
import cv2, numpy as np, torch
from torch import nn
from liquid_depth.scene_classifier import PretrainedSceneHead, image_tensor, geometry_features,TAGS, MATERIALS


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(190731);np.random.seed(190731);random.seed(190731)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    rows=[];images=[];geometry=[]
    for f in sorted(a.data.glob('*/complete.json')):
        req=json.loads((f.parent/'request.json').read_text());group=req['index']
        rgb=cv2.imread(str(f.parent/'rgb.png'))
        if rgb is None:raise RuntimeError('incomplete_rgb')
        tags=[req['darkness']>0,req['glare']>0,req['wave_m']>0,req['float_count']>0]
        rows.append(dict(group=group,material=MATERIALS.index(req['material']),tags=tags,path=str(f.parent/'rgb.png')))
        with np.load(f.parent/'routing_prior.npz',allow_pickle=False) as prior:
            images.append(image_tensor(rgb,prior['roi']))
            geometry.append(geometry_features(np.load(f.parent/'depth_m.npy'),prior['dry_depth_m'],prior['roi']))
    groups=sorted({r['group'] for r in rows})
    # Split complete geometries, including all variants, before loading labels into training.
    order=np.random.default_rng(20261019117).permutation(groups)
    train_groups=set(map(int,order[:int(len(groups)*.67)]));val_groups=set(map(int,order[int(len(groups)*.67):int(len(groups)*.84)]))
    split={k:np.array([i for i,r in enumerate(rows) if (r['group'] in train_groups if k=='train' else r['group'] in val_groups if k=='validation' else r['group'] not in train_groups|val_groups)]) for k in ('train','validation','test')}
    if min(map(len,split.values()))<24:raise RuntimeError('insufficient_grouped_corpus')
    x=torch.stack(images);gx=torch.tensor(np.stack(geometry));ym=torch.tensor([r['material'] for r in rows]);yt=torch.tensor([r['tags'] for r in rows],dtype=torch.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=PretrainedSceneHead(pretrained=True).to(device)
    model.geometry_mean.copy_(gx[split['train']].mean(0).to(device))
    model.geometry_scale.copy_(gx[split['train']].std(0).clamp_min(.05).to(device))
    opt=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':.0001},
                          {'params':model.head.parameters(),'lr':.001}],weight_decay=.02)
    best=float('inf');history=[];start=time.time()
    for epoch in range(a.epochs):
        model.train();losses=[]
        order=np.random.permutation(split['train'])
        for offset in range(0,len(order),24):
            ids=order[offset:offset+24]
            batch=x[ids].to(device)
            # Mild color/exposure variation must not turn darkness labels into a contradiction.
            gain=torch.empty((len(ids),3,1,1),device=device).uniform_(.92,1.08)
            batch=(batch*gain).clamp(0,1)
            if epoch%2:batch=batch.flip(-1)
            pred=model(batch,gx[ids].to(device));loss=nn.functional.cross_entropy(pred[:,:3],ym[ids].to(device))+nn.functional.binary_cross_entropy_with_logits(pred[:,3:],yt[ids].to(device))
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            pred=model(x[split['validation']].to(device),gx[split['validation']].to(device))
            val=(nn.functional.cross_entropy(pred[:,:3],ym[split['validation']].to(device))+nn.functional.binary_cross_entropy_with_logits(pred[:,3:],yt[split['validation']].to(device))).item()
        history.append(dict(epoch=epoch,train=float(np.mean(losses)),validation=val))
        if val<best:
            best=val;beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=epoch
        if epoch%10==0:print('ROUTER_TRAIN',epoch,history[-1],flush=True)
    model.load_state_dict(beststate);model.eval()
    def scores(ids):
        with torch.inference_mode():
            p=model(x[ids].to(device),gx[ids].to(device))
            return p[:,:3].softmax(1).cpu().numpy(),p[:,3:].sigmoid().cpu().numpy()
    vm,vt=scores(split['validation']);truth=yt[split['validation']].numpy();thresholds={}
    threshold_audit={}
    for j,tag in enumerate(TAGS):
        candidates=[]
        for threshold in np.arange(.15,.91,.05):
            use=vt[:,j]>=threshold
            # Routing proxy cost: wrongly invoking float highest is penalized 2x.
            # This is NOT a replacement for end-to-end height-regret evaluation.
            fp=float(np.mean(use & (truth[:,j]==0)));fn=float(np.mean(~use & (truth[:,j]==1)))
            candidates.append((fp*(2 if tag=='floating_objects' else 1)+fn,float(threshold)))
        thresholds[tag]=min(candidates)[1];threshold_audit[tag]=dict(selection='validation_weighted_label_cost',candidates=candidates)
    confidence=vm.max(1);correct=vm.argmax(1)==ym[split['validation']].numpy()
    material_threshold=.75
    qualified=False
    for t in np.arange(.55,.96,.05):
        accepted=confidence>=t
        if accepted.sum()>=12 and correct[accepted].mean()>=.95:
            material_threshold=float(t);qualified=True;break
    state=dict(schema=4,model=beststate,thresholds=thresholds,material_threshold=material_threshold,
               model_id='rgbd_mobilenet_router_sim_v4_20261019',requires_material_confirmation=not qualified,
               pretrained_source='torchvision_MobileNet_V3_Small_Weights_IMAGENET1K_V1')
    torch.save(state,a.out/'scene_head.pt')
    metrics={}
    for name,ids in split.items():
        pm,pt=scores(ids);target=yt[ids].numpy();pred=pt>=np.array([thresholds[t] for t in TAGS])
        per={}
        for j,tag in enumerate(TAGS):
            tp=int(((pred[:,j]==1)&(target[:,j]==1)).sum());fp=int(((pred[:,j]==1)&(target[:,j]==0)).sum());fn=int(((pred[:,j]==0)&(target[:,j]==1)).sum())
            per[tag]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
        metrics[name]=dict(frames=len(ids),groups=sorted({rows[i]['group'] for i in ids}),
            material_accuracy=float(np.mean(pm.argmax(1)==ym[ids].numpy())),
            material_confusion=[[int(((ym[ids].numpy()==i)&(pm.argmax(1)==j)).sum()) for j in range(3)] for i in range(3)],
            exact_tag_accuracy=float(np.mean((pred==target).all(1))),tags=per)
    report=dict(seed=190731,split_seed=20261019117,source='new_Blender_EEVEE_rendered_installations',
        metrics=metrics,thresholds=thresholds,threshold_selection=threshold_audit,
        material_threshold=material_threshold,material_validation_qualified=qualified,
        training_seconds=time.time()-start,best_epoch=bestepoch,history=history,
        limitations=['synthetic_only_not_field_qualified','RGB_does_not_identify_refractive_index',
                    'thresholds_label_cost_only_end_to_end_regret_required','no_real_world_material_guarantee'])
    (a.out/'training_report.json').write_text(json.dumps(report,indent=2))
    (a.out/'corpus_manifest.json').write_text(json.dumps(dict(rows=rows,split={k:v.tolist() for k,v in split.items()},
        checkpoint_sha256=hashlib.sha256((a.out/'scene_head.pt').read_bytes()).hexdigest()),indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','threshold_selection')}),flush=True)


if __name__=='__main__':main()
