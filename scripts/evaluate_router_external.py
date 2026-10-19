import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
from liquid_depth.scene_classifier import SceneClassifier,TAGS,MATERIALS
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2);cv2.setNumThreads(1);net=SceneClassifier(a.checkpoint)
rows=[]
for file in sorted(a.data.glob('*/complete.json')):
    folder=file.parent;req=json.loads((folder/'request.json').read_text());rgb=cv2.imread(str(folder/'rgb.png'))
    with np.load(folder/'routing_prior.npz',allow_pickle=False) as prior:
        value=net.predict(rgb,depth_m=np.load(folder/'depth_m.npy'),dry_depth_m=prior['dry_depth_m'],roi=prior['roi'])
    labels=[tag for tag,key in zip(TAGS,('darkness','glare','wave_m','float_count')) if req[key]>0]
    rows.append(dict(group=req['index'],true_material=req['material'],true_tags=labels,prediction=value))
if len(rows)!=288:raise RuntimeError('external_test_incomplete')
tags={}
for tag in TAGS:
    tp=sum(tag in r['true_tags'] and tag in r['prediction']['tags'] for r in rows)
    fp=sum(tag not in r['true_tags'] and tag in r['prediction']['tags'] for r in rows)
    fn=sum(tag in r['true_tags'] and tag not in r['prediction']['tags'] for r in rows)
    tags[tag]=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))
result=dict(physical_installations=12,frames=288,seed=202610192877,
    material_accuracy=sum(r['true_material']==r['prediction']['material'] for r in rows)/len(rows),tags=tags,
    exact_tag_accuracy=sum(set(r['true_tags'])==set(r['prediction']['tags']) for r in rows)/len(rows),
    scope='new_independent_Blender_RGBD_classification_only_not_height_qualification',
    thresholds_locked_before_test=True,records=rows)
a.out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}))
