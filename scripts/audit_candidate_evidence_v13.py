#!/usr/bin/env python3
"""Validation-only candidate evidence audit; no fitting or test access."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import torch
import train_candidate_ranker_v12 as v12

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ('base','candidates','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    args.feature_context=True
    args.batch_size=8
    args.workers=4
    torch.set_num_threads(4)
    d=v12.get_data(args,'val',0)
    m=v12.load_candidates(args)
    counts=defaultdict(lambda:defaultdict(int))
    offset=0
    for x,y,r in v12.loader(d,args):
        x=x.cuda()
        target=y['depth_m'].cuda()
        valid=y['valid'].cuda()>0
        q=v12.candidate_prediction(m,x,args)
        depths=q['layer_depths_m']
        order=depths.argsort(1)
        raw=.1*torch.exp(x[:,3:4]*torch.log(torch.tensor(100.,device='cuda')))
        index={f'near_to_far_{k}':order[:,k:k+1] for k in range(4)}
        index.update(prior_nearest=(depths-q['depth_m']).abs().argmin(1,keepdim=True),
                     raw_nearest=(depths-raw).abs().argmin(1,keepdim=True),
                     interface_max=q['liquid_interface_probability'].argmax(1,keepdim=True),
                     presence_max=q['layer_presence_logits'].argmax(1,keepdim=True),
                     confidence_max=q['layer_confidence'].argmax(1,keepdim=True))
        good=((depths-target).abs()<=(target*.02).clamp_min(.005)) & valid
        hits={name:good.gather(1,idx) for name,idx in index.items()}
        hits['oracle']=good.any(1,keepdim=True)
        for i in range(len(x)):
            scene=d.rows[offset+i]['scenario']
            for group in ('all',scene):
                counts[group]['n']+=int(valid[i].sum())
                for name,h in hits.items():
                    counts[group][name]+=int(h[i].sum())
        offset+=len(x)
    result={s:{k:(v/r['n'] if k!='n' else v) for k,v in r.items()} for s,r in counts.items()}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
