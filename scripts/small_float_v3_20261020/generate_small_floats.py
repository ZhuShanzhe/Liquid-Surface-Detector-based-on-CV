import argparse,json,sys
from pathlib import Path
sys.path.insert(0,'/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/src');sys.path.insert(0,'/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV/scripts');sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_scene_console import render,corpus_request
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out.mkdir(parents=True,exist_ok=True)
    cohorts=[('train',128,30000,202610202147),('validation',32,40000,202610202239),('test',100,50000,202610202351)]
    for split,count,offset,seed in cohorts:
        for g in range(a.worker,count,a.workers):
            for bits in (0,1,2,4,16,8,9,10,12,24):
                folder=a.out/split/f'{g:04d}_{bits:02d}'
                if (folder/'complete.json').exists():continue
                req=corpus_request(offset+g,bits*3,seed);req['radius_m']=(req['bottom_m']-req['capacity_m'])*.18
                req['float_count']=(1,3,12)[g%3] if bits&8 else 0
                req['small_float_training']=True;req['float_shape_mode']=('small','mixed','thin_high')[g%3]
                render(req,folder)
            print('SMALL_FLOAT_GROUP',split,g,flush=True)
