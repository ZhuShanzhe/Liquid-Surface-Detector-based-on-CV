import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_scene_console import render,corpus_request
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out.mkdir(parents=True,exist_ok=True)
    for g in range(a.worker,12,a.workers):
        for bits in (0,1,2,4,8,3,9,12):
            for material in range(3):
                req=corpus_request(1000+g,bits*3+material,202610192877)
                folder=a.out/f'{g:04d}_{bits*3+material:02d}'
                if not (folder/'complete.json').exists():render(req,folder,corpus=True)
        print('EXTERNAL_GROUP',g,flush=True)
