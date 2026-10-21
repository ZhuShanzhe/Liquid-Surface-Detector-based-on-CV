import concurrent.futures,subprocess,sys,os,time,json
from pathlib import Path
from range_protocol import ROOT
CODE=Path(__file__).resolve().parent
def worker(i):
    with (ROOT/f'capture-{i:02d}.log').open('w') as log:
        command=['/root/autodl-tmp/tools/blender/blender-4.2.0-linux-x64/blender','-b','-t','2','--python-exit-code','1','-P',str(CODE/'diverse_capture.py'),'--','--category',str(i)]
        if '--smoke-high' in sys.argv:command.append('--smoke-high')
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(worker,[0,8,12,13] if '--smoke-high' in sys.argv else range(25)))
(ROOT/('capture-smoke.complete' if '--smoke-high' in sys.argv else 'capture-all.complete')).write_text('true')
