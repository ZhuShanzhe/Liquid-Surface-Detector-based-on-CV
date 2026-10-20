"""Validation-only early stopping; independent test labels are never read."""
import json,time,os,signal
from pathlib import Path
r=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020');out=r/'pixel-training';target=str(r/'train_small_float.py').encode()
rule={'minimum_epochs':12,'patience':8,'monitor':'validation score','test_used':False}
(r/'early-stop-rule.json').write_text(json.dumps(rule))
while not (out/'complete.json').exists():
    try:
        history=json.loads((out/'training.json').read_text())['history'];scores=[x['score'] for x in history];best=max(range(len(scores)),key=scores.__getitem__)
        if len(scores)>=rule['minimum_epochs'] and len(scores)-1-best>=rule['patience']:
            stopped=[]
            for p in Path('/proc').iterdir():
                if p.name.isdigit() and (p/'cmdline').exists() and target in (p/'cmdline').read_bytes().split(bytes([0])):
                    os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
            (out/'complete.json').write_text(json.dumps({'complete':True,'epochs_run':len(scores),'best_epoch':best+1,'reason':'validation_patience','rule':rule,'stopped_pids':stopped}));print('EARLY_STOP',len(scores),best+1,flush=True);break
    except (FileNotFoundError,json.JSONDecodeError,ValueError):pass
    time.sleep(20)
