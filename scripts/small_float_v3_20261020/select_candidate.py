import json,time,shutil,hashlib
from pathlib import Path
r=Path('/root/autodl-tmp/liquid-depth-artifacts/small-float-20261020')
while not all((r/f'candidate-{s}/complete.json').exists() for s in ('old','new')):time.sleep(15)
policies={s:json.loads((r/f'candidate-{s}/policy.json').read_text()) for s in ('old','new')}
selected=min(policies,key=lambda s:policies[s]['objective']);out=r/'candidate-training';out.mkdir(exist_ok=True)
for name in ('models.joblib','policy.json','summary.json'):shutil.copyfile(r/f'candidate-{selected}'/name,out/name)
p=json.loads((r/f'candidate-{selected}/protocol.json').read_text());p['family_selection_validation_only']=policies;p['selected_source']=selected
(r/'frozen-test-protocol.json').write_text(json.dumps(p,indent=2));(out/'complete.json').write_text(json.dumps({'complete':True,'selected':selected}));print('SELECTED',selected,policies,flush=True)
