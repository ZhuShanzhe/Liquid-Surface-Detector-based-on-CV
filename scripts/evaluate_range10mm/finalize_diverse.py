import hashlib,json,subprocess,tarfile
from pathlib import Path
from range_protocol import ROOT as R
CODE=Path(__file__).resolve().parent;REPO=Path('/root/autodl-tmp/Liquid-Surface-Detector-based-on-CV')
assert (R/'measurements.complete').exists()
snap=R/'formal-source-and-models.tar.gz'
with tarfile.open(snap,'w:gz',compresslevel=4) as tar:
    for name in ('src','scripts','configs','tests','pyproject.toml','README.md','assets/models'):
        tar.add(REPO/name,arcname=name,filter=lambda info:None if '__pycache__' in info.name or info.name.endswith('.pyc') else info)
p=json.loads((R/'protocol.json').read_text());p['evaluation_helpers_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in CODE.glob('*.py')}
p['source_snapshot_sha256']=hashlib.sha256(snap.read_bytes()).hexdigest()
p['narrow_control_root']=str(R.parent/'formal-v14-single-composite-20261021')
p['preflight_geometry']=json.loads((R/'preflight-geometry.json').read_text())
p['capture_domain_audit']=json.loads((R/'capture-domain-audit.json').read_text())
(R/'protocol.json').write_text(json.dumps(p,indent=2))
subprocess.run(['/root/autodl-tmp/envs/liquid-depth/bin/python',str(CODE/'summarize_evaluation.py'),'--root',str(R),'--diverse','--control-root',str(R.parent/'formal-v14-single-composite-20261021')],check=True)
subprocess.run(['/root/autodl-tmp/envs/liquid-depth/bin/python',str(CODE/'derive_report_conclusions.py')],check=True)
subprocess.run(['/root/autodl-tmp/envs/liquid-depth/bin/python',str(CODE/'final_results_audit.py'),'--root',str(R),'--strict'],check=True)
subprocess.run(['/root/autodl-tmp/envs/liquid-depth/bin/python',str(CODE/'augment_report_audit.py'),'--root',str(R)],check=True)
(R/'report-ready.complete').write_text('true')
