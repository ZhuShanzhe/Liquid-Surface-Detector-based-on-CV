"""Build a reproducible, indexed V20 delivery without modifying prior experiments."""
import json,hashlib,zipfile,subprocess
from pathlib import Path
O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v20')
DATA=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
DEST=Path('/root/autodl-tmp/liquid-depth-artifacts/deliveries/V20-refraction-delivery.zip')
def main():
 if DEST.exists():raise FileExistsError(DEST)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 subprocess.run(['git','archive','--format=zip','-o',str(O/'source.zip'),'HEAD'],check=True)
 patch=subprocess.check_output(['git','diff','--binary','b0c3687d99ba8420791486e95fa8f603cc5e5b4c','HEAD'])
 (O/'V20.patch').write_bytes(patch)
 selected={}
 for tag in ['heldoutA','heldoutB','heldoutC','heldoutD']:
  rows=json.loads((O/(tag+'-v20/results.json')).read_text())
  ids={0,2}
  for accepted in [True,False]:
   rs=[r for r in rows if r['group']=='strong' and r['accepted']==accepted]
   if rs:ids.add(rs[0]['id'])
  selected[tag]=sorted(ids)
 entries={}
 with zipfile.ZipFile(DEST,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  def add(path,name):
   b=path.read_bytes();archive.writestr(name,b);entries[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
  for p in sorted(O.rglob('*')):
   if p.is_file() and p.name not in ['archive-verification.json','delivery.json']:
    add(p,'artifacts/'+str(p.relative_to(O)))
  for tag in ['heldoutA','heldoutB','heldoutC','heldoutD']:
   root=DATA/('stratified-v20-'+tag)
   add(root/'manifest.json','data/'+root.name+'/manifest.json')
   for f in sorted(root.iterdir()):
    if not f.is_dir():continue
    for name in ['camera.json','scene.json']:
     add(f/name,'data/'+root.name+'/'+f.name+'/'+name)
    if int(f.name) in selected[tag]:
     for name in ['rgb.png','empty_rgb.png','observation.npz','burst_persistent64.npz','arrays.npz']:
      add(f/name,'data/'+root.name+'/'+f.name+'/'+name)
  manifest=dict(commit=commit,selected_samples=selected,files=entries)
  archive.writestr('archive-manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(DEST) as archive:
  assert archive.testzip() is None
  for name,rec in entries.items():assert hashlib.sha256(archive.read(name)).hexdigest()==rec['sha256'],name
 result=dict(path=str(DEST),bytes=DEST.stat().st_size,sha256=hashlib.sha256(DEST.read_bytes()).hexdigest(),
  commit=commit,files=len(entries),entries=len(entries)+1,selected_samples=selected,crc_and_hashes_verified=True)
 (O/'delivery.json').write_text(json.dumps(result,indent=2))
 (O/'archive-verification.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
