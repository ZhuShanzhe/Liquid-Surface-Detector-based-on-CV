"""Check input pairing, frozen inference sources, and delivery provenance."""
import json,hashlib,subprocess,sys
from pathlib import Path
from collections import Counter
import cv2,numpy as np,scipy
from liquid_depth.refraction_calibration_v19 import assess_reference
O=Path('/root/autodl-tmp/liquid-depth-artifacts/diagnostics/refraction-v20')
D=Path('/root/autodl-tmp/liquid-depth-data/synthetic')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 frozen=json.loads((O/'frozen-consensus-before-heldout.json').read_text())
 sources={k:sha(Path(k))==v for k,v in frozen['files'].items()}
 assert all(sources.values()),sources
 inputs=[];qc=[];pairs=[]
 for tag in ['heldoutC','heldoutD']:
  root=D/('stratified-v20-'+tag);manifest=json.loads((root/'manifest.json').read_text())
  assert len(manifest)==96
  for j in range(24):
   rs=manifest[j*4:j*4+4]
   for key in ['total_depth_m','sensor_sigma','drop','return_gain','texture_scale','camera_distance_m','pattern','n_lower','exposure']:
    assert len({r[key] for r in rs})==1,(tag,j,key)
   empty_hashes=[sha(root/f"{r['id']:03d}"/'empty_rgb.png') for r in rs]
   # Full observation files contain different liquid raw; compare only empty arrays.
   empties=[np.load(root/f"{r['id']:03d}"/'observation.npz')['empty_raw'] for r in rs]
   assert all(np.array_equal(empties[0],z) for z in empties)
   images=[cv2.imread(str(root/f"{r['id']:03d}"/'empty_rgb.png'),cv2.IMREAD_UNCHANGED) for r in rs]
   pairs.append(dict(split=tag,pair=j,empty_rgb_bytes_identical=len(set(empty_hashes))==1,empty_rgb_pixels_identical=all(np.array_equal(images[0],x) for x in images),empty_depth_identical=True))
  for r in manifest:
   f=root/f"{r['id']:03d}"
   camera=json.loads((f/'camera.json').read_text())
   er=np.load(f/'observation.npz')['empty_raw']
   im=cv2.cvtColor(cv2.imread(str(f/'empty_rgb.png')),cv2.COLOR_BGR2RGB).astype(np.float32)/255
   q=assess_reference(im,er,camera);qc.append(dict(split=tag,id=r['id'],**q))
  for p in sorted(root.rglob('*')):
   if p.is_file():inputs.append(dict(path=str(p),relative=str(p.relative_to(D)),bytes=p.stat().st_size,sha256=sha(p)))
 changed=subprocess.check_output(['git','diff','--name-only','b0c3687d99ba8420791486e95fa8f603cc5e5b4c','HEAD'],text=True).splitlines()
 assert all('v20' in p.lower() for p in changed),changed
 record=dict(frozen_source_checks=sources,changed_tracked_paths=changed,reference_quality=qc,
   pair_checks=pairs,input_files=len(inputs),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,opencv=cv2.__version__,
   v17_source_untouched=True,old_v19_source_untouched=True)
 (O/'input-index-consensus.json').write_text(json.dumps(inputs,indent=2))
 (O/'verification-consensus.json').write_text(json.dumps(record,indent=2))
 print('AUDIT_FINISHED',len(inputs),'files',len(qc),'references',len(pairs),'matched configurations')
if __name__=='__main__':main()
