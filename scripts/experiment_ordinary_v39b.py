"""Frozen second revision, new independent seeds; shares only scene protocol."""
import sys,json,hashlib
from pathlib import Path
import experiment_ordinary_v39 as base
from liquid_depth.ordinary_surface_v39b import estimate_v39
base.estimate_v39=estimate_v39

if __name__=='__main__':
    base.main()
    root=Path(sys.argv[sys.argv.index('--output')+1])
    protocol=json.loads((root/'protocol.json').read_text())
    protocol['estimator_revision']='v39b_observed_return_mixture'
    for f in [Path(__file__),Path(__file__).parents[1]/'src/liquid_depth/ordinary_surface_v39b.py']:
        protocol['source_hashes'][str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
