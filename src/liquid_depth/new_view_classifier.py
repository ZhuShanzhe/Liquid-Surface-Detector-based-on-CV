"""Four nonexclusive observation tags; labels are not metrology confidence."""
import hashlib,json,time
from pathlib import Path
import joblib,numpy as np,torch
from .new_view_native import NativeMIL
from .new_view_features import features
from .new_view_glare import GlarePredictor
from .compiled_native_forest import CompiledForest
TAGS=('low_light','glare','uneven_surface','floating_objects')
class NewViewClassifier:
    def __init__(self,bundle,asset_dir=None):
        import sklearn
        directory=Path(asset_dir) if asset_dir else Path(__file__).resolve().parents[2]/'assets/models/new_view_v14'
        manifest=json.loads((directory/'manifest.json').read_text())
        if sklearn.__version__!=manifest['sklearn_version']:raise RuntimeError('trusted_forest_sklearn_version_mismatch')
        for name,expected in manifest['sha256'].items():
            if Path(name).name!=name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('formal_classifier_asset_checksum_mismatch')
        if not torch.cuda.is_available():raise RuntimeError('formal_auxiliary_cuda_profile_requires_gpu')
        self.bundle=bundle;self.native=NativeMIL().cuda().eval()
        self.native.load_state_dict(torch.load(directory/'native_best.pt',map_location='cpu',weights_only=True)['state_dict'])
        state=joblib.load(directory/'candidate.joblib')
        self.thresholds=np.asarray(state['thresholds']);self.models=[CompiledForest(m) for m in state['models']]
        self.glare=GlarePredictor(directory/'glare_head.pt');self.last_ms=None
    def predict(self,rgb_bgr,*,depth_m=None,dry_depth_m=None,roi=None):
        begin=time.perf_counter();x=features(rgb_bgr,depth_m,dry_depth_m,self.bundle,self.native)
        scores=np.array([m.predict_proba(x[None])[0,1] for m in self.models])
        scores[1]=self.glare.predict(rgb_bgr,depth_m,self.bundle);pred=scores>=self.thresholds
        self.last_ms=(time.perf_counter()-begin)*1000
        return dict(material='ordinary_diffuse',material_score=1.,material_ambiguous=False,
            material_source='independently_confirmed_installation_not_visual_prediction',
            tags=[t for t,p in zip(TAGS,pred) if p],tag_scores=dict(zip(TAGS,map(float,scores))),
            float_peak_recommended=bool(pred[3]),model_id='formal_tof_v14_20261021',probabilities_are_metrology_confidence=False)
