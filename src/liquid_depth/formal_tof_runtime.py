"""Official ToF factory: V14 geometry plus verified V13 auxiliary routing.

Material confirmation and raw phase requirements are unchanged. Scene tags are
diagnostics; numeric error, availability and latency determine validation.
"""
import time
from pathlib import Path
import cv2,numpy as np,torch
from .unified_tof import UnifiedToFMeter
VERSION='formal_tof_v14_20261021'
def create_formal_meter(installation,*,profile_path,classifier=None,compound_repair=True):
    if installation.confirmed_material!='ordinary_diffuse' or not compound_repair:
        if classifier is None:
            from .scene_classifier import SceneClassifier
            classifier=SceneClassifier(Path(__file__).resolve().parents[2]/'assets/models/scene_router/scene_head.pt')
        meter=UnifiedToFMeter(installation,classifier,profile_path=profile_path,compound_repair=compound_repair)
        meter.formal_runtime=dict(version=VERSION,material_route='preserved_phase' if installation.confirmed_material!='ordinary_diffuse' else 'explicit_geometry_repair_disabled')
        return meter
    from .new_view_classifier import NewViewClassifier
    from .compiled_native_forest import CompiledForest
    from .rgbd_evidence_model import RGBDEvidencePredictor
    from .small_float_peak_core import float_features
    from .compound_tof_repair import sanitize_capture
    start=time.perf_counter();head=NewViewClassifier(installation.geometry)
    meter=UnifiedToFMeter(installation,head,profile_path=profile_path,compound_repair=True)
    # Keep the archived symmetric baseline source byte-identical. The execution
    # optimized copy is installed only on THIS formal instance, never globally.
    from .formal_symmetric_meter import SymmetricVisualMeter
    frozen=meter.geometric.backend.backend.legacy.adaptive.frozen
    previous=frozen._meter
    frozen._meter=SymmetricVisualMeter(previous.base,predictor=previous.predictor,policy=previous.policy)
    meter.warm_optional_peak_models()
    peak=meter._small_float_refiner()
    peak.models={name:CompiledForest(model) for name,model in peak.models.items()}
    pixel=Path(__file__).resolve().parents[2]/'assets/models/small_float_peak/pixel_best.pt'
    peak.predictor=RGBDEvidencePredictor(pixel,device='cuda')
    dry,_=sanitize_capture(installation.dry_depth_m,.08,np.asarray(installation.dry_depth_m)>0)
    rgb=np.zeros((*dry.shape,3),np.uint8)
    head.predict(rgb,depth_m=dry,dry_depth_m=dry,roi=meter.roi)
    xx,_=float_features(rgb,dry,dry,installation.geometry);peak.predictor.predict(xx)
    torch.cuda.synchronize()
    meter.formal_runtime=dict(version=VERSION,material_route='ordinary_geometry',geometry='dynamic_wet_interior_v14_20261021',
        classification='new_view_v13_frozen',device='cuda',warmup_ms=(time.perf_counter()-start)*1000,
        truth_used_for_measurement=False,semantic_accuracy_is_release_gate=False,field_metrology_qualified=False)
    return meter
