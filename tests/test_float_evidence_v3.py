import numpy as np
import pytest
import torch

from liquid_depth.float_evidence_v3 import (
    MultiScaleEvidencePredictor,
    OptionalFloatMeterV3,
    choose_peak,
    region_candidates,
    score_regions,
    sensor_sigma,
)
from liquid_depth.frozen_symmetric_geometry import FrozenSymmetricGeometryMeter
from liquid_depth.rgbd_evidence_model import RGBDEvidenceNet
from liquid_depth.supported_scene_meter import FormalSceneMeter
from liquid_depth.uneven_surface_meter import UnevenSurfaceMeter


def base_meter():
    pose = np.diag([1., -1., -1., 1.]); pose[2, 3] = 6.
    bundle = {'source': 'test_empty_geometry', 'sensor_family': 'tof', 'image_size': [80, 60], 'rim_z_m': 2.,
        'calibration': {'intrinsics': [[100., 0., 40.], [0., 100., 30.], [0., 0., 1.]],
            'camera_to_world_cv': pose.tolist(), 'bottom_z_m': 0., 'radii_m': [1., 1.], 'footprint_exponent': 2.}}
    return FormalSceneMeter.from_empty_geometry(bundle, np.full((60, 80), 6., np.float32))


class Prediction:
    def predict(self, features):
        h, w = features.shape[-2:]
        semantic = np.zeros((4, h, w), np.float32); semantic[2] = 1
        origin = np.zeros((7, h, w), np.float32); origin[1] = 1
        semantic[:, 25:32, 35:42] = 0; semantic[3, 25:32, 35:42] = 1
        origin[:, 25:32, 35:42] = 0; origin[2, 25:32, 35:42] = 1
        return {'semantic': semantic, 'origin': origin, 'reliability': np.ones((1, h, w), np.float32)}


class Constant:
    def __init__(self, value): self.value = value
    def predict(self, x): return np.full(len(x), self.value)
    def predict_proba(self, x): return np.tile([1-self.value, self.value], (len(x), 1))


def verifier(probability=1., interval=0.):
    return {'classifier': Constant(probability), 'regressor': Constant(0.), 'lower': Constant(0.),
            'upper': Constant(interval), 'policy': {'threshold': .9, 'interval_factor': 2.}}


def test_off_needs_no_rgb_or_models_and_matches_legacy_peak():
    base = base_meter(); dry = np.full((60, 80), 6.); raw = np.full((60, 80), 4.6)
    result = OptionalFloatMeterV3(base, dry).measure(depth_m=raw)
    assert result['highest_height_m'] == UnevenSurfaceMeter(base, dry).measure(raw)['highest_visible_height_m']
    assert result['liquid_depth_m'] == FrozenSymmetricGeometryMeter(base).measure(depth_m=raw)['liquid_depth_m']
    assert not result['visual_module_executed']


def test_float_peak_cannot_change_liquid():
    base = base_meter(); dry = np.full((60, 80), 6.); raw = np.full((60, 80), 4.6)
    raw[25:32, 35:42] = 4.35
    meter = OptionalFloatMeterV3(base, dry, Prediction()); meter.verifier = verifier()
    result = meter.measure(depth_m=raw, rgb_bgr=np.zeros((60, 80, 3), np.uint8), visual_mode='auto')
    assert result['liquid_depth_m'] == FrozenSymmetricGeometryMeter(base).measure(depth_m=raw)['liquid_depth_m']
    assert abs(result['highest_height_m']-1.65) < .005
    assert result['visual_guidance_used']
    assert not result['liquid_override_used']


def test_rejected_float_preserves_geometric_peak():
    base = base_meter(); dry = np.full((60, 80), 6.); raw = np.full((60, 80), 4.6)
    meter = OptionalFloatMeterV3(base, dry, Prediction()); meter.verifier = verifier(.1)
    result = meter.measure(depth_m=raw, rgb_bgr=np.zeros((60, 80, 3), np.uint8), visual_mode='auto')
    assert not result['visual_guidance_used']
    assert result['highest_height_m'] == UnevenSurfaceMeter(base, dry).measure(raw)['highest_visible_height_m']


def test_large_height_uncertainty_rejects_region_not_numeric_frame():
    candidate = {'object_score': .999, 'height_m': .2, 'interval_width_m': .5}
    chosen, accepted = choose_peak([candidate], {'threshold': .9, 'interval_factor': 2.})
    assert chosen is None and accepted == []


def test_crossed_quantiles_do_not_create_zero_uncertainty():
    model = verifier(); model['lower'] = Constant(2.); model['upper'] = Constant(-2.)
    candidate = {'features': np.zeros(34, np.float32), 'height_raw_m': .2, 'noise_m': .1}
    scored = score_regions([candidate], model)
    assert abs(scored[0]['interval_width_m']-.4) < 1e-6


def test_empty_depth_does_not_create_visual_height():
    base = base_meter(); meter = OptionalFloatMeterV3(base, np.full((60, 80), 6.), Prediction())
    meter.verifier = verifier()
    result = meter.measure(depth_m=np.zeros((60, 80)), rgb_bgr=np.zeros((60, 80, 3), np.uint8), visual_mode='auto')
    assert result['highest_height_m'] is None
    assert result['liquid_depth_m'] is None


def test_noise_grows_with_distance():
    base = base_meter()
    assert sensor_sigma(np.array([1., 9.]), base.bundle)[1] > sensor_sigma(np.array([1., 9.]), base.bundle)[0]


def test_multiscale_feature_fusion_preserves_original_pixel_grid():
    predictor = MultiScaleEvidencePredictor.__new__(MultiScaleEvidencePredictor)
    predictor.net = RGBDEvidenceNet().eval(); predictor.device = 'cpu'; predictor.temperature = {}
    torch.set_num_threads(2)
    result = predictor.predict(np.zeros((12, 24, 40), np.float32))
    assert result['semantic'].shape == (4, 24, 40)
    assert result['origin'].shape == (7, 24, 40)
    assert result['reliability'].shape == (1, 24, 40)
    assert np.allclose(result['semantic'].sum(0), 1., atol=1e-5)


def test_low_metric_support_has_no_float_candidates():
    base = base_meter(); raw = np.full((60, 80), 4.6); rgb = np.zeros((60, 80, 3), np.uint8)
    pred = Prediction().predict(np.zeros((12, 60, 80))); pred['reliability'][:] = 0
    assert region_candidates(base, np.full_like(raw, 6.), rgb, raw, pred, 1.4) == []


def test_visual_auto_explicitly_requires_models():
    base = base_meter(); meter = OptionalFloatMeterV3(base, np.full((60, 80), 6.))
    with pytest.raises(ValueError, match='optional_model'):
        meter.measure(depth_m=np.full((60, 80), 4.6), visual_mode='auto')


def test_unknown_visual_mode_is_rejected():
    base = base_meter()
    with pytest.raises(ValueError, match='visual_mode'):
        OptionalFloatMeterV3(base, np.full((60, 80), 6.)).measure(depth_m=np.zeros((60, 80)), visual_mode='truth')


def test_numeric_depth_stratum_is_safe_for_report_labels():
    from evaluate_float_v3 import stratum_labels
    assert stratum_labels({'sensor': 'tof', 'band': 3}) == ['all', 'sensor=tof', 'band=3']


def test_new_regions_are_not_hidden_by_inherited_empty_legacy_field():
    from evaluate_float_v3 import used_float_region
    value = {'diagnostics': {'float_components': 0, 'verified_regions': [{'pixels': 5}]}}
    assert used_float_region('v3_auto', value)
    assert not used_float_region('v2_auto', value)
