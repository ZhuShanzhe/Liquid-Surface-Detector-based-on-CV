import pytest

from liquid_depth.supported_scene_meter import SCENES, FormalSceneMeter, _measurement


def test_only_three_supported_scene_types():
    assert SCENES == {
        'ordinary_diffuse', 'homogeneous_transparent',
        'homogeneous_translucent', 'partial_depth_loss',
    }
    with pytest.raises(ValueError, match='unsupported_scene_route'):
        FormalSceneMeter().measure('glare')


def test_homogeneous_scene_requires_its_own_calibration():
    with pytest.raises(ValueError, match='scene_specific_five_prior_calibration_required'):
        FormalSceneMeter().measure('homogeneous_transparent', stereo_images={})
    with pytest.raises(ValueError, match='one_empty_plus_four'):
        FormalSceneMeter.from_five_priors({}, {}, [], [0.0, 0.1],
                                           scene='homogeneous_translucent')


def test_negative_measurement_is_not_snapped_or_clipped():
    result = _measurement('ordinary_diffuse', 'broad_container_median_v51b',
                          -0.003, {})
    assert result['liquid_depth_m'] == -0.003
    assert result['numeric_output'] is True
    assert result['field_metrology_qualified'] is False
    assert result['software_route_status'] == 'supported_three_scene_v3'


def test_partial_loss_requires_empty_geometry_first():
    with pytest.raises(ValueError, match='empty_geometry_and_dry_depth_reference_required'):
        FormalSceneMeter().measure('partial_depth_loss', depth_m=None)


def test_structured_profile_enables_supported_partial_route():
    profile = {'disparity_step_px': 1/16, 'disparity_noise_px': .26}
    meter = FormalSceneMeter(bundle={'sensor_family': 'structured_light'},
                             max_return_depth_m=10.,
                             quantized_disparity_profile=profile)
    assert meter.elastic_evidence_enabled
    assert meter.quantized_disparity_profile == profile
    with pytest.raises(ValueError, match="max_return_depth"):
        FormalSceneMeter(bundle={'sensor_family': 'structured_light'},
                         quantized_disparity_profile=profile)
