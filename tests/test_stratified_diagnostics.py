import math
import pytest
from liquid_depth.stratified_diagnostics import fresnel_unpolarized, slab_geometry, match_interfaces


def test_one_prediction_cannot_explain_two_interfaces():
    r = match_interfaces([.5025], [.5, .505])
    assert r['legacy_recall'] == 1 and r['recall'] == .5
    assert not r['all_truth_matched']


def test_matching_is_order_independent_and_penalizes_extra_layers():
    r = match_interfaces([.55, .5, .9], [.5, .55], relative_tolerance=0)
    assert r['recall'] == 1 and r['precision'] == 2 / 3
    assert not r['complete_set_correct']


def test_matching_maximizes_cardinality_before_error():
    r = match_interfaces([1.004, .996], [1., 1.008], absolute_tolerance_m=.006, relative_tolerance=0)
    assert r['matched_count'] == 2


def test_empty_predictions_are_misses():
    r = match_interfaces([], [.5, .55])
    assert r['recall'] == 0 and r['precision'] is None


def test_equal_indices_hide_internal_boundary_in_ideal_slab():
    a = slab_geometry([.02, .08], [1.4, 1.4], math.radians(30))
    b = slab_geometry([.06, .04], [1.4, 1.4], math.radians(30))
    assert a['geometric_interface_depths_m'] != b['geometric_interface_depths_m']
    assert a['lateral_displacement_m'] == pytest.approx(b['lateral_displacement_m'])
    assert a['paraxial_apparent_depth_m'] == pytest.approx(b['paraxial_apparent_depth_m'])
    assert fresnel_unpolarized(1.4, 1.4, math.radians(30)) == 0


def test_fresnel_normal_incidence_and_total_internal_reflection():
    assert fresnel_unpolarized(1., 1.5) == pytest.approx(.04)
    assert fresnel_unpolarized(1.5, 1., math.radians(60)) == 1


def test_no_refraction_matches_air_path():
    x = slab_geometry([.02, .08], [1., 1.], math.radians(30))
    assert x['lateral_displacement_m'] == pytest.approx(x['straight_air_displacement_m'])
    assert x['paraxial_apparent_depth_m'] == pytest.approx(.1)


def test_invalid_optical_parameters_rejected():
    with pytest.raises(ValueError):
        slab_geometry([0, .1], [1.4, 1.33])
    with pytest.raises(ValueError):
        fresnel_unpolarized(float('nan'), 1.33)
