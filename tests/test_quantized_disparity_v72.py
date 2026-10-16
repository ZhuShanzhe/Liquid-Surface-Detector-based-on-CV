"""Quantized stereo returns must not be mistaken for two physical surfaces."""
import numpy as np

from liquid_depth.quantized_disparity_v72 import fit_quantized_disparity


def _fit(depth, heights):
    return fit_quantized_disparity(
        depth, heights, camera_z_m=10., bottom_z_m=0.,
        fx_px=314., baseline_m=.035, disparity_step_px=1/16,
        disparity_noise_px=.26, max_return_depth_m=10.,
        coarse_height_m=10 - 314*.035/1.1875,
        upper_height_m=10 - 314*.035/1.25, radius_m=.19)


def test_adjacent_bins_recover_sub_bin_surface():
    rng = np.random.default_rng(72)
    scale = 314 * .035
    expected = scale / 9.
    disparity = np.rint((expected + rng.normal(0, .26, 5000)) * 16) / 16
    depth = scale / disparity
    depth = depth[(depth > 0) & (depth <= 10)]
    fit = _fit(depth, 10 - depth)
    assert fit['ambiguous'] and fit['applied']
    assert abs(fit['height_m'] - 1.) < .08


def test_unquantized_returns_do_not_activate():
    depth = np.linspace(8.7, 9.3, 500)
    fit = _fit(depth, 10 - depth)
    assert not fit['ambiguous'] and not fit['applied']


def test_ambiguous_fit_without_surviving_returns_falls_back():
    scale = 314 * .035
    disparity = np.tile(np.array([1.1875, 1.25]), 150)
    depth = scale / disparity
    fit = fit_quantized_disparity(
        depth, 10 - depth, camera_z_m=10., bottom_z_m=0.,
        fx_px=314., baseline_m=.035, disparity_step_px=1/16,
        disparity_noise_px=.26, max_return_depth_m=1.,
        coarse_height_m=10 - scale/1.1875,
        upper_height_m=10 - scale/1.25, radius_m=.19)
    assert fit['ambiguous']
    assert not fit['applied']
    assert fit['reason'] == 'too_few_quantized_returns'
