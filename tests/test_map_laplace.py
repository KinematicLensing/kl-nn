import math

import numpy as np

from map_laplace import (
    WALL_ABS,
    one_dimensional_kde_map_laplace,
    tf_weighted_1d_shear_map_laplace,
)


def test_kde_map_and_laplace_recover_gaussian():
    rng = np.random.default_rng(0)
    values = rng.normal(0.02, 0.015, size=8000)
    weights = np.ones_like(values)
    mode, variance, ok = one_dimensional_kde_map_laplace(values, weights)
    assert ok
    assert abs(mode - 0.02) < 0.003
    assert math.isfinite(variance) and 5e-5 < variance < 8e-4


def test_kde_marks_wall_pileup_invalid():
    values = np.full(2000, 0.099)
    weights = np.ones_like(values)
    mode, variance, ok = one_dimensional_kde_map_laplace(values, weights)
    assert not ok
    assert abs(mode) >= WALL_ABS or not np.isfinite(variance)


def test_tf_weighted_1d_maps_overwrite_only_shear():
    rng = np.random.default_rng(1)
    n_galaxies, n_draws = 3, 4000
    names = (
        "g1",
        "g2",
        "theta_int",
        "cosi",
        "v0",
        "vcirc",
        "rscale",
        "hlr",
        "halpha_flux_true",
    )
    samples = np.zeros((n_galaxies, n_draws, 9), dtype=np.float64)
    samples[..., 0] = rng.normal(0.01, 0.02, size=(n_galaxies, n_draws))
    samples[..., 1] = rng.normal(-0.015, 0.018, size=(n_galaxies, n_draws))
    samples[..., 5] = 200.0
    weights = np.ones((n_galaxies, n_draws), dtype=np.float64)
    maps, cov, ok = tf_weighted_1d_shear_map_laplace(
        samples, weights, feature_names=names
    )
    assert maps.shape == (n_galaxies, 2)
    assert cov.shape == (n_galaxies, 2, 2)
    assert ok.all()
    np.testing.assert_allclose(maps[:, 0], 0.01, atol=0.004)
    np.testing.assert_allclose(maps[:, 1], -0.015, atol=0.004)
    assert np.all(cov[:, 0, 1] == 0.0)
    assert np.all(cov[:, 1, 0] == 0.0)
    assert np.all(cov[:, 0, 0] > 0.0)
    assert np.all(cov[:, 1, 1] > 0.0)
