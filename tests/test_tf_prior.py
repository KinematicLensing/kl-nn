import numpy as np
import pytest
from scipy.integrate import quad

from tf_prior import (
    TFPrior,
    normalize_population_log_weights,
    population_log_importance_ratio,
    posterior_importance_weights,
    truncated_tf_log_prob,
)


def test_truncated_tf_density_normalizes_in_physical_velocity():
    prior = TFPrior(scatter_dex=0.12, vcirc_min=60.0, vcirc_max=540.0)
    for magnitude in (17.0, 20.0, 23.0):
        integral = quad(
            lambda velocity: np.exp(
                truncated_tf_log_prob(velocity, magnitude, prior)
            ),
            prior.vcirc_min,
            prior.vcirc_max,
        )[0]
        assert integral == pytest.approx(1.0, rel=2e-9)
    assert np.isneginf(
        truncated_tf_log_prob(
            np.asarray((59.9, 540.1)), np.asarray((20.0, 20.0)), prior
        )
    ).all()


def test_posterior_weights_are_within_galaxy_and_preserve_joint_rows():
    prior = TFPrior(scatter_dex=0.08)
    candidates = np.asarray(
        [[80.0, 120.0, 180.0, 300.0], [90.0, 160.0, 260.0, 500.0]]
    )
    result = posterior_importance_weights(candidates, np.asarray((20.0, 18.0)), prior)
    np.testing.assert_allclose(result.weight.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.exp(result.log_weight), result.weight)
    np.testing.assert_allclose(
        result.effective_sample_size,
        1.0 / np.sum(result.weight**2, axis=1),
    )
    # The API returns weights/index-aligned ratios only; it never resamples rows.
    assert result.weight.shape == candidates.shape
    assert result.log_ratio.shape == candidates.shape
    assert np.all(result.max_weight >= 0.25)


def test_population_log_ratios_are_float64_and_normalized_only_globally():
    prior = TFPrior()
    log_ratio = population_log_importance_ratio(
        np.asarray((100.0, 180.0, 320.0)),
        np.asarray((21.0, 20.0, 19.0)),
        prior,
    )
    assert log_ratio.dtype == np.float64
    weight = normalize_population_log_weights(log_ratio)
    np.testing.assert_allclose(weight.sum(), 1.0)
    expected = np.exp(log_ratio - np.max(log_ratio))
    expected /= expected.sum()
    np.testing.assert_allclose(weight, expected)


def test_no_finite_candidate_row_raises_instead_of_silent_fallback():
    with pytest.raises(RuntimeError, match="no finite candidates"):
        posterior_importance_weights(
            np.asarray([[1.0, 2.0], [100.0, 200.0]]),
            np.asarray((20.0, 20.0)),
            TFPrior(),
        )
