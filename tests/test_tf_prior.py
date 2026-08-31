import numpy as np
import pytest
from scipy.integrate import quad

from tf_prior import (
    TFPrior,
    normalize_population_log_weights,
    population_log_importance_ratio,
    posterior_importance_from_log_ratio,
    posterior_importance_weights,
    sample_truncated_tf_vcirc,
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


def test_generic_posterior_importance_normalizes_composed_log_ratios_once():
    first = np.log(np.asarray([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]]))
    second = np.log(np.asarray([[4.0, 1.0, 2.0], [1.0, 5.0, 2.0]]))
    result = posterior_importance_from_log_ratio(first + second)
    expected = np.exp(first + second)
    expected /= np.sum(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(result.weight, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(np.sum(result.weight, axis=1), 1.0)
    np.testing.assert_allclose(
        result.effective_sample_size,
        1.0 / np.sum(np.square(expected), axis=1),
    )
    np.testing.assert_allclose(result.max_weight, np.max(expected, axis=1))


def test_generic_posterior_importance_rejects_invalid_log_ratio_rows():
    with pytest.raises(RuntimeError, match="no finite candidates"):
        posterior_importance_from_log_ratio(
            np.asarray([[0.0, -1.0], [-np.inf, -np.inf]])
        )
    with pytest.raises(ValueError, match="finite values or -inf"):
        posterior_importance_from_log_ratio(np.asarray([[0.0, np.inf]]))


def test_tf_sampler_is_the_inverse_of_the_truncated_conditional_cdf():
    prior = TFPrior(scatter_dex=0.12)
    magnitude = np.asarray((17.0, 20.0, 23.0))
    quantiles = np.asarray((0.1, 0.5, 0.9))
    velocity = sample_truncated_tf_vcirc(
        magnitude,
        prior,
        quantiles=quantiles,
    )
    mean = (magnitude - prior.intercept) / prior.slope
    lower = (np.log10(prior.vcirc_min) - mean) / prior.scatter_dex
    upper = (np.log10(prior.vcirc_max) - mean) / prior.scatter_dex
    standardized = (np.log10(velocity) - mean) / prior.scatter_dex
    from scipy.stats import truncnorm

    np.testing.assert_allclose(
        truncnorm.cdf(standardized, lower, upper),
        quantiles,
        rtol=2e-13,
        atol=2e-13,
    )
    assert np.all((velocity > prior.vcirc_min) & (velocity < prior.vcirc_max))
    assert np.isfinite(truncated_tf_log_prob(velocity, magnitude, prior)).all()


def test_tf_sampler_rng_is_reproducible_and_validates_quantiles():
    magnitude = np.full(512, 20.0)
    first = sample_truncated_tf_vcirc(
        magnitude,
        TFPrior(),
        rng=np.random.default_rng(9),
    )
    second = sample_truncated_tf_vcirc(
        magnitude,
        TFPrior(),
        rng=np.random.default_rng(9),
    )
    np.testing.assert_array_equal(first, second)
    assert not np.any(np.isin(first, (60.0, 540.0)))
    with pytest.raises(ValueError, match="strictly between"):
        sample_truncated_tf_vcirc(
            np.asarray((20.0, 20.0)),
            TFPrior(),
            quantiles=np.asarray((0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="either rng or quantiles"):
        sample_truncated_tf_vcirc(
            np.asarray((20.0,)),
            TFPrior(),
            rng=np.random.default_rng(4),
            quantiles=np.asarray((0.5,)),
        )
