import numpy as np
import pytest
from scipy.integrate import quad

from inclination_prior import (
    InclinationPrior,
    isotropic_inclination_log_prior_ratio,
)


def test_full_support_ratio_matches_isotropic_jacobian():
    prior = InclinationPrior()
    sini = np.asarray((0.1, 0.5, 0.9), dtype=np.float32)

    actual = isotropic_inclination_log_prior_ratio(sini, prior)
    expected = np.log(sini.astype(np.float64)) - 0.5 * np.log1p(
        -np.square(sini.astype(np.float64))
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_full_support_target_density_normalizes_and_zero_has_zero_density():
    prior = InclinationPrior()
    integral = quad(
        lambda value: np.exp(isotropic_inclination_log_prior_ratio(value, prior)),
        prior.sini_min,
        prior.sini_max,
        epsabs=2e-11,
        epsrel=2e-11,
    )[0]

    assert integral == pytest.approx(1.0, rel=2e-10, abs=2e-10)
    assert np.isneginf(isotropic_inclination_log_prior_ratio(0.0, prior))


def test_truncated_support_ratio_has_the_correct_normalization():
    prior = InclinationPrior(sini_min=0.2, sini_max=0.8)
    sini = np.asarray((0.2, 0.4, 0.8))
    lower_cos = np.sqrt(1.0 - prior.sini_min**2)
    upper_cos = np.sqrt(1.0 - prior.sini_max**2)
    expected_ratio = (
        sini
        / np.sqrt(1.0 - sini**2)
        * (prior.sini_max - prior.sini_min)
        / (lower_cos - upper_cos)
    )

    actual = np.exp(isotropic_inclination_log_prior_ratio(sini, prior))
    np.testing.assert_allclose(actual, expected_ratio, rtol=2e-15, atol=2e-15)

    training_density = 1.0 / (prior.sini_max - prior.sini_min)
    integral = quad(
        lambda value: training_density
        * np.exp(isotropic_inclination_log_prior_ratio(value, prior)),
        prior.sini_min,
        prior.sini_max,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]
    assert integral == pytest.approx(1.0, rel=2e-11, abs=2e-11)


@pytest.mark.parametrize(
    ("sini_min", "sini_max"),
    (
        (-0.1, 0.8),
        (0.2, 0.2),
        (0.8, 0.2),
        (0.2, 1.1),
        (np.nan, 0.8),
        (0.2, np.inf),
    ),
)
def test_invalid_support_is_rejected(sini_min, sini_max):
    with pytest.raises(ValueError, match="sini bounds"):
        InclinationPrior(sini_min=sini_min, sini_max=sini_max)


def test_values_outside_support_are_rejected_without_clipping():
    prior = InclinationPrior(sini_min=0.2, sini_max=0.8)
    with pytest.raises(ValueError, match="configured support"):
        isotropic_inclination_log_prior_ratio(np.asarray((0.19, 0.4)), prior)
    with pytest.raises(ValueError, match="configured support"):
        isotropic_inclination_log_prior_ratio(np.asarray((0.4, 0.81)), prior)
    with pytest.raises(ValueError, match="must be finite"):
        isotropic_inclination_log_prior_ratio(np.asarray((0.4, np.nan)), prior)


def test_exact_full_support_singularity_is_rejected_without_capping():
    with pytest.raises(ValueError, match="singular upper endpoint"):
        isotropic_inclination_log_prior_ratio(
            np.asarray((0.5, 1.0)), InclinationPrior()
        )


def test_truncated_upper_endpoint_is_finite():
    prior = InclinationPrior(sini_min=0.1, sini_max=0.95)
    result = isotropic_inclination_log_prior_ratio(prior.sini_max, prior)
    assert np.isfinite(result)


def test_manifest_dict_records_exact_support():
    assert InclinationPrior(0.1, 0.9).to_dict() == {
        "sini_min": 0.1,
        "sini_max": 0.9,
    }
