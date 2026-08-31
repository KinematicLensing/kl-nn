"""Post-training Tully--Fisher importance weights.

The simulator and neural posterior are deliberately trained with an independent,
uniform ``vcirc`` prior.  This module is the only place where the assumed
Tully--Fisher (TF) population enters: after drawing joint posterior candidates.

Two weights must not be conflated:

``posterior_weight``
    Normalized across posterior candidates *within one galaxy*.  These weights
    replace the uniform training prior on ``vcirc`` with the TF prior.

``population_log_ratio``
    One unnormalised log importance ratio per simulated galaxy, evaluated at
    its true ``vcirc`` and true magnitude.  These ratios are normalized only
    after all cache partitions have been joined and are used for ensemble
    statistics such as shear calibration and coverage.

Magnitude is treated as perfectly known in the current proof-of-concept setup.
Consequently the TF width below is only the intrinsic scatter; there is no
measurement-error broadening term.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import truncnorm


@dataclass(frozen=True)
class TFPrior:
    """A truncated normal in ``log10(vcirc)`` conditional on true magnitude."""

    slope: float = -7.22
    intercept: float = 36.0
    scatter_dex: float = 0.1
    vcirc_min: float = 60.0
    vcirc_max: float = 540.0

    def __post_init__(self) -> None:
        values = (
            self.slope,
            self.intercept,
            self.scatter_dex,
            self.vcirc_min,
            self.vcirc_max,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("TF parameters must be finite")
        if self.slope == 0.0:
            raise ValueError("TF slope must be non-zero")
        if self.scatter_dex <= 0.0:
            raise ValueError("TF scatter must be positive")
        if not 0.0 < self.vcirc_min < self.vcirc_max:
            raise ValueError("vcirc bounds must satisfy 0 < min < max")

    @property
    def base_log_density(self) -> float:
        """Log density of the uniform physical-velocity training prior."""

        return -math.log(self.vcirc_max - self.vcirc_min)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PosteriorImportance:
    """Candidate-level importance products for a batch of galaxies."""

    log_ratio: np.ndarray
    log_weight: np.ndarray
    weight: np.ndarray
    effective_sample_size: np.ndarray
    effective_sample_fraction: np.ndarray
    max_weight: np.ndarray
    log_mean_ratio: np.ndarray


def sample_truncated_tf_vcirc(
    rmag_true: np.ndarray,
    prior: TFPrior,
    *,
    rng: np.random.Generator | None = None,
    quantiles: np.ndarray | None = None,
) -> np.ndarray:
    """Draw ``vcirc`` from the configured TF population.

    Sampling is performed in ``log10(vcirc)`` with the same truncated-normal
    convention used by :func:`truncated_tf_log_prob`.  Supplying ``quantiles``
    makes the transform deterministic and is useful when a Latin hypercube is
    used to cover the TF scatter.  Otherwise draws come from ``rng`` (or a new
    default NumPy generator).
    """

    magnitude = np.asarray(rmag_true, dtype=np.float64)
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("rmag_true must contain only finite values")

    if quantiles is None:
        generator = np.random.default_rng() if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        probability = generator.random(magnitude.shape)
        # Avoid exact endpoints if a custom bit generator ever returns one.
        probability = np.clip(
            probability,
            np.nextafter(0.0, 1.0),
            np.nextafter(1.0, 0.0),
        )
    else:
        if rng is not None:
            raise ValueError("provide either rng or quantiles, not both")
        probability = np.asarray(quantiles, dtype=np.float64)
        if probability.shape != magnitude.shape:
            raise ValueError("quantiles must have the same shape as rmag_true")
        if not np.all(np.isfinite(probability)) or np.any(
            (probability <= 0.0) | (probability >= 1.0)
        ):
            raise ValueError("quantiles must be finite and strictly between 0 and 1")

    mean_log10 = (magnitude - prior.intercept) / prior.slope
    lower = (math.log10(prior.vcirc_min) - mean_log10) / prior.scatter_dex
    upper = (math.log10(prior.vcirc_max) - mean_log10) / prior.scatter_dex
    standardized = truncnorm.ppf(probability, lower, upper)
    velocity = np.power(10.0, mean_log10 + prior.scatter_dex * standardized)
    if not np.all(np.isfinite(velocity)):
        raise RuntimeError("TF sampling produced non-finite velocities")
    if np.any((velocity < prior.vcirc_min) | (velocity > prior.vcirc_max)):
        raise RuntimeError("TF sampling produced velocities outside configured support")
    return velocity


def _log_standard_normal_interval(
    lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return ``log(Phi(upper) - Phi(lower))`` without tail cancellation."""

    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )
    if np.any(lower >= upper):
        raise ValueError("normal interval lower bound must be below upper bound")

    def log_difference(log_larger: np.ndarray, log_smaller: np.ndarray) -> np.ndarray:
        return log_larger + np.log(-np.expm1(log_smaller - log_larger))

    result = np.empty_like(lower)
    negative = upper <= 0.0
    positive = lower >= 0.0
    crossing = ~(negative | positive)
    if np.any(negative):
        result[negative] = log_difference(
            log_ndtr(upper[negative]), log_ndtr(lower[negative])
        )
    if np.any(positive):
        result[positive] = log_difference(
            log_ndtr(-lower[positive]), log_ndtr(-upper[positive])
        )
    if np.any(crossing):
        result[crossing] = log_difference(
            log_ndtr(upper[crossing]), log_ndtr(lower[crossing])
        )
    return result


def truncated_tf_log_prob(
    vcirc: np.ndarray,
    rmag_true: np.ndarray,
    prior: TFPrior,
) -> np.ndarray:
    """Log TF density with respect to physical velocity in km/s.

    ``vcirc`` and ``rmag_true`` follow NumPy broadcasting rules.  Values outside
    the configured velocity support have density zero (log density ``-inf``).
    """

    velocity, magnitude = np.broadcast_arrays(
        np.asarray(vcirc, dtype=np.float64),
        np.asarray(rmag_true, dtype=np.float64),
    )
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("rmag_true must contain only finite values")

    mean_log10 = (magnitude - prior.intercept) / prior.slope
    sigma = prior.scatter_dex
    lower = (math.log10(prior.vcirc_min) - mean_log10) / sigma
    upper = (math.log10(prior.vcirc_max) - mean_log10) / sigma
    log_truncation = _log_standard_normal_interval(lower, upper)

    on_support = (
        np.isfinite(velocity)
        & (velocity >= prior.vcirc_min)
        & (velocity <= prior.vcirc_max)
    )
    safe_velocity = np.where(on_support, velocity, 1.0)
    standardized = (np.log10(safe_velocity) - mean_log10) / sigma
    log_density = (
        -0.5 * np.square(standardized)
        - math.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - log_truncation
        - np.log(safe_velocity)
        - math.log(math.log(10.0))
    )
    return np.where(on_support, log_density, -np.inf)


def tf_log_prior_ratio(
    vcirc: np.ndarray,
    rmag_true: np.ndarray,
    prior: TFPrior,
) -> np.ndarray:
    """Return ``log p_TF(v|m_true) - log p_uniform(v)``."""

    return truncated_tf_log_prob(vcirc, rmag_true, prior) - prior.base_log_density


def posterior_importance_from_log_ratio(
    log_ratio: np.ndarray,
) -> PosteriorImportance:
    """Normalize candidate-level log prior ratios within each galaxy.

    Keeping this normalization independent of any particular prior lets callers
    compose several prior replacements in log space and normalize exactly once.
    Rows may contain ``-inf`` for zero target density, but each row must retain
    at least one finite candidate.
    """

    ratio = np.asarray(log_ratio, dtype=np.float64)
    if ratio.ndim != 2:
        raise ValueError("log_ratio must have shape (galaxy, candidate)")
    if ratio.shape[1] == 0:
        raise ValueError("each galaxy must have at least one posterior candidate")
    if np.any(np.isnan(ratio)) or np.any(np.isposinf(ratio)):
        raise ValueError("log_ratio may contain finite values or -inf only")

    finite_row = np.any(np.isfinite(ratio), axis=1)
    if not np.all(finite_row):
        invalid = np.flatnonzero(~finite_row).tolist()
        raise RuntimeError(
            "importance sampling has no finite candidates for galaxy rows "
            f"{invalid}; increase the candidate bank or inspect posterior support"
        )

    log_normalizer = logsumexp(ratio, axis=1, keepdims=True)
    log_weight = ratio - log_normalizer
    weight = np.exp(log_weight)
    if not np.all(np.isfinite(weight)):
        raise RuntimeError("importance sampling produced non-finite weights")
    weight_sum = np.sum(weight, axis=1)
    if not np.allclose(weight_sum, 1.0, rtol=2e-12, atol=2e-12):
        raise RuntimeError("posterior importance weights do not normalize to one")

    ess = 1.0 / np.sum(np.square(weight), axis=1)
    return PosteriorImportance(
        log_ratio=ratio,
        log_weight=log_weight,
        weight=weight,
        effective_sample_size=ess,
        effective_sample_fraction=ess / ratio.shape[1],
        max_weight=np.max(weight, axis=1),
        log_mean_ratio=log_normalizer[:, 0] - math.log(ratio.shape[1]),
    )


def posterior_importance_weights(
    vcirc_candidates: np.ndarray,
    rmag_true: np.ndarray,
    prior: TFPrior,
) -> PosteriorImportance:
    """Compute normalized TF weights across candidates within each galaxy.

    Parameters
    ----------
    vcirc_candidates
        Physical velocities with shape ``(galaxy, candidate)``.
    rmag_true
        One true magnitude per galaxy, shape ``(galaxy,)``.
    """

    velocity = np.asarray(vcirc_candidates, dtype=np.float64)
    magnitude = np.asarray(rmag_true, dtype=np.float64)
    if velocity.ndim != 2:
        raise ValueError("vcirc_candidates must have shape (galaxy, candidate)")
    if magnitude.shape != (velocity.shape[0],):
        raise ValueError("rmag_true must contain one value per galaxy")
    if velocity.shape[1] == 0:
        raise ValueError("each galaxy must have at least one posterior candidate")

    return posterior_importance_from_log_ratio(
        tf_log_prior_ratio(velocity, magnitude[:, None], prior)
    )


def population_log_importance_ratio(
    vcirc_true: np.ndarray,
    rmag_true: np.ndarray,
    prior: TFPrior,
) -> np.ndarray:
    """Return the unnormalised population log ratio for simulated galaxies."""

    velocity = np.asarray(vcirc_true, dtype=np.float64)
    magnitude = np.asarray(rmag_true, dtype=np.float64)
    if velocity.shape != magnitude.shape or velocity.ndim != 1:
        raise ValueError("vcirc_true and rmag_true must be matching vectors")
    log_ratio = tf_log_prior_ratio(velocity, magnitude, prior)
    if not np.all(np.isfinite(log_ratio)):
        invalid = np.flatnonzero(~np.isfinite(log_ratio)).tolist()
        raise ValueError(
            "population TF ratios are non-finite for galaxy rows " f"{invalid}"
        )
    return log_ratio


def normalize_population_log_weights(log_ratio: np.ndarray) -> np.ndarray:
    """Globally normalize concatenated galaxy-level log importance ratios."""

    values = np.asarray(log_ratio, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("population log ratios must be a non-empty vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("population log ratios must be finite")
    normalized = np.exp(values - logsumexp(values))
    if not np.isclose(np.sum(normalized), 1.0, rtol=2e-12, atol=2e-12):
        raise RuntimeError("population weights do not normalize to one")
    return normalized


def effective_sample_size(weight: np.ndarray) -> float:
    """Effective number of independent equal-weight observations."""

    values = np.asarray(weight, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0):
        raise ValueError("weights must be a non-empty non-negative vector")
    total = np.sum(values)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive finite total")
    values = values / total
    return float(1.0 / np.sum(np.square(values)))
