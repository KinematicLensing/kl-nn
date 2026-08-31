"""Importance ratios for replacing a uniform-``sini`` training prior.

The neural posterior is trained with a uniform prior on ``sini = sin(i)``.
For an isotropically oriented population, ``cos(i)`` is uniform instead.  This
module evaluates the exact density ratio on the same (possibly truncated)
``sini`` support so it can be combined with other candidate-level log weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class InclinationPrior:
    """Bounds shared by the uniform-``sini`` and uniform-``cosi`` priors.

    The target prior is uniform over the ``cos(i)`` interval corresponding to
    ``sini_min <= sin(i) <= sini_max``.  This makes the target and training
    priors have identical support, including when the model bounds are
    truncated.
    """

    sini_min: float = 0.0
    sini_max: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sini_min) or not math.isfinite(self.sini_max):
            raise ValueError("sini bounds must be finite")
        if not 0.0 <= self.sini_min < self.sini_max <= 1.0:
            raise ValueError("sini bounds must satisfy 0 <= min < max <= 1")

    @property
    def log_ratio_normalization(self) -> float:
        """Log of the support-normalization factor in the prior ratio.

        Directly evaluating ``(b-a)/(sqrt(1-a**2)-sqrt(1-b**2))`` can lose
        precision for a narrow interval.  Rationalizing the denominator gives
        the equivalent and stable expression
        ``(sqrt(1-a**2)+sqrt(1-b**2))/(a+b)``.
        """

        lower_cos = math.sqrt((1.0 - self.sini_min) * (1.0 + self.sini_min))
        upper_cos = math.sqrt((1.0 - self.sini_max) * (1.0 + self.sini_max))
        return math.log(lower_cos + upper_cos) - math.log(self.sini_min + self.sini_max)

    def to_dict(self) -> dict[str, float]:
        """Return serializable bounds for cache manifests."""

        return asdict(self)


def isotropic_inclination_log_prior_ratio(
    sini: np.ndarray | float,
    prior: InclinationPrior,
) -> np.ndarray:
    """Return ``log p_uniform-cosi(sini) - log p_uniform-sini(sini)``.

    For ``s = sin(i)`` on ``[a, b]``, the normalized ratio is

    ``s / sqrt(1-s**2) * (b-a) / (sqrt(1-a**2)-sqrt(1-b**2))``.

    All calculations are performed in float64.  Values outside the configured
    support are rejected because silently assigning zero weight would conceal
    a mismatch between the model and target supports.  At full support,
    ``sini=1`` is an integrable density singularity but is not a finite
    candidate weight, so that exact endpoint is rejected rather than clipped.
    ``sini=0`` correctly has zero isotropic density and returns ``-inf``.
    """

    values = np.asarray(sini, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("sini values must be finite")
    if np.any((values < prior.sini_min) | (values > prior.sini_max)):
        raise ValueError(
            "sini values must lie within the configured support "
            f"[{prior.sini_min}, {prior.sini_max}]"
        )
    if prior.sini_max == 1.0 and np.any(values == 1.0):
        raise ValueError(
            "sini=1 is the singular upper endpoint of the isotropic "
            "inclination density; exact endpoint candidates are not allowed"
        )

    # log1p retains accuracy when sini is close to one.  Divide-by-zero in
    # log(0) is intentional: an exactly face-on galaxy has zero density under
    # a continuous uniform-cosi population.
    with np.errstate(divide="ignore"):
        return (
            np.log(values)
            - 0.5 * np.log1p(-np.square(values))
            + prior.log_ratio_normalization
        )
