"""Importance ratios for replacing a uniform-``sini`` training prior.

The neural posterior is trained with a uniform prior on ``sini = sin(i)``.
For an isotropically oriented population, ``cos(i)`` is uniform instead.  This
module evaluates a stabilized approximation to that density ratio on the same
(possibly truncated) ``sini`` support so it can be combined with other
candidate-level log weights without a singular weight at ``sini=1``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class InclinationPrior:
    """Bounds and stabilizer for the inclination prior replacement.

    The exact uniform-``cosi`` density has denominator ``sqrt(1-sini**2)``.
    Adding ``denominator_stabilizer`` inside that radicand keeps the importance
    ratio finite at the upper endpoint.  The stabilized density is normalized
    over the configured support, which remains identical to the training-prior
    support even when the model bounds are truncated.
    """

    sini_min: float = 0.0
    sini_max: float = 1.0
    denominator_stabilizer: float = 0.1

    def __post_init__(self) -> None:
        if not math.isfinite(self.sini_min) or not math.isfinite(self.sini_max):
            raise ValueError("sini bounds must be finite")
        if not 0.0 <= self.sini_min < self.sini_max <= 1.0:
            raise ValueError("sini bounds must satisfy 0 <= min < max <= 1")
        if (
            not math.isfinite(self.denominator_stabilizer)
            or self.denominator_stabilizer <= 0.0
        ):
            raise ValueError("denominator stabilizer must be finite and positive")

    @property
    def denominator_constant(self) -> float:
        """Constant in the stabilized denominator ``sqrt(c-sini**2)``."""

        return 1.0 + self.denominator_stabilizer

    @property
    def log_ratio_normalization(self) -> float:
        """Log of the support-normalization factor in the prior ratio.

        Directly evaluating
        ``(b-a)/(sqrt(c-a**2)-sqrt(c-b**2))`` can lose precision for a narrow
        interval.  Rationalizing the denominator gives the equivalent and
        stable expression ``(sqrt(c-a**2)+sqrt(c-b**2))/(a+b)``.
        """

        lower_term = math.sqrt(
            self.denominator_constant - self.sini_min**2
        )
        upper_term = math.sqrt(
            self.denominator_constant - self.sini_max**2
        )
        return math.log(lower_term + upper_term) - math.log(
            self.sini_min + self.sini_max
        )

    def to_dict(self) -> dict[str, float]:
        """Return serializable bounds for cache manifests."""

        return asdict(self)


def isotropic_inclination_log_prior_ratio(
    sini: np.ndarray | float,
    prior: InclinationPrior,
) -> np.ndarray:
    """Return the stabilized inclination-prior log density ratio.

    For ``s = sin(i)`` on ``[a, b]``, the normalized ratio is

    ``s / sqrt(c-s**2) * (b-a) / (sqrt(c-a**2)-sqrt(c-b**2))``,

    where ``c = 1 + prior.denominator_stabilizer`` (``1.1`` by default).

    All calculations are performed in float64.  Values outside the configured
    support are rejected because silently assigning zero weight would conceal
    a mismatch between the model and target supports.  The stabilizer makes
    ``sini=1`` finite, while ``sini=0`` correctly has zero target density and
    returns ``-inf``.
    """

    values = np.asarray(sini, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("sini values must be finite")
    if np.any((values < prior.sini_min) | (values > prior.sini_max)):
        raise ValueError(
            "sini values must lie within the configured support "
            f"[{prior.sini_min}, {prior.sini_max}]"
        )

    # Divide-by-zero in log(0) is intentional: an exactly face-on galaxy has
    # zero density under both the exact and stabilized target populations.
    with np.errstate(divide="ignore"):
        return (
            np.log(values)
            - 0.5
            * np.log(prior.denominator_constant - np.square(values))
            + prior.log_ratio_normalization
        )
