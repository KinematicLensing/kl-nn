"""TF-weighted 1D shear MAP and Laplace variance from a sample bank.

The 1D MAP is the mode of a weighted Gaussian KDE of ``g1`` and of ``g2`` on
the shear prior interval. Laplace variance is the curvature of that same 1D
log-density at the mode. Nuisance coordinates are not estimated here.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from .utils import resolve_feature_index
except ImportError:  # Direct execution from arch/.
    from utils import resolve_feature_index


SHEAR_LOW = -0.1
SHEAR_HIGH = 0.1
KDE_GRID_SIZE = 401
BANDWIDTH_FLOOR = 0.002
SILVERMAN_FACTOR = 1.06
WALL_ABS = 0.095
CURVATURE_MIN = 1e-12


def _weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float]:
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        return float("nan"), float("nan"), 0.0
    normalized = weights / total
    mean = float(np.dot(normalized, values))
    variance = float(np.dot(normalized, np.square(values - mean)))
    n_eff = float(1.0 / np.dot(normalized, normalized))
    return mean, max(variance, 0.0), n_eff


def silverman_bandwidth(values: np.ndarray, weights: np.ndarray) -> float:
    """Silverman bandwidth from the weighted sample, floored against collapse."""

    _, variance, n_eff = _weighted_moments(values, weights)
    if not math.isfinite(n_eff) or n_eff <= 1.0 or not math.isfinite(variance):
        return BANDWIDTH_FLOOR
    width = SILVERMAN_FACTOR * math.sqrt(variance) * n_eff ** (-0.2)
    if not math.isfinite(width) or width < BANDWIDTH_FLOOR:
        return BANDWIDTH_FLOOR
    return float(width)


def gaussian_kde_density(
    grid: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Evaluate a weighted Gaussian KDE on ``grid``."""

    total = float(np.sum(weights))
    delta = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * np.square(delta)) / (bandwidth * math.sqrt(2.0 * math.pi))
    return kernel @ weights / total


def _quadratic_peak(grid: np.ndarray, density: np.ndarray) -> float:
    index = int(np.argmax(density))
    if index <= 0 or index >= len(grid) - 1:
        return float(grid[index])
    y0, y1, y2 = density[index - 1], density[index], density[index + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom >= 0.0 or not math.isfinite(denom):
        return float(grid[index])
    delta = 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    step = float(grid[1] - grid[0])
    return float(grid[index] + delta * step)


def gaussian_kde_log_curvature(
    location: float,
    values: np.ndarray,
    weights: np.ndarray,
    bandwidth: float,
) -> float:
    """Return ``d^2 log p / dg^2`` of a weighted Gaussian KDE at ``location``."""

    total = float(np.sum(weights))
    delta = location - values
    kernel = np.exp(-0.5 * np.square(delta / bandwidth)) / (
        bandwidth * math.sqrt(2.0 * math.pi)
    )
    density = float(np.dot(weights, kernel) / total)
    if not math.isfinite(density) or density <= 0.0:
        return float("nan")
    first = float(np.dot(weights, (-delta / bandwidth**2) * kernel) / total)
    second = float(
        np.dot(
            weights,
            ((np.square(delta) / bandwidth**4) - (1.0 / bandwidth**2)) * kernel,
        )
        / total
    )
    return (second * density - first * first) / (density * density)


def one_dimensional_kde_map_laplace(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    low: float = SHEAR_LOW,
    high: float = SHEAR_HIGH,
) -> tuple[float, float, bool]:
    """Return ``(mode, variance, ok)`` for one bounded 1D weighted KDE."""

    invalid = (float("nan"), float("nan"), False)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape != weights.shape or values.size == 0:
        return invalid
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(finite):
        return invalid
    values = values[finite]
    weights = weights[finite]
    bandwidth = silverman_bandwidth(values, weights)
    grid = np.linspace(low, high, KDE_GRID_SIZE, dtype=np.float64)
    density = gaussian_kde_density(grid, values, weights, bandwidth)
    if not np.any(np.isfinite(density)) or np.nanmax(density) <= 0.0:
        return invalid
    mode = _quadratic_peak(grid, density)
    if not math.isfinite(mode) or abs(mode) >= WALL_ABS:
        return invalid
    curvature = gaussian_kde_log_curvature(mode, values, weights, bandwidth)
    if not math.isfinite(curvature) or curvature >= -CURVATURE_MIN:
        return invalid
    variance = -1.0 / curvature
    if not math.isfinite(variance) or variance <= 0.0:
        return invalid
    return float(mode), float(variance), True


def tf_weighted_1d_shear_map_laplace(
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D TF-weighted KDE MAP and diagonal Laplace covariance for ``g1``, ``g2``.

    ``samples`` has shape ``(galaxy, draw, feature)``. Returned maps overwrite
    only the shear coordinates of a copy of the TF-weighted mean; the caller
    should pass that mean as the starting map if it wants mean nuisances.
    """

    samples = np.asarray(samples, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("samples must have shape (galaxy, draw, feature)")
    n_galaxies, n_draws, _ = samples.shape
    if weights.shape != (n_galaxies, n_draws):
        raise ValueError("weights must have shape (galaxy, draw)")
    names = tuple(feature_names)
    g1 = resolve_feature_index(names, "g1")
    g2 = resolve_feature_index(names, "g2")
    maps = np.full((n_galaxies, 2), np.nan, dtype=np.float64)
    covariances = np.zeros((n_galaxies, 2, 2), dtype=np.float64)
    ok = np.zeros(n_galaxies, dtype=bool)
    for index in range(n_galaxies):
        component_ok = True
        for axis, feature_index in enumerate((g1, g2)):
            mode, variance, valid = one_dimensional_kde_map_laplace(
                samples[index, :, feature_index],
                weights[index],
            )
            maps[index, axis] = mode
            covariances[index, axis, axis] = variance
            component_ok = component_ok and valid
        ok[index] = component_ok
        if not component_ok:
            covariances[index] = np.nan
    return maps, covariances, ok
