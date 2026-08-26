#!/usr/bin/env python3
"""Build a proposal-versus-TF shear report from current posterior caches."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import html
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from cache_contract import load_cache_partitions, load_partitioned_array
from tf_prior import effective_sample_size, normalize_population_log_weights
from utils import img_to_gal_axis


SHEAR_PP_BIN_EDGES = np.asarray(
    [-0.1, -1.0 / 30.0, 1.0 / 30.0, 0.1], dtype=np.float64
)
LOW_G_DEFAULT = 0.02
ADDITIVE_DISPLAY_SCALE = 1.0e4
MULTIPLICATIVE_DISPLAY_SCALE = 1.0e2
DEFAULT_PRECISION_CAP_PERCENTILE = 95.0
PRECISION_CAP_SWEEP = (90.0, 95.0, 99.0, None)
LOGGER = logging.getLogger(__name__)

NUISANCE_DISPLAY = {
    "theta_int": (1.0, "rad"),
    "sini": (1.0, ""),
    "v0": (1.0, "km s<sup>-1</sup>"),
    "vcirc": (1.0, "km s<sup>-1</sup>"),
    "rscale": (1.0, "arcsec"),
    "hlr": (1.0, "arcsec"),
    "halpha_flux_true": (
        1.0e16,
        "10<sup>-16</sup> erg s<sup>-1</sup> cm<sup>-2</sup>",
    ),
}
NUISANCE_PLOT_UNIT = {
    "theta_int": "rad",
    "sini": "",
    "v0": r"km s$^{-1}$",
    "vcirc": r"km s$^{-1}$",
    "rscale": "arcsec",
    "hlr": "arcsec",
    "halpha_flux_true": r"$10^{-16}$ erg s$^{-1}$ cm$^{-2}$",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path('/ocean/projects/phy250048p/shared/cache/'))
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="MODEL:DATASET cache pair; repeat to compare cases.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--low-g", type=float, default=LOW_G_DEFAULT)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--weighted",
        action="store_true",
        help=(
            "Compose each population weight with posterior shear precision, "
            "capped at its population-weighted 95th percentile."
        ),
    )
    return parser.parse_args(argv)


def load_case(
    cache_root: Path,
    case: str,
    max_galaxies: int | None = None,
) -> dict:
    try:
        model, dataset = case.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Case must be MODEL:DATASET, got {case!r}") from exc
    root = cache_root / model / dataset
    cache_partitions = load_cache_partitions(root)
    feature_names = cache_partitions.feature_names
    truth = np.asarray(load_partitioned_array(cache_partitions, "truth"))
    proposal_map = np.asarray(
        load_partitioned_array(cache_partitions, "proposal_map_estimates")
    )
    proposal_summary = np.asarray(
        load_partitioned_array(cache_partitions, "proposal_mean_estimates")
    )
    target_map = np.asarray(
        load_partitioned_array(cache_partitions, "tf_target_map_estimates")
    )
    target_summary = np.asarray(
        load_partitioned_array(cache_partitions, "tf_target_mean_estimates")
    )
    population_log_ratio = np.asarray(
        load_partitioned_array(cache_partitions, "population_tf_log_ratio"),
        dtype=np.float64,
    )
    rmag_true = np.asarray(
        load_partitioned_array(cache_partitions, "rmag_true"), dtype=float
    )
    if "central_halpha_snr" in cache_partitions.files:
        spectral_condition = np.asarray(
            load_partitioned_array(cache_partitions, "central_halpha_snr"),
            dtype=float,
        )
        spectral_condition_name = "central H-alpha S/N"
        spectral_condition_log_scale = False
    else:
        spectral_condition = np.asarray(
            load_partitioned_array(
                cache_partitions, "spectral_reference_quality"
            ),
            dtype=float,
        )
        spectral_condition_name = "spectral reference quality"
        spectral_condition_log_scale = True
    posterior_ess = np.asarray(
        load_partitioned_array(cache_partitions, "posterior_tf_ess"), dtype=float
    )
    posterior_ess_fraction = np.asarray(
        load_partitioned_array(cache_partitions, "posterior_tf_ess_fraction"),
        dtype=float,
    )
    posterior_max_weight = np.asarray(
        load_partitioned_array(cache_partitions, "posterior_tf_max_weight"),
        dtype=float,
    )

    n = len(truth)
    expected_summary = (n, 3, len(feature_names))
    for name, value, expected in (
        ("proposal_map", proposal_map, truth.shape),
        ("target_map", target_map, truth.shape),
        ("proposal_summary", proposal_summary, expected_summary),
        ("target_summary", target_summary, expected_summary),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} shape {value.shape}; expected {expected}")
    vectors = (
        population_log_ratio,
        rmag_true,
        spectral_condition,
        posterior_ess,
        posterior_ess_fraction,
        posterior_max_weight,
    )
    if any(value.shape != (n,) for value in vectors):
        raise ValueError(f"Scalar cache length mismatch for {case}")
    if max_galaxies is not None:
        if max_galaxies <= 0:
            raise ValueError("max_galaxies must be positive")
        take = min(max_galaxies, n)
        truth = truth[:take]
        proposal_map = proposal_map[:take]
        proposal_summary = proposal_summary[:take]
        target_map = target_map[:take]
        target_summary = target_summary[:take]
        population_log_ratio = population_log_ratio[:take]
        rmag_true = rmag_true[:take]
        spectral_condition = spectral_condition[:take]
        posterior_ess = posterior_ess[:take]
        posterior_ess_fraction = posterior_ess_fraction[:take]
        posterior_max_weight = posterior_max_weight[:take]
        n = take

    proposal_weight = np.full(n, 1.0 / n, dtype=np.float64)
    target_weight = normalize_population_log_weights(population_log_ratio)
    return {
        "case": case,
        "model": model,
        "dataset": dataset,
        "root": root,
        "cache_partitions": cache_partitions,
        "feature_names": feature_names,
        "truth": truth,
        "rmag_true": rmag_true,
        "spectral_condition": spectral_condition,
        "spectral_condition_name": spectral_condition_name,
        "spectral_condition_log_scale": spectral_condition_log_scale,
        "posterior_tf_ess": posterior_ess,
        "posterior_tf_ess_fraction": posterior_ess_fraction,
        "posterior_tf_max_weight": posterior_max_weight,
        "populations": {
            "Proposal population / base posterior": {
                "key": "proposal",
                "map": proposal_map,
                "summary": proposal_summary,
                "mean": proposal_summary[:, 1],
                "galaxy_weight": proposal_weight,
                "population_weight": proposal_weight.copy(),
            },
            "TF target population / TF posterior": {
                "key": "tf_target",
                "map": target_map,
                "summary": target_summary,
                "mean": target_summary[:, 1],
                "galaxy_weight": target_weight,
                "population_weight": target_weight.copy(),
            },
        },
    }


def normalize_subset_weights(
    weight: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(weight, dtype=np.float64)
    finite = np.isfinite(values) & (values >= 0.0)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    if not np.any(finite):
        return finite, np.array([], dtype=np.float64)
    selected = values[finite]
    total = np.sum(selected)
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros_like(finite), np.array([], dtype=np.float64)
    return finite, selected / total


def weighted_mean_and_se(
    values: np.ndarray, weight: np.ndarray
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mask, normalized = normalize_subset_weights(
        weight, np.isfinite(values)
    )
    if not len(normalized):
        return float("nan"), float("nan"), 0.0
    selected = values[mask]
    mean = float(np.sum(normalized * selected))
    ess = effective_sample_size(normalized)
    variance = float(np.sum(normalized * np.square(selected - mean)))
    se = math.sqrt(variance / max(ess - 1.0, 1.0))
    return mean, se, ess


def posterior_component_variance(
    values: np.ndarray, log_weight: np.ndarray | None = None
) -> float:
    """Marginal posterior variance with optional normalized log weights."""

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if log_weight is None:
        selected = values[finite]
        if not len(selected):
            return float("nan")
        weight = np.full(len(selected), 1.0 / len(selected))
    else:
        log_weight = np.asarray(log_weight, dtype=np.float64)
        if log_weight.shape != values.shape:
            raise ValueError("values and log_weight must have matching shapes")
        finite &= np.isfinite(log_weight)
        selected = values[finite]
        selected_log_weight = log_weight[finite]
        if not len(selected):
            return float("nan")
        maximum = float(np.max(selected_log_weight))
        scaled = np.exp(selected_log_weight - maximum)
        weight = scaled / np.sum(scaled)
    mean = float(np.sum(weight * selected))
    return float(np.sum(weight * np.square(selected - mean)))


def fit_design(
    y: np.ndarray, design: np.ndarray, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if design.ndim != 2 or y.shape != (design.shape[0],):
        raise ValueError("inconsistent regression shapes")
    finite = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    mask, normalized = normalize_subset_weights(weight, finite)
    x = design[mask]
    y = y[mask]
    if len(y) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        nan = np.full(x.shape[1], np.nan)
        return nan, nan.copy(), effective_sample_size(normalized) if len(normalized) else 0.0
    bread = np.linalg.pinv(x.T @ (normalized[:, None] * x))
    coefficient = bread @ (x.T @ (normalized * y))
    residual = y - x @ coefficient
    score = x * (normalized * residual)[:, None]
    covariance = bread @ (score.T @ score) @ bread
    ess = effective_sample_size(normalized)
    correction = ess / max(ess - x.shape[1], 1.0)
    uncertainty = np.sqrt(np.clip(np.diag(covariance) * correction, 0.0, None))
    return coefficient, uncertainty, ess


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap a directed angle to ``[-pi, pi)``."""

    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def weighted_quantile(
    values: np.ndarray, quantiles: np.ndarray, weight: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if values.ndim != 1 or weight.shape != values.shape:
        raise ValueError("values and weight must be matching vectors")
    finite = np.isfinite(values) & np.isfinite(weight) & (weight >= 0.0)
    values, weight = values[finite], weight[finite]
    if not len(values) or np.sum(weight) <= 0.0:
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values, kind="stable")
    values, weight = values[order], weight[order]
    weight = weight / np.sum(weight)
    coordinate = np.cumsum(weight) - 0.5 * weight
    return np.interp(
        quantiles, coordinate, values, left=values[0], right=values[-1]
    )


def population_weight(population: dict) -> np.ndarray:
    """Return the population-only weights, before optional precision weighting."""

    return np.asarray(
        population.get("population_weight", population["galaxy_weight"]),
        dtype=np.float64,
    )


def compose_precision_weights(
    base_weight: np.ndarray,
    g1_variance: np.ndarray,
    g2_variance: np.ndarray,
    cap_percentile: float | None,
) -> tuple[np.ndarray, dict]:
    """Compose population mass with common spin-symmetric shear precision.

    The precision is ``2 / (Var(g1) + Var(g2))``. A percentile cap is
    evaluated with respect to the valid base-population mass, rather than row
    count, which is important for the TF target population. Galaxies with
    non-finite or non-positive total posterior variance receive zero analysis
    weight and are reported explicitly in the returned diagnostics.
    """

    base_weight = np.asarray(base_weight, dtype=np.float64)
    g1_variance = np.asarray(g1_variance, dtype=np.float64)
    g2_variance = np.asarray(g2_variance, dtype=np.float64)
    if base_weight.ndim != 1 or not (
        g1_variance.shape == g2_variance.shape == base_weight.shape
    ):
        raise ValueError("population weights and shear variances must be vectors")
    if cap_percentile is not None and not 0.0 < cap_percentile < 100.0:
        raise ValueError("precision cap percentile must lie strictly between 0 and 100")

    valid_base = np.isfinite(base_weight) & (base_weight >= 0.0)
    variance_sum = g1_variance + g2_variance
    valid_variance = (
        np.isfinite(g1_variance)
        & np.isfinite(g2_variance)
        & (g1_variance >= 0.0)
        & (g2_variance >= 0.0)
        & np.isfinite(variance_sum)
        & (variance_sum > 0.0)
    )
    usable = valid_base & valid_variance
    if not np.any(usable) or np.sum(base_weight[usable]) <= 0.0:
        raise ValueError("no positive population mass has a valid shear variance")

    precision = np.full(base_weight.shape, np.nan, dtype=np.float64)
    precision[usable] = 2.0 / variance_sum[usable]
    if cap_percentile is None:
        threshold = float("inf")
        capped = np.zeros(base_weight.shape, dtype=bool)
        clipped_precision = precision.copy()
    else:
        threshold = float(
            weighted_quantile(
                precision[usable],
                np.asarray([cap_percentile / 100.0]),
                base_weight[usable],
            )[0]
        )
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("precision percentile cap is not finite and positive")
        capped = usable & (precision > threshold)
        clipped_precision = np.minimum(precision, threshold)

    combined = np.zeros(base_weight.shape, dtype=np.float64)
    combined[usable] = base_weight[usable] * clipped_precision[usable]
    total = float(np.sum(combined))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("precision-weighted population mass is not positive")
    combined /= total

    valid_base_mass = float(np.sum(base_weight[valid_base]))
    usable_base_mass = float(np.sum(base_weight[usable]))
    capped_mass = float(np.sum(base_weight[capped]))
    invalid_variance = ~valid_variance
    invalid_mass = float(np.sum(base_weight[valid_base & invalid_variance]))
    diagnostics = {
        "cap_percentile": cap_percentile,
        "precision_threshold": threshold,
        "shape_noise_floor": 0.0 if cap_percentile is None else 1.0 / math.sqrt(threshold),
        "capped_count": int(np.count_nonzero(capped)),
        "capped_fraction": float(np.count_nonzero(capped) / np.count_nonzero(usable)),
        "capped_population_mass": capped_mass / usable_base_mass,
        "invalid_variance_count": int(np.count_nonzero(invalid_variance)),
        "invalid_variance_fraction": float(np.mean(invalid_variance)),
        "invalid_variance_population_mass": (
            invalid_mass / valid_base_mass if valid_base_mass > 0.0 else float("nan")
        ),
        "usable_count": int(np.count_nonzero(usable)),
        "ess": effective_sample_size(combined),
    }
    return combined, diagnostics


def component_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    low_g: float,
    weight: np.ndarray,
) -> dict:
    """Weighted additive and multiplicative shear calibration metrics."""

    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if truth.shape != estimate.shape or weight.shape != truth.shape:
        raise ValueError("truth, estimate, and weight must be matching vectors")
    finite = np.isfinite(truth) & np.isfinite(estimate)
    residual = estimate - truth
    additive, additive_se, ess = weighted_mean_and_se(residual[finite], weight[finite])
    low = finite & (np.abs(truth) < low_g)
    linear, linear_se, low_ess = fit_design(
        residual[low],
        np.column_stack((np.ones(np.count_nonzero(low)), truth[low])),
        weight[low],
    )
    cubic, cubic_se, cubic_ess = fit_design(
        residual[finite],
        np.column_stack(
            (
                np.ones(np.count_nonzero(finite)),
                truth[finite],
                truth[finite] ** 3,
            )
        ),
        weight[finite],
    )
    return {
        "c": additive,
        "c_se": additive_se,
        "low_c": float(linear[0]),
        "low_c_se": float(linear_se[0]),
        "low_m": float(linear[1]),
        "low_m_se": float(linear_se[1]),
        "cubic_c": float(cubic[0]),
        "cubic_m": float(cubic[1]),
        "cubic_q": float(cubic[2]),
        "cubic_m_se": float(cubic_se[1]),
        "cubic_q_se": float(cubic_se[2]),
        "n": int(np.count_nonzero(finite)),
        "ess": ess,
        "n_low": int(np.count_nonzero(low)),
        "ess_low": low_ess,
        "ess_cubic": cubic_ess,
    }


def precision_cap_sweep(
    case: dict,
    posterior_diagnostics: dict[str, dict[str, np.ndarray]],
    low_g: float,
) -> list[dict]:
    """Evaluate Mean-estimator low-|g| calibration across precision caps."""

    rows = []
    truth = np.asarray(case["truth"], dtype=np.float64)
    for population_label, population in case["populations"].items():
        posterior = posterior_diagnostics[population["key"]]
        base_weight = population_weight(population)
        for percentile in PRECISION_CAP_SWEEP:
            weight, diagnostics = compose_precision_weights(
                base_weight,
                posterior["g1_variance"],
                posterior["g2_variance"],
                percentile,
            )
            component_rows = [
                component_metrics(
                    truth[:, index], population["mean"][:, index], low_g, weight
                )
                for index in range(2)
            ]
            rows.append(
                {
                    "population": population_label,
                    "population_key": population["key"],
                    **diagnostics,
                    "g1_m": component_rows[0]["low_m"],
                    "g1_m_se": component_rows[0]["low_m_se"],
                    "g1_ess": component_rows[0]["ess_low"],
                    "g1_n": component_rows[0]["n_low"],
                    "g2_m": component_rows[1]["low_m"],
                    "g2_m_se": component_rows[1]["low_m_se"],
                    "g2_ess": component_rows[1]["ess_low"],
                    "g2_n": component_rows[1]["n_low"],
                }
            )
    return rows


def apply_precision_weighting(
    case: dict,
    posterior_diagnostics: dict[str, dict[str, np.ndarray]],
    cap_percentile: float = DEFAULT_PRECISION_CAP_PERCENTILE,
) -> dict[str, dict]:
    """Install capped precision-composed weights for all downstream sections."""

    applied = {}
    for population_label, population in case["populations"].items():
        posterior = posterior_diagnostics[population["key"]]
        weight, diagnostics = compose_precision_weights(
            population_weight(population),
            posterior["g1_variance"],
            posterior["g2_variance"],
            cap_percentile,
        )
        population["galaxy_weight"] = weight
        population["precision_weighting"] = diagnostics
        applied[population_label] = diagnostics
    case["precision_weighted"] = True
    case["precision_cap_percentile"] = cap_percentile
    return applied


def galaxy_frame_components(
    truth: np.ndarray, estimate: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Use true theta to rotate truth and estimate into the galaxy frame."""

    theta = truth[:, 2]
    true_components = img_to_gal_axis(truth[:, 0], truth[:, 1], theta)
    estimated_components = img_to_gal_axis(
        estimate[:, 0], estimate[:, 1], theta
    )
    return true_components, estimated_components


def compute_metrics(case: dict, low_g: float) -> list[dict]:
    rows = []
    truth = case["truth"]
    for population_label, population in case["populations"].items():
        weight = population["galaxy_weight"]
        for estimator, estimate in (
            ("MAP", population["map"]),
            ("Mean", population["mean"]),
        ):
            frames = {
                "image": (
                    (truth[:, 0], truth[:, 1]),
                    (estimate[:, 0], estimate[:, 1]),
                    ("g1", "g2"),
                )
            }
            true_gal, estimate_gal = galaxy_frame_components(truth, estimate)
            frames["galaxy"] = (true_gal, estimate_gal, ("g+ (E)", "gx (B)"))
            for frame, (true_components, estimated_components, labels) in frames.items():
                for component_truth, component_estimate, component in zip(
                    true_components, estimated_components, labels
                ):
                    row = component_metrics(
                        component_truth, component_estimate, low_g, weight
                    )
                    row.update(
                        population=population_label,
                        population_key=population["key"],
                        estimator=estimator,
                        frame=frame,
                        component=component,
                    )
                    rows.append(row)
    return rows


def nuisance_bias_metrics(case: dict) -> list[dict]:
    """Population-weighted additive biases for every non-shear target."""

    rows = []
    truth = np.asarray(case["truth"], dtype=np.float64)
    for population_label, population in case["populations"].items():
        weight = np.asarray(population["galaxy_weight"], dtype=np.float64)
        for estimator, estimate in (
            ("Mean", population["mean"]),
            ("MAP", population["map"]),
        ):
            estimate = np.asarray(estimate, dtype=np.float64)
            for index, parameter in enumerate(case["feature_names"]):
                if parameter in {"g1", "g2"}:
                    continue
                residual = estimate[:, index] - truth[:, index]
                if parameter == "theta_int":
                    residual = wrap_angle(residual)
                finite = (
                    np.isfinite(truth[:, index])
                    & np.isfinite(estimate[:, index])
                    & np.isfinite(weight)
                    & (weight >= 0.0)
                )
                bias, bias_se, ess = weighted_mean_and_se(
                    residual[finite], weight[finite]
                )
                rows.append(
                    {
                        "population": population_label,
                        "estimator": estimator,
                        "parameter": parameter,
                        "bias": bias,
                        "bias_se": bias_se,
                        "n": int(np.count_nonzero(finite)),
                        "ess": ess,
                    }
                )
    return rows


def coverage_metrics(
    truth: np.ndarray,
    summary: np.ndarray,
    weight: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    population: str = "",
) -> list[dict]:
    """Weighted 16th--84th posterior coverage for every target."""

    truth = np.asarray(truth, dtype=np.float64)
    summary = np.asarray(summary, dtype=np.float64)
    feature_names = tuple(feature_names)
    if summary.shape != (len(truth), 3, len(feature_names)):
        raise ValueError("posterior summary has an unexpected shape")
    rows = []
    for index, parameter in enumerate(feature_names):
        lower, upper = summary[:, 0, index], summary[:, 2, index]
        target = truth[:, index]
        finite = np.isfinite(target) & np.isfinite(lower) & np.isfinite(upper)
        if parameter == "theta_int":
            target = wrap_angle(target)
            lower, upper = wrap_angle(lower), wrap_angle(upper)
            inside = np.where(
                lower <= upper,
                (target >= lower) & (target <= upper),
                (target >= lower) | (target <= upper),
            )
        else:
            inside = (target >= lower) & (target <= upper)
        mask, normalized = normalize_subset_weights(weight, finite)
        if len(normalized):
            coverage = float(np.sum(normalized * inside[mask]))
            ess = effective_sample_size(normalized)
            se = math.sqrt(coverage * (1.0 - coverage) / ess)
        else:
            coverage = se = float("nan")
            ess = 0.0
        rows.append(
            {
                "population": population,
                "parameter": parameter,
                "coverage": coverage,
                "coverage_se": se,
                "delta": coverage - 0.68,
                "n": int(np.count_nonzero(finite)),
                "ess": ess,
            }
        )
    return rows


def quantile_binned(
    x: np.ndarray, y: np.ndarray, bins: int, weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted-quantile bins with weighted centers, means, and errors."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight >= 0)
    x, y, weight = x[finite], y[finite], weight[finite]
    if not len(x) or np.sum(weight) <= 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty.astype(int), empty
    edges = weighted_quantile(x, np.linspace(0.0, 1.0, bins + 1), weight)
    edges[0], edges[-1] = -np.inf, np.inf
    assignment = np.searchsorted(edges[1:-1], x, side="right")
    center, mean, error, count, ess = [], [], [], [], []
    for bin_index in range(bins):
        selected = assignment == bin_index
        if not np.any(selected) or np.sum(weight[selected]) <= 0:
            continue
        selected_weight = weight[selected] / np.sum(weight[selected])
        center_value = float(np.sum(selected_weight * x[selected]))
        mean_value, se_value, ess_value = weighted_mean_and_se(
            y[selected], selected_weight
        )
        center.append(center_value)
        mean.append(mean_value)
        error.append(se_value)
        count.append(int(np.count_nonzero(selected)))
        ess.append(ess_value)
    return (
        np.asarray(center),
        np.asarray(mean),
        np.asarray(error),
        np.asarray(count, dtype=int),
        np.asarray(ess),
    )


def nuisance_bias_curves(case: dict, bins: int) -> dict[str, dict]:
    """Mean-estimator nuisance residuals in common proposal-truth bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    truth = np.asarray(case["truth"], dtype=np.float64)
    proposal = next(
        (
            population
            for population in case["populations"].values()
            if population["key"] == "proposal"
        ),
        None,
    )
    if proposal is None:
        raise ValueError("nuisance plots require a proposal population")
    proposal_weight = population_weight(proposal)
    curves = {}
    for index, parameter in enumerate(case["feature_names"]):
        if parameter in {"g1", "g2"}:
            continue
        true = truth[:, index]
        finite_truth = (
            np.isfinite(true)
            & np.isfinite(proposal_weight)
            & (proposal_weight >= 0.0)
        )
        if not np.any(finite_truth) or np.sum(proposal_weight[finite_truth]) <= 0.0:
            continue
        edges = weighted_quantile(
            true[finite_truth],
            np.linspace(0.0, 1.0, bins + 1),
            proposal_weight[finite_truth],
        )
        assignment_edges = edges.copy()
        assignment_edges[0], assignment_edges[-1] = -np.inf, np.inf
        assignment = np.full(len(true), -1, dtype=int)
        assignment[finite_truth] = np.searchsorted(
            assignment_edges[1:-1], true[finite_truth], side="right"
        )

        centers = np.full(bins, np.nan, dtype=np.float64)
        for bin_index in range(bins):
            selected = finite_truth & (assignment == bin_index)
            mask, normalized = normalize_subset_weights(
                proposal_weight, selected
            )
            if len(normalized):
                centers[bin_index] = float(np.sum(normalized * true[mask]))

        parameter_curves = {"edges": edges, "center": centers, "populations": {}}
        for population_label, population in case["populations"].items():
            estimate = np.asarray(population["mean"], dtype=np.float64)[:, index]
            weight = np.asarray(population["galaxy_weight"], dtype=np.float64)
            residual = estimate - true
            if parameter == "theta_int":
                residual = wrap_angle(residual)
            mean = np.full(bins, np.nan, dtype=np.float64)
            error = np.full(bins, np.nan, dtype=np.float64)
            count = np.zeros(bins, dtype=int)
            ess = np.zeros(bins, dtype=np.float64)
            for bin_index in range(bins):
                selected = assignment == bin_index
                value, se, bin_ess = weighted_mean_and_se(
                    residual[selected], weight[selected]
                )
                mean[bin_index] = value
                error[bin_index] = se
                count[bin_index] = np.count_nonzero(
                    selected & np.isfinite(residual) & np.isfinite(weight)
                )
                ess[bin_index] = bin_ess
            finite = np.isfinite(true) & np.isfinite(residual)
            regression_truth = true[finite]
            truth_scale = float(np.ptp(regression_truth))
            if np.isfinite(truth_scale) and truth_scale > 0.0:
                truth_center = 0.5 * float(
                    np.min(regression_truth) + np.max(regression_truth)
                )
                scaled_truth = (regression_truth - truth_center) / truth_scale
                coefficient, uncertainty, slope_ess = fit_design(
                    residual[finite],
                    np.column_stack(
                        (np.ones(np.count_nonzero(finite)), scaled_truth)
                    ),
                    weight[finite],
                )
                slope = float(coefficient[1] / truth_scale)
                slope_se = float(uncertainty[1] / truth_scale)
            else:
                slope = float("nan")
                slope_se = float("nan")
                slope_ess = 0.0
            parameter_curves["populations"][population_label] = {
                "population_key": population["key"],
                "mean": mean,
                "se": error,
                "n": count,
                "ess": ess,
                "m": slope,
                "m_se": slope_se,
                "slope_ess": slope_ess,
            }
        curves[parameter] = parameter_curves
    return curves


def conditional_shear_calibration(
    case: dict,
    population_label: str,
    estimator: str,
    bins: int,
    shape_noise: np.ndarray,
) -> dict:
    """Weighted full-range m, c, and posterior shape noise in condition bins."""

    population = case["populations"][population_label]
    estimator_key = estimator.lower()
    if estimator_key not in {"mean", "map"}:
        raise ValueError("estimator must be Mean or MAP")
    truth = np.asarray(case["truth"], dtype=np.float64)
    estimate = np.asarray(population[estimator_key], dtype=np.float64)
    weights = np.asarray(population["galaxy_weight"], dtype=np.float64)
    shape_noise = np.asarray(shape_noise, dtype=np.float64)
    if shape_noise.shape != (len(truth),):
        raise ValueError("shape_noise must contain one value per galaxy")
    names = case["feature_names"]
    spectral_condition_name = case.get(
        "spectral_condition_name", "spectral reference quality"
    )
    spectral_condition = case.get(
        "spectral_condition", case.get("spectral_reference_quality")
    )
    conditions = {
        "true magnitude": case["rmag_true"],
        spectral_condition_name: spectral_condition,
        "true hlr": truth[:, names.index("hlr")],
        "true sini": truth[:, names.index("sini")],
    }
    result = {}
    for condition, axis in conditions.items():
        axis = np.asarray(axis, dtype=np.float64)
        finite = np.isfinite(axis) & np.isfinite(weights) & (weights >= 0)
        edges = weighted_quantile(
            axis[finite], np.linspace(0.0, 1.0, bins + 1), weights[finite]
        )
        edges[0], edges[-1] = -np.inf, np.inf
        assignment = np.full(len(axis), -1, dtype=int)
        assignment[finite] = np.searchsorted(
            edges[1:-1], axis[finite], side="right"
        )
        result[condition] = {}
        for component, index in (("g1", 0), ("g2", 1)):
            curve = {key: [] for key in ("x", "m", "m_se", "c", "c_se", "n", "ess")}
            residual = estimate[:, index] - truth[:, index]
            for bin_index in range(bins):
                selected = assignment == bin_index
                design = np.column_stack(
                    (np.ones(np.count_nonzero(selected)), truth[selected, index])
                )
                coefficient, uncertainty, ess = fit_design(
                    residual[selected], design, weights[selected]
                )
                mask, normalized = normalize_subset_weights(weights, selected)
                center = (
                    float(np.sum(normalized * axis[mask]))
                    if len(normalized)
                    else float("nan")
                )
                for key, value in (
                    ("x", center),
                    ("c", float(coefficient[0])),
                    ("c_se", float(uncertainty[0])),
                    ("m", float(coefficient[1])),
                    ("m_se", float(uncertainty[1])),
                    ("n", int(np.count_nonzero(selected))),
                    ("ess", ess),
                ):
                    curve[key].append(value)
            result[condition][component] = {
                key: np.asarray(value) for key, value in curve.items()
            }
        shape_curve = {
            key: [] for key in ("x", "value", "se", "n", "ess")
        }
        for bin_index in range(bins):
            selected = assignment == bin_index
            mask, normalized = normalize_subset_weights(
                weights, selected & np.isfinite(shape_noise)
            )
            center = (
                float(np.sum(normalized * axis[mask]))
                if len(normalized)
                else float("nan")
            )
            value, se, ess = weighted_mean_and_se(
                shape_noise[mask], normalized
            )
            for key, item in (
                ("x", center),
                ("value", value),
                ("se", se),
                ("n", int(np.count_nonzero(mask))),
                ("ess", ess),
            ):
                shape_curve[key].append(item)
        result[condition]["shape_noise"] = {
            key: np.asarray(value) for key, value in shape_curve.items()
        }
    return result


def shear_pp_bin_indices(
    values: np.ndarray, edges: np.ndarray = SHEAR_PP_BIN_EDGES
) -> np.ndarray:
    """Assign shear to half-open bins, with a closed final interval."""

    raw = np.asarray(values)
    values = np.asarray(raw, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("shear P-P bin edges must be strictly increasing")
    tolerance = 4.0 * (
        np.finfo(raw.dtype).eps
        if np.issubdtype(raw.dtype, np.floating)
        else np.finfo(float).eps
    )
    snapped = values.copy()
    for edge in edges:
        snapped[np.abs(snapped - edge) <= tolerance] = edge
    assigned = np.searchsorted(edges, snapped, side="right") - 1
    assigned[snapped == edges[-1]] = len(edges) - 2
    valid = (
        np.isfinite(snapped)
        & (snapped >= edges[0])
        & (snapped <= edges[-1])
        & (assigned >= 0)
        & (assigned < len(edges) - 1)
    )
    result = np.full(values.shape, -1, dtype=np.int8)
    result[valid] = assigned[valid]
    return result


def shear_pp_bin_masks(
    values: np.ndarray, edges: np.ndarray = SHEAR_PP_BIN_EDGES
) -> list[np.ndarray]:
    indices = shear_pp_bin_indices(values, edges)
    return [indices == index for index in range(len(edges) - 1)]


def _empty_shear_posterior_diagnostic(size: int) -> dict[str, np.ndarray]:
    return {
        "pit": np.full((size, 2), np.nan, dtype=np.float64),
        "posterior_variance": np.full((size, 2), np.nan, dtype=np.float64),
        "retained_count": np.zeros((size, 2), dtype=np.int64),
        "retained_mass": np.zeros((size, 2), dtype=np.float64),
        "conditional_ess": np.zeros((size, 2), dtype=np.float64),
    }


def _conditional_candidate_mass(
    values: np.ndarray,
    keep: np.ndarray,
    log_weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the retained mask, normalized conditional weights, and mass."""

    if log_weight is None:
        count = int(np.count_nonzero(keep))
        conditional = np.full(count, 1.0 / count) if count else np.array([])
        return keep, conditional, count / len(values)
    keep = keep & np.isfinite(log_weight)
    selected_log_weight = log_weight[keep]
    if not len(selected_log_weight):
        return keep, np.array([]), 0.0
    maximum = float(np.max(selected_log_weight))
    scaled = np.exp(selected_log_weight - maximum)
    conditional = scaled / np.sum(scaled)
    log_mass = maximum + math.log(float(np.sum(scaled)))
    return keep, conditional, float(np.exp(log_mass))


def load_shear_posterior_diagnostics(
    case: dict, block_size: int = 64
) -> dict[str, dict[str, np.ndarray]]:
    """Stream proposal and TF shear ranks and variances in one candidate pass."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    truth = np.asarray(case["truth"])
    root = Path(case["root"])
    cache_partitions = case.get("cache_partitions")
    if cache_partitions is None:
        cache_partitions = load_cache_partitions(root)
    sample_files = cache_partitions.files["sample"]
    truth_files = cache_partitions.files["truth"]
    weight_files = cache_partitions.files["posterior_tf_log_weight"]
    result = {
        key: _empty_shear_posterior_diagnostic(len(truth))
        for key in ("proposal", "tf_target")
    }

    offset = 0
    for part_index, (sample_path, truth_path, weight_path) in enumerate(
        zip(sample_files, truth_files, weight_files)
    ):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        stored_truth = np.load(truth_path, mmap_mode="r")
        cached_log_weight = np.load(weight_path, mmap_mode="r")
        if samples.ndim != 3 or samples.shape[-1] != len(case["feature_names"]):
            raise ValueError(
                f"Expected flat candidate shape (galaxy, draw, feature), got "
                f"{samples.shape} in {sample_path}"
            )
        if stored_truth.shape != (samples.shape[0], samples.shape[-1]):
            raise ValueError(f"Truth/sample shape mismatch in {sample_path}")
        if cached_log_weight.shape != samples.shape[:2]:
            raise ValueError(f"Candidate-weight shape mismatch in {weight_path}")
        take = min(samples.shape[0], len(truth) - offset)
        expected_truth = truth[offset : offset + take]
        if not np.array_equal(
            np.asarray(stored_truth[:take]), expected_truth, equal_nan=True
        ):
            raise ValueError(
                f"Truth rows in {truth_path} do not align with the report prefix"
            )

        for local_start in range(0, take, block_size):
            local_end = min(take, local_start + block_size)
            draws = np.asarray(
                samples[local_start:local_end, :, :2], dtype=np.float64
            )
            tf_log_weight = np.asarray(
                cached_log_weight[local_start:local_end], dtype=np.float64
            )
            targets = expected_truth[local_start:local_end, :2]
            truth_bins = shear_pp_bin_indices(targets)
            draw_bins = shear_pp_bin_indices(draws)
            for row in range(len(targets)):
                global_row = offset + local_start + row
                candidate_logs = {
                    "proposal": None,
                    "tf_target": tf_log_weight[row],
                }
                for component in range(2):
                    finite_in_bin = (
                        np.isfinite(draws[row, :, component])
                        & (draw_bins[row, :, component] == truth_bins[row, component])
                    )
                    for population_key, log_weight in candidate_logs.items():
                        diagnostic = result[population_key]
                        diagnostic["posterior_variance"][global_row, component] = (
                            posterior_component_variance(
                                draws[row, :, component], log_weight
                            )
                        )
                        if truth_bins[row, component] < 0:
                            continue
                        keep, conditional, mass = _conditional_candidate_mass(
                            draws[row, :, component], finite_in_bin, log_weight
                        )
                        diagnostic["retained_count"][global_row, component] = (
                            np.count_nonzero(keep)
                        )
                        diagnostic["retained_mass"][global_row, component] = mass
                        if not len(conditional):
                            continue
                        values = draws[row, keep, component]
                        target = targets[row, component]
                        rank = (
                            np.sum(conditional[values < target])
                            + 0.5 * np.sum(conditional[values == target])
                        )
                        diagnostic["pit"][global_row, component] = np.clip(
                            rank, 0.0, 1.0
                        )
                        diagnostic["conditional_ess"][global_row, component] = (
                            effective_sample_size(conditional)
                        )
        offset += take
        del samples, stored_truth, cached_log_weight
    if offset != len(truth):
        raise ValueError(
            f"Sample cache has {offset} galaxies, report uses {len(truth)}"
        )

    flattened = {}
    for population_key, diagnostic in result.items():
        pit = diagnostic["pit"]
        variance = diagnostic["posterior_variance"]
        retained = diagnostic["retained_count"]
        retained_mass = diagnostic["retained_mass"]
        conditional_ess = diagnostic["conditional_ess"]
        flattened[population_key] = {
            "g1": pit[:, 0],
            "g2": pit[:, 1],
            "g1_retained": retained[:, 0],
            "g2_retained": retained[:, 1],
            "g1_retained_mass": retained_mass[:, 0],
            "g2_retained_mass": retained_mass[:, 1],
            "g1_conditional_ess": conditional_ess[:, 0],
            "g2_conditional_ess": conditional_ess[:, 1],
            "g1_variance": variance[:, 0],
            "g2_variance": variance[:, 1],
            "shape_noise": np.sqrt(np.mean(variance, axis=1)),
        }
    return flattened


def load_shear_pit_values(
    case: dict, population_key: str, block_size: int = 64
) -> dict[str, np.ndarray]:
    """Compatibility wrapper returning one population's streamed diagnostics."""

    if population_key not in {"proposal", "tf_target"}:
        raise ValueError("population_key must be proposal or tf_target")
    return load_shear_posterior_diagnostics(case, block_size)[population_key]


def posterior_pp_curve(
    values: np.ndarray, galaxy_weight: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Weighted PIT ECDF, two-sided KS distance, and galaxy ESS."""

    values = np.asarray(values, dtype=np.float64)
    galaxy_weight = np.asarray(galaxy_weight, dtype=np.float64)
    finite = (
        np.isfinite(values)
        & np.isfinite(galaxy_weight)
        & (galaxy_weight >= 0.0)
    )
    values, galaxy_weight = values[finite], galaxy_weight[finite]
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("posterior PIT values must lie in [0, 1]")
    if not len(values) or np.sum(galaxy_weight) <= 0.0:
        return values, np.array([]), float("nan"), 0.0
    order = np.argsort(values, kind="stable")
    values, galaxy_weight = values[order], galaxy_weight[order]
    galaxy_weight = galaxy_weight / np.sum(galaxy_weight)
    empirical = np.cumsum(galaxy_weight)
    before = empirical - galaxy_weight
    distance = float(
        max(np.max(empirical - values), np.max(values - before))
    )
    return values, empirical, distance, effective_sample_size(galaxy_weight)


def fig_data_uri(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def bias_figure(case: dict, bins: int) -> str:
    """Weighted residual trends for proposal and TF-target populations."""

    populations = list(case["populations"].items())
    fig, axes = plt.subplots(
        len(populations), 2, figsize=(11, 4.2 * len(populations)), squeeze=False
    )
    for row, (label, population) in enumerate(populations):
        for component, axis in enumerate(axes[row]):
            true = case["truth"][:, component]
            for estimator, estimate, color in (
                ("MAP", population["map"], "tab:blue"),
                ("Mean", population["mean"], "tab:orange"),
            ):
                center, residual, error, _, _ = quantile_binned(
                    true,
                    estimate[:, component] - true,
                    bins,
                    population["galaxy_weight"],
                )
                axis.errorbar(
                    center, residual, yerr=error, marker="o", ms=3,
                    lw=1, color=color, label=estimator,
                )
            axis.axhline(0.0, color="black", ls="--", lw=0.8)
            axis.set_xlabel(f"true g{component + 1}")
            axis.set_ylabel("estimate - truth")
            axis.set_title(f"{label} — g{component + 1}")
            axis.legend()
    fig.suptitle(f"{case['case']} — weighted shear residuals")
    fig.tight_layout()
    return fig_data_uri(fig)


def nuisance_bias_figure(case: dict, bins: int) -> str:
    """Overlay proposal and TF posterior-mean nuisance residual trends."""

    curves = nuisance_bias_curves(case, bins)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), squeeze=False)
    colors = {"proposal": "tab:blue", "tf_target": "tab:orange"}
    short_labels = {"proposal": "Original", "tf_target": "TF"}
    for axis, (parameter, parameter_curves) in zip(axes.flat, curves.items()):
        scale, _ = NUISANCE_DISPLAY[parameter]
        unit = NUISANCE_PLOT_UNIT[parameter]
        for population_label, curve in parameter_curves["populations"].items():
            population_key = curve["population_key"]
            label = short_labels.get(population_key, population_label)
            label += (
                f": m={curve['m']:.3g}±{curve['m_se']:.2g}"
                if np.isfinite(curve["m"])
                else ": m=n/a"
            )
            axis.errorbar(
                scale * parameter_curves["center"],
                scale * curve["mean"],
                yerr=scale * curve["se"],
                marker="o",
                ms=3,
                lw=1,
                color=colors.get(population_key),
                label=label,
            )
        axis.axhline(0.0, color="black", ls="--", lw=0.8)
        suffix = f" [{unit}]" if unit else ""
        axis.set_xlabel(f"true {parameter}{suffix}")
        residual_label = (
            "wrapped posterior mean - truth"
            if parameter == "theta_int"
            else "posterior mean - truth"
        )
        axis.set_ylabel(f"{residual_label}{suffix}")
        axis.set_title(parameter)
        axis.legend(fontsize="x-small")
    for axis in axes.flat[len(curves):]:
        axis.set_visible(False)
    weighting = (
        f"precision weighted (p{case['precision_cap_percentile']:g} cap)"
        if case.get("precision_weighted")
        else "population weighted"
    )
    fig.suptitle(
        f"{case['case']} — nuisance bias vs truth, Mean estimator — {weighting}"
    )
    fig.tight_layout()
    return fig_data_uri(fig)


def conditional_shear_calibration_figure(
    case: dict, population_label: str, estimator: str, curves: dict
) -> str:
    spectral_condition_name = case.get(
        "spectral_condition_name", "spectral reference quality"
    )
    spectral_condition = np.asarray(
        case.get("spectral_condition", case.get("spectral_reference_quality"))
    )
    spectral_condition_log_scale = case.get(
        "spectral_condition_log_scale", True
    )
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), squeeze=False)
    colors = {"g1": "tab:blue", "g2": "tab:orange"}
    for column, (condition, components) in enumerate(curves.items()):
        for row, (metric, scale, ylabel) in enumerate(
            (("m", 100.0, r"$10^2 m$"), ("c", 1.0e4, r"$10^4 c$"))
        ):
            axis = axes[row, column]
            for component in ("g1", "g2"):
                curve = components[component]
                axis.errorbar(
                    curve["x"],
                    scale * curve[metric],
                    yerr=scale * curve[f"{metric}_se"],
                    marker="o", ms=3, lw=1,
                    color=colors[component], label=component,
                )
            axis.axhline(0.0, color="black", ls="--", lw=0.8)
            axis.set_xlabel(condition)
            axis.set_ylabel(ylabel)
            axis.set_title(f"{metric} vs {condition}")
            if (
                condition == spectral_condition_name
                and spectral_condition_log_scale
                and np.all(spectral_condition > 0)
            ):
                axis.set_xscale("log")
            axis.legend()
        axis = axes[2, column]
        curve = components["shape_noise"]
        axis.errorbar(
            curve["x"],
            curve["value"],
            yerr=curve["se"],
            marker="o", ms=3, lw=1,
            color="tab:green",
        )
        axis.set_xlabel(condition)
        axis.set_ylabel(r"$\sigma_\epsilon$")
        axis.set_title(f"shape noise vs {condition}")
        if (
            condition == spectral_condition_name
            and spectral_condition_log_scale
            and np.all(spectral_condition > 0)
        ):
            axis.set_xscale("log")
    fig.suptitle(f"{case['case']} — {population_label} — {estimator} estimator")
    fig.tight_layout()
    return fig_data_uri(fig)


def theta_bias_figure(case: dict, bins: int) -> str:
    theta_index = case["feature_names"].index("theta_int")
    theta_true = case["truth"][:, theta_index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for axis, (label, population) in zip(axes, case["populations"].items()):
        for estimator, estimate, color in (
            ("MAP", population["map"], "tab:blue"),
            ("Mean", population["mean"], "tab:orange"),
        ):
            residual = wrap_angle(estimate[:, theta_index] - theta_true)
            center, mean, error, _, _ = quantile_binned(
                theta_true, residual, bins, population["galaxy_weight"]
            )
            axis.errorbar(
                center, mean, yerr=error, marker="o", ms=3, lw=1,
                color=color, label=estimator,
            )
        axis.axhline(0.0, color="black", ls="--", lw=0.8)
        axis.set_title(label)
        axis.set_xlabel("true theta_int [rad]")
        axis.legend()
    axes[0].set_ylabel("wrapped estimate - truth [rad]")
    fig.suptitle(f"{case['case']} — directed theta_int bias")
    fig.tight_layout()
    return fig_data_uri(fig)


def _smooth_circular_histogram(histogram: np.ndarray) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.float64)
    if histogram.ndim != 1 or len(histogram) < 9:
        raise ValueError("circular histogram requires at least nine bins")
    smoothed = histogram.copy()
    for _ in range(2):
        smoothed = (
            np.roll(smoothed, 2)
            + 4.0 * np.roll(smoothed, 1)
            + 6.0 * smoothed
            + 4.0 * np.roll(smoothed, -1)
            + np.roll(smoothed, -2)
        ) / 16.0
    return smoothed


def significant_circular_modes(histogram: np.ndarray) -> np.ndarray:
    """Descriptive peaks in a weighted, smoothed circular histogram."""

    smoothed = _smooth_circular_histogram(histogram)
    maximum = float(np.max(smoothed))
    if not np.isfinite(maximum) or maximum <= 0.0:
        return np.array([], dtype=int)
    candidates = np.flatnonzero(
        (smoothed > np.roll(smoothed, 1))
        & (smoothed >= np.roll(smoothed, -1))
        & (smoothed >= 0.10 * maximum)
    )
    selected = []
    for peak in candidates:
        left = min(smoothed[(peak - step) % len(smoothed)] for step in range(1, 5))
        right = min(smoothed[(peak + step) % len(smoothed)] for step in range(1, 5))
        if smoothed[peak] - max(left, right) >= 0.05 * maximum:
            selected.append(int(peak))
    return np.asarray(selected, dtype=int)


def load_theta_posterior_diagnostics(
    case: dict,
    population_key: str,
    *,
    histogram_bins: int = 72,
    block_size: int = 64,
) -> dict:
    """Stream directed-theta branch mass and weighted posterior mode counts."""

    if population_key not in {"proposal", "tf_target"}:
        raise ValueError("population_key must be proposal or tf_target")
    if histogram_bins < 9 or block_size <= 0:
        raise ValueError("invalid theta histogram or block size")
    theta_index = case["feature_names"].index("theta_int")
    truth = np.asarray(case["truth"])
    theta_truth = truth[:, theta_index]
    root = Path(case["root"])
    cache_partitions = case.get("cache_partitions")
    if cache_partitions is None:
        cache_partitions = load_cache_partitions(root)
    sample_files = cache_partitions.files["sample"]
    truth_files = cache_partitions.files["truth"]
    weight_files = (
        cache_partitions.files["posterior_tf_log_weight"]
        if population_key == "tf_target"
        else None
    )
    result = {
        "directional_mean": np.full(len(truth), np.nan),
        "directional_resultant": np.full(len(truth), np.nan),
        "true_branch_mass": np.full(len(truth), np.nan),
        "opposite_branch_mass": np.full(len(truth), np.nan),
        "middle_mass": np.full(len(truth), np.nan),
        "mode_count": np.zeros(len(truth), dtype=np.int16),
    }
    edges = np.linspace(-np.pi, np.pi, histogram_bins + 1)
    galaxy_histograms = np.zeros((len(truth), histogram_bins), dtype=np.float64)
    offset = 0
    for part_index, sample_path in enumerate(sample_files):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        stored_truth = np.load(truth_files[part_index], mmap_mode="r")
        if samples.ndim != 3 or samples.shape[-1] != len(case["feature_names"]):
            raise ValueError(f"Invalid flat candidate bank {samples.shape} in {sample_path}")
        take = min(samples.shape[0], len(truth) - offset)
        if not np.array_equal(
            np.asarray(stored_truth[:take]), truth[offset : offset + take], equal_nan=True
        ):
            raise ValueError(f"Truth rows do not align in {sample_path}")
        cached_weight = (
            np.load(weight_files[part_index], mmap_mode="r")
            if weight_files is not None
            else None
        )
        for local_start in range(0, take, block_size):
            local_end = min(take, local_start + block_size)
            theta = np.asarray(
                samples[local_start:local_end, :, theta_index], dtype=np.float64
            )
            if cached_weight is None:
                weight = np.full(theta.shape, 1.0 / theta.shape[1])
            else:
                log_weight = np.asarray(
                    cached_weight[local_start:local_end], dtype=np.float64
                )
                row_maximum = np.max(log_weight, axis=1, keepdims=True)
                weight = np.exp(log_weight - row_maximum)
                weight = weight / np.sum(weight, axis=1, keepdims=True)
            for row in range(len(theta)):
                global_row = offset + local_start + row
                finite = np.isfinite(theta[row]) & np.isfinite(weight[row]) & (weight[row] >= 0)
                if not np.any(finite) or np.sum(weight[row, finite]) <= 0.0:
                    continue
                angle = theta[row, finite]
                posterior_weight = weight[row, finite]
                posterior_weight /= np.sum(posterior_weight)
                sine = np.sum(posterior_weight * np.sin(angle))
                cosine = np.sum(posterior_weight * np.cos(angle))
                result["directional_mean"][global_row] = math.atan2(sine, cosine)
                result["directional_resultant"][global_row] = math.hypot(sine, cosine)
                residual = wrap_angle(angle - theta_truth[global_row])
                primary = np.abs(residual) < np.pi / 4.0
                opposite = np.abs(residual) > 3.0 * np.pi / 4.0
                result["true_branch_mass"][global_row] = np.sum(posterior_weight[primary])
                result["opposite_branch_mass"][global_row] = np.sum(posterior_weight[opposite])
                result["middle_mass"][global_row] = np.sum(
                    posterior_weight[~(primary | opposite)]
                )
                histogram = np.histogram(
                    residual, bins=edges, weights=posterior_weight
                )[0]
                galaxy_histograms[global_row] = histogram
                result["mode_count"][global_row] = len(
                    significant_circular_modes(histogram)
                )
        offset += take
        del samples, stored_truth, cached_weight
    if offset != len(truth):
        raise ValueError("theta sample cache is shorter than the report case")
    population = next(
        value for value in case["populations"].values()
        if value["key"] == population_key
    )
    galaxy_weight = population["galaxy_weight"]
    result["aggregate_residual_histogram"] = np.sum(
        galaxy_weight[:, None] * galaxy_histograms, axis=0
    )
    result["angle_edges"] = edges
    return result


def theta_modality_table(case: dict, diagnostics: dict[str, dict]) -> str:
    body = []
    for label, population in case["populations"].items():
        diagnostic = diagnostics[population["key"]]
        weight = population["galaxy_weight"]
        mode_count = diagnostic["mode_count"]
        antipodal = (
            (diagnostic["true_branch_mass"] >= 0.15)
            & (diagnostic["opposite_branch_mass"] >= 0.15)
        )
        cells = []
        for selected in (
            mode_count == 1, mode_count == 2, mode_count >= 3, antipodal
        ):
            cells.append(100.0 * float(np.sum(weight[selected])))
        true_mass, _, _ = weighted_mean_and_se(
            diagnostic["true_branch_mass"], weight
        )
        opposite_mass, _, _ = weighted_mean_and_se(
            diagnostic["opposite_branch_mass"], weight
        )
        body.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{cells[0]:.2f}%</td><td>{cells[1]:.2f}%</td>"
            f"<td>{cells[2]:.2f}%</td><td>{cells[3]:.2f}%</td>"
            f"<td>{true_mass:.3f}</td><td>{opposite_mass:.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>1 mode</th>"
        "<th>2 modes</th><th>3+ modes</th><th>antipodal bimodality</th>"
        "<th>mean true-branch mass</th><th>mean opposite-branch mass</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def theta_modality_figure(case: dict, diagnostics: dict[str, dict]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = ("tab:blue", "tab:orange")
    for color, (label, population) in zip(colors, case["populations"].items()):
        diagnostic = diagnostics[population["key"]]
        edges = diagnostic["angle_edges"]
        center = 0.5 * (edges[:-1] + edges[1:])
        axes[0].plot(
            center, diagnostic["aggregate_residual_histogram"],
            color=color, lw=1.4, label=label,
        )
        mode_count = diagnostic["mode_count"]
        weights = population["galaxy_weight"]
        fractions = [
            np.sum(weights[mode_count == 1]),
            np.sum(weights[mode_count == 2]),
            np.sum(weights[mode_count >= 3]),
        ]
        offset = -0.18 if population["key"] == "proposal" else 0.18
        axes[1].bar(
            np.arange(3) + offset, fractions, width=0.34,
            color=color, alpha=0.8, label=label,
        )
    axes[0].axvline(0.0, color="black", ls="--", lw=0.8)
    axes[0].set_xlabel("wrapped theta sample - truth [rad]")
    axes[0].set_ylabel("population-weighted posterior mass / bin")
    axes[0].legend(fontsize="small")
    axes[1].set_xticks(np.arange(3), ("1", "2", "3+"))
    axes[1].set_xlabel("significant posterior mode count")
    axes[1].set_ylabel("population-weighted galaxy fraction")
    axes[1].legend(fontsize="small")
    fig.suptitle(f"{case['case']} — directed theta posterior modality")
    fig.tight_layout()
    return fig_data_uri(fig)


def posterior_pp_figure(case: dict, pits_by_population: dict[str, dict]) -> str:
    populations = list(case["populations"].items())
    fig, axes = plt.subplots(
        len(populations), 2, figsize=(12, 5 * len(populations)),
        sharex=True, sharey=True, squeeze=False,
    )
    grid = np.linspace(0.0, 1.0, 501)
    colors = ("tab:blue", "tab:green", "tab:red")
    for row, (population_label, population) in enumerate(populations):
        pits = pits_by_population[population["key"]]
        galaxy_weight = population["galaxy_weight"]
        for component_index, component in enumerate(("g1", "g2")):
            axis = axes[row, component_index]
            truth_component = case["truth"][:, component_index]
            for bin_index, mask in enumerate(shear_pp_bin_masks(truth_component)):
                selected = mask & np.isfinite(pits[component])
                x, y, distance, ess = posterior_pp_curve(
                    pits[component][selected], galaxy_weight[selected]
                )
                lower, upper = (
                    SHEAR_PP_BIN_EDGES[bin_index],
                    SHEAR_PP_BIN_EDGES[bin_index + 1],
                )
                closing = "]" if bin_index == 2 else ")"
                label = (
                    f"[{lower:+.4f}, {upper:+.4f}{closing}; "
                    f"N={len(x):,}, ESS={ess:.1f}, KS={distance:.3f}"
                )
                if len(x):
                    selected_weight = galaxy_weight[selected]
                    candidate_ess = weighted_quantile(
                        pits[f"{component}_conditional_ess"][selected],
                        np.asarray([0.5]),
                        selected_weight,
                    )[0]
                    retained_mass = weighted_quantile(
                        pits[f"{component}_retained_mass"][selected],
                        np.asarray([0.5]),
                        selected_weight,
                    )[0]
                    label += (
                        f", median draw ESS={candidate_ess:.0f}, "
                        f"mass={retained_mass:.2f}"
                    )
                    epsilon = math.sqrt(math.log(40.0) / (2.0 * ess))
                    axis.fill_between(
                        grid,
                        np.maximum(0.0, grid - epsilon),
                        np.minimum(1.0, grid + epsilon),
                        color=colors[bin_index], alpha=0.07,
                    )
                axis.step(
                    x, y, where="post", color=colors[bin_index], lw=1.3,
                    label=label,
                )
            axis.plot(grid, grid, color="black", ls="--", lw=0.9, label="Uniform")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("conditional posterior quantile")
            axis.set_title(f"{population_label} — {component}")
            axis.legend(fontsize="xx-small", loc="upper left")
        axes[row, 0].set_ylabel("weighted fraction of truths")
    fig.suptitle(
        f"{case['case']} — prior-matched conditional P-P (three true-shear bins)"
    )
    fig.tight_layout()
    return fig_data_uri(fig)


def _format_scaled(value: float, scale: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{scale * value:.3f}"


def metrics_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{row['estimator']}</td><td>{row['frame']}</td>"
            f"<td>{row['component']}</td>"
            f"<td>{_format_scaled(row['c'], 1e4)} ± "
            f"{_format_scaled(row['c_se'], 1e4)}</td>"
            f"<td>{_format_scaled(row['low_m'], 1e2)} ± "
            f"{_format_scaled(row['low_m_se'], 1e2)}</td>"
            f"<td>{_format_scaled(row['cubic_m'], 1e2)} ± "
            f"{_format_scaled(row['cubic_m_se'], 1e2)}</td>"
            f"<td>{_format_scaled(row['cubic_q'], 1.0)}</td>"
            f"<td>{row['n']:,} / {row['ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Estimator</th>"
        "<th>Frame</th><th>Component</th><th>10<sup>4</sup> c</th>"
        "<th>10<sup>2</sup> low-|g| m</th><th>10<sup>2</sup> cubic m</th>"
        "<th>cubic q</th><th>N / galaxy ESS</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>"
    )


def precision_cap_sweep_table(rows: list[dict]) -> str:
    """Render precision caps, invalid-variance diagnostics, ESS, and Mean m."""

    body = []
    for row in rows:
        percentile = row["cap_percentile"]
        cap_label = "uncapped / all" if percentile is None else f"{percentile:g}th"
        threshold = (
            "∞"
            if percentile is None
            else f"{row['precision_threshold']:.5g}"
        )
        floor = (
            "0"
            if percentile is None
            else f"{row['shape_noise_floor']:.5g}"
        )
        body.append(
            "<tr>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{cap_label}</td><td>{threshold}</td><td>{floor}</td>"
            f"<td>{row['capped_count']:,} / {100 * row['capped_fraction']:.2f}% "
            f"rows / {100 * row['capped_population_mass']:.2f}% mass</td>"
            f"<td>{row['invalid_variance_count']:,} / "
            f"{100 * row['invalid_variance_fraction']:.3f}% rows / "
            f"{100 * row['invalid_variance_population_mass']:.3f}% mass</td>"
            f"<td>{row['ess']:.1f}</td>"
            f"<td>{_format_scaled(row['g1_m'], 1e2)} ± "
            f"{_format_scaled(row['g1_m_se'], 1e2)}</td>"
            f"<td>{row['g1_n']:,} / {row['g1_ess']:.1f}</td>"
            f"<td>{_format_scaled(row['g2_m'], 1e2)} ± "
            f"{_format_scaled(row['g2_m_se'], 1e2)}</td>"
            f"<td>{row['g2_n']:,} / {row['g2_ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Precision cap</th>"
        "<th>max p</th><th>equivalent σ<sub>ε</sub> floor</th>"
        "<th>Capped</th><th>Invalid variance</th><th>Overall ESS</th>"
        "<th>10<sup>2</sup> g1 m</th><th>g1 N / fit ESS</th>"
        "<th>10<sup>2</sup> g2 m</th><th>g2 N / fit ESS</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def nuisance_bias_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        scale, unit = NUISANCE_DISPLAY[row["parameter"]]
        unit_display = unit or "dimensionless"
        body.append(
            "<tr>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{row['estimator']}</td>"
            f"<td><code>{row['parameter']}</code></td>"
            f"<td>{_format_scaled(row['bias'], scale)} ± "
            f"{_format_scaled(row['bias_se'], scale)}</td>"
            f"<td>{unit_display}</td>"
            f"<td>{row['n']:,} / {row['ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Estimator</th>"
        "<th>Parameter</th><th>weighted additive bias ± SE</th><th>Units</th>"
        "<th>N / galaxy ESS</th></tr></thead><tbody>" + "".join(body)
        + "</tbody></table>"
    )


def coverage_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{row['parameter']}</td>"
            f"<td>{100 * row['coverage']:.2f}% ± "
            f"{100 * row['coverage_se']:.2f}%</td>"
            f"<td>{100 * row['delta']:+.2f} pp</td>"
            f"<td>{row['n']:,} / {row['ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Parameter</th>"
        "<th>16th–84th coverage</th><th>Difference from 68%</th>"
        "<th>N / galaxy ESS</th></tr></thead><tbody>" + "".join(body)
        + "</tbody></table>"
    )


def importance_table(case: dict) -> str:
    target_label = "TF target population / TF posterior"
    target_weight = population_weight(case["populations"][target_label])
    population_ess = effective_sample_size(target_weight)
    rows = []
    for label, values in (
        ("posterior candidate ESS", case["posterior_tf_ess"]),
        ("posterior candidate ESS fraction", case["posterior_tf_ess_fraction"]),
        ("largest posterior candidate weight", case["posterior_tf_max_weight"]),
    ):
        quantiles = weighted_quantile(
            values, np.asarray((0.16, 0.5, 0.84)), target_weight
        )
        mean, _, _ = weighted_mean_and_se(values, target_weight)
        rows.append(
            f"<tr><td>{label}</td><td>{mean:.4g}</td>"
            f"<td>{quantiles[0]:.4g}</td><td>{quantiles[1]:.4g}</td>"
            f"<td>{quantiles[2]:.4g}</td></tr>"
        )
    return (
        f"<p>Globally normalized TF population weights have effective sample "
        f"size <b>{population_ess:.1f}</b> of {len(target_weight):,} galaxies.</p>"
        "<table><thead><tr><th>Diagnostic</th><th>weighted mean</th>"
        "<th>16th</th><th>median</th><th>84th</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.bins < 2:
        raise ValueError("bins must be at least 2")
    if args.low_g <= 0.0:
        raise ValueError("low-g must be positive")
    cases = [
        load_case(args.cache_root, value, args.max_galaxies)
        for value in args.case
    ]
    sections = []
    for case in cases:
        case_name = case["case"]
        LOGGER.info("%s: starting report", case_name)
        importance_html = importance_table(case)
        LOGGER.info("%s: finished section: importance-sampling health", case_name)

        LOGGER.info("%s: streaming posterior candidate diagnostics", case_name)
        pits = load_shear_posterior_diagnostics(case)
        LOGGER.info("%s: finished posterior candidate diagnostics", case_name)
        cap_sweep_rows = precision_cap_sweep(case, pits, args.low_g)
        cap_sweep_html = precision_cap_sweep_table(cap_sweep_rows)
        if args.weighted:
            applied = apply_precision_weighting(
                case, pits, DEFAULT_PRECISION_CAP_PERCENTILE
            )
            for population_label, diagnostic in applied.items():
                LOGGER.info(
                    "%s: %s p95 precision weighting: ESS %.1f, capped %d, "
                    "invalid variance %d",
                    case_name,
                    population_label,
                    diagnostic["ess"],
                    diagnostic["capped_count"],
                    diagnostic["invalid_variance_count"],
                )

        metric_rows = compute_metrics(case, args.low_g)
        shear_html = (
            metrics_table(metric_rows)
            + f'<img src="{bias_figure(case, args.bins)}" '
            'alt="weighted residual trends">'
        )
        LOGGER.info("%s: finished section: shear calibration", case_name)
        nuisance_rows = nuisance_bias_metrics(case)
        nuisance_html = (
            nuisance_bias_table(nuisance_rows)
            + f'<img src="{nuisance_bias_figure(case, args.bins)}" '
            'alt="nuisance posterior-mean bias versus truth">'
        )
        LOGGER.info("%s: finished section: nuisance calibration", case_name)

        coverage_rows = []
        for population_label, population in case["populations"].items():
            coverage_rows.extend(
                coverage_metrics(
                    case["truth"], population["summary"],
                    population["galaxy_weight"], case["feature_names"],
                    population_label,
                )
            )
        coverage_html = coverage_table(coverage_rows)
        LOGGER.info("%s: finished section: posterior coverage", case_name)

        conditional_html = {}
        for estimator in ("Mean", "MAP"):
            panels = []
            for population_label, population in case["populations"].items():
                curves = conditional_shear_calibration(
                    case,
                    population_label,
                    estimator,
                    args.bins,
                    pits[population["key"]]["shape_noise"],
                )
                figure = conditional_shear_calibration_figure(
                    case, population_label, estimator, curves
                )
                panels.append(
                    f"<h4>{html.escape(population_label)}</h4>"
                    f"<img src=\"{figure}\" "
                    "alt=\"conditional shear and shape-noise calibration\">"
                )
            conditional_html[estimator] = "".join(panels)
            LOGGER.info(
                "%s: finished section: conditional %s calibration",
                case_name,
                estimator,
            )
        pp_html = (
            f"<img src=\"{posterior_pp_figure(case, pits)}\" "
            "alt=\"weighted posterior P-P plots\">"
        )
        LOGGER.info("%s: finished section: conditional P-P diagnostic", case_name)
        sections.append(
            f"<section><h2>{html.escape(case['case'])}</h2>"
            "<h3>Importance-sampling health</h3>"
            + importance_html
            + "<h3>Posterior-precision cap sweep</h3>"
            "<p>For each galaxy, p = 2 / [Var(g1) + Var(g2)]. Population "
            "weights are multiplied by p after clipping p at the named "
            "population-mass percentile; <i>uncapped / all</i> retains the "
            "full finite precision. These are caps, not sample cuts. The "
            "equivalent σ<sub>ε</sub> floor is 1/√p<sub>max</sub>. Invalid "
            "variances receive zero precision-analysis weight. The m fits use "
            "the posterior Mean in the image frame and the same low-|g| range "
            "as the main table.</p>"
            + cap_sweep_html
            + "<h3>Shear calibration</h3>"
            "<p>Every number is an ensemble statistic with the galaxy weights "
            "appropriate to the named population. Low-|g| fits use "
            f"|g| &lt; {args.low_g:g}; cubic fits use the full range. "
            + (
                "This report applies posterior precision with a "
                "population-weighted 95th-percentile cap."
                if args.weighted
                else "This report does not apply posterior-precision weights."
            )
            + "</p>"
            + shear_html
            + "<h3>Nuisance-parameter calibration</h3>"
            "<p>Entries are population-weighted means of estimate minus truth. "
            "The combined 2×4 figure uses the posterior Mean only and overlays "
            "the original/base and TF/TF populations in common truth bins "
            "defined from the original proposal. Error bars are weighted "
            "standard errors; legend values are full-range residual slopes. "
            "The <code>theta_int</code> residual is wrapped to [-π, π).</p>"
            + nuisance_html
            + "<h3>Conditional Mean calibration</h3>"
            "<p>Bin edges, fit coefficients, and errors are population-weighted. "
            "Spectral reference quality is the independently drawn noise-level "
            "control (an SNR-like reference), not a claim that every galaxy has "
            "that achieved emission-line SNR. "
            "The TF-target panel therefore describes the target population, not "
            "the uniform simulator proposal. Shape noise follows KL-I equation 20: "
            "σ<sub>ε,j</sub> = √[(σ²<sub>g1,j</sub> + σ²<sub>g2,j</sub>)/2]. "
            "Per-galaxy values are population-weighted inside each bin.</p>"
            + conditional_html["Mean"]
            + "<h3>Conditional MAP calibration</h3>"
            "<p>The m and c panels use MAP point estimates. Shape noise is a property "
            "of the corresponding posterior, so its panels match the Mean section.</p>"
            + conditional_html["MAP"]
            + "<h3>Posterior coverage</h3>"
            "<p>The base posterior uses equal candidate mass and equal galaxy "
            "mass. The TF posterior uses within-galaxy TF candidate weights; its "
            "coverage average additionally uses globally normalized TF population "
            "weights.</p>"
            + coverage_html
            + "<h3>Prior-matched conditional P-P diagnostic</h3>"
            "<p>For each component, candidates are restricted to the same one of "
            "three true-shear intervals as the truth. TF-target ranks use the "
            "within-galaxy TF weights and their curves use global target-population "
            "galaxy weights. Candidate retention mass and conditional candidate "
            "ESS are reported separately from galaxy ESS.</p>"
            + pp_html
            + "</section>"
        )
        LOGGER.info("%s: finished report assembly", case_name)

    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>KL-NN shear bias report</title><style>"
        "body{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;"
        "padding:0 1rem;color:#1b1b1b}section{margin:3rem 0;border-top:2px solid #ddd}"
        "table{border-collapse:collapse;width:100%;font-size:.88rem;margin:1rem 0}"
        "th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:right}"
        "th:first-child,td:first-child{text-align:left}th{background:#f3f3f3}"
        "img{display:block;max-width:100%;height:auto;margin:1rem auto}"
        "code{background:#f4f4f4;padding:.1rem .25rem}</style></head><body>"
        "<h1>KL-NN shear bias diagnostics</h1>"
        "<p>This report deliberately keeps two estimands separate: "
        "<b>Proposal population / base posterior</b> is the uninformative "
        "simulator population and raw NPE posterior; <b>TF target population / "
        "TF posterior</b> applies post-training prior replacement within each "
        "joint posterior and global importance weighting across galaxies. There "
        "is no model mode, resampling, or hidden TF-conditioned network.</p>"
        + "".join(sections) + "</body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
