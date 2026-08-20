#!/usr/bin/env python3
"""Build a proposal-versus-TF shear report from current posterior caches."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import html
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
SINI_CUTS = (np.inf, 0.9, 0.8, 0.7, 0.6)
LOW_G_DEFAULT = 0.02
ADDITIVE_DISPLAY_SCALE = 1.0e4
MULTIPLICATIVE_DISPLAY_SCALE = 1.0e2


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help="MODEL:DATASET cache pair; repeat to compare cases.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--low-g", type=float, default=LOW_G_DEFAULT)
    parser.add_argument("--bins", type=int, default=8)
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
    spectral_quality = np.asarray(
        load_partitioned_array(cache_partitions, "spectral_reference_quality"),
        dtype=float,
    )
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
        spectral_quality,
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
        spectral_quality = spectral_quality[:take]
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
        "spectral_reference_quality": spectral_quality,
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
            },
            "TF target population / TF posterior": {
                "key": "tf_target",
                "map": target_map,
                "summary": target_summary,
                "mean": target_summary[:, 1],
                "galaxy_weight": target_weight,
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
    sini_index = case["feature_names"].index("sini")
    sini = truth[:, sini_index]
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
                    cuts = {}
                    for cut in SINI_CUTS:
                        cut_mask = (
                            np.ones(len(sini), dtype=bool)
                            if np.isinf(cut)
                            else sini < cut
                        )
                        cut_metrics = component_metrics(
                            component_truth[cut_mask],
                            component_estimate[cut_mask],
                            low_g,
                            weight[cut_mask],
                        )
                        cuts["all" if np.isinf(cut) else f"{cut:.1f}"] = {
                            "m": cut_metrics["low_m"],
                            "m_se": cut_metrics["low_m_se"],
                            "n": cut_metrics["n_low"],
                            "ess": cut_metrics["ess_low"],
                        }
                    row.update(
                        population=population_label,
                        population_key=population["key"],
                        estimator=estimator,
                        frame=frame,
                        component=component,
                        sini_cuts=cuts,
                    )
                    rows.append(row)
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


def conditional_shear_calibration(
    case: dict, population_label: str, bins: int
) -> dict:
    """Weighted full-range Mean-estimator m and c in nuisance bins."""

    population = case["populations"][population_label]
    truth, estimate = case["truth"], population["mean"]
    weights = population["galaxy_weight"]
    names = case["feature_names"]
    conditions = {
        "true magnitude": case["rmag_true"],
        "spectral reference quality": case["spectral_reference_quality"],
        "true hlr": truth[:, names.index("hlr")],
    }
    result = {}
    for condition, axis in conditions.items():
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


def load_shear_pit_values(
    case: dict, population_key: str, block_size: int = 64
) -> dict[str, np.ndarray]:
    """Stream prior-matched conditional ranks from the joint candidate bank.

    Proposal ranks give every candidate equal mass. TF-target ranks instead use
    the cached, within-galaxy TF weights. In both cases the selected shear
    component is restricted to the same fixed prior bin as its truth and the
    retained mass is renormalized before evaluating the rank.
    """

    if population_key not in {"proposal", "tf_target"}:
        raise ValueError("population_key must be proposal or tf_target")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    truth = np.asarray(case["truth"])
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

    pit = np.full((len(truth), 2), np.nan, dtype=np.float64)
    retained_count = np.zeros((len(truth), 2), dtype=np.int64)
    retained_mass = np.zeros((len(truth), 2), dtype=np.float64)
    conditional_ess = np.zeros((len(truth), 2), dtype=np.float64)
    offset = 0
    for part_index, (sample_path, truth_path) in enumerate(
        zip(sample_files, truth_files)
    ):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        stored_truth = np.load(truth_path, mmap_mode="r")
        if samples.ndim != 3 or samples.shape[-1] != len(case["feature_names"]):
            raise ValueError(
                f"Expected flat candidate shape (galaxy, draw, feature), got "
                f"{samples.shape} in {sample_path}"
            )
        if stored_truth.shape != (samples.shape[0], samples.shape[-1]):
            raise ValueError(f"Truth/sample shape mismatch in {sample_path}")
        candidate_weight = (
            np.load(weight_files[part_index], mmap_mode="r")
            if weight_files is not None
            else None
        )
        if candidate_weight is not None and candidate_weight.shape != samples.shape[:2]:
            raise ValueError(f"Candidate-weight shape mismatch in {weight_files[part_index]}")
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
            draws = np.asarray(samples[local_start:local_end, :, :2], dtype=np.float64)
            n_draw = draws.shape[1]
            log_weights = (
                np.asarray(candidate_weight[local_start:local_end], dtype=np.float64)
                if candidate_weight is not None
                else None
            )
            targets = expected_truth[local_start:local_end, :2]
            truth_bins = shear_pp_bin_indices(targets)
            draw_bins = shear_pp_bin_indices(draws)
            for row in range(len(targets)):
                for component in range(2):
                    selected_bin = truth_bins[row, component]
                    if selected_bin < 0:
                        continue
                    keep = (
                        np.isfinite(draws[row, :, component])
                        & (draw_bins[row, :, component] == selected_bin)
                    )
                    if log_weights is None:
                        count = int(np.count_nonzero(keep))
                        conditional = (
                            np.full(count, 1.0 / count) if count else np.array([])
                        )
                        mass = count / n_draw
                    else:
                        keep &= np.isfinite(log_weights[row])
                        selected_log_weight = log_weights[row, keep]
                        if len(selected_log_weight):
                            maximum = float(np.max(selected_log_weight))
                            scaled = np.exp(selected_log_weight - maximum)
                            conditional = scaled / np.sum(scaled)
                            log_mass = maximum + math.log(float(np.sum(scaled)))
                            mass = float(np.exp(log_mass))
                        else:
                            conditional = np.array([])
                            mass = 0.0
                    global_row = offset + local_start + row
                    retained_count[global_row, component] = int(np.count_nonzero(keep))
                    retained_mass[global_row, component] = mass
                    if not len(conditional):
                        continue
                    values = draws[row, keep, component]
                    target = targets[row, component]
                    pit[global_row, component] = float(
                        np.sum(conditional[values < target])
                        + 0.5 * np.sum(conditional[values == target])
                    )
                    conditional_ess[global_row, component] = effective_sample_size(
                        conditional
                    )
        offset += take
        del samples, stored_truth, candidate_weight
    if offset != len(truth):
        raise ValueError(
            f"Sample cache has {offset} galaxies, report uses {len(truth)}"
        )
    return {
        "g1": pit[:, 0],
        "g2": pit[:, 1],
        "g1_retained": retained_count[:, 0],
        "g2_retained": retained_count[:, 1],
        "g1_retained_mass": retained_mass[:, 0],
        "g2_retained_mass": retained_mass[:, 1],
        "g1_conditional_ess": conditional_ess[:, 0],
        "g2_conditional_ess": conditional_ess[:, 1],
    }


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


def conditional_shear_calibration_figure(
    case: dict, population_label: str, curves: dict
) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
    colors = {"g1": "tab:blue", "g2": "tab:orange"}
    for column, (condition, components) in enumerate(curves.items()):
        for row, (metric, scale, ylabel) in enumerate(
            (("m", 100.0, r"$10^2 m$"), ("c", 1.0e4, r"$10^4 c$"))
        ):
            axis = axes[row, column]
            for component, curve in components.items():
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
            if condition == "spectral reference quality" and np.all(
                np.asarray(case["spectral_reference_quality"]) > 0
            ):
                axis.set_xscale("log")
            axis.legend()
    fig.suptitle(f"{case['case']} — {population_label} — Mean estimator")
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


def cuts_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        if row["frame"] != "image":
            continue
        for cut, values in row["sini_cuts"].items():
            body.append(
                "<tr>"
                f"<td>{html.escape(row['population'])}</td>"
                f"<td>{row['estimator']}</td><td>{row['component']}</td>"
                f"<td>{cut}</td>"
                f"<td>{_format_scaled(values['m'], 1e2)} ± "
                f"{_format_scaled(values['m_se'], 1e2)}</td>"
                f"<td>{values['n']:,} / {values['ess']:.1f}</td>"
                "</tr>"
            )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Estimator</th>"
        "<th>Component</th><th>sin i cut</th><th>10<sup>2</sup> low-|g| m</th>"
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
    target_weight = case["populations"][target_label]["galaxy_weight"]
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
        metric_rows = compute_metrics(case, args.low_g)
        coverage_rows = []
        for population_label, population in case["populations"].items():
            coverage_rows.extend(
                coverage_metrics(
                    case["truth"], population["summary"],
                    population["galaxy_weight"], case["feature_names"],
                    population_label,
                )
            )
        pits = {
            population["key"]: load_shear_pit_values(case, population["key"])
            for population in case["populations"].values()
        }
        theta_diagnostics = {
            population["key"]: load_theta_posterior_diagnostics(
                case, population["key"]
            )
            for population in case["populations"].values()
        }
        conditional = []
        for population_label in case["populations"]:
            curves = conditional_shear_calibration(
                case, population_label, args.bins
            )
            conditional.append(
                f"<h4>{html.escape(population_label)}</h4>"
                f"<img src=\"{conditional_shear_calibration_figure(case, population_label, curves)}\" "
                "alt=\"conditional shear calibration\">"
            )
        sections.append(
            f"<section><h2>{html.escape(case['case'])}</h2>"
            "<h3>Importance-sampling health</h3>"
            + importance_table(case)
            + "<h3>Shear calibration</h3>"
            "<p>Every number is an ensemble statistic with the galaxy weights "
            "appropriate to the named population. Low-|g| fits use "
            f"|g| &lt; {args.low_g:g}; cubic fits use the full range.</p>"
            + metrics_table(metric_rows)
            + f"<img src=\"{bias_figure(case, args.bins)}\" alt=\"weighted residual trends\">"
            "<h3>Inclination-cut statistics</h3>"
            "<p>Weights are renormalized inside each cut before fitting.</p>"
            + cuts_table(metric_rows)
            + "<h3>Conditional Mean calibration</h3>"
            "<p>Bin edges, fit coefficients, and errors are population-weighted. "
            "Spectral reference quality is the independently drawn noise-level "
            "control (an SNR-like reference), not a claim that every galaxy has "
            "that achieved emission-line SNR. "
            "The TF-target panel therefore describes the target population, not "
            "the uniform simulator proposal.</p>"
            + "".join(conditional)
            + "<h3>Directed theta_int bias</h3>"
            + f"<img src=\"{theta_bias_figure(case, args.bins)}\" alt=\"theta bias versus truth\">"
            "<p>The mode audit treats theta_int as directed on a 2π domain; "
            "mass near theta+π is not folded onto the true branch. Candidate "
            "histograms use equal proposal mass or the within-galaxy TF weights, "
            "and reported galaxy fractions use the corresponding population "
            "weights. Peaks are descriptive: a twice-smoothed 72-bin circular "
            "histogram requires 10% relative height and 5% local prominence.</p>"
            + theta_modality_table(case, theta_diagnostics)
            + f"<img src=\"{theta_modality_figure(case, theta_diagnostics)}\" alt=\"theta posterior modality\">"
            "<h3>Posterior coverage</h3>"
            "<p>The base posterior uses equal candidate mass and equal galaxy "
            "mass. The TF posterior uses within-galaxy TF candidate weights; its "
            "coverage average additionally uses globally normalized TF population "
            "weights.</p>"
            + coverage_table(coverage_rows)
            + "<h3>Prior-matched conditional P-P diagnostic</h3>"
            "<p>For each component, candidates are restricted to the same one of "
            "three true-shear intervals as the truth. TF-target ranks use the "
            "within-galaxy TF weights and their curves use global target-population "
            "galaxy weights. Candidate retention mass and conditional candidate "
            "ESS are reported separately from galaxy ESS.</p>"
            + f"<img src=\"{posterior_pp_figure(case, pits)}\" alt=\"weighted posterior P-P plots\">"
            "</section>"
        )

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
