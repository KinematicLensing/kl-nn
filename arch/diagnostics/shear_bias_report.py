#!/usr/bin/env python3
"""Build a proposal-versus-TF shear report from current posterior caches."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import html
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstwobign, truncnorm

ARCH_DIR = Path(__file__).resolve().parents[1]
import sys

if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from cache_contract import load_cache_partitions, load_partitioned_array
from tf_prior import effective_sample_size, normalize_population_log_weights
from utils import img_to_gal_axis


LOW_G_DEFAULT = 0.02
ADDITIVE_DISPLAY_SCALE = 1.0e4
MULTIPLICATIVE_DISPLAY_SCALE = 1.0e2
TF_AUDIT_ALPHA = 1.0e-6
TF_AUDIT_QUANTILES = np.asarray(
    [0.01, 0.16, 0.5, 0.84, 0.99], dtype=np.float64
)
COMBINED_TEST_SET_CANDIDATE_WEIGHTING = (
    "tf_x_isotropic_inclination_importance"
)
ORACLE_VCIRC_CANDIDATE_WEIGHTING = "oracle_vcirc_importance"
LOGGER = logging.getLogger(__name__)

NUISANCE_DISPLAY = {
    "theta_int": (1.0, "rad"),
    "cosi": (1.0, ""),
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
    "cosi": "",
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
            "Compose each population weight with posterior shear precision "
            "regularized by the population's cached-posterior shape noise."
        ),
    )
    parser.add_argument(
        "--test-set",
        action="store_true",
        help=(
            "Analyze a compact TF-conformed catalog test cache: use only the "
            "cached prior-replaced posterior Mean and uniform truth-population "
            "mass. "
            "Combine with --weighted to apply shape-noise-regularized "
            "posterior precision across galaxies."
        ),
    )
    parser.add_argument(
        "--response-calibrate",
        type=Path,
        default=None,
        help=(
            "Optional R(C)=aI+bC JSON applied to the last --case. Maps that "
            "case's cached shear Means and streamed shear candidates by "
            "unclipped R^{-1} and recomputes 16th/84th shear summaries on the "
            "transformed draws. Drops galaxies whose R is not invertible or "
            "whose calibrated RMS is at least --max-final-variance. "
            "Not the default Mean."
        ),
    )
    parser.add_argument(
        "--max-final-variance",
        type=float,
        default=None,
        help=(
            "Zero analysis weight when the calibrated RMS "
            "sqrt(tr(C_cal)/2) is at least this value. Default 0.2 when "
            "--response-calibrate is set; ignored otherwise. The threshold "
            "is the prior-box width, not a second moment."
        ),
    )
    return parser.parse_args(argv)


RESPONSE_CALIBRATED_SUFFIX = " · R(C) calibrated"
DEFAULT_RESPONSE_R_MAX = 1.0
DEFAULT_MAX_FINAL_VARIANCE = 0.2
MATRIX_MODEL = "aI_plus_bC"


def shear_interval_sigma(summary: np.ndarray) -> np.ndarray:
    """Combined TF 16–84 half-width of (g1, g2)."""

    summary = np.asarray(summary, dtype=np.float64)
    if summary.ndim != 3 or summary.shape[1] != 3 or summary.shape[2] < 2:
        raise ValueError("posterior summary must have shape (galaxy, 3, feature)")
    half = 0.5 * (summary[:, 2, :2] - summary[:, 0, :2])
    if not np.all(np.isfinite(half)):
        raise ValueError("uncalibrated 16–84 shear width must be finite")
    if np.any(half < 0.0):
        raise ValueError("84th percentile must be at least the 16th for g1 and g2")
    return np.sqrt(0.5 * np.sum(np.square(half), axis=1))


def response_from_sigma(
    sigma: np.ndarray,
    a: float,
    b: float,
    r_min: float,
    r_max: float = DEFAULT_RESPONSE_R_MAX,
) -> np.ndarray:
    """Clip R = a + b σ² into [R_min, R_max]."""

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("R(σ) coefficients must be finite")
    if not (0.0 < r_min <= r_max) or not np.isfinite(r_min) or not np.isfinite(r_max):
        raise ValueError("R clip bounds must satisfy 0 < R_min <= R_max")
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim != 1 or not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("σ must be a finite non-negative vector")
    return np.clip(a + b * np.square(sigma), r_min, r_max)


def load_response_calibration(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"response calibration {path} must be a JSON object")
    model = payload.get("model")
    if model != MATRIX_MODEL:
        raise ValueError(
            f"response calibration {path} has model {model!r}; "
            f"required {MATRIX_MODEL!r}"
        )
    missing = [key for key in ("a", "b", "R_min") if key not in payload]
    if missing:
        raise ValueError(
            f"response calibration {path} is missing {missing}"
        )
    r_max = float(payload.get("R_max", DEFAULT_RESPONSE_R_MAX))
    calibration = {
        "a": float(payload["a"]),
        "b": float(payload["b"]),
        "R_min": float(payload["R_min"]),
        "R_max": r_max,
        "model": MATRIX_MODEL,
        "path": Path(path),
        "payload": payload,
    }
    if not np.isfinite(calibration["a"]) or not np.isfinite(calibration["b"]):
        raise ValueError(f"response calibration {path} has non-finite a or b")
    if not (0.0 < calibration["R_min"] <= calibration["R_max"]):
        raise ValueError(
            f"response calibration {path} has invalid R_min/R_max"
        )
    return calibration


def symmetrize_matrices(matrices: np.ndarray) -> np.ndarray:
    """Return the symmetric part of one or more 2×2 matrices."""

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim == 2:
        if values.shape != (2, 2):
            raise ValueError("single matrix must have shape (2, 2)")
        return 0.5 * (values + values.T)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("matrices must have shape (n, 2, 2) or (2, 2)")
    return 0.5 * (values + np.swapaxes(values, -1, -2))


def batch_weighted_shear_covariance(
    draws: np.ndarray,
    log_weight: np.ndarray | None = None,
) -> np.ndarray:
    """TF- or equal-weight 2×2 covariance of (g1, g2) per galaxy."""

    draws = np.asarray(draws, dtype=np.float64)
    if draws.ndim != 3 or draws.shape[-1] != 2:
        raise ValueError("draws must have shape (galaxy, draw, 2)")
    n_galaxies, n_draw, _ = draws.shape
    finite_draw = np.all(np.isfinite(draws), axis=-1)
    if log_weight is None:
        weight = finite_draw.astype(np.float64)
    else:
        log_weight = np.asarray(log_weight, dtype=np.float64)
        if log_weight.shape != (n_galaxies, n_draw):
            raise ValueError("log_weight must have shape (galaxy, draw)")
        usable = finite_draw & np.isfinite(log_weight)
        filled = np.where(usable, log_weight, -np.inf)
        maximum = np.max(filled, axis=1, keepdims=True)
        weight = np.exp(filled - maximum)
        weight[~usable] = 0.0
        weight[np.squeeze(maximum, axis=1) == -np.inf] = 0.0
    total = np.sum(weight, axis=1, keepdims=True)
    normalized = np.divide(
        weight, total, out=np.zeros_like(weight), where=total > 0.0
    )
    filled_draws = np.where(np.isfinite(draws), draws, 0.0)
    mean = np.sum(normalized[..., None] * filled_draws, axis=1)
    centered = filled_draws - mean[:, None, :]
    centered[~finite_draw] = 0.0
    covariance = np.einsum("nk,nki,nkj->nij", normalized, centered, centered)
    covariance[np.squeeze(total, axis=1) <= 0.0] = np.nan
    return symmetrize_matrices(covariance)


def inverse_matrices(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Invert stacked 2×2 matrices; invalid rows are NaN with ok=False."""

    values = symmetrize_matrices(matrices)
    inverse = np.full_like(values, np.nan)
    finite = np.all(np.isfinite(values), axis=(1, 2))
    ok = np.zeros(len(values), dtype=bool)
    if not np.any(finite):
        return inverse, ok
    subset = values[finite]
    determinant = (
        subset[:, 0, 0] * subset[:, 1, 1] - subset[:, 0, 1] * subset[:, 1, 0]
    )
    invertible = np.abs(determinant) > 1.0e-18
    selected = np.zeros(len(values), dtype=bool)
    selected[np.flatnonzero(finite)[invertible]] = True
    if np.any(selected):
        inverse[selected] = np.linalg.inv(values[selected])
        ok[selected] = True
    return inverse, ok


def calibrated_rms(response: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """σ_final = sqrt(tr(R^{-1} C R^{-T}) / 2). NaN where R is singular."""

    inverse, ok = inverse_matrices(response)
    covariance = symmetrize_matrices(covariance)
    calibrated = np.einsum("nij,njk,nlk->nil", inverse, covariance, inverse)
    variance = 0.5 * (calibrated[:, 0, 0] + calibrated[:, 1, 1])
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    return np.where(ok & np.isfinite(sigma), sigma, np.nan)


def measured_quality_mask(response: np.ndarray) -> np.ndarray:
    """Keep finite SPD measured R for OLS. No σ_final cut."""

    values = symmetrize_matrices(response)
    keep = np.zeros(len(values), dtype=bool)
    finite = np.all(np.isfinite(values), axis=(1, 2))
    if not np.any(finite):
        return keep
    evals = np.linalg.eigvalsh(values[finite])
    ok = np.all(evals > 0.0, axis=1)
    keep[np.flatnonzero(finite)[ok]] = True
    return keep


def measured_keep_mask(
    response: np.ndarray,
    covariance: np.ndarray,
    max_final_variance: float = DEFAULT_MAX_FINAL_VARIANCE,
) -> np.ndarray:
    """Keep galaxies whose measured-R inflated RMS is below the prior-box width."""

    if not np.isfinite(max_final_variance) or max_final_variance <= 0.0:
        raise ValueError("max-final-variance must be a positive finite value")
    sigma = calibrated_rms(response, covariance)
    return np.isfinite(sigma) & (sigma < max_final_variance)


def response_from_covariance(
    covariance: np.ndarray,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unclipped R = aI + bC. Valid where R is SPD (all eigenvalues > 0)."""

    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("R(C) coefficients must be finite")
    covariance = symmetrize_matrices(covariance)
    identity = np.eye(2, dtype=np.float64)
    response = a * identity + b * covariance
    valid = np.zeros(len(covariance), dtype=bool)
    finite = np.all(np.isfinite(response), axis=(1, 2))
    response[~finite] = np.nan
    if not np.any(finite):
        return response, valid
    evals = np.linalg.eigvalsh(response[finite])
    ok = np.all(evals > 0.0, axis=1)
    dest = np.flatnonzero(finite)
    response[dest[~ok]] = np.nan
    valid[dest[ok]] = True
    return response, valid


def fit_ai_plus_bc(response: np.ndarray, covariance: np.ndarray) -> tuple[float, float]:
    """Frobenius OLS of R ≈ a I + b C on stacked 2×2 entries."""

    response = symmetrize_matrices(response)
    covariance = symmetrize_matrices(covariance)
    if response.shape != covariance.shape:
        raise ValueError("response and covariance must have the same shape")
    target = response.reshape(len(response), 4)
    identity = np.broadcast_to(
        np.eye(2, dtype=np.float64).reshape(4), target.shape
    )
    predictor = covariance.reshape(len(covariance), 4)
    finite = np.all(np.isfinite(target), axis=1) & np.all(
        np.isfinite(predictor), axis=1
    )
    if np.count_nonzero(finite) < 2:
        raise ValueError("R = aI + bC requires at least two finite leftover galaxies")
    design = np.column_stack(
        (identity[finite].reshape(-1), predictor[finite].reshape(-1))
    )
    outcome = target[finite].reshape(-1)
    coef, _, rank, _ = np.linalg.lstsq(design, outcome, rcond=None)
    if rank < 2:
        raise ValueError("R = aI + bC design is rank-deficient")
    return float(coef[0]), float(coef[1])


def apply_inverse_to_vectors(
    vectors: np.ndarray, inverse: np.ndarray
) -> np.ndarray:
    """Map shear 2-vectors by per-galaxy R^{-1}."""

    values = np.asarray(vectors, dtype=np.float64)
    inverse = np.asarray(inverse, dtype=np.float64)
    if inverse.ndim != 3 or inverse.shape[1:] != (2, 2):
        raise ValueError("inverse must have shape (galaxy, 2, 2)")
    if values.ndim == 2:
        if values.shape != (len(inverse), 2):
            raise ValueError("vectors must have shape (galaxy, 2)")
        return np.einsum("nij,nj->ni", inverse, values)
    if values.ndim != 3 or values.shape[0] != len(inverse) or values.shape[-1] != 2:
        raise ValueError("vectors must have shape (galaxy, draw, 2)")
    return np.einsum("nij,nkj->nki", inverse, values)


def tf_or_test_population(case: dict) -> dict:
    for population in case["populations"].values():
        if population["key"] in {"tf_target", "test_set"}:
            return population
    raise ValueError(
        "TF or test-set 16–84 summaries are required for response calibration "
        "and shear-width cuts"
    )


def max_sigma_at_response_floor(a: float, b: float, r_min: float) -> float:
    """Unclipped R = a + b σ² equals R_min. Requires b < 0 and a > R_min."""

    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(r_min):
        raise ValueError("R(σ) floor sigma requires finite a, b, and R_min")
    if b >= 0.0 or a <= r_min:
        raise ValueError("R(σ) floor sigma requires a > R_min and b < 0")
    return float(np.sqrt((r_min - a) / b))


def apply_max_shear_sigma_cut(case: dict, max_sigma: float) -> dict:
    """Zero analysis weight for galaxies wider than `max_sigma`. Mutates `case`."""

    if not np.isfinite(max_sigma) or max_sigma <= 0.0:
        raise ValueError("max-shear-sigma must be a positive finite value")
    source = tf_or_test_population(case)
    if case.get("response_calibration") is not None:
        raise ValueError(
            "shear-width cuts must use uncalibrated 16–84 widths; "
            "apply --max-shear-sigma before --response-calibrate"
        )
    sigma = shear_interval_sigma(source["summary"])
    keep = sigma <= max_sigma
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        raise ValueError(
            f"max-shear-sigma {max_sigma:g} dropped every galaxy"
        )
    for population in case["populations"].values():
        galaxy_weight = np.array(
            population["galaxy_weight"], dtype=np.float64, copy=True
        )
        population_mass = np.array(
            population["population_weight"], dtype=np.float64, copy=True
        )
        galaxy_weight[~keep] = 0.0
        population_mass[~keep] = 0.0
        population["galaxy_weight"] = galaxy_weight
        population["population_weight"] = population_mass
    case["shear_sigma_cut"] = {
        "max_shear_sigma": float(max_sigma),
        "n": int(len(keep)),
        "n_kept": n_kept,
        "keep": keep,
        "sigma": sigma,
    }
    suffix = f" · σ ≤ {max_sigma:g}"
    name = str(case["case"])
    if suffix not in name:
        case["case"] = name + suffix
    return case


def shear_sigma_cut_note(case: dict) -> str:
    info = case.get("final_variance_cut") or case.get("shear_sigma_cut")
    if not info:
        return ""
    n_kept = info["n_kept"]
    n = info["n"]
    threshold = info.get("max_final_variance", info.get("max_shear_sigma"))
    kept_pct = 100.0 * n_kept / n
    return (
        "<p>This case zeros analysis weight for galaxies whose calibrated "
        "shear RMS "
        f"<code>σ<sub>final</sub>=√(tr(C<sub>cal</sub>)/2)</code> is at least "
        f"<code>{threshold:g}</code> (prior-box width). Kept {n_kept:,} / {n:,} "
        f"({kept_pct:.1f}%). Cache rows are unchanged; posterior-candidate "
        "diagnostics still stream every galaxy.</p>"
    )


def load_shear_covariances(case: dict, block_size: int = 64) -> np.ndarray:
    """Stream TF/test-set weighted 2×2 shear covariances from the candidate bank."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    truth = np.asarray(case["truth"])
    root = Path(case["root"])
    cache_partitions = case.get("cache_partitions")
    if cache_partitions is None:
        cache_partitions = load_cache_partitions(root)
    candidate_array = case.get("candidate_array", "sample")
    sample_files = cache_partitions.files[candidate_array]
    weight_array = case.get(
        "candidate_log_weight_array", "posterior_tf_log_weight"
    )
    population_keys = {
        population["key"] for population in case["populations"].values()
    }
    weighted = bool({"tf_target", "test_set"} & population_keys)
    weight_files = (
        cache_partitions.files[weight_array] if weighted else None
    )
    covariance = np.full((len(truth), 2, 2), np.nan, dtype=np.float64)
    offset = 0
    for part_index, sample_path in enumerate(sample_files):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        cached_log_weight = (
            np.load(weight_files[part_index], mmap_mode="r")
            if weight_files is not None
            else None
        )
        take = min(samples.shape[0], len(truth) - offset)
        for local_start in range(0, take, block_size):
            local_end = min(take, local_start + block_size)
            draws = np.asarray(
                samples[local_start:local_end, :, :2], dtype=np.float64
            )
            log_weight = (
                np.asarray(
                    cached_log_weight[local_start:local_end], dtype=np.float64
                )
                if cached_log_weight is not None
                else None
            )
            covariance[offset + local_start : offset + local_end] = (
                batch_weighted_shear_covariance(draws, log_weight)
            )
        offset += take
        del samples
        if cached_log_weight is not None:
            del cached_log_weight
    if offset != len(truth):
        raise ValueError(
            f"Sample cache has {offset} galaxies, report uses {len(truth)}"
        )
    return covariance


def apply_response_calibration_to_case(
    case: dict,
    calibration: dict,
    *,
    max_final_variance: float = DEFAULT_MAX_FINAL_VARIANCE,
) -> dict:
    """Apply R(C)=aI+bC inverse to Means and zero weight on σ_final. Mutates `case`."""

    if not np.isfinite(max_final_variance) or max_final_variance <= 0.0:
        raise ValueError("max-final-variance must be a positive finite value")
    covariance = load_shear_covariances(case)
    response, valid = response_from_covariance(
        covariance,
        calibration["a"],
        calibration["b"],
    )
    inverse, invertible = inverse_matrices(response)
    sigma_final = calibrated_rms(response, covariance)
    keep = (
        valid
        & invertible
        & np.isfinite(sigma_final)
        & (sigma_final < max_final_variance)
    )
    n_kept = int(np.count_nonzero(keep))
    if n_kept == 0:
        raise ValueError(
            f"max-final-variance {max_final_variance:g} dropped every galaxy"
        )
    for population in case["populations"].values():
        summary = np.array(population["summary"], dtype=np.float64, copy=True)
        mean_shear = apply_inverse_to_vectors(summary[:, 1, :2], inverse)
        summary[:, 1, :2] = mean_shear
        population["summary"] = summary
        population["mean"] = summary[:, 1]
        galaxy_weight = np.array(
            population["galaxy_weight"], dtype=np.float64, copy=True
        )
        population_mass = np.array(
            population["population_weight"], dtype=np.float64, copy=True
        )
        galaxy_weight[~keep] = 0.0
        population_mass[~keep] = 0.0
        population["galaxy_weight"] = galaxy_weight
        population["population_weight"] = population_mass
        if "map" in population:
            mapped = np.array(population["map"], dtype=np.float64, copy=True)
            mapped[:, :2] = apply_inverse_to_vectors(mapped[:, :2], inverse)
            population["map"] = mapped
        if "map_galaxy_weight" in population:
            map_weight = np.array(
                population["map_galaxy_weight"], dtype=np.float64, copy=True
            )
            map_weight[~keep] = 0.0
            population["map_galaxy_weight"] = map_weight
    if "base_summary" in case:
        base_summary = np.array(case["base_summary"], dtype=np.float64, copy=True)
        base_summary[:, 1, :2] = apply_inverse_to_vectors(
            base_summary[:, 1, :2], inverse
        )
        case["base_summary"] = base_summary
    case["response_calibration"] = {
        "a": calibration["a"],
        "b": calibration["b"],
        "R_min": calibration["R_min"],
        "R_max": calibration.get("R_max", DEFAULT_RESPONSE_R_MAX),
        "model": MATRIX_MODEL,
        "path": calibration["path"],
        "covariance": covariance,
        "R": response,
        "R_inv": inverse,
        "sigma_final": sigma_final,
        "keep": keep,
    }
    case["final_variance_cut"] = {
        "max_final_variance": float(max_final_variance),
        "n": int(len(keep)),
        "n_kept": n_kept,
        "keep": keep,
        "sigma_final": sigma_final,
    }
    name = str(case["case"])
    if not name.endswith(RESPONSE_CALIBRATED_SUFFIX):
        case["case"] = name + RESPONSE_CALIBRATED_SUFFIX
    return case


def response_calibration_note(case: dict) -> str:
    info = case.get("response_calibration")
    if not info:
        return ""
    path = html.escape(str(info["path"]))
    cut = case.get("final_variance_cut") or {}
    n_kept = cut.get("n_kept")
    n = cut.get("n")
    threshold = cut.get("max_final_variance", DEFAULT_MAX_FINAL_VARIANCE)
    kept_text = ""
    if n_kept is not None and n:
        kept_text = (
            f" Kept {n_kept:,} / {n:,} galaxies with invertible "
            f"<code>R=aI+bC</code> and "
            f"σ<sub>final</sub> &lt; {threshold:g}."
        )
    return (
        "<p>This case applies an optional report-time matrix calibration "
        f"<code>R(C)=aI+bC</code> from "
        f"<code>{path}</code> "
        f"(a={info['a']:.6g}, b={info['b']:.6g}), unclipped. "
        "Galaxies whose <code>R</code> is not positive-definite, not "
        "invertible, or whose calibrated RMS "
        f"<code>σ<sub>final</sub>=√(tr(C<sub>cal</sub>)/2)</code> is at least "
        f"<code>{threshold:g}</code> receive zero analysis weight. "
        "Shear Means and streamed candidates are mapped by "
        "<code>R<sup>-1</sup></code>; 16th/84th shear summaries are recomputed "
        "on the transformed draws. Posterior errors inflate through "
        "<code>C<sub>cal</sub>=R<sup>-1</sup> C (R<sup>-1</sup>)<sup>T</sup></code>. "
        "Nuisances and truth are unchanged. "
        f"This is not the default Mean estimator.{kept_text}</p>"
    )


def load_case(
    cache_root: Path,
    case: str,
    max_galaxies: int | None = None,
    *,
    test_set: bool = False,
) -> dict:
    try:
        model, dataset = case.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Case must be MODEL:DATASET, got {case!r}") from exc
    root = cache_root / model / dataset
    cache_partitions = load_cache_partitions(root)
    manifest = cache_partitions.manifests[0]
    analysis_mode = getattr(cache_partitions, "analysis_mode", None)
    if analysis_mode is None:
        analysis_mode = manifest.get("analysis_mode", "standard")
    if test_set and analysis_mode != "test_set":
        raise ValueError(
            f"{case} is not a compact test-set cache; its analysis mode is "
            f"{analysis_mode!r}"
        )
    if not test_set and analysis_mode == "test_set":
        raise ValueError(
            f"{case} is a compact test-set cache; rerun with --test-set"
        )

    feature_names = cache_partitions.feature_names
    truth = np.asarray(load_partitioned_array(cache_partitions, "truth"))
    proposal_summary = np.asarray(
        load_partitioned_array(cache_partitions, "proposal_mean_estimates")
    )
    rmag_true = np.asarray(
        load_partitioned_array(cache_partitions, "rmag_true"), dtype=float
    )
    image_snr = (
        np.asarray(
            load_partitioned_array(cache_partitions, "image_snr"), dtype=float
        )
        if "image_snr" in cache_partitions.files
        else None
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

    n = len(truth)
    expected_summary = (n, 3, len(feature_names))
    if proposal_summary.shape != expected_summary:
        raise ValueError(
            f"proposal_summary shape {proposal_summary.shape}; "
            f"expected {expected_summary}"
        )
    vectors = [rmag_true, spectral_condition]
    if image_snr is not None:
        vectors.append(image_snr)
    if any(value.shape != (n,) for value in vectors):
        raise ValueError(f"Scalar cache length mismatch for {case}")

    proposal_map = target_map = target_summary = population_log_ratio = None
    posterior_ess = posterior_ess_fraction = posterior_max_weight = None
    test_set_provenance = tf_conformance_audit = None
    posterior_candidate_weighting = None
    map_laplace_cov = map_laplace_ok = None
    map_computed = False
    candidate_log_weight_array = "posterior_tf_log_weight"
    target_summary_array = "tf_target_mean_estimates"
    test_set_population_label = "TF-conformed test set / TF posterior"
    candidate_weight_name = "TF importance"
    candidate_weight_health_heading = "TF posterior candidate-weight health"
    if not test_set:
        proposal_map = np.asarray(
            load_partitioned_array(cache_partitions, "proposal_map_estimates")
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
        posterior_ess = np.asarray(
            load_partitioned_array(cache_partitions, "posterior_tf_ess"),
            dtype=float,
        )
        posterior_ess_fraction = np.asarray(
            load_partitioned_array(cache_partitions, "posterior_tf_ess_fraction"),
            dtype=float,
        )
        posterior_max_weight = np.asarray(
            load_partitioned_array(cache_partitions, "posterior_tf_max_weight"),
            dtype=float,
        )
        for name, value, expected in (
            ("proposal_map", proposal_map, truth.shape),
            ("target_map", target_map, truth.shape),
            ("target_summary", target_summary, expected_summary),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} shape {value.shape}; expected {expected}")
        for value in (
            population_log_ratio,
            posterior_ess,
            posterior_ess_fraction,
            posterior_max_weight,
        ):
            if value.shape != (n,):
                raise ValueError(f"Scalar cache length mismatch for {case}")
    else:
        test_set_provenance = getattr(
            cache_partitions, "mode_metadata", manifest.get("test_set")
        )
        if not isinstance(test_set_provenance, dict):
            raise ValueError(
                f"Compact test-set cache {case} is missing manifest.test_set provenance"
            )
        posterior_candidate_weighting = test_set_provenance.get(
            "posterior_candidate_weighting"
        )
        map_computed = test_set_provenance.get("map_computed") is True
        if map_computed:
            proposal_map = np.asarray(
                load_partitioned_array(
                    cache_partitions, "proposal_map_estimates"
                )
            )
            target_map = np.asarray(
                load_partitioned_array(
                    cache_partitions, "tf_target_map_estimates"
                )
            )
            map_laplace_cov = np.asarray(
                load_partitioned_array(cache_partitions, "tf_map_laplace_cov"),
                dtype=np.float64,
            )
            map_laplace_ok = np.asarray(
                load_partitioned_array(cache_partitions, "tf_map_laplace_ok")
            ).astype(bool)
            if proposal_map.shape != truth.shape or target_map.shape != truth.shape:
                raise ValueError(
                    f"MAP estimate shape mismatch for {case}; "
                    f"proposal {proposal_map.shape}, target {target_map.shape}, "
                    f"truth {truth.shape}"
                )
            if map_laplace_cov.shape != (n, 2, 2) or map_laplace_ok.shape != (n,):
                raise ValueError(f"Laplace array shape mismatch for {case}")
        combined_prior = (
            posterior_candidate_weighting
            == COMBINED_TEST_SET_CANDIDATE_WEIGHTING
        )
        if combined_prior:
            target_summary_array = "target_mean_estimates"
            candidate_log_weight_array = "posterior_target_log_weight"
            posterior_ess_array = "posterior_target_ess"
            posterior_ess_fraction_array = "posterior_target_ess_fraction"
            posterior_max_weight_array = "posterior_target_max_weight"
            test_set_population_label = (
                "TF-conformed test set / TF + isotropic-inclination posterior"
            )
            candidate_weight_name = "combined prior-replacement"
            candidate_weight_health_heading = (
                "Combined prior-replacement candidate-weight health"
            )
        else:
            posterior_ess_array = "posterior_tf_ess"
            posterior_ess_fraction_array = "posterior_tf_ess_fraction"
            posterior_max_weight_array = "posterior_tf_max_weight"
            if (
                posterior_candidate_weighting
                == ORACLE_VCIRC_CANDIDATE_WEIGHTING
            ):
                test_set_population_label = (
                    "TF-conformed test set / oracle-vcirc posterior"
                )
                candidate_weight_name = "oracle vcirc importance"
                candidate_weight_health_heading = (
                    "Oracle-vcirc candidate-weight health"
                )
        target_summary = np.asarray(
            load_partitioned_array(cache_partitions, target_summary_array)
        )
        posterior_ess = np.asarray(
            load_partitioned_array(cache_partitions, posterior_ess_array),
            dtype=float,
        )
        posterior_ess_fraction = np.asarray(
            load_partitioned_array(cache_partitions, posterior_ess_fraction_array),
            dtype=float,
        )
        posterior_max_weight = np.asarray(
            load_partitioned_array(cache_partitions, posterior_max_weight_array),
            dtype=float,
        )
        if target_summary.shape != expected_summary:
            raise ValueError(
                f"target_summary shape {target_summary.shape}; "
                f"expected {expected_summary}"
            )
        for value in (
            posterior_ess,
            posterior_ess_fraction,
            posterior_max_weight,
        ):
            if value.shape != (n,):
                raise ValueError(f"Scalar cache length mismatch for {case}")
        generation_manifest = test_set_provenance.get("generation_manifest")
        if not isinstance(generation_manifest, dict):
            raise ValueError(
                f"Compact test-set cache {case} is missing "
                "manifest.test_set.generation_manifest provenance"
            )
        if not isinstance(generation_manifest.get("tf"), dict):
            raise ValueError(
                f"Compact test-set cache {case} is missing "
                "manifest.test_set.generation_manifest.tf provenance"
            )
        if not np.all(np.isfinite(truth)):
            raise ValueError(f"Test-set truth contains non-finite values for {case}")
        cosi = truth[:, feature_names.index("cosi")]
        if np.any((cosi < 0.0) | (cosi > 1.0)):
            raise ValueError(f"Test-set cosi lies outside [0, 1] for {case}")
        hlr = truth[:, feature_names.index("hlr")]
        if np.max(hlr) > 5.0 + 1.0e-6:
            raise ValueError(
                f"Test-set hlr exceeds the 5 arcsec model limit for {case}"
            )
        if image_snr is None:
            raise ValueError(f"Compact test-set cache {case} is missing image_snr")
        for name, value in (
            ("rmag_true", rmag_true),
            ("image_snr", image_snr),
            ("central_halpha_snr", spectral_condition),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Test-set {name} contains non-finite values for {case}"
                )
        if np.any(image_snr <= 0.0) or np.any(spectral_condition <= 0.0):
            raise ValueError(f"Test-set S/N values must be positive for {case}")
        tf_conformance_audit = compute_test_set_tf_conformance_audit(
            truth[:, feature_names.index("vcirc")],
            rmag_true,
            test_set_provenance.get("tf"),
            generation_tf=generation_manifest.get("tf"),
        )

    if max_galaxies is not None:
        if max_galaxies <= 0:
            raise ValueError("max_galaxies must be positive")
        take = min(max_galaxies, n)
        truth = truth[:take]
        proposal_summary = proposal_summary[:take]
        rmag_true = rmag_true[:take]
        if image_snr is not None:
            image_snr = image_snr[:take]
        spectral_condition = spectral_condition[:take]
        if not test_set:
            proposal_map = proposal_map[:take]
            target_map = target_map[:take]
            target_summary = target_summary[:take]
            population_log_ratio = population_log_ratio[:take]
            posterior_ess = posterior_ess[:take]
            posterior_ess_fraction = posterior_ess_fraction[:take]
            posterior_max_weight = posterior_max_weight[:take]
        else:
            target_summary = target_summary[:take]
            posterior_ess = posterior_ess[:take]
            posterior_ess_fraction = posterior_ess_fraction[:take]
            posterior_max_weight = posterior_max_weight[:take]
            if map_computed:
                proposal_map = proposal_map[:take]
                target_map = target_map[:take]
                map_laplace_cov = map_laplace_cov[:take]
                map_laplace_ok = map_laplace_ok[:take]
        n = take

    proposal_weight = np.full(n, 1.0 / n, dtype=np.float64)
    common = {
        "case": case,
        "model": model,
        "dataset": dataset,
        "root": root,
        "cache_partitions": cache_partitions,
        "feature_names": feature_names,
        "truth": truth,
        "rmag_true": rmag_true,
        "image_snr": image_snr,
        "spectral_condition": spectral_condition,
        "spectral_condition_name": spectral_condition_name,
        "spectral_condition_log_scale": spectral_condition_log_scale,
        "analysis_mode": analysis_mode,
        "dataset_size": cache_partitions.dataset_size,
        "analyzed_size": n,
    }
    if test_set:
        test_set_population = {
            "key": "test_set",
            "summary": target_summary,
            "mean": target_summary[:, 1],
            "galaxy_weight": proposal_weight,
            "population_weight": proposal_weight.copy(),
        }
        if map_computed:
            test_set_population["map"] = target_map
            test_set_population["map_galaxy_weight"] = proposal_weight.copy()
        common.update(
            report_map=map_computed,
            map_computed=map_computed,
            map_laplace_cov=map_laplace_cov,
            map_laplace_ok=map_laplace_ok,
            candidate_array="shear_sample",
            candidate_log_weight_array=candidate_log_weight_array,
            posterior_candidate_weighting=posterior_candidate_weighting,
            target_summary_array=target_summary_array,
            test_set_population_label=test_set_population_label,
            candidate_weight_name=candidate_weight_name,
            candidate_weight_health_heading=candidate_weight_health_heading,
            test_set_provenance=test_set_provenance,
            tf_conformance_audit=tf_conformance_audit,
            posterior_candidate_ess=posterior_ess,
            posterior_candidate_ess_fraction=posterior_ess_fraction,
            posterior_candidate_max_weight=posterior_max_weight,
            base_summary=proposal_summary,
            populations={
                test_set_population_label: test_set_population
            },
        )
        if posterior_candidate_weighting != COMBINED_TEST_SET_CANDIDATE_WEIGHTING:
            common.update(
                posterior_tf_ess=posterior_ess,
                posterior_tf_ess_fraction=posterior_ess_fraction,
                posterior_tf_max_weight=posterior_max_weight,
            )
        return common

    target_weight = normalize_population_log_weights(population_log_ratio)
    common.update(
        posterior_tf_ess=posterior_ess,
        posterior_tf_ess_fraction=posterior_ess_fraction,
        posterior_tf_max_weight=posterior_max_weight,
        populations={
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
    )
    return common


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


def compose_shape_noise_regularized_weights(
    base_weight: np.ndarray,
    g1_variance: np.ndarray,
    g2_variance: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Compose population mass with shape-noise-regularized shear precision.

    The per-galaxy posterior shape noise is
    ``sqrt((Var(g1) + Var(g2)) / 2)``. Its population-weighted first-pass mean
    is frozen as the ensemble shape-noise floor. The common spin-symmetric
    precision is then ``1 / (sigma_gal**2 + sigma_shape**2)``. Galaxies with
    non-finite or negative posterior variance receive zero analysis weight.
    """

    base_weight = np.asarray(base_weight, dtype=np.float64)
    g1_variance = np.asarray(g1_variance, dtype=np.float64)
    g2_variance = np.asarray(g2_variance, dtype=np.float64)
    if base_weight.ndim != 1 or not (
        g1_variance.shape == g2_variance.shape == base_weight.shape
    ):
        raise ValueError("population weights and shear variances must be vectors")

    valid_base = np.isfinite(base_weight) & (base_weight >= 0.0)
    galaxy_variance = 0.5 * (g1_variance + g2_variance)
    valid_variance = (
        np.isfinite(g1_variance)
        & np.isfinite(g2_variance)
        & (g1_variance >= 0.0)
        & (g2_variance >= 0.0)
        & np.isfinite(galaxy_variance)
    )
    usable = valid_base & valid_variance
    if not np.any(usable) or np.sum(base_weight[usable]) <= 0.0:
        raise ValueError("no positive population mass has valid shear variances")

    galaxy_shape_noise = np.sqrt(np.clip(galaxy_variance, 0.0, None))
    shape_noise, shape_noise_se, shape_noise_ess = weighted_mean_and_se(
        galaxy_shape_noise[usable], base_weight[usable]
    )
    if not np.isfinite(shape_noise) or shape_noise <= 0.0:
        raise ValueError("ensemble posterior shape noise must be finite and positive")

    regularized_variance = galaxy_variance + shape_noise**2
    valid_regularized = usable & np.isfinite(regularized_variance) & (
        regularized_variance > 0.0
    )
    precision = np.zeros(base_weight.shape, dtype=np.float64)
    precision[valid_regularized] = 1.0 / regularized_variance[valid_regularized]

    combined = np.zeros(base_weight.shape, dtype=np.float64)
    combined[valid_regularized] = (
        base_weight[valid_regularized] * precision[valid_regularized]
    )
    total = float(np.sum(combined))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("regularized precision-weighted mass is not positive")
    combined /= total

    weighted_shape_noise, weighted_shape_noise_se, weighted_shape_noise_ess = (
        weighted_mean_and_se(galaxy_shape_noise, combined)
    )

    valid_base_mass = float(np.sum(base_weight[valid_base]))
    invalid_variance = ~valid_variance
    invalid_mass = float(np.sum(base_weight[valid_base & invalid_variance]))
    diagnostics = {
        "shape_noise": shape_noise,
        "shape_noise_se": shape_noise_se,
        "shape_noise_ess": shape_noise_ess,
        "shape_noise_variance": shape_noise**2,
        "weighted_shape_noise": weighted_shape_noise,
        "weighted_shape_noise_se": weighted_shape_noise_se,
        "weighted_shape_noise_ess": weighted_shape_noise_ess,
        "invalid_variance_count": int(np.count_nonzero(invalid_variance)),
        "invalid_variance_fraction": float(np.mean(invalid_variance)),
        "invalid_variance_population_mass": (
            invalid_mass / valid_base_mass if valid_base_mass > 0.0 else float("nan")
        ),
        "usable_count": int(np.count_nonzero(valid_regularized)),
        "population_ess": effective_sample_size(
            base_weight[valid_base]
            / np.sum(base_weight[valid_base])
        ),
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


def shape_noise_weight_comparison(
    case: dict,
    posterior_diagnostics: dict[str, dict[str, np.ndarray]],
    low_g: float,
) -> list[dict]:
    """Compare population-only and regularized Mean-estimator summaries."""

    rows = []
    truth = np.asarray(case["truth"], dtype=np.float64)
    for population_label, population in case["populations"].items():
        posterior = posterior_diagnostics[population["key"]]
        base_weight = population_weight(population)
        regularized_weight, diagnostics = compose_shape_noise_regularized_weights(
            base_weight,
            posterior["g1_variance"],
            posterior["g2_variance"],
        )
        for weighting, weight in (
            ("Population only", base_weight),
            ("Shape-noise regularized", regularized_weight),
        ):
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
                    "weighting": weighting,
                    **diagnostics,
                    "reported_shape_noise": (
                        diagnostics["shape_noise"]
                        if weighting == "Population only"
                        else diagnostics["weighted_shape_noise"]
                    ),
                    "reported_shape_noise_se": (
                        diagnostics["shape_noise_se"]
                        if weighting == "Population only"
                        else diagnostics["weighted_shape_noise_se"]
                    ),
                    "reported_shape_noise_ess": (
                        diagnostics["shape_noise_ess"]
                        if weighting == "Population only"
                        else diagnostics["weighted_shape_noise_ess"]
                    ),
                    "reported_ess": (
                        diagnostics["population_ess"]
                        if weighting == "Population only"
                        else diagnostics["ess"]
                    ),
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


def mean_vs_map_weight_comparison(
    case: dict,
    posterior_diagnostics: dict[str, dict[str, np.ndarray]],
    low_g: float,
) -> list[dict]:
    """Compare Mean vs MAP under population and estimator-specific weights."""

    truth = np.asarray(case["truth"], dtype=np.float64)
    rows = []
    for population_label, population in case["populations"].items():
        if "map" not in population:
            continue
        posterior = posterior_diagnostics[population["key"]]
        base_weight = population_weight(population)
        mean_reg, mean_diag = compose_shape_noise_regularized_weights(
            base_weight,
            posterior["g1_variance"],
            posterior["g2_variance"],
        )
        cov = np.asarray(case["map_laplace_cov"], dtype=np.float64)
        ok = np.asarray(case["map_laplace_ok"], dtype=bool)
        g1_var = np.where(ok, cov[:, 0, 0], np.nan)
        g2_var = np.where(ok, cov[:, 1, 1], np.nan)
        try:
            map_reg, map_diag = compose_shape_noise_regularized_weights(
                base_weight, g1_var, g2_var
            )
        except ValueError:
            map_reg = np.zeros_like(base_weight)
            map_diag = {
                "shape_noise": float("nan"),
                "shape_noise_se": float("nan"),
                "shape_noise_ess": 0.0,
                "weighted_shape_noise": float("nan"),
                "weighted_shape_noise_se": float("nan"),
                "weighted_shape_noise_ess": 0.0,
                "invalid_variance_count": int(np.size(ok)),
                "invalid_variance_fraction": 1.0,
                "invalid_variance_population_mass": 1.0,
                "population_ess": float("nan"),
                "ess": 0.0,
            }
        map_abs = np.abs(np.asarray(population["map"][:, :2], dtype=np.float64))
        wall_fraction = float(np.mean(np.any(map_abs >= 0.095, axis=1)))
        estimators = (
            ("Mean", population["mean"], "Population only", base_weight, mean_diag),
            (
                "Mean",
                population["mean"],
                "Shape-noise regularized",
                mean_reg,
                mean_diag,
            ),
            ("MAP", population["map"], "Population only", base_weight, map_diag),
            (
                "MAP",
                population["map"],
                "Laplace shape-noise regularized",
                map_reg,
                map_diag,
            ),
        )
        for estimator, estimate, weighting, weight, diagnostics in estimators:
            component_rows = [
                component_metrics(
                    truth[:, index], estimate[:, index], low_g, weight
                )
                for index in range(2)
            ]
            first_pass = weighting == "Population only"
            rows.append(
                {
                    "population": population_label,
                    "population_key": population["key"],
                    "estimator": estimator,
                    "weighting": weighting,
                    **diagnostics,
                    "reported_shape_noise": (
                        diagnostics["shape_noise"]
                        if first_pass
                        else diagnostics["weighted_shape_noise"]
                    ),
                    "reported_shape_noise_se": (
                        diagnostics["shape_noise_se"]
                        if first_pass
                        else diagnostics["weighted_shape_noise_se"]
                    ),
                    "reported_shape_noise_ess": (
                        diagnostics["shape_noise_ess"]
                        if first_pass
                        else diagnostics["weighted_shape_noise_ess"]
                    ),
                    "reported_ess": (
                        diagnostics["population_ess"]
                        if first_pass
                        else diagnostics["ess"]
                    ),
                    "g1_m": component_rows[0]["low_m"],
                    "g1_m_se": component_rows[0]["low_m_se"],
                    "g1_c": component_rows[0]["c"],
                    "g1_c_se": component_rows[0]["c_se"],
                    "g1_ess": component_rows[0]["ess_low"],
                    "g1_n": component_rows[0]["n_low"],
                    "g2_m": component_rows[1]["low_m"],
                    "g2_m_se": component_rows[1]["low_m_se"],
                    "g2_c": component_rows[1]["c"],
                    "g2_c_se": component_rows[1]["c_se"],
                    "g2_ess": component_rows[1]["ess_low"],
                    "g2_n": component_rows[1]["n_low"],
                    "laplace_invalid_fraction": float(np.mean(~ok)),
                    "map_wall_fraction": wall_fraction,
                }
            )
    return rows


def apply_precision_weighting(
    case: dict,
    posterior_diagnostics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict]:
    """Install shape-noise-regularized weights for downstream sections."""

    applied = {}
    for population_label, population in case["populations"].items():
        posterior = posterior_diagnostics[population["key"]]
        weight, diagnostics = compose_shape_noise_regularized_weights(
            population_weight(population),
            posterior["g1_variance"],
            posterior["g2_variance"],
        )
        population["galaxy_weight"] = weight
        population["precision_weighting"] = diagnostics
        applied[population_label] = diagnostics
    case["precision_weighted"] = True
    case["shape_noise_regularized"] = True
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
        estimators = [("Mean", population["mean"])]
        if case.get("report_map", True):
            estimators.insert(0, ("MAP", population["map"]))
        for estimator, estimate in estimators:
            weight = population["galaxy_weight"]
            if estimator == "MAP" and "map_galaxy_weight" in population:
                weight = population["map_galaxy_weight"]
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
        estimators = [("Mean", population["mean"])]
        if case.get("report_map", True):
            estimators.append(("MAP", population["map"]))
        for estimator, estimate in estimators:
            if estimator == "MAP" and "map_galaxy_weight" in population:
                weight = np.asarray(
                    population["map_galaxy_weight"], dtype=np.float64
                )
            else:
                weight = np.asarray(population["galaxy_weight"], dtype=np.float64)
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
    """Mean-estimator nuisance residuals in common reference-truth bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    truth = np.asarray(case["truth"], dtype=np.float64)
    reference = next(
        (
            population
            for population in case["populations"].values()
            if population["key"] == "proposal"
        ),
        next(iter(case["populations"].values()), None),
    )
    if reference is None:
        raise ValueError("nuisance plots require at least one population")
    reference_weight = population_weight(reference)
    curves = {}
    for index, parameter in enumerate(case["feature_names"]):
        if parameter in {"g1", "g2"}:
            continue
        true = truth[:, index]
        finite_truth = (
            np.isfinite(true)
            & np.isfinite(reference_weight)
            & (reference_weight >= 0.0)
        )
        if not np.any(finite_truth) or np.sum(reference_weight[finite_truth]) <= 0.0:
            continue
        edges = weighted_quantile(
            true[finite_truth],
            np.linspace(0.0, 1.0, bins + 1),
            reference_weight[finite_truth],
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
                reference_weight, selected
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
    }
    if case.get("image_snr") is not None:
        conditions["image S/N"] = case["image_snr"]
    conditions.update(
        {
            spectral_condition_name: spectral_condition,
            "true hlr": truth[:, names.index("hlr")],
            "true cosi": truth[:, names.index("cosi")],
        }
    )
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
    """Stream active-population shear ranks and variances in one candidate pass."""

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    truth = np.asarray(case["truth"])
    root = Path(case["root"])
    cache_partitions = case.get("cache_partitions")
    if cache_partitions is None:
        cache_partitions = load_cache_partitions(root)
    candidate_array = case.get("candidate_array", "sample")
    sample_files = cache_partitions.files[candidate_array]
    truth_files = cache_partitions.files["truth"]
    population_keys = tuple(
        population["key"] for population in case["populations"].values()
    )
    weighted_population_keys = {"tf_target", "test_set"} & set(population_keys)
    candidate_log_weight_array = case.get(
        "candidate_log_weight_array", "posterior_tf_log_weight"
    )
    weight_files = (
        cache_partitions.files[candidate_log_weight_array]
        if weighted_population_keys
        else None
    )
    result = {
        key: _empty_shear_posterior_diagnostic(len(truth))
        for key in population_keys
    }
    inverse_all = case.get("response_calibration", {}).get("R_inv")
    if inverse_all is not None:
        for population in case["populations"].values():
            population["summary"] = np.array(
                population["summary"], dtype=np.float64, copy=True
            )

    offset = 0
    for part_index, (sample_path, truth_path) in enumerate(
        zip(sample_files, truth_files)
    ):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        stored_truth = np.load(truth_path, mmap_mode="r")
        cached_log_weight = (
            np.load(weight_files[part_index], mmap_mode="r")
            if weight_files is not None
            else None
        )
        expected_candidate_features = (
            2 if candidate_array == "shear_sample" else len(case["feature_names"])
        )
        if samples.ndim != 3 or samples.shape[-1] != expected_candidate_features:
            raise ValueError(
                f"Expected candidate shape (galaxy, draw, "
                f"{expected_candidate_features}), got "
                f"{samples.shape} in {sample_path}"
            )
        if stored_truth.shape != (
            samples.shape[0], len(case["feature_names"])
        ):
            raise ValueError(f"Truth/sample shape mismatch in {sample_path}")
        if (
            cached_log_weight is not None
            and cached_log_weight.shape != samples.shape[:2]
        ):
            raise ValueError(
                "Candidate-weight shape mismatch in "
                f"{weight_files[part_index]}"
            )
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
            inverse = case.get("response_calibration", {}).get("R_inv")
            if inverse is not None:
                draws = apply_inverse_to_vectors(
                    draws,
                    inverse[offset + local_start : offset + local_end],
                )
            candidate_log_weight = (
                np.asarray(
                    cached_log_weight[local_start:local_end], dtype=np.float64
                )
                if cached_log_weight is not None
                else None
            )
            targets = expected_truth[local_start:local_end, :2]
            for row in range(len(targets)):
                global_row = offset + local_start + row
                candidate_logs = {
                    key: (
                        candidate_log_weight[row]
                        if key in weighted_population_keys
                        and candidate_log_weight is not None
                        else None
                    )
                    for key in population_keys
                }
                for component in range(2):
                    finite = np.isfinite(draws[row, :, component])
                    for population_key, log_weight in candidate_logs.items():
                        diagnostic = result[population_key]
                        diagnostic["posterior_variance"][global_row, component] = (
                            posterior_component_variance(
                                draws[row, :, component], log_weight
                            )
                        )
                        keep, conditional, mass = _conditional_candidate_mass(
                            draws[row, :, component], finite, log_weight
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
                        if inverse is None:
                            continue
                        lower, upper = weighted_quantile(
                            values,
                            np.asarray((0.16, 0.84), dtype=np.float64),
                            conditional,
                        )
                        for population in case["populations"].values():
                            if population["key"] != population_key:
                                continue
                            summary = population["summary"]
                            summary[global_row, 0, component] = lower
                            summary[global_row, 2, component] = upper
                            population["mean"] = summary[:, 1]
        offset += take
        del samples, stored_truth
        if cached_log_weight is not None:
            del cached_log_weight
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

    diagnostics = load_shear_posterior_diagnostics(case, block_size)
    if population_key not in diagnostics:
        raise ValueError(
            f"population_key must be one of {sorted(diagnostics)}, "
            f"got {population_key!r}"
        )
    return diagnostics[population_key]


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
            estimators = [("Mean", population["mean"], "tab:orange")]
            if case.get("report_map", True):
                estimators.insert(0, ("MAP", population["map"], "tab:blue"))
            for estimator, estimate, color in estimators:
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
    """Overlay active-population posterior-mean nuisance residual trends."""

    curves = nuisance_bias_curves(case, bins)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), squeeze=False)
    colors = {
        "proposal": "tab:blue",
        "tf_target": "tab:orange",
        "test_set": "tab:purple",
    }
    short_labels = {
        "proposal": "Original",
        "tf_target": "TF",
        "test_set": "Test set",
    }
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
        "shape-noise-regularized precision weighted"
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
    column_count = len(curves)
    fig, axes = plt.subplots(
        3,
        column_count,
        figsize=(4.5 * column_count, 11),
        squeeze=False,
    )
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
    for row, (population_label, population) in enumerate(populations):
        pits = pits_by_population[population["key"]]
        galaxy_weight = population["galaxy_weight"]
        for component_index, component in enumerate(("g1", "g2")):
            axis = axes[row, component_index]
            selected = np.isfinite(pits[component])
            x, y, distance, ess = posterior_pp_curve(
                pits[component][selected], galaxy_weight[selected]
            )
            label = f"N={len(x):,}, ESS={ess:.1f}, KS={distance:.3f}"
            if len(x):
                epsilon = math.sqrt(math.log(40.0) / (2.0 * ess))
                axis.fill_between(
                    grid,
                    np.maximum(0.0, grid - epsilon),
                    np.minimum(1.0, grid + epsilon),
                    color="tab:blue", alpha=0.12,
                )
            axis.step(
                x, y, where="post", color="tab:blue", lw=1.4, label=label,
            )
            axis.plot(grid, grid, color="black", ls="--", lw=0.9, label="Uniform")
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, 1.0)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("posterior quantile")
            axis.set_title(f"{population_label} — {component}")
            axis.legend(fontsize="small", loc="upper left")
        axes[row, 0].set_ylabel("weighted fraction of truths")
    description = (
        "empirical test-set P-P"
        if case.get("analysis_mode") == "test_set"
        else "prior-matched P-P"
    )
    fig.suptitle(f"{case['case']} — {description}")
    fig.tight_layout()
    return fig_data_uri(fig)


def _flatten_provenance(value: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten bounded JSON provenance for compact HTML display."""

    rows = []
    for key in sorted(value):
        item = value[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            rows.extend(_flatten_provenance(item, name))
        elif isinstance(item, list) and len(item) > 12:
            rows.append((name, f"<{len(item)} values>"))
        else:
            rows.append((name, json.dumps(item, sort_keys=True)))
    return rows


def test_set_provenance_table(case: dict) -> str:
    """Render the generation and analysis contract for a compact test cache."""

    provenance = case.get("test_set_provenance")
    if not isinstance(provenance, dict) or not isinstance(
        provenance.get("generation_manifest"), dict
    ):
        raise ValueError(
            f"{case['case']} lacks required compact test-set generation provenance"
        )
    operative_weight = (
        "1 / (posterior shear variance + fixed ensemble shape-noise variance)"
        if case.get("precision_weighted")
        else "equal over generated test galaxies (population only)"
    )
    rows = [
        ("cache dataset rows", f"{case['dataset_size']:,}"),
        ("report rows", f"{case['analyzed_size']:,}"),
        ("posterior estimator", "Mean and 1D MAP" if case.get("map_computed") else "Mean only"),
        (
            "posterior candidate mass",
            f"{case.get('candidate_weight_name', 'TF importance')} weights "
            "normalized within each galaxy",
        ),
        ("pre-precision galaxy mass", "equal over generated test galaxies"),
        (
            "operative analysis weight",
            operative_weight,
        ),
        ("cache analysis mode", "test_set"),
    ]
    cut = case.get("final_variance_cut") or case.get("shear_sigma_cut")
    if cut is not None:
        threshold = cut.get("max_final_variance", cut.get("max_shear_sigma"))
        rows.append(
            (
                "calibrated shear RMS cut",
                f"σ_final < {threshold:g}; kept "
                f"{cut['n_kept']:,} / {cut['n']:,} analysis weight",
            )
        )
    display_provenance = {
        key: value
        for key, value in provenance.items()
        if key != "map_computed"
    }
    rows.extend(_flatten_provenance(display_provenance, "test_set"))
    body = "".join(
        "<tr>"
        f"<td><code>{html.escape(name)}</code></td>"
        f"<td>{html.escape(value)}</td></tr>"
        for name, value in rows
    )
    return (
        "<table><thead><tr><th>Contract field</th><th>Value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _uniform_ks(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    values = values[np.isfinite(values)]
    if not len(values) or np.any((values < 0.0) | (values > 1.0)):
        return float("nan")
    after = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
    before = np.arange(len(values), dtype=np.float64) / len(values)
    return float(max(np.max(after - values), np.max(values - before)))


def compute_test_set_tf_conformance_audit(
    vcirc: np.ndarray,
    rmag_true: np.ndarray,
    embedded_tf: dict,
    *,
    generation_tf: dict | None = None,
) -> dict:
    """Audit cached truth against its embedded truncated TF conditional.

    This calculation deliberately does not call the generation sampler. It
    independently maps every cached (rmag_true, vcirc) row through the CDF of
    the configured normal in log10(vcirc), truncated to the configured
    physical velocity support. Correctly generated rows therefore have a
    uniform probability-integral transform (PIT).
    """

    expected_fields = {
        "slope", "intercept", "scatter_dex", "vcirc_min", "vcirc_max"
    }
    if not isinstance(embedded_tf, dict) or set(embedded_tf) != expected_fields:
        raise ValueError(
            "test_set.tf must contain exactly slope, intercept, scatter_dex, "
            "vcirc_min, and vcirc_max"
        )
    config = {}
    for name in expected_fields:
        value = embedded_tf[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"test_set.tf.{name} must be finite numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"test_set.tf.{name} must be finite numeric")
        config[name] = value
    if config["slope"] == 0.0:
        raise ValueError("test_set.tf.slope must be non-zero")
    if config["scatter_dex"] <= 0.0:
        raise ValueError("test_set.tf.scatter_dex must be positive")
    if not 0.0 < config["vcirc_min"] < config["vcirc_max"]:
        raise ValueError(
            "test_set.tf velocity bounds must satisfy 0 < vcirc_min < vcirc_max"
        )
    if generation_tf is not None and generation_tf != embedded_tf:
        raise ValueError(
            "test_set.generation_manifest.tf must exactly match test_set.tf"
        )

    velocity = np.asarray(vcirc, dtype=np.float64)
    magnitude = np.asarray(rmag_true, dtype=np.float64)
    if velocity.ndim != 1 or magnitude.ndim != 1 or velocity.shape != magnitude.shape:
        raise ValueError("cached vcirc and rmag_true must be matching vectors")
    if not len(velocity):
        raise ValueError("TF-conformance audit requires at least one cached row")
    if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(magnitude)):
        raise ValueError(
            "cached vcirc and rmag_true must be finite for TF-conformance audit"
        )
    on_support = (velocity >= config["vcirc_min"]) & (
        velocity <= config["vcirc_max"]
    )
    if not np.all(on_support):
        invalid = np.flatnonzero(~on_support)
        preview = invalid[:10].tolist()
        raise ValueError(
            "cached vcirc lies outside embedded test_set.tf support at rows "
            f"{preview}" + ("..." if len(invalid) > len(preview) else "")
        )

    mean_log10 = (magnitude - config["intercept"]) / config["slope"]
    residual_dex = np.log10(velocity) - mean_log10
    standardized = residual_dex / config["scatter_dex"]
    lower = (math.log10(config["vcirc_min"]) - mean_log10) / config[
        "scatter_dex"
    ]
    upper = (math.log10(config["vcirc_max"]) - mean_log10) / config[
        "scatter_dex"
    ]
    intermediates = (mean_log10, residual_dex, standardized, lower, upper)
    if not all(np.all(np.isfinite(value)) for value in intermediates):
        raise ValueError("TF-conformance transform produced non-finite values")
    if np.any(lower >= upper):
        raise ValueError("embedded TF truncation interval is empty")

    pit = np.asarray(truncnorm.cdf(standardized, lower, upper), dtype=np.float64)
    if not np.all(np.isfinite(pit)) or np.any((pit < 0.0) | (pit > 1.0)):
        raise ValueError("truncated TF conditional CDF produced invalid values")

    ks_distance = _uniform_ks(pit)
    if not math.isfinite(ks_distance):
        raise ValueError("TF-conformance KS distance is not finite")
    row_count = len(pit)
    dkw_threshold = min(
        1.0, math.sqrt(math.log(2.0 / TF_AUDIT_ALPHA) / (2.0 * row_count))
    )
    ks_pvalue = float(kstwobign.sf(math.sqrt(row_count) * ks_distance))
    if not math.isfinite(ks_pvalue):
        raise ValueError("TF-conformance KS probability is not finite")
    pit_quantiles = np.quantile(pit, TF_AUDIT_QUANTILES)
    residual_quantiles = np.quantile(residual_dex, TF_AUDIT_QUANTILES)
    standardized_quantiles = np.quantile(standardized, TF_AUDIT_QUANTILES)
    quantile_max_error = float(
        np.max(np.abs(pit_quantiles - TF_AUDIT_QUANTILES))
    )
    return {
        "config": config,
        "row_count": row_count,
        "pit_min": float(np.min(pit)),
        "pit_max": float(np.max(pit)),
        "ks_distance": ks_distance,
        "ks_pvalue_asymptotic": ks_pvalue,
        "significance_alpha": TF_AUDIT_ALPHA,
        "dkw_distance_threshold": dkw_threshold,
        "uniformity_status": "PASS" if ks_distance <= dkw_threshold else "FAIL",
        "residual_status": "PASS",
        "quantile_status": (
            "PASS" if quantile_max_error <= dkw_threshold else "FAIL"
        ),
        "quantile_max_abs_error": quantile_max_error,
        "quantile_probabilities": TF_AUDIT_QUANTILES.copy(),
        "pit_quantiles": pit_quantiles,
        "residual_dex_quantiles": residual_quantiles,
        "standardized_residual_quantiles": standardized_quantiles,
    }


def test_set_tf_conformance_table(case: dict) -> str:
    """Render the independent cached-truth TF probability-integral audit."""

    audit = case.get("tf_conformance_audit")
    if not isinstance(audit, dict):
        raise ValueError(f"{case['case']} lacks a cached-truth TF audit")
    config = audit.get("config")
    probabilities = np.asarray(audit.get("quantile_probabilities"), dtype=float)
    pit_quantiles = np.asarray(audit.get("pit_quantiles"), dtype=float)
    residual_quantiles = np.asarray(
        audit.get("residual_dex_quantiles"), dtype=float
    )
    standardized_quantiles = np.asarray(
        audit.get("standardized_residual_quantiles"), dtype=float
    )
    expected_shape = TF_AUDIT_QUANTILES.shape
    if not isinstance(config, dict) or any(
        value.shape != expected_shape
        for value in (
            probabilities,
            pit_quantiles,
            residual_quantiles,
            standardized_quantiles,
        )
    ):
        raise ValueError(f"{case['case']} has malformed cached-truth TF audit data")

    status_rows = (
        ("rows audited", f"{audit['row_count']:,}"),
        (
            "embedded conditional",
            "log10(vcirc) | rmag ~ Normal((rmag - intercept) / slope, "
            "scatter_dex), truncated to [vcirc_min, vcirc_max]",
        ),
        (
            "embedded TF parameters",
            ", ".join(f"{name}={config[name]:g}" for name in sorted(config)),
        ),
        ("velocity support / finite residual status", audit["residual_status"]),
        (
            "truncated-CDF PIT range",
            f"[{audit['pit_min']:.6g}, {audit['pit_max']:.6g}]",
        ),
        ("uniform KS distance D", f"{audit['ks_distance']:.6g}"),
        (
            "asymptotic uniform-KS p-value",
            f"{audit['ks_pvalue_asymptotic']:.6g}",
        ),
        (
            f"DKW distance threshold (alpha={audit['significance_alpha']:.0e})",
            f"{audit['dkw_distance_threshold']:.6g}",
        ),
        ("uniformity status", audit["uniformity_status"]),
        (
            "PIT quantile status / maximum absolute deviation",
            f"{audit['quantile_status']} / {audit['quantile_max_abs_error']:.6g}",
        ),
    )
    status_body = "".join(
        "<tr>"
        f"<td><code>{html.escape(label)}</code></td>"
        f"<td>{html.escape(str(value))}</td></tr>"
        for label, value in status_rows
    )
    quantile_body = "".join(
        "<tr>"
        f"<td>{probability:.2f}</td>"
        f"<td>{residual:.6g}</td>"
        f"<td>{standardized_value:.6g}</td>"
        f"<td>{pit_value:.6g}</td>"
        f"<td>{pit_value - probability:+.6g}</td></tr>"
        for probability, residual, standardized_value, pit_value in zip(
            probabilities,
            residual_quantiles,
            standardized_quantiles,
            pit_quantiles,
        )
    )
    return (
        "<table><thead><tr><th>TF audit field</th><th>Value</th></tr></thead>"
        f"<tbody>{status_body}</tbody></table>"
        "<table><thead><tr><th>quantile</th><th>TF residual [dex]</th>"
        "<th>standardized residual</th><th>truncated-CDF PIT</th>"
        "<th>PIT − nominal</th></tr></thead>"
        f"<tbody>{quantile_body}</tbody></table>"
    )


def test_set_audit_table(case: dict) -> str:
    """Summarize cached empirical inputs and core generation invariants."""

    truth = np.asarray(case["truth"], dtype=np.float64)
    names = case["feature_names"]
    cosi = truth[:, names.index("cosi")]
    sini = np.sqrt(np.clip(1.0 - np.square(cosi), 0.0, 1.0))
    values = {
        "rmag_true": case["rmag_true"],
        "image_snr": case["image_snr"],
        "central_halpha_snr": case["spectral_condition"],
        "vcirc [km/s]": truth[:, names.index("vcirc")],
        "hlr [arcsec]": truth[:, names.index("hlr")],
        "cosi": cosi,
        "reconstructed sini": sini,
    }
    body = []
    for label, raw in values.items():
        raw = np.asarray(raw, dtype=np.float64)
        finite = raw[np.isfinite(raw)]
        quantile = (
            np.quantile(finite, [0.0, 0.16, 0.5, 0.84, 1.0])
            if len(finite)
            else np.full(5, np.nan)
        )
        body.append(
            "<tr>"
            f"<td><code>{html.escape(label)}</code></td>"
            f"<td>{len(finite):,} / {len(raw):,}</td>"
            + "".join(f"<td>{value:.5g}</td>" for value in quantile)
            + "</tr>"
        )
    cached_rows_status = (
        "PASS"
        if case["dataset_size"] == 100_000
        else "CHECK: final requested size is 100,000"
    )
    truncation = (
        "none"
        if case["analyzed_size"] == case["dataset_size"]
        else f"debug prefix {case['analyzed_size']:,}/{case['dataset_size']:,}"
    )
    hlr_max = float(np.max(truth[:, names.index("hlr")]))
    return (
        f"<p><b>Row-count check:</b> {cached_rows_status}; report truncation: "
        f"{html.escape(truncation)}. <b>HLR upper-limit check:</b> "
        f"{'PASS' if hlr_max <= 5.0 + 1.0e-6 else 'FAIL'} "
        f"(max {hlr_max:.5g} arcsec). <b>Inclination check:</b> reconstructed "
        f"cos(i) uniform KS distance = {_uniform_ks(cosi):.5g}.</p>"
        "<table><thead><tr><th>Cached quantity</th><th>finite / N</th>"
        "<th>min</th><th>16th</th><th>median</th><th>84th</th><th>max</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def cross_case_summary_table(results: list[dict]) -> str:
    """Compare operative Mean shear calibration across test-set cuts."""

    body = []
    for result in results:
        weighting = result["weighting"]
        operative = next(
            row
            for row in result["weight_rows"]
            if row["weighting"] == weighting
        )
        for component in ("g1", "g2"):
            metric = next(
                row
                for row in result["metric_rows"]
                if row["estimator"] == "Mean"
                and row["frame"] == "image"
                and row["component"] == component
            )
            body.append(
                "<tr>"
                f"<td>{html.escape(result['case'])}</td><td>{component}</td>"
                f"<td>{html.escape(weighting)}</td>"
                f"<td>{_format_scaled(metric['c'], 1e4)} ± "
                f"{_format_scaled(metric['c_se'], 1e4)}</td>"
                f"<td>{_format_scaled(metric['low_m'], 1e2)} ± "
                f"{_format_scaled(metric['low_m_se'], 1e2)}</td>"
                f"<td>{operative['reported_shape_noise']:.5g}</td>"
                f"<td>{metric['n']:,} / {metric['ess']:.1f}</td>"
                "</tr>"
            )
    return (
        "<table><thead><tr><th>Case / catalog cut</th><th>Component</th>"
        "<th>Operative galaxy weighting</th>"
        "<th>10<sup>4</sup> c</th><th>10<sup>2</sup> low-|g| m</th>"
        "<th>reported σ<sub>shape</sub></th><th>N / ESS</th></tr></thead>"
        "<tbody>" + "".join(body) + "</tbody></table>"
    )


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


def shape_noise_weight_comparison_table(rows: list[dict]) -> str:
    """Render two-pass shape noise, invalid variances, ESS, and Mean m."""

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['population'])}</td>"
            f"<td>{row['weighting']}</td>"
            f"<td>{row['shape_noise']:.5g}</td>"
            f"<td>{row['reported_shape_noise']:.5g} ± "
            f"{row['reported_shape_noise_se']:.3g} / "
            f"{row['reported_shape_noise_ess']:.1f}</td>"
            f"<td>{row['invalid_variance_count']:,} / "
            f"{100 * row['invalid_variance_fraction']:.3f}% rows / "
            f"{100 * row['invalid_variance_population_mass']:.3f}% mass</td>"
            f"<td>{row['reported_ess']:.1f}</td>"
            f"<td>{_format_scaled(row['g1_m'], 1e2)} ± "
            f"{_format_scaled(row['g1_m_se'], 1e2)}</td>"
            f"<td>{row['g1_n']:,} / {row['g1_ess']:.1f}</td>"
            f"<td>{_format_scaled(row['g2_m'], 1e2)} ± "
            f"{_format_scaled(row['g2_m_se'], 1e2)}</td>"
            f"<td>{row['g2_n']:,} / {row['g2_ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Population / posterior</th><th>Weighting</th>"
        "<th>fixed first-pass σ<sub>shape</sub></th>"
        "<th>reported σ<sub>shape</sub> ± SE / ESS</th>"
        "<th>Invalid variance</th><th>Overall ESS</th>"
        "<th>10<sup>2</sup> g1 m</th><th>g1 N / fit ESS</th>"
        "<th>10<sup>2</sup> g2 m</th><th>g2 N / fit ESS</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def mean_vs_map_weight_comparison_table(rows: list[dict]) -> str:
    """Render Mean vs MAP m, c, and estimator-specific shape-noise weights."""

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['estimator'])}</td>"
            f"<td>{html.escape(row['weighting'])}</td>"
            f"<td>{_format_scaled(row['g1_c'], 1e4)} ± "
            f"{_format_scaled(row['g1_c_se'], 1e4)}</td>"
            f"<td>{_format_scaled(row['g1_m'], 1e2)} ± "
            f"{_format_scaled(row['g1_m_se'], 1e2)}</td>"
            f"<td>{_format_scaled(row['g2_c'], 1e4)} ± "
            f"{_format_scaled(row['g2_c_se'], 1e4)}</td>"
            f"<td>{_format_scaled(row['g2_m'], 1e2)} ± "
            f"{_format_scaled(row['g2_m_se'], 1e2)}</td>"
            f"<td>{row['reported_shape_noise']:.5g}</td>"
            f"<td>{row['reported_ess']:.1f}</td>"
            f"<td>{100 * row['laplace_invalid_fraction']:.2f}%</td>"
            f"<td>{100 * row['map_wall_fraction']:.2f}%</td>"
            f"<td>{row['g1_n']:,} / {row['g1_ess']:.1f}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Weighting</th>"
        "<th>10<sup>4</sup> g1 c</th><th>10<sup>2</sup> g1 m</th>"
        "<th>10<sup>4</sup> g2 c</th><th>10<sup>2</sup> g2 m</th>"
        "<th>reported σ<sub>shape</sub></th><th>ESS</th>"
        "<th>Laplace invalid</th><th>|g<sub>MAP</sub>|≥0.095</th>"
        "<th>g1 N / fit ESS</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
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


def test_set_candidate_weight_table(case: dict) -> str:
    """Summarize candidate weights without reweighting truth galaxies."""

    population = next(iter(case["populations"].values()))
    galaxy_weight = population_weight(population)
    rows = []
    for label, values in (
        ("posterior candidate ESS", case["posterior_candidate_ess"]),
        (
            "posterior candidate ESS fraction",
            case["posterior_candidate_ess_fraction"],
        ),
        (
            "largest posterior candidate weight",
            case["posterior_candidate_max_weight"],
        ),
    ):
        quantiles = weighted_quantile(
            values, np.asarray((0.16, 0.5, 0.84)), galaxy_weight
        )
        mean, _, _ = weighted_mean_and_se(values, galaxy_weight)
        invalid = int(np.count_nonzero(~np.isfinite(values)))
        rows.append(
            f"<tr><td>{label}</td><td>{mean:.4g}</td>"
            f"<td>{quantiles[0]:.4g}</td><td>{quantiles[1]:.4g}</td>"
            f"<td>{quantiles[2]:.4g}</td><td>{invalid:,}</td></tr>"
        )
    candidate_weight_name = html.escape(
        case.get("candidate_weight_name", "TF importance")
    )
    return (
        f"<p>These diagnostics describe {candidate_weight_name} weights over posterior "
        "candidates within each galaxy. Their summary uses uniform "
        "pre-precision galaxy mass; no TF population ratio is applied across "
        "the already-TF-conformed truth rows.</p>"
        "<table><thead><tr><th>Diagnostic</th><th>mean</th>"
        "<th>16th</th><th>median</th><th>84th</th>"
        "<th>non-finite rows</th></tr></thead><tbody>"
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
    calibration = (
        load_response_calibration(args.response_calibrate)
        if args.response_calibrate is not None
        else None
    )
    cases = [
        load_case(
            args.cache_root,
            value,
            args.max_galaxies,
            test_set=args.test_set,
        )
        for value in args.case
    ]
    if args.max_final_variance is not None and calibration is None:
        raise ValueError("max-final-variance requires --response-calibrate")
    if calibration is not None:
        max_final_variance = (
            DEFAULT_MAX_FINAL_VARIANCE
            if args.max_final_variance is None
            else args.max_final_variance
        )
        if max_final_variance <= 0.0:
            raise ValueError("max-final-variance must be positive")
        apply_response_calibration_to_case(
            cases[-1],
            calibration,
            max_final_variance=max_final_variance,
        )
        LOGGER.info(
            "%s: R(C) calibrated; kept %d / %d galaxies with σ_final < %g",
            cases[-1]["case"],
            cases[-1]["final_variance_cut"]["n_kept"],
            cases[-1]["final_variance_cut"]["n"],
            max_final_variance,
        )
    sections = []
    cross_case_results = []
    for case in cases:
        case_name = case["case"]
        case["report_map"] = not args.weighted
        if args.test_set:
            case["report_map"] = bool(case.get("map_computed"))
        LOGGER.info("%s: starting report", case_name)
        if args.test_set:
            importance_html = test_set_candidate_weight_table(case)
            LOGGER.info(
                "%s: finished section: %s",
                case_name,
                case["candidate_weight_health_heading"],
            )
        else:
            importance_html = importance_table(case)
            LOGGER.info(
                "%s: finished section: importance-sampling health", case_name
            )

        LOGGER.info("%s: streaming posterior candidate diagnostics", case_name)
        pits = load_shear_posterior_diagnostics(case)
        LOGGER.info("%s: finished posterior candidate diagnostics", case_name)
        weight_comparison_rows = shape_noise_weight_comparison(
            case, pits, args.low_g
        )
        weight_comparison_html = shape_noise_weight_comparison_table(
            weight_comparison_rows
        )
        map_weight_rows = []
        map_weight_html = ""
        if case.get("map_computed"):
            map_weight_rows = mean_vs_map_weight_comparison(
                case, pits, args.low_g
            )
            map_weight_html = (
                "<h3>Mean vs MAP with estimator-specific weights</h3>"
                "<p>MAP is the TF-weighted 1D Gaussian-KDE mode of "
                "<code>g1</code> and of <code>g2</code> on "
                "[-0.1, 0.1]. Nuisance coordinates in the stored MAP array "
                "are the TF-weighted Mean, not a 9D density argmax. Laplace "
                "weights use the 1D curvature "
                "σ² = -1 / d²log p̂/dg² at each shear mode and set wall or "
                "non-curved densities to zero weight. Mean shape-noise weights "
                "still use posterior sample variance; they are not applied to "
                "MAP.</p>"
                + mean_vs_map_weight_comparison_table(map_weight_rows)
            )
        if args.weighted:
            applied = apply_precision_weighting(case, pits)
            for population_label, diagnostic in applied.items():
                LOGGER.info(
                    "%s: %s shape-noise-regularized precision: "
                    "sigma_shape %.5g -> %.5g, ESS %.1f, invalid variance %d",
                    case_name,
                    population_label,
                    diagnostic["shape_noise"],
                    diagnostic["weighted_shape_noise"],
                    diagnostic["ess"],
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
        conditional_estimators = (
            ("Mean", "MAP") if case.get("report_map", True) else ("Mean",)
        )
        for estimator in conditional_estimators:
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
        conditional_map_section = ""
        if case.get("report_map", True):
            conditional_map_section = (
                "<h3>Conditional MAP calibration</h3>"
                "<p>The m and c panels use MAP point estimates. Shape noise is a "
                "property of the corresponding posterior, so its panels match the "
                "Mean section.</p>"
                + conditional_html["MAP"]
            )
        pp_html = (
            f"<img src=\"{posterior_pp_figure(case, pits)}\" "
            "alt=\"weighted posterior P-P plots\">"
        )
        LOGGER.info("%s: finished section: P-P diagnostic", case_name)
        if args.test_set:
            combined_prior = (
                case["posterior_candidate_weighting"]
                == COMBINED_TEST_SET_CANDIDATE_WEIGHTING
            )
            operative_weighting = (
                "Shape-noise regularized" if args.weighted else "Population only"
            )
            operative_weight_description = (
                "shape-noise-regularized posterior precision"
                if args.weighted
                else "equal truth-galaxy mass without posterior-precision weighting"
            )
            oracle_prior = (
                case["posterior_candidate_weighting"]
                == ORACLE_VCIRC_CANDIDATE_WEIGHTING
            )
            if combined_prior:
                prior_replacement_description = (
                    "Inference replaces the uniform training priors on circular "
                    "velocity and inclination with the declared TF and isotropic-"
                    "inclination priors using combined prior-replacement importance "
                    "weights over joint posterior candidates within each galaxy. "
                    "Truth galaxies retain uniform population mass."
                )
                posterior_variance_description = (
                    "Each galaxy's posterior shear variance after combined "
                    "prior-replacement weighting is"
                )
                interval_description = (
                    "Intervals come from posterior candidates after combined "
                    "prior-replacement weighting and"
                )
                rank_description = (
                    "posterior ranks use renormalized combined prior-replacement "
                    "candidate weights."
                )
            elif oracle_prior:
                prior_replacement_description = (
                    "Inference replaces the uniform training prior on circular "
                    "velocity with an oracle log-normal centered on each galaxy's "
                    "true vcirc (same 0.1 dex TF scatter). This is not a real-data "
                    "estimator; the prior mean is unavailable on the sky. "
                    "Truth galaxies retain uniform population mass."
                )
                posterior_variance_description = (
                    "Each galaxy's oracle-vcirc-weighted posterior shear variance is"
                )
                interval_description = (
                    "Intervals come from oracle-vcirc-weighted posterior candidates and"
                )
                rank_description = (
                    "posterior ranks use renormalized oracle-vcirc candidate weights."
                )
            else:
                prior_replacement_description = (
                    "Inference replaces the uniform training prior on circular "
                    "velocity with the declared TF prior by importance weighting "
                    "joint posterior candidates within each galaxy. Truth galaxies "
                    "retain uniform population mass."
                )
                posterior_variance_description = (
                    "Each galaxy's TF-weighted posterior shear variance is"
                )
                interval_description = (
                    "Intervals come from TF-weighted posterior candidates and"
                )
                rank_description = (
                    "posterior ranks use renormalized TF candidate weights."
                )
            cross_case_results.append(
                {
                    "case": case_name,
                    "metric_rows": metric_rows,
                    "weight_rows": weight_comparison_rows,
                    "weighting": operative_weighting,
                }
            )
            sections.append(
                f"<section><h2>{html.escape(case_name)}</h2>"
                + shear_sigma_cut_note(case)
                + response_calibration_note(case)
                + "<h3>Test-set provenance and generation contract</h3>"
                "<p>This section is rendered from the compact cache's embedded "
                "generation manifest. The catalog-selected truth population is "
                "already TF-conformed. "
                + prior_replacement_description
                + "</p>"
                + test_set_provenance_table(case)
                + "<h3>Independent TF-conformance audit</h3>"
                "<p>Every cached truth row is transformed independently through "
                "the embedded, correctly truncated conditional TF CDF. A "
                "TF-conformed cache has a uniform probability-integral transform. "
                "The KS and quantile statuses therefore expose a generation, "
                "LMDB, or cache row-alignment mismatch without applying any TF "
                "truth-population weight to the shear analysis.</p>"
                + test_set_tf_conformance_table(case)
                + "<h3>Cached-input audit</h3>"
                "<p>The table reports the exact values passed through the held-out "
                "dataset and cache. Image and central H-alpha S/N are record-backed "
                "and are neither redrawn nor clipped at posterior-cache time.</p>"
                + test_set_audit_table(case)
                + f"<h3>{html.escape(case['candidate_weight_health_heading'])}</h3>"
                + importance_html
                + "<h3>Galaxy-weighting and shape-noise comparison</h3>"
                f"<p>{posterior_variance_description} "
                "σ²<sub>gal,j</sub> = [Var(g1) + Var(g2)] / 2. The fixed "
                "ensemble floor is the equal-population mean of "
                "σ<sub>gal,j</sub>, and the regularized alternative is "
                "w<sub>j</sub> ∝ 1 / "
                "[σ²<sub>gal,j</sub> + σ²<sub>shape</sub>]. Both population-only "
                "and regularized results are shown here. The operative row for "
                f"subsequent ensemble statistics is <b>{operative_weighting}</b>. "
                "Invalid variances receive zero regularized weight.</p>"
                + weight_comparison_html
                + map_weight_html
                + (
                    "<h3>Shear calibration — Mean and MAP</h3>"
                    "<p>Low-|g| fits use "
                    f"|g| &lt; {args.low_g:g}; cubic fits use the full range. "
                    f"Mean results use {operative_weight_description}. "
                    "MAP ensemble metrics use equal galaxy mass (Laplace "
                    "reweighting is only in the Mean vs MAP table). MAP shear "
                    "is the 1D KDE mode; MAP nuisances are TF-weighted Means.</p>"
                    if case.get("map_computed")
                    else (
                        "<h3>Shear calibration — posterior Mean</h3>"
                        "<p>Only the posterior Mean is reported. Low-|g| fits use "
                        f"|g| &lt; {args.low_g:g}; cubic fits use the full range. "
                        f"Results use {operative_weight_description}.</p>"
                    )
                )
                + shear_html
                + "<h3>Nuisance-parameter calibration — posterior Mean</h3>"
                "<p>Entries are weighted means of estimate minus truth. Error "
                "bars are weighted standard errors; legend values are full-range "
                "residual slopes. The <code>theta_int</code> residual is wrapped "
                "to [-π, π).</p>"
                + nuisance_html
                + "<h3>Conditional Mean calibration</h3>"
                "<p>Weighted-quantile bins show m, c, and posterior shape noise "
                "against catalog r magnitude, catalog image S/N, catalog central "
                "H-alpha S/N, true HLR, and true sin(i). The two S/N coordinates "
                "are the exact cached observation controls.</p>"
                + conditional_html["Mean"]
                + "<h3>Empirical test-set posterior coverage</h3>"
                f"<p>{interval_description} "
                "are averaged over the selected test population with the operative "
                f"{operative_weight_description}. This is empirical "
                "held-out coverage, not simulation-based calibration under the "
                "training proposal.</p>"
                + coverage_html
                + "<h3>Empirical test-set P-P diagnostic</h3>"
                "<p>Each panel is an ordinary one-dimensional P-P plot of the "
                "weighted posterior PIT against the corresponding true shear, "
                "with no binning of the true shear. "
                + rank_description
                + " Curves use the operative galaxy weights. Because the "
                "test-set shear proposal is still uniform, this diagnostic "
                "does not by itself distinguish a prior-dominated shear "
                "posterior from a well-calibrated informative one.</p>"
                + pp_html
                + "</section>"
            )
            LOGGER.info("%s: finished test-set report assembly", case_name)
            continue

        sections.append(
            f"<section><h2>{html.escape(case['case'])}</h2>"
            + shear_sigma_cut_note(case)
            + response_calibration_note(case)
            + "<h3>Importance-sampling health</h3>"
            + importance_html
            + "<h3>Shape-noise-regularized precision comparison</h3>"
            "<p>The first pass defines each galaxy's posterior variance as "
            "σ²<sub>gal,j</sub> = [Var(g1) + Var(g2)] / 2 and estimates the "
            "ensemble σ<sub>shape</sub> as the population-weighted mean of "
            "σ<sub>gal,j</sub>. The fixed floor then gives "
            "w<sub>j</sub> ∝ w<sub>pop,j</sub> / "
            "[σ²<sub>gal,j</sub> + σ²<sub>shape</sub>]. There are no caps or "
            "percentile cuts. The table compares population-only and "
            "regularized Mean-estimator m, ESS, and ensemble shape noise; the "
            "regularized shape noise is recomputed without iterating the "
            "floor. Invalid posterior variances receive zero regularized "
            "weight.</p>"
            + weight_comparison_html
            + "<h3>Shear calibration</h3>"
            "<p>Every number is an ensemble statistic with the galaxy weights "
            "appropriate to the named population. Low-|g| fits use "
            f"|g| &lt; {args.low_g:g}; cubic fits use the full range. "
            + (
                "This report applies shape-noise-regularized posterior precision. "
                "MAP diagnostics are omitted because these weights are defined "
                "from posterior spread around the Mean."
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
            + conditional_map_section
            + "<h3>Posterior coverage</h3>"
            "<p>The base posterior uses equal candidate mass and equal galaxy "
            "mass. The TF posterior uses within-galaxy TF candidate weights; its "
            "coverage average additionally uses globally normalized TF population "
            "weights.</p>"
            + coverage_html
            + "<h3>Prior-matched P-P diagnostic</h3>"
            "<p>Each panel is an ordinary one-dimensional P-P plot of the "
            "weighted posterior PIT against the corresponding true shear, with "
            "no binning of the true shear. TF-target ranks use the "
            "within-galaxy TF weights and their curves use global "
            "target-population galaxy weights. Because the proposal shear is "
            "still uniform, this diagnostic does not by itself distinguish a "
            "prior-dominated shear posterior from a well-calibrated "
            "informative one.</p>"
            + pp_html
            + "</section>"
        )
        LOGGER.info("%s: finished report assembly", case_name)

    if args.test_set:
        operative_weight_description = (
            "shape-noise-regularized posterior precision"
            if args.weighted
            else "equal truth-galaxy mass without posterior-precision weighting"
        )
        report_heading = "KL-NN TF-conformed catalog test-set diagnostics"
        combined_modes = [
            case["posterior_candidate_weighting"]
            == COMBINED_TEST_SET_CANDIDATE_WEIGHTING
            for case in cases
        ]
        oracle_modes = [
            case["posterior_candidate_weighting"]
            == ORACLE_VCIRC_CANDIDATE_WEIGHTING
            for case in cases
        ]
        if all(combined_modes):
            posterior_population_description = (
                "TF + isotropic-inclination prior-replaced"
            )
        elif all(oracle_modes):
            posterior_population_description = "oracle-vcirc prior-replaced"
        elif any(combined_modes):
            posterior_population_description = "declared prior-replaced"
        elif any(oracle_modes):
            posterior_population_description = "TF-weighted or oracle-vcirc"
        else:
            posterior_population_description = "TF-weighted"
        map_modes = [bool(case.get("map_computed")) for case in cases]
        if all(map_modes):
            map_description = (
                "and the TF-weighted 1D KDE MAP of g1 and g2 with 1D Laplace "
                "variances."
            )
        elif any(map_modes):
            map_description = (
                "and sample MAP plus Laplace covariances when the cache stored them."
            )
        else:
            map_description = "and no MAP estimator."
        report_intro = (
            "<p>Each named case is an independently generated empirical catalog "
            "selection. The truth population already follows the declared "
            "Tully–Fisher relation. The report uses one "
            f"{posterior_population_description} posterior "
            f"population, the posterior Mean, {operative_weight_description}, "
            f"{map_description}</p>"
        )
        cross_case_html = (
            "<section><h2>Cross-cut operative shear summary</h2>"
            f"<p>All rows use the same Mean estimator and "
            f"{operative_weight_description} defined in the per-case sections.</p>"
            + cross_case_summary_table(cross_case_results)
            + "</section>"
        )
    else:
        report_heading = "KL-NN shear bias diagnostics"
        report_intro = (
            "<p>This report deliberately keeps two estimands separate: "
            "<b>Proposal population / base posterior</b> is the uninformative "
            "simulator population and raw NPE posterior; <b>TF target population / "
            "TF posterior</b> applies post-training prior replacement within each "
            "joint posterior and global importance weighting across galaxies. There "
            "is no model mode, resampling, or hidden TF-conditioned network.</p>"
        )
        cross_case_html = ""

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
        f"<h1>{report_heading}</h1>"
        + report_intro
        + cross_case_html
        + "".join(sections)
        + "</body></html>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
