#!/usr/bin/env python3
"""Fit a matrix shear response R(C)=aI+bC on a matched five-point stencil cache.

Writes a JSON production input for report-time R^{-1} and a companion HTML
scatter. This diagnostic is not METHODS-monitored.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIAG_DIR = Path(__file__).resolve().parent
ARCH_DIR = DIAG_DIR.parent
for path in (str(DIAG_DIR), str(ARCH_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cache_contract import load_cache_partitions, load_partitioned_array
from shear_bias_report import (
    DEFAULT_MAX_FINAL_VARIANCE,
    MATRIX_MODEL,
    batch_weighted_shear_covariance,
    calibrated_rms,
    fit_ai_plus_bc,
    measured_quality_mask,
    symmetrize_matrices,
)
from shear_response_report import STATE_ORDER, build_matched_cubes


DEFAULT_CACHE = Path(
    "/ocean/projects/phy250048p/shared/cache/"
    "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_45255702/"
    "shear_response_simv3_cosi_5k_shear_response_candidates"
)
DEFAULT_MANIFEST = Path(
    "/ocean/projects/phy250048p/shared/samples/"
    "shear_response_simv3_cosi_5k_manifest.csv"
)
DEFAULT_OUTPUT = Path(
    "/ocean/projects/phy250048p/shared/reports/xu3-estimators/"
    "r_matrix_allpoints_frozen_concat_45255702.json"
)
DEFAULT_SEED = 31415
DEFAULT_R_MIN = 0.25
DEFAULT_R_MAX = 1.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--r-min", type=float, default=DEFAULT_R_MIN)
    parser.add_argument("--r-max", type=float, default=DEFAULT_R_MAX)
    parser.add_argument(
        "--max-final-variance",
        type=float,
        default=DEFAULT_MAX_FINAL_VARIANCE,
        help=(
            "Recorded apply-time σ_final threshold for the HTML caption. "
            "Not used to select OLS points."
        ),
    )
    return parser.parse_args(argv)


def load_zero_shear_covariances(
    cache_partitions,
    base_column: np.ndarray,
    state_column: np.ndarray,
    nbase: int,
) -> np.ndarray:
    """Equal-weight 2×2 C_α from zero-shear candidate draws, streamed by partition."""

    zero = STATE_ORDER.index("zero")
    sample_files = cache_partitions.files["sample"]
    n_rows = len(base_column)
    if n_rows != len(state_column):
        raise ValueError("base_id and state columns must have the same length")
    covariance = np.full((nbase, 2, 2), np.nan, dtype=np.float64)
    filled = np.zeros(nbase, dtype=bool)
    offset = 0
    for path in sample_files:
        if offset >= n_rows:
            break
        samples = np.load(path, mmap_mode="r")
        if samples.ndim != 3 or samples.shape[-1] < 2:
            raise ValueError(f"sample cache must contain g1 and g2 in {path}")
        take = min(samples.shape[0], n_rows - offset)
        states = state_column[offset : offset + take]
        bases = base_column[offset : offset + take]
        local = np.flatnonzero(states == zero)
        if len(local):
            dest = bases[local]
            if np.any(dest < 0) or np.any(dest >= nbase):
                raise ValueError("zero-shear base_id is outside the matched cube")
            if np.any(filled[dest]):
                raise ValueError("duplicate zero-shear rows in the stencil cache")
            for start in range(0, len(local), 64):
                idx = local[start : start + 64]
                dest_chunk = dest[start : start + 64]
                draws = np.asarray(samples[idx, :, :2], dtype=np.float64)
                covariance[dest_chunk] = batch_weighted_shear_covariance(draws)
            filled[dest] = True
        offset += take
        del samples
    if offset != n_rows:
        raise ValueError(
            f"sample cache has {offset} galaxies, stencil uses {n_rows}"
        )
    if not np.all(filled):
        raise ValueError("every matched base needs a zero-shear sample row")
    return covariance


def finite_difference_response(
    estimate_cube: np.ndarray, truth_cube: np.ndarray
) -> tuple[np.ndarray, float]:
    """Per-base Mean finite-difference Jacobian (nbase, 2, 2)."""

    code = {name: index for index, name in enumerate(STATE_ORDER)}
    delta_g1 = 0.5 * (
        truth_cube[:, code["g1_plus"], 0] - truth_cube[:, code["g1_minus"], 0]
    )
    delta_g2 = 0.5 * (
        truth_cube[:, code["g2_plus"], 1] - truth_cube[:, code["g2_minus"], 1]
    )
    if (
        np.any(delta_g1 <= 0.0)
        or np.any(delta_g2 <= 0.0)
        or not np.allclose(delta_g1, delta_g1[0], rtol=1e-6, atol=1e-7)
        or not np.allclose(delta_g2, delta_g2[0], rtol=1e-6, atol=1e-7)
        or not np.isclose(delta_g1[0], delta_g2[0], rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("matched truths do not have one common positive shear step")
    delta = float(delta_g1[0])
    response = np.empty((len(estimate_cube), 2, 2), dtype=np.float64)
    response[:, :, 0] = (
        estimate_cube[:, code["g1_plus"]] - estimate_cube[:, code["g1_minus"]]
    ) / (2.0 * delta)
    response[:, :, 1] = (
        estimate_cube[:, code["g2_plus"]] - estimate_cube[:, code["g2_minus"]]
    ) / (2.0 * delta)
    if not np.all(np.isfinite(response)):
        raise ValueError("finite-difference response must be finite")
    return symmetrize_matrices(response), delta


def frobenius_residual_rms(
    response: np.ndarray, covariance: np.ndarray, a: float, b: float
) -> float:
    residual = symmetrize_matrices(response) - (
        a * np.eye(2, dtype=np.float64) + b * symmetrize_matrices(covariance)
    )
    return float(np.sqrt(np.mean(np.sum(np.square(residual), axis=(1, 2)))))


def fit_with_holdout(
    response: np.ndarray,
    covariance: np.ndarray,
    keep: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float,
) -> dict:
    keep = np.asarray(keep, dtype=bool)
    leftover = np.flatnonzero(keep)
    if len(leftover) < 4:
        raise ValueError("at least four leftover bases are required for R(C)")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie strictly between 0 and 1")
    a, b = fit_ai_plus_bc(response[leftover], covariance[leftover])
    rng = np.random.default_rng(seed)
    order = rng.permutation(leftover)
    split = int(round(calibration_fraction * len(leftover)))
    split = min(max(split, 2), len(leftover) - 2)
    calibration, holdout = order[:split], order[split:]
    predicted = a * np.eye(2) + b * covariance
    residual = response - predicted
    cal_residual = residual[calibration]
    ss_res = float(np.sum(np.square(cal_residual)))
    centered = response[calibration] - np.mean(response[calibration], axis=0)
    ss_tot = float(np.sum(np.square(centered)))
    r_squared = float("nan") if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    return {
        "a": a,
        "b": b,
        "r_squared": r_squared,
        "holdout_residual_rms": frobenius_residual_rms(
            response[holdout], covariance[holdout], a, b
        ),
        "n_calibration": int(len(calibration)),
        "n_holdout": int(len(holdout)),
        "calibration_index": calibration.astype(np.int64),
        "holdout_index": holdout.astype(np.int64),
        "keep": keep,
        "n_kept": int(np.count_nonzero(keep)),
        "n_dropped": int(np.count_nonzero(~keep)),
    }


def fig_data_uri(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scatter_html(
    covariance: np.ndarray,
    response: np.ndarray,
    fit: dict,
    *,
    max_final_variance: float,
) -> str:
    keep = np.asarray(fit["keep"], dtype=bool)
    dropped = ~keep
    pairs = (
        (r"$C_{11}$", r"$R_{11}$", covariance[:, 0, 0], response[:, 0, 0], True),
        (r"$C_{22}$", r"$R_{22}$", covariance[:, 1, 1], response[:, 1, 1], True),
        (r"$C_{12}$", r"$R_{12}$", covariance[:, 0, 1], response[:, 0, 1], False),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    for axis, (xlabel, ylabel, x_values, y_values, with_intercept) in zip(
        axes, pairs
    ):
        axis.scatter(
            x_values[keep],
            y_values[keep],
            s=12,
            alpha=0.75,
            label=f"OLS (n={int(np.count_nonzero(keep))})",
        )
        if np.any(dropped):
            axis.scatter(
                x_values[dropped],
                y_values[dropped],
                s=12,
                alpha=0.75,
                marker="x",
                label=f"not SPD (n={int(np.count_nonzero(dropped))})",
            )
        finite_x = x_values[np.isfinite(x_values)]
        if len(finite_x):
            grid = np.linspace(float(np.min(finite_x)), float(np.max(finite_x)), 200)
            overlay = (
                fit["a"] + fit["b"] * grid if with_intercept else fit["b"] * grid
            )
            axis.plot(grid, overlay, color="black", lw=1.2, label=r"$aI+bC$")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Proposal Mean matrix response vs posterior covariance")
    fig.tight_layout()
    uri = fig_data_uri(fig)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>R(C) fit (all SPD points)</title>"
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;"
        "padding:.35rem .6rem;text-align:right}th:first-child,td:first-child"
        "{text-align:left}</style></head><body>"
        "<h1>Matrix shear response vs posterior covariance</h1>"
        "<p>Proposal Mean five-point stencil. "
        "R = aI + bC is Frobenius ordinary least squares on every base with "
        "a finite positive-definite measured Jacobian. "
        "There is no σ<sub>final</sub> cut at fit time. "
        "Report-time apply uses unclipped R = aI + bC and drops galaxies "
        "whose predicted R is not invertible or whose σ<sub>final</sub> is at least "
        f"{max_final_variance:g}. "
        "Crosses are non-SPD measured R, not an analysis cut.</p>"
        "<table><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>a</td><td>{fit['a']:.8g}</td></tr>"
        f"<tr><td>b</td><td>{fit['b']:.8g}</td></tr>"
        f"<tr><td>calibration R²</td><td>{fit['r_squared']:.6g}</td></tr>"
        f"<tr><td>holdout Frobenius RMS</td><td>{fit['holdout_residual_rms']:.6g}</td></tr>"
        f"<tr><td>n OLS</td><td>{fit['n_kept']}</td></tr>"
        f"<tr><td>n not SPD</td><td>{fit['n_dropped']}</td></tr>"
        f"<tr><td>seed</td><td>{fit['seed']}</td></tr>"
        "</tbody></table>"
        f'<p><img src="{uri}" alt="R versus C scatter"></p>'
        "</body></html>"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.r_min <= 0.0 or args.r_min > args.r_max:
        raise ValueError("R clip bounds must satisfy 0 < R_min <= R_max")
    if args.max_final_variance <= 0.0:
        raise ValueError("max-final-variance must be positive")
    html_path = args.html if args.html is not None else args.output.with_suffix(".html")
    manifest = pd.read_csv(args.manifest)
    cache_partitions = load_cache_partitions(args.cache_root)
    if cache_partitions.observation_provenance["matched_group_size"] != len(
        STATE_ORDER
    ):
        raise ValueError(
            "R(C) fit requires cache noise shared within five-state groups"
        )
    truth = np.asarray(load_partitioned_array(cache_partitions, "truth"))
    rmag_true = np.asarray(
        load_partitioned_array(cache_partitions, "rmag_true"), dtype=np.float64
    )
    summary = np.asarray(
        load_partitioned_array(cache_partitions, "proposal_mean_estimates")
    )
    if summary.ndim != 3 or summary.shape[1] != 3 or summary.shape[2] < 2:
        raise ValueError("proposal Mean summaries must have shape (galaxy, 3, feature)")
    mean_cube, truth_cube, _, base_ids = build_matched_cubes(
        manifest, truth, summary[:, 1], rmag_true
    )
    nbase = len(base_ids)
    ordered = manifest.sort_values("ID").reset_index(drop=True)
    base_column = pd.to_numeric(ordered["base_id"], errors="raise").to_numpy(
        dtype=np.int64
    )
    state_code = {name: index for index, name in enumerate(STATE_ORDER)}
    unknown = sorted(set(ordered["state"]) - set(state_code))
    if unknown:
        raise ValueError(f"Unknown matched states: {unknown}")
    state_column = ordered["state"].map(state_code).to_numpy(dtype=np.int64)
    covariance = load_zero_shear_covariances(
        cache_partitions, base_column, state_column, nbase
    )
    response, delta = finite_difference_response(mean_cube, truth_cube)
    keep = measured_quality_mask(response)
    fit = fit_with_holdout(
        response,
        covariance,
        keep,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    payload = {
        "model": MATRIX_MODEL,
        "a": fit["a"],
        "b": fit["b"],
        "R_min": float(args.r_min),
        "R_max": float(args.r_max),
        "r_squared": fit["r_squared"],
        "holdout_residual_rms": fit["holdout_residual_rms"],
        "stencil_path": str(args.cache_root),
        "manifest_path": str(args.manifest),
        "seed": int(args.seed),
        "nbase": int(nbase),
        "n_calibration": fit["n_calibration"],
        "n_holdout": fit["n_holdout"],
        "n_kept": fit["n_kept"],
        "n_dropped": fit["n_dropped"],
        "fit_keep": "spd_measured_R",
        "max_final_variance": float(args.max_final_variance),
        "delta_g": delta,
        "posterior_source": "proposal",
        "estimator": "mean",
        "mean_measured_sigma_final_kept": float(
            np.nanmean(calibrated_rms(response[keep], covariance[keep]))
        )
        if np.any(keep)
        else float("nan"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    document = scatter_html(
        covariance,
        response,
        {**fit, "seed": args.seed},
        max_final_variance=float(args.max_final_variance),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {html_path}")
    print(
        f"R = {payload['a']:.6g} I + {payload['b']:.6g} C, "
        f"R²={payload['r_squared']:.4f}, "
        f"kept={payload['n_kept']}/{nbase}, "
        f"holdout RMS={payload['holdout_residual_rms']:.4f}"
    )


if __name__ == "__main__":
    main()
