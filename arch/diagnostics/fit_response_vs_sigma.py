#!/usr/bin/env python3
"""Fit a scalar shear response R(σ) on a matched five-point stencil cache.

Writes a small JSON (the production input for report-time stretch) and a
companion HTML scatter. This diagnostic is not METHODS-monitored.

σ is posterior RMS from equal-weight candidate variance at the zero-shear
state, not a 16–84 half-width. Production a, b are OLS on leftover bases
after an iterated σ/R keep cut.
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
from shear_bias_report import combined_posterior_rms, response_from_sigma
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
    "/ocean/projects/phy250048p/shared/reports/"
    "r_sigma_frozen_concat_45255702_variance.json"
)
DEFAULT_SEED = 31415
DEFAULT_R_MIN = 0.25
DEFAULT_R_MAX = 1.0
DEFAULT_MAX_FINAL_VARIANCE = 0.2
MAX_KEEP_ITERATIONS = 10


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
            "Drop bases whose inflated RMS σ_final = σ/R is at least this "
            "threshold, using the clipped R map. Default 0.2 (prior box)."
        ),
    )
    return parser.parse_args(argv)


def zero_shear_sigma(draws: np.ndarray) -> np.ndarray:
    """Combined posterior RMS of (g1, g2) at the zero-shear state.

    `draws` has shape (nbase, ndraw, ≥2). Weights are equal over candidates.
    """

    sigma = combined_posterior_rms(draws, log_weight=None)
    if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError("zero-shear posterior RMS must be finite and non-negative")
    return sigma


def load_zero_shear_candidate_draws(
    cache_partitions,
    base_column: np.ndarray,
    state_column: np.ndarray,
    nbase: int,
) -> np.ndarray:
    """Equal-weight g1/g2 candidate draws at the zero-shear matched state."""

    zero = STATE_ORDER.index("zero")
    sample_files = cache_partitions.files["sample"]
    first = np.load(sample_files[0], mmap_mode="r")
    if first.ndim != 3 or first.shape[-1] < 2:
        raise ValueError(
            f"sample array must have shape (galaxy, draw, ≥2), got {first.shape}"
        )
    ndraw = int(first.shape[1])
    del first
    out = np.empty((nbase, ndraw, 2), dtype=np.float64)
    filled = np.zeros(nbase, dtype=bool)
    offset = 0
    for path in sample_files:
        samples = np.load(path, mmap_mode="r")
        nloc = int(samples.shape[0])
        global_index = np.arange(offset, offset + nloc)
        if offset + nloc > len(state_column):
            raise ValueError("sample cache is longer than the matched manifest")
        local = np.flatnonzero(state_column[global_index] == zero)
        if len(local):
            bases = base_column[global_index[local]]
            if np.any(bases < 0) or np.any(bases >= nbase):
                raise ValueError("zero-shear base_id is outside the matched cube")
            if np.any(filled[bases]):
                raise ValueError("duplicate zero-shear samples for a base galaxy")
            out[bases] = np.asarray(samples[local, :, :2], dtype=np.float64)
            filled[bases] = True
        offset += nloc
        del samples
    if offset != len(state_column):
        raise ValueError(
            f"sample cache has {offset} galaxies, manifest has {len(state_column)}"
        )
    if not np.all(filled):
        raise ValueError("every base galaxy must have a zero-shear sample row")
    return out


def diagonal_response(estimate_cube: np.ndarray, truth_cube: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-base Mean finite-difference R_diag = (R11 + R22) / 2."""

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
    r_diag = 0.5 * (response[:, 0, 0] + response[:, 1, 1])
    if not np.all(np.isfinite(r_diag)):
        raise ValueError("diagonal response must be finite")
    return r_diag, delta


def production_ols(sigma: np.ndarray, r_diag: np.ndarray) -> dict:
    """Ordinary least squares R = a + b σ² on every supplied point."""

    sigma = np.asarray(sigma, dtype=np.float64)
    r_diag = np.asarray(r_diag, dtype=np.float64)
    if len(sigma) < 2 or len(r_diag) != len(sigma):
        raise ValueError("OLS requires matching sigma and R of length >= 2")
    sigma2 = np.square(sigma)
    design = np.column_stack((np.ones(len(sigma), dtype=np.float64), sigma2))
    coef, _, _, _ = np.linalg.lstsq(design, r_diag, rcond=None)
    intercept, slope = (float(coef[0]), float(coef[1]))
    predicted = intercept + slope * sigma2
    residual = r_diag - predicted
    ss_res = float(np.sum(np.square(residual)))
    centered = r_diag - np.mean(r_diag)
    ss_tot = float(np.sum(np.square(centered)))
    r_squared = float("nan") if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    if len(sigma) >= 2 and float(np.std(sigma2)) > 0.0 and float(np.std(r_diag)) > 0.0:
        correlation = float(np.corrcoef(r_diag, sigma2)[0, 1])
    else:
        correlation = float("nan")
    return {
        "a": intercept,
        "b": slope,
        "r_squared": r_squared,
        "corr_R_sigma2": correlation,
    }


def holdout_residual_rms(
    sigma: np.ndarray,
    r_diag: np.ndarray,
    a: float,
    b: float,
    *,
    seed: int,
    calibration_fraction: float,
) -> dict:
    """Seeded holdout RMS of production a, b on leftover points (diagnostic)."""

    sigma = np.asarray(sigma, dtype=np.float64)
    r_diag = np.asarray(r_diag, dtype=np.float64)
    nbase = len(sigma)
    if nbase < 4 or len(r_diag) != nbase:
        raise ValueError("holdout residual requires matching leftover length >= 4")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(nbase)
    split = int(round(calibration_fraction * nbase))
    split = min(max(split, 2), nbase - 2)
    holdout = order[split:]
    predicted = a + b * np.square(sigma[holdout])
    residual_rms = float(np.sqrt(np.mean(np.square(r_diag[holdout] - predicted))))
    return {
        "holdout_residual_rms": residual_rms,
        "n_calibration": int(split),
        "n_holdout": int(len(holdout)),
    }


def keep_from_clipped_response(
    sigma: np.ndarray,
    response: np.ndarray,
    max_final_variance: float,
) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    ratio = sigma / response
    return (
        np.isfinite(sigma)
        & (sigma >= 0.0)
        & np.isfinite(response)
        & (response > 0.0)
        & np.isfinite(ratio)
        & (ratio < max_final_variance)
    )


def iterated_keep_fit(
    sigma: np.ndarray,
    r_diag: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float,
    r_min: float,
    r_max: float,
    max_final_variance: float,
    max_iterations: int = MAX_KEEP_ITERATIONS,
) -> dict:
    """Fit a+bσ² on leftover bases, iterating the clipped σ/R keep mask."""

    sigma = np.asarray(sigma, dtype=np.float64)
    r_diag = np.asarray(r_diag, dtype=np.float64)
    if len(sigma) < 4 or len(r_diag) != len(sigma):
        raise ValueError("calibration requires matching sigma and R of length >= 4")
    if not np.isfinite(max_final_variance) or max_final_variance <= 0.0:
        raise ValueError("max-final-variance must be a positive finite value")
    if max_iterations < 1:
        raise ValueError("keep iteration cap must be at least 1")
    keep = np.isfinite(sigma) & (sigma >= 0.0) & np.isfinite(r_diag)
    if not np.any(keep):
        raise ValueError("no finite zero-shear σ values to initialize the keep set")
    fit = None
    n_iterations = 0
    response = None
    for n_iterations in range(1, max_iterations + 1):
        if int(np.count_nonzero(keep)) < 4:
            raise ValueError("keep set is too small for OLS after the σ/R cut")
        fit = production_ols(sigma[keep], r_diag[keep])
        response = response_from_sigma(
            sigma, fit["a"], fit["b"], r_min, r_max
        )
        new_keep = keep_from_clipped_response(sigma, response, max_final_variance)
        new_keep &= np.isfinite(r_diag)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    holdout = holdout_residual_rms(
        sigma[keep],
        r_diag[keep],
        fit["a"],
        fit["b"],
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    n_kept = int(np.count_nonzero(keep))
    return {
        **fit,
        **holdout,
        "keep": keep,
        "response": response,
        "predicted": fit["a"] + fit["b"] * np.square(sigma),
        "n_kept": n_kept,
        "n_dropped": int(len(keep) - n_kept),
        "n_iterations": int(n_iterations),
        "seed": int(seed),
        "max_final_variance": float(max_final_variance),
    }


def fig_data_uri(figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scatter_html(
    sigma: np.ndarray,
    r_diag: np.ndarray,
    fit: dict,
    *,
    r_min: float,
    r_max: float,
    max_final_variance: float,
) -> str:
    keep = np.asarray(fit["keep"], dtype=bool)
    dropped = ~keep
    grid = np.linspace(float(np.min(sigma)), float(np.max(sigma)), 200)
    unclipped = fit["a"] + fit["b"] * np.square(grid)
    clipped = np.clip(unclipped, r_min, r_max)
    boundary = grid / max_final_variance
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.scatter(
        sigma[keep],
        r_diag[keep],
        s=12,
        alpha=0.75,
        label=rf"kept $\sigma/R < {max_final_variance:g}$ (n={int(np.count_nonzero(keep))})",
    )
    if np.any(dropped):
        axis.scatter(
            sigma[dropped],
            r_diag[dropped],
            s=18,
            alpha=0.8,
            marker="x",
            color="tab:orange",
            label=rf"dropped $\sigma/R \geq {max_final_variance:g}$ (n={int(np.count_nonzero(dropped))})",
        )
    axis.plot(grid, unclipped, color="black", lw=1.2, label=r"$a + b\sigma^2$")
    axis.plot(
        grid,
        clipped,
        color="tab:red",
        lw=1.2,
        ls="--",
        label=rf"clip$[{r_min:g}, {r_max:g}]$",
    )
    axis.plot(
        grid,
        boundary,
        color="tab:green",
        lw=1.2,
        ls=":",
        label=rf"$\sigma/R={max_final_variance:g}$",
    )
    axis.set_xlabel(r"zero-shear $\sigma=\sqrt{\frac{1}{2}(\mathrm{Var}\,g_1+\mathrm{Var}\,g_2)}$")
    axis.set_ylabel(r"$R_{\mathrm{diag}}=(R_{11}+R_{22})/2$")
    axis.set_title("Proposal Mean response vs posterior RMS")
    axis.legend()
    axis.grid(True, alpha=0.3)
    uri = fig_data_uri(fig)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>R(σ) variance fit</title>"
        "<style>body{font-family:system-ui;max-width:800px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;"
        "padding:.35rem .6rem;text-align:right}th:first-child,td:first-child"
        "{text-align:left}</style></head><body>"
        "<h1>Scalar shear response vs posterior RMS</h1>"
        "<p>Proposal Mean five-point stencil. "
        "X is posterior RMS "
        "σ = sqrt((Var g<sub>1</sub> + Var g<sub>2</sub>)/2) from equal-weight "
        "zero-shear candidates, not a 16–84 half-width. "
        "R = a + b σ<sup>2</sup> is ordinary least squares on leftover "
        f"points after iterating the keep rule σ/R &lt; {max_final_variance:g} "
        "with clipped R; applied response is clipped to "
        f"[{r_min:g}, {r_max:g}]. "
        "Wide-tertile floor R<sub>min</sub> never amplifies by more than "
        "&times;4 and never shrinks.</p>"
        "<table><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>a</td><td>{fit['a']:.8g}</td></tr>"
        f"<tr><td>b</td><td>{fit['b']:.8g}</td></tr>"
        f"<tr><td>leftover R²</td><td>{fit['r_squared']:.6g}</td></tr>"
        f"<tr><td>holdout residual RMS</td><td>{fit['holdout_residual_rms']:.6g}</td></tr>"
        f"<tr><td>corr(R, σ²)</td><td>{fit['corr_R_sigma2']:.6g}</td></tr>"
        f"<tr><td>n<sub>kept</sub></td><td>{fit['n_kept']}</td></tr>"
        f"<tr><td>n<sub>dropped</sub></td><td>{fit['n_dropped']}</td></tr>"
        f"<tr><td>iterations</td><td>{fit['n_iterations']}</td></tr>"
        f"<tr><td>seed</td><td>{fit['seed']}</td></tr>"
        "</tbody></table>"
        f'<p><img src="{uri}" alt="R versus sigma scatter"></p>'
        "</body></html>"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.r_min <= 0.0 or args.r_min > args.r_max:
        raise ValueError("R clip bounds must satisfy 0 < R_min <= R_max")
    if not np.isfinite(args.max_final_variance) or args.max_final_variance <= 0.0:
        raise ValueError("max-final-variance must be a positive finite value")
    html_path = args.html if args.html is not None else args.output.with_suffix(".html")
    manifest = pd.read_csv(args.manifest)
    cache_partitions = load_cache_partitions(args.cache_root)
    if cache_partitions.observation_provenance["matched_group_size"] != len(
        STATE_ORDER
    ):
        raise ValueError(
            "R(σ) fit requires cache noise shared within five-state groups"
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
    draws = load_zero_shear_candidate_draws(
        cache_partitions, base_column, state_column, nbase
    )
    sigma = zero_shear_sigma(draws)
    r_diag, delta = diagonal_response(mean_cube, truth_cube)
    fit = iterated_keep_fit(
        sigma,
        r_diag,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
        r_min=float(args.r_min),
        r_max=float(args.r_max),
        max_final_variance=float(args.max_final_variance),
    )
    payload = {
        "a": fit["a"],
        "b": fit["b"],
        "R_min": float(args.r_min),
        "R_max": float(args.r_max),
        "r_squared": fit["r_squared"],
        "holdout_residual_rms": fit["holdout_residual_rms"],
        "corr_R_sigma2": fit["corr_R_sigma2"],
        "stencil_path": str(args.cache_root),
        "manifest_path": str(args.manifest),
        "seed": int(args.seed),
        "nbase": int(nbase),
        "n_calibration": fit["n_calibration"],
        "n_holdout": fit["n_holdout"],
        "n_kept": fit["n_kept"],
        "n_dropped": fit["n_dropped"],
        "n_iterations": fit["n_iterations"],
        "max_final_variance": float(args.max_final_variance),
        "predictor": "posterior_variance",
        "delta_g": delta,
        "posterior_source": "proposal",
        "estimator": "mean",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    document = scatter_html(
        sigma,
        r_diag,
        fit,
        r_min=float(args.r_min),
        r_max=float(args.r_max),
        max_final_variance=float(args.max_final_variance),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {html_path}")
    print(
        f"R = {payload['a']:.6g} + {payload['b']:.6g} σ², "
        f"R²={payload['r_squared']:.4f}, "
        f"holdout RMS={payload['holdout_residual_rms']:.4f}, "
        f"kept {payload['n_kept']}/{payload['nbase']} "
        f"after {payload['n_iterations']} iteration(s)"
    )


if __name__ == "__main__":
    main()
