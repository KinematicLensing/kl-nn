#!/usr/bin/env python3
"""Fit a scalar shear response R(σ) on a matched five-point stencil cache.

Writes a small JSON (the production input for report-time stretch) and a
companion HTML scatter. This diagnostic is not METHODS-monitored.
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
    "r_sigma_frozen_concat_45255702.json"
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
    return parser.parse_args(argv)


def cube_rows(
    values: np.ndarray,
    base_column: np.ndarray,
    state_column: np.ndarray,
    nbase: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    cube = np.empty((nbase, len(STATE_ORDER), values.shape[1]), dtype=np.float64)
    cube[base_column, state_column] = values
    return cube


def zero_shear_sigma(p16: np.ndarray, p84: np.ndarray) -> np.ndarray:
    """Combined 16–84 half-width of (g1, g2) at the zero-shear state."""

    half = 0.5 * (np.asarray(p84, dtype=np.float64) - np.asarray(p16, dtype=np.float64))
    if half.ndim != 2 or half.shape[1] != 2:
        raise ValueError("zero-shear 16/84 must have shape (nbase, 2)")
    if not np.all(np.isfinite(half)):
        raise ValueError("zero-shear 16–84 shear width must be finite")
    if np.any(half < 0.0):
        raise ValueError("zero-shear 84th percentile must be at least the 16th")
    return np.sqrt(0.5 * np.sum(np.square(half), axis=1))


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


def fit_linear_response(
    sigma: np.ndarray,
    r_diag: np.ndarray,
    *,
    seed: int,
    calibration_fraction: float,
) -> dict:
    sigma = np.asarray(sigma, dtype=np.float64)
    r_diag = np.asarray(r_diag, dtype=np.float64)
    nbase = len(sigma)
    if nbase < 4 or len(r_diag) != nbase:
        raise ValueError("calibration requires matching sigma and R of length >= 4")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(nbase)
    split = int(round(calibration_fraction * nbase))
    split = min(max(split, 2), nbase - 2)
    calibration, holdout = order[:split], order[split:]
    sigma2 = np.square(sigma)
    design = np.column_stack(
        (np.ones(len(calibration), dtype=np.float64), sigma2[calibration])
    )
    coef, _, _, _ = np.linalg.lstsq(design, r_diag[calibration], rcond=None)
    intercept, slope = (float(coef[0]), float(coef[1]))
    predicted = intercept + slope * sigma2
    residual = r_diag - predicted
    cal_residual = residual[calibration]
    ss_res = float(np.sum(np.square(cal_residual)))
    centered = r_diag[calibration] - np.mean(r_diag[calibration])
    ss_tot = float(np.sum(np.square(centered)))
    r_squared = float("nan") if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    holdout_residual_rms = float(np.sqrt(np.mean(np.square(residual[holdout]))))
    correlation = float(np.corrcoef(r_diag, sigma2)[0, 1])
    return {
        "a": intercept,
        "b": slope,
        "r_squared": r_squared,
        "holdout_residual_rms": holdout_residual_rms,
        "n_calibration": int(len(calibration)),
        "n_holdout": int(len(holdout)),
        "corr_R_sigma2": correlation,
        "calibration_index": calibration.astype(np.int64),
        "holdout_index": holdout.astype(np.int64),
        "predicted": predicted,
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
) -> str:
    calibration = fit["calibration_index"]
    holdout = fit["holdout_index"]
    grid = np.linspace(float(np.min(sigma)), float(np.max(sigma)), 200)
    unclipped = fit["a"] + fit["b"] * np.square(grid)
    clipped = np.clip(unclipped, r_min, r_max)
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    axis.scatter(
        sigma[calibration],
        r_diag[calibration],
        s=12,
        alpha=0.75,
        label=f"calibration (n={len(calibration)})",
    )
    axis.scatter(
        sigma[holdout],
        r_diag[holdout],
        s=12,
        alpha=0.75,
        marker="s",
        label=f"holdout (n={len(holdout)})",
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
    axis.set_xlabel(r"zero-shear $\sigma$")
    axis.set_ylabel(r"$R_{\mathrm{diag}}=(R_{11}+R_{22})/2$")
    axis.set_title("Proposal Mean response vs posterior width")
    axis.legend()
    axis.grid(True, alpha=0.3)
    uri = fig_data_uri(fig)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>R(σ) fit</title>"
        "<style>body{font-family:system-ui;max-width:800px;margin:2rem auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #ccc;"
        "padding:.35rem .6rem;text-align:right}th:first-child,td:first-child"
        "{text-align:left}</style></head><body>"
        "<h1>Scalar shear response vs posterior width</h1>"
        "<p>Proposal Mean five-point stencil. "
        "R = a + b σ<sup>2</sup> is ordinary least squares on the calibration "
        f"half; applied response is clipped to [{r_min:g}, {r_max:g}]. "
        "Wide-tertile floor R<sub>min</sub> never amplifies by more than "
        "&times;4 and never shrinks.</p>"
        "<table><thead><tr><th>Quantity</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>a</td><td>{fit['a']:.8g}</td></tr>"
        f"<tr><td>b</td><td>{fit['b']:.8g}</td></tr>"
        f"<tr><td>calibration R²</td><td>{fit['r_squared']:.6g}</td></tr>"
        f"<tr><td>holdout residual RMS</td><td>{fit['holdout_residual_rms']:.6g}</td></tr>"
        f"<tr><td>corr(R, σ²)</td><td>{fit['corr_R_sigma2']:.6g}</td></tr>"
        f"<tr><td>seed</td><td>{fit['seed']}</td></tr>"
        "</tbody></table>"
        f'<p><img src="{uri}" alt="R versus sigma scatter"></p>'
        "</body></html>"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.r_min <= 0.0 or args.r_min > args.r_max:
        raise ValueError("R clip bounds must satisfy 0 < R_min <= R_max")
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
    p16_cube = cube_rows(summary[:, 0, :2], base_column, state_column, nbase)
    p84_cube = cube_rows(summary[:, 2, :2], base_column, state_column, nbase)
    zero = STATE_ORDER.index("zero")
    sigma = zero_shear_sigma(p16_cube[:, zero], p84_cube[:, zero])
    r_diag, delta = diagonal_response(mean_cube, truth_cube)
    fit = fit_linear_response(
        sigma,
        r_diag,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
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
        "delta_g": delta,
        "posterior_source": "proposal",
        "estimator": "mean",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    document = scatter_html(
        sigma,
        r_diag,
        {**fit, "seed": args.seed},
        r_min=float(args.r_min),
        r_max=float(args.r_max),
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(document, encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {html_path}")
    print(
        f"R = {payload['a']:.6g} + {payload['b']:.6g} σ², "
        f"R²={payload['r_squared']:.4f}, "
        f"holdout RMS={payload['holdout_residual_rms']:.4f}"
    )


if __name__ == "__main__":
    main()
