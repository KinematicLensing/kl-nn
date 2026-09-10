#!/usr/bin/env python3
"""Gaussian Fisher shear bound from a noiseless finite-difference FITS stencil.

Diagnostic only: not METHODS-monitored. Reads simulator-v3 FITS, uses the
training white-noise covariance, and writes JSON + HTML.
"""

from __future__ import annotations

import argparse
import base64
from html import escape
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


DIAG_DIR = Path(__file__).resolve().parent
ARCH_DIR = DIAG_DIR.parent
REPO_ROOT = ARCH_DIR.parent
for path in (str(DIAG_DIR), str(ARCH_DIR), str(REPO_ROOT / "data_generate")):
    if path not in sys.path:
        sys.path.insert(0, path)

from make_fisher_stencil_samples import FISHER_PARAMETERS  # noqa: E402

NSPEC = 5
IMAGE_HDU = NSPEC + 1
CENTER_FIBER = 2
SIGMA_PI = 0.1 / np.sqrt(3.0)
IMAGE_SNR_GRID = (10.0, 30.0, 100.0, 200.0, 500.0, 1000.0)
SPEC_SNR_GRID = (1.0, 5.0, 15.0, 30.0, 80.0, 150.0)
XU_IMAGE_SNR = 174.0
XU_SPEC_SNR = 26.0
CENTER_EXPOSURE = 180.0
OFFSET_EXPOSURE = 600.0
OFFSET_COUNTS_RATIO = float(np.sqrt(OFFSET_EXPOSURE / CENTER_EXPOSURE))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-fits", type=Path, required=True)
    parser.add_argument("--full-samples", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--g5-fits", type=Path, required=True)
    parser.add_argument("--g5-samples", type=Path, required=True)
    parser.add_argument("--g5-manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/ocean/projects/phy250048p/shared/reports/fisher_shear_bound.json"
        ),
    )
    parser.add_argument("--html", type=Path, default=None)
    parser.add_argument("--sini-bins", type=int, default=3)
    return parser.parse_args(argv)


def finite_difference(center, *, plus, minus, delta, side):
    center = np.asarray(center, dtype=np.float64)
    delta = float(delta)
    if delta <= 0.0 or not np.isfinite(delta):
        raise ValueError("delta must be a positive finite step")
    if side == "two_sided":
        return (np.asarray(plus, dtype=np.float64) - np.asarray(minus, dtype=np.float64)) / (
            2.0 * delta
        )
    if side == "plus_only":
        return (np.asarray(plus, dtype=np.float64) - center) / delta
    if side == "minus_only":
        return (center - np.asarray(minus, dtype=np.float64)) / delta
    raise ValueError(f"unknown finite-difference side {side!r}")


def fisher_matrix(dmu_dtheta: np.ndarray, inv_var: np.ndarray) -> np.ndarray:
    dmu = np.asarray(dmu_dtheta, dtype=np.float64)
    weight = np.sqrt(np.asarray(inv_var, dtype=np.float64))
    if dmu.ndim != 2 or weight.shape != (dmu.shape[1],):
        raise ValueError("dmu_dtheta must be (n_param, n_pix) with matching inv_var")
    if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
        raise ValueError("inv_var must be finite and non-negative")
    weighted = dmu * weight
    return weighted @ weighted.T


def scale_fisher(
    image_unit: np.ndarray, spec_unit: np.ndarray, *, image_snr: float, spec_snr: float
) -> np.ndarray:
    return (float(image_snr) ** 2) * np.asarray(image_unit) + (
        float(spec_snr) ** 2
    ) * np.asarray(spec_unit)


def profile_shear_block(info: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    info = np.asarray(info, dtype=np.float64)
    if info.shape != (9, 9):
        raise ValueError("Fisher matrix must be 9x9")
    shear = info[:2, :2]
    cross = info[:2, 2:]
    nuisance = info[2:, 2:]
    try:
        nuisance_inv = np.linalg.inv(nuisance)
    except np.linalg.LinAlgError:
        nuisance_inv = np.linalg.pinv(nuisance)
    profiled = shear - cross @ nuisance_inv @ cross.T
    try:
        covariance = np.linalg.inv(profiled)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(profiled)
    sigma_cr = float(np.sqrt(0.5 * (covariance[0, 0] + covariance[1, 1])))
    return profiled, covariance, sigma_cr


def unprofiled_sigma(info: np.ndarray) -> float:
    shear = np.asarray(info, dtype=np.float64)[:2, :2]
    try:
        covariance = np.linalg.inv(shear)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(shear)
    return float(np.sqrt(0.5 * (covariance[0, 0] + covariance[1, 1])))


def shrinkage_m_r(sigma_cr: float, sigma_pi: float = SIGMA_PI) -> tuple[float, float]:
    sigma_cr = float(sigma_cr)
    variance = sigma_cr * sigma_cr
    prior = float(sigma_pi) * float(sigma_pi)
    m = -variance / (variance + prior)
    return m, 1.0 + m


def continuum_subtracted_line_norm(spectrum: np.ndarray) -> float:
    values = np.asarray(spectrum, dtype=np.float64)
    valid = values != 0.0
    if int(np.count_nonzero(valid)) < 2:
        raise ValueError("spectrum must contain at least two non-zero pixels")
    continuum = float(np.median(values[valid]))
    residual = np.where(valid, values - continuum, 0.0)
    return float(np.linalg.norm(residual))


def image_inv_var_unit(image: np.ndarray) -> np.ndarray:
    flat = np.asarray(image, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(flat))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("clean image matched-filter norm must be positive")
    return np.full(flat.size, 1.0 / (norm * norm), dtype=np.float64)


def spectral_inv_var_unit(spectra: np.ndarray, *, center_fiber: int = CENTER_FIBER) -> np.ndarray:
    spectra = np.asarray(spectra, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[0] != NSPEC:
        raise ValueError(f"spectra must have shape ({NSPEC}, nwave)")
    central_norm = continuum_subtracted_line_norm(spectra[center_fiber])
    if not np.isfinite(central_norm) or central_norm <= 0.0:
        raise ValueError("central H-alpha line norm must be positive")
    inv = np.zeros(spectra.size, dtype=np.float64)
    fiber_sigma_unit = np.full(spectra.shape[0], 1.0 / central_norm, dtype=np.float64)
    offset = np.ones(spectra.shape[0], dtype=bool)
    offset[center_fiber] = False
    fiber_sigma_unit[offset] *= OFFSET_COUNTS_RATIO
    valid = spectra != 0.0
    for fiber in range(spectra.shape[0]):
        sigma = fiber_sigma_unit[fiber]
        sl = slice(fiber * spectra.shape[1], (fiber + 1) * spectra.shape[1])
        inv[sl] = np.where(valid[fiber], 1.0 / (sigma * sigma), 0.0)
    return inv


def load_datavector(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path) as hdul:
        image = np.asarray(hdul[IMAGE_HDU].data, dtype=np.float64)
        spectra = np.zeros((NSPEC, 64), dtype=np.float64)
        for fiber in range(NSPEC):
            spec = np.asarray(hdul[fiber + 1].data, dtype=np.float64).ravel()
            spectra[fiber, : spec.shape[0]] = spec
    return image, spectra


def index_fits(root: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"FITS root does not exist: {root}")
    for path in root.glob("part_*/gal_*.fits"):
        sample_id = int(path.stem.split("_", 1)[1])
        mapping[sample_id] = path
    return mapping


def flatten_datavector(image: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.asarray(image, dtype=np.float64).ravel(), np.asarray(spectra, dtype=np.float64).ravel()]
    )


def group_states(manifest: pd.DataFrame) -> dict[int, pd.DataFrame]:
    groups = {}
    for base_id, group in manifest.groupby("base_id"):
        groups[int(base_id)] = group.set_index("state", drop=False)
    return groups


def load_group_datavectors(
    group: pd.DataFrame,
    fits_index: dict[int, Path],
    *,
    required_center: str,
) -> dict[str, np.ndarray]:
    vectors = {}
    for _, row in group.iterrows():
        sample_id = int(row["ID"])
        if sample_id not in fits_index:
            raise FileNotFoundError(f"missing FITS for ID {sample_id}")
        image, spectra = load_datavector(fits_index[sample_id])
        vectors[str(row["state"])] = flatten_datavector(image, spectra)
        if str(row["state"]) == required_center:
            vectors["_image"] = image
            vectors["_spectra"] = spectra
    if required_center not in vectors:
        raise KeyError(f"group is missing {required_center} state")
    return vectors


def full_unit_fishers(
    samples: pd.DataFrame,
    manifest: pd.DataFrame,
    fits_index: dict[int, Path],
) -> pd.DataFrame:
    joined = manifest.merge(samples, on="ID", suffixes=("", "_sample"), validate="one_to_one")
    records = []
    n_pix_image = 48 * 48
    for base_id, group in joined.groupby("base_id"):
        states = group.set_index("state", drop=False)
        vectors = load_group_datavectors(states, fits_index, required_center="center")
        image = vectors["_image"]
        spectra = vectors["_spectra"]
        image_weight = image_inv_var_unit(image)
        spec_weight = spectral_inv_var_unit(spectra)
        dmu = np.zeros((len(FISHER_PARAMETERS), n_pix_image + spectra.size), dtype=np.float64)
        missing = []
        for index, name in enumerate(FISHER_PARAMETERS):
            plus = vectors.get(f"{name}_plus")
            minus = vectors.get(f"{name}_minus")
            if plus is None and minus is None:
                missing.append(name)
                continue
            if plus is not None and minus is not None:
                side = "two_sided"
                delta = float(states.loc[f"{name}_plus", "delta"])
            elif plus is not None:
                side = "plus_only"
                delta = float(states.loc[f"{name}_plus", "delta"])
            else:
                side = "minus_only"
                delta = float(states.loc[f"{name}_minus", "delta"])
            dmu[index] = finite_difference(
                vectors["center"], plus=plus, minus=minus, delta=delta, side=side
            )
        if missing:
            raise ValueError(f"base {base_id} missing derivatives for {missing}")
        image_info = fisher_matrix(dmu[:, :n_pix_image], image_weight)
        spec_info = fisher_matrix(dmu[:, n_pix_image:], spec_weight)
        center_row = states.loc["center"]
        records.append(
            {
                "base_id": int(base_id),
                "source_row": int(center_row["source_row"]),
                "sini": float(center_row["sini"]),
                "hlr": float(center_row["hlr"]),
                "vcirc": float(center_row["vcirc"]),
                "g1": float(center_row["g1"]),
                "g2": float(center_row["g2"]),
                "image_snr_drawn": float(center_row["image_snr"]),
                "spec_snr_drawn": float(center_row["central_halpha_snr"]),
                "image_unit": image_info,
                "spec_unit": spec_info,
            }
        )
    return records


def g5_unit_fishers(
    samples: pd.DataFrame,
    manifest: pd.DataFrame,
    fits_index: dict[int, Path],
) -> list[dict]:
    joined = manifest.merge(samples, on="ID", suffixes=("", "_sample"), validate="one_to_one")
    records = []
    n_pix_image = 48 * 48
    for base_id, group in joined.groupby("base_id"):
        states = group.set_index("state", drop=False)
        vectors = load_group_datavectors(states, fits_index, required_center="zero")
        # Map zero -> center-like keys for derivative helper.
        vectors["center"] = vectors["zero"]
        image = vectors["_image"]
        spectra = vectors["_spectra"]
        image_weight = image_inv_var_unit(image)
        spec_weight = spectral_inv_var_unit(spectra)
        dmu = np.zeros((2, n_pix_image + spectra.size), dtype=np.float64)
        for index, name in enumerate(("g1", "g2")):
            plus = vectors[f"{name}_plus"]
            minus = vectors[f"{name}_minus"]
            delta = float(states.loc[f"{name}_plus", "delta"])
            dmu[index] = finite_difference(
                vectors["center"], plus=plus, minus=minus, delta=delta, side="two_sided"
            )
        image_info = fisher_matrix(dmu[:, :n_pix_image], image_weight)
        spec_info = fisher_matrix(dmu[:, n_pix_image:], spec_weight)
        # Embed in 9x9 so scale_fisher / unprofiled_sigma still work on [:2,:2].
        image_unit = np.zeros((9, 9), dtype=np.float64)
        spec_unit = np.zeros((9, 9), dtype=np.float64)
        image_unit[:2, :2] = image_info
        spec_unit[:2, :2] = spec_info
        zero = states.loc["zero"]
        records.append(
            {
                "base_id": int(base_id),
                "source_row": int(zero["source_row"]),
                "sini": float(zero["sini"]),
                "hlr": float(zero["hlr"]),
                "vcirc": float(zero["vcirc"]),
                "g1": 0.0,
                "g2": 0.0,
                "image_snr_drawn": float(zero["image_snr"]),
                "spec_snr_drawn": float(zero["central_halpha_snr"]),
                "image_unit": image_unit,
                "spec_unit": spec_unit,
            }
        )
    return records


def evaluate_record(record: dict, image_snr: float, spec_snr: float, *, profile: bool) -> dict:
    info = scale_fisher(
        record["image_unit"], record["spec_unit"], image_snr=image_snr, spec_snr=spec_snr
    )
    image_only = scale_fisher(
        record["image_unit"], np.zeros_like(record["spec_unit"]), image_snr=image_snr, spec_snr=0.0
    )
    spec_only = scale_fisher(
        np.zeros_like(record["image_unit"]), record["spec_unit"], image_snr=0.0, spec_snr=spec_snr
    )
    if profile:
        _, _, sigma = profile_shear_block(info)
        _, _, sigma_img = profile_shear_block(image_only)
        _, _, sigma_spec = profile_shear_block(spec_only)
        unprof = unprofiled_sigma(info)
    else:
        sigma = unprofiled_sigma(info)
        sigma_img = unprofiled_sigma(image_only)
        sigma_spec = unprofiled_sigma(spec_only)
        unprof = sigma
    m, response = shrinkage_m_r(sigma)
    return {
        "sigma_cr": sigma,
        "sigma_cr_unprofiled": unprof,
        "sigma_cr_image": sigma_img,
        "sigma_cr_spec": sigma_spec,
        "m": m,
        "R": response,
        "profile_ratio": unprof / sigma if sigma > 0 else np.nan,
    }


def sini_bin_edges(sini: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(sini, quantiles)
    edges[0] = min(edges[0], 0.0)
    edges[-1] = max(edges[-1], 1.0)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def transfer_ratios(full_eval: list[dict], sini: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = sini_bin_edges(sini, n_bins)
    ratios = np.array([row["sigma_cr"] / row["sigma_cr_unprofiled"] for row in full_eval])
    digitized = np.digitize(sini, edges[1:-1], right=False)
    medians = np.array(
        [
            float(np.median(ratios[digitized == index]))
            if np.any(digitized == index)
            else float(np.median(ratios))
            for index in range(n_bins)
        ]
    )
    return edges, medians


def summarize(values: np.ndarray) -> dict:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "median": None, "p16": None, "p84": None}
    return {
        "n": int(finite.size),
        "median": float(np.median(finite)),
        "p16": float(np.quantile(finite, 0.16)),
        "p84": float(np.quantile(finite, 0.84)),
    }


def figure_to_data_uri(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def scatter_vs(x, y, *, xlabel, ylabel, title, hline=None, hlabel=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.scatter(x, y, s=18, alpha=0.75, color="#1f4e79")
    if hline is not None:
        ax.axhline(hline, color="#b85c38", ls="--", lw=1.2, label=hlabel)
        if hlabel:
            ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    return figure_to_data_uri(fig)


def line_vs_snr(grid, curves, *, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for label, values in curves:
        ax.plot(grid, values, marker="o", label=label)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    return figure_to_data_uri(fig)


def render_html(payload: dict, figures: dict[str, str]) -> str:
    head = payload["headline"]
    blocks = []
    for key, uri in figures.items():
        blocks.append(
            f'<figure><img src="{uri}" alt="{escape(key)}">'
            f"<figcaption>{escape(key)}</figcaption></figure>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Simulator Fisher shear bound</title>
<style>
body {{ font: 16px/1.55 "Iowan Old Style", Palatino, serif; color: #1b1f24; background: #fbfbf9; margin: 0; }}
header {{ background: #1f4e79; color: #f4f7fa; padding: 2rem 1.5rem; }}
main {{ max-width: 920px; margin: 0 auto; padding: 1.4rem 1.2rem 3.5rem; }}
.stat {{ display: inline-block; border: 1px solid #d9dee3; padding: .7rem .9rem; margin: .3rem .4rem .3rem 0; }}
.stat b {{ display: block; font-size: 1.35rem; color: #1f4e79; }}
table {{ border-collapse: collapse; width: 100%; font: 14px/1.4 ui-sans-serif, system-ui, sans-serif; }}
th, td {{ border-bottom: 1px solid #d9dee3; padding: .35rem .45rem; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
figure img {{ width: 100%; height: auto; }}
.callout {{ border-left: 4px solid #1f4e79; padding-left: 1rem; color: #5c6770; }}
.warn {{ border-left-color: #b85c38; }}
</style>
</head>
<body>
<header>
<h1>Simulator Fisher bound on Mean-estimator m</h1>
<p>Noiseless KL forward model, training white-noise covariance, posterior-Mean shrinkage under g ~ U[-0.1, 0.1].</p>
</header>
<main>
<p class="callout">This is an information bound <em>inside the simulator</em> for the <strong>posterior Mean</strong>, not a claim about an R-corrected estimator. A perfect network cannot beat this m as a Mean under this prior.</p>
<div class="stat"><span>Xu-like S/N (ρ_I={XU_IMAGE_SNR:g}, ρ_S={XU_SPEC_SNR:g})</span><b>m = {head["m"]:.3f}</b>median profiled, 250 full bases</div>
<div class="stat"><span>σ_CR</span><b>{head["sigma_cr"]:.4f}</b>16–84: {head["sigma_cr_p16"]:.4f} – {head["sigma_cr_p84"]:.4f}</div>
<div class="stat"><span>R_Fisher</span><b>{head["R"]:.3f}</b>xu3 stencil R_diag ≈ 0.56</div>
<h2>What was computed</h2>
<p>{payload["n_full"]} full 9-parameter two-sided galaxies plus {payload["n_g5"]} shear-only 5-points.
Nuisance profiling uses the 250. The 500 inherit a sini-binned σ_CR,profiled / σ_CR,unprofiled transfer from the 250; that transfer is an estimate, not a gold-standard Fisher.</p>
<p class="callout warn">Drawn-S/N rows use the catalog image/Hα S/N stored on each FITS, which for train_1m is the independent uniform proposal, not the xu3 population. The headline uses the xu3-like S/N pair instead.</p>
<h2>Headline vs existing NN numbers</h2>
<table>
<tr><th>Quantity</th><th>Value</th></tr>
<tr><td>Fisher m (250, xu3-like S/N, profiled)</td><td>{head["m"]:.4f}</td></tr>
<tr><td>xu3 uncalibrated Mean m</td><td>≈ −0.50</td></tr>
<tr><td>Finite-shear stencil R_diag (NN)</td><td>≈ 0.56 (m ≈ −0.44)</td></tr>
<tr><td>Bayes shrinkage from σ_shape=0.039</td><td>≈ −0.45</td></tr>
<tr><td>Empirical R(σ) = 1.06 − 231 σ²</td><td>at σ=0.039 → R≈0.71</td></tr>
</table>
<h2>Figures</h2>
{''.join(blocks)}
<h2>S/N grid medians (profiled, 250)</h2>
{payload["snr_table_html"]}
<h2>sin i transfer used on the 500</h2>
{payload["transfer_table_html"]}
</main>
</body>
</html>
"""


def _median_eval(records: list[dict], image_snr: float, spec_snr: float, *, profile: bool) -> dict:
    rows = [evaluate_record(record, image_snr, spec_snr, profile=profile) for record in records]
    sigma = np.array([row["sigma_cr"] for row in rows])
    m = np.array([row["m"] for row in rows])
    response = np.array([row["R"] for row in rows])
    stats = summarize(sigma)
    m_stats = summarize(m)
    r_stats = summarize(response)
    return {
        "sigma_cr": stats["median"],
        "sigma_cr_p16": stats["p16"],
        "sigma_cr_p84": stats["p84"],
        "m": m_stats["median"],
        "R": r_stats["median"],
        "rows": rows,
    }


def snr_table_html(full_records: list[dict]) -> str:
    header = "<tr><th>ρ_I \\ ρ_S</th>" + "".join(f"<th>{s:g}</th>" for s in SPEC_SNR_GRID) + "</tr>"
    rows = []
    for image_snr in IMAGE_SNR_GRID:
        cells = [f"<td>{image_snr:g}</td>"]
        for spec_snr in SPEC_SNR_GRID:
            med = _median_eval(full_records, image_snr, spec_snr, profile=True)
            cells.append(f"<td>{med['m']:.3f}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + header + "".join(rows) + "</table>"


def transfer_table_html(edges: np.ndarray, medians: np.ndarray) -> str:
    rows = []
    for i, median in enumerate(medians):
        rows.append(
            f"<tr><td>[{edges[i]:.3f}, {edges[i+1]:.3f})</td><td>{median:.3f}</td></tr>"
        )
    return (
        "<table><tr><th>sin i bin</th><th>median σ_CR,profiled / σ_CR,unprofiled</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    html_path = args.html if args.html is not None else args.output.with_suffix(".html")
    full_samples = pd.read_csv(args.full_samples, float_precision="round_trip")
    full_manifest = pd.read_csv(args.full_manifest, float_precision="round_trip")
    g5_samples = pd.read_csv(args.g5_samples, float_precision="round_trip")
    g5_manifest = pd.read_csv(args.g5_manifest, float_precision="round_trip")
    full_fits = index_fits(args.full_fits)
    g5_fits = index_fits(args.g5_fits)
    print(f"Indexed {len(full_fits)} full FITS and {len(g5_fits)} g5 FITS", flush=True)
    full_records = full_unit_fishers(full_samples, full_manifest, full_fits)
    g5_records = g5_unit_fishers(g5_samples, g5_manifest, g5_fits)
    headline = _median_eval(full_records, XU_IMAGE_SNR, XU_SPEC_SNR, profile=True)
    full_sini = np.array([record["sini"] for record in full_records])
    edges, medians = transfer_ratios(headline["rows"], full_sini, args.sini_bins)
    g5_eval = []
    g5_sini = np.array([record["sini"] for record in g5_records])
    g5_bins = np.digitize(g5_sini, edges[1:-1], right=False)
    for record, bin_index in zip(g5_records, g5_bins):
        raw = evaluate_record(record, XU_IMAGE_SNR, XU_SPEC_SNR, profile=False)
        scale = float(medians[int(bin_index)])
        sigma = raw["sigma_cr"] * scale
        m, response = shrinkage_m_r(sigma)
        g5_eval.append({**raw, "sigma_cr": sigma, "m": m, "R": response, "transfer": scale})

    full_m = np.array([row["m"] for row in headline["rows"]])
    full_sigma = np.array([row["sigma_cr"] for row in headline["rows"]])
    full_r = np.array([row["R"] for row in headline["rows"]])
    # Inverse-variance split: larger weight means that modality dominates σ_CR.
    spec_weight = []
    for row in headline["rows"]:
        i_img = 1.0 / max(row["sigma_cr_image"] ** 2, 1e-18)
        i_spec = 1.0 / max(row["sigma_cr_spec"] ** 2, 1e-18)
        spec_weight.append(i_spec / (i_img + i_spec))
    spec_weight = np.array(spec_weight)

    grid_image_m = [
        _median_eval(full_records, image_snr, XU_SPEC_SNR, profile=True)["m"]
        for image_snr in IMAGE_SNR_GRID
    ]
    grid_spec_m = [
        _median_eval(full_records, XU_IMAGE_SNR, spec_snr, profile=True)["m"]
        for spec_snr in SPEC_SNR_GRID
    ]

    figures = {
        "Profiled m vs sin i at xu3-like S/N (250 gold)": scatter_vs(
            full_sini,
            full_m,
            xlabel=r"$\sin i$",
            ylabel=r"$m_{\mathrm{shrink}}$",
            title="Fisher Mean-shrinkage m vs inclination",
            hline=-0.50,
            hlabel="xu3 Mean m ≈ −50%",
        ),
        "Profiled σ_CR vs sin i": scatter_vs(
            full_sini,
            full_sigma,
            xlabel=r"$\sin i$",
            ylabel=r"$\sigma_{\mathrm{CR}}$",
            title="Cramér–Rao shear width vs inclination",
            hline=0.039,
            hlabel="xu3 σ_shape ≈ 0.039",
        ),
        "Spectral information fraction vs sin i": scatter_vs(
            full_sini,
            spec_weight,
            xlabel=r"$\sin i$",
            ylabel="spectra / (image + spectra)",
            title="Which modality sets σ_CR (inverse-variance split)",
        ),
        "Median m vs image S/N (ρ_S = 26)": line_vs_snr(
            IMAGE_SNR_GRID,
            [("profiled 250", grid_image_m)],
            xlabel=r"$\rho_I$",
            ylabel=r"median $m$",
            title="Shrinkage m vs image S/N",
        ),
        "Median m vs Hα S/N (ρ_I = 174)": line_vs_snr(
            SPEC_SNR_GRID,
            [("profiled 250", grid_spec_m)],
            xlabel=r"$\rho_S$",
            ylabel=r"median $m$",
            title="Shrinkage m vs central Hα S/N",
        ),
        "500 transferred m vs sin i": scatter_vs(
            g5_sini,
            np.array([row["m"] for row in g5_eval]),
            xlabel=r"$\sin i$",
            ylabel=r"$m$ (transferred)",
            title="g5-only galaxies after sini-binned profiling transfer",
            hline=headline["m"],
            hlabel="250 median",
        ),
        "Profiled vs unprofiled σ_CR (250)": scatter_vs(
            np.array([row["sigma_cr_unprofiled"] for row in headline["rows"]]),
            full_sigma,
            xlabel=r"unprofiled $\sigma_{\mathrm{CR}}$",
            ylabel=r"profiled $\sigma_{\mathrm{CR}}$",
            title="Nuisance profiling inflates the shear CR width",
        ),
    }

    payload = {
        "n_full": len(full_records),
        "n_g5": len(g5_records),
        "n_full_fits": len(full_fits),
        "n_g5_fits": len(g5_fits),
        "sigma_pi": SIGMA_PI,
        "xu_image_snr": XU_IMAGE_SNR,
        "xu_spec_snr": XU_SPEC_SNR,
        "headline": {
            "m": headline["m"],
            "R": headline["R"],
            "sigma_cr": headline["sigma_cr"],
            "sigma_cr_p16": headline["sigma_cr_p16"],
            "sigma_cr_p84": headline["sigma_cr_p84"],
        },
        "g5_transferred": summarize(np.array([row["m"] for row in g5_eval])),
        "sini_edges": edges.tolist(),
        "transfer_medians": medians.tolist(),
        "snr_table_html": snr_table_html(full_records),
        "transfer_table_html": transfer_table_html(edges, medians),
    }
    serializable = {
        key: value
        for key, value in payload.items()
        if key not in {"snr_table_html", "transfer_table_html"}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(payload, figures), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {html_path}")
    print(
        f"xu3-like median σ_CR={headline['sigma_cr']:.4f} "
        f"m={headline['m']:.4f} R={headline['R']:.4f}"
    )


if __name__ == "__main__":
    main()
