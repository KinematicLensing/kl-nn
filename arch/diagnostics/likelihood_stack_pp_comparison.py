#!/usr/bin/env python3
"""Compare marginal shear P-P curves for the three xu3 estimators.

The report uses one common prefix of the test-set cache and deliberately keeps
inference separate from aggregation.  The 9D NPE uses the cached TF-weighted
shear samples, the 2D NRE uses its saved shear-grid log-ratios, and the 9D
NRE uses weighted Nautilus posterior samples.
"""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import sys

import numpy as np

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from cache_contract import load_cache_partitions
from diagnostics.likelihood_stack_core import (
    ABS_G_SPLIT,
    CACHE_ROOT,
    G01_CASE,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    NRE9D_NAME,
    REPORT_ROOT,
    json_safe,
    write_json,
)
from diagnostics.likelihood_stack_nre_9d_nautilus import (
    artifact_path,
    chain_run_dir,
    load_artifact,
    normalized_weights,
)
from diagnostics.likelihood_stack_nre_9d_nautilus_report import (
    physical_posterior,
)
from diagnostics.likelihood_stack_select import split_case

FIGURE_DPI = 150
DEFAULT_N_ROWS = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_REPORT_DIR = REPORT_ROOT / "05_pp_comparison"
DEFAULT_NRE2D_CACHE = DEFAULT_REPORT_DIR / "cache" / "nre2d_xu3_n1000_s42.npz"
DEFAULT_NRE9D_RUN = (
    CACHE_ROOT
    / NRE9D_NAME
    / "nautilus_factorized_xu3_n1000_s42"
)
DEFAULT_NRE9D_REPORT = DEFAULT_REPORT_DIR / "nre9d"
ESTIMATOR_ORDER = ("npe_9d", "nre_2d", "nre_9d")
ESTIMATOR_LABELS = {
    "npe_9d": "9D NPE (TF-weighted)",
    "nre_2d": "2D NRE",
    "nre_9d": "9D NRE (TF-weighted)",
}
ESTIMATOR_COLORS = {
    "npe_9d": "#1f4e79",
    "nre_2d": "#b85c38",
    "nre_9d": "#2e7d32",
}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--npe-case", default=G01_CASE)
    parser.add_argument("--nre2d-cache", type=Path, default=DEFAULT_NRE2D_CACHE)
    parser.add_argument("--nre9d-run", type=Path, default=DEFAULT_NRE9D_RUN)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--n-rows", type=int, default=DEFAULT_N_ROWS)
    parser.add_argument("--abs-g-split", type=float, default=ABS_G_SPLIT)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _prefix_array(partitions, name: str, n_rows: int) -> np.ndarray:
    """Read only the requested prefix without concatenating the full cache."""

    if name not in partitions.files:
        raise ValueError(f"cache does not contain required array {name!r}")
    if n_rows < 1 or n_rows > partitions.total_rows:
        raise ValueError(
            f"n_rows={n_rows} is outside the cache range 1..{partitions.total_rows}"
        )
    chunks = []
    offset = 0
    for path in partitions.files[name]:
        if offset >= n_rows:
            break
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        take = min(values.shape[0], n_rows - offset)
        chunks.append(np.asarray(values[:take]))
        offset += take
    if offset != n_rows:
        raise ValueError(f"{name} contains only {offset} rows; expected {n_rows}")
    return np.concatenate(chunks, axis=0)


def weighted_sample_pit(
    samples: np.ndarray,
    log_weights: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Compute row-wise weighted PITs with midpoint handling for ties."""

    samples = np.asarray(samples, dtype=np.float64)
    log_weights = np.asarray(log_weights, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape (rows, draws, 2)")
    if log_weights.shape != samples.shape[:2]:
        raise ValueError("log_weights must match the first two sample axes")
    if truth.shape != (samples.shape[0], 2):
        raise ValueError("truth must have shape (rows, 2)")
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("sample log weights must be finite")
    shifted = log_weights - np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    below = samples < truth[:, None, :]
    equal = samples == truth[:, None, :]
    pits = np.sum(weights[:, :, None] * below, axis=1)
    pits += 0.5 * np.sum(weights[:, :, None] * equal, axis=1)
    return np.clip(pits, 0.0, 1.0)


def weighted_sample_ess(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    shifted = log_weights - np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return 1.0 / np.sum(weights**2, axis=1)


def _trapezoid_weights(axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("grid axes must be strictly increasing with at least two points")
    weights = np.empty(axis.size, dtype=np.float64)
    weights[0] = 0.5 * (axis[1] - axis[0])
    weights[-1] = 0.5 * (axis[-1] - axis[-2])
    weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
    return weights


def grid_marginal_pit(
    log_r: np.ndarray,
    grid_phys: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Integrate a 2D grid posterior and evaluate both marginal CDFs."""

    log_r = np.asarray(log_r, dtype=np.float64)
    grid_phys = np.asarray(grid_phys, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if log_r.ndim != 2:
        raise ValueError("log_r must have shape (rows, grid_points)")
    if grid_phys.ndim != 2 or grid_phys.shape[1] != 2:
        raise ValueError("grid_phys must have shape (grid_points, 2)")
    if log_r.shape[1] != len(grid_phys):
        raise ValueError("grid and log_r point counts do not match")
    if truth.shape != (len(log_r), 2):
        raise ValueError("truth must match the number of grid rows")
    axis_g1 = np.unique(grid_phys[:, 0])
    axis_g2 = np.unique(grid_phys[:, 1])
    if len(axis_g1) * len(axis_g2) != len(grid_phys):
        raise ValueError("grid_phys is not a rectangular grid")
    expected = np.stack(
        np.meshgrid(axis_g1, axis_g2, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    if not np.allclose(expected, grid_phys, rtol=0.0, atol=1e-12):
        raise ValueError("grid_phys ordering is not meshgrid(indexing='ij')")

    values = np.exp(log_r - np.max(log_r, axis=1, keepdims=True))
    values = values.reshape(len(log_r), len(axis_g1), len(axis_g2))
    weights_g1 = _trapezoid_weights(axis_g1)
    weights_g2 = _trapezoid_weights(axis_g2)
    marginal_g1 = np.sum(values * weights_g2[None, None, :], axis=2)
    marginal_g2 = np.sum(values * weights_g1[None, :, None], axis=1)

    cdf_g1 = np.concatenate(
        (
            np.zeros((len(values), 1)),
            np.cumsum(
                0.5 * (marginal_g1[:, :-1] + marginal_g1[:, 1:])
                * np.diff(axis_g1)[None, :],
                axis=1,
            ),
        ),
        axis=1,
    )
    cdf_g2 = np.concatenate(
        (
            np.zeros((len(values), 1)),
            np.cumsum(
                0.5 * (marginal_g2[:, :-1] + marginal_g2[:, 1:])
                * np.diff(axis_g2)[None, :],
                axis=1,
            ),
        ),
        axis=1,
    )
    cdf_g1 /= cdf_g1[:, -1, None]
    cdf_g2 /= cdf_g2[:, -1, None]
    return np.column_stack(
        (
            np.array(
                [
                    np.interp(truth[row, 0], axis_g1, cdf_g1[row])
                    for row in range(len(values))
                ]
            ),
            np.array(
                [
                    np.interp(truth[row, 1], axis_g2, cdf_g2[row])
                    for row in range(len(values))
                ]
            ),
        )
    )


def load_npe_pits(
    cache_root: Path,
    case: str,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model, dataset = split_case(case)
    root = Path(cache_root) / model / dataset
    partitions = load_cache_partitions(root)
    if partitions.analysis_mode != "test_set":
        raise ValueError("the comparison requires the test-set NPE cache")
    names = tuple(partitions.feature_names)
    if "g1" not in names or "g2" not in names:
        raise ValueError(f"cache feature names omit g1/g2: {names}")
    g1, g2 = names.index("g1"), names.index("g2")
    truth = _prefix_array(partitions, "truth", n_rows)[:, (g1, g2)]
    samples = _prefix_array(partitions, "shear_sample", n_rows)
    log_weights = _prefix_array(partitions, "posterior_tf_log_weight", n_rows)
    if samples.shape[-1] != 2 or samples.shape[:2] != log_weights.shape:
        raise ValueError("NPE shear samples and TF weights have incompatible shapes")
    pits = weighted_sample_pit(samples, log_weights, truth)
    ess = weighted_sample_ess(log_weights)
    metadata = {
        "cache_root": str(root),
        "cache_schema": partitions.manifests[0].get("schema"),
        "analysis_mode": partitions.analysis_mode,
        "feature_names": list(names),
        "posterior": "test-set TF-weighted shear_sample",
        "draws_per_galaxy": int(samples.shape[1]),
        "effective_sample_size_mean": float(np.mean(ess)),
        "effective_sample_size_median": float(np.median(ess)),
    }
    return truth, pits, metadata


def load_nre2d_pits(
    path: Path,
    truth: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"indices", "truth_g", "grid_phys", "log_r"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing fields {sorted(missing)}")
        indices = np.asarray(data["indices"], dtype=np.int64)
        cached_truth = np.asarray(data["truth_g"], dtype=np.float64)
        grid_phys = np.asarray(data["grid_phys"], dtype=np.float64)
        log_r = np.asarray(data["log_r"], dtype=np.float64)
    expected_indices = np.arange(n_rows, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("2D NRE cache is not the common first-row selection")
    if cached_truth.shape != truth.shape or not np.allclose(
        cached_truth, truth, rtol=0.0, atol=1e-6
    ):
        raise ValueError("2D NRE truth rows do not match the NPE cache")
    pits = grid_marginal_pit(log_r, grid_phys, truth)
    return pits, {
        "cache_path": str(path),
        "posterior": "21x21 shear-grid likelihood ratio with uniform shear prior",
        "grid_points": int(len(grid_phys)),
        "grid_shape": [
            int(len(np.unique(grid_phys[:, 0]))),
            int(len(np.unique(grid_phys[:, 1]))),
        ],
    }


def load_nre9d_pits(
    run_dir: Path,
    truth: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    selection_path = run_dir / "selection.npz"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    with np.load(selection_path, allow_pickle=False) as data:
        indices = np.asarray(data["index"], dtype=np.int64)
    expected_indices = np.arange(n_rows, dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise ValueError("9D NRE selection is not the common first-row selection")

    pits = np.empty((n_rows, 2), dtype=np.float64)
    ess = np.empty(n_rows, dtype=np.float64)
    names = tuple(config.TARGET_NAMES)
    for row, index in enumerate(indices):
        path = artifact_path(run_dir, int(index))
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact = load_artifact(path)
        artifact_truth = np.asarray(artifact["truth"], dtype=np.float64)[:2]
        if not np.allclose(artifact_truth, truth[row], rtol=0.0, atol=1e-6):
            raise ValueError(f"9D NRE truth does not match row {row}")
        physical, weights = physical_posterior(artifact, names=names)
        below = physical[:, :2] < truth[row]
        equal = physical[:, :2] == truth[row]
        pits[row] = np.sum(weights[:, None] * below, axis=0)
        pits[row] += 0.5 * np.sum(weights[:, None] * equal, axis=0)
        pits[row] = np.clip(pits[row], 0.0, 1.0)
        ess[row] = 1.0 / np.sum(weights**2)
    return pits, ess, {
        "run_dir": str(run_dir),
        "posterior": "factorized 9D NRE weighted Nautilus samples with TF prior",
        "effective_sample_size_mean": float(np.mean(ess)),
        "effective_sample_size_median": float(np.median(ess)),
        "n_artifacts": int(n_rows),
    }


def pp_curve(values: np.ndarray, alpha: float = DEFAULT_ALPHA) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PIT values must be finite and lie in [0, 1]")
    values = np.sort(values)
    n = len(values)
    empirical = np.arange(1, n + 1, dtype=np.float64) / n
    before = np.arange(n, dtype=np.float64) / n
    ks = float(max(np.max(empirical - values), np.max(values - before)))
    epsilon = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))
    return {
        "pit_sorted": values,
        "empirical_cdf": empirical,
        "ks_distance": ks,
        "dkw_epsilon": epsilon,
        "n": int(n),
        "central_coverage": float(np.mean((values >= 0.16) & (values <= 0.84))),
    }


def summarize_pits(
    pits: np.ndarray,
    truth: np.ndarray,
    alpha: float,
    abs_g_split: float,
) -> dict:
    pits = np.asarray(pits, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pits.shape != truth.shape:
        raise ValueError("PIT and truth arrays must have the same shape")
    amplitude = np.hypot(truth[:, 0], truth[:, 1])
    masks = {
        "full": np.isfinite(amplitude),
        "inner": np.isfinite(amplitude) & (amplitude < abs_g_split),
        "outer": np.isfinite(amplitude) & (amplitude > abs_g_split),
    }
    components = {}
    for component, name in enumerate(("g1", "g2")):
        components[name] = {
            slice_name: pp_curve(pits[mask, component], alpha=alpha)
            for slice_name, mask in masks.items()
            if np.count_nonzero(mask) > 0
        }
    return components


def save_figures(
    report_dir: Path,
    summaries: dict,
    alpha: float,
) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures = {}
    grid = np.linspace(0.0, 1.0, 501)
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.8), sharex=True, sharey=True)
    for axis, component in zip(axes, ("g1", "g2")):
        for estimator in ESTIMATOR_ORDER:
            diagnostic = summaries[estimator][component]["full"]
            axis.step(
                diagnostic["pit_sorted"],
                diagnostic["empirical_cdf"],
                where="post",
                color=ESTIMATOR_COLORS[estimator],
                label=(
                    f"{ESTIMATOR_LABELS[estimator]} "
                    f"(KS={diagnostic['ks_distance']:.3f})"
                ),
            )
        n = summaries[ESTIMATOR_ORDER[0]][component]["full"]["n"]
        epsilon = float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))
        axis.fill_between(
            grid,
            np.maximum(0.0, grid - epsilon),
            np.minimum(1.0, grid + epsilon),
            color="#777777",
            alpha=0.15,
            label=f"{100 * (1 - alpha):.0f}% DKW band",
        )
        axis.plot(grid, grid, "--", color="#333333", lw=0.9, label="uniform")
        axis.set_title(component)
        axis.set_xlabel("posterior quantile at truth")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("fraction of galaxies")
    figure.suptitle("Three-way marginal shear P–P comparison")
    figure.tight_layout()
    figure.savefig(report_dir / "posterior_pp_comparison.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["posterior_pp_comparison"] = "posterior_pp_comparison.png"

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    bins = np.linspace(0.0, 1.0, 21)
    for axis, component in zip(axes, ("g1", "g2")):
        for estimator in ESTIMATOR_ORDER:
            pits = np.asarray(
                summaries[estimator][component]["full"]["pit_sorted"],
                dtype=np.float64,
            )
            axis.hist(
                pits,
                bins=bins,
                histtype="step",
                lw=1.5,
                color=ESTIMATOR_COLORS[estimator],
                label=ESTIMATOR_LABELS[estimator],
            )
        axis.axhline(len(pits) / len(bins[:-1]), color="#333333", ls="--", lw=0.8)
        axis.set_xlabel(f"{component} posterior quantile at truth")
        axis.set_xlim(0.0, 1.0)
        axis.set_title(component)
    axes[0].set_ylabel("galaxies per bin")
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("PIT histograms")
    figure.tight_layout()
    figure.savefig(report_dir / "posterior_pit_histograms.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["posterior_pit_histograms"] = "posterior_pit_histograms.png"
    return figures


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_html(
    path: Path,
    payload: dict,
    figures: dict[str, str],
) -> None:
    rows = []
    for estimator in ESTIMATOR_ORDER:
        for component in ("g1", "g2"):
            diagnostic = payload["estimators"][estimator][component]["full"]
            rows.append(
                "<tr>"
                f"<td>{html.escape(ESTIMATOR_LABELS[estimator])}</td>"
                f"<td>{component}</td><td>{diagnostic['n']}</td>"
                f"<td>{_fmt(diagnostic['ks_distance'])}</td>"
                f"<td>{_fmt(diagnostic['central_coverage'])}</td>"
                "</tr>"
            )
    split_rows = []
    for estimator in ESTIMATOR_ORDER:
        for slice_name in ("inner", "outer"):
            values = []
            for component in ("g1", "g2"):
                values.append(
                    payload["estimators"][estimator][component][slice_name]
                )
            split_rows.append(
                "<tr>"
                f"<td>{html.escape(ESTIMATOR_LABELS[estimator])}</td>"
                f"<td>{slice_name}</td><td>{values[0]['n']}</td>"
                f"<td>{_fmt(values[0]['ks_distance'])}</td>"
                f"<td>{_fmt(values[1]['ks_distance'])}</td>"
                "</tr>"
            )
    figure_html = "".join(
        f'<figure><img src="{html.escape(filename)}" alt="{html.escape(key)}">'
        f"<figcaption>{html.escape(key.replace('_', ' ').title())}</figcaption></figure>"
        for key, filename in figures.items()
    )
    metadata = payload["metadata"]
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Three-way xu3 P-P comparison</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Three-way marginal shear P–P comparison</h1>
<p class="lead">The three estimators are evaluated on the identical first
{metadata['n_rows']:,} rows of the xu3 catalog. The expected calibration curve
is the diagonal.</p>
{figure_html}
<h2>Full-subset calibration</h2>
<table><thead><tr><th>Estimator</th><th>component</th><th>N</th>
<th>KS distance</th><th>central 68% coverage</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Optional |g| split</h2>
<p>The inner/outer split is diagnostic only: rows exactly on the cut are
excluded from both split panels.</p>
<table><thead><tr><th>Estimator</th><th>slice</th><th>N</th>
<th>g1 KS</th><th>g2 KS</th></tr></thead>
<tbody>{''.join(split_rows)}</tbody></table>
<h2>Cache and PIT conventions</h2>
<ul>
<li>9D NPE: TF-weighted <code>shear_sample</code> and
<code>posterior_tf_log_weight</code> from the existing test-set cache.</li>
<li>2D NRE: trapezoid-integrated marginal CDF of the saved 21×21 shear
log-ratio grid with its uniform shear prior.</li>
<li>9D NRE: TF-weighted Nautilus samples from the factorized 9D posterior.</li>
<li>All sample/grid ties use midpoint PIT handling. The 2D grid is treated as
a continuous piecewise-linear density for its marginal CDF.</li>
</ul>
<p class="note">The 2D shear-only NRE does not contain nuisance parameters, so
it cannot receive the 9D model's explicit Tully–Fisher nuisance-prior
replacement. This distinction is recorded rather than hidden in the comparison.</p>
<p class="note">Report metadata: <code>{html.escape(str(metadata))}</code></p>
</body></html>"""
    path.write_text(body, encoding="utf-8")


def write_report(args) -> None:
    output = args.report_dir / "report.html"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")
    if args.n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    truth, npe_pits, npe_meta = load_npe_pits(
        args.cache_root, args.npe_case, args.n_rows
    )
    nre2d_pits, nre2d_meta = load_nre2d_pits(
        args.nre2d_cache, truth, args.n_rows
    )
    nre9d_pits, nre9d_ess, nre9d_meta = load_nre9d_pits(
        args.nre9d_run, truth, args.n_rows
    )
    pit_values = {
        "npe_9d": npe_pits,
        "nre_2d": nre2d_pits,
        "nre_9d": nre9d_pits,
    }
    summaries = {
        estimator: summarize_pits(
            pits, truth, args.alpha, args.abs_g_split
        )
        for estimator, pits in pit_values.items()
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figures = save_figures(args.report_dir, summaries, args.alpha)
    metadata = {
        "n_rows": int(args.n_rows),
        "row_indices": [0, int(args.n_rows) - 1],
        "npe_case": args.npe_case,
        "abs_g_split": float(args.abs_g_split),
        "alpha": float(args.alpha),
        "npe": npe_meta,
        "nre2d": nre2d_meta,
        "nre9d": nre9d_meta,
    }
    payload = {
        "metadata": metadata,
        "estimators": summaries,
        "pit_values": pit_values,
        "effective_sample_size": {
            "npe_9d_mean": float(npe_meta["effective_sample_size_mean"]),
            "nre_9d_mean": float(np.mean(nre9d_ess)),
        },
        "figures": figures,
    }
    write_json(args.report_dir / "report.json", payload)
    write_html(output, json_safe(payload), figures)
    print(f"Wrote {output}")
    print(f"Wrote {args.report_dir / 'report.json'}")


def main(argv=None) -> None:
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
