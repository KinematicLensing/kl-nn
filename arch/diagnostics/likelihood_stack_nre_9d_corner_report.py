#!/usr/bin/env python3
"""Build corner plots and aggregate diagnostics from persisted 9D NRE chains."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
import sys
from pathlib import Path

import numpy as np

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from diagnostics.likelihood_stack_benchmark_core import fit_component_mc
from diagnostics.likelihood_stack_core import (
    CACHE_ROOT,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    NRE9D_DIR,
    NRE9D_NAME,
    SELECT_DIR,
    STACK_DIR,
    json_safe,
    write_json,
)
from model_registry import load_model_config
from utils import denormalize


DEFAULT_RUN_NAME = "emcee_w48_b2500_p6000_s42"
NUISANCE_PREFIXES = (8, 16, 32, 64, 128)
FIGURE_DPI = 140


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--galaxies", type=Path, default=SELECT_DIR / "galaxies.npz")
    parser.add_argument("--stack-draws", type=Path, default=STACK_DIR / "stack.npz")
    parser.add_argument("--n-featured", type=int, default=8)
    parser.add_argument("--corner-max-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_indices(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "index" not in data.files:
        raise ValueError(f"{path} does not contain an index array")
    indices = np.asarray(data["index"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{path} index must be non-empty and one-dimensional")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{path} index contains duplicates")
    return indices


def _load_artifact(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {
        "index",
        "chain",
        "log_prob",
        "truth_physical",
        "parameter_names",
        "summary_json",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return {
        "index": int(data["index"].item()),
        "chain": np.asarray(data["chain"], dtype=np.float64),
        "log_prob": np.asarray(data["log_prob"], dtype=np.float64),
        "truth": np.asarray(data["truth_physical"], dtype=np.float64),
        "parameter_names": tuple(str(name) for name in data["parameter_names"]),
        "summary": json.loads(str(data["summary_json"].item())),
    }


def _finite_tau(diagnostics: dict) -> np.ndarray:
    raw = diagnostics.get("autocorrelation_time", [])
    return np.asarray(
        [float(value) if value is not None else np.nan for value in raw],
        dtype=np.float64,
    )


def _posterior_record(
    artifact: dict,
    *,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    target_transforms,
    corner_dir: Path,
    corner_max_samples: int,
) -> dict:
    import corner

    chain = artifact["chain"]
    log_prob = artifact["log_prob"]
    physical = np.asarray(
        denormalize(
            chain,
            par_ranges,
            feature_names=names,
            target_transforms=target_transforms,
        ),
        dtype=np.float64,
    )
    finite = np.isfinite(log_prob) & np.all(np.isfinite(physical), axis=1)
    if not np.any(finite):
        raise ValueError(f"galaxy {artifact['index']} has no finite samples")
    samples = physical[finite]
    finite_log_prob = log_prob[finite]
    map_index = int(np.argmax(finite_log_prob))
    mean = np.mean(samples, axis=0)
    lower = np.quantile(samples, 0.16, axis=0)
    upper = np.quantile(samples, 0.84, axis=0)
    map_like = samples[map_index]

    diagnostics = artifact["summary"]["diagnostics"]
    tau = _finite_tau(diagnostics)
    nwalkers = int(diagnostics["nwalkers"])
    production = int(diagnostics["production"])
    effective_sample_size = np.divide(
        nwalkers * production,
        tau,
        out=np.full_like(tau, np.nan),
        where=np.isfinite(tau) & (tau > 0),
    )

    stride = max(1, int(np.ceil(len(samples) / corner_max_samples)))
    corner_samples = samples[::stride]
    figure = corner.corner(
        corner_samples,
        labels=list(names),
        truths=artifact["truth"].tolist(),
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3g",
        levels=(0.68, 0.95),
        plot_datapoints=False,
        fill_contours=True,
        color="#1f4e79",
    )
    figure.suptitle(f"9D NRE posterior — dataset index {artifact['index']}", y=1.02)
    figure_path = corner_dir / f"galaxy_{artifact['index']:06d}.png"
    figure.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)

    return {
        "index": int(artifact["index"]),
        "truth": artifact["truth"],
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "map_like": map_like,
        "residual": mean - artifact["truth"],
        "diagnostics": diagnostics,
        "tau": tau,
        "effective_sample_size": effective_sample_size,
        "max_tau": float(np.nanmax(tau)),
        "min_effective_sample_size": float(np.nanmin(effective_sample_size)),
        "acceptance_mean": float(diagnostics["acceptance_mean"]),
        "converged": bool(diagnostics["converged"]),
        "n_samples": int(len(samples)),
        "corner_path": str(figure_path.name),
    }


def _featured_indices(records: list[dict], n_featured: int) -> list[tuple[int, str]]:
    if n_featured <= 0:
        return []
    n_featured = min(int(n_featured), len(records))
    amplitude = np.asarray(
        [np.hypot(row["truth"][0], row["truth"][1]) for row in records]
    )
    order = np.argsort(amplitude)
    selected: list[int] = []
    reasons: dict[int, str] = {}
    for group in np.array_split(order, min(4, n_featured)):
        if len(group) == 0:
            continue
        target = float(np.median(amplitude[group]))
        candidate = min(group.tolist(), key=lambda index: abs(amplitude[index] - target))
        selected.append(candidate)
        reasons[candidate] = "true |g| stratification representative"

    remaining = [index for index in range(len(records)) if index not in selected]
    slots = n_featured - len(selected)
    diagnostic_order = sorted(
        remaining,
        key=lambda index: (
            -records[index]["max_tau"],
            records[index]["acceptance_mean"],
            records[index]["index"],
        ),
    )
    for index in diagnostic_order[: slots // 2 + slots % 2]:
        selected.append(index)
        reasons[index] = "highest autocorrelation diagnostic"
    remaining = [index for index in remaining if index not in selected]
    diagnostic_order = sorted(
        remaining,
        key=lambda index: (
            records[index]["max_tau"],
            -records[index]["acceptance_mean"],
            records[index]["index"],
        ),
    )
    for index in diagnostic_order[: n_featured - len(selected)]:
        selected.append(index)
        reasons[index] = "lowest autocorrelation diagnostic"

    return [
        (records[index]["index"], reasons.get(index, "diagnostic selection"))
        for index in selected[:n_featured]
    ]


def _load_controls(
    selection_path: Path,
    stack_path: Path,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    controls: dict[str, np.ndarray] = {}
    selection = np.load(selection_path, allow_pickle=False)
    selected_indices = np.asarray(selection["index"], dtype=np.int64)
    if np.array_equal(selected_indices, indices) and "mean_g" in selection.files:
        controls["NPE posterior mean"] = np.asarray(selection["mean_g"], dtype=float)
    if stack_path.is_file():
        stack = np.load(stack_path, allow_pickle=False)
        if np.array_equal(np.asarray(stack["index"], dtype=np.int64), indices):
            controls["NPE shared 16-noise MAP"] = np.asarray(
                stack["stack_map"][:, -1], dtype=float
            )
    return controls


def _nuisance_rows(
    truth_g: np.ndarray,
    estimates: dict[str, np.ndarray],
    *,
    replicates: int,
) -> list[dict]:
    rng = np.random.default_rng(420042)
    rows = []
    for count in NUISANCE_PREFIXES:
        if count > len(truth_g):
            continue
        draws = rng.integers(0, count, size=(replicates, count), dtype=np.int64)
        for name, estimate in estimates.items():
            metrics = fit_component_mc(truth_g[:count], estimate[:count])
            bootstrap = np.asarray(
                [
                    fit_component_mc(
                        truth_g[:count][draw], estimate[:count][draw]
                    )["combined_m"]
                    for draw in draws
                ]
            )
            rows.append(
                {
                    "estimator": name,
                    "nuisance_count": int(count),
                    "combined_m_lower": float(np.quantile(bootstrap, 0.16)),
                    "combined_m_upper": float(np.quantile(bootstrap, 0.84)),
                    **metrics,
                }
            )
    return rows


def _save_summary_figures(
    report_dir: Path,
    records: list[dict],
    parameter_names: tuple[str, ...],
    nuisance_rows: list[dict],
) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures: dict[str, str] = {}
    acceptance = np.asarray([row["acceptance_mean"] for row in records])
    max_tau = np.asarray([row["max_tau"] for row in records])
    min_ess = np.asarray([row["min_effective_sample_size"] for row in records])
    converged = np.asarray([row["converged"] for row in records])

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(acceptance, bins=15, color="#1f4e79")
    axes[0].set_xlabel("acceptance fraction")
    axes[1].hist(max_tau, bins=15, color="#8f3b76")
    axes[1].set_xlabel("maximum autocorrelation time")
    axes[2].hist(min_ess, bins=15, color="#2e7d32")
    axes[2].set_xlabel("minimum effective sample size")
    figure.suptitle(f"Sampling diagnostics ({np.count_nonzero(converged)}/{len(records)} converged)")
    figure.tight_layout()
    path = report_dir / "sampling_diagnostics.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    figures["sampling_diagnostics"] = path.name

    truth = np.stack([row["truth"] for row in records])
    mean = np.stack([row["mean"] for row in records])
    lower = np.stack([row["lower"] for row in records])
    upper = np.stack([row["upper"] for row in records])
    coverage = np.mean((truth >= lower) & (truth <= upper), axis=0)
    residual = mean - truth
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(np.arange(len(parameter_names)), coverage, color="#1f4e79")
    axis.axhline(0.68, color="#555", ls="--", lw=0.8, label="nominal 68%")
    axis.set_xticks(np.arange(len(parameter_names)), parameter_names, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("fraction containing truth")
    axis.set_title("16–84% marginal interval coverage")
    axis.legend(frameon=False)
    figure.tight_layout()
    path = report_dir / "coverage_by_parameter.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    figures["coverage"] = path.name

    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.errorbar(
        np.arange(len(parameter_names)),
        np.mean(residual, axis=0),
        yerr=np.std(residual, axis=0),
        fmt="o",
        capsize=3,
        color="#8f3b76",
    )
    axis.set_xticks(np.arange(len(parameter_names)), parameter_names, rotation=35, ha="right")
    axis.set_ylabel("posterior mean − truth")
    axis.set_title("Posterior-mean residuals across selected galaxies")
    figure.tight_layout()
    path = report_dir / "residuals_by_parameter.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    figures["residuals"] = path.name

    colors = ("#1f4e79", "#8f3b76", "#a05a2c", "#2e7d32")
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, name in zip(
        colors,
        sorted({row["estimator"] for row in nuisance_rows}),
    ):
        subset = sorted(
            [row for row in nuisance_rows if row["estimator"] == name],
            key=lambda row: row["nuisance_count"],
        )
        x = np.asarray([row["nuisance_count"] for row in subset], dtype=float)
        y = np.asarray([row["combined_m"] for row in subset], dtype=float)
        lower_m = np.asarray([row["combined_m_lower"] for row in subset])
        upper_m = np.asarray([row["combined_m_upper"] for row in subset])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower_m, upper_m - y)),
            marker="o",
            capsize=3,
            color=color,
            label=name,
        )
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(NUISANCE_PREFIXES))
    axis.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: str(int(value))))
    axis.set_xlabel("number of selected galaxies")
    axis.set_ylabel("combined multiplicative bias m")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("Ensemble multiplicative bias versus cumulative galaxy count")
    figure.tight_layout()
    path = report_dir / "m_vs_nuisance_stacks.png"
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    figures["m_vs_nuisance"] = path.name
    return figures


def _write_html(
    report_dir: Path,
    metadata: dict,
    figures: dict[str, str],
    records: list[dict],
    parameter_names: tuple[str, ...],
    parameter_stats: list[dict],
    ensemble_metrics: list[dict],
    featured: list[tuple[int, str]],
) -> None:
    record_by_index = {row["index"]: row for row in records}
    featured_html = []
    for index, reason in featured:
        row = record_by_index[index]
        featured_html.append(
            f"<figure><img src=\"corner_plots/{html.escape(row['corner_path'])}\" "
            f"alt=\"9D corner plot for galaxy {index}\">"
            f"<figcaption>Dataset index {index}: {html.escape(reason)}; "
            f"true |g|={np.hypot(row['truth'][0], row['truth'][1]):.4f}, "
            f"max tau={row['max_tau']:.1f}, "
            f"acceptance={row['acceptance_mean']:.3f}, "
            f"converged={row['converged']}.</figcaption></figure>"
        )
    parameter_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['parameter'])}</td>"
        f"<td>{row['mean_bias']:.5g}</td>"
        f"<td>{row['rmse']:.5g}</td>"
        f"<td>{row['coverage']:.3f}</td>"
        f"<td>{row['mean_interval_width']:.5g}</td>"
        "</tr>"
        for row in parameter_stats
    )
    ensemble_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{row['g1_m']:.5g}</td><td>{row['g2_m']:.5g}</td>"
        f"<td>{row['g1_c']:.5g}</td><td>{row['g2_c']:.5g}</td>"
        f"<td>{row['combined_m']:.5g}</td>"
        "</tr>"
        for row in ensemble_metrics
    )
    diagnostic_rows = "".join(
        "<tr>"
        f"<td>{row['index']}</td>"
        f"<td>{np.hypot(row['truth'][0], row['truth'][1]):.5f}</td>"
        f"<td>{row['acceptance_mean']:.3f}</td>"
        f"<td>{row['max_tau']:.2f}</td>"
        f"<td>{row['min_effective_sample_size']:.1f}</td>"
        f"<td>{row['converged']}</td>"
        "</tr>"
        for row in records
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>9D NRE persisted-chain corner report</title>
<style>{HTML_STYLE}
section {{ margin-top: 2rem; }}
.figure-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
.figure-grid figure {{ margin: 0; }}
.figure-grid img {{ width: 100%; }}
</style></head><body>
<h1>9D NRE persisted-chain corner report</h1>
<p class="lead">This report summarizes the persisted full-9D emcee chains
for all {len(records)} galaxies. All corner plots are stored in
<code>corner_plots/</code>; only eight deterministic diagnostic examples are
shown here.</p>
<section><h2>Run provenance</h2>
<pre>{html.escape(json.dumps(metadata, indent=2))}</pre></section>
<section><h2>Aggregate sampling diagnostics</h2>
<p><img src="{html.escape(figures['sampling_diagnostics'])}" alt="sampling diagnostics"></p>
<p>Convergence uses the inference criterion
<code>production &gt;= 50 * max(autocorrelation_time)</code>. Effective sample
size is estimated per parameter as
<code>nwalkers * production / autocorrelation_time</code>.</p>
<table><thead><tr><th>Index</th><th>true |g|</th><th>acceptance</th>
<th>max tau</th><th>min ESS</th><th>converged</th></tr></thead>
<tbody>{diagnostic_rows}</tbody></table></section>
<section><h2>Posterior quality by parameter</h2>
<p><img src="{html.escape(figures['coverage'])}" alt="coverage by parameter"></p>
<p><img src="{html.escape(figures['residuals'])}" alt="posterior residuals"></p>
<table><thead><tr><th>Parameter</th><th>mean bias</th><th>RMSE</th>
<th>16–84% coverage</th><th>mean interval width</th></tr></thead>
<tbody>{parameter_rows}</tbody></table></section>
<section><h2>Ensemble shear bias</h2>
<img src="{html.escape(figures['m_vs_nuisance'])}" alt="m versus galaxy count">
<table><thead><tr><th>Estimator</th><th>m g1</th><th>m g2</th>
<th>c g1</th><th>c g2</th><th>combined m</th></tr></thead>
<tbody>{ensemble_rows}</tbody></table></section>
<section><h2>Featured 9D corner plots</h2>
<div class="figure-grid">{"".join(featured_html)}</div></section>
</body></html>
"""
    (report_dir / "report.html").write_text(body, encoding="utf-8")


def build_report(args) -> None:
    report_dir = args.report_dir or NRE9D_DIR / args.run_name
    if (report_dir / "report.html").exists() and not args.overwrite:
        raise FileExistsError(f"{report_dir / 'report.html'} exists; use --overwrite")
    if args.n_featured < 1 or args.n_featured > 8:
        raise ValueError("--n-featured must be between 1 and 8")
    if args.corner_max_samples < 100:
        raise ValueError("--corner-max-samples must be at least 100")
    if args.bootstrap_replicates < 1:
        raise ValueError("--bootstrap-replicates must be positive")

    run_dir = args.cache_root / args.nre_name / args.run_name
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    indices = _load_indices(args.galaxies)
    artifact_paths = [run_dir / f"galaxy_{int(index):06d}.npz" for index in indices]
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} chain artifacts are missing; first missing: {missing[0]}"
        )

    model_config = load_model_config(
        args.parent_npe,
        configs_root=str(args.model_root.parent / "configs"),
    )
    config.set_model_config(model_config)
    names = tuple(metadata["parameter_names"])
    if names != tuple(config.TARGET_NAMES):
        raise ValueError("cached parameter names do not match the current schema")
    config.require_matching_dataset_par_ranges(
        Path(metadata["data_dir"]),
        config.par_ranges,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    corner_dir = report_dir / "corner_plots"
    corner_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in artifact_paths:
        artifact = _load_artifact(path)
        if artifact["parameter_names"] != names:
            raise ValueError(f"{path} has incompatible parameter names")
        records.append(
            _posterior_record(
                artifact,
                names=names,
                par_ranges=config.par_ranges,
                target_transforms=config.TARGET_TRANSFORMS,
                corner_dir=corner_dir,
                corner_max_samples=args.corner_max_samples,
            )
        )

    truth = np.stack([row["truth"] for row in records])
    means = np.stack([row["mean"] for row in records])
    lower = np.stack([row["lower"] for row in records])
    upper = np.stack([row["upper"] for row in records])
    residual = means - truth
    parameter_stats = []
    for column, name in enumerate(names):
        parameter_stats.append(
            {
                "parameter": name,
                "mean_bias": float(np.mean(residual[:, column])),
                "rmse": float(np.sqrt(np.mean(residual[:, column] ** 2))),
                "coverage": float(
                    np.mean(
                        (truth[:, column] >= lower[:, column])
                        & (truth[:, column] <= upper[:, column])
                    )
                ),
                "mean_interval_width": float(
                    np.mean(upper[:, column] - lower[:, column])
                ),
            }
        )

    g1, g2 = names.index("g1"), names.index("g2")
    truth_g = truth[:, [g1, g2]]
    estimates = {
        "9D NRE posterior mean": means[:, [g1, g2]],
        "9D NRE MAP-like": np.stack([row["map_like"][[g1, g2]] for row in records]),
    }
    if args.galaxies.is_file():
        estimates.update(_load_controls(args.galaxies, args.stack_draws, indices))
    ensemble_metrics = [
        {"estimator": name, **fit_component_mc(truth_g, estimate)}
        for name, estimate in estimates.items()
    ]
    nuisance_rows = _nuisance_rows(
        truth_g,
        estimates,
        replicates=args.bootstrap_replicates,
    )
    figures = _save_summary_figures(report_dir, records, names, nuisance_rows)
    featured = _featured_indices(records, args.n_featured)
    report_metadata = {
        **metadata,
        "report_dir": str(report_dir),
        "corner_plot_dir": str(corner_dir),
        "n_corner_plots": len(records),
        "featured_indices": [index for index, _ in featured],
        "figures": figures,
    }
    payload = {
        "metadata": report_metadata,
        "parameter_names": list(names),
        "sampling_summary": {
            "n_galaxies": len(records),
            "converged_fraction": float(np.mean([row["converged"] for row in records])),
            "acceptance_mean": float(np.mean([row["acceptance_mean"] for row in records])),
            "max_tau": float(np.max([row["max_tau"] for row in records])),
            "min_ess": float(
                np.min([row["min_effective_sample_size"] for row in records])
            ),
        },
        "parameter_statistics": parameter_stats,
        "ensemble_metrics": ensemble_metrics,
        "nuisance_stack_table": nuisance_rows,
        "featured": [
            {"index": index, "reason": reason} for index, reason in featured
        ],
        "figures": figures,
        "per_galaxy": [
            {
                "index": row["index"],
                "truth": row["truth"],
                "mean": row["mean"],
                "lower": row["lower"],
                "upper": row["upper"],
                "map_like": row["map_like"],
                "residual": row["residual"],
                "diagnostics": row["diagnostics"],
                "effective_sample_size": row["effective_sample_size"],
                "corner_path": row["corner_path"],
            }
            for row in records
        ],
    }
    np.savez_compressed(
        report_dir / "aggregate_summary.npz",
        index=np.asarray([row["index"] for row in records], dtype=np.int64),
        truth=truth,
        mean=means,
        lower=lower,
        upper=upper,
        residual=residual,
    )
    write_json(report_dir / "corner_report.json", payload)
    _write_html(
        report_dir,
        report_metadata,
        figures,
        records,
        names,
        parameter_stats,
        ensemble_metrics,
        featured,
    )
    print(f"Wrote {report_dir / 'report.html'}", flush=True)


def main(argv=None) -> None:
    build_report(parse_args(argv))


if __name__ == "__main__":
    main()
