#!/usr/bin/env python3
"""Build an HTML report from a Nautilus factorized 9D NRE campaign."""

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
from diagnostics.likelihood_stack_benchmark_core import fit_component_mc
from diagnostics.likelihood_stack_core import (
    CACHE_ROOT,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    NRE9D_DIR,
    NRE9D_NAME,
    json_safe,
    write_json,
)
from diagnostics.likelihood_stack_nre_9d_nautilus import (
    artifact_path,
    chain_run_dir,
    load_artifact,
    normalized_weights,
)
from tf_prior import TFPrior
from utils import denormalize


DEFAULT_RUN_NAME = "nautilus_factorized_n100_s42"
FIGURE_DPI = 140


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--selection-file", type=Path, default=None)
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


def load_selection(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "index" not in data.files:
        raise ValueError(f"{path} does not contain an index array")
    indices = np.asarray(data["index"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{path} index must be non-empty and one-dimensional")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{path} index contains duplicates")
    return indices


def physical_posterior(
    artifact: dict,
    *,
    names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    points = artifact["points"]
    weights = normalized_weights(artifact["log_weights"])
    physical = denormalize(
        points,
        config.par_ranges,
        feature_names=names,
        target_transforms=config.TARGET_TRANSFORMS,
    )
    return physical, weights


def _weighted_quantile(
    values: np.ndarray, quantiles: tuple[float, float], weights: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    coordinate = np.cumsum(weights) - 0.5 * weights
    return np.interp(np.asarray(quantiles), coordinate, values)


def posterior_record(
    artifact: dict,
    *,
    names: tuple[str, ...],
    corner_dir: Path,
    corner_max_samples: int,
    make_corner: bool,
) -> dict:
    physical, weights = physical_posterior(artifact, names=names)
    summary = artifact["summary"]
    means = np.sum(physical * weights[:, None], axis=0)
    intervals = np.stack(
        [
            _weighted_quantile(physical[:, column], (0.16, 0.84), weights)
            for column in range(physical.shape[1])
        ],
        axis=0,
    )
    record = {
        "index": int(artifact["index"]),
        "truth": artifact["truth"],
        "mean": means,
        "lower": intervals[:, 0],
        "upper": intervals[:, 1],
        "map_like": np.asarray(
            [
                summary["g1_map_like"],
                summary["g2_map_like"],
            ],
            dtype=np.float64,
        ),
        "residual": means - artifact["truth"],
        "interval_width": intervals[:, 1] - intervals[:, 0],
        "effective_sample_size": float(summary["effective_sample_size"]),
        "nautilus_n_eff": float(summary["nautilus_n_eff"]),
        "n_like": int(summary["n_like"]),
        "eta": float(summary["eta"]),
        "log_z_relative": float(summary["log_z_relative"]),
        "runtime_s": float(summary["runtime_s"]),
        "status": summary.get("status", "completed"),
    }
    if make_corner:
        import corner

        corner_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(420042 + int(artifact["index"]))
        plot_count = min(int(corner_max_samples), max(1000, len(physical)))
        choice = rng.choice(
            len(physical),
            size=plot_count,
            replace=True,
            p=weights,
        )
        points = physical[choice]
        plot_weights = None
        figure = corner.corner(
            points,
            labels=list(names),
            truths=artifact["truth"],
            weights=plot_weights,
            show_titles=True,
            title_fmt=".3g",
            plot_datapoints=False,
            fill_contours=True,
        )
        figure.savefig(
            corner_dir / f"galaxy_{int(artifact['index']):06d}.png",
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
        record["corner_path"] = f"corner_plots/galaxy_{int(artifact['index']):06d}.png"
    return record


def posterior_pit(
    artifact: dict,
    *,
    names: tuple[str, ...],
) -> np.ndarray:
    """Return each component's weighted posterior CDF evaluated at truth."""

    physical, weights = physical_posterior(artifact, names=names)
    truth = np.asarray(artifact["truth"], dtype=np.float64)
    pits = np.empty(len(names), dtype=np.float64)
    for column in range(len(names)):
        below = physical[:, column] < truth[column]
        equal = physical[:, column] == truth[column]
        pits[column] = np.sum(weights[below]) + 0.5 * np.sum(weights[equal])
    return pits


def posterior_pp_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a one-dimensional posterior P-P curve and KS distance."""

    values = np.sort(np.asarray(values, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size == 0 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PIT values must be finite and lie in [0, 1]")
    empirical = np.arange(1, len(values) + 1, dtype=np.float64) / len(values)
    before = (np.arange(len(values), dtype=np.float64)) / len(values)
    distance = float(
        max(np.max(empirical - values), np.max(values - before))
    )
    return values, empirical, distance


def featured_records(records: list[dict], n_featured: int) -> list[dict]:
    if n_featured <= 0:
        return []
    n_featured = min(int(n_featured), len(records))
    amplitude = np.asarray([np.hypot(*row["truth"][:2]) for row in records])
    selected: list[int] = []
    for group in np.array_split(np.argsort(amplitude), min(4, n_featured)):
        if len(group):
            selected.append(
                min(group.tolist(), key=lambda i: abs(amplitude[i] - np.median(amplitude[group])))
            )
    remaining = [i for i in range(len(records)) if i not in selected]
    remaining.sort(key=lambda i: records[i]["effective_sample_size"])
    selected.extend(remaining[: n_featured - len(selected)])
    return [records[i] for i in selected[:n_featured]]


def save_figures(
    report_dir: Path,
    records: list[dict],
    names: tuple[str, ...],
    estimates: dict[str, np.ndarray],
    ensemble_metrics: list[dict],
    pp_plot_data: dict[str, dict],
) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures: dict[str, str] = {}

    def symmetric_limit(values: np.ndarray) -> float:
        scale = float(np.nanmax(np.abs(values)))
        return 1.1 * scale if scale > 0.0 else 1.0

    n_eff = np.asarray([row["effective_sample_size"] for row in records])
    n_like = np.asarray([row["n_like"] for row in records])
    eta = np.asarray([row["eta"] for row in records])
    runtime = np.asarray([row["runtime_s"] for row in records])
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].hist(n_eff, bins=15, color="#1f4e79")
    axes[0].set_xlabel("posterior effective sample size")
    axes[1].hist(n_like, bins=15, color="#8f3b76")
    axes[1].set_xlabel("likelihood calls")
    axes[2].hist(runtime, bins=15, color="#2e7d32")
    axes[2].set_xlabel("runtime (s)")
    figure.suptitle("Nautilus sampling diagnostics")
    figure.tight_layout()
    figure.savefig(report_dir / "sampling_diagnostics.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["sampling_diagnostics"] = "sampling_diagnostics.png"

    truth = np.stack([row["truth"] for row in records])
    lower = np.stack([row["lower"] for row in records])
    upper = np.stack([row["upper"] for row in records])
    mean = np.stack([row["mean"] for row in records])
    coverage = np.mean((truth >= lower) & (truth <= upper), axis=0)
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(np.arange(len(names)), coverage, color="#1f4e79")
    axis.axhline(0.68, color="#555", ls="--", lw=0.8, label="nominal 68%")
    axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("fraction containing truth")
    axis.set_title("16–84% marginal interval coverage")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(report_dir / "coverage_by_parameter.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["coverage"] = "coverage_by_parameter.png"

    residual = mean - truth
    figure, axes = plt.subplots(3, 3, figsize=(11, 9), squeeze=False)
    for column, name in enumerate(names):
        axis = axes.flat[column]
        values = residual[:, column]
        x_limit = symmetric_limit(truth[:, column])
        y_limit = symmetric_limit(values)
        axis.axhline(0.0, color="#555", ls="--", lw=0.8)
        axis.scatter(
            truth[:, column],
            values,
            s=10,
            alpha=0.35,
            linewidths=0,
            color="#1f4e79",
        )
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_title(name)
        axis.set_xlabel("truth")
        axis.set_ylabel("posterior mean − truth")
    figure.suptitle("Posterior-mean bias versus truth by parameter")
    figure.tight_layout()
    figure.savefig(
        report_dir / "bias_vs_truth_by_parameter.png",
        dpi=FIGURE_DPI,
    )
    plt.close(figure)
    figures["bias_vs_truth"] = "bias_vs_truth_by_parameter.png"

    figure, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True)
    panels = (
        (axes[0, 0], estimates["Nautilus posterior mean"], "posterior mean", 0),
        (axes[0, 1], estimates["Nautilus MAP-like"], "MAP-like", 0),
        (axes[1, 0], estimates["Nautilus posterior mean"], "posterior mean", 1),
        (axes[1, 1], estimates["Nautilus MAP-like"], "MAP-like", 1),
    )
    for axis, estimate, estimator, component in panels:
        metric = next(
            row
            for row in ensemble_metrics
            if row["estimator"] == f"Nautilus {estimator}"
        )
        axis.scatter(
            truth[:, component],
            estimate[:, component] - truth[:, component],
            s=10,
            alpha=0.35,
            linewidths=0,
            color="#1f4e79",
        )
        x_limit = max(
            0.1,
            float(np.max(np.abs(truth[:, component]))),
        )
        y_limit = max(
            0.1,
            float(np.max(np.abs(estimate[:, component] - truth[:, component]))),
        )
        axis.axhline(0.0, color="#555", ls="--", lw=0.8)
        axis.set_xlim(-x_limit, x_limit)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_title(
            f"{estimator}, {'g1' if component == 0 else 'g2'}\n"
            f"m={metric['g1_m' if component == 0 else 'g2_m']:.3f}, "
            f"c={metric['g1_c' if component == 0 else 'g2_c']:.3g}"
        )
    axes[1, 0].set_xlabel("true shear")
    axes[1, 1].set_xlabel("true shear")
    axes[0, 0].set_ylabel("estimated shear − truth")
    axes[1, 0].set_ylabel("estimated shear − truth")
    figure.suptitle("MAP and posterior-mean shear bias versus truth")
    figure.tight_layout()
    figure.savefig(
        report_dir / "g_estimates_vs_truth.png",
        dpi=FIGURE_DPI,
    )
    plt.close(figure)
    figures["g_estimates_vs_truth"] = "g_estimates_vs_truth.png"

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    grid = np.linspace(0.0, 1.0, 501)
    for axis, component in zip(axes, ("g1", "g2")):
        diagnostic = pp_plot_data[component]
        x = np.asarray(diagnostic["pit_sorted"])
        y = np.asarray(diagnostic["empirical_cdf"])
        epsilon = diagnostic["dkw_epsilon"]
        axis.fill_between(
            grid,
            np.maximum(0.0, grid - epsilon),
            np.minimum(1.0, grid + epsilon),
            color="#1f4e79",
            alpha=0.12,
        )
        axis.step(
            x,
            y,
            where="post",
            color="#1f4e79",
            label=f"KS={diagnostic['ks_distance']:.3f}",
        )
        axis.plot(grid, grid, color="#555", ls="--", label="uniform")
        axis.set_title(component)
        axis.set_xlabel("posterior quantile at truth")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(frameon=False)
    axes[0].set_ylabel("fraction of galaxies")
    figure.suptitle("Posterior P-P coverage diagnostic")
    figure.tight_layout()
    figure.savefig(report_dir / "posterior_pp.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["posterior_pp"] = "posterior_pp.png"

    return figures


def write_html(
    path: Path,
    *,
    metadata: dict,
    figures: dict[str, str],
    records: list[dict],
    parameter_stats: list[dict],
    ensemble_metrics: list[dict],
    pp_summary: dict[str, dict],
    featured: list[dict],
) -> None:
    figure_html = "".join(
        f'<figure><img src="{html.escape(name)}" alt="{html.escape(key)}">'
        f"<figcaption>{html.escape(key.replace('_', ' ').title())}</figcaption></figure>"
        for key, name in figures.items()
    )
    featured_html = "".join(
        f'<figure><img src="{html.escape(row["corner_path"])}" '
        f'alt="corner plot for galaxy {row["index"]}">'
        f"<figcaption>Galaxy {row['index']}; "
        f"effective sample size {row['effective_sample_size']:.1f}, "
        f"{row['n_like']} likelihood calls.</figcaption></figure>"
        for row in featured
    )
    parameter_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['parameter'])}</td>"
        f"<td>{row['mean_bias']:.5g}</td><td>{row['rmse']:.5g}</td>"
        f"<td>{row['coverage']:.3f}</td>"
        f"<td>{row['mean_interval_width']:.5g}</td>"
        "</tr>"
        for row in parameter_stats
    )
    ensemble_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['estimator'])}</td>"
        f"<td>{row['g1_m']:.5g}</td><td>{row['g2_m']:.5g}</td>"
        f"<td>{row['combined_m']:.5g}</td>"
        "</tr>"
        for row in ensemble_metrics
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Nautilus 9D NRE report</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Nautilus 9D NRE likelihood-stack report</h1>
<p class="lead">A seeded random sample of {metadata['n_expected']} galaxies from
<code>{html.escape(metadata['data_dir'])}</code> using the factorized joint NRE.</p>
<p>The sampled population is the TF-conformed xu3 set. The likelihood includes
the truncated-normal Tully–Fisher prior for <code>vcirc</code>; its exact
parameters are recorded below.</p>
<p>Nautilus posterior evidence is reported only as a relative diagnostic because
the NRE ratio is defined up to an additive log constant.</p>
<h2>Summary figures</h2>{figure_html}
<h2>Shear calibration</h2>
<table><thead><tr><th>Estimator</th><th>g1 m</th><th>g2 m</th><th>combined m</th>
</tr></thead><tbody>{ensemble_rows}</tbody></table>
<table><thead><tr><th>Parameter</th><th>mean bias</th><th>RMSE</th>
<th>68% coverage</th><th>mean interval width</th></tr></thead>
<tbody>{parameter_rows}</tbody></table>
<h2>Posterior P-P coverage</h2>
<p>The P-P curve uses each galaxy's weighted posterior CDF evaluated at
the true shear. Uniformity is the expected calibration target.</p>
<table><thead><tr><th>Component</th><th>galaxies</th><th>KS distance</th>
<th>ESS</th><th>16–84% coverage</th></tr></thead><tbody>
{"".join(
    f"<tr><td>{html.escape(component)}</td><td>{diag['n']}</td>"
    f"<td>{diag['ks_distance']:.5g}</td>"
    f"<td>{diag['effective_sample_size']:.2f}</td>"
    f"<td>{diag['central_coverage']:.3f}</td></tr>"
    for component, diag in pp_summary.items()
)}
</tbody></table>
<h2>Representative posterior plots</h2>{featured_html}
</body></html>"""
    path.write_text(body, encoding="utf-8")


def main(argv=None) -> None:
    args = parse_args(argv)
    names = tuple(config.TARGET_NAMES)
    run_dir = Path(args.run_dir or chain_run_dir(args.cache_root, args.nre_name, args.run_name))
    report_dir = Path(args.report_dir or NRE9D_DIR / args.run_name)
    selection_file = Path(args.selection_file or run_dir / "selection.npz")
    output = report_dir / "report.html"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite")
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["tf_prior"] = TFPrior().to_dict()
    indices = load_selection(selection_file)
    records = []
    failed = []
    for index in indices:
        path = artifact_path(run_dir, int(index))
        if not path.is_file():
            failed.append(int(index))
            continue
        artifact = load_artifact(path)
        records.append(artifact)
    if len(records) < 2:
        raise RuntimeError("at least two successful artifacts are required for a report")
    records.sort(key=lambda row: row["index"])
    report_dir.mkdir(parents=True, exist_ok=True)
    featured_indices = set(
        row["index"] for row in featured_records(
            [
                {
                    "index": row["index"],
                    "truth": row["truth"],
                    "effective_sample_size": row["summary"]["effective_sample_size"],
                }
                for row in records
            ],
            args.n_featured,
        )
    )
    report_records = [
        posterior_record(
            artifact,
            names=names,
            corner_dir=report_dir / "corner_plots",
            corner_max_samples=args.corner_max_samples,
            make_corner=artifact["index"] in featured_indices,
        )
        for artifact in records
    ]
    truth = np.stack([row["truth"] for row in report_records])
    mean = np.stack([row["mean"] for row in report_records])
    lower = np.stack([row["lower"] for row in report_records])
    upper = np.stack([row["upper"] for row in report_records])
    parameter_stats = []
    for column, name in enumerate(names):
        parameter_stats.append(
            {
                "parameter": name,
                "mean_bias": float(np.mean(mean[:, column] - truth[:, column])),
                "rmse": float(np.sqrt(np.mean((mean[:, column] - truth[:, column]) ** 2))),
                "coverage": float(
                    np.mean(
                        (truth[:, column] >= lower[:, column])
                        & (truth[:, column] <= upper[:, column])
                    )
                ),
                "mean_interval_width": float(np.mean(upper[:, column] - lower[:, column])),
            }
        )
    truth_g = truth[:, :2]
    estimates = {
        "Nautilus posterior mean": mean[:, :2],
        "Nautilus MAP-like": np.asarray(
            [row["map_like"] for row in report_records], dtype=np.float64
        ),
    }
    ensemble_metrics = [
        {"estimator": name, **fit_component_mc(truth_g, estimate)}
        for name, estimate in estimates.items()
    ]
    pits = np.stack(
        [posterior_pit(artifact, names=names) for artifact in records],
        axis=0,
    )
    pp_plot_data = {}
    pp_summary = {}
    mean_effective_sample_size = float(
        np.mean([row["effective_sample_size"] for row in report_records])
    )
    for column, name in enumerate(names[:2]):
        pit_sorted, empirical_cdf, ks_distance = posterior_pp_curve(pits[:, column])
        pp_plot_data[name] = {
            "pit_sorted": pit_sorted,
            "empirical_cdf": empirical_cdf,
            "ks_distance": ks_distance,
            "dkw_epsilon": 1.36 / np.sqrt(len(pit_sorted)),
        }
        pp_summary[name] = {
            "n": int(len(pit_sorted)),
            "ks_distance": float(ks_distance),
            "effective_sample_size": mean_effective_sample_size,
            "central_coverage": float(
                np.mean((pits[:, column] >= 0.16) & (pits[:, column] <= 0.84))
            ),
        }
    (report_dir / "m_vs_nuisance_stacks.png").unlink(missing_ok=True)
    (report_dir / "normalized_residuals_by_parameter.png").unlink(missing_ok=True)
    figures = save_figures(
        report_dir,
        report_records,
        names,
        estimates,
        ensemble_metrics,
        pp_plot_data,
    )
    featured = [row for row in report_records if "corner_path" in row]
    payload = {
        "metadata": {
            **metadata,
            "status": "completed" if not failed else "partial",
            "n_successful": len(records),
            "n_failed": len(failed),
            "failed_indices": failed,
            "report_dir": str(report_dir),
            "figures": figures,
            "n_corner_plots": len(featured),
        },
        "parameter_names": list(names),
        "parameter_statistics": parameter_stats,
        "ensemble_metrics": ensemble_metrics,
        "posterior_pp": pp_summary,
        "per_galaxy": report_records,
        "featured": [row["index"] for row in featured],
    }
    write_json(report_dir / "report.json", payload)
    # Keep the campaign index compatible with the existing 9D corner-report
    # convention while the report itself is Nautilus-specific.
    write_json(report_dir / "corner_report.json", payload)
    write_html(
        output,
        metadata=payload["metadata"],
        figures=figures,
        records=report_records,
        parameter_stats=parameter_stats,
        ensemble_metrics=ensemble_metrics,
        pp_summary=pp_summary,
        featured=featured,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
