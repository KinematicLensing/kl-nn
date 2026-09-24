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
from utils import denormalize


DEFAULT_RUN_NAME = "nautilus_factorized_n100_s42"
NAUTILUS_PREFIXES = (8, 16, 32, 64, 100)
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
        points = physical
        if len(points) > corner_max_samples:
            rng = np.random.default_rng(420042 + int(artifact["index"]))
            choice = rng.choice(
                len(points),
                size=corner_max_samples,
                replace=False,
                p=weights,
            )
            points = points[choice]
            plot_weights = None
        else:
            plot_weights = weights
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


def nuisance_rows(
    truth_g: np.ndarray,
    estimate: np.ndarray,
    *,
    replicates: int,
) -> list[dict]:
    rng = np.random.default_rng(420042)
    rows = []
    for count in NAUTILUS_PREFIXES:
        if count > len(truth_g):
            continue
        metrics = fit_component_mc(truth_g[:count], estimate[:count])
        draws = rng.integers(0, count, size=(replicates, count), dtype=np.int64)
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
                "nuisance_count": int(count),
                "combined_m_lower": float(np.quantile(bootstrap, 0.16)),
                "combined_m_upper": float(np.quantile(bootstrap, 0.84)),
                **metrics,
            }
        )
    return rows


def save_figures(
    report_dir: Path,
    records: list[dict],
    names: tuple[str, ...],
    nuisance: list[dict],
) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures: dict[str, str] = {}
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
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.errorbar(
        np.arange(len(names)),
        residual.mean(axis=0),
        yerr=residual.std(axis=0),
        fmt="o",
        capsize=3,
        color="#8f3b76",
    )
    axis.set_xticks(np.arange(len(names)), names, rotation=35, ha="right")
    axis.set_ylabel("posterior mean − truth")
    axis.set_title("Posterior-mean residuals")
    figure.tight_layout()
    figure.savefig(report_dir / "residuals_by_parameter.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["residuals"] = "residuals_by_parameter.png"

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    if nuisance:
        x = np.asarray([row["nuisance_count"] for row in nuisance], dtype=float)
        y = np.asarray([row["combined_m"] for row in nuisance], dtype=float)
        lower_m = np.asarray([row["combined_m_lower"] for row in nuisance])
        upper_m = np.asarray([row["combined_m_upper"] for row in nuisance])
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower_m, upper_m - y)),
            marker="o",
            capsize=3,
            color="#1f4e79",
            label="Nautilus posterior mean",
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(NAUTILUS_PREFIXES))
        axis.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda value, _: str(int(value)))
        )
        axis.legend(frameon=False)
    else:
        axis.text(
            0.5,
            0.5,
            "fewer than 8 successful galaxies",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.set_xlabel("number of sampled galaxies")
    axis.set_ylabel("combined multiplicative bias m")
    axis.set_title("Nautilus m versus cumulative galaxy count")
    figure.tight_layout()
    figure.savefig(report_dir / "m_vs_nuisance_stacks.png", dpi=FIGURE_DPI)
    plt.close(figure)
    figures["m_vs_nuisance"] = "m_vs_nuisance_stacks.png"

    return figures


def write_html(
    path: Path,
    *,
    metadata: dict,
    figures: dict[str, str],
    records: list[dict],
    parameter_stats: list[dict],
    ensemble_metrics: list[dict],
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
        f"<td>{row['coverage']:.3f}</td><td>{row['mean_interval_width']:.5g}</td>"
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
    galaxy_rows = "".join(
        "<tr>"
        f"<td>{row['index']}</td><td>{row['effective_sample_size']:.1f}</td>"
        f"<td>{row['n_like']}</td><td>{row['runtime_s']:.2f}</td>"
        f"<td>{row['mean'][0]:.5g} [{row['lower'][0]:.5g}, {row['upper'][0]:.5g}]</td>"
        f"<td>{row['mean'][1]:.5g} [{row['lower'][1]:.5g}, {row['upper'][1]:.5g}]</td>"
        "</tr>"
        for row in records
    )
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Nautilus 9D NRE report</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Nautilus 9D NRE likelihood-stack report</h1>
<p class="lead">A seeded random sample of {metadata['n_expected']} galaxies from
<code>{html.escape(metadata['data_dir'])}</code> using the factorized joint NRE.</p>
<p>Nautilus posterior evidence is reported only as a relative diagnostic because
the NRE ratio is defined up to an additive log constant.</p>
<h2>Run metadata</h2><pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
<h2>Summary figures</h2>{figure_html}
<h2>Shear calibration</h2>
<table><thead><tr><th>Estimator</th><th>g1 m</th><th>g2 m</th><th>combined m</th>
</tr></thead><tbody>{ensemble_rows}</tbody></table>
<table><thead><tr><th>Parameter</th><th>mean bias</th><th>RMSE</th>
<th>68% coverage</th><th>mean interval width</th></tr></thead>
<tbody>{parameter_rows}</tbody></table>
<h2>Representative posterior plots</h2>{featured_html}
<h2>Per-galaxy posterior summaries</h2>
<table><thead><tr><th>Index</th><th>ESS</th><th>likelihood calls</th>
<th>runtime (s)</th><th>g1 mean [16,84]</th><th>g2 mean [16,84]</th>
</tr></thead><tbody>{galaxy_rows}</tbody></table>
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
    nuisance = nuisance_rows(
        truth_g,
        estimates["Nautilus posterior mean"],
        replicates=args.bootstrap_replicates,
    )
    figures = save_figures(
        report_dir,
        report_records,
        names,
        nuisance,
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
        "nuisance_stack_table": nuisance,
        "per_galaxy": report_records,
        "featured": [row["index"] for row in featured],
    }
    write_json(report_dir / "report.json", payload)
    write_html(
        output,
        metadata=payload["metadata"],
        figures=figures,
        records=report_records,
        parameter_stats=parameter_stats,
        ensemble_metrics=ensemble_metrics,
        featured=featured,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
