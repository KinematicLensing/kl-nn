#!/usr/bin/env python3
"""Select 128 large-Mean-shrinkage galaxies for the shared-η stack campaign."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import sys
from pathlib import Path

import numpy as np

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from cache_contract import load_cache_partitions, load_partitioned_array
from diagnostics.likelihood_stack_core import (
    ABS_G_MIN,
    CACHE_ROOT,
    G01_CASE,
    N_SELECTED,
    SELECT_DIR,
    HTML_STYLE,
    abs_g,
    dumps_meta,
    galaxies_npz,
    loads_meta,
    select_large_shrinkage,
    shear_columns,
    shrinkage_along_truth,
    write_json,
)
from utils import resolve_feature_index

FIGURE_DPI = 140
NUISANCE_KEYS = ("cosi", "vcirc", "hlr", "rscale")


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--case", default=G01_CASE)
    parser.add_argument("--report-dir", type=Path, default=SELECT_DIR)
    parser.add_argument("--n-selected", type=int, default=N_SELECTED)
    parser.add_argument("--abs-g-min", type=float, default=ABS_G_MIN)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-galaxies",
        action="store_true",
        help="Rebuild figures and HTML from galaxies.npz without rereading the cache",
    )
    return parser.parse_args(argv)


def split_case(case: str) -> tuple[str, str]:
    try:
        model, dataset = case.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Case must be MODEL:DATASET, got {case!r}") from exc
    return model, dataset


def load_proposal_catalog(cache_root: Path, case: str) -> dict:
    model, dataset = split_case(case)
    root = Path(cache_root) / model / dataset
    partitions = load_cache_partitions(root)
    names = tuple(partitions.feature_names)
    g1_idx, g2_idx = shear_columns(names)
    truth = np.asarray(load_partitioned_array(partitions, "truth"), dtype=np.float64)
    summary = np.asarray(
        load_partitioned_array(partitions, "proposal_mean_estimates"),
        dtype=np.float64,
    )
    if summary.ndim != 3 or summary.shape[0] != len(truth) or summary.shape[1] != 3:
        raise ValueError(
            f"proposal_mean_estimates shape {summary.shape}; expected "
            f"({len(truth)}, 3, {len(names)})"
        )
    mean = summary[:, 1, :]
    columns = {}
    for name in names:
        index = resolve_feature_index(names, name)
        columns[name] = truth[:, index]
    return {
        "feature_names": names,
        "index": np.arange(len(truth), dtype=np.int64),
        "truth": truth,
        "mean": mean,
        "truth_g": np.column_stack((truth[:, g1_idx], truth[:, g2_idx])),
        "mean_g": np.column_stack((mean[:, g1_idx], mean[:, g2_idx])),
        "columns": columns,
        "g1_index": g1_idx,
        "g2_index": g2_idx,
        "cache_root": str(root),
        "dataset_size": int(len(truth)),
    }


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figures(parent: dict, selected: dict, report_dir: Path) -> dict[str, str]:
    plt = _setup_matplotlib()
    figures = {}
    parent_g = parent["truth_g"]
    selected_g = selected["truth_g"]
    parent_mean = parent["mean_g"]
    selected_mean = selected["mean_g"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for axis, component in zip(axes, (0, 1)):
        axis.scatter(
            parent_g[:, component],
            parent_mean[:, component],
            s=8,
            color="#b8c0c8",
            alpha=0.25,
            linewidths=0,
            label="parent catalog",
        )
        axis.scatter(
            selected_g[:, component],
            selected_mean[:, component],
            s=22,
            color="#1f4e79",
            alpha=0.9,
            label="selected",
        )
        lo = float(np.min(parent_g[:, component]))
        hi = float(np.max(parent_g[:, component]))
        pad = 0.05 * max(hi - lo, 0.02)
        axis.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#bbbbbb", lw=1)
        name = "g1" if component == 0 else "g2"
        axis.set_xlabel(f"true {name}")
        axis.set_ylabel(f"cached Mean {name}")
        axis.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    path = report_dir / "mean_vs_truth.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["mean_vs_truth"] = path.name

    hist_specs = (
        ("abs_g", "true shear amplitude", abs_g(parent_g[:, 0], parent_g[:, 1]),
         abs_g(selected_g[:, 0], selected_g[:, 1])),
        ("cosi", "cosine of inclination", parent["columns"]["cosi"], selected["columns"]["cosi"]),
        ("vcirc", "circular velocity (km s$^{-1}$)", parent["columns"]["vcirc"],
         selected["columns"]["vcirc"]),
        ("hlr", "half-light radius (arcsec)", parent["columns"]["hlr"], selected["columns"]["hlr"]),
        ("rscale", "scale radius (arcsec)", parent["columns"]["rscale"],
         selected["columns"]["rscale"]),
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.4))
    axes = axes.ravel()
    for axis, (key, xlabel, parent_values, selected_values) in zip(axes, hist_specs):
        axis.hist(
            parent_values,
            bins=24,
            density=True,
            histtype="step",
            color="#8a939c",
            lw=1.5,
            label="parent catalog",
        )
        axis.hist(
            selected_values,
            bins=18,
            density=True,
            histtype="step",
            color="#1f4e79",
            lw=1.8,
            label="selected",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("density")
        figures[key] = key
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.tight_layout()
    path = report_dir / "nuisance_histograms.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["nuisance_histograms"] = path.name
    return figures


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "p16": None, "p84": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p16": float(np.percentile(finite, 16)),
        "p84": float(np.percentile(finite, 84)),
    }


def fmt(value, digits=3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    selected = payload["selected_summary"]
    parent = payload["parent_summary"]
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Galaxies with a large Mean shrinkage</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Which galaxies does the Mean pull hardest toward zero?</h1>
<p class="lead">This page only chooses a sample. It does not combine noise
realizations, and it does not replace the catalog multiplicative-bias tables.
We start from the cached posterior Mean of the frozen network trained out to
shear 0.1, keep galaxies whose true shear amplitude is above 0.02, and draw
128 objects from the more-shrunk half of that set.</p>

<p>Shrinkage along the true shear is
<code>(ĝ − g) · g / |g|²</code>. Negative values mean the Mean is pulled
toward zero. The parent catalog still has its original nuisance parameters;
we do not match inclination, size, or rotation speed. The 128 are a random
draw from the biased half, so those nuisances come along as they are.</p>

<table>
<thead><tr>
<th></th><th>N</th><th>median |g|</th><th>median shrinkage</th>
<th>median cos i</th><th>median v<sub>circ</sub></th>
</tr></thead>
<tbody>
<tr><td>Parent, |g| &gt; 0.02</td>
<td class="num">{parent["n"]}</td>
<td class="num">{fmt(parent["abs_g"]["median"])}</td>
<td class="num">{fmt(parent["shrinkage"]["median"])}</td>
<td class="num">{fmt(parent["cosi"]["median"])}</td>
<td class="num">{fmt(parent["vcirc"]["median"])}</td></tr>
<tr><td>Biased half</td>
<td class="num">{payload["n_pool"]}</td>
<td class="num">{fmt(payload["pool_abs_g"]["median"])}</td>
<td class="num">{fmt(payload["pool_shrinkage"]["median"])}</td>
<td class="num">{fmt(payload["pool_cosi"]["median"])}</td>
<td class="num">{fmt(payload["pool_vcirc"]["median"])}</td></tr>
<tr><td>Selected</td>
<td class="num">{selected["n"]}</td>
<td class="num">{fmt(selected["abs_g"]["median"])}</td>
<td class="num">{fmt(selected["shrinkage"]["median"])}</td>
<td class="num">{fmt(selected["cosi"]["median"])}</td>
<td class="num">{fmt(selected["vcirc"]["median"])}</td></tr>
</tbody>
</table>

<figure>
<img src="{html.escape(figures["mean_vs_truth"])}" alt="Cached Mean versus true shear">
<figcaption>Cached posterior Mean against truth. Grey is the parent catalog
with |g| above 0.02. Navy is the 128 galaxies that go on to the noise-stack
page. A slope of one would mean the Mean tracks the truth.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures["nuisance_histograms"])}" alt="Nuisance histograms">
<figcaption>True shear amplitude and four nuisance parameters. Densities, not
counts. The selected set is a random slice of the biased half, not a cut on
inclination or size.</figcaption>
</figure>

<p class="note">The next stage redraws independent noise on these 128 stamps
and compares a shared-parameter likelihood stack to an inverse-variance mean
of Means. That is a diagnostic of the estimator, not a catalog shear.</p>
</body></html>
"""
    path = report_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def save_galaxies(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "index": payload["index"],
        "truth": payload["truth"],
        "mean": payload["mean"],
        "truth_g": payload["truth_g"],
        "mean_g": payload["mean_g"],
        "shrinkage": payload["shrinkage"],
        "feature_names": np.asarray(payload["feature_names"]),
        "meta_json": dumps_meta(payload["meta"]),
    }
    parent = payload.get("parent")
    if parent is not None:
        arrays["parent_truth_g"] = parent["truth_g"]
        arrays["parent_mean_g"] = parent["mean_g"]
        arrays["parent_shrinkage"] = shrinkage_along_truth(
            parent["mean_g"], parent["truth_g"]
        )
        for key in NUISANCE_KEYS:
            arrays[f"parent_{key}"] = parent["columns"][key]
    np.savez_compressed(path, **arrays)


def load_galaxies(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    names = tuple(str(name) for name in np.asarray(data["feature_names"]).tolist())
    columns = {
        name: np.asarray(data["truth"][:, resolve_feature_index(names, name)], dtype=np.float64)
        for name in names
    }
    payload = {
        "index": np.asarray(data["index"], dtype=np.int64),
        "truth": np.asarray(data["truth"], dtype=np.float64),
        "mean": np.asarray(data["mean"], dtype=np.float64),
        "truth_g": np.asarray(data["truth_g"], dtype=np.float64),
        "mean_g": np.asarray(data["mean_g"], dtype=np.float64),
        "shrinkage": np.asarray(data["shrinkage"], dtype=np.float64),
        "feature_names": names,
        "columns": columns,
        "meta": loads_meta(data["meta_json"]),
    }
    if "parent_truth_g" in data.files:
        payload["parent"] = {
            "truth_g": np.asarray(data["parent_truth_g"], dtype=np.float64),
            "mean_g": np.asarray(data["parent_mean_g"], dtype=np.float64),
            "shrinkage": np.asarray(data["parent_shrinkage"], dtype=np.float64),
            "columns": {
                key: np.asarray(data[f"parent_{key}"], dtype=np.float64)
                for key in NUISANCE_KEYS
            },
        }
    return payload


def summarize_subset(
    truth_g: np.ndarray,
    shrinkage: np.ndarray,
    columns: dict[str, np.ndarray],
) -> dict:
    return {
        "n": int(len(truth_g)),
        "abs_g": distribution_summary(abs_g(truth_g[:, 0], truth_g[:, 1])),
        "shrinkage": distribution_summary(shrinkage),
        "cosi": distribution_summary(columns["cosi"]),
        "vcirc": distribution_summary(columns["vcirc"]),
        "hlr": distribution_summary(columns["hlr"]),
        "rscale": distribution_summary(columns["rscale"]),
    }


def assemble_payload(
    catalog: dict,
    choice: dict,
    args,
) -> tuple[dict, dict, dict]:
    selected = choice["selected"]
    pool = choice["pool"]
    eligible = choice["eligible"]
    selected_payload = {
        "index": catalog["index"][selected],
        "truth": catalog["truth"][selected],
        "mean": catalog["mean"][selected],
        "truth_g": catalog["truth_g"][selected],
        "mean_g": catalog["mean_g"][selected],
        "shrinkage": choice["shrinkage"][selected],
        "feature_names": catalog["feature_names"],
        "columns": {
            name: values[selected] for name, values in catalog["columns"].items()
        },
    }
    parent_mask = eligible
    parent = {
        "truth_g": catalog["truth_g"][parent_mask],
        "mean_g": catalog["mean_g"][parent_mask],
        "columns": {
            name: values[parent_mask] for name, values in catalog["columns"].items()
        },
    }
    meta = {
        "case": args.case,
        "cache_root": catalog["cache_root"],
        "dataset_size": catalog["dataset_size"],
        "n_eligible": choice["n_eligible"],
        "n_pool": choice["n_pool"],
        "n_selected": choice["n_selected"],
        "abs_g_min": choice["abs_g_min"],
        "seed": choice["seed"],
        "median_eligible_shrinkage": choice["median_eligible_shrinkage"],
        "feature_names": list(catalog["feature_names"]),
    }
    selected_payload["meta"] = meta
    selected_payload["parent"] = parent
    report = {
        **meta,
        "parent_summary": summarize_subset(
            catalog["truth_g"][eligible],
            choice["shrinkage"][eligible],
            {name: values[eligible] for name, values in catalog["columns"].items()},
        ),
        "pool_abs_g": distribution_summary(
            abs_g(catalog["truth_g"][pool, 0], catalog["truth_g"][pool, 1])
        ),
        "pool_shrinkage": distribution_summary(choice["shrinkage"][pool]),
        "pool_cosi": distribution_summary(catalog["columns"]["cosi"][pool]),
        "pool_vcirc": distribution_summary(catalog["columns"]["vcirc"][pool]),
        "selected_summary": summarize_subset(
            selected_payload["truth_g"],
            selected_payload["shrinkage"],
            selected_payload["columns"],
        ),
        "n_pool": choice["n_pool"],
    }
    return selected_payload, parent, report


def write_outputs(selected: dict, parent: dict, report: dict, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = save_figures(parent, selected, report_dir)
    report = {**report, "figures": figures}
    write_html(report, figures, report_dir)
    write_json(report_dir / "report.json", report)
    print(f"Wrote {report_dir / 'report.json'}", flush=True)


def write_report(args) -> None:
    html_path = args.report_dir / "report.html"
    galaxies_path = galaxies_npz(args.report_dir)
    if html_path.exists() and not args.overwrite and not args.reuse_galaxies:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    if args.reuse_galaxies:
        selected = load_galaxies(galaxies_path)
        parent = selected.get("parent")
        if parent is None:
            parent = {
                "truth_g": selected["truth_g"],
                "mean_g": selected["mean_g"],
                "shrinkage": selected["shrinkage"],
                "columns": selected["columns"],
            }
        parent_shrinkage = parent.get("shrinkage")
        if parent_shrinkage is None:
            parent_shrinkage = shrinkage_along_truth(parent["mean_g"], parent["truth_g"])
        report = {
            **selected["meta"],
            "parent_summary": summarize_subset(
                parent["truth_g"], parent_shrinkage, parent["columns"]
            ),
            "pool_abs_g": summarize_subset(
                selected["truth_g"], selected["shrinkage"], selected["columns"]
            )["abs_g"],
            "pool_shrinkage": summarize_subset(
                selected["truth_g"], selected["shrinkage"], selected["columns"]
            )["shrinkage"],
            "pool_cosi": summarize_subset(
                selected["truth_g"], selected["shrinkage"], selected["columns"]
            )["cosi"],
            "pool_vcirc": summarize_subset(
                selected["truth_g"], selected["shrinkage"], selected["columns"]
            )["vcirc"],
            "selected_summary": summarize_subset(
                selected["truth_g"], selected["shrinkage"], selected["columns"]
            ),
            "n_pool": selected["meta"].get("n_pool", selected["meta"].get("n_selected")),
        }
        write_outputs(selected, parent, report, args.report_dir)
        return

    catalog = load_proposal_catalog(args.cache_root, args.case)
    choice = select_large_shrinkage(
        catalog["mean_g"],
        catalog["truth_g"],
        n=args.n_selected,
        abs_g_min=args.abs_g_min,
        seed=args.seed,
    )
    print(
        f"eligible={choice['n_eligible']} pool={choice['n_pool']} "
        f"selected={choice['n_selected']} median_s={choice['median_eligible_shrinkage']:.4f}",
        flush=True,
    )
    selected, parent, report = assemble_payload(catalog, choice, args)
    save_galaxies(galaxies_path, selected)
    print(f"Wrote {galaxies_path}", flush=True)
    write_outputs(selected, parent, report, args.report_dir)


def main(argv=None) -> None:
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
