#!/usr/bin/env python3
"""16–84% shear coverage on the g01 xu3 cache, split at |g| = 0.05."""

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
    ABS_G_SPLIT,
    CACHE_ROOT,
    COVERAGE_DIR,
    G01_CASE,
    HTML_STYLE,
    NOMINAL_COVERAGE,
    abs_g,
    abs_g_split_masks,
    coverage_takeaway,
    interval_coverage,
    shear_columns,
    write_json,
)
from diagnostics.likelihood_stack_select import split_case

FIGURE_DPI = 140
SLICE_ORDER = ("full", "inner", "outer")
SLICE_LABELS = {
    "full": "full catalog",
    "inner": "|g| < 0.05",
    "outer": "|g| > 0.05",
}
POSTERIOR_ORDER = ("proposal", "tf")
POSTERIOR_LABELS = {
    "proposal": "training-prior",
    "tf": "Tully–Fisher weighted",
}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--case", default=G01_CASE)
    parser.add_argument("--report-dir", type=Path, default=COVERAGE_DIR)
    parser.add_argument("--abs-g-split", type=float, default=ABS_G_SPLIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fmt_pct(value, digits=2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}"


def fmt_pp(value, digits=2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):+.{digits}f}"


def load_coverage_catalog(cache_root: Path, case: str) -> dict:
    model, dataset = split_case(case)
    root = Path(cache_root) / model / dataset
    partitions = load_cache_partitions(root)
    names = tuple(partitions.feature_names)
    g1_idx, g2_idx = shear_columns(names)
    truth = np.asarray(load_partitioned_array(partitions, "truth"), dtype=np.float64)
    summaries = {}
    for key, array_name in (
        ("proposal", "proposal_mean_estimates"),
        ("tf", "tf_target_mean_estimates"),
    ):
        summary = np.asarray(
            load_partitioned_array(partitions, array_name),
            dtype=np.float64,
        )
        if summary.ndim != 3 or summary.shape[0] != len(truth) or summary.shape[1] != 3:
            raise ValueError(
                f"{array_name} shape {summary.shape}; expected "
                f"({len(truth)}, 3, {len(names)})"
            )
        summaries[key] = {
            "lower": np.column_stack(
                (summary[:, 0, g1_idx], summary[:, 0, g2_idx])
            ),
            "mean": np.column_stack(
                (summary[:, 1, g1_idx], summary[:, 1, g2_idx])
            ),
            "upper": np.column_stack(
                (summary[:, 2, g1_idx], summary[:, 2, g2_idx])
            ),
        }
    truth_g = np.column_stack((truth[:, g1_idx], truth[:, g2_idx]))
    return {
        "feature_names": names,
        "truth_g": truth_g,
        "summaries": summaries,
        "cache_root": str(root),
        "dataset_size": int(len(truth)),
        "g1_index": g1_idx,
        "g2_index": g2_idx,
    }


def coverage_table(
    catalog: dict,
    *,
    split: float = ABS_G_SPLIT,
) -> list[dict]:
    masks = abs_g_split_masks(catalog["truth_g"], split=split)
    truth_g = catalog["truth_g"]
    rows = []
    for posterior in POSTERIOR_ORDER:
        bounds = catalog["summaries"][posterior]
        for slice_name in SLICE_ORDER:
            mask = masks[slice_name]
            row = {
                "posterior": posterior,
                "slice": slice_name,
                "n": int(np.count_nonzero(mask)),
            }
            for axis, label in enumerate(("g1", "g2")):
                metrics = interval_coverage(
                    truth_g[mask, axis],
                    bounds["lower"][mask, axis],
                    bounds["upper"][mask, axis],
                )
                row[f"{label}_coverage"] = metrics["coverage"]
                row[f"{label}_coverage_se"] = metrics["coverage_se"]
                row[f"{label}_delta"] = metrics["delta"]
                row[f"{label}_n"] = metrics["n"]
            rows.append(row)
    return rows


def save_figure(rows: list[dict], report_dir: Path) -> dict[str, str]:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    slice_names = list(SLICE_ORDER)
    x = np.arange(len(slice_names))
    width = 0.36
    for axis, posterior in zip(axes, POSTERIOR_ORDER):
        selected = [row for row in rows if row["posterior"] == posterior]
        by_slice = {row["slice"]: row for row in selected}
        g1 = [100.0 * by_slice[name]["g1_coverage"] for name in slice_names]
        g2 = [100.0 * by_slice[name]["g2_coverage"] for name in slice_names]
        g1_se = [100.0 * by_slice[name]["g1_coverage_se"] for name in slice_names]
        g2_se = [100.0 * by_slice[name]["g2_coverage_se"] for name in slice_names]
        axis.bar(
            x - width / 2,
            g1,
            width,
            yerr=g1_se,
            color="#1f4e79",
            label="g1",
            capsize=3,
        )
        axis.bar(
            x + width / 2,
            g2,
            width,
            yerr=g2_se,
            color="#b85c38",
            label="g2",
            capsize=3,
        )
        axis.axhline(100.0 * NOMINAL_COVERAGE, color="#444444", ls="--", lw=1)
        axis.set_xticks(x)
        axis.set_xticklabels([SLICE_LABELS[name] for name in slice_names])
        axis.set_ylim(0.0, 100.0)
        axis.set_title(POSTERIOR_LABELS[posterior])
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("16th–84th coverage (%)")
    fig.tight_layout()
    path = report_dir / "coverage_split.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"coverage_split": path.name}


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    rows = []
    for row in payload["table"]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(POSTERIOR_LABELS[row['posterior']])}</td>"
            f"<td>{html.escape(SLICE_LABELS[row['slice']])}</td>"
            f"<td class='num'>{row['n']}</td>"
            f"<td class='num'>{fmt_pct(row['g1_coverage'])}</td>"
            f"<td class='num'>{fmt_pct(row['g2_coverage'])}</td>"
            f"<td class='num'>{fmt_pp(row['g1_delta'])} pp</td>"
            f"<td class='num'>{fmt_pp(row['g2_delta'])} pp</td>"
            "</tr>"
        )
    split = payload["abs_g_split"]
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Inner versus outer 16–84% coverage</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Does the 68% interval stay honest when we split on |g|?</h1>
<p class="lead">The global 16th–84th interval can sit on 68% while the inner
and outer halves of the shear box do not. This page splits the same cached
posteriors at |g| = {split:g} and asks whether coverage is the same in both
halves. Equal galaxy mass; no new sampling.</p>

<p>Inner is |g| &lt; {split:g}. Outer is |g| &gt; {split:g}. Galaxies sitting
exactly on the cut are kept in the full catalog only. The training-prior
posterior is the one the network was trained with. The Tully–Fisher weighted
posterior is the same candidates after replacing the circular-velocity prior.
A calibrated 16th–84th interval covers 68% of the truths.</p>

<figure>
<img src="{html.escape(figures["coverage_split"])}" alt="Coverage versus |g| slice">
<figcaption>Equal-mass 16th–84th coverage of true g1 and g2. The dashed line
is 68%. Error bars are binomial. Left: training-prior posterior. Right:
Tully–Fisher weighted posterior.</figcaption>
</figure>

<table>
<thead><tr>
<th>Posterior</th><th>Slice</th><th>N</th>
<th>coverage of g1</th><th>coverage of g2</th>
<th>g1 − 68%</th><th>g2 − 68%</th>
</tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>

<p>{html.escape(payload["takeaway"])}</p>
<p class="note">This is a diagnostic of local calibration. It does not replace
the multiplicative-bias tables, and it is not a catalog shear estimator.</p>
</body></html>
"""
    path = report_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def write_report(args) -> None:
    html_path = args.report_dir / "report.html"
    if html_path.exists() and not args.overwrite:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    catalog = load_coverage_catalog(args.cache_root, args.case)
    table = coverage_table(catalog, split=args.abs_g_split)
    masks = abs_g_split_masks(catalog["truth_g"], split=args.abs_g_split)
    payload = {
        "case": args.case,
        "cache_root": catalog["cache_root"],
        "dataset_size": catalog["dataset_size"],
        "abs_g_split": float(args.abs_g_split),
        "nominal_coverage": NOMINAL_COVERAGE,
        "n_full": int(np.count_nonzero(masks["full"])),
        "n_inner": int(np.count_nonzero(masks["inner"])),
        "n_outer": int(np.count_nonzero(masks["outer"])),
        "n_on_cut": int(
            np.count_nonzero(masks["full"] & ~masks["inner"] & ~masks["outer"])
        ),
        "median_abs_g": float(np.median(abs_g(
            catalog["truth_g"][masks["full"], 0],
            catalog["truth_g"][masks["full"], 1],
        ))),
        "table": table,
        "takeaway": coverage_takeaway(table),
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    figures = save_figure(table, args.report_dir)
    payload["figures"] = figures
    write_html(payload, figures, args.report_dir)
    write_json(args.report_dir / "report.json", payload)
    print(f"Wrote {args.report_dir / 'report.json'}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    write_report(args)


if __name__ == "__main__":
    main()
