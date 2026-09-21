#!/usr/bin/env python3
"""xu3 NRE MAP, 16–84% coverage, and Bernstein stacks versus cached NPE Means."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import sys
from pathlib import Path

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from diagnostics.likelihood_stack_core import (
    ABS_G_SPLIT,
    CACHE_ROOT,
    DATA_ROOT,
    G01_CASE,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    NOMINAL_COVERAGE,
    NRE_DIR,
    NRE_GRID_N,
    NRE_INFER_DATASET,
    NRE_NAME,
    NRE_PREFIXES,
    abs_g,
    interval_coverage,
    write_json,
)
from diagnostics.likelihood_stack_coverage import load_coverage_catalog
from diagnostics.likelihood_stack_nre_core import (
    AMP_SLICE_NAMES,
    SLICE_LABELS,
    RatioHead,
    component_metrics,
    grid_posterior_moments,
    grid_marginal_intervals,
    grouped_mean,
    grouped_stack_map,
    nre_slice_masks,
    nre_takeaway,
    normalized_shear_grid,
    prefix_groups,
    score_shear_grid,
)
from diagnostics.likelihood_stack_nre_train import (
    extract_contexts,
    load_frozen_parent,
    load_ratio_head,
    nre_checkpoint_path,
    refuse_parent_overwrite,
)
from utils import denormalize

FIGURE_DPI = 140
GALAXY_SLICES = ("full", "inner", "outer")
STACK_SLICES = GALAXY_SLICES + AMP_SLICE_NAMES
ESTIMATOR_LABELS = {
    "nre_map": "NRE peak",
    "nre_mean": "NRE posterior mean",
    "npe_mean": "NPE Mean",
    "nre_stack_map": "stacked NRE peak",
    "nre_mean_of_means": "mean of NRE posterior means",
    "npe_mean_of_means": "mean of Means",
}


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--parent-checkpoint-suffix", default="best")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_ROOT / NRE_INFER_DATASET
    )
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--case", default=G01_CASE)
    parser.add_argument("--report-dir", type=Path, default=NRE_DIR)
    parser.add_argument("--grid-n", type=int, default=NRE_GRID_N)
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=list(NRE_PREFIXES)
    )
    parser.add_argument("--abs-g-split", type=float, default=ABS_G_SPLIT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fmt(value, digits=3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def fmt_pct(value, digits=2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * float(value):.{digits}f}"


def physical_shear_grid(n: int, par_ranges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_norm, axis_norm = normalized_shear_grid(n)
    grid_phys = np.asarray(
        denormalize(grid_norm, par_ranges, feature_names=("g1", "g2")),
        dtype=np.float64,
    )
    axis_dummy = np.stack((axis_norm, np.zeros_like(axis_norm)), axis=1)
    axis_phys = np.asarray(
        denormalize(axis_dummy, par_ranges, feature_names=("g1", "g2")),
        dtype=np.float64,
    )[:, 0]
    return grid_norm, grid_phys, axis_phys


@torch.inference_mode()
def score_all_logits(
    head: RatioHead,
    context: torch.Tensor,
    grid_norm: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    grid = torch.as_tensor(grid_norm, device=device, dtype=torch.float32)
    n = int(context.shape[0])
    out = np.empty((n, grid.shape[0]), dtype=np.float32)
    head.eval()
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        logits = score_shear_grid(head, context[start:stop].to(device), grid)
        out[start:stop] = logits.float().cpu().numpy()
    return out


def maps_from_logits(log_r: np.ndarray, grid_phys: np.ndarray) -> np.ndarray:
    index = np.argmax(np.where(np.isfinite(log_r), log_r, -np.inf), axis=1)
    return np.asarray(grid_phys[index], dtype=np.float64)


def galaxy_table(truth_g, estimates: dict[str, np.ndarray], masks) -> list[dict]:
    rows = []
    for name, values in estimates.items():
        for slice_name in GALAXY_SLICES:
            mask = masks[slice_name]
            row = {
                "estimator": name,
                "slice": slice_name,
                **component_metrics(truth_g[mask], values[mask]),
            }
            rows.append(row)
    return rows


def coverage_table(truth_g, intervals: dict[str, dict], masks) -> list[dict]:
    rows = []
    for estimator, bounds in intervals.items():
        for slice_name in GALAXY_SLICES:
            mask = masks[slice_name]
            row = {
                "estimator": estimator,
                "slice": slice_name,
                "n": int(np.count_nonzero(mask)),
            }
            for axis in ("g1", "g2"):
                metrics = interval_coverage(
                    truth_g[mask, 0 if axis == "g1" else 1],
                    bounds[f"{axis}_lower"][mask],
                    bounds[f"{axis}_upper"][mask],
                )
                row[f"{axis}_coverage"] = metrics["coverage"]
                row[f"{axis}_coverage_se"] = metrics["coverage_se"]
                row[f"{axis}_delta"] = metrics["delta"]
            rows.append(row)
    return rows


def stack_table(
    truth_g: np.ndarray,
    nre_log_r: np.ndarray,
    npe_mean: np.ndarray,
    grid_phys: np.ndarray,
    masks: dict[str, np.ndarray],
    prefixes: tuple[int, ...],
    *,
    seed: int,
    nre_mean: np.ndarray | None = None,
) -> list[dict]:
    rows = []
    for slice_name in STACK_SLICES:
        members = np.flatnonzero(masks[slice_name])
        if members.size == 0:
            continue
        slice_truth = truth_g[members]
        slice_log_r = nre_log_r[members]
        slice_mean = npe_mean[members]
        slice_nre_mean = None if nre_mean is None else nre_mean[members]
        for count in prefixes:
            groups = prefix_groups(len(members), int(count), seed=seed)
            if groups.shape[0] < 2:
                continue
            truth_hat = grouped_mean(slice_truth, groups)
            nre_hat = grouped_stack_map(slice_log_r, grid_phys, groups)
            mean_hat = grouped_mean(slice_mean, groups)
            estimates = [
                ("nre_stack_map", nre_hat),
                ("npe_mean_of_means", mean_hat),
            ]
            if slice_nre_mean is not None:
                estimates.insert(
                    1,
                    ("nre_mean_of_means", grouped_mean(slice_nre_mean, groups)),
                )
            for estimator, estimate in estimates:
                row = {
                    **component_metrics(truth_hat, estimate),
                    "estimator": estimator,
                    "slice": slice_name,
                    "stack_size": int(count),
                    "n_groups": int(len(groups)),
                }
                rows.append(row)
    return rows


def save_figures(payload: dict, report_dir: Path) -> dict[str, str]:
    plt = _setup_matplotlib()
    truth = np.asarray(payload["truth_g"])
    nre_map = np.asarray(payload["nre_map"])
    nre_mean = np.asarray(payload["nre_mean"])
    npe_mean = np.asarray(payload["npe_mean"])
    rng = np.random.default_rng(payload["seed"])
    if len(truth) > 8000:
        shown = np.sort(rng.choice(len(truth), size=8000, replace=False))
    else:
        shown = np.arange(len(truth))
    limit = 0.12
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), sharex=True, sharey=True)
    panels = (
        (axes[0, 0], nre_map, "NRE peak, g1"),
        (axes[0, 1], nre_mean, "NRE posterior mean, g1"),
        (axes[0, 2], npe_mean, "NPE Mean, g1"),
        (axes[1, 0], nre_map, "NRE peak, g2"),
        (axes[1, 1], nre_mean, "NRE posterior mean, g2"),
        (axes[1, 2], npe_mean, "NPE Mean, g2"),
    )
    for axis, estimate, title in panels:
        component = 0 if title.endswith("g1") else 1
        axis.plot([-limit, limit], [-limit, limit], color="#888", lw=1)
        axis.scatter(
            truth[shown, component],
            estimate[shown, component],
            s=6,
            alpha=0.22,
            linewidths=0,
            color="#1f4e79",
        )
        axis.set_title(title)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal", adjustable="box")
    for axis in axes[1]:
        axis.set_xlabel("true shear")
    axes[0, 0].set_ylabel("estimated shear")
    axes[1, 0].set_ylabel("estimated shear")
    fig.tight_layout()
    scatter_name = "nre_vs_mean_scatter.png"
    fig.savefig(report_dir / scatter_name, dpi=FIGURE_DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True)
    for axis, slice_name, title in (
        (axes[0], "inner", "|g| < 0.05"),
        (axes[1], "outer", "|g| > 0.05"),
    ):
        for estimator, color, label in (
            ("nre_stack_map", "#1f4e79", "stacked NRE peak"),
            ("nre_mean_of_means", "#2f855a", "mean of NRE posterior means"),
            ("npe_mean_of_means", "#a03c3c", "mean of Means"),
        ):
            rows = [
                row
                for row in payload["stack_table"]
                if row["estimator"] == estimator and row["slice"] == slice_name
            ]
            rows = sorted(rows, key=lambda row: int(row["stack_size"]))
            if not rows:
                continue
            ns = [int(row["stack_size"]) for row in rows]
            axis.plot(
                ns,
                [row["g1_m"] for row in rows],
                marker="o",
                color=color,
                label=f"{label}, g1",
            )
            axis.plot(
                ns,
                [row["g2_m"] for row in rows],
                marker="s",
                linestyle="--",
                color=color,
                label=f"{label}, g2",
            )
        axis.axhline(0.0, color="#888", lw=1)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel("galaxies in the stack")
        axis.set_xticks(list(payload["prefixes"]))
        axis.set_xticklabels([str(n) for n in payload["prefixes"]])
    axes[0].set_ylabel("m (zero is unbiased slope)")
    axes[1].legend(frameon=False, loc="best", fontsize="small")
    fig.tight_layout()
    stack_name = "stack_m_vs_n.png"
    fig.savefig(report_dir / stack_name, dpi=FIGURE_DPI)
    plt.close(fig)
    return {"scatter": scatter_name, "stack_m": stack_name}


def write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    galaxy_rows = []
    for row in payload["galaxy_table"]:
        galaxy_rows.append(
            "<tr>"
            f"<td>{html.escape(ESTIMATOR_LABELS[row['estimator']])}</td>"
            f"<td>{html.escape(SLICE_LABELS[row['slice']])}</td>"
            f"<td class='num'>{row['n']}</td>"
            f"<td class='num'>{fmt(row['g1_m'])}</td>"
            f"<td class='num'>{fmt(row['g2_m'])}</td>"
            f"<td class='num'>{fmt(row['g1_c'], 4)}</td>"
            f"<td class='num'>{fmt(row['g2_c'], 4)}</td>"
            "</tr>"
        )
    coverage_rows = []
    coverage_labels = {
        "nre": "NRE 16–84% on the grid",
        "npe_proposal": "cached NPE Mean 16–84%",
    }
    for row in payload["coverage_table"]:
        coverage_rows.append(
            "<tr>"
            f"<td>{html.escape(coverage_labels[row['estimator']])}</td>"
            f"<td>{html.escape(SLICE_LABELS[row['slice']])}</td>"
            f"<td class='num'>{row['n']}</td>"
            f"<td class='num'>{fmt_pct(row['g1_coverage'])}</td>"
            f"<td class='num'>{fmt_pct(row['g2_coverage'])}</td>"
            "</tr>"
        )
    stack_rows = []
    for row in payload["stack_table"]:
        stack_rows.append(
            "<tr>"
            f"<td>{html.escape(ESTIMATOR_LABELS[row['estimator']])}</td>"
            f"<td>{html.escape(SLICE_LABELS[row['slice']])}</td>"
            f"<td class='num'>{row['stack_size']}</td>"
            f"<td class='num'>{row['n_groups']}</td>"
            f"<td class='num'>{fmt(row['g1_m'])}</td>"
            f"<td class='num'>{fmt(row['g2_m'])}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Frozen-encoder NRE peak and posterior mean versus the Mean</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Does a shear likelihood head on the frozen encoder undo the Mean shrinkage?</h1>
<p class="lead">The network that produces the catalog Mean was not retrained.
A small classifier on its frozen features learns how much more likely each
stamp is at one shear than at another. This page compares its per-galaxy
peak and posterior mean with the Mean, then shows a grouped-product stress
test inside broad |g| slices.</p>

<p>Training used the same 100k catalog the parent network saw, with the
convolutional encoder held fixed. Inference is the xu3 test catalog in the
±0.1 shear box. Each stamp is encoded once. The shear plane is scored on a
21×21 grid. The peak is the NRE MAP; weighting every grid point by exp(log-r)
gives the NRE posterior mean. The Mean control is the cached training-prior
Mean; it is not recomputed.</p>

<figure>
<img src="{html.escape(figures['scatter'])}" alt="NRE peak and posterior mean versus NPE Mean">
<figcaption>Eight thousand random galaxies. Each row compares the NRE MAP,
NRE posterior mean, and cached NPE Mean. Top: g1. Bottom: g2. The line is
truth. The posterior mean is included as a lower-variance alternative to the
quantized MAP.</figcaption>
</figure>

<table>
<thead><tr>
<th>Estimator</th><th>Slice</th><th>N</th>
<th>m of g1</th><th>m of g2</th>
<th>c of g1</th><th>c of g2</th>
</tr></thead>
<tbody>
{"".join(galaxy_rows)}
</tbody>
</table>

<table>
<thead><tr>
<th>Interval</th><th>Slice</th><th>N</th>
<th>coverage of g1</th><th>coverage of g2</th>
</tr></thead>
<tbody>
{"".join(coverage_rows)}
</tbody>
</table>

<figure>
<img src="{html.escape(figures['stack_m'])}" alt="Stack m versus number of galaxies">
<figcaption>Galaxies in the same broad |g| slice are shuffled once (seed 42)
and split into groups of N. Blue adds log-ratios as if they shared one shear;
green averages NRE posterior means; red averages cached Means. Circles are
g1 and dashed squares are g2. Because a |g| slice does not fix shear
direction, this remains a grouped-product stress test, not a valid
common-shear likelihood.</figcaption>
</figure>

<table>
<thead><tr>
<th>Estimator</th><th>Slice</th><th>galaxies per group</th><th>groups</th>
<th>m of g1</th><th>m of g2</th>
</tr></thead>
<tbody>
{"".join(stack_rows)}
</tbody>
</table>

<p>{html.escape(payload["takeaway"])}</p>
<p class="note">Diagnostic only. The parent NPE weights, FITS, and LMDBs were
not modified. This is not a catalog shear estimator and does not replace the
multiplicative-bias tables. Identity noise; no hopping; shear box ±0.1.</p>
</body></html>
"""
    path = report_dir / "report.html"
    path.write_text(body, encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def write_report(args) -> None:
    html_path = args.report_dir / "report.html"
    if html_path.exists() and not args.overwrite:
        raise FileExistsError(f"{html_path} exists; use --overwrite")
    refuse_parent_overwrite(args.model_root, args.parent_npe, args.nre_name)
    if args.nre_name == args.parent_npe:
        raise ValueError("nre-name must differ from the parent NPE")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    prefixes = tuple(int(value) for value in args.prefixes)

    encoder, model_config, parent_ckpt = load_frozen_parent(
        parent_npe=args.parent_npe,
        model_root=args.model_root,
        checkpoint_suffix=args.parent_checkpoint_suffix,
        device=device,
    )
    names = tuple(config.TARGET_NAMES)
    config.require_matching_dataset_par_ranges(args.data_dir, config.par_ranges)
    checkpoint = nre_checkpoint_path(args.model_root, args.nre_name)
    head, nre_meta = load_ratio_head(
        checkpoint, device=device, expected_parent=args.parent_npe
    )
    catalog = load_coverage_catalog(args.cache_root, args.case)
    dataset = pxt.TorchDataset(str(args.data_dir))
    if len(dataset) != catalog["dataset_size"]:
        raise ValueError(
            f"LMDB size {len(dataset)} != cache size {catalog['dataset_size']}"
        )
    indices = np.arange(len(dataset), dtype=np.int64)
    channels_last = bool(model_config.train.channels_last)
    print(f"encoding xu3 n={len(indices)}", flush=True)
    context, shear_norm = extract_contexts(
        encoder,
        dataset,
        indices,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        channels_last=channels_last,
        names=names,
        split_name="xu3",
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    truth_from_lmdb = np.asarray(
        denormalize(shear_norm.numpy(), config.par_ranges, feature_names=("g1", "g2")),
        dtype=np.float64,
    )
    truth_g = catalog["truth_g"]
    if truth_from_lmdb.shape != truth_g.shape or not np.allclose(
        truth_from_lmdb, truth_g, rtol=0.0, atol=1e-5
    ):
        raise ValueError("LMDB shear truth does not match the cached NPE catalog")
    npe_mean = catalog["summaries"]["proposal"]["mean"]
    grid_norm, grid_phys, axis_phys = physical_shear_grid(args.grid_n, config.par_ranges)
    print(f"scoring {args.grid_n}x{args.grid_n} shear grid", flush=True)
    log_r = score_all_logits(
        head,
        context,
        grid_norm,
        batch_size=args.batch_size,
        device=device,
    )
    nre_map = maps_from_logits(log_r, grid_phys)
    nre_moments = grid_posterior_moments(log_r, grid_phys)
    nre_mean = nre_moments["mean"]
    nre_intervals = grid_marginal_intervals(log_r, axis_phys, n_grid=args.grid_n)
    npe_intervals = {
        "g1_lower": catalog["summaries"]["proposal"]["lower"][:, 0],
        "g1_upper": catalog["summaries"]["proposal"]["upper"][:, 0],
        "g2_lower": catalog["summaries"]["proposal"]["lower"][:, 1],
        "g2_upper": catalog["summaries"]["proposal"]["upper"][:, 1],
    }
    masks = nre_slice_masks(truth_g, split=args.abs_g_split)
    galaxy_rows = galaxy_table(
        truth_g,
        {"nre_map": nre_map, "nre_mean": nre_mean, "npe_mean": npe_mean},
        masks,
    )
    coverage_rows = coverage_table(
        truth_g,
        {"nre": nre_intervals, "npe_proposal": npe_intervals},
        masks,
    )
    stack_rows = stack_table(
        truth_g,
        log_r.astype(np.float64, copy=False),
        npe_mean,
        grid_phys,
        masks,
        prefixes,
        seed=args.seed,
        nre_mean=nre_mean,
    )
    args.report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.report_dir / "maps.npz",
        truth_g=truth_g,
        nre_map=nre_map,
        nre_mean=nre_mean,
        nre_std=nre_moments["std"],
        npe_mean=npe_mean,
        nre_g1_lower=nre_intervals["g1_lower"],
        nre_g1_upper=nre_intervals["g1_upper"],
        nre_g2_lower=nre_intervals["g2_lower"],
        nre_g2_upper=nre_intervals["g2_upper"],
    )
    payload = {
        "parent_npe": args.parent_npe,
        "parent_checkpoint": str(parent_ckpt),
        "nre_name": args.nre_name,
        "nre_checkpoint": str(checkpoint),
        "nre_meta": nre_meta,
        "data_dir": str(args.data_dir),
        "cache_root": catalog["cache_root"],
        "case": args.case,
        "dataset_size": int(len(truth_g)),
        "grid_n": int(args.grid_n),
        "prefixes": list(prefixes),
        "abs_g_split": float(args.abs_g_split),
        "seed": int(args.seed),
        "nominal_coverage": NOMINAL_COVERAGE,
        "galaxy_table": galaxy_rows,
        "coverage_table": coverage_rows,
        "stack_table": stack_rows,
        "takeaway": nre_takeaway(galaxy_rows, stack_rows),
        "truth_g": truth_g,
        "nre_map": nre_map,
        "nre_mean": nre_mean,
        "npe_mean": npe_mean,
    }
    figures = save_figures(payload, args.report_dir)
    payload["figures"] = figures
    write_html(payload, figures, args.report_dir)
    stored = dict(payload)
    stored.pop("truth_g", None)
    stored.pop("nre_map", None)
    stored.pop("nre_mean", None)
    stored.pop("npe_mean", None)
    write_json(args.report_dir / "report.json", stored)
    print(f"Wrote {args.report_dir / 'report.json'}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    write_report(args)


if __name__ == "__main__":
    main()
