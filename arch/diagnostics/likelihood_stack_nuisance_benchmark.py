#!/usr/bin/env python3
"""Benchmark six estimators while stacking only nuisance galaxies."""

from __future__ import annotations

from argparse import ArgumentParser
import html
import json
from pathlib import Path
import sys

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from cache_posteriors import physical_log_prob_from_normalized
from diagnostics.likelihood_stack_benchmark import (
    _apply_benchmark_noise,
    _checkpoint_file,
    _flow_context,
    _load_cell,
    _setup_matplotlib,
)
from diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_DATASET,
    BENCHMARK_GRID_N,
    BENCHMARK_N_GALAXIES,
    BENCHMARK_N_CELLS,
    BENCHMARK_NOISE_SEED,
    BENCHMARK_SHEAR_VALUES,
    fit_component_mc,
    fixed_shear_grid,
    rmse,
)
from diagnostics.likelihood_stack_core import (
    DATA_ROOT,
    G01_NPE,
    MODEL_ROOT,
    NRE9D_NAME,
    NRE_NAME,
    json_safe,
    shear_columns,
    write_json,
)
from diagnostics.likelihood_stack_nre_core import (
    grid_map,
    grid_posterior_moments,
    normalized_shear_grid,
    score_shear_grid,
)
from diagnostics.likelihood_stack_nre_train import (
    load_ratio_head,
    nre_checkpoint_path,
)
from model_registry import load_model_config
from networks import KLNPE
from tf_prior import TFPrior, tf_log_prior_ratio
from train import (
    load_model,
    seed_everything,
)
from utils import denormalize


REPORT_DIR = Path(
    "/ocean/projects/phy250048p/shared/reports/likelihood-stack/"
    "04_benchmark_nuisance"
)
NUISANCE_PREFIXES = (1, 2, 4, 8, 16, 32)
REALIZATION = 0
DEFAULT_NPE_SAMPLES = 1024
FIGURE_DPI = 150
ESTIMATORS = (
    ("npe_mean", "9D NPE posterior mean"),
    ("npe_map", "9D NPE posterior MAP"),
    ("nre2d_mean", "2D NRE posterior mean"),
    ("nre2d_map", "2D NRE posterior MAP"),
    ("nre9d_mean", "9D NRE posterior mean"),
    ("nre9d_map", "9D NRE posterior MAP"),
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT / BENCHMARK_DATASET,
    )
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-2d-name", default=NRE_NAME)
    parser.add_argument("--nre-9d-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--parent-checkpoint-suffix", default="best")
    parser.add_argument("--nre-checkpoint-suffix", default="best")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--npe-samples", type=int, default=DEFAULT_NPE_SAMPLES)
    parser.add_argument("--grid-n", type=int, default=BENCHMARK_GRID_N)
    parser.add_argument("--seed", type=int, default=BENCHMARK_NOISE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _normalize_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("log_weights must have shape (galaxy, candidate)")
    finite = np.isfinite(values)
    if np.any(~np.any(finite, axis=1)):
        raise ValueError("each galaxy needs at least one finite candidate weight")
    shifted = np.where(finite, values - np.nanmax(values, axis=1, keepdims=True), -np.inf)
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def candidate_nre9d_estimators(
    samples_normalized: np.ndarray,
    nre_log_ratio: np.ndarray,
    proposal_log_density: np.ndarray,
    *,
    rmag_true: np.ndarray,
    parameter_names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    prior: TFPrior,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 9D NRE mean, MAP shear, and candidate ESS per galaxy."""

    samples = np.asarray(samples_normalized, dtype=np.float64)
    log_ratio = np.asarray(nre_log_ratio, dtype=np.float64)
    proposal = np.asarray(proposal_log_density, dtype=np.float64)
    if samples.ndim != 3:
        raise ValueError("samples_normalized must have shape (B, K, D)")
    if log_ratio.shape != samples.shape[:2]:
        raise ValueError("nre_log_ratio must match the candidate dimensions")
    if proposal.shape != samples.shape[:2]:
        raise ValueError("proposal_log_density must match the candidate dimensions")
    names = tuple(parameter_names)
    if names[:2] != ("g1", "g2") or len(names) != 9:
        raise ValueError("candidate_nre9d_estimators requires the current 9D schema")
    physical = np.asarray(
        denormalize(
            samples,
            par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        ),
        dtype=np.float64,
    )
    vcirc_index = names.index("vcirc")
    tf_ratio = tf_log_prior_ratio(
        physical[..., vcirc_index],
        np.asarray(rmag_true, dtype=np.float64)[:, None],
        prior,
    )
    target_log_density = log_ratio + tf_ratio
    log_weights = target_log_density - proposal
    weights = _normalize_weights(log_weights)
    mean = np.sum(weights[..., None] * physical[..., :2], axis=1)
    map_index = np.argmax(
        np.where(np.isfinite(target_log_density), target_log_density, -np.inf),
        axis=1,
    )
    map_estimate = physical[np.arange(len(physical)), map_index, :2]
    effective_sample_size = 1.0 / np.sum(np.square(weights), axis=1)
    return mean, map_estimate, effective_sample_size


def evaluate_cell(
    model,
    nre_2d_head,
    nre_9d_head,
    cell: dict,
    *,
    cell_id: int,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    grid_norm: np.ndarray,
    grid_phys: np.ndarray,
    npe_samples: int,
    seed: int,
    device: torch.device,
    channels_last: bool,
    tf_prior: TFPrior,
) -> dict[str, np.ndarray]:
    """Evaluate all six estimators for one fixed-shear cell."""

    images, spectra = _apply_benchmark_noise(
        cell,
        cell_id=cell_id,
        realization=REALIZATION,
        seed=seed,
        device=device,
    )
    context = _flow_context(
        model,
        images,
        spectra,
        cell,
        channels_last=channels_last,
    )
    sample_seed = int(
        np.random.SeedSequence(
            [int(seed), int(cell_id), REALIZATION, 991]
        ).generate_state(1, dtype=np.uint64)[0]
        % (2**63 - 1)
    )
    torch.manual_seed(sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(sample_seed)
    samples, normalized_log_prob = model.flow.sample_and_log_prob(
        int(npe_samples),
        context=context,
    )
    samples_np = samples.detach().cpu().numpy()
    normalized_log_prob_np = normalized_log_prob.detach().cpu().numpy()
    physical = np.asarray(
        denormalize(
            samples_np,
            par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        ),
        dtype=np.float64,
    )
    g1_index, g2_index = shear_columns(names)
    npe_mean = np.stack(
        (
            physical[..., g1_index].mean(axis=1),
            physical[..., g2_index].mean(axis=1),
        ),
        axis=-1,
    )
    proposal_log_density = physical_log_prob_from_normalized(
        samples_np,
        normalized_log_prob_np,
        par_ranges=par_ranges,
        feature_names=names,
        target_transforms=config.TARGET_TRANSFORMS,
    )
    npe_map_index = np.argmax(
        np.where(
            np.isfinite(proposal_log_density),
            proposal_log_density,
            -np.inf,
        ),
        axis=1,
    )
    npe_map = physical[
        np.arange(len(physical)),
        npe_map_index,
    ][:, [g1_index, g2_index]]

    grid_tensor = torch.as_tensor(
        grid_norm,
        device=device,
        dtype=torch.float32,
    )
    nre2d_log_ratio = score_shear_grid(
        nre_2d_head,
        context,
        grid_tensor,
    ).detach().cpu().numpy()
    nre2d_mean = grid_posterior_moments(
        nre2d_log_ratio,
        grid_phys,
    )["mean"]
    nre2d_map = np.asarray(
        [grid_map(row, grid_phys) for row in nre2d_log_ratio],
        dtype=np.float64,
    )

    batch, candidates, dimensions = samples.shape
    candidate_tensor = samples.reshape(batch * candidates, dimensions)
    candidate_context = context[:, None, :].expand(
        batch,
        candidates,
        context.shape[-1],
    ).reshape(batch * candidates, context.shape[-1])
    nre9d_log_ratio = nre_9d_head(
        candidate_context,
        candidate_tensor,
    ).reshape(batch, candidates).detach().cpu().numpy()
    nre9d_mean, nre9d_map, nre9d_ess = candidate_nre9d_estimators(
        samples_np,
        nre9d_log_ratio,
        proposal_log_density,
        rmag_true=cell["rmag"].detach().cpu().numpy(),
        parameter_names=names,
        par_ranges=par_ranges,
        prior=tf_prior,
    )
    return {
        "npe_mean": npe_mean,
        "npe_map": npe_map,
        "nre2d_mean": np.asarray(nre2d_mean, dtype=np.float64),
        "nre2d_map": nre2d_map,
        "nre9d_mean": nre9d_mean,
        "nre9d_map": nre9d_map,
        "nre9d_ess": nre9d_ess,
        "truth_g": np.asarray(
            fixed_shear_grid()[cell_id],
            dtype=np.float64,
        ).reshape(1, 2).repeat(BENCHMARK_N_GALAXIES, axis=0),
        "truth_full": np.asarray(cell["truth"], dtype=np.float64),
    }


def run_benchmark(
    dataset,
    model,
    nre_2d_head,
    nre_9d_head,
    *,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    channels_last: bool,
    device: torch.device,
    npe_samples: int,
    grid_n: int,
    seed: int,
    tf_prior: TFPrior,
) -> dict[str, np.ndarray]:
    if BENCHMARK_N_CELLS != len(fixed_shear_grid()):
        raise ValueError("fixed-shear cell count does not match the shear grid")
    grid_norm, _ = normalized_shear_grid(grid_n)
    grid_phys = np.asarray(
        denormalize(
            grid_norm,
            par_ranges,
            feature_names=("g1", "g2"),
        ),
        dtype=np.float64,
    )
    outputs = {
        name: np.empty((BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, 2))
        for name, _ in ESTIMATORS
    }
    nre9d_ess = np.empty((BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES))
    truth_full = np.empty(
        (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, len(names)),
        dtype=np.float64,
    )
    for cell_id in range(BENCHMARK_N_CELLS):
        print(f"nuisance benchmark cell {cell_id + 1}/{BENCHMARK_N_CELLS}", flush=True)
        cell = _load_cell(dataset, cell_id, device, names, par_ranges)
        evaluated = evaluate_cell(
            model,
            nre_2d_head,
            nre_9d_head,
            cell,
            cell_id=cell_id,
            names=names,
            par_ranges=par_ranges,
            grid_norm=grid_norm,
            grid_phys=grid_phys,
            npe_samples=npe_samples,
            seed=seed,
            device=device,
            channels_last=channels_last,
            tf_prior=tf_prior,
        )
        for name, _ in ESTIMATORS:
            outputs[name][cell_id] = evaluated[name]
        nre9d_ess[cell_id] = evaluated["nre9d_ess"]
        truth_full[cell_id] = evaluated["truth_full"]
    return {
        **outputs,
        "truth_g": np.asarray(
            fixed_shear_grid(),
            dtype=np.float64,
        )[:, None, :].repeat(BENCHMARK_N_GALAXIES, axis=1),
        "truth_full": truth_full,
        "nre9d_ess": nre9d_ess,
        "grid_phys": grid_phys,
        "prefixes": np.asarray(NUISANCE_PREFIXES, dtype=np.int64),
    }


def aggregate_results(result: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    truth = np.asarray(result["truth_g"], dtype=np.float64)
    metrics = []
    per_shear = []
    for estimator, _ in ESTIMATORS:
        values = np.asarray(result[estimator], dtype=np.float64)
        for prefix in NUISANCE_PREFIXES:
            cell_estimate = np.mean(values[:, :prefix], axis=1)
            fit = fit_component_mc(truth[:, 0], cell_estimate)
            metrics.append(
                {
                    "estimator": estimator,
                    "n_nuisance": int(prefix),
                    **fit,
                    **rmse(truth[:, 0], cell_estimate),
                }
            )
            for cell_id, shear in enumerate(fixed_shear_grid()):
                estimate = cell_estimate[cell_id]
                per_shear.append(
                    {
                        "estimator": estimator,
                        "n_nuisance": int(prefix),
                        "cell": int(cell_id),
                        "g1_truth": float(shear[0]),
                        "g2_truth": float(shear[1]),
                        "g1_bias": float(estimate[0] - shear[0]),
                        "g2_bias": float(estimate[1] - shear[1]),
                    }
                )
    return metrics, per_shear


def _plot_outputs(
    result: dict[str, np.ndarray],
    metrics: list[dict],
    report_dir: Path,
) -> dict[str, str]:
    plt = _setup_matplotlib()
    report_dir.mkdir(parents=True, exist_ok=True)
    labels = dict(ESTIMATORS)
    colors = (
        "#1f4e79",
        "#8f3b76",
        "#a05a2c",
        "#2e7d32",
        "#6a4c93",
        "#b56576",
    )
    figures = {}
    for metric_name, ylabel, filename in (
        ("combined_m", "combined multiplicative bias m", "m_vs_nuisance.png"),
        ("combined_rmse", "combined RMSE", "rmse_vs_nuisance.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        for color, (name, label) in zip(colors, ESTIMATORS):
            subset = [row for row in metrics if row["estimator"] == name]
            axis.plot(
                [row["n_nuisance"] for row in subset],
                [row[metric_name] for row in subset],
                marker="o",
                color=color,
                label=label,
            )
        if metric_name == "combined_m":
            axis.axhline(0.0, color="#555", ls="--", lw=0.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks(list(NUISANCE_PREFIXES))
        axis.get_xaxis().set_major_formatter(
            plt.FuncFormatter(lambda value, _: str(int(value)))
        )
        axis.set_xlabel("nuisance galaxies combined")
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        figure.savefig(report_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        key = filename.removesuffix(".png")
        figures[key] = filename

    for prefix in (1, 32):
        figure, axes = plt.subplots(
            2,
            len(ESTIMATORS),
            figsize=(18, 6),
            sharex=True,
            sharey="row",
        )
        for column, (name, label) in enumerate(ESTIMATORS):
            values = np.mean(result[name][:, :prefix], axis=1)
            truth = result["truth_g"][:, 0]
            for axis, component in zip(axes[:, column], (0, 1)):
                axis.scatter(
                    truth[:, component],
                    values[:, component] - truth[:, component],
                    s=18,
                    alpha=0.75,
                    color=colors[column],
                )
                axis.axhline(0.0, color="#555", ls="--", lw=0.8)
                axis.set_title(label, fontsize=9)
                axis.set_xlabel(f"true g{component + 1}")
                if column == 0:
                    axis.set_ylabel(f"bias in g{component + 1}")
        figure.suptitle(f"Shear bias versus truth, N nuisance={prefix}")
        figure.tight_layout()
        filename = f"scatter_n{prefix}.png"
        figure.savefig(report_dir / filename, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        figures[f"scatter_n{prefix}"] = filename
    return figures


def _write_html(
    payload: dict,
    figures: dict[str, str],
    report_dir: Path,
) -> None:
    labels = dict(ESTIMATORS)
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(labels[row['estimator']])}</td>"
        f"<td>{row['n_nuisance']}</td>"
        f"<td>{row['g1_m']:.5f}</td><td>{row['g2_m']:.5f}</td>"
        f"<td>{row['g1_c']:.5f}</td><td>{row['g2_c']:.5f}</td>"
        f"<td>{row['combined_rmse']:.5f}</td></tr>"
        for row in payload["metrics"]
    )
    ess = payload["nre9d_candidate_ess"]
    ess_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['median']:.1f}</td>"
        f"<td>{value['minimum']:.1f}</td><td>{value['mean']:.1f}</td></tr>"
        for key, value in ess.items()
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Nuisance-only estimator benchmark</title>
<style>
body {{ font: 17px/1.55 Palatino, serif; margin: 2rem auto; max-width: 1180px; }}
h1, h2 {{ font-weight: 600; }} figure {{ margin: 1.4rem 0 2rem; }}
figcaption, .note {{ color: #444; }} img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: .32rem .45rem; }}
th {{ background: #f4f4f4; text-align: left; }}
</style></head><body>
<h1>Nuisance-only estimator benchmark</h1>
<p class="lead">A controlled 5×5 fixed-shear grid compares six estimators
as paired nuisance-galaxy prefixes grow. Every galaxy uses one deterministic
identity-noise realization; no noise realizations are stacked or averaged.</p>
<h2>Design</h2>
<p>There are 25 exact shear cells and 32 paired nuisance galaxies per cell.
The nuisance prefixes are {", ".join(str(value) for value in NUISANCE_PREFIXES)}.
The 9D NRE estimates use the factorized ratio head with the configured
truncated-normal Tully–Fisher prior on <code>vcirc</code>. Its mean uses the
common NPE candidate bank as an importance proposal; its MAP is the maximum
corrected 9D log density among those candidates.</p>
<h2>Bias and RMSE versus nuisance count</h2>
<figure><img src="{html.escape(figures['m_vs_nuisance'])}"
alt="bias versus nuisance count"></figure>
<figure><img src="{html.escape(figures['rmse_vs_nuisance'])}"
alt="RMSE versus nuisance count"></figure>
<h2>Candidate effective sample size</h2>
<table><thead><tr><th>Estimator</th><th>median</th><th>minimum</th><th>mean</th>
</tr></thead><tbody>{ess_rows}</tbody></table>
<h2>Fixed-shear bias versus truth</h2>
<figure><img src="{html.escape(figures['scatter_n1'])}"
alt="bias versus truth at one nuisance galaxy"></figure>
<figure><img src="{html.escape(figures['scatter_n32'])}"
alt="bias versus truth at 32 nuisance galaxies"></figure>
<h2>Metrics</h2>
<table><thead><tr><th>Estimator</th><th>N nuisance</th><th>m g1</th>
<th>m g2</th><th>c g1</th><th>c g2</th><th>combined RMSE</th>
</tr></thead><tbody>{metric_rows}</tbody></table>
</body></html>"""
    (report_dir / "report.html").write_text(body, encoding="utf-8")


def _summarize_ess(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.npe_samples <= 0 or args.grid_n < 2:
        raise ValueError("npe-samples must be positive and grid-n must be at least 2")
    if args.report_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.report_dir} exists; use --overwrite")
    if not args.device.startswith("cuda") and args.device != "cpu":
        raise ValueError("device must be cuda or cpu")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but CUDA is unavailable")

    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)
    configs_root = args.model_root.parent / "configs"
    model_config = load_model_config(
        args.parent_npe,
        configs_root=str(configs_root),
    )
    config.set_model_config(model_config)
    names = tuple(config.TARGET_NAMES)
    par_ranges = config.par_ranges
    config.require_matching_dataset_par_ranges(args.data_dir, par_ranges)
    parent_checkpoint = _checkpoint_file(
        args.model_root,
        args.parent_npe,
        args.parent_checkpoint_suffix,
    )
    model = load_model(
        KLNPE,
        path=str(parent_checkpoint),
        model_name=args.parent_npe,
        device=str(device),
        strict=True,
        networks_root=str(args.model_root.parent / "networks"),
    )
    model.eval()
    channels_last = bool(model_config.train.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    nre2d_checkpoint = nre_checkpoint_path(
        args.model_root,
        args.nre_2d_name,
        args.nre_checkpoint_suffix,
    )
    nre9d_checkpoint = nre_checkpoint_path(
        args.model_root,
        args.nre_9d_name,
        args.nre_checkpoint_suffix,
    )
    nre2d_head, nre2d_meta = load_ratio_head(
        nre2d_checkpoint,
        device=device,
        expected_parent=args.parent_npe,
    )
    nre9d_head, nre9d_meta = load_ratio_head(
        nre9d_checkpoint,
        device=device,
        expected_parent=args.parent_npe,
    )
    if int(nre2d_meta.get("parameter_dim", 0)) != 2:
        raise ValueError("the 2D NRE checkpoint must have parameter_dim=2")
    if int(nre9d_meta.get("parameter_dim", 0)) != 9:
        raise ValueError("the 9D NRE checkpoint must have parameter_dim=9")
    dataset = pxt.TorchDataset(str(args.data_dir))
    tf_prior = TFPrior()
    result = run_benchmark(
        dataset,
        model,
        nre2d_head,
        nre9d_head,
        names=names,
        par_ranges=par_ranges,
        channels_last=channels_last,
        device=device,
        npe_samples=args.npe_samples,
        grid_n=args.grid_n,
        seed=args.seed,
        tf_prior=tf_prior,
    )
    metrics, per_shear = aggregate_results(result)
    figures = _plot_outputs(result, metrics, args.report_dir)
    ess_summary = {
        "9D NRE posterior candidate ESS": _summarize_ess(result["nre9d_ess"]),
    }
    meta = {
        "schema": "likelihood-stack-nuisance-only-benchmark-v1",
        "data_dir": str(args.data_dir),
        "parent_npe": args.parent_npe,
        "nre_2d_name": args.nre_2d_name,
        "nre_9d_name": args.nre_9d_name,
        "parent_checkpoint": str(parent_checkpoint),
        "nre_2d_checkpoint": str(nre2d_checkpoint),
        "nre_9d_checkpoint": str(nre9d_checkpoint),
        "seed": int(args.seed),
        "shear_values": list(BENCHMARK_SHEAR_VALUES),
        "n_cells": int(BENCHMARK_N_CELLS),
        "n_galaxies_per_cell": int(BENCHMARK_N_GALAXIES),
        "nuisance_prefixes": list(NUISANCE_PREFIXES),
        "noise_realizations": 1,
        "noise_realization": REALIZATION,
        "npe_samples": int(args.npe_samples),
        "grid_n": int(args.grid_n),
        "tf_prior": tf_prior.to_dict(),
        "nre_2d_meta": nre2d_meta,
        "nre_9d_meta": nre9d_meta,
        "truth_is_candidate": False,
        "identity_only": True,
    }
    payload = {
        **meta,
        "metrics": metrics,
        "per_shear": per_shear,
        "nre9d_candidate_ess": ess_summary,
        "figures": figures,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.report_dir / "benchmark.npz",
        **result,
        meta_json=json.dumps(json_safe(meta)),
    )
    write_json(args.report_dir / "report.json", payload)
    _write_html(payload, figures, args.report_dir)
    print(f"Wrote {args.report_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
