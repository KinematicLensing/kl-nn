#!/usr/bin/env python3
"""Bounded emcee inference for the full nine-dimensional NRE ratio head."""

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
    DATA_ROOT,
    G01_NPE,
    HTML_STYLE,
    MODEL_ROOT,
    NRE9D_BURNIN,
    NRE9D_DIR,
    NRE9D_NAME,
    NRE9D_NWALKERS,
    NRE9D_PRODUCTION,
    NRE_INFER_DATASET,
    SELECT_DIR,
    STACK_DIR,
    json_safe,
    write_json,
)
from diagnostics.likelihood_stack_nre_train import (
    apply_identity_noise,
    encode_flow_context,
    load_frozen_parent,
    load_observation_batch,
    load_ratio_head,
    nre_checkpoint_path,
)
from tf_prior import TFPrior, tf_log_prior_ratio
from train import seed_everything
from utils import denormalize
from diagnostics.likelihood_stack_benchmark_core import fit_component_mc


NUISANCE_STACK_PREFIXES = (8, 16, 32, 64, 128)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_ROOT / NRE_INFER_DATASET
    )
    parser.add_argument("--report-dir", type=Path, default=NRE9D_DIR)
    parser.add_argument(
        "--galaxies",
        type=Path,
        default=SELECT_DIR / "galaxies.npz",
        help="galaxies.npz selection used by the shared stacking test",
    )
    parser.add_argument(
        "--stack-draws",
        type=Path,
        default=STACK_DIR / "stack.npz",
        help="optional shared-stack draws for NPE control curves",
    )
    parser.add_argument("--example-index", type=int, default=None)
    parser.add_argument("--nwalkers", type=int, default=NRE9D_NWALKERS)
    parser.add_argument("--burnin", type=int, default=NRE9D_BURNIN)
    parser.add_argument("--production", type=int, default=NRE9D_PRODUCTION)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_selection_indices(path: Path, dataset_size: int) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "index" not in data.files:
        raise ValueError(f"{path} does not contain an index array")
    indices = np.asarray(data["index"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(f"{path} index must be a non-empty one-dimensional array")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{path} index contains duplicates")
    if np.any(indices < 0) or np.any(indices >= int(dataset_size)):
        raise ValueError(f"{path} index contains out-of-range dataset rows")
    return indices


def _as_parameter_matrix(theta_norm, parameter_dim: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(theta_norm, dtype=np.float64)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != parameter_dim:
        raise ValueError(
            f"theta_norm must have shape ({parameter_dim},) or "
            f"(N, {parameter_dim}); got {values.shape}"
        )
    return values, scalar


def bounded_nre_log_density(
    theta_norm,
    *,
    head,
    context: torch.Tensor,
    rmag_true: float,
    parameter_names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    prior: TFPrior,
) -> float | np.ndarray:
    """Evaluate ``log r_NRE + log p_TF(vcirc|rmag) - log p_uniform(vcirc)``.

    The sampler coordinates are the normalized LMDB coordinates.  All nine
    coordinates must remain in the closed normalized box before the ratio head
    or the TF correction is evaluated.
    """

    names = tuple(parameter_names)
    values, scalar = _as_parameter_matrix(theta_norm, len(names))
    result = np.full(values.shape[0], -np.inf, dtype=np.float64)
    supported = np.all(np.isfinite(values), axis=1) & np.all(
        (values >= -1.0) & (values <= 1.0), axis=1
    )
    if np.any(supported):
        supported_values = values[supported]
        tensor_values = torch.as_tensor(
            supported_values,
            dtype=context.dtype,
            device=context.device,
        )
        expanded_context = context.reshape(1, -1).expand(
            tensor_values.shape[0], -1
        )
        with torch.inference_mode():
            nre_log_ratio = head(expanded_context, tensor_values).detach().cpu().numpy()
        physical = denormalize(
            supported_values,
            par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        )
        vcirc_index = names.index("vcirc")
        tf_ratio = tf_log_prior_ratio(
            physical[:, vcirc_index],
            float(rmag_true),
            prior,
        )
        combined = np.asarray(nre_log_ratio, dtype=np.float64) + tf_ratio
        finite = np.isfinite(combined)
        supported_indices = np.flatnonzero(supported)
        result[supported_indices[finite]] = combined[finite]
    if scalar:
        return float(result[0])
    return result


def run_emcee(
    log_density,
    initial_walkers: np.ndarray,
    *,
    burnin: int,
    production: int,
    seed: int,
    vectorized: bool = False,
) -> dict[str, np.ndarray | dict]:
    """Run a bounded emcee chain and retain sampler diagnostics."""

    try:
        import emcee
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("emcee is required for 9D NRE inference") from exc

    initial = np.asarray(initial_walkers, dtype=np.float64)
    if initial.ndim != 2:
        raise ValueError("initial_walkers must have shape (nwalkers, ndim)")
    nwalkers, parameter_dim = initial.shape
    if nwalkers < 2 * parameter_dim:
        raise ValueError("emcee requires at least 2 * ndim walkers")
    if burnin < 0 or production <= 0:
        raise ValueError("burnin must be non-negative and production positive")
    if np.any(~np.isfinite(initial)):
        raise ValueError("initial walkers must be finite")
    if np.any((initial < -1.0) | (initial > 1.0)):
        raise ValueError("initial walkers must lie in the normalized box")

    rng = np.random.default_rng(seed)
    centered = initial - np.mean(initial, axis=0, keepdims=True)
    if np.linalg.matrix_rank(centered) < parameter_dim:
        initial = np.clip(
            initial + rng.normal(0.0, 1.0e-6, size=initial.shape),
            -1.0 + 1.0e-8,
            1.0 - 1.0e-8,
        )

    np.random.seed(seed)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        parameter_dim,
        log_density,
        vectorize=bool(vectorized),
    )
    if burnin:
        sampler.run_mcmc(initial, burnin, progress=False)
        sampler.reset()
        sampler.run_mcmc(None, production, progress=False)
    else:
        sampler.run_mcmc(initial, production, progress=False)

    chain = np.asarray(sampler.get_chain(flat=True), dtype=np.float64)
    log_prob = np.asarray(sampler.get_log_prob(flat=True), dtype=np.float64)
    acceptance = np.asarray(sampler.acceptance_fraction, dtype=np.float64)
    try:
        autocorrelation = np.asarray(
            sampler.get_autocorr_time(tol=0), dtype=np.float64
        )
        converged = bool(
            np.all(np.isfinite(autocorrelation))
            and production >= 50.0 * float(np.max(autocorrelation))
        )
        autocorrelation_error = None
    except Exception as exc:  # emcee raises when the chain is too short
        autocorrelation = np.full(parameter_dim, np.nan, dtype=np.float64)
        converged = False
        autocorrelation_error = str(exc)
    diagnostics = {
        "nwalkers": int(nwalkers),
        "parameter_dim": int(parameter_dim),
        "burnin": int(burnin),
        "production": int(production),
        "acceptance_mean": float(np.mean(acceptance)),
        "acceptance_min": float(np.min(acceptance)),
        "acceptance_max": float(np.max(acceptance)),
        "autocorrelation_time": autocorrelation,
        "autocorrelation_error": autocorrelation_error,
        "converged": converged,
    }
    return {"chain": chain, "log_prob": log_prob, "diagnostics": diagnostics}


def summarize_shear_samples(
    samples_norm: np.ndarray,
    log_prob: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
) -> dict:
    """Return marginal shear intervals and a maximum-density sample summary."""

    samples_norm = np.asarray(samples_norm, dtype=np.float64)
    log_prob = np.asarray(log_prob, dtype=np.float64)
    if samples_norm.ndim != 2 or samples_norm.shape[1] != len(parameter_names):
        raise ValueError("samples_norm has the wrong parameter dimension")
    if log_prob.shape != (len(samples_norm),):
        raise ValueError("log_prob must have one value per sample")
    physical = denormalize(
        samples_norm,
        par_ranges,
        feature_names=parameter_names,
        target_transforms=config.TARGET_TRANSFORMS,
    )
    g1_index = parameter_names.index("g1")
    g2_index = parameter_names.index("g2")
    finite = np.isfinite(log_prob)
    if not np.any(finite):
        raise ValueError("emcee produced no finite log-density samples")
    map_index = int(np.nanargmax(np.where(finite, log_prob, -np.inf)))
    summary = {
        "g1_mean": float(np.mean(physical[:, g1_index])),
        "g2_mean": float(np.mean(physical[:, g2_index])),
        "g1_lower": float(np.quantile(physical[:, g1_index], 0.16)),
        "g1_upper": float(np.quantile(physical[:, g1_index], 0.84)),
        "g2_lower": float(np.quantile(physical[:, g2_index], 0.16)),
        "g2_upper": float(np.quantile(physical[:, g2_index], 0.84)),
        "g1_map_like": float(physical[map_index, g1_index]),
        "g2_map_like": float(physical[map_index, g2_index]),
        "map_like_log_density": float(log_prob[map_index]),
        "n_samples": int(len(samples_norm)),
    }
    return summary


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_control_estimates(
    selection_path: Path,
    stack_path: Path,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    controls: dict[str, np.ndarray] = {}
    selection = np.load(selection_path, allow_pickle=False)
    selected_indices = np.asarray(selection["index"], dtype=np.int64)
    if np.array_equal(selected_indices, indices) and "mean_g" in selection.files:
        controls["NPE posterior mean"] = np.asarray(selection["mean_g"], dtype=np.float64)
    if stack_path.is_file():
        stack = np.load(stack_path, allow_pickle=False)
        stack_indices = np.asarray(stack["index"], dtype=np.int64)
        if np.array_equal(stack_indices, indices) and "stack_map" in stack.files:
            controls["NPE shared 16-noise MAP"] = np.asarray(
                stack["stack_map"][:, -1], dtype=np.float64
            )
    return controls


def _nuisance_stack_rows(
    truth: np.ndarray,
    estimates: dict[str, np.ndarray],
    *,
    n_bootstrap: int = 1000,
) -> list[dict]:
    rng = np.random.default_rng(420042)
    rows = []
    for count in NUISANCE_STACK_PREFIXES:
        if count > len(truth):
            continue
        bootstrap_indices = rng.integers(
            0,
            count,
            size=(n_bootstrap, count),
            dtype=np.int64,
        )
        for name, values in estimates.items():
            metrics = fit_component_mc(truth[:count], values[:count])
            bootstrap_m = np.asarray(
                [
                    fit_component_mc(truth[:count][draw], values[:count][draw])[
                        "combined_m"
                    ]
                    for draw in bootstrap_indices
                ],
                dtype=np.float64,
            )
            rows.append(
                {
                    "estimator": name,
                    "nuisance_count": int(count),
                    "combined_m_lower": float(np.quantile(bootstrap_m, 0.16)),
                    "combined_m_upper": float(np.quantile(bootstrap_m, 0.84)),
                    **metrics,
                }
            )
    return rows


def save_report_figures(
    payload: dict,
    report_dir: Path,
    *,
    example_chain: np.ndarray,
    example_index: int,
    example_log_prob: np.ndarray,
) -> dict[str, str]:
    plt = _setup_matplotlib()
    report_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    physical = np.asarray(
        denormalize(
            example_chain,
            config.par_ranges,
            feature_names=tuple(payload["parameter_names"]),
            target_transforms=config.TARGET_TRANSFORMS,
        ),
        dtype=np.float64,
    )
    names = tuple(payload["parameter_names"])
    g1_index, g2_index = names.index("g1"), names.index("g2")
    finite = np.isfinite(example_log_prob)
    map_index = int(np.nanargmax(np.where(finite, example_log_prob, -np.inf)))
    summary = next(
        row for row in payload["summaries"] if int(row["index"]) == int(example_index)
    )
    fig, axis = plt.subplots(figsize=(6.3, 5.2))
    axis.hist2d(
        physical[:, g1_index],
        physical[:, g2_index],
        bins=40,
        cmap="Blues",
    )
    axis.scatter(
        [summary["truth_g1"]],
        [summary["truth_g2"]],
        marker="*",
        s=120,
        color="#c0392b",
        label="truth",
        zorder=3,
    )
    axis.scatter(
        [physical[map_index, g1_index]],
        [physical[map_index, g2_index]],
        marker="x",
        s=70,
        color="#1f4e79",
        label="MAP-like retained sample",
        zorder=3,
    )
    axis.set_xlabel("g1")
    axis.set_ylabel("g2")
    axis.set_title(f"9D NRE + TF posterior, dataset index {example_index}")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = report_dir / "example_posterior_g1_g2.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figures["example_posterior"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    colors = (
        "#1f4e79",
        "#8f3b76",
        "#a05a2c",
        "#2e7d32",
    )
    for color, (name, rows) in zip(
        colors,
        (
            (name, [row for row in payload["nuisance_stack_table"] if row["estimator"] == name])
            for name in payload["nuisance_estimators"]
        ),
    ):
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["nuisance_count"])
        x = np.asarray([row["nuisance_count"] for row in rows], dtype=float)
        y = np.asarray([row["combined_m"] for row in rows], dtype=float)
        lower = np.asarray([row["combined_m_lower"] for row in rows], dtype=float)
        upper = np.asarray([row["combined_m_upper"] for row in rows], dtype=float)
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - lower, upper - y)),
            marker="o",
            capsize=3,
            label=name,
            color=color,
        )
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(NUISANCE_STACK_PREFIXES))
    axis.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda value, _: str(int(value)))
    )
    axis.set_xlabel("number of selected galaxies in cumulative ensemble")
    axis.set_ylabel("combined multiplicative bias m")
    axis.set_title("9D NRE ensemble slope versus selected-galaxy count")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = report_dir / "m_vs_nuisance_stacks.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    figures["m_vs_nuisance"] = path.name
    return figures


def _write_html(payload: dict, report_dir: Path, figures: dict[str, str]) -> None:
    rows = []
    for row in payload["summaries"]:
        diagnostics = row["diagnostics"]
        rows.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td>{row['g1_mean']:.5f}</td>"
            f"<td>[{row['g1_lower']:.5f}, {row['g1_upper']:.5f}]</td>"
            f"<td>{row['g2_mean']:.5f}</td>"
            f"<td>[{row['g2_lower']:.5f}, {row['g2_upper']:.5f}]</td>"
            f"<td>{row['g1_map_like']:.5f}, {row['g2_map_like']:.5f}</td>"
            f"<td>{diagnostics['acceptance_mean']:.3f}</td>"
            f"<td>{html.escape(str(diagnostics['converged']))}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Full-9D NRE with emcee and TF prior replacement</title>
<style>{HTML_STYLE}</style></head><body>
<h1>Full-9D NRE inference with bounded emcee</h1>
<p class="lead">The frozen parent encoder supplies a context, while the
separate nine-dimensional NRE head is evaluated continuously in normalized
coordinates. Walkers start from frozen-NPE posterior draws and never leave the
normalized box.</p>
<p>The sampled log density is
<code>log r_NRE(x, theta) + log p_TF(vcirc | rmag_true)
- log p_uniform(vcirc)</code>, using physical <code>vcirc</code> after
denormalization. The MAP-like columns are the highest-density retained sample,
not an optimization result.</p>
<p class="note">Warning: the 9D NRE ratio is not the original NPE posterior.
This report preserves the existing 2D grid report and its artifacts.</p>
<figure>
<img src="{html.escape(figures['example_posterior'])}" alt="Example 9D NRE posterior in g1 and g2">
<figcaption>Example marginal posterior from the longer emcee run. The
histogram is the g1/g2 projection of the full 9D chain; the red star is the
simulator truth and the blue cross is the highest-density retained sample.</figcaption>
</figure>
<figure>
<img src="{html.escape(figures['m_vs_nuisance'])}" alt="m versus selected galaxy count">
<figcaption>At each cumulative count, the slope is fit over the selected
galaxies used by the shared stacking test. This is a heterogeneous-galaxy
ensemble diagnostic, not the fixed-shear nuisance bootstrap from report 04.
The error bars are the 16th–84th percentiles of 1,000 deterministic
bootstrap resamples of the cumulative selected-galaxy set. The controls are
the cached NPE posterior mean and the original shared 16-noise NPE stack MAP
when available.</figcaption>
</figure>
<table><thead><tr>
<th>Index</th><th>g1 mean</th><th>g1 16–84%</th>
<th>g2 mean</th><th>g2 16–84%</th><th>MAP-like (g1,g2)</th>
<th>acceptance</th><th>converged</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table>
</body></html>
"""
    (report_dir / "report.html").write_text(body, encoding="utf-8")


def write_report(args) -> None:
    if args.report_dir.joinpath("report.html").exists() and not args.overwrite:
        raise FileExistsError(f"{args.report_dir / 'report.html'} exists; use --overwrite")
    if args.nwalkers < 2 * len(config.TARGET_NAMES):
        raise ValueError("nwalkers must be at least 2 * the nine target dimensions")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    names = tuple(config.TARGET_NAMES)
    config.require_matching_dataset_par_ranges(args.data_dir, config.par_ranges)
    encoder, model_config, parent_checkpoint = load_frozen_parent(
        parent_npe=args.parent_npe,
        model_root=args.model_root,
        checkpoint_suffix="best",
        device=device,
    )
    checkpoint = nre_checkpoint_path(args.model_root, args.nre_name)
    head, nre_meta = load_ratio_head(
        checkpoint,
        device=device,
        expected_parent=args.parent_npe,
    )
    if tuple(nre_meta.get("target_names") or ()) != names:
        raise ValueError("9D checkpoint target_names do not match the current schema")
    if int(nre_meta.get("parameter_dim", 0)) != len(names):
        raise ValueError("9D checkpoint parameter_dim is not nine")

    dataset = pxt.TorchDataset(str(args.data_dir))
    dataset_size = len(dataset)
    indices = load_selection_indices(args.galaxies, dataset_size)
    if args.max_galaxies is not None:
        if args.max_galaxies <= 0:
            raise ValueError("max-galaxies must be positive")
        indices = indices[: args.max_galaxies]
    count = len(indices)
    if args.example_index is None:
        example_index = int(indices[0])
    else:
        example_index = int(args.example_index)
        if example_index not in set(int(index) for index in indices):
            raise ValueError("--example-index must be in the selected galaxy list")
    channels_last = bool(model_config.train.channels_last)
    prior = TFPrior()
    summaries = []
    summary_arrays = {
        "index": [],
        "truth_g": [],
        "g1_mean": [],
        "g2_mean": [],
        "g1_lower": [],
        "g1_upper": [],
        "g2_lower": [],
        "g2_upper": [],
        "g1_map_like": [],
        "g2_map_like": [],
    }
    diagnostics_rows = []
    example_chain = None
    example_log_prob = None
    for start in range(0, count, args.batch_size):
        batch_indices = indices[start : start + args.batch_size]
        batch = load_observation_batch(dataset, batch_indices, device)
        image, spec = apply_identity_noise(
            batch["img"],
            batch["spec"],
            batch["image_snr"],
            batch["spec_snr"],
            batch_indices,
            seed=args.seed,
            device=device,
        )
        if channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spec = spec.contiguous(memory_format=torch.channels_last)
        context = encode_flow_context(
            encoder,
            image,
            spec,
            batch["fib_pos"],
            batch["observation_context"],
        )
        with torch.inference_mode():
            npe_samples = encoder.sample(
                image,
                spec,
                args.nwalkers,
                fiber_positions=batch["fib_pos"],
                observation_context=batch["observation_context"],
            )
        truth_physical = denormalize(
            batch["fid_pars"].cpu().numpy(),
            config.par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        )
        for row, index in enumerate(batch_indices):
            row_context = context[row].detach()
            rmag_true = float(batch["observation_context"]["rmag_true"][row].item())
            log_density = lambda theta, c=row_context, r=rmag_true: bounded_nre_log_density(
                theta,
                head=head,
                context=c,
                rmag_true=r,
                parameter_names=names,
                par_ranges=config.par_ranges,
                prior=prior,
            )
            chain = run_emcee(
                log_density,
                npe_samples[row].detach().cpu().numpy(),
                burnin=args.burnin,
                production=args.production,
                seed=args.seed + int(index),
                vectorized=True,
            )
            summary = summarize_shear_samples(
                chain["chain"],
                chain["log_prob"],
                parameter_names=names,
                par_ranges=config.par_ranges,
            )
            summary["index"] = int(index)
            summary["truth_g1"] = float(truth_physical[row, names.index("g1")])
            summary["truth_g2"] = float(truth_physical[row, names.index("g2")])
            summary["diagnostics"] = chain["diagnostics"]
            summaries.append(summary)
            diagnostics_rows.append(chain["diagnostics"])
            if int(index) == example_index:
                example_chain = np.asarray(chain["chain"], dtype=np.float64)
                example_log_prob = np.asarray(chain["log_prob"], dtype=np.float64)
            summary_arrays["index"].append(int(index))
            summary_arrays["truth_g"].append(
                [summary["truth_g1"], summary["truth_g2"]]
            )
            for key in summary_arrays:
                if key in ("index", "truth_g"):
                    continue
                summary_arrays[key].append(summary[key])
        print(f"inferred galaxies {min(start + args.batch_size, count)}/{count}", flush=True)

    if example_chain is None or example_log_prob is None:
        raise RuntimeError("the requested example galaxy was not inferred")
    nre_mean = np.column_stack(
        (summary_arrays["g1_mean"], summary_arrays["g2_mean"])
    )
    nre_map = np.column_stack(
        (summary_arrays["g1_map_like"], summary_arrays["g2_map_like"])
    )
    truth_g = np.asarray(summary_arrays["truth_g"], dtype=np.float64)
    estimates = {
        "9D NRE posterior mean": nre_mean,
        "9D NRE MAP-like": nre_map,
    }
    if args.galaxies.is_file():
        estimates.update(
            _load_control_estimates(args.galaxies, args.stack_draws, indices)
        )
    nuisance_rows = _nuisance_stack_rows(truth_g, estimates)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.report_dir / "summaries.npz",
        example_chain=example_chain,
        example_log_prob=example_log_prob,
        example_index=np.asarray(example_index, dtype=np.int64),
        **{key: np.asarray(value) for key, value in summary_arrays.items()},
    )
    payload = {
        "parent_npe": args.parent_npe,
        "parent_checkpoint": str(parent_checkpoint),
        "nre_name": args.nre_name,
        "nre_checkpoint": str(checkpoint),
        "nre_meta": nre_meta,
        "data_dir": str(args.data_dir),
        "dataset_size": int(dataset_size),
        "n_galaxies": int(count),
        "selection_file": str(args.galaxies),
        "stack_draws": str(args.stack_draws),
        "example_index": int(example_index),
        "parameter_names": list(names),
        "parameter_dim": len(names),
        "normalized_support": [[-1.0, 1.0] for _ in names],
        "tf_prior": prior.to_dict(),
        "log_density_formula": (
            "log r_NRE(x, theta_norm) + "
            "log p_TF(vcirc | rmag_true) - log p_uniform(vcirc)"
        ),
        "walker_initialization": "frozen-NPE posterior draws",
        "burnin": int(args.burnin),
        "production": int(args.production),
        "nwalkers": int(args.nwalkers),
        "seed": int(args.seed),
        "summaries": summaries,
        "nuisance_estimators": list(estimates),
        "nuisance_stack_table": nuisance_rows,
        "warning": "The 9D NRE ratio is not the original NPE posterior.",
    }
    if diagnostics_rows:
        payload["diagnostics_summary"] = {
            "acceptance_mean": float(
                np.mean([row["acceptance_mean"] for row in diagnostics_rows])
            ),
            "converged_fraction": float(
                np.mean([row["converged"] for row in diagnostics_rows])
            ),
        }
    figures = save_report_figures(
        payload,
        args.report_dir,
        example_chain=example_chain,
        example_index=example_index,
        example_log_prob=example_log_prob,
    )
    payload["figures"] = figures
    _write_html(payload, args.report_dir, figures)
    write_json(args.report_dir / "report.json", payload)
    print(f"Wrote {args.report_dir / 'report.json'}", flush=True)


def main(argv=None) -> None:
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
