#!/usr/bin/env python3
"""Benchmark six estimators with strict BA14 noise/nuisance stacking."""

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
    NUISANCE_BOOTSTRAP_COUNT,
    NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
    NUISANCE_BOOTSTRAP_SEED_OFFSET,
    NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
    paired_nuisance_bootstrap_indices,
)
from diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_DATASET,
    BENCHMARK_GRID_N,
    BENCHMARK_N_GALAXIES,
    BENCHMARK_N_CELLS,
    BENCHMARK_NOISE_SEED,
    BENCHMARK_N_REALIZATIONS,
    BENCHMARK_SHEAR_VALUES,
    deterministic_nuisance_bank,
    fit_component_mc,
    fixed_shear_grid,
    marginalize_nuisance_log_surface,
    rmse,
    stack_log_surfaces,
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
DEFAULT_NUISANCE_SAMPLES = 4096
DEFAULT_NUISANCE_SEED = BENCHMARK_NOISE_SEED + 1
DEFAULT_CANDIDATE_CHUNK_SIZE = 2048
NORMALIZED_BOUNDARY_EPS = 1.0e-4
FIGURE_DPI = 150
ESTIMATORS = (
    ("npe_mean", "stacked 9D NPE marginalized mean"),
    ("npe_map", "stacked 9D NPE marginalized MAP"),
    ("nre2d_mean", "stacked 2D NRE posterior mean"),
    ("nre2d_map", "stacked 2D NRE grid MAP"),
    ("nre9d_mean", "stacked 9D NRE marginalized mean"),
    ("nre9d_map", "stacked 9D NRE marginalized MAP"),
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
    parser.add_argument(
        "--nuisance-samples",
        type=int,
        default=DEFAULT_NUISANCE_SAMPLES,
        help="uniform normalized-prior samples used for 9D nuisance integration",
    )
    parser.add_argument("--nuisance-seed", type=int, default=DEFAULT_NUISANCE_SEED)
    parser.add_argument(
        "--candidate-chunk-size",
        type=int,
        default=DEFAULT_CANDIDATE_CHUNK_SIZE,
    )
    parser.add_argument("--grid-n", type=int, default=BENCHMARK_GRID_N)
    parser.add_argument("--seed", type=int, default=BENCHMARK_NOISE_SEED)
    parser.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="regenerate the report from a saved benchmark.npz without inference",
    )
    parser.add_argument(
        "--bootstrap-count",
        type=int,
        default=NUISANCE_BOOTSTRAP_COUNT,
        help="paired galaxy bootstrap replicates for report error bars",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        help="bootstrap seed; defaults to benchmark seed plus its fixed offset",
    )
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


def _candidate_bank(
    grid_norm: np.ndarray,
    nuisance_bank_norm: np.ndarray,
) -> np.ndarray:
    grid = np.asarray(grid_norm, dtype=np.float64)
    nuisance = np.asarray(nuisance_bank_norm, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[-1] != 2:
        raise ValueError("grid_norm must have shape (n_shear, 2)")
    if nuisance.ndim != 2 or nuisance.shape[-1] != 7:
        raise ValueError("nuisance_bank_norm must have shape (n_samples, 7)")
    candidates = np.concatenate(
        (
            np.repeat(grid, nuisance.shape[0], axis=0),
            np.tile(nuisance, (grid.shape[0], 1)),
        ),
        axis=1,
    )
    # The trained rational-quadratic spline has an open transform domain;
    # exact normalized prior endpoints can therefore raise InputOutsideDomain.
    return np.clip(
        candidates,
        -1.0 + NORMALIZED_BOUNDARY_EPS,
        1.0 - NORMALIZED_BOUNDARY_EPS,
    )


@torch.inference_mode()
def _flow_log_density(
    model,
    context: torch.Tensor,
    candidates: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate normalized 9D NPE log density on a shared candidate bank."""

    if context.ndim != 2 or context.shape[0] <= 0:
        raise ValueError("context must have shape (batch, context_dim)")
    values = np.asarray(candidates, dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] != 9:
        raise ValueError("candidates must have shape (n_candidates, 9)")
    batch = int(context.shape[0])
    scores = np.empty((batch, values.shape[0]), dtype=np.float64)
    for start in range(0, values.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), values.shape[0])
        chunk = torch.as_tensor(
            values[start:stop],
            device=context.device,
            dtype=torch.float32,
        )
        parameters = chunk.unsqueeze(0).expand(batch, -1, -1)
        score_context = context[:, None, :].expand(
            batch, stop - start, context.shape[-1]
        )
        scores[:, start:stop] = (
            model.flow.log_prob(
                parameters.reshape(-1, values.shape[-1]),
                context=score_context.reshape(-1, context.shape[-1]),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(batch, stop - start)
        )
    return scores


@torch.inference_mode()
def _nre_log_ratio(
    head,
    context: torch.Tensor,
    candidates: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate a 9D NRE ratio head on a shared candidate bank."""

    if context.ndim != 2 or context.shape[0] <= 0:
        raise ValueError("context must have shape (batch, context_dim)")
    values = np.asarray(candidates, dtype=np.float32)
    if values.ndim != 2 or values.shape[-1] != 9:
        raise ValueError("candidates must have shape (n_candidates, 9)")
    batch = int(context.shape[0])
    scores = np.empty((batch, values.shape[0]), dtype=np.float64)
    for start in range(0, values.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), values.shape[0])
        chunk = torch.as_tensor(
            values[start:stop],
            device=context.device,
            dtype=torch.float32,
        )
        parameters = chunk.unsqueeze(0).expand(batch, -1, -1)
        score_context = context[:, None, :].expand(
            batch, stop - start, context.shape[-1]
        )
        scores[:, start:stop] = (
            head(
                score_context.reshape(-1, context.shape[-1]),
                parameters.reshape(-1, values.shape[-1]),
            )
            .detach()
            .cpu()
            .numpy()
            .reshape(batch, stop - start)
        )
    return scores


def _log_integrand_9d_surfaces(
    model,
    nre_9d_head,
    context: torch.Tensor,
    *,
    grid_norm: np.ndarray,
    nuisance_bank_norm: np.ndarray,
    rmag_true: np.ndarray,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    tf_prior: TFPrior,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build TF-corrected 9D log integrands on a shared nuisance bank."""

    if context.ndim != 2:
        raise ValueError("context must have shape (galaxy, context_dim)")
    candidates = _candidate_bank(grid_norm, nuisance_bank_norm)
    physical = np.asarray(
        denormalize(
            candidates,
            par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        ),
        dtype=np.float64,
    )
    vcirc_index = names.index("vcirc")
    tf_ratio = tf_log_prior_ratio(
        physical[None, :, vcirc_index],
        np.asarray(rmag_true, dtype=np.float64)[:, None],
        tf_prior,
    )
    n_shear = int(np.asarray(grid_norm).shape[0])
    n_nuisance = int(np.asarray(nuisance_bank_norm).shape[0])
    shape = (context.shape[0], n_shear, n_nuisance)
    tf_ratio = np.asarray(tf_ratio, dtype=np.float64).reshape(shape)
    npe_scores = _flow_log_density(
        model,
        context,
        candidates,
        chunk_size=chunk_size,
    ).reshape(shape)
    nre_scores = _nre_log_ratio(
        nre_9d_head,
        context,
        candidates,
        chunk_size=chunk_size,
    ).reshape(shape)
    return npe_scores, nre_scores, tf_ratio


def _marginalized_9d_surfaces(
    model,
    nre_9d_head,
    context: torch.Tensor,
    *,
    grid_norm: np.ndarray,
    nuisance_bank_norm: np.ndarray,
    rmag_true: np.ndarray,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    tf_prior: TFPrior,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build TF-corrected 9D NPE/NRE shear likelihood surfaces."""

    npe_scores, nre_scores, tf_ratio = _log_integrand_9d_surfaces(
        model,
        nre_9d_head,
        context,
        grid_norm=grid_norm,
        nuisance_bank_norm=nuisance_bank_norm,
        rmag_true=rmag_true,
        names=names,
        par_ranges=par_ranges,
        tf_prior=tf_prior,
        chunk_size=chunk_size,
    )
    npe_integrand = npe_scores + tf_ratio
    nre_integrand = nre_scores + tf_ratio
    npe_surface = np.empty(npe_integrand.shape[:2], dtype=np.float64)
    nre_surface = np.empty_like(npe_surface)
    npe_ess = np.empty_like(npe_surface)
    nre_ess = np.empty_like(npe_surface)
    for galaxy in range(context.shape[0]):
        npe_surface[galaxy], npe_ess[galaxy] = (
            marginalize_nuisance_log_surface(
                npe_integrand[galaxy],
                return_ess=True,
            )
        )
        nre_surface[galaxy], nre_ess[galaxy] = (
            marginalize_nuisance_log_surface(
                nre_integrand[galaxy],
                return_ess=True,
            )
        )
    return npe_surface, npe_ess, nre_surface, nre_ess


def evaluate_cell_surfaces(
    model,
    nre_2d_head,
    nre_9d_head,
    cell: dict,
    *,
    cell_id: int,
    realization: int,
    names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    grid_norm: np.ndarray,
    nuisance_bank_norm: np.ndarray,
    seed: int,
    device: torch.device,
    channels_last: bool,
    tf_prior: TFPrior,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Evaluate per-galaxy shear likelihood surfaces for one cell."""

    images, spectra = _apply_benchmark_noise(
        cell,
        cell_id=cell_id,
        realization=realization,
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
    grid_tensor = torch.as_tensor(
        grid_norm,
        device=device,
        dtype=torch.float32,
    )
    nre2d_surface = score_shear_grid(
        nre_2d_head,
        context,
        grid_tensor,
    ).detach().cpu().numpy()
    npe_scores, nre9d_scores, tf_ratio = _log_integrand_9d_surfaces(
        model,
        nre_9d_head,
        context,
        grid_norm=grid_norm,
        nuisance_bank_norm=nuisance_bank_norm,
        rmag_true=cell["rmag"].detach().cpu().numpy(),
        names=names,
        par_ranges=par_ranges,
        tf_prior=tf_prior,
        chunk_size=chunk_size,
    )
    return {
        "npe_log_integrand": npe_scores,
        "nre2d_log_surface": np.asarray(nre2d_surface, dtype=np.float64),
        "nre9d_log_integrand": nre9d_scores,
        "tf_log_prior": tf_ratio,
        "truth_full": np.asarray(cell["truth"], dtype=np.float64),
    }


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
    grid_n: int,
    seed: int,
    nuisance_samples: int,
    nuisance_seed: int,
    chunk_size: int,
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
    nuisance_bank_norm = deterministic_nuisance_bank(
        nuisance_samples,
        seed=nuisance_seed,
    )
    grid_size = grid_norm.shape[0]
    outputs = {
        "npe_log_surface": np.empty(
            (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, grid_size)
        ),
        "nre2d_log_surface": np.empty(
            (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, grid_size)
        ),
        "nre9d_log_surface": np.empty(
            (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, grid_size)
        ),
    }
    npe_ess = np.empty(
        (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, grid_size)
    )
    nre9d_ess = np.empty_like(npe_ess)
    truth_full = np.empty(
        (BENCHMARK_N_CELLS, BENCHMARK_N_GALAXIES, len(names)),
        dtype=np.float64,
    )
    for cell_id in range(BENCHMARK_N_CELLS):
        print(f"nuisance benchmark cell {cell_id + 1}/{BENCHMARK_N_CELLS}", flush=True)
        cell = _load_cell(dataset, cell_id, device, names, par_ranges)
        npe_log_integrand = None
        nre9d_log_integrand = None
        nre2d_log_surface = None
        tf_log_prior = None
        for realization in range(BENCHMARK_N_REALIZATIONS):
            print(
                f"  realization {realization + 1}/{BENCHMARK_N_REALIZATIONS}",
                flush=True,
            )
            evaluated = evaluate_cell_surfaces(
                model,
                nre_2d_head,
                nre_9d_head,
                cell,
                cell_id=cell_id,
                realization=realization,
                names=names,
                par_ranges=par_ranges,
                grid_norm=grid_norm,
                nuisance_bank_norm=nuisance_bank_norm,
                seed=seed,
                device=device,
                channels_last=channels_last,
                tf_prior=tf_prior,
                chunk_size=chunk_size,
            )
            if npe_log_integrand is None:
                npe_log_integrand = np.zeros_like(
                    evaluated["npe_log_integrand"],
                    dtype=np.float64,
                )
                nre9d_log_integrand = np.zeros_like(
                    evaluated["nre9d_log_integrand"],
                    dtype=np.float64,
                )
                nre2d_log_surface = np.zeros_like(
                    evaluated["nre2d_log_surface"],
                    dtype=np.float64,
                )
                tf_log_prior = np.asarray(
                    evaluated["tf_log_prior"],
                    dtype=np.float64,
                )
            npe_log_integrand += evaluated["npe_log_integrand"]
            nre9d_log_integrand += evaluated["nre9d_log_integrand"]
            nre2d_log_surface += evaluated["nre2d_log_surface"]
            truth_full[cell_id] = evaluated["truth_full"]

        if (
            npe_log_integrand is None
            or nre9d_log_integrand is None
            or nre2d_log_surface is None
            or tf_log_prior is None
        ):
            raise RuntimeError("no noise realizations were evaluated")
        # The galaxy-level prior is applied once, after multiplying the
        # repeated-noise likelihoods conditional on the same latent nuisance.
        npe_log_integrand += tf_log_prior
        nre9d_log_integrand += tf_log_prior
        npe_surface = np.empty((BENCHMARK_N_GALAXIES, grid_size))
        nre9d_surface = np.empty_like(npe_surface)
        npe_cell_ess = np.empty_like(npe_surface)
        nre9d_cell_ess = np.empty_like(npe_surface)
        for galaxy in range(BENCHMARK_N_GALAXIES):
            npe_surface[galaxy], npe_cell_ess[galaxy] = (
                marginalize_nuisance_log_surface(
                    npe_log_integrand[galaxy],
                    return_ess=True,
                )
            )
            nre9d_surface[galaxy], nre9d_cell_ess[galaxy] = (
                marginalize_nuisance_log_surface(
                    nre9d_log_integrand[galaxy],
                    return_ess=True,
                )
            )
        outputs["npe_log_surface"][cell_id] = npe_surface
        outputs["nre2d_log_surface"][cell_id] = nre2d_log_surface
        outputs["nre9d_log_surface"][cell_id] = nre9d_surface
        npe_ess[cell_id] = npe_cell_ess
        nre9d_ess[cell_id] = nre9d_cell_ess
    return {
        **outputs,
        "truth_g": np.asarray(
            fixed_shear_grid(),
            dtype=np.float64,
        )[:, None, :].repeat(BENCHMARK_N_GALAXIES, axis=1),
        "truth_full": truth_full,
        "npe_nuisance_ess": npe_ess,
        "nre9d_nuisance_ess": nre9d_ess,
        "nuisance_bank_norm": nuisance_bank_norm,
        "grid_phys": grid_phys,
        "prefixes": np.asarray(NUISANCE_PREFIXES, dtype=np.int64),
        "noise_realizations": np.asarray(BENCHMARK_N_REALIZATIONS),
    }


def _bootstrap_metric_intervals(
    truth: np.ndarray,
    galaxy_log_surfaces: np.ndarray,
    grid_phys: np.ndarray,
    *,
    n_nuisance: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return paired-bootstrap intervals for posterior means and grid MAPs."""

    truth = np.asarray(truth, dtype=np.float64)
    surfaces = np.asarray(galaxy_log_surfaces, dtype=np.float64)
    grid_phys = np.asarray(grid_phys, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth must have shape (cells, 2)")
    if surfaces.ndim != 3 or surfaces.shape[0] != len(truth):
        raise ValueError(
            "galaxy_log_surfaces must have shape (cells, galaxies, grid)"
        )
    if grid_phys.ndim != 2 or grid_phys.shape[-1] != 2:
        raise ValueError("grid_phys must have shape (grid, 2)")
    if not 1 <= int(n_nuisance) <= surfaces.shape[1]:
        raise ValueError("n_nuisance must be within the galaxy axis")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be positive")

    metric_names = (
        "g1_m",
        "g2_m",
        "combined_m",
        "g1_c",
        "g2_c",
        "combined_c",
        "g1_rmse",
        "g2_rmse",
        "combined_rmse",
    )
    samples = {
        "mean": {name: np.empty(n_bootstrap, dtype=np.float64) for name in metric_names},
        "map": {name: np.empty(n_bootstrap, dtype=np.float64) for name in metric_names},
    }
    # Keep the temporary sampled surface small; a full (bootstrap, galaxy,
    # grid) allocation is unnecessarily large for the 441-point grid.
    chunk_size = 32
    bootstrap_ids = paired_nuisance_bootstrap_indices(
        surfaces.shape[1],
        int(n_nuisance),
        n_bootstrap=int(n_bootstrap),
        seed=int(seed),
    )
    for start in range(0, int(n_bootstrap), chunk_size):
        stop = min(start + chunk_size, int(n_bootstrap))
        sampled = np.take(surfaces, bootstrap_ids[start:stop], axis=1)
        stacked = np.sum(sampled, axis=2)
        flat_stacked = stacked.reshape(-1, stacked.shape[-1])
        means = grid_posterior_moments(flat_stacked, grid_phys)["mean"].reshape(
            len(truth), stop - start, 2
        )
        maps = grid_phys[np.nanargmax(flat_stacked, axis=1)].reshape(
            len(truth), stop - start, 2
        )
        for index in range(stop - start):
            for kind, estimates in (("mean", means[:, index]), ("map", maps[:, index])):
                fit = fit_component_mc(truth, estimates)
                fit.update(rmse(truth, estimates))
                for name in metric_names:
                    samples[kind][name][start + index] = fit[name]

    return {
        kind: {
            **{
                f"{name}_median": float(np.median(values))
                for name, values in values_by_name.items()
            },
            **{
                interval_name: float(
                    np.percentile(
                        values,
                        (
                            NUISANCE_BOOTSTRAP_LOWER_PERCENTILE
                            if interval_name.endswith("_lower")
                            else NUISANCE_BOOTSTRAP_UPPER_PERCENTILE
                        ),
                    ),
                )
                for name, values in values_by_name.items()
                for interval_name in (f"{name}_lower", f"{name}_upper")
            }
        }
        for kind, values_by_name in samples.items()
    }


def aggregate_results(
    result: dict[str, np.ndarray],
    *,
    bootstrap_count: int = NUISANCE_BOOTSTRAP_COUNT,
    bootstrap_seed: int = NUISANCE_BOOTSTRAP_SEED_OFFSET,
) -> tuple[list[dict], list[dict]]:
    truth = np.asarray(result["truth_g"], dtype=np.float64)[:, 0]
    grid_phys = np.asarray(result["grid_phys"], dtype=np.float64)
    surface_names = {
        "npe_mean": "npe_log_surface",
        "npe_map": "npe_log_surface",
        "nre2d_mean": "nre2d_log_surface",
        "nre2d_map": "nre2d_log_surface",
        "nre9d_mean": "nre9d_log_surface",
        "nre9d_map": "nre9d_log_surface",
    }
    if int(bootstrap_count) <= 0:
        raise ValueError("bootstrap_count must be positive")
    bootstrap_cache: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    metrics = []
    per_shear = []
    for estimator, _ in ESTIMATORS:
        surface_name = surface_names[estimator]
        surface = np.asarray(result[surface_name], dtype=np.float64)
        stacked = stack_log_surfaces(
            surface,
            NUISANCE_PREFIXES,
        )
        for prefix_index, prefix in enumerate(NUISANCE_PREFIXES):
            posterior = grid_posterior_moments(
                stacked[:, prefix_index],
                grid_phys,
            )
            mean_estimate = np.asarray(posterior["mean"], dtype=np.float64)
            map_estimate = np.asarray(
                [grid_map(row, grid_phys) for row in stacked[:, prefix_index]],
                dtype=np.float64,
            )
            cell_estimate = (
                mean_estimate
                if estimator.endswith("_mean")
                else map_estimate
            )
            fit = fit_component_mc(truth, cell_estimate)
            cache_key = (surface_name, int(prefix))
            if cache_key not in bootstrap_cache:
                bootstrap_cache[cache_key] = _bootstrap_metric_intervals(
                    truth,
                    surface,
                    grid_phys,
                    n_nuisance=int(prefix),
                    n_bootstrap=int(bootstrap_count),
                    seed=int(bootstrap_seed),
                )
            estimator_kind = "mean" if estimator.endswith("_mean") else "map"
            metrics.append(
                {
                    "estimator": estimator,
                    "n_nuisance": int(prefix),
                    **fit,
                    **rmse(truth, cell_estimate),
                    "bootstrap_count": int(bootstrap_count),
                    "bootstrap_seed": int(bootstrap_seed),
                    "bootstrap_lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
                    "bootstrap_upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
                    "bootstrap_center": "prefix",
                    **bootstrap_cache[cache_key][estimator_kind],
                }
            )
            for cell_id, shear in enumerate(fixed_shear_grid()):
                per_shear.append(
                    {
                        "estimator": estimator,
                        "n_nuisance": int(prefix),
                        "cell": int(cell_id),
                        "g1_truth": float(shear[0]),
                        "g2_truth": float(shear[1]),
                        "g1_bias": float(
                            cell_estimate[cell_id, 0] - shear[0]
                        ),
                        "g2_bias": float(
                            cell_estimate[cell_id, 1] - shear[1]
                        ),
                    }
                )
    return metrics, per_shear


def _plot_outputs(
    metrics: list[dict],
    per_shear: list[dict],
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
    # The x-axis is logarithmic base 2, so separate nearby series by small
    # multiplicative offsets in log2 space rather than additive linear shifts.
    log2_offsets = np.linspace(-0.16, 0.16, len(ESTIMATORS))
    figures = {}
    for metric_name, ylabel, filename in (
        ("combined_m", "combined multiplicative bias m", "m_vs_nuisance.png"),
        ("combined_rmse", "combined RMSE", "rmse_vs_nuisance.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        for estimator_index, (color, (name, label)) in enumerate(
            zip(colors, ESTIMATORS)
        ):
            subset = sorted(
                (row for row in metrics if row["estimator"] == name),
                key=lambda row: row["n_nuisance"],
            )
            base_x_values = np.asarray(
                [row["n_nuisance"] for row in subset],
                dtype=np.float64,
            )
            x_values = base_x_values * (2.0 ** log2_offsets[estimator_index])
            point_values = np.asarray(
                [row[f"{metric_name}_median"] for row in subset],
                dtype=np.float64,
            )
            lower_values = np.asarray(
                [row[f"{metric_name}_lower"] for row in subset],
                dtype=np.float64,
            )
            upper_values = np.asarray(
                [row[f"{metric_name}_upper"] for row in subset],
                dtype=np.float64,
            )
            # The bootstrap median lies inside its percentile interval, so
            # these are genuine asymmetric errors around the plotted center.
            lower_errors = point_values - lower_values
            upper_errors = upper_values - point_values
            axis.errorbar(
                x_values,
                point_values,
                yerr=np.vstack((lower_errors, upper_errors)),
                fmt="o-",
                color=color,
                label=label,
                capsize=3,
                capthick=0.9,
                elinewidth=0.9,
                markersize=4,
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
            rows = [
                row
                for row in per_shear
                if row["estimator"] == name and row["n_nuisance"] == prefix
            ]
            rows = sorted(rows, key=lambda row: row["cell"])
            truth = np.asarray(
                [[row["g1_truth"], row["g2_truth"]] for row in rows],
                dtype=np.float64,
            )
            bias = np.asarray(
                [[row["g1_bias"], row["g2_bias"]] for row in rows],
                dtype=np.float64,
            )
            for axis, component in zip(axes[:, column], (0, 1)):
                axis.scatter(
                    truth[:, component],
                    bias[:, component],
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
    ess = payload["nuisance_integration_ess"]
    ess_rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['median']:.1f}</td>"
        f"<td>{value['minimum']:.1f}</td><td>{value['mean']:.1f}</td></tr>"
        for key, value in ess.items()
    )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Strict BA14 stacked estimator benchmark</title>
<script>
MathJax = {{tex: {{inlineMath: [['\\\\(', '\\\\)']]}}, svg: {{fontCache: 'global'}}}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
body {{ font: 17px/1.55 Palatino, serif; margin: 2rem auto; max-width: 1180px; }}
h1, h2 {{ font-weight: 600; }} figure {{ margin: 1.4rem 0 2rem; }}
figcaption, .note {{ color: #444; }} img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: .32rem .45rem; }}
th {{ background: #f4f4f4; text-align: left; }}
.equation {{ overflow-x: auto; margin: 1.1rem 0; }}
</style></head><body>
<h1>Strict Bernstein–Armstrong stacked estimator benchmark</h1>
<p class="lead">A controlled 5×5 fixed-shear grid compares six
likelihood-stack-then-estimate estimators as paired nuisance-galaxy prefixes
grow. The benchmark uses the first {BENCHMARK_N_REALIZATIONS} independent
noise realizations for every latent galaxy and holds the noise prefix fixed at
R={BENCHMARK_N_REALIZATIONS} while varying the nuisance-galaxy count.</p>
<h2>Design</h2>
<p>There are 25 exact shear cells and 32 paired nuisance galaxies per cell.
The nuisance prefixes are {", ".join(str(value) for value in NUISANCE_PREFIXES)}.
For every estimator, the likelihood contributions from all R noise
realizations of a galaxy are combined first at fixed galaxy-specific nuisance
parameters. Those nuisance dimensions are then marginalized, and the resulting
per-galaxy log likelihoods are summed over each nuisance prefix before
computing the shared-shear posterior mean or grid MAP. Both 9D paths use the
configured truncated-normal Tully–Fisher prior on <code>vcirc</code>; the 2D
NRE head is already nuisance-marginalized.</p>
<h2>Connection to Bernstein &amp; Armstrong (2014)</h2>
<p>The <a href="https://arxiv.org/abs/1304.1843">Bernstein &amp; Armstrong
(2014)</a> construction treats the shear as shared by independent galaxies
and combines their likelihood contributions before estimating the shear. Here
<code>a</code> denotes one of the six estimators, <code>D</code> is the noisy
observation, <code>g</code> is the shared two-component shear, and
<code>&eta;</code> contains the seven nuisance parameters of one latent
galaxy.</p>
<div class="equation">\\[
\\mathcal{{L}}_{{a,n,R}}(\\mathbf{{g}})
=
\\int
\\exp\\!\\left[
\\sum_{{r=1}}^R
\\ell_{{a,n,r}}(\\mathbf{{g}},\\boldsymbol{{\\eta}}_n)
\\right]
\\pi(\\boldsymbol{{\\eta}}_n)
\\,d\\boldsymbol{{\\eta}}_n .
\\]</div>
<p>This is the strict repeated-exposure order: multiply the likelihoods for
the R noise realizations conditional on the same latent galaxy nuisance
parameters, then marginalize those parameters. The nuisance-galaxy stack is</p>
<div class="equation">\\[
\\log \\mathcal{{L}}_{{a,N,R}}(\\mathbf{{g}})
=
\\sum_{{n=1}}^N
\\log \\mathcal{{L}}_{{a,n,R}}(\\mathbf{{g}}),
\\qquad
p(\\mathbf{{g}}\\mid D)
\\propto
\\pi(\\mathbf{{g}})
\\exp\\!\\left[
\\log \\mathcal{{L}}_{{a,N,R}}(\\mathbf{{g}})
\\right].
\\]</div>
<p>The posterior mean and MAP are evaluated only after this complete
likelihood stack:</p>
<div class="equation">\\[
\\widehat{{\\mathbf{{g}}}}_{{\\rm mean}}
=
\\frac{{\\sum_k \\mathbf{{g}}_k w_k}}{{\\sum_k w_k}},
\\qquad
\\widehat{{\\mathbf{{g}}}}_{{\\rm MAP}}
=
\\mathbf{{g}}_{{\\arg\\max_k w_k}},
\\qquad
w_k=\\pi(\\mathbf{{g}}_k)
\\exp\\!\\left[
\\log \\mathcal{{L}}_{{a,N,R}}(\\mathbf{{g}}_k)
\\right].
\\]</div>
<p>For the 2D NRE estimator, <code>&ell;</code> is its directly predicted
two-dimensional log-ratio surface. For the 9D NPE and NRE estimators,
<code>&ell;</code> is the prior-corrected 9D log-ratio surface; the seven
nuisance coordinates are integrated after the realization-axis sum. Thus no
per-galaxy mean or MAP is formed before either likelihood stack.</p>
<h2>Bias and RMSE versus nuisance count</h2>
<p class="note">Markers show the bootstrap median. Clean vertical error bars
show the {payload["bootstrap"]["lower_percentile"]:.0f}th–{payload["bootstrap"]["upper_percentile"]:.0f}th
percentile interval from {payload["bootstrap"]["count"]:,} paired bootstrap
resamples of the nuisance galaxies. The six series use small multiplicative
offsets in log<sub>2</sub>(N) space to keep overlapping error bars legible;
the tick labels remain the actual nuisance counts. The error bars include no
shaded regions. The un-suffixed metric values in the JSON retain the
deterministic fixed-prefix results.</p>
<figure><img src="{html.escape(figures['m_vs_nuisance'])}"
alt="bias versus nuisance count"></figure>
<figure><img src="{html.escape(figures['rmse_vs_nuisance'])}"
alt="RMSE versus nuisance count"></figure>
<h2>Nuisance integration effective sample size</h2>
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


def _load_saved_result(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Load inference surfaces and metadata for report-only regeneration."""

    with np.load(path, allow_pickle=False) as saved:
        if "meta_json" not in saved:
            raise ValueError(f"{path} does not contain benchmark metadata")
        result = {
            name: np.asarray(saved[name])
            for name in saved.files
            if name != "meta_json"
        }
        meta = json.loads(str(np.asarray(saved["meta_json"]).item()))
    return result, meta


def _write_report(
    result: dict[str, np.ndarray],
    meta: dict,
    report_dir: Path,
    *,
    bootstrap_count: int,
    bootstrap_seed: int,
    save_npz: bool,
) -> None:
    """Aggregate saved surfaces and write the HTML/JSON report and figures."""

    metrics, per_shear = aggregate_results(
        result,
        bootstrap_count=bootstrap_count,
        bootstrap_seed=bootstrap_seed,
    )
    figures = _plot_outputs(metrics, per_shear, report_dir)
    ess_summary = {
        "9D NPE nuisance integration ESS after noise stack": _summarize_ess(
            result["npe_nuisance_ess"]
        ),
        "9D NRE nuisance integration ESS after noise stack": _summarize_ess(
            result["nre9d_nuisance_ess"]
        ),
    }
    meta = {
        **meta,
        "schema": "likelihood-stack-strict-ba14-nuisance-benchmark-v4",
        "bootstrap": {
            "count": int(bootstrap_count),
            "seed": int(bootstrap_seed),
            "lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
            "upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
            "resampling_unit": "nuisance galaxies, paired across shear cells",
            "center": "fixed prefix stack",
        },
    }
    payload = {
        **meta,
        "metrics": metrics,
        "per_shear": per_shear,
        "nuisance_integration_ess": ess_summary,
        "figures": figures,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    if save_npz:
        np.savez_compressed(
            report_dir / "benchmark.npz",
            **result,
            meta_json=json.dumps(json_safe(meta)),
        )
    write_json(report_dir / "report.json", payload)
    _write_html(payload, figures, report_dir)
    print(f"Wrote {report_dir / 'report.json'}", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if (
        args.nuisance_samples <= 0
        or args.candidate_chunk_size <= 0
        or args.grid_n < 2
    ):
        raise ValueError(
            "sample counts and candidate-chunk-size must be positive; "
            "grid-n must be at least 2"
        )
    if args.bootstrap_count <= 0:
        raise ValueError("bootstrap-count must be positive")
    if args.report_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.report_dir} exists; use --overwrite")
    if args.from_npz is not None:
        result, meta = _load_saved_result(args.from_npz)
        bootstrap_seed = (
            int(meta.get("seed", BENCHMARK_NOISE_SEED))
            + NUISANCE_BOOTSTRAP_SEED_OFFSET
            if args.bootstrap_seed is None
            else int(args.bootstrap_seed)
        )
        _write_report(
            result,
            meta,
            args.report_dir,
            bootstrap_count=args.bootstrap_count,
            bootstrap_seed=bootstrap_seed,
            save_npz=False,
        )
        return
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
    # The original 2D NRE checkpoint predates the explicit parameter_dim
    # metadata field; load_ratio_head derives the dimension from target_names
    # (or the 2D default) and constructs the correct head in that case.
    if int(getattr(nre2d_head, "parameter_dim", 0)) != 2:
        raise ValueError("the 2D NRE checkpoint must have parameter_dim=2")
    if int(getattr(nre9d_head, "parameter_dim", 0)) != 9:
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
        grid_n=args.grid_n,
        seed=args.seed,
        nuisance_samples=args.nuisance_samples,
        nuisance_seed=args.nuisance_seed,
        chunk_size=args.candidate_chunk_size,
        tf_prior=tf_prior,
    )
    meta = {
        "schema": "likelihood-stack-strict-ba14-nuisance-benchmark-v4",
        "surface_definition": (
            "sum log likelihood-ratio surfaces over the realization prefix "
            "at fixed galaxy nuisance, marginalize galaxy-specific nuisance "
            "parameters, sum log surfaces over the shared-shear prefix, then "
            "estimate"
        ),
        "stack_order": [
            "noise_realizations_at_fixed_galaxy_nuisance",
            "marginalize_galaxy_specific_nuisance",
            "nuisance_galaxy_prefix",
            "posterior_mean_or_map",
        ],
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
        "noise_realizations": int(BENCHMARK_N_REALIZATIONS),
        "nuisance_samples": int(args.nuisance_samples),
        "nuisance_seed": int(args.nuisance_seed),
        "candidate_chunk_size": int(args.candidate_chunk_size),
        "grid_n": int(args.grid_n),
        "tf_prior": tf_prior.to_dict(),
        "tf_prior_applied_to": ["npe_9d", "nre_9d"],
        "nre_2d_meta": nre2d_meta,
        "nre_9d_meta": nre9d_meta,
        "estimators": [
            {"name": name, "label": label}
            for name, label in ESTIMATORS
        ],
        "truth_is_candidate": False,
        "identity_only": True,
    }
    bootstrap_seed = (
        int(meta["seed"]) + NUISANCE_BOOTSTRAP_SEED_OFFSET
        if args.bootstrap_seed is None
        else int(args.bootstrap_seed)
    )
    _write_report(
        result,
        meta,
        args.report_dir,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=bootstrap_seed,
        save_npz=True,
    )


if __name__ == "__main__":
    main()
