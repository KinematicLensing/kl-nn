#!/usr/bin/env python3
"""Evaluate five estimators on the exact fixed-shear benchmark."""

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
from data import apply_central_halpha_snr_noise, apply_image_noise_for_snr
from diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_DATASET,
    BENCHMARK_GRID_N,
    BENCHMARK_NPE_SAMPLES,
    BENCHMARK_N_GALAXIES,
    BENCHMARK_N_REALIZATIONS,
    BENCHMARK_PREFIXES,
    BENCHMARK_REPORT_DIR,
    BENCHMARK_SHEAR_VALUES,
    BENCHMARK_SURFACE_SAMPLES,
    BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY,
    BENCHMARK_NOISE_SEED,
    benchmark_noise_seeds,
    fit_component_mc,
    fixed_shear_grid,
    prefix_surface_map,
    prefix_surface_mean,
    rmse,
    row_index,
)
from diagnostics.likelihood_stack_core import (
    DATA_ROOT,
    G01_NPE,
    MODEL_ROOT,
    json_safe,
    shear_columns,
    write_json,
)
from diagnostics.likelihood_stack_nre_core import (
    RatioHead,
    grid_map,
    grid_posterior_moments,
    normalized_shear_grid,
    score_shear_grid,
)
from diagnostics.likelihood_stack_nre_train import (
    load_frozen_parent,
    load_ratio_head,
    nre_checkpoint_path,
)
from model_registry import load_model_config
from train import _seeded_generator, build_observation_levels, load_model, seed_everything, validate_observation_record
from utils import denormalize
from cache_posteriors import physical_log_prob_from_normalized
from networks import KLNPE


FIGURE_DPI = 150
ESTIMATORS = (
    ("npe_mean", "NPE posterior mean"),
    ("nre_mean", "NRE 2D posterior mean"),
    ("npe_map", "NPE sampled MAP"),
    ("nre_map", "NRE 2D grid MAP"),
    ("npe_9d_stack_map", "stacked 9D NPE q/pi MAP"),
    ("npe_9d_stack_mean", "stacked 9D NPE q/pi posterior mean"),
    ("nre_stack_mean", "stacked NRE posterior mean"),
)
PER_GALAXY_ESTIMATORS = ESTIMATORS[:-1]
NUISANCE_PREFIXES = (1, 2, 4, 8, 16, 32)
NUISANCE_BOOTSTRAP_COUNT = 1000
NUISANCE_BOOTSTRAP_LOWER_PERCENTILE = 16.0
NUISANCE_BOOTSTRAP_UPPER_PERCENTILE = 84.0
NUISANCE_BOOTSTRAP_SEED_OFFSET = 1_000_003


def available_estimators(result: dict[str, np.ndarray]) -> tuple[tuple[str, str], ...]:
    """Return estimators supported by the saved benchmark arrays."""

    if {"nre_log_r", "nre_grid_phys", "surface_mean"} <= set(result):
        return ESTIMATORS
    return PER_GALAXY_ESTIMATORS


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_ROOT / BENCHMARK_DATASET)
    parser.add_argument("--noise-data", type=Path, default=None)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=None)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--parent-checkpoint-suffix", default="best")
    parser.add_argument("--nre-checkpoint-suffix", default="best")
    parser.add_argument("--report-dir", type=Path, default=BENCHMARK_REPORT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--npe-samples", type=int, default=BENCHMARK_NPE_SAMPLES)
    parser.add_argument("--surface-samples", type=int, default=BENCHMARK_SURFACE_SAMPLES)
    parser.add_argument("--grid-n", type=int, default=BENCHMARK_GRID_N)
    parser.add_argument("--batch-size", type=int, default=BENCHMARK_N_GALAXIES)
    parser.add_argument("--seed", type=int, default=BENCHMARK_NOISE_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-draws", action="store_true")
    return parser.parse_args(argv)


def _checkpoint_file(model_root: Path, model_name: str, suffix: str) -> Path:
    return Path(model_root) / model_name / f"{model_name}{suffix}"


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _load_cell(dataset, cell: int, device: torch.device, names, par_ranges):
    indices = np.asarray(
        [row_index(cell, galaxy) for galaxy in range(BENCHMARK_N_GALAXIES)],
        dtype=np.int64,
    )
    records = [dataset[int(index)] for index in indices]
    metadata = [
        validate_observation_record(record, location=f"benchmark record {int(index)}")
        for record, index in zip(records, indices)
    ]
    image = torch.stack(
        [torch.as_tensor(record["img"]).float() for record in records]
    ).to(device)
    spec = torch.stack(
        [torch.as_tensor(record["spec"]).float() for record in records]
    ).to(device)
    fib_pos = torch.stack(
        [torch.as_tensor(record["fib_pos"]).float() for record in records]
    ).to(device)
    rmag = torch.tensor([row[0] for row in metadata], device=device)
    image_snr, spec_snr = build_observation_levels(
        torch.tensor([row[2] for row in metadata], device=device),
        torch.tensor([row[3] for row in metadata], device=device),
    )
    labels = torch.stack(
        [torch.as_tensor(record["fid_pars"]).float() for record in records]
    )
    truth = np.asarray(
        denormalize(
            labels,
            par_ranges,
            feature_names=names,
            target_transforms=config.TARGET_TRANSFORMS,
        ),
        dtype=np.float64,
    )
    return {
        "indices": indices,
        "img": image,
        "spec": spec,
        "fib_pos": fib_pos,
        "rmag": rmag,
        "image_snr": image_snr,
        "spec_snr": spec_snr,
        "truth": truth,
    }


def _observation_context(cell: dict) -> dict[str, torch.Tensor]:
    return {
        "rmag_true": cell["rmag"],
        "image_snr": cell["image_snr"],
        "central_halpha_snr": cell["spec_snr"],
    }


def _apply_benchmark_noise(
    cell: dict,
    *,
    cell_id: int,
    realization: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = int(config.observation["center_fiber_index"])
    images = []
    spectra = []
    for galaxy in range(BENCHMARK_N_GALAXIES):
        image_seed, spec_seed = benchmark_noise_seeds(
            seed, cell_id, galaxy, realization
        )
        images.append(
            apply_image_noise_for_snr(
                cell["img"][galaxy : galaxy + 1],
                cell["image_snr"][galaxy : galaxy + 1],
                randgen=_seeded_generator(device, image_seed),
            )
        )
        spectra.append(
            apply_central_halpha_snr_noise(
                cell["spec"][galaxy : galaxy + 1],
                cell["spec_snr"][galaxy : galaxy + 1],
                center_fiber_index=center,
                center_exposure_s=config.observation["center_exposure_s"],
                offset_exposure_s=config.observation["offset_exposure_s"],
                spectral_units=config.observation["spectral_units"],
                randgen=_seeded_generator(device, spec_seed),
                device=device,
            )
        )
    return torch.cat(images, dim=0), torch.cat(spectra, dim=0)


@torch.inference_mode()
def _flow_context(model, image, spec, cell, *, channels_last: bool):
    if channels_last:
        image = image.contiguous(memory_format=torch.channels_last)
        spec = spec.contiguous(memory_format=torch.channels_last)
    return model._flow_context(
        model._raw_features(
            image,
            spec,
            cell["fib_pos"],
            _observation_context(cell),
        )
    )


@torch.inference_mode()
def _per_stamp_estimators(
    model,
    nre_head: RatioHead,
    image,
    spec,
    cell,
    *,
    names,
    par_ranges,
    grid_norm: np.ndarray,
    grid_phys: np.ndarray,
    npe_samples: int,
    seed: int,
    cell_id: int,
    realization: int,
    channels_last: bool,
):
    # Sampling randomness is keyed separately from image/spectral noise so a
    # rerun cannot accidentally couple a posterior draw to an image stream.
    sample_seed = int(
        np.random.SeedSequence(
            [int(seed), int(cell_id), int(realization), 991]
        ).generate_state(1, dtype=np.uint64)[0]
        % (2**63 - 1)
    )
    torch.manual_seed(sample_seed)
    if image.is_cuda:
        torch.cuda.manual_seed_all(sample_seed)
    context = _flow_context(model, image, spec, cell, channels_last=channels_last)
    samples, normalized_log_prob = model.flow.sample_and_log_prob(
        int(npe_samples), context=context
    )
    physical = denormalize(
        samples,
        par_ranges,
        feature_names=names,
        target_transforms=config.TARGET_TRANSFORMS,
    )
    g1_index, g2_index = shear_columns(names)
    npe_mean = torch.stack(
        (
            physical[..., g1_index].mean(dim=1),
            physical[..., g2_index].mean(dim=1),
        ),
        dim=-1,
    )
    physical_log_prob = physical_log_prob_from_normalized(
        samples.detach().cpu().numpy(),
        normalized_log_prob.detach().cpu().numpy(),
        par_ranges=par_ranges,
        feature_names=names,
        target_transforms=config.TARGET_TRANSFORMS,
    )
    map_index = np.argmax(
        np.where(np.isfinite(physical_log_prob), physical_log_prob, -np.inf),
        axis=1,
    )
    physical_np = physical.detach().cpu().numpy()
    npe_map = physical_np[np.arange(len(physical_np)), map_index][:, [g1_index, g2_index]]
    candidates = samples[:, :BENCHMARK_SURFACE_SAMPLES].detach().cpu().numpy()

    grid_tensor = torch.as_tensor(grid_norm, device=image.device, dtype=torch.float32)
    nre_log_r = score_shear_grid(nre_head, context, grid_tensor).detach().cpu().numpy()
    nre_mean = grid_posterior_moments(nre_log_r, grid_phys)["mean"]
    nre_map = np.asarray(
        [grid_map(row, grid_phys) for row in nre_log_r], dtype=np.float64
    )
    return {
        "npe_mean": npe_mean.detach().cpu().numpy(),
        "npe_map": np.asarray(npe_map, dtype=np.float64),
        "nre_mean": np.asarray(nre_mean, dtype=np.float64),
        "nre_map": nre_map,
        "nre_log_r": np.asarray(nre_log_r, dtype=np.float64),
        "surface_candidates": candidates,
        "context": context,
    }


@torch.inference_mode()
def _surface_stack(
    model,
    images: torch.Tensor,
    spectra: torch.Tensor,
    cell: dict,
    candidates: np.ndarray,
    *,
    prefixes=BENCHMARK_PREFIXES,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every observation on its galaxy's unioned 9D candidate bank."""

    n_gal, n_real, n_candidates, n_features = candidates.shape
    if n_real != BENCHMARK_N_REALIZATIONS:
        raise ValueError("surface candidates must contain 16 realizations")
    score_width = n_real * n_candidates
    scores = np.empty((n_gal, n_real, score_width), dtype=np.float32)
    # Call KLNPE.posterior_log_prob in bounded candidate chunks. The method
    # expands the flow context across candidates, so an all-at-once
    # (512 observations × 4096 candidates × 1152 context) call is needlessly
    # large on a 32 GB V100.
    for galaxy in range(n_gal):
        image = images[galaxy]
        spectrum = spectra[galaxy]
        fib_pos = cell["fib_pos"][galaxy].unsqueeze(0).expand(
            n_real, *cell["fib_pos"].shape[1:]
        )
        observation_context = {
            "rmag_true": cell["rmag"][galaxy].expand(n_real),
            "image_snr": cell["image_snr"][galaxy].expand(n_real),
            "central_halpha_snr": cell["spec_snr"][galaxy].expand(n_real),
        }
        bank = torch.as_tensor(
            candidates[galaxy].reshape(score_width, n_features),
            device=images.device,
            dtype=torch.float32,
        )
        for start in range(0, score_width, 512):
            stop = min(start + 512, score_width)
            parameters = bank[start:stop].unsqueeze(0).expand(
                n_real, stop - start, n_features
            )
            scores[galaxy, :, start:stop] = (
                model.posterior_log_prob(
                    image,
                    spectrum,
                    parameters,
                    fib_pos,
                    observation_context,
                )
                .detach()
                .cpu()
                .numpy()
            )
    scores_np = scores
    output_map = np.empty((n_gal, len(prefixes), n_features), dtype=np.float64)
    output_mean = np.empty((n_gal, len(prefixes), n_features), dtype=np.float64)
    for galaxy in range(n_gal):
        mapped, _ = prefix_surface_map(
            candidates[galaxy],
            scores_np[galaxy],
            prefixes=prefixes,
            candidates_per_realization=n_candidates,
            prior_log_density=BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY,
        )
        means = prefix_surface_mean(
            candidates[galaxy],
            scores_np[galaxy],
            prefixes=prefixes,
            candidates_per_realization=n_candidates,
            prior_log_density=BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY,
        )
        output_map[galaxy] = denormalize(
            mapped,
            config.par_ranges,
            feature_names=tuple(config.TARGET_NAMES),
            target_transforms=config.TARGET_TRANSFORMS,
        )
        output_mean[galaxy] = denormalize(
            means,
            config.par_ranges,
            feature_names=tuple(config.TARGET_NAMES),
            target_transforms=config.TARGET_TRANSFORMS,
        )
    return output_map, output_mean


def run_benchmark(
    dataset,
    model,
    nre_head,
    *,
    names,
    par_ranges,
    channels_last: bool,
    device: torch.device,
    npe_samples: int,
    grid_n: int,
    seed: int,
    noisy_data: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    grid_norm, _ = normalized_shear_grid(grid_n)
    grid_phys = np.asarray(
        denormalize(
            grid_norm,
            par_ranges,
            feature_names=("g1", "g2"),
        ),
        dtype=np.float64,
    )
    g1_index, g2_index = shear_columns(names)
    n_cells = len(fixed_shear_grid())
    shape = (n_cells, BENCHMARK_N_GALAXIES, BENCHMARK_N_REALIZATIONS, 2)
    point = {name: np.empty(shape, dtype=np.float64) for name in (
        "npe_mean", "nre_mean", "npe_map", "nre_map"
    )}
    surface_map = np.empty(
        (n_cells, BENCHMARK_N_GALAXIES, len(BENCHMARK_PREFIXES), len(names)),
        dtype=np.float64,
    )
    surface_mean = np.empty_like(surface_map)
    nre_log_r = np.empty(
        (
            n_cells,
            BENCHMARK_N_GALAXIES,
            BENCHMARK_N_REALIZATIONS,
            grid_phys.shape[0],
        ),
        dtype=np.float64,
    )
    truth_g = np.empty((n_cells, BENCHMARK_N_GALAXIES, 2), dtype=np.float64)
    truth_full = np.empty((n_cells, BENCHMARK_N_GALAXIES, len(names)), dtype=np.float64)
    for cell_id in range(n_cells):
        print(f"benchmark cell {cell_id + 1}/{n_cells}", flush=True)
        cell = _load_cell(dataset, cell_id, device, names, par_ranges)
        truth_full[cell_id] = cell["truth"]
        truth_g[cell_id, :, 0] = fixed_shear_grid()[cell_id, 0]
        truth_g[cell_id, :, 1] = fixed_shear_grid()[cell_id, 1]
        candidates = np.empty(
            (
                BENCHMARK_N_GALAXIES,
                BENCHMARK_N_REALIZATIONS,
                BENCHMARK_SURFACE_SAMPLES,
                len(names),
            ),
            dtype=np.float32,
        )
        noisy_images = []
        noisy_spectra = []
        for realization in range(BENCHMARK_N_REALIZATIONS):
            if noisy_data is None:
                image, spec = _apply_benchmark_noise(
                    cell,
                    cell_id=cell_id,
                    realization=realization,
                    seed=seed,
                    device=device,
                )
            else:
                start = cell_id * BENCHMARK_N_GALAXIES
                stop = start + BENCHMARK_N_GALAXIES
                image = torch.as_tensor(
                    noisy_data["images"][start:stop, realization],
                    device=device,
                    dtype=torch.float32,
                )
                spec = torch.as_tensor(
                    noisy_data["spectra"][start:stop, realization],
                    device=device,
                    dtype=torch.float32,
                )
            current = _per_stamp_estimators(
                model,
                nre_head,
                image,
                spec,
                cell,
                names=names,
                par_ranges=par_ranges,
                grid_norm=grid_norm,
                grid_phys=grid_phys,
                npe_samples=npe_samples,
                seed=seed,
                cell_id=cell_id,
                realization=realization,
                channels_last=channels_last,
            )
            for name in point:
                point[name][cell_id, :, realization] = current[name]
            nre_log_r[cell_id, :, realization] = current["nre_log_r"]
            candidates[:, realization] = current["surface_candidates"]
            noisy_images.append(image)
            noisy_spectra.append(spec)
        image_stack = torch.stack(noisy_images, dim=1)
        spec_stack = torch.stack(noisy_spectra, dim=1)
        surface_map[cell_id], surface_mean[cell_id] = _surface_stack(
            model,
            image_stack,
            spec_stack,
            cell,
            candidates,
        )
        del image_stack, spec_stack
    return {
        **point,
        "nre_log_r": nre_log_r,
        "nre_grid_phys": grid_phys,
        "surface_map": surface_map,
        "surface_mean": surface_mean,
        "truth_g": truth_g,
        "truth_full": truth_full,
        "prefixes": np.asarray(BENCHMARK_PREFIXES, dtype=np.int64),
    }


def stacked_nre_means(
    log_r: np.ndarray,
    grid_phys: np.ndarray,
    prefixes: tuple[int, ...] = BENCHMARK_PREFIXES,
) -> np.ndarray:
    """Posterior means after summing NRE log-ratio surfaces over nuisances."""

    scores = np.asarray(log_r, dtype=np.float64)
    grid_phys = np.asarray(grid_phys, dtype=np.float64)
    if scores.ndim != 4:
        raise ValueError(
            "log_r must have shape (cells, galaxies, realizations, grid)"
        )
    if grid_phys.ndim != 2 or grid_phys.shape[-1] != 2:
        raise ValueError("grid_phys must have shape (grid, 2)")
    if scores.shape[-1] != grid_phys.shape[0]:
        raise ValueError("log_r width must match grid_phys")
    output = np.empty(
        (scores.shape[0], len(prefixes), grid_phys.shape[1]),
        dtype=np.float64,
    )
    for prefix_index, prefix in enumerate(prefixes):
        prefix = int(prefix)
        if not 1 <= prefix <= scores.shape[2]:
            raise ValueError("prefix is outside the available realization axis")
        stacked = np.sum(scores[:, :, :prefix, :], axis=(1, 2))
        output[:, prefix_index] = grid_posterior_moments(
            stacked, grid_phys
        )["mean"]
    return output


def stacked_nre_nuisance_means(
    log_r: np.ndarray,
    grid_phys: np.ndarray,
    nuisance_prefixes: tuple[int, ...] = NUISANCE_PREFIXES,
    *,
    realization_prefix: int = BENCHMARK_N_REALIZATIONS,
) -> np.ndarray:
    """NRE posterior means after stacking nuisance prefixes at fixed R."""

    scores = np.asarray(log_r, dtype=np.float64)
    grid_phys = np.asarray(grid_phys, dtype=np.float64)
    if scores.ndim != 4:
        raise ValueError(
            "log_r must have shape (cells, galaxies, realizations, grid)"
        )
    if not 1 <= int(realization_prefix) <= scores.shape[2]:
        raise ValueError("realization_prefix is outside the available axis")
    output = np.empty(
        (scores.shape[0], len(nuisance_prefixes), grid_phys.shape[1]),
        dtype=np.float64,
    )
    for prefix_index, prefix in enumerate(nuisance_prefixes):
        prefix = int(prefix)
        if not 1 <= prefix <= scores.shape[1]:
            raise ValueError("nuisance prefix is outside the available axis")
        stacked = np.sum(
            scores[:, :prefix, :realization_prefix, :],
            axis=(1, 2),
        )
        output[:, prefix_index] = grid_posterior_moments(
            stacked, grid_phys
        )["mean"]
    return output


def aggregate_results(result: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], dict]:
    truth = result["truth_g"]
    prefixes = tuple(int(value) for value in result["prefixes"])
    estimators = available_estimators(result)
    rows = []
    cells = []
    surface = result["surface_map"][..., [0, 1]]
    surface_mean = result["surface_mean"][..., [0, 1]]
    estimates = {
        "npe_mean": result["npe_mean"],
        "nre_mean": result["nre_mean"],
        "npe_map": result["npe_map"],
        "nre_map": result["nre_map"],
        "npe_9d_stack_map": surface,
        "npe_9d_stack_mean": result["surface_mean"][..., [0, 1]],
    }
    if "nre_stack_mean" in dict(estimators):
        estimates["nre_stack_mean"] = stacked_nre_means(
            result["nre_log_r"],
            result["nre_grid_phys"],
            prefixes,
        )
    for estimator, _ in estimators:
        values = estimates[estimator]
        for prefix_index, prefix in enumerate(prefixes):
            if estimator == "npe_9d_stack_map":
                galaxy_values = values[:, :, prefix_index]
            elif estimator == "nre_stack_mean":
                galaxy_values = values[:, prefix_index]
            else:
                galaxy_values = np.mean(values[:, :, :prefix], axis=2)
            if estimator == "nre_stack_mean":
                cell_values = galaxy_values
            else:
                cell_values = np.mean(galaxy_values, axis=1)
            truth_cells = truth.mean(axis=1)
            fit = fit_component_mc(truth_cells, cell_values)
            metrics = rmse(truth_cells, cell_values)
            rows.append(
                {
                    "estimator": estimator,
                    "prefix": int(prefix),
                    **fit,
                    **metrics,
                }
            )
            for cell_id, shear in enumerate(fixed_shear_grid()):
                values_cell = (
                    galaxy_values[cell_id]
                    if estimator != "nre_stack_mean"
                    else galaxy_values[cell_id : cell_id + 1]
                )
                cells.append(
                    {
                        "estimator": estimator,
                        "prefix": int(prefix),
                        "cell": int(cell_id),
                        "g1_truth": float(shear[0]),
                        "g2_truth": float(shear[1]),
                        "g1_mean": float(np.mean(values_cell[:, 0])),
                        "g2_mean": float(np.mean(values_cell[:, 1])),
                        "g1_se": (
                            None
                            if estimator == "nre_stack_mean"
                            else float(
                                np.std(values_cell[:, 0], ddof=1)
                                / np.sqrt(len(values_cell))
                            )
                        ),
                        "g2_se": (
                            None
                            if estimator == "nre_stack_mean"
                            else float(
                                np.std(values_cell[:, 1], ddof=1)
                                / np.sqrt(len(values_cell))
                            )
                        ),
                    }
                )
    takeaway = {
        estimator: {
            str(row["prefix"]): {
                "combined_m": row["combined_m"],
                "combined_rmse": row["combined_rmse"],
            }
            for row in rows
            if row["estimator"] == estimator
        }
        for estimator, _ in estimators
    }
    return rows, cells, takeaway


def paired_nuisance_bootstrap_indices(
    n_galaxies: int,
    n_nuisance: int,
    *,
    n_bootstrap: int = NUISANCE_BOOTSTRAP_COUNT,
    seed: int = NUISANCE_BOOTSTRAP_SEED_OFFSET,
) -> np.ndarray:
    """Return paired with-replacement nuisance IDs for one nuisance count.

    Each returned row contains ``n_nuisance`` IDs drawn from all available
    nuisance galaxies.  It is one bootstrap replicate, and callers must apply
    that same row to every fixed-shear cell.  Seeding each nuisance count
    independently makes the intervals reproducible without coupling one count
    to another.
    """

    n_galaxies = int(n_galaxies)
    n_nuisance = int(n_nuisance)
    n_bootstrap = int(n_bootstrap)
    if n_galaxies <= 0:
        raise ValueError("n_galaxies must be positive")
    if not 1 <= n_nuisance <= n_galaxies:
        raise ValueError("n_nuisance must be between 1 and n_galaxies")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), n_nuisance])
    )
    return rng.integers(
        0,
        n_galaxies,
        size=(n_bootstrap, n_nuisance),
        dtype=np.int64,
    )


def _nuisance_bootstrap_fits(
    truth_cells: np.ndarray,
    galaxy_values: np.ndarray,
    *,
    n_nuisance: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, tuple[float, float]],
]:
    """Fit prefix provenance and paired-bootstrap summary statistics."""

    values = np.asarray(galaxy_values, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("galaxy_values must have shape (cells, galaxies, 2)")
    if values.shape[0] != len(truth_cells):
        raise ValueError("galaxy_values and truth_cells must share the cell axis")
    bootstrap_ids = paired_nuisance_bootstrap_indices(
        values.shape[1],
        n_nuisance,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    prefix_fit = fit_component_mc(
        truth_cells,
        np.mean(values[:, :n_nuisance], axis=1),
    )
    prefix_summary = {
        **prefix_fit,
        **rmse(truth_cells, np.mean(values[:, :n_nuisance], axis=1)),
    }
    # Advanced indexing gives (cell, bootstrap, nuisance, component).  The
    # bootstrap ID row is therefore shared across all 25 exact shear cells.
    bootstrap_cell_values = np.mean(values[:, bootstrap_ids, :], axis=2)
    bootstrap_fits = [
        fit_component_mc(truth_cells, bootstrap_cell_values[:, index, :])
        for index in range(n_bootstrap)
    ]
    bootstrap_rmses = [
        rmse(truth_cells, bootstrap_cell_values[:, index, :])
        for index in range(n_bootstrap)
    ]
    bootstrap_names = (
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
    bootstrap_median = {
        name: float(
            np.median(
                [
                    fit[name] if name in fit else rmse_row[name]
                    for fit, rmse_row in zip(bootstrap_fits, bootstrap_rmses)
                ]
            )
        )
        for name in bootstrap_names
    }
    interval = {
        name: (
            float(
                np.percentile(
                    [fit[name] for fit in bootstrap_fits],
                    NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
                )
            ),
            float(
                np.percentile(
                    [fit[name] for fit in bootstrap_fits],
                    NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
                )
            ),
        )
        for name in ("g1_m", "g2_m", "combined_m")
    }
    return prefix_summary, bootstrap_median, interval


def _nre_nuisance_bootstrap_fits(
    truth_cells: np.ndarray,
    log_r: np.ndarray,
    grid_phys: np.ndarray,
    *,
    n_nuisance: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[float, float]]]:
    """Bootstrap NRE score products over paired nuisance IDs."""

    scores = np.asarray(log_r, dtype=np.float64)
    if scores.ndim != 4:
        raise ValueError(
            "log_r must have shape (cells, galaxies, realizations, grid)"
        )
    bootstrap_ids = paired_nuisance_bootstrap_indices(
        scores.shape[1],
        n_nuisance,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    prefix_scores = np.sum(
        scores[:, :n_nuisance, :BENCHMARK_N_REALIZATIONS, :],
        axis=(1, 2),
    )
    prefix_values = grid_posterior_moments(prefix_scores, grid_phys)["mean"]
    prefix_fit = fit_component_mc(truth_cells, prefix_values)
    prefix_summary = {**prefix_fit, **rmse(truth_cells, prefix_values)}

    galaxy_scores = np.sum(
        scores[:, :, :BENCHMARK_N_REALIZATIONS, :],
        axis=2,
    )
    bootstrap_fits = []
    bootstrap_rmses = []
    for start in range(0, n_bootstrap, 16):
        stop = min(start + 16, n_bootstrap)
        sampled_scores = np.take(
            galaxy_scores,
            bootstrap_ids[start:stop],
            axis=1,
        )
        sampled_scores = np.sum(sampled_scores, axis=2)
        sampled_values = grid_posterior_moments(
            sampled_scores.reshape(-1, sampled_scores.shape[-1]),
            grid_phys,
        )["mean"].reshape(scores.shape[0], stop - start, 2)
        bootstrap_fits.extend(
            fit_component_mc(truth_cells, sampled_values[:, index, :])
            for index in range(stop - start)
        )
        bootstrap_rmses.extend(
            rmse(truth_cells, sampled_values[:, index, :])
            for index in range(stop - start)
        )
    bootstrap_names = (
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
    bootstrap_median = {
        name: float(
            np.median(
                [
                    fit[name] if name in fit else rmse_row[name]
                    for fit, rmse_row in zip(bootstrap_fits, bootstrap_rmses)
                ]
            )
        )
        for name in bootstrap_names
    }
    interval = {
        name: (
            float(
                np.percentile(
                    [fit[name] for fit in bootstrap_fits],
                    NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
                )
            ),
            float(
                np.percentile(
                    [fit[name] for fit in bootstrap_fits],
                    NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
                )
            ),
        )
        for name in ("g1_m", "g2_m", "combined_m")
    }
    return prefix_summary, bootstrap_median, interval


def aggregate_nuisance_results(
    result: dict[str, np.ndarray],
    *,
    bootstrap_count: int = NUISANCE_BOOTSTRAP_COUNT,
    bootstrap_seed: int = NUISANCE_BOOTSTRAP_SEED_OFFSET,
) -> list[dict]:
    """Fit m/c after fixing R=16 and bootstrapping nuisance galaxies."""

    truth_cells = np.mean(result["truth_g"], axis=1)
    prefixes = tuple(int(value) for value in result["prefixes"])
    try:
        r16_index = prefixes.index(BENCHMARK_N_REALIZATIONS)
    except ValueError as exc:
        raise ValueError(
            f"benchmark result does not contain R={BENCHMARK_N_REALIZATIONS}"
        ) from exc

    surface = result["surface_map"][..., [0, 1]]
    surface_mean = result["surface_mean"][..., [0, 1]]
    estimates = {
        "npe_mean": np.mean(
            result["npe_mean"][:, :, :BENCHMARK_N_REALIZATIONS, :], axis=2
        ),
        "nre_mean": np.mean(
            result["nre_mean"][:, :, :BENCHMARK_N_REALIZATIONS, :], axis=2
        ),
        "npe_map": np.mean(
            result["npe_map"][:, :, :BENCHMARK_N_REALIZATIONS, :], axis=2
        ),
        "nre_map": np.mean(
            result["nre_map"][:, :, :BENCHMARK_N_REALIZATIONS, :], axis=2
        ),
        "npe_9d_stack_map": surface[:, :, r16_index],
        "npe_9d_stack_mean": surface_mean[:, :, r16_index],
    }
    rows = []
    for estimator, _ in PER_GALAXY_ESTIMATORS:
        galaxy_values = estimates[estimator]
        for n_nuisance in NUISANCE_PREFIXES:
            prefix_summary, bootstrap_median, intervals = _nuisance_bootstrap_fits(
                truth_cells,
                galaxy_values,
                n_nuisance=n_nuisance,
                n_bootstrap=bootstrap_count,
                seed=int(bootstrap_seed),
            )
            rows.append(
                {
                    "estimator": estimator,
                    "n_nuisance": int(n_nuisance),
                    "noise_realizations": BENCHMARK_N_REALIZATIONS,
                    "bootstrap_count": int(bootstrap_count),
                    "bootstrap_seed": int(bootstrap_seed),
                    "bootstrap_lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
                    "bootstrap_upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
                    "bootstrap_center": "median",
                    **bootstrap_median,
                    "prefix_summary": prefix_summary,
                    **{
                        f"{name}_lower": bounds[0]
                        for name, bounds in intervals.items()
                    },
                    **{
                        f"{name}_upper": bounds[1]
                        for name, bounds in intervals.items()
                    },
                    **rmse(
                        truth_cells,
                        np.mean(galaxy_values[:, :n_nuisance], axis=1),
                    ),
                }
            )
    nre_scores = result["nre_log_r"]
    for n_nuisance in NUISANCE_PREFIXES:
        prefix_summary, bootstrap_median, intervals = _nre_nuisance_bootstrap_fits(
            truth_cells,
            nre_scores,
            result["nre_grid_phys"],
            n_nuisance=n_nuisance,
            n_bootstrap=bootstrap_count,
            seed=int(bootstrap_seed),
        )
        rows.append(
            {
                "estimator": "nre_stack_mean",
                "n_nuisance": int(n_nuisance),
                "noise_realizations": BENCHMARK_N_REALIZATIONS,
                "bootstrap_count": int(bootstrap_count),
                "bootstrap_seed": int(bootstrap_seed),
                "bootstrap_lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
                "bootstrap_upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
                "bootstrap_center": "median",
                **bootstrap_median,
                "prefix_summary": prefix_summary,
                **{
                    f"{name}_lower": bounds[0]
                    for name, bounds in intervals.items()
                },
                **{
                    f"{name}_upper": bounds[1]
                    for name, bounds in intervals.items()
                },
            }
        )
    return rows


def benchmark_takeaway(metrics: list[dict]) -> str:
    """State whether increasing R actually removes the ensemble bias."""

    metric_names = {row["estimator"] for row in metrics}
    estimators = tuple(
        (name, label)
        for name, label in ESTIMATORS
        if name in metric_names
    )
    by_name = {
        estimator: {
            int(row["prefix"]): row
            for row in metrics
            if row["estimator"] == estimator
        }
        for estimator, _ in estimators
    }
    changes = []
    for estimator, label in estimators:
        first = by_name[estimator][1]["combined_m"]
        last = by_name[estimator][16]["combined_m"]
        changes.append((abs(last), estimator, label, first, last))
    _, closest_name, closest_label, first, last = min(changes)
    improving = []
    for estimator, label in estimators:
        values = by_name[estimator]
        improving.append(
            (abs(values[16]["combined_m"]) - abs(values[1]["combined_m"]), label)
        )
    _, best_improver = min(improving)
    if "nre_stack_mean" in by_name:
        stack_r1 = by_name["nre_stack_mean"][1]
        stack_r16 = by_name["nre_stack_mean"][16]
        if abs(stack_r16["combined_m"]) < 0.05:
            return (
                "The all-nuisance NRE likelihood product is the first estimator "
                "close to zero in this benchmark: combined m="
                f"{stack_r16['combined_m']:.3f} at R=16 "
                f"(R=1: {stack_r1['combined_m']:.3f}), with component slopes "
                f"m_g1={stack_r16['g1_m']:.3f} and "
                f"m_g2={stack_r16['g2_m']:.3f}. This is substantially closer "
                "to an unbiased ensemble response than the per-galaxy NRE "
                "estimators and the 9D per-galaxy stack. The residual component "
                "difference should be treated as remaining calibration error, "
                "not as evidence that the likelihood is fully calibrated."
            )
    return (
        "No estimator approaches m=0 under this exact shared-shear benchmark. "
        f"At R=16 the closest is {closest_label} with combined m="
        f"{last:.3f} (R=1: {first:.3f}). {best_improver} shows the "
        "largest movement toward zero, but the remaining slope is still a "
        "substantial point-estimate bias. The 9D q/pi stack improves its "
        "combined slope modestly while the NRE grid MAP remains the closest "
        "single estimator; increasing R alone does not make the bias vanish."
    )


def _plot_outputs(
    result,
    rows,
    nuisance_rows,
    report_dir: Path,
) -> dict[str, str]:
    plt = _setup_matplotlib()
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    estimators = available_estimators(result)
    labels = dict(estimators)
    colors = (
        "#1f4e79",
        "#8f3b76",
        "#a05a2c",
        "#2e7d32",
        "#6a4c93",
        "#b56576",
        "#4d908e",
    )
    truth = result["truth_g"][:, 0]
    estimate_map = {
        "npe_mean": result["npe_mean"],
        "nre_mean": result["nre_mean"],
        "npe_map": result["npe_map"],
        "nre_map": result["nre_map"],
        "npe_9d_stack_map": result["surface_map"][..., [0, 1]],
        "npe_9d_stack_mean": result["surface_mean"][..., [0, 1]],
    }
    if "nre_stack_mean" in dict(estimators):
        estimate_map["nre_stack_mean"] = stacked_nre_means(
            result["nre_log_r"],
            result["nre_grid_phys"],
            tuple(int(value) for value in result["prefixes"]),
        )
    for prefix in (1, 16):
        fig, axes = plt.subplots(
            2, len(estimators), figsize=(18, 6.2), sharex=True, sharey=True
        )
        for column, (name, label) in enumerate(estimators):
            if name in ("npe_9d_stack_map", "npe_9d_stack_mean"):
                prefix_index = BENCHMARK_PREFIXES.index(prefix)
                estimate = estimate_map[name][:, :, prefix_index]
            elif name == "nre_stack_mean":
                prefix_index = BENCHMARK_PREFIXES.index(prefix)
                estimate = estimate_map[name][:, prefix_index]
            else:
                estimate = np.mean(estimate_map[name][:, :, :prefix], axis=2)
            cell_estimate = (
                estimate
                if name == "nre_stack_mean"
                else np.mean(estimate, axis=1)
            )
            for axis, component in zip(axes[:, column], (0, 1)):
                bias = cell_estimate[:, component] - truth[:, component]
                axis.scatter(
                    truth[:, component],
                    bias,
                    s=22,
                    alpha=0.85,
                    color=colors[column],
                )
                axis.axhline(0.0, color="#555", ls="--", lw=0.8)
                axis.set_title(label, fontsize=9)
                axis.set_xlabel("true g" + str(component + 1))
                if column == 0:
                    axis.set_ylabel("ĝ" + str(component + 1) + " − true g" + str(component + 1))
        fig.suptitle(f"Fixed-shear shear bias versus true shear, R={prefix}")
        fig.tight_layout()
        path = report_dir / f"scatter_r{prefix}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        figures[f"scatter_r{prefix}"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, (name, label) in zip(colors, estimators):
        subset = [row for row in rows if row["estimator"] == name]
        axis.plot(
            [row["prefix"] for row in subset],
            [row["combined_m"] for row in subset],
            marker="o",
            color=color,
            label=label,
        )
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(BENCHMARK_PREFIXES))
    axis.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: str(int(value))))
    axis.set_xlabel("noise realizations combined (R)")
    axis.set_ylabel("combined multiplicative bias m")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = report_dir / "m_vs_r.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["m_vs_r"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, (name, label) in zip(colors, estimators):
        subset = [row for row in nuisance_rows if row["estimator"] == name]
        x_values = np.asarray(
            [row["n_nuisance"] for row in subset],
            dtype=np.float64,
        )
        point_values = np.asarray(
            [row["combined_m"] for row in subset],
            dtype=np.float64,
        )
        lower_values = np.asarray(
            [row["combined_m_lower"] for row in subset],
            dtype=np.float64,
        )
        upper_values = np.asarray(
            [row["combined_m_upper"] for row in subset],
            dtype=np.float64,
        )
        axis.fill_between(
            x_values,
            lower_values,
            upper_values,
            color=color,
            alpha=0.16,
            linewidth=0,
        )
        axis.plot(
            x_values,
            point_values,
            marker="o",
            color=color,
            label=label,
        )
    axis.axhline(0.0, color="#555", ls="--", lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(NUISANCE_PREFIXES))
    axis.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda value, _: str(int(value)))
    )
    axis.set_xlabel("nuisance galaxies combined (N_nuisance)")
    axis.set_ylabel("combined multiplicative bias m (paired-bootstrap median)")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = report_dir / "m_vs_nuisance_r16.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["m_vs_nuisance_r16"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, (name, label) in zip(colors, estimators):
        subset = [row for row in rows if row["estimator"] == name]
        axis.plot(
            [row["prefix"] for row in subset],
            [row["combined_rmse"] for row in subset],
            marker="o",
            color=color,
            label=label,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(list(BENCHMARK_PREFIXES))
    axis.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda value, _: str(int(value))))
    axis.set_xlabel("noise realizations combined (R)")
    axis.set_ylabel("combined RMSE")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = report_dir / "rmse_vs_r.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["rmse_vs_r"] = path.name
    return figures


def _write_html(payload: dict, figures: dict[str, str], report_dir: Path) -> None:
    rows_html = []
    labels = dict(ESTIMATORS)
    metric_names = {row["estimator"] for row in payload["metrics"]}
    has_nre_stack = "nre_stack_mean" in metric_names
    estimator_count = len(metric_names)
    nre_stack_description = (
        "The 2D stacked NRE estimator sums the NRE log-ratio surfaces over "
        "all 32 nuisance galaxies and the first R noise realizations, then "
        "takes the posterior mean on that grid. "
        if has_nre_stack
        else ""
    )
    npe9d_mean_description = (
        "The 9D surface produces both a candidate-weighted posterior mean "
        "and a MAP for each latent galaxy. "
        if "npe_9d_stack_mean" in metric_names
        else ""
    )
    grouping_warning = (
        "only the 9D surface stack and the 2D NRE nuisance stack combine "
        "likelihood information. "
        if has_nre_stack
        else "only the 9D surface stack combines likelihood information. "
    )

    def format_se(value) -> str:
        if value is None or not np.isfinite(float(value)):
            return "not applicable"
        return f"{float(value):.5f}"

    for row in payload["metrics"]:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(labels[row['estimator']])}</td>"
            f"<td>{row['prefix']}</td>"
            f"<td>{row['g1_m']:.4f}</td><td>{row['g2_m']:.4f}</td>"
            f"<td>{row['g1_c']:.5f}</td><td>{row['g2_c']:.5f}</td>"
            f"<td>{row['combined_rmse']:.5f}</td>"
            "</tr>"
        )
    cell_rows = []
    for row in payload["per_shear"]:
        if row["prefix"] not in (1, 16):
            continue
        cell_rows.append(
            "<tr>"
            f"<td>{html.escape(labels[row['estimator']])}</td>"
            f"<td>{row['prefix']}</td><td>{row['cell']}</td>"
            f"<td>{row['g1_truth']:.2f}</td><td>{row['g2_truth']:.2f}</td>"
            f"<td>{row['g1_mean'] - row['g1_truth']:.5f} ± {format_se(row['g1_se'])}</td>"
            f"<td>{row['g2_mean'] - row['g2_truth']:.5f} ± {format_se(row['g2_se'])}</td>"
            "</tr>"
        )
    nuisance_rows_html = []
    for row in payload["nuisance_metrics"]:
        nuisance_rows_html.append(
            "<tr>"
            f"<td>{html.escape(labels[row['estimator']])}</td>"
            f"<td>{row['n_nuisance']}</td>"
            f"<td>{row['g1_m']:.4f}</td><td>{row['g2_m']:.4f}</td>"
            f"<td>{row['g1_c']:.5f}</td><td>{row['g2_c']:.5f}</td>"
            f"<td>{row['combined_m']:.4f}"
            f" [{row['combined_m_lower']:.4f}, {row['combined_m_upper']:.4f}]</td>"
            f"<td>{row['combined_rmse']:.5f}</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Fixed-shear estimator benchmark</title>
<style>
body {{ font: 17px/1.55 Palatino, "Palatino Linotype", serif; margin: 2rem auto; max-width: 1180px; color: #1b1b1b; }}
h1, h2 {{ font-weight: 600; }} figure {{ margin: 1.4rem 0 2rem; }}
figcaption, .note {{ color: #444; }} img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; margin: 1rem 0 1.6rem; }}
th, td {{ border: 1px solid #ccc; padding: .32rem .45rem; }}
th {{ background: #f4f4f4; text-align: left; }}
</style></head><body>
<h1>Fixed-shear estimator benchmark</h1>
<p class="lead">A controlled 5×5 grid of exact shears compares {estimator_count} ensemble
estimators as independent identity-noise realizations accumulate. The 32
latent galaxies are paired across every shear cell: nuisance parameters and
observation-quality controls are held fixed, while only the exact g1 and g2
values change.</p>

<h2>Design</h2>
<p>There are 25 exact shear cells, 32 paired latent galaxies per cell, and
16 deterministic independent image/spectral noise streams per galaxy. The
streams are derived from (seed, cell, galaxy, realization), so the result is
repeatable. Prefixes R = 1, 2, 4, 8, 16 use the first R realizations.
Each per-stamp NPE mean and sampled MAP uses {payload["npe_samples"]:,}
posterior candidates. The NRE mean and MAP use a {payload["grid_n"]}×{payload["grid_n"]}
shear grid. {nre_stack_description}{npe9d_mean_description}The 9D surface uses {payload["surface_samples"]}
candidates per noise, unions them into a common per-galaxy bank, removes the
uniform normalized prior, sums log(q/π), and selects a 9D MAP.</p>
<p class="note"><strong>Important grouping warning:</strong> {grouping_warning}
The per-galaxy estimators are averaged across repeated noises and then across
the 32 galaxies; the stacked NRE estimator instead sums its 2D score surfaces
over nuisance galaxies. Truth is a diagnostic marker and is never inserted
into a candidate bank.</p>

<h2>m/c and RMSE versus R</h2>
<table><thead><tr><th>Estimator</th><th>R</th><th>m g1</th><th>m g2</th>
<th>c g1</th><th>c g2</th><th>combined RMSE</th></tr></thead>
<tbody>{"".join(rows_html)}</tbody></table>
<figure><img src="{html.escape(figures['m_vs_r'])}" alt="multiplicative bias versus R">
<figcaption>Direct test of whether the ensemble slope moves toward m=0 as
the number of independent noises increases.</figcaption></figure>
<h2>Nuisance/galaxy convergence at R=16</h2>
<table><thead><tr><th>Estimator</th><th>N_nuisance</th><th>m g1</th><th>m g2</th>
<th>c g1</th><th>c g2</th><th>combined m [16th, 84th percentile]</th>
<th>combined RMSE</th></tr></thead>
<tbody>{"".join(nuisance_rows_html)}</tbody></table>
<figure><img src="{html.escape(figures['m_vs_nuisance_r16'])}"
alt="multiplicative bias versus nuisance galaxy count at R=16">
<figcaption>Noise averaging is held fixed at R=16: each per-galaxy estimator
first averages its 16 noise realizations per nuisance galaxy, while the two
9D summaries use each nuisance galaxy's R=16 stacked surface. The 2D NRE
likelihood product is recomputed over each nuisance prefix. For each
N_nuisance, the curve is the median of 1,000 paired bootstrap
replicates. The shaded interval is the 16th–84th percentile of those same
replicates; each replicate resamples N IDs with replacement from the 32
available nuisance IDs and reuses the same ID vector across all 25 shear
cells before refitting component-wise and combined m/c. The curve and band
therefore describe the same random-subset estimand and represent paired
finite-nuisance-sample variation in this benchmark.</figcaption>
</figure>
<figure><img src="{html.escape(figures['rmse_vs_r'])}" alt="RMSE versus R"></figure>

<h2>Fixed-shear recovery</h2>
<figure><img src="{html.escape(figures['scatter_r1'])}" alt="shear bias versus true shear at R=1">
<figcaption>Component shear bias, ĝ − g<sub>true</sub>, versus true shear
at R=1. Error bars are omitted here; per-cell standard errors across
latent galaxies are tabulated below.</figcaption></figure>
<figure><img src="{html.escape(figures['scatter_r16'])}" alt="shear bias versus true shear at R=16">
<figcaption>Component shear bias, ĝ − g<sub>true</sub>, versus true shear
at R=16. These are exact fixed-shear cells, not a broad-|g| product
diagnostic.</figcaption></figure>

<h2>Per-shear cells at R=1 and R=16</h2>
<table><thead><tr><th>Estimator</th><th>R</th><th>Cell</th><th>true g1</th>
<th>true g2</th><th>g1 bias ± SE</th><th>g2 bias ± SE</th></tr></thead>
<tbody>{"".join(cell_rows)}</tbody></table>
<p>{html.escape(payload["takeaway"])}</p>
</body></html>"""
    (report_dir / "report.html").write_text(body, encoding="utf-8")


def _load_draws(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def write_report(args) -> None:
    report_dir = args.report_dir
    draws_path = report_dir / "benchmark.npz"
    if args.reuse_draws:
        result = _load_draws(draws_path)
        meta = json.loads(str(result.pop("meta_json")))
    else:
        if report_dir.exists() and (report_dir / "report.html").exists() and not args.overwrite:
            raise FileExistsError(f"{report_dir / 'report.html'} exists; use --overwrite")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but CUDA is unavailable")
        device = torch.device(args.device)
        seed_everything(args.seed, deterministic=True)
        configs_root = args.model_root.parent / "configs"
        model_config = load_model_config(args.parent_npe, configs_root=str(configs_root))
        config.set_model_config(model_config)
        names = tuple(config.TARGET_NAMES)
        par_ranges = config.par_ranges
        config.require_matching_dataset_par_ranges(args.data_dir, par_ranges)
        checkpoint = _checkpoint_file(
            args.model_root, args.parent_npe, args.parent_checkpoint_suffix
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = load_model(
            KLNPE,
            path=str(checkpoint),
            model_name=args.parent_npe,
            device=str(device),
            strict=True,
            networks_root=str(args.model_root.parent / "networks"),
        )
        model.eval()
        channels_last = bool(model_config.train.channels_last)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        nre_name = args.nre_name
        if nre_name is None:
            from diagnostics.likelihood_stack_core import NRE_NAME

            nre_name = NRE_NAME
        nre_checkpoint = nre_checkpoint_path(
            args.model_root, nre_name, args.nre_checkpoint_suffix
        )
        nre_head, nre_meta = load_ratio_head(
            nre_checkpoint,
            device=device,
            expected_parent=args.parent_npe,
        )
        dataset = pxt.TorchDataset(str(args.data_dir))
        noisy_data = None
        if args.noise_data is not None:
            noisy_payload = np.load(args.noise_data, allow_pickle=False)
            noisy_data = {
                "images": np.asarray(noisy_payload["images"]),
                "spectra": np.asarray(noisy_payload["spectra"]),
            }
            expected_shape = (
                25 * BENCHMARK_N_GALAXIES,
                BENCHMARK_N_REALIZATIONS,
            )
            if noisy_data["images"].shape[:2] != expected_shape:
                raise ValueError(
                    f"noise image shape must start with {expected_shape}, "
                    f"got {noisy_data['images'].shape}"
                )
        result = run_benchmark(
            dataset,
            model,
            nre_head,
            names=names,
            par_ranges=par_ranges,
            channels_last=channels_last,
            device=device,
            npe_samples=args.npe_samples,
            grid_n=args.grid_n,
            seed=args.seed,
            noisy_data=noisy_data,
        )
        meta = {
            "schema": "likelihood-stack-fixed-shear-benchmark-v1",
            "data_dir": str(args.data_dir),
            "noise_data": str(args.noise_data) if args.noise_data else None,
            "parent_npe": args.parent_npe,
            "nre_name": nre_name,
            "seed": int(args.seed),
            "shear_values": list(BENCHMARK_SHEAR_VALUES),
            "n_cells": 25,
            "n_galaxies_per_cell": 32,
            "n_realizations": 16,
            "prefixes": list(BENCHMARK_PREFIXES),
            "npe_samples": int(args.npe_samples),
            "surface_samples": int(BENCHMARK_SURFACE_SAMPLES),
            "grid_n": int(args.grid_n),
            "uniform_normalized_prior_log_density": float(
                BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY
            ),
            "truth_is_candidate": False,
            "identity_only": True,
            "nre_meta": nre_meta,
        }
        report_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            draws_path,
            **result,
            meta_json=json.dumps(json_safe(meta)),
        )
    metrics, per_shear, summary = aggregate_results(result)
    nuisance_bootstrap_seed = int(meta.get("seed", BENCHMARK_NOISE_SEED)) + (
        NUISANCE_BOOTSTRAP_SEED_OFFSET
    )
    nuisance_metrics = aggregate_nuisance_results(
        result,
        bootstrap_count=NUISANCE_BOOTSTRAP_COUNT,
        bootstrap_seed=nuisance_bootstrap_seed,
    )
    figures = _plot_outputs(result, metrics, nuisance_metrics, report_dir)
    nre_stack_metadata = {}
    if "nre_stack_mean" in {row["estimator"] for row in metrics}:
        nre_stack_metadata = {
            "nre_nuisance_stack": {
                "estimator": "nre_stack_mean",
                "nuisance_galaxies": BENCHMARK_N_GALAXIES,
                "noise_prefixes": list(BENCHMARK_PREFIXES),
                "summary": "posterior mean of summed 2D NRE log-ratio surfaces",
                "prior": "uniform g01 shear-box prior",
            }
        }
    payload = {
        **meta,
        "metrics": metrics,
        "nuisance_prefixes": list(NUISANCE_PREFIXES),
        "nuisance_realizations": BENCHMARK_N_REALIZATIONS,
        "nuisance_bootstrap": {
            "count": NUISANCE_BOOTSTRAP_COUNT,
            "lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
            "upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
            "center": "median",
            "seed": nuisance_bootstrap_seed,
            "paired_across_shear_cells": True,
            "source_ids": "0 through n_galaxies_per_cell - 1",
            "fixed_noise_realizations": BENCHMARK_N_REALIZATIONS,
        },
        "nuisance_metrics": nuisance_metrics,
        "per_shear": per_shear,
        "figures": figures,
        "takeaway_data": summary,
        **nre_stack_metadata,
        "takeaway": benchmark_takeaway(metrics),
    }
    write_json(report_dir / "report.json", payload)
    _write_html(payload, figures, report_dir)
    print(f"Wrote {report_dir / 'report.json'}", flush=True)


def main(argv=None):
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
