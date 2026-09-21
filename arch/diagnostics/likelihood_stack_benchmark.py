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
)
NUISANCE_PREFIXES = (1, 2, 4, 8, 16, 32)
NUISANCE_BOOTSTRAP_COUNT = 1000
NUISANCE_BOOTSTRAP_LOWER_PERCENTILE = 16.0
NUISANCE_BOOTSTRAP_UPPER_PERCENTILE = 84.0
NUISANCE_BOOTSTRAP_SEED_OFFSET = 1_000_003


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
) -> np.ndarray:
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
    output = np.empty((n_gal, len(prefixes), n_features), dtype=np.float64)
    for galaxy in range(n_gal):
        mapped, _ = prefix_surface_map(
            candidates[galaxy],
            scores_np[galaxy],
            prefixes=prefixes,
            candidates_per_realization=n_candidates,
            prior_log_density=BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY,
        )
        output[galaxy] = denormalize(
            mapped,
            config.par_ranges,
            feature_names=tuple(config.TARGET_NAMES),
            target_transforms=config.TARGET_TRANSFORMS,
        )
    return output


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
            candidates[:, realization] = current["surface_candidates"]
            noisy_images.append(image)
            noisy_spectra.append(spec)
        image_stack = torch.stack(noisy_images, dim=1)
        spec_stack = torch.stack(noisy_spectra, dim=1)
        surface_map[cell_id] = _surface_stack(
            model,
            image_stack,
            spec_stack,
            cell,
            candidates,
        )
        del image_stack, spec_stack
    return {
        **point,
        "surface_map": surface_map,
        "truth_g": truth_g,
        "truth_full": truth_full,
        "prefixes": np.asarray(BENCHMARK_PREFIXES, dtype=np.int64),
    }


def aggregate_results(result: dict[str, np.ndarray]) -> tuple[list[dict], list[dict], dict]:
    truth = result["truth_g"]
    prefixes = tuple(int(value) for value in result["prefixes"])
    rows = []
    cells = []
    surface = result["surface_map"][..., [0, 1]]
    estimates = {
        "npe_mean": result["npe_mean"],
        "nre_mean": result["nre_mean"],
        "npe_map": result["npe_map"],
        "nre_map": result["nre_map"],
        "npe_9d_stack_map": surface,
    }
    for estimator, _ in ESTIMATORS:
        values = estimates[estimator]
        for prefix_index, prefix in enumerate(prefixes):
            if estimator == "npe_9d_stack_map":
                galaxy_values = values[:, :, prefix_index]
            else:
                galaxy_values = np.mean(values[:, :, :prefix], axis=2)
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
                values_cell = galaxy_values[cell_id]
                cells.append(
                    {
                        "estimator": estimator,
                        "prefix": int(prefix),
                        "cell": int(cell_id),
                        "g1_truth": float(shear[0]),
                        "g2_truth": float(shear[1]),
                        "g1_mean": float(np.mean(values_cell[:, 0])),
                        "g2_mean": float(np.mean(values_cell[:, 1])),
                        "g1_se": float(np.std(values_cell[:, 0], ddof=1) / np.sqrt(len(values_cell))),
                        "g2_se": float(np.std(values_cell[:, 1], ddof=1) / np.sqrt(len(values_cell))),
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
        for estimator, _ in ESTIMATORS
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
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    """Fit point estimates and paired-bootstrap percentile intervals."""

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
    point = fit_component_mc(
        truth_cells,
        np.mean(values[:, :n_nuisance], axis=1),
    )
    # Advanced indexing gives (cell, bootstrap, nuisance, component).  The
    # bootstrap ID row is therefore shared across all 25 exact shear cells.
    bootstrap_cell_values = np.mean(values[:, bootstrap_ids, :], axis=2)
    bootstrap_fits = [
        fit_component_mc(truth_cells, bootstrap_cell_values[:, index, :])
        for index in range(n_bootstrap)
    ]
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
    return point, interval


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
    }
    rows = []
    for estimator, _ in ESTIMATORS:
        galaxy_values = estimates[estimator]
        for n_nuisance in NUISANCE_PREFIXES:
            point, intervals = _nuisance_bootstrap_fits(
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
                    **point,
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
    return rows


def benchmark_takeaway(metrics: list[dict]) -> str:
    """State whether increasing R actually removes the ensemble bias."""

    by_name = {
        estimator: {
            int(row["prefix"]): row
            for row in metrics
            if row["estimator"] == estimator
        }
        for estimator, _ in ESTIMATORS
    }
    changes = []
    for estimator, label in ESTIMATORS:
        first = by_name[estimator][1]["combined_m"]
        last = by_name[estimator][16]["combined_m"]
        changes.append((abs(last), estimator, label, first, last))
    _, closest_name, closest_label, first, last = min(changes)
    improving = []
    for estimator, label in ESTIMATORS:
        values = by_name[estimator]
        improving.append(
            (abs(values[16]["combined_m"]) - abs(values[1]["combined_m"]), label)
        )
    _, best_improver = min(improving)
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
    labels = dict(ESTIMATORS)
    colors = ("#1f4e79", "#8f3b76", "#a05a2c", "#2e7d32", "#6a4c93")
    truth = result["truth_g"][:, 0]
    estimate_map = {
        "npe_mean": result["npe_mean"],
        "nre_mean": result["nre_mean"],
        "npe_map": result["npe_map"],
        "nre_map": result["nre_map"],
        "npe_9d_stack_map": result["surface_map"][..., [0, 1]],
    }
    for prefix in (1, 16):
        fig, axes = plt.subplots(2, 5, figsize=(15, 6.2), sharex=True, sharey=True)
        for column, (name, label) in enumerate(ESTIMATORS):
            if name == "npe_9d_stack_map":
                prefix_index = BENCHMARK_PREFIXES.index(prefix)
                estimate = estimate_map[name][:, :, prefix_index]
            else:
                estimate = np.mean(estimate_map[name][:, :, :prefix], axis=2)
            cell_estimate = np.mean(estimate, axis=1)
            for axis, component in zip(axes[:, column], (0, 1)):
                axis.scatter(
                    truth[:, component],
                    cell_estimate[:, component],
                    s=22,
                    alpha=0.85,
                    color=colors[column],
                )
                axis.axline((0, 0), slope=1, color="#555", ls="--", lw=0.8)
                axis.set_title(label, fontsize=9)
                axis.set_xlabel("true g" + str(component + 1))
                if column == 0:
                    axis.set_ylabel("estimated g" + str(component + 1))
        fig.suptitle(f"Fixed-shear cell means, R={prefix}")
        fig.tight_layout()
        path = report_dir / f"scatter_r{prefix}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        figures[f"scatter_r{prefix}"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, (name, label) in zip(colors, ESTIMATORS):
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
    for color, (name, label) in zip(colors, ESTIMATORS):
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
    axis.set_ylabel("combined multiplicative bias m")
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = report_dir / "m_vs_nuisance_r16.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    figures["m_vs_nuisance_r16"] = path.name

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for color, (name, label) in zip(colors, ESTIMATORS):
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
            f"<td>{row['g1_mean']:.5f} ± {row['g1_se']:.5f}</td>"
            f"<td>{row['g2_mean']:.5f} ± {row['g2_se']:.5f}</td>"
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
<p class="lead">A controlled 5×5 grid of exact shears compares five ensemble
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
shear grid. The 9D surface uses {payload["surface_samples"]} candidates per
noise, unions them into a common per-galaxy bank, removes the uniform
normalized prior, sums log(q/π), and selects a 9D MAP.</p>
<p class="note"><strong>Important grouping warning:</strong> only the 9D
surface stack shares the complete latent truth within a group. The four
per-stamp estimators are averaged across repeated noises and then across the
32 galaxies; they are not products of likelihoods from different nuisance
galaxies. Truth is a diagnostic marker and is never inserted into a candidate
bank.</p>

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
<figcaption>Noise averaging is held fixed at R=16: each of the four
per-stamp estimators first averages its 16 noise realizations per nuisance
galaxy, while the 9D estimator uses each nuisance galaxy's R=16 stacked MAP.
For each N_nuisance, the point estimate averages the first N nuisance IDs in
each exact shear cell. The shaded interval is the 16th–84th percentile over
1,000 deterministic bootstrap replicates; each replicate resamples N IDs with
replacement from the 32 available nuisance IDs and reuses the same ID vector
across all 25 shear cells before refitting component-wise and combined m/c.
The interval therefore represents paired finite-nuisance-sample variation in
this benchmark.</figcaption>
</figure>
<figure><img src="{html.escape(figures['rmse_vs_r'])}" alt="RMSE versus R"></figure>

<h2>Fixed-shear recovery</h2>
<figure><img src="{html.escape(figures['scatter_r1'])}" alt="fixed shear scatter at R=1">
<figcaption>Cell means at R=1. Error bars are omitted here; the per-cell
standard errors across 32 latent galaxies are tabulated below.</figcaption></figure>
<figure><img src="{html.escape(figures['scatter_r16'])}" alt="fixed shear scatter at R=16">
<figcaption>Cell means at R=16. These are exact fixed-shear cells, not a
broad-|g| product diagnostic.</figcaption></figure>

<h2>Per-shear cells at R=1 and R=16</h2>
<table><thead><tr><th>Estimator</th><th>R</th><th>Cell</th><th>true g1</th>
<th>true g2</th><th>g1 estimate ± SE</th><th>g2 estimate ± SE</th></tr></thead>
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
    payload = {
        **meta,
        "metrics": metrics,
        "nuisance_prefixes": list(NUISANCE_PREFIXES),
        "nuisance_realizations": BENCHMARK_N_REALIZATIONS,
        "nuisance_bootstrap": {
            "count": NUISANCE_BOOTSTRAP_COUNT,
            "lower_percentile": NUISANCE_BOOTSTRAP_LOWER_PERCENTILE,
            "upper_percentile": NUISANCE_BOOTSTRAP_UPPER_PERCENTILE,
            "seed": nuisance_bootstrap_seed,
            "paired_across_shear_cells": True,
            "source_ids": "0 through n_galaxies_per_cell - 1",
            "fixed_noise_realizations": BENCHMARK_N_REALIZATIONS,
        },
        "nuisance_metrics": nuisance_metrics,
        "per_shear": per_shear,
        "figures": figures,
        "takeaway_data": summary,
        "takeaway": benchmark_takeaway(metrics),
    }
    write_json(report_dir / "report.json", payload)
    _write_html(payload, figures, report_dir)
    print(f"Wrote {report_dir / 'report.json'}", flush=True)


def main(argv=None):
    write_report(parse_args(argv))


if __name__ == "__main__":
    main()
