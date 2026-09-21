"""Math and contracts for the exact fixed-shear likelihood-stack benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from diagnostics.likelihood_stack_core import json_safe


BENCHMARK_DATASET = "likelihood_stack_04_benchmark"
BENCHMARK_SAMPLE = "likelihood_stack_04_benchmark"
BENCHMARK_REPORT_DIR = Path(
    "/ocean/projects/phy250048p/shared/reports/likelihood-stack/04_benchmark"
)
BENCHMARK_PREFIXES = (1, 2, 4, 8, 16)
BENCHMARK_SHEAR_VALUES = (-0.08, -0.04, 0.0, 0.04, 0.08)
BENCHMARK_N_CELLS = 25
BENCHMARK_N_GALAXIES = 32
BENCHMARK_N_REALIZATIONS = 16
BENCHMARK_NPE_SAMPLES = 1024
BENCHMARK_SURFACE_SAMPLES = 256
BENCHMARK_GRID_N = 21
BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY = -9.0 * np.log(2.0)
BENCHMARK_NOISE_SEED = 420042


def fixed_shear_grid(
    values: tuple[float, ...] = BENCHMARK_SHEAR_VALUES,
) -> np.ndarray:
    """Return the 25 exact (g1, g2) cells in row-major order."""

    return np.asarray(
        [(float(g1), float(g2)) for g1 in values for g2 in values],
        dtype=np.float64,
    )


def row_index(cell: int, galaxy: int, *, n_galaxies: int = BENCHMARK_N_GALAXIES) -> int:
    cell = int(cell)
    galaxy = int(galaxy)
    if not 0 <= cell < BENCHMARK_N_CELLS:
        raise ValueError(f"cell must be in [0, {BENCHMARK_N_CELLS})")
    if not 0 <= galaxy < int(n_galaxies):
        raise ValueError(f"galaxy must be in [0, {n_galaxies})")
    return cell * int(n_galaxies) + galaxy


def fixed_shear_group_ids(
    n_cells: int = BENCHMARK_N_CELLS,
    n_galaxies: int = BENCHMARK_N_GALAXIES,
    n_realizations: int = BENCHMARK_N_REALIZATIONS,
) -> np.ndarray:
    """Return (cell, galaxy, realization) IDs for the materialized design."""

    if min(int(n_cells), int(n_galaxies), int(n_realizations)) <= 0:
        raise ValueError("all group sizes must be positive")
    cell = np.repeat(
        np.arange(n_cells, dtype=np.int64),
        int(n_galaxies) * int(n_realizations),
    )
    galaxy = np.tile(
        np.repeat(np.arange(n_galaxies, dtype=np.int64), int(n_realizations)),
        int(n_cells),
    )
    realization = np.tile(np.arange(n_realizations, dtype=np.int64), int(n_cells) * int(n_galaxies))
    return np.stack((cell, galaxy, realization), axis=1)


def benchmark_noise_seeds(
    seed: int,
    cell: int,
    galaxy: int,
    realization: int,
) -> tuple[int, int]:
    """Derive independent image/spectral streams from all benchmark IDs."""

    if min(int(cell), int(galaxy), int(realization)) < 0:
        raise ValueError("benchmark IDs must be non-negative")
    sequence = np.random.SeedSequence(
        [int(seed), int(cell), int(galaxy), int(realization)]
    )
    values = sequence.generate_state(2, dtype=np.uint64)
    max_seed = 2**63 - 1
    return int(values[0] % max_seed), int(values[1] % max_seed)


def validate_exact_truth_groups(
    truth: np.ndarray,
    *,
    n_cells: int = BENCHMARK_N_CELLS,
    n_galaxies: int = BENCHMARK_N_GALAXIES,
    n_realizations: int = BENCHMARK_N_REALIZATIONS,
) -> None:
    """Assert that only the realization axis varies within each latent group."""

    values = np.asarray(truth)
    expected = (int(n_cells), int(n_galaxies), int(n_realizations), values.shape[-1])
    if values.shape != expected:
        raise ValueError(f"truth must have shape {expected}, got {values.shape}")
    reference = values[:, :, :1, :]
    if not np.array_equal(values, np.broadcast_to(reference, values.shape)):
        raise ValueError("complete latent truth is not fixed across realizations")
    if np.array_equal(values[:, 0, 0, :], values[:, 1, 0, :]):
        raise ValueError("paired latent galaxy rows must be distinct")


def prefix_mean(values: np.ndarray, prefixes=BENCHMARK_PREFIXES) -> np.ndarray:
    """Average realization-level values over prefixes on axis -2."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-2] < max(prefixes):
        raise ValueError("values do not contain all requested realizations")
    return np.stack(
        [np.mean(values[..., : int(prefix), :], axis=-2) for prefix in prefixes],
        axis=-2,
    )


def prefix_surface_map(
    candidate_bank: np.ndarray,
    log_q: np.ndarray,
    *,
    prefixes=BENCHMARK_PREFIXES,
    candidates_per_realization: int = BENCHMARK_SURFACE_SAMPLES,
    prior_log_density: float = BENCHMARK_UNIFORM_PRIOR_LOG_DENSITY,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack q/pi surfaces on a cumulative unioned candidate bank.

    ``candidate_bank`` has shape (R, K, D), and ``log_q`` has shape
    (R, R*K). For prefix R, the first R*K candidates form the common bank,
    while every one of the first R observations is scored on that bank.
    """

    bank = np.asarray(candidate_bank, dtype=np.float64)
    scores = np.asarray(log_q, dtype=np.float64)
    if bank.ndim != 3:
        raise ValueError("candidate_bank must have shape (R, K, D)")
    n_real, per_realization, n_features = bank.shape
    if per_realization != int(candidates_per_realization):
        raise ValueError("candidate bank does not have the requested K")
    if scores.shape != (n_real, n_real * per_realization):
        raise ValueError("log_q must have shape (R, R*K)")
    # The prior is retained explicitly in the computation even though it is
    # constant for this benchmark's normalized uniform prior.
    log_q_over_pi = scores - float(prior_log_density)
    estimates = []
    stacked_scores = []
    for prefix in prefixes:
        prefix = int(prefix)
        if not 1 <= prefix <= n_real:
            raise ValueError(f"prefix {prefix} is outside 1..{n_real}")
        width = prefix * per_realization
        surface = np.sum(log_q_over_pi[:prefix, :width], axis=0)
        index = int(np.nanargmax(np.where(np.isfinite(surface), surface, -np.inf)))
        estimates.append(bank[:prefix].reshape(width, n_features)[index])
        stacked_scores.append(surface[index])
    return np.asarray(estimates), np.asarray(stacked_scores)


def fit_component_mc(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> dict[str, float]:
    """Fit y=(1+m)x+c independently for a two-component shear estimate."""

    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth and estimate must have shape (N, 2)")
    result: dict[str, float] = {"n": int(len(truth))}
    for index, name in enumerate(("g1", "g2")):
        finite = np.isfinite(truth[:, index]) & np.isfinite(estimate[:, index])
        if np.count_nonzero(finite) < 2:
            result[f"{name}_m"] = float("nan")
            result[f"{name}_c"] = float("nan")
            continue
        design = np.column_stack((truth[finite, index], np.ones(np.count_nonzero(finite))))
        slope, intercept = np.linalg.lstsq(
            design, estimate[finite, index], rcond=None
        )[0]
        result[f"{name}_m"] = float(slope - 1.0)
        result[f"{name}_c"] = float(intercept)
    result["combined_m"] = float(
        np.nanmean([result["g1_m"], result["g2_m"]])
    )
    result["combined_c"] = float(
        np.nanmean([result["g1_c"], result["g2_c"]])
    )
    return result


def rmse(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth and estimate must have shape (N, 2)")
    residual = estimate - truth
    return {
        "g1_rmse": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
        "g2_rmse": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
        "combined_rmse": float(np.sqrt(np.mean(residual**2))),
    }


def json_safe_benchmark(value):
    return json_safe(value)
