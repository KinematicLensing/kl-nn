"""Shear-only NRE head: joint-vs-shuffled-g classifier, log r, and grid stacks."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from diagnostics.likelihood_stack_core import (
    ABS_G_SPLIT,
    NRE_ABS_G_EDGES,
    NRE_CONTEXT_DIM,
    NRE_GRID_N,
    NRE_HIDDEN_DIMS,
    NRE_PREFIXES,
    abs_g,
    abs_g_split_masks,
    interval_coverage,
    map_index,
    residual_calibration,
)


QUANTILES = (0.16, 0.84)
AMP_SLICE_NAMES = (
    "amp_00_025",
    "amp_025_050",
    "amp_050_075",
    "amp_075_100",
)
SLICE_LABELS = {
    "full": "full catalog",
    "inner": "|g| < 0.05",
    "outer": "|g| > 0.05",
    "amp_00_025": "|g| in [0, 0.025)",
    "amp_025_050": "|g| in [0.025, 0.05)",
    "amp_050_075": "|g| in [0.05, 0.075)",
    "amp_075_100": "|g| in [0.075, 0.1]",
}


class RatioHead(nn.Module):
    """MLP on frozen context concatenated with normalized (g1, g2). One logit."""

    def __init__(
        self,
        context_dim: int = NRE_CONTEXT_DIM,
        hidden_dims: tuple[int, ...] = NRE_HIDDEN_DIMS,
    ):
        super().__init__()
        context_dim = int(context_dim)
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        layers: list[nn.Module] = []
        width_in = context_dim + 2
        for width_out in hidden_dims:
            layers.extend((nn.Linear(width_in, width_out), nn.ReLU()))
            width_in = width_out
        layers.append(nn.Linear(width_in, 1))
        self.network = nn.Sequential(*layers)
        self.context_dim = context_dim
        self.hidden_dims = hidden_dims

    def forward(self, context: torch.Tensor, shear: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2 or context.shape[-1] != self.context_dim:
            raise ValueError(
                "context must have shape (B, "
                f"{self.context_dim}); got {tuple(context.shape)}"
            )
        if shear.ndim != 2 or shear.shape[-1] != 2:
            raise ValueError(f"shear must have shape (B, 2); got {tuple(shear.shape)}")
        if context.shape[0] != shear.shape[0]:
            raise ValueError("context and shear batch sizes must match")
        logit = self.network(torch.cat((context, shear), dim=-1))
        return logit.squeeze(-1)


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def shuffle_shear(shear: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Permute g within the batch. Same x, different g is the marginal class."""

    if shear.ndim != 2 or shear.shape[-1] != 2:
        raise ValueError(f"shear must have shape (B, 2); got {tuple(shear.shape)}")
    batch = int(shear.shape[0])
    if batch < 2:
        raise ValueError("shuffling g requires batch size >= 2")
    perm = torch.randperm(batch, device=shear.device, generator=generator)
    identity = torch.arange(batch, device=shear.device)
    if torch.equal(perm, identity):
        perm = torch.roll(identity, shifts=1)
    return shear.index_select(0, perm)


def nre_pair_logits_and_labels(
    head: nn.Module,
    context: torch.Tensor,
    shear: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint (label 1) then shuffled-g (label 0). Logits are log r at the BCE optimum."""

    shuffled = shuffle_shear(shear, generator=generator)
    joint = head(context, shear)
    marginal = head(context, shuffled)
    logits = torch.cat((joint, marginal), dim=0)
    labels = torch.cat((torch.ones_like(joint), torch.zeros_like(marginal)), dim=0)
    return logits, labels


def nre_bce_loss(
    head: nn.Module,
    context: torch.Tensor,
    shear: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    logits, labels = nre_pair_logits_and_labels(
        head, context, shear, generator=generator
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def normalized_shear_grid(n: int = NRE_GRID_N) -> tuple[np.ndarray, np.ndarray]:
    n = int(n)
    if n < 2:
        raise ValueError("grid size must be at least 2")
    axis = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    g1, g2 = np.meshgrid(axis, axis, indexing="ij")
    grid = np.stack((g1.reshape(-1), g2.reshape(-1)), axis=1)
    return grid, axis


def score_shear_grid(
    head: nn.Module,
    context: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Return (B, G) logits. Each logit is log r(x, g) on the grid."""

    if context.ndim != 2:
        raise ValueError("context must have shape (B, C)")
    if grid.ndim != 2 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape (G, 2)")
    batch, _ = context.shape
    n_grid = int(grid.shape[0])
    expanded_context = context.unsqueeze(1).expand(batch, n_grid, context.shape[-1])
    expanded_grid = grid.unsqueeze(0).expand(batch, n_grid, 2)
    logits = head(
        expanded_context.reshape(batch * n_grid, context.shape[-1]),
        expanded_grid.reshape(batch * n_grid, 2),
    )
    return logits.reshape(batch, n_grid)


def stack_log_r(log_r: np.ndarray) -> np.ndarray:
    scores = np.asarray(log_r, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("log r must have shape (n_galaxies, n_grid)")
    return np.sum(scores, axis=0)


def grid_map(log_r: np.ndarray, grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape (n_grid, 2)")
    scores = np.asarray(log_r, dtype=np.float64)
    if scores.ndim != 1:
        scores = stack_log_r(scores)
    if scores.shape[0] != grid.shape[0]:
        raise ValueError("log r length must match the grid")
    return grid[map_index(scores)].copy()


def discrete_posterior(log_r: np.ndarray) -> np.ndarray:
    scores = np.asarray(log_r, dtype=np.float64)
    if scores.ndim == 1:
        shifted = scores - np.max(scores)
        probability = np.exp(shifted)
        total = probability.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("log r does not yield a finite posterior")
        return probability / total
    if scores.ndim != 2:
        raise ValueError("log r must have shape (n_grid,) or (n, n_grid)")
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probability = np.exp(shifted)
    total = probability.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(total) | (total <= 0.0)):
        raise ValueError("log r does not yield a finite posterior")
    return probability / total


def grid_posterior_moments(
    log_r: np.ndarray,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Posterior mean and standard deviation on a finite shear grid."""

    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[-1] != 2:
        raise ValueError("grid must have shape (n_grid, 2)")
    probability = discrete_posterior(log_r)
    if probability.ndim == 1:
        probability = probability[None, :]
    if probability.shape[-1] != grid.shape[0]:
        raise ValueError("log r width must match the grid")
    mean = probability @ grid
    centered = grid[None, :, :] - mean[:, None, :]
    variance = np.sum(probability[:, :, None] * centered**2, axis=1)
    return {
        "mean": mean,
        "std": np.sqrt(np.maximum(variance, 0.0)),
    }


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    quantiles: tuple[float, ...] = QUANTILES,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1 or values.size == 0:
        raise ValueError("values and weights must be non-empty 1D arrays of one length")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = np.clip(weights[order], 0.0, None)
    total = float(sorted_weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full(len(quantiles), np.nan, dtype=np.float64)
    cdf = np.cumsum(sorted_weights) / total
    return np.interp(np.asarray(quantiles, dtype=np.float64), cdf, sorted_values)


def grid_marginal_intervals(
    log_r: np.ndarray,
    axis: np.ndarray,
    *,
    n_grid: int = NRE_GRID_N,
    quantiles: tuple[float, float] = QUANTILES,
) -> dict[str, np.ndarray]:
    """16–84% intervals of the 1D g1 and g2 marginals on an (n, n) g grid."""

    axis = np.asarray(axis, dtype=np.float64)
    n_grid = int(n_grid)
    if axis.shape != (n_grid,):
        raise ValueError(f"axis must have shape ({n_grid},)")
    probability = discrete_posterior(log_r)
    if probability.ndim == 1:
        probability = probability[None, :]
    n_rows, n_cells = probability.shape
    if n_cells != n_grid * n_grid:
        raise ValueError("log r width must be n_grid ** 2")
    mass = probability.reshape(n_rows, n_grid, n_grid)
    p_g1 = mass.sum(axis=2)
    p_g2 = mass.sum(axis=1)
    lower_g1 = np.empty(n_rows, dtype=np.float64)
    upper_g1 = np.empty(n_rows, dtype=np.float64)
    lower_g2 = np.empty(n_rows, dtype=np.float64)
    upper_g2 = np.empty(n_rows, dtype=np.float64)
    for row in range(n_rows):
        lo1, hi1 = weighted_quantiles(axis, p_g1[row], quantiles)
        lo2, hi2 = weighted_quantiles(axis, p_g2[row], quantiles)
        lower_g1[row] = lo1
        upper_g1[row] = hi1
        lower_g2[row] = lo2
        upper_g2[row] = hi2
    return {
        "g1_lower": lower_g1,
        "g1_upper": upper_g1,
        "g2_lower": lower_g2,
        "g2_upper": upper_g2,
    }


def nre_slice_masks(
    truth_g: np.ndarray,
    *,
    split: float = ABS_G_SPLIT,
    amp_edges: tuple[float, ...] = NRE_ABS_G_EDGES,
) -> dict[str, np.ndarray]:
    masks = dict(abs_g_split_masks(truth_g, split=split))
    amplitude = abs_g(truth_g[:, 0], truth_g[:, 1])
    finite = np.isfinite(amplitude)
    edges = tuple(float(edge) for edge in amp_edges)
    if len(edges) - 1 != len(AMP_SLICE_NAMES):
        raise ValueError("amplitude edges must have one more entry than slice names")
    for index, name in enumerate(AMP_SLICE_NAMES):
        low = edges[index]
        high = edges[index + 1]
        if index + 1 == len(AMP_SLICE_NAMES):
            masks[name] = finite & (amplitude >= low) & (amplitude <= high)
        else:
            masks[name] = finite & (amplitude >= low) & (amplitude < high)
    return masks


def prefix_groups(n_items: int, count: int, *, seed: int = 42) -> np.ndarray:
    """Disjoint random groups of `count` indices. Remainder is dropped."""

    n_items = int(n_items)
    count = int(count)
    if n_items < 0:
        raise ValueError("n_items must be non-negative")
    if count < 1:
        raise ValueError("group size must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    n_groups = n_items // count
    if n_groups == 0:
        return np.empty((0, count), dtype=np.int64)
    return order[: n_groups * count].reshape(n_groups, count).astype(np.int64, copy=False)


def grouped_mean(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError("values must have shape (n, 2)")
    if groups.ndim != 2:
        raise ValueError("groups must have shape (n_groups, n)")
    if groups.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.mean(values[groups], axis=1)


def grouped_stack_map(
    log_r: np.ndarray,
    grid: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    log_r = np.asarray(log_r, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if log_r.ndim != 2:
        raise ValueError("log r must have shape (n_galaxies, n_grid)")
    if grid.shape != (log_r.shape[1], 2):
        raise ValueError("grid must have shape (n_grid, 2)")
    if groups.ndim != 2:
        raise ValueError("groups must have shape (n_groups, n)")
    if groups.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    stacked = np.sum(log_r[groups], axis=1)
    finite = np.isfinite(stacked)
    stacked = np.where(finite, stacked, -np.inf)
    index = np.argmax(stacked, axis=1)
    return grid[index].copy()


def component_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth and estimate must have shape (N, 2)")
    row = {"n": int(len(truth))}
    for axis, name in enumerate(("g1", "g2")):
        metrics = residual_calibration(truth[:, axis], estimate[:, axis])
        row[f"{name}_m"] = metrics["m"]
        row[f"{name}_c"] = metrics["c"]
        row[f"{name}_median_abs_residual"] = metrics["median_abs_residual"]
    return row


def mean_m(row: dict) -> float:
    return 0.5 * (float(row["g1_m"]) + float(row["g2_m"]))


def nre_takeaway(galaxy_rows: list[dict], stack_rows: list[dict]) -> str:
    """One-line reading of whether stacked NRE unshrinks relative to the Mean."""

    def galaxy_m(estimator: str, slice_name: str = "full") -> float:
        match = [
            row
            for row in galaxy_rows
            if row.get("estimator") == estimator and row.get("slice") == slice_name
        ]
        if not match:
            return float("nan")
        return mean_m(match[0])

    def stack_m(estimator: str, slice_name: str = "outer", n: int = NRE_PREFIXES[-1]) -> float:
        match = [
            row
            for row in stack_rows
            if row.get("estimator") == estimator
            and row.get("slice") == slice_name
            and int(row.get("stack_size") or 0) == int(n)
        ]
        if not match:
            return float("nan")
        return mean_m(match[0])

    def stack_row(
        estimator: str,
        slice_name: str = "outer",
        n: int = NRE_PREFIXES[-1],
    ) -> dict | None:
        return next(
            (
                row
                for row in stack_rows
                if row.get("estimator") == estimator
                and row.get("slice") == slice_name
                and int(row.get("stack_size") or 0) == int(n)
            ),
            None,
        )

    nre_stack = stack_m("nre_stack_map")
    mean_stack = stack_m("npe_mean_of_means")
    nre_stack_row = stack_row("nre_stack_map")
    nre_gal = galaxy_m("nre_map")
    nre_mean_gal = galaxy_m("nre_mean")
    mean_gal = galaxy_m("npe_mean")
    recovered = 0.15
    shrunk = 0.25
    if np.isfinite(nre_mean_gal) and np.isfinite(mean_gal):
        return (
            f"Per galaxy, the NRE MAP has mean m {nre_gal:.2f} and the NRE "
            f"posterior mean has mean m {nre_mean_gal:.2f}, versus {mean_gal:.2f} "
            "for the cached NPE Mean. The posterior mean is the lower-variance "
            "NRE estimator; its intervals, rather than the MAP, should be used "
            "for local coverage. The broad-|g| grouped product remains only a "
            "stress test because its galaxies do not share one shear vector."
        )
    if np.isfinite(nre_stack) and np.isfinite(mean_stack):
        component_m = (
            float(nre_stack_row["g1_m"]),
            float(nre_stack_row["g2_m"]),
        )
        if max(component_m) > recovered and abs(mean_stack) > shrunk:
            return (
                "The grouped NRE product crosses m = 0 and overshoots: at "
                f"N = {NRE_PREFIXES[-1]} in the outer slice, m is "
                f"{component_m[0]:.2f} for g1 and {component_m[1]:.2f} for g2, "
                "while the mean of Means stays pulled toward zero. This is not "
                "evidence of convergence. The galaxies share only a broad |g| "
                "slice, not one shear vector, so multiplying their ratios is a "
                "stress test rather than a valid common-shear likelihood."
            )
        if abs(nre_stack) < recovered and abs(mean_stack) > shrunk:
            return (
                "The stacked NRE peak sits near the true shear while the mean of "
                "Means stays pulled toward zero. The frozen encoder still carries "
                "a useable shear likelihood; the catalog slope is a property of "
                "the Mean as a point estimator."
            )
        if abs(nre_stack) > shrunk and abs(mean_stack) > shrunk:
            return (
                "Both the stacked NRE peak and the mean of Means stay pulled "
                "toward zero. A new shear head on this frozen encoder does not "
                "invent a likelihood that the Mean was hiding."
            )
        return (
            "The stacked NRE peak and the mean of Means do not separate cleanly. "
            f"At N = {NRE_PREFIXES[-1]} in the outer half, NRE m is {nre_stack:.2f} "
            f"and the mean of Means m is {mean_stack:.2f}. Look at the versus-N "
            "table before calling the encoder weak or useable in shear."
        )
    if np.isfinite(nre_gal) and np.isfinite(mean_gal):
        if abs(nre_gal) < recovered and abs(mean_gal) > shrunk:
            return (
                "Per galaxy, the NRE peak sits nearer the true shear than the "
                "Mean. The encoder still carries shear information that the Mean "
                "shrinks."
            )
        if abs(nre_gal) > shrunk and abs(mean_gal) > shrunk:
            return (
                "Per galaxy, both the NRE peak and the Mean stay pulled toward "
                "zero. The frozen encoder context is weak in shear."
            )
    return "The NRE comparison did not produce a finite slope."


def zero_logit_bce() -> float:
    return math.log(2.0)
