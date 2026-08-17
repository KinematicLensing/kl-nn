#!/usr/bin/env python3
"""Build a self-contained shear-bias report from cached posterior products.

Most diagnostics use the compact truth, SNR, MAP, and posterior-summary
arrays. The shear P-P diagnostic memory-maps posterior sample partitions and
reduces only g1/g2 in small galaxy blocks; it never concatenates the raw sample
bank in memory.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import the module directly because ``arch.__init__`` still references removed
# legacy modules. This works for direct execution from the repository root,
# from inside ``arch``, and for importlib-based tests.
arch_dir = Path(__file__).resolve().parents[1]
if str(arch_dir) not in sys.path:
    sys.path.insert(0, str(arch_dir))
from utils import img_to_gal_axis as _img_to_gal_axis


PART_RE = re.compile(r"part(\d+)of(\d+)\.npy$")
SINI_CUTS = (0.3, 0.5, 0.7, np.inf)
ADDITIVE_DISPLAY_SCALE = 1e4
MULTIPLICATIVE_DISPLAY_SCALE = 1e2
SHEAR_PP_BIN_EDGES = np.linspace(-0.1, 0.1, 4)
FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
NUISANCE_INDICES = {"sini": 3, "vcirc": 5, "hlr": 7}
NUISANCE_LABELS = {
    "sini": "sini estimate - truth",
    "vcirc": "vcirc estimate - truth [km/s]",
    "hlr": "hlr estimate - truth [arcsec]",
}
GALAXY_COMPONENT_COLORS = {"g+": "tab:red", "gx": "tab:blue"}
ESTIMATOR_COLORS = {
    "MAP": "tab:blue",
    "Mean": "tab:orange",
    "In-support Mean": "tab:green",
}


class PosteriorPPUnavailable(RuntimeError):
    """Raised when cached draws cannot represent the reported posterior."""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="MODEL:DATASET",
        help="Cached model/dataset pair; repeat for comparisons.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/ocean/projects/phy250048p/shared/cache"),
    )
    parser.add_argument("--mode", type=int, default=0, help="TF-analysis mode index.")
    parser.add_argument("--low-g", type=float, default=0.02)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument(
        "--max-galaxies",
        type=int,
        default=None,
        help="Use only the first N galaxies after combining cache partitions.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def partition_files(directory: Path) -> list[Path]:
    files = []
    for path in directory.glob("part*of*.npy"):
        match = PART_RE.match(path.name)
        if match:
            files.append((int(match.group(1)), int(match.group(2)), path))
    if not files:
        raise FileNotFoundError(f"No partition arrays in {directory}")
    totals = {item[1] for item in files}
    if len(totals) != 1:
        raise ValueError(f"Mixed partition totals in {directory}: {totals}")
    expected = next(iter(totals))
    files.sort(key=lambda item: item[0])
    indices = [item[0] for item in files]
    if indices != list(range(expected)):
        raise ValueError(f"Incomplete partitions in {directory}: {indices} of {expected}")
    return [item[2] for item in files]


def load_concat(directory: Path, axis: int) -> np.ndarray:
    arrays = [np.load(path, mmap_mode="r") for path in partition_files(directory)]
    return np.concatenate(arrays, axis=axis)


def load_case(
    cache_root: Path, case: str, mode: int, max_galaxies: int | None = None
) -> dict:
    try:
        model, dataset = case.split(":", 1)
    except ValueError as exc:
        raise ValueError(f"Case must be MODEL:DATASET, got {case!r}") from exc
    root = cache_root / model / dataset
    truth = load_concat(root / "truth", axis=0)
    snr = load_concat(root / "snr", axis=0)
    map_all = load_concat(root / "map_estimates", axis=1)
    mean_all = load_concat(root / "mean_estimates", axis=1)
    support_summary_dir = root / "in_support_mean_estimates"
    support_retention_dir = root / "in_support_retention"
    support_dirs_present = (
        support_summary_dir.is_dir(),
        support_retention_dir.is_dir(),
    )
    if any(support_dirs_present) and not all(support_dirs_present):
        raise FileNotFoundError(
            "Incomplete in-support cache: both in_support_mean_estimates and "
            "in_support_retention are required"
        )
    support_all = None
    retention_all = None
    if all(support_dirs_present):
        support_all = load_concat(support_summary_dir, axis=1)
        retention_all = load_concat(support_retention_dir, axis=1)
    if not (0 <= mode < map_all.shape[0]):
        raise IndexError(f"Mode {mode} outside [0, {map_all.shape[0]}) for {case}")
    if support_all is not None and not (0 <= mode < support_all.shape[0]):
        raise IndexError(f"Mode {mode} outside in-support summaries for {case}")
    if max_galaxies is not None:
        if max_galaxies <= 0:
            raise ValueError("max_galaxies must be positive")
        truth = truth[:max_galaxies]
        snr = snr[:max_galaxies]
        map_all = map_all[:, :max_galaxies]
        mean_all = mean_all[:, :max_galaxies]
        if support_all is not None:
            support_all = support_all[:, :max_galaxies]
            retention_all = retention_all[:, :max_galaxies]
    estimates = {
        "MAP": np.asarray(map_all[mode]),
        "Mean": np.asarray(mean_all[mode, :, 1]),
    }
    summaries = {"Mean": np.asarray(mean_all[mode])}
    support_retention = None
    if support_all is not None:
        estimates["In-support Mean"] = np.asarray(support_all[mode, :, 1])
        summaries["In-support Mean"] = np.asarray(support_all[mode])
        support_retention = np.asarray(retention_all[mode])
    if any(value.shape[0] != truth.shape[0] for value in [snr, *estimates.values()]):
        raise ValueError(f"Length mismatch among cached summaries for {case}")
    if support_retention is not None and support_retention.shape != (truth.shape[0],):
        raise ValueError(f"In-support retention shape mismatch for {case}")
    return {
        "case": case,
        "model": model,
        "dataset": dataset,
        "root": root,
        "truth": np.asarray(truth),
        "snr": np.asarray(snr),
        "estimates": estimates,
        "mean_summary": np.asarray(mean_all[mode]),
        "summaries": summaries,
        "support_retention": support_retention,
        "support_manifest": (
            root / "in_support_meta" / "manifest.json"
            if support_retention is not None
            else None
        ),
    }


def _validate_pp_provenance(root: Path, sample_files: list[Path]) -> None:
    """Ensure raw draws correspond to the posterior summaries being reported."""
    meta_dir = root / "meta"
    flags = []
    for sample_path in sample_files:
        meta_path = meta_dir / sample_path.with_suffix(".json").name
        if not meta_path.is_file():
            raise PosteriorPPUnavailable(
                f"P-P plot unavailable: missing provenance file {meta_path}"
            )
        try:
            payload = json.loads(meta_path.read_text())
            flag = payload["args"]["cancel_add_noise"]
        except (KeyError, TypeError, ValueError) as exc:
            raise PosteriorPPUnavailable(
                f"P-P plot unavailable: invalid provenance in {meta_path}"
            ) from exc
        if not isinstance(flag, bool):
            raise PosteriorPPUnavailable(
                f"P-P plot unavailable: cancel_add_noise is not Boolean in {meta_path}"
            )
        flags.append(flag)
    if len(set(flags)) != 1:
        raise PosteriorPPUnavailable(
            "P-P plot unavailable: cancel_add_noise differs across partitions"
        )
    if flags[0]:
        raise PosteriorPPUnavailable(
            "P-P plot unavailable: this legacy cache stores only the original "
            "draw bank, while its Mean summaries combine original and rotated "
            "draws for additive-noise cancellation"
        )


def load_shear_pit_values(
    case: dict, mode: int, block_size: int = 64
) -> dict[str, np.ndarray]:
    """Return prior-matched conditional midpoint ranks for g1 and g2.

    For each component, the galaxy's true shear selects one of the fixed
    intervals in ``SHEAR_PP_BIN_EDGES``. The posterior is then conditioned on
    the matching restricted prior by retaining only draws of that component in
    the same interval. The finite-sample rank within those retained draws is
    ``(n_less + 0.5 * n_equal + 0.5) / (n_retained + 1)``. Raw sample
    partitions are memory-mapped and only a small block of their two shear
    columns is materialized at once.
    """
    if mode < 0:
        raise IndexError("mode must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    truth = np.asarray(case["truth"])
    if truth.ndim != 2 or truth.shape[1] < 2:
        raise ValueError("case truth must have shape (galaxy, feature>=2)")
    if not len(truth):
        raise ValueError("cannot build a P-P plot for zero galaxies")

    root = Path(case["root"])
    sample_files = partition_files(root / "sample")
    truth_files = partition_files(root / "truth")
    if len(sample_files) != len(truth_files):
        raise ValueError("sample and truth caches have different partition counts")
    if [path.name for path in sample_files] != [path.name for path in truth_files]:
        raise ValueError("sample and truth partition labels do not match")
    _validate_pp_provenance(root, sample_files)

    pit = np.full((len(truth), 2), np.nan, dtype=np.float64)
    retained = np.zeros((len(truth), 2), dtype=np.int64)
    offset = 0
    for sample_path, truth_path in zip(sample_files, truth_files):
        if offset >= len(truth):
            break
        samples = np.load(sample_path, mmap_mode="r")
        partition_truth = np.load(truth_path, mmap_mode="r")
        if samples.ndim != 4 or samples.shape[-1] < 2:
            raise ValueError(
                f"Expected sample shape (mode, galaxy, draw, feature>=2), got "
                f"{samples.shape} in {sample_path}"
            )
        if partition_truth.ndim != 2 or partition_truth.shape[1] < 2:
            raise ValueError(f"Invalid truth shape {partition_truth.shape} in {truth_path}")
        if samples.shape[1] != partition_truth.shape[0]:
            raise ValueError(f"Galaxy-count mismatch between {sample_path} and {truth_path}")
        if mode >= samples.shape[0]:
            raise IndexError(
                f"Mode {mode} outside [0, {samples.shape[0]}) for {sample_path}"
            )

        take = min(samples.shape[1], len(truth) - offset)
        expected_truth = truth[offset : offset + take, :2]
        stored_truth = np.asarray(partition_truth[:take, :2])
        if not np.array_equal(stored_truth, expected_truth, equal_nan=True):
            raise ValueError(
                f"Truth rows in {truth_path} do not align with the loaded case prefix"
            )

        for local_start in range(0, take, block_size):
            local_end = min(take, local_start + block_size)
            draws = np.asarray(samples[mode, local_start:local_end, :, :2])
            targets = expected_truth[local_start:local_end]
            ranks = np.full(targets.shape, np.nan, dtype=np.float64)
            block_retained = np.zeros(targets.shape, dtype=np.int64)
            for component in range(2):
                truth_bins = shear_pp_bin_indices(targets[:, component])
                draw_bins = shear_pp_bin_indices(draws[:, :, component])
                for bin_index in range(len(SHEAR_PP_BIN_EDGES) - 1):
                    galaxy_mask = truth_bins == bin_index
                    if not np.any(galaxy_mask):
                        continue
                    component_draws = draws[galaxy_mask, :, component]
                    kept = (
                        np.isfinite(component_draws)
                        & (draw_bins[galaxy_mask] == bin_index)
                    )
                    n_kept = kept.sum(axis=1)
                    component_targets = targets[galaxy_mask, component]
                    n_less = (
                        (component_draws < component_targets[:, None]) & kept
                    ).sum(axis=1)
                    n_equal = (
                        (component_draws == component_targets[:, None]) & kept
                    ).sum(axis=1)
                    component_ranks = np.full(
                        len(component_targets), np.nan, dtype=np.float64
                    )
                    valid = np.isfinite(component_targets) & (n_kept > 0)
                    component_ranks[valid] = (
                        n_less[valid] + 0.5 * n_equal[valid] + 0.5
                    ) / (n_kept[valid] + 1.0)
                    ranks[galaxy_mask, component] = component_ranks
                    block_retained[galaxy_mask, component] = n_kept
            start = offset + local_start
            stop = offset + local_end
            pit[start:stop] = ranks
            retained[start:stop] = block_retained
        offset += take
        del samples, partition_truth

    if offset != len(truth):
        raise ValueError(
            f"Sample cache contains {offset} galaxies, but report case has {len(truth)}"
        )
    return {
        "g1": pit[:, 0],
        "g2": pit[:, 1],
        "g1_retained": retained[:, 0],
        "g2_retained": retained[:, 1],
    }


def fit_design(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    dof = max(1, len(y) - design.shape[1])
    cov = np.linalg.pinv(design.T @ design) * (residual @ residual / dof)
    return coef, np.sqrt(np.maximum(np.diag(cov), 0))


def component_metrics(truth: np.ndarray, estimate: np.ndarray, low_g: float) -> dict:
    residual = estimate - truth
    additive = float(np.mean(residual))
    additive_se = float(np.std(residual, ddof=1) / np.sqrt(len(residual)))
    low = np.abs(truth) < low_g
    linear, linear_se = fit_design(
        residual[low], np.column_stack([np.ones(low.sum()), truth[low]])
    )
    cubic, cubic_se = fit_design(
        residual,
        np.column_stack([np.ones(len(truth)), truth, truth**3]),
    )
    return {
        "c": additive,
        "c_se": additive_se,
        "low_c": float(linear[0]),
        "low_c_se": float(linear_se[0]),
        "low_m": float(linear[1]),
        "low_m_se": float(linear_se[1]),
        "cubic_c": float(cubic[0]),
        "cubic_m": float(cubic[1]),
        "cubic_q": float(cubic[2]),
        "cubic_m_se": float(cubic_se[1]),
        "cubic_q_se": float(cubic_se[2]),
    }


def spin_metrics(theta: np.ndarray, residual: np.ndarray) -> dict:
    design = np.column_stack(
        [
            np.ones(len(theta)),
            np.cos(2 * theta),
            np.sin(2 * theta),
            np.cos(4 * theta),
            np.sin(4 * theta),
        ]
    )
    coef, _ = fit_design(residual, design)
    return {
        "offset": float(coef[0]),
        "spin2": float(np.hypot(coef[1], coef[2])),
        "spin4": float(np.hypot(coef[3], coef[4])),
        "coef": coef,
    }


def img_to_galaxy_clockwise(
    g1: np.ndarray, g2: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use the shared clockwise ``theta_int`` shear transform."""
    return _img_to_gal_axis(g1, g2, theta)


def galaxy_frame_components(
    truth: np.ndarray, estimate: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return true and estimated (g_plus, g_cross) using true theta_int."""
    theta = truth[:, 2]
    true_components = img_to_galaxy_clockwise(truth[:, 0], truth[:, 1], theta)
    estimate_components = img_to_galaxy_clockwise(
        estimate[:, 0], estimate[:, 1], theta
    )
    return true_components, estimate_components


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Return a finite-pair Pearson correlation and contributing sample size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def galaxy_frame_diagnostics(case: dict, low_g: float) -> tuple[list[dict], list[dict]]:
    """Compute galaxy-frame bias and coupled nuisance-error diagnostics."""
    metric_rows = []
    correlation_rows = []
    truth = case["truth"]
    for estimator, estimate in case["estimates"].items():
        true_components, estimate_components = galaxy_frame_components(truth, estimate)
        for label, true_component, estimate_component in zip(
            ("g+ (E)", "gx (B)"), true_components, estimate_components
        ):
            residual = estimate_component - true_component
            row = component_metrics(true_component, estimate_component, low_g)
            row.update(
                estimator=estimator,
                component=label,
                n_low=int(np.count_nonzero(np.abs(true_component) < low_g)),
            )
            metric_rows.append(row)
            for nuisance, index in NUISANCE_INDICES.items():
                nuisance_error = estimate[:, index] - truth[:, index]
                correlation, n = pearson_correlation(residual, nuisance_error)
                correlation_rows.append(
                    {
                        "estimator": estimator,
                        "component": label,
                        "nuisance": nuisance,
                        "correlation": correlation,
                        "n": n,
                    }
                )
    return metric_rows, correlation_rows


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def coverage_metrics(
    truth: np.ndarray,
    summary: np.ndarray,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> list[dict]:
    """Compute empirical coverage of cached 16th--84th posterior intervals.

    The theta interval is represented around its cached circular center and may
    cross the -pi/pi seam, so it cannot use a direct lower <= truth <= upper
    comparison.
    """
    truth = np.asarray(truth)
    summary = np.asarray(summary)
    if summary.ndim != 3 or summary.shape[1] != 3:
        raise ValueError("mean summary must have shape (galaxy, 3, feature)")
    if truth.shape != (summary.shape[0], summary.shape[2]):
        raise ValueError("truth and mean-summary shapes are inconsistent")
    if len(feature_names) != truth.shape[1]:
        raise ValueError("feature_names must match the cached feature dimension")

    rows = []
    for index, name in enumerate(feature_names):
        lower = summary[:, 0, index]
        center = summary[:, 1, index]
        upper = summary[:, 2, index]
        target = truth[:, index]
        finite = (
            np.isfinite(lower)
            & np.isfinite(center)
            & np.isfinite(upper)
            & np.isfinite(target)
        )
        if name == "theta_int":
            lower_delta = wrap_angle(lower - center)
            upper_delta = wrap_angle(upper - center)
            target_delta = wrap_angle(target - center)
            covered = (target_delta >= lower_delta) & (target_delta <= upper_delta)
        else:
            covered = (target >= lower) & (target <= upper)
        covered = covered[finite]
        n = len(covered)
        fraction = float(np.mean(covered)) if n else float("nan")
        se = (
            float(np.sqrt(fraction * (1.0 - fraction) / n))
            if n and np.isfinite(fraction)
            else float("nan")
        )
        rows.append(
            {
                "parameter": name,
                "coverage": fraction,
                "coverage_se": se,
                "delta": fraction - 0.68,
                "n": n,
            }
        )
    return rows


def binned(x: np.ndarray, y: np.ndarray, bins: int):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1)
    index = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    center, mean, error = [], [], []
    for i in range(bins):
        mask = index == i
        if mask.sum() < 2:
            continue
        center.append(np.mean(x[mask]))
        mean.append(np.mean(y[mask]))
        error.append(np.std(y[mask], ddof=1) / np.sqrt(mask.sum()))
    return np.asarray(center), np.asarray(mean), np.asarray(error)


def quantile_binned(x: np.ndarray, y: np.ndarray, bins: int):
    """Bin finite pairs into nearly equal-count bins and return mean +/- SEM."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have matching shapes")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return (np.array([], dtype=float),) * 3
    n_bins = min(bins, len(x) // 2)
    groups = np.array_split(np.argsort(x, kind="stable"), n_bins)
    center = np.asarray([np.mean(x[group]) for group in groups])
    mean = np.asarray([np.mean(y[group]) for group in groups])
    error = np.asarray(
        [np.std(y[group], ddof=1) / np.sqrt(len(group)) for group in groups]
    )
    return center, mean, error


def fig_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def bias_figure(case: dict, bins: int) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for component in range(2):
        true = case["truth"][:, component]
        for name, estimate in case["estimates"].items():
            residual = estimate[:, component] - true
            x, y, error = binned(true, residual, bins)
            axes[0, component].errorbar(
                x, y,
                yerr=error,
                marker="o", ms=3, lw=1, label=name,
                color=ESTIMATOR_COLORS.get(name, "tab:gray"),
            )
            theta = case["truth"][:, 2]
            tx, ty, terror = binned(theta, residual, bins)
            axes[1, component].errorbar(
                tx, ty,
                yerr=terror,
                marker="o", ms=3, lw=1,
                label=name, color=ESTIMATOR_COLORS.get(name, "tab:gray"),
            )
        axes[0, component].axhline(0, color="black", ls="--", lw=0.8)
        axes[1, component].axhline(0, color="black", ls="--", lw=0.8)
        axes[0, component].set_title(f"g{component + 1} residual vs truth")
        axes[1, component].set_title(f"g{component + 1} residual vs theta_int")
        axes[0, component].set_ylabel("estimate - truth")
        axes[1, component].set_ylabel("estimate - truth")
        axes[0, component].legend()
    axes[0, 0].set_xlabel("true g1")
    axes[0, 1].set_xlabel("true g2")
    axes[1, 0].set_xlabel("theta_int [rad]")
    axes[1, 1].set_xlabel("theta_int [rad]")
    fig.suptitle(case["case"])
    fig.tight_layout()
    return fig_data_uri(fig)


def galaxy_frame_figure(case: dict, bins: int) -> str:
    """Plot galaxy-frame residuals against inclination in physical shear units."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sini = case["truth"][:, 3]
    for estimator, estimate in case["estimates"].items():
        true_components, estimate_components = galaxy_frame_components(
            case["truth"], estimate
        )
        for axis, label, true_component, estimate_component in zip(
            axes,
            (r"$g_+$ (E)", r"$g_\times$ (B)"),
            true_components,
            estimate_components,
        ):
            residual = estimate_component - true_component
            x, y, error = binned(sini, residual, bins)
            axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                ms=3,
                lw=1,
                label=estimator,
                color=ESTIMATOR_COLORS.get(estimator, "tab:gray"),
            )
            axis.set_title(f"{label} residual vs sin i")
            axis.set_xlabel("true sin i")
            axis.set_ylabel("estimate - truth")
            axis.axhline(0, color="black", ls="--", lw=0.8)
            axis.legend()
    fig.suptitle(f"{case['case']} — clockwise galaxy frame")
    fig.tight_layout()
    return fig_data_uri(fig)


def nuisance_correlation_figure(case: dict, bins: int) -> str:
    """Plot Mean galaxy-frame shear residuals versus nuisance errors."""
    try:
        estimate = case["estimates"]["Mean"]
    except KeyError as exc:
        raise ValueError("nuisance plots require the Mean estimator") from exc
    truth = case["truth"]
    true_components, estimate_components = galaxy_frame_components(truth, estimate)
    residuals = tuple(
        estimated - target
        for target, estimated in zip(true_components, estimate_components)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, (nuisance, index) in zip(axes, NUISANCE_INDICES.items()):
        nuisance_error = estimate[:, index] - truth[:, index]
        for component, residual, label in zip(
            ("g+", "gx"),
            residuals,
            (r"$g_+$ (E)", r"$g_\times$ (B)"),
        ):
            x, y, error = quantile_binned(nuisance_error, residual, bins)
            errorbar = axis.errorbar(
                x,
                y,
                yerr=error,
                marker="o",
                ms=3,
                lw=1,
                label="_nolegend_",
                color=GALAXY_COMPONENT_COLORS[component],
            )
            errorbar.lines[0].set_label(label)
        axis.axhline(0, color="black", ls="--", lw=0.8)
        axis.axvline(0, color="0.5", ls=":", lw=0.8)
        axis.set_title(nuisance)
        axis.set_xlabel(NUISANCE_LABELS[nuisance])
        axis.set_ylabel("shear estimate - truth")
        axis.legend()
    fig.suptitle(
        f"{case['case']} — Mean galaxy-frame shear residual vs nuisance error"
    )
    fig.tight_layout()
    return fig_data_uri(fig)


def posterior_pp_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return sorted finite PIT values, their ECDF, and the KS distance."""
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("posterior PIT values must lie in [0, 1]")
    if not len(values):
        return values, np.array([], dtype=float), float("nan")
    empirical = np.arange(1, len(values) + 1, dtype=float) / len(values)
    lower = np.arange(len(values), dtype=float) / len(values)
    distance = float(
        max(np.max(empirical - values), np.max(values - lower))
    )
    return values, empirical, distance


def shear_pp_bin_indices(
    values: np.ndarray, edges: np.ndarray = SHEAR_PP_BIN_EDGES
) -> np.ndarray:
    """Assign values to half-open shear bins, closing the final interval.

    Values within a few source-dtype ulps of any edge are snapped to that edge
    before assignment. This keeps float32 cached values and float64 report
    edges from disagreeing about boundary ownership.
    """
    raw_values = np.asarray(values)
    values = np.asarray(raw_values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("shear P-P bin edges must be strictly increasing")
    dtype_epsilon = (
        np.finfo(raw_values.dtype).eps
        if np.issubdtype(raw_values.dtype, np.floating)
        else np.finfo(float).eps
    )
    endpoint_tolerance = 4.0 * dtype_epsilon * max(1.0, np.max(np.abs(edges)))
    snapped = values.copy()
    for edge in edges:
        snapped[np.abs(snapped - edge) <= endpoint_tolerance] = edge

    finite_inside = (
        np.isfinite(snapped)
        & (snapped >= edges[0])
        & (snapped <= edges[-1])
    )
    assigned = np.searchsorted(edges, snapped, side="right") - 1
    assigned[snapped == edges[-1]] = len(edges) - 2
    valid_assignment = finite_inside & (assigned >= 0) & (assigned < len(edges) - 1)
    result = np.full(values.shape, -1, dtype=np.int8)
    result[valid_assignment] = assigned[valid_assignment]
    return result


def shear_pp_bin_masks(
    truth: np.ndarray, edges: np.ndarray = SHEAR_PP_BIN_EDGES
) -> list[np.ndarray]:
    """Return non-overlapping true-shear masks with a closed final interval."""
    truth = np.asarray(truth)
    if truth.ndim != 1:
        raise ValueError("true shear must be one-dimensional")
    indices = shear_pp_bin_indices(truth, edges)
    return [indices == index for index in range(len(edges) - 1)]


def posterior_pp_figure(pits: dict[str, np.ndarray], truth: np.ndarray) -> str:
    """Plot prior-matched g1/g2 rank ECDFs in fixed true-shear bins."""
    truth = np.asarray(truth)
    if truth.ndim != 2 or truth.shape[1] < 2:
        raise ValueError("truth must have shape (galaxy, feature>=2)")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    grid = np.linspace(0.0, 1.0, 501)
    colors = ("tab:blue", "tab:green", "tab:red")
    for component_index, (axis, component) in enumerate(zip(axes, ("g1", "g2"))):
        if component not in pits:
            raise KeyError(f"Missing posterior PIT values for {component}")
        component_pits = np.asarray(pits[component], dtype=float)
        if component_pits.shape != (len(truth),):
            raise ValueError(
                f"{component} PIT values must have shape ({len(truth)},)"
            )
        retained_key = f"{component}_retained"
        if retained_key not in pits:
            raise KeyError(f"Missing retained-draw counts for {component}")
        component_retained = np.asarray(pits[retained_key])
        if component_retained.shape != (len(truth),):
            raise ValueError(
                f"{component} retained counts must have shape ({len(truth)},)"
            )
        curves = []
        masks = shear_pp_bin_masks(truth[:, component_index])
        for bin_index, mask in enumerate(masks):
            values, empirical, distance = posterior_pp_curve(component_pits[mask])
            retained_in_bin = component_retained[mask]
            finite_rank = np.isfinite(component_pits[mask])
            median_retained = (
                float(np.median(retained_in_bin[finite_rank]))
                if np.any(finite_rank)
                else float("nan")
            )
            zero_retained = int(np.count_nonzero(retained_in_bin == 0))
            curves.append(
                (
                    bin_index,
                    values,
                    empirical,
                    distance,
                    int(np.count_nonzero(mask)),
                    median_retained,
                    zero_retained,
                )
            )

        nonempty_sizes = [
            len(values) for _, values, _, _, _, _, _ in curves if len(values)
        ]
        for (
            bin_index,
            values,
            empirical,
            distance,
            total,
            median_retained,
            zero_retained,
        ) in curves:
            lower = SHEAR_PP_BIN_EDGES[bin_index]
            upper = SHEAR_PP_BIN_EDGES[bin_index + 1]
            closing = "]" if bin_index == len(curves) - 1 else ")"
            interval = f"[{lower:+.4f}, {upper:+.4f}{closing}"
            n = len(values)
            label = f"true {component} in {interval}; N={n:,}/{total:,}"
            if n:
                label += (
                    f", median retained={median_retained:,.0f}, "
                    f"zero retained={zero_retained:,}, KS D={distance:.3f}"
                )
                epsilon = np.sqrt(np.log(2.0 / 0.05) / (2.0 * n))
                axis.fill_between(
                    grid,
                    np.maximum(0.0, grid - epsilon),
                    np.minimum(1.0, grid + epsilon),
                    color=colors[bin_index],
                    alpha=0.08,
                )
            axis.step(
                values,
                empirical,
                where="post",
                color=colors[bin_index],
                lw=1.4,
                label=label,
            )
        if not nonempty_sizes:
            axis.text(
                0.5,
                0.5,
                "No finite posterior ranks in requested shear range",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        included = np.logical_or.reduce(masks)
        excluded = int(np.count_nonzero(np.isfinite(truth[:, component_index]) & ~included))
        suffix = f"; {excluded} outside range" if excluded else ""
        axis.set_title(f"{component}{suffix}")
        axis.plot(grid, grid, color="black", ls="--", lw=0.9, label="Uniform")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Restricted-posterior quantile")
        axis.legend(loc="upper left", fontsize="x-small")
    axes[0].set_ylabel("Fraction of truths below quantile")
    fig.suptitle("Prior-matched conditional posterior P-P diagnostic")
    fig.tight_layout()
    return fig_data_uri(fig)


def compute_metrics(case: dict, low_g: float) -> list[dict]:
    rows = []
    theta = case["truth"][:, 2]
    sini = case["truth"][:, 3]
    for estimator, estimate in case["estimates"].items():
        for component in range(2):
            true = case["truth"][:, component]
            residual = estimate[:, component] - true
            base = component_metrics(true, estimate[:, component], low_g)
            base.update(spin_metrics(theta, residual))
            cuts = {}
            for cut in SINI_CUTS:
                mask = np.ones(len(sini), dtype=bool) if np.isinf(cut) else sini < cut
                mask &= np.abs(true) < low_g
                coef, se = fit_design(
                    residual[mask], np.column_stack([np.ones(mask.sum()), true[mask]])
                )
                cuts["all" if np.isinf(cut) else f"{cut:.1f}"] = (
                    float(coef[1]), float(se[1]), int(mask.sum())
                )
            base.update(
                estimator=estimator,
                component=f"g{component + 1}",
                sini_cuts=cuts,
            )
            rows.append(base)
    return rows


def fmt(value: float) -> str:
    return f"{value:.3e}"


def fmt_scaled(value: float, scale: float) -> str:
    """Format a metric in report units without changing stored physical units."""
    return f"{scale * value:.3f}"


def metrics_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['estimator']}</td><td>{row['component']}</td>"
            f"<td>{fmt_scaled(row['c'], ADDITIVE_DISPLAY_SCALE)} ± "
            f"{fmt_scaled(row['c_se'], ADDITIVE_DISPLAY_SCALE)}</td>"
            f"<td>{fmt_scaled(row['low_m'], MULTIPLICATIVE_DISPLAY_SCALE)} ± "
            f"{fmt_scaled(row['low_m_se'], MULTIPLICATIVE_DISPLAY_SCALE)}</td>"
            f"<td>{fmt_scaled(row['cubic_m'], MULTIPLICATIVE_DISPLAY_SCALE)} ± "
            f"{fmt_scaled(row['cubic_m_se'], MULTIPLICATIVE_DISPLAY_SCALE)}</td>"
            f"<td>{fmt(row['cubic_q'])} ± {fmt(row['cubic_q_se'])}</td>"
            f"<td>{fmt_scaled(row['spin2'], ADDITIVE_DISPLAY_SCALE)}</td>"
            f"<td>{fmt_scaled(row['spin4'], ADDITIVE_DISPLAY_SCALE)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Component</th>"
        "<th>10<sup>4</sup> mean c</th><th>10<sup>2</sup> low-|g| m</th>"
        "<th>10<sup>2</sup> cubic m</th><th>cubic q (unscaled)</th>"
        "<th>10<sup>4</sup> spin-2 amp</th>"
        "<th>10<sup>4</sup> spin-4 amp</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def cuts_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        for cut, (slope, se, n) in row["sini_cuts"].items():
            body.append(
                f"<tr><td>{row['estimator']}</td><td>{row['component']}</td>"
                f"<td>{html.escape(cut)}</td>"
                f"<td>{fmt_scaled(slope, MULTIPLICATIVE_DISPLAY_SCALE)} ± "
                f"{fmt_scaled(se, MULTIPLICATIVE_DISPLAY_SCALE)}</td>"
                f"<td>{n:,}</td></tr>"
            )
    return (
        "<table><thead><tr><th>Estimator</th><th>Component</th><th>sin i cut</th>"
        "<th>10<sup>2</sup> low-|g| m</th><th>N</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def galaxy_metrics_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            f"<tr><td>{row['estimator']}</td><td>{row['component']}</td>"
            f"<td>{fmt_scaled(row['c'], ADDITIVE_DISPLAY_SCALE)} ± "
            f"{fmt_scaled(row['c_se'], ADDITIVE_DISPLAY_SCALE)}</td>"
            f"<td>{fmt_scaled(row['low_m'], MULTIPLICATIVE_DISPLAY_SCALE)} ± "
            f"{fmt_scaled(row['low_m_se'], MULTIPLICATIVE_DISPLAY_SCALE)}</td>"
            f"<td>{row['n_low']:,}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Galaxy component</th>"
        "<th>10<sup>4</sup> mean c</th>"
        "<th>10<sup>2</sup> low-|g| m</th><th>N low-|g|</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def correlations_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        correlation = row["correlation"]
        rendered = "n/a" if not np.isfinite(correlation) else f"{correlation:+.3f}"
        body.append(
            f"<tr><td>{row['estimator']}</td><td>{row['component']}</td>"
            f"<td>{row['nuisance']}</td><td>{rendered}</td>"
            f"<td>{row['n']:,}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Galaxy component</th>"
        "<th>Nuisance error</th><th>Pearson r</th><th>N</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def coverage_table(rows: list[dict]) -> str:
    body = []
    for row in rows:
        body.append(
            f"<tr><td>{row['estimator']}</td><td>{row['parameter']}</td>"
            f"<td>{100.0 * row['coverage']:.1f}% ± "
            f"{100.0 * row['coverage_se']:.1f}%</td>"
            f"<td>{100.0 * row['delta']:+.1f} pp</td>"
            f"<td>{row['n']:,}</td></tr>"
        )
    return (
        "<table><thead><tr><th>Estimator</th><th>Parameter</th>"
        "<th>16th–84th coverage</th>"
        "<th>Difference from 68%</th><th>N finite</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def retention_table(retention: np.ndarray) -> str:
    """Render joint all-parameter draw retention for the selected mode."""
    values = np.asarray(retention, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError("No finite in-support retention fractions")
    lower, median, upper = np.percentile(finite, (16, 50, 84))
    body = (
        "<tr><td>In-support Mean</td>"
        f"<td>{100.0 * np.mean(finite):.2f}%</td>"
        f"<td>{100.0 * median:.2f}%</td>"
        f"<td>{100.0 * lower:.2f}%–{100.0 * upper:.2f}%</td>"
        f"<td>{100.0 * np.min(finite):.2f}%</td>"
        f"<td>{np.count_nonzero(finite == 0.0):,}</td>"
        f"<td>{len(finite):,}</td></tr>"
    )
    return (
        "<table><thead><tr><th>Estimator</th><th>Mean retained</th>"
        "<th>Median retained</th><th>16th–84th galaxy range</th>"
        "<th>Minimum</th><th>Galaxies with zero draws</th><th>N</th>"
        "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def main():
    args = parse_args()
    cases = [
        load_case(args.cache_root, item, args.mode, args.max_galaxies)
        for item in args.case
    ]
    sections = []
    for case in cases:
        metrics = compute_metrics(case, args.low_g)
        galaxy_metrics, correlations = galaxy_frame_diagnostics(case, args.low_g)
        coverage = []
        for estimator, summary in case["summaries"].items():
            estimator_rows = coverage_metrics(case["truth"], summary)
            for row in estimator_rows:
                row["estimator"] = estimator
            coverage.extend(estimator_rows)
        try:
            shear_pits = load_shear_pit_values(case, args.mode)
            pp_section = (
                "<h3>Prior-matched conditional posterior P-P diagnostic</h3>"
                "<p>These curves use the raw posterior draws for the selected "
                "cache mode. For each component, the truth selects one of three "
                "equal-width intervals from -0.1 to 0.1, and that component's "
                "posterior draws are restricted to the same true-shear interval "
                "and renormalized before computing the finite-sample midpoint "
                "rank. Other parameters are not restricted. This matches the "
                "posterior and truth-generating prior within each bin, so an exact "
                "posterior should follow the diagonal. The legend reports the "
                "median retained draws per galaxy; a galaxy with zero retained "
                "draws has no rank and is excluded from its curve. "
                "Each faint 95% DKW band uses that bin's finite galaxy count and "
                "represents galaxy-sampling uncertainty only, not posterior Monte "
                "Carlo error. Exact-D4 draws are balanced across group components "
                "rather than independent.</p>"
                f"<img src=\"{posterior_pp_figure(shear_pits, case['truth'])}\" "
                "alt=\"g1 and g2 posterior P-P calibration\">"
            )
        except (PosteriorPPUnavailable, FileNotFoundError) as exc:
            pp_section = (
                "<h3>Prior-matched conditional posterior P-P diagnostic</h3>"
                f"<p class=\"note\">{html.escape(str(exc))}</p>"
            )
        support_section = ""
        if case["support_retention"] is not None:
            manifest = case["support_manifest"]
            manifest_note = (
                f" Provenance: <code>{html.escape(str(manifest))}</code>."
                if manifest is not None and manifest.is_file()
                else ""
            )
            support_section = (
                "<h3>Archived-prior support diagnostic</h3>"
                "<p><b>In-support Mean</b> conditions the cached draws on all eight "
                "parameters being finite and inside their archived inclusive prior "
                "bounds. It is a diagnostic truncated posterior, not the original "
                "model posterior, and can expose rather than repair learned support "
                "mismatch."
                + manifest_note
                + "</p>"
                + retention_table(case["support_retention"])
            )
        sections.append(
            f"<section><h2>{html.escape(case['case'])}</h2>"
            f"<p><b>N:</b> {len(case['truth']):,}; <b>mode index:</b> {args.mode}; "
            f"<b>cache:</b> <code>{html.escape(str(case['root']))}</code></p>"
            + metrics_table(metrics)
            + support_section
            + "<h3>Inclination-cut diagnostic</h3>"
            + cuts_table(metrics)
            + f"<img src=\"{bias_figure(case, args.bins)}\" alt=\"bias diagnostics\">"
            + "<h3>Clockwise galaxy-frame E/B diagnostic</h3>"
            + "<p>The transform uses the true <code>theta_int</code>, whose positive "
            + "direction is clockwise in this repository. A nonzero E residual can "
            + "therefore expose shear leakage aligned with each galaxy even when "
            + "detector-frame additive bias averages to zero.</p>"
            + galaxy_metrics_table(galaxy_metrics)
            + f"<img src=\"{galaxy_frame_figure(case, args.bins)}\" "
            + "alt=\"clockwise galaxy-frame residuals versus inclination\">"
            + "<h3>Coupled nuisance-error diagnostic</h3>"
            + "<p>Pearson correlations compare each galaxy-frame shear residual "
            + "with the same estimator's nuisance-parameter error. They are "
            + "descriptive associations, not evidence of causation. The three "
            + "plots show only the primary Mean estimator, with equal-count bins "
            + "and unscaled physical shear residuals.</p>"
            + correlations_table(correlations)
            + f"<img src=\"{nuisance_correlation_figure(case, args.bins)}\" "
            + "alt=\"Mean galaxy-frame shear residuals versus nuisance errors\">"
            + "<h3>Posterior interval coverage</h3>"
            + "<p>Coverage uses each available estimator's cached 16th/84th "
            + "percentiles. The theta interval is evaluated circularly across the "
            + "-pi/pi seam. In-support coverage, when present, describes the "
            + "diagnostically truncated draws rather than the original posterior.</p>"
            + coverage_table(coverage)
            + pp_section
            + "</section>"
        )
    generated = datetime.now(timezone.utc).isoformat()
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>KL-NN shear-bias diagnostics</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 1.5rem; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ccd1d5; padding: 0.45rem; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
th {{ background: #f1f3f4; }} img {{ max-width: 100%; height: auto; }}
.note {{ background: #fff7df; border-left: 4px solid #e6a700; padding: 0.8rem 1rem; }}
code {{ overflow-wrap: anywhere; }}
</style></head><body>
<h1>KL-NN shear-bias diagnostics</h1>
<p>Generated {generated}. Most diagnostics use compact truth/SNR/MAP/mean
summaries. The P-P diagnostic streams memory-mapped g1/g2 posterior draws
partition by partition and never concatenates the raw sample bank, keeping the
report suitable for a 4 GB CPU job.</p>
<section><h2>Estimator and cache provenance</h2>
<p>This report evaluates the point estimates and posterior summaries already
stored in each requested cache. It does not apply an additional symmetry,
rotation-pairing, or response correction. Exact-D4 caches therefore retain their
native D4 posterior treatment; legacy caches retain whatever treatment was used
when they were generated. If a legacy cache used 90° cancellation, verify that
it was generated after the rotation-pairing ordering fix before interpreting it.</p>
<p>If <code>--max-galaxies N</code> is supplied, every case uses its first N
combined cache rows. This supports matched-prefix comparisons only when those
caches were generated from the same ordered galaxy sample and seed.</p></section>
<section><h2>Cached MAP/mean audit</h2>
<p class="note"><b>Interpretation:</b> Reported additive quantities use 10<sup>4</sup> shear units and multiplicative slopes use 10<sup>2</sup>; the approximate requirements are |10<sup>4</sup> c| &lt; 1 and |10<sup>2</sup> m| &lt; 1. Uncertainties must still be considered when judging either threshold. Mean <i>c</i> is the unbinned average residual. The low-|g| fit uses |g| &lt; {args.low_g:g}. The cubic fit is residual = c + m g + q g³; q is left unscaled because it has different units. Spin amplitudes come from a joint constant + spin-2 + spin-4 regression against true theta_int.</p>
<p class="note"><b>Estimator caveats:</b> Treat posterior <b>Mean</b> as the primary point estimator. <b>In-support Mean</b>, if present, is a diagnostic conditioning on all archived prior bounds and is not a replacement for a bounded-support model. Cached MAP values depend on a finite posterior draw set and an argmax over cached log probabilities, so they can be unstable—especially if non-finite log probabilities are present. MAP has no interval-coverage entry. Galaxy-frame results use the true orientation to diagnose conditional leakage; they are not an inference procedure that assumes true orientations will be known for observed galaxies.</p>
{''.join(sections)}
</section>
</body></html>"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
