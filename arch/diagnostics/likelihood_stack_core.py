"""Shared math for the shared-η likelihood-stack campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ABS_G_MIN = 0.02
ABS_G_SPLIT = 0.05
NOMINAL_COVERAGE = 0.68
N_SELECTED = 128
N_REALIZATIONS = 16
N_SAMPLES = 2048
PROPOSAL_PER_REALIZATION = 256
PREFIXES = (1, 2, 4, 8, 16)
NOISE_STRIDE = 1009
SPEC_SEED_OFFSET = 17
G01_NPE = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
G01_DATASET = "test_100k_simv3_cosi_xu3_tf"
G01_CACHE_TAG = "testset_tfweighted_v2_10k_s42_righthanded"
G01_CASE = f"{G01_NPE}:{G01_DATASET}_{G01_CACHE_TAG}"
CACHE_ROOT = Path("/ocean/projects/phy250048p/shared/cache")
DATA_ROOT = Path("/ocean/projects/phy250048p/shared/datasets")
MODEL_ROOT = Path("/ocean/projects/phy250048p/shared/models")
REPORT_ROOT = Path("/ocean/projects/phy250048p/shared/reports/likelihood-stack")
SELECT_DIR = REPORT_ROOT / "00_select"
STACK_DIR = REPORT_ROOT / "01_stack"
COVERAGE_DIR = REPORT_ROOT / "02_coverage"
NRE_DIR = REPORT_ROOT / "03_nre"
NRE_NAME = "CNN-CNN-Meta-nre2d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
NRE_TRAIN_DATASET = "valid_100k_simv3_cosi"
NRE_VALID_DATASET = "small_10k_simv3_cosi"
NRE_INFER_DATASET = G01_DATASET
NRE_PREFIXES = (1, 8, 32, 128)
NRE_GRID_N = 21
NRE_HIDDEN_DIMS = (512, 256)
NRE_CONTEXT_DIM = 1152
NRE_EPOCHS = 80
NRE_BATCH_SIZE = 256
NRE_LR = 1e-3
NRE_ABS_G_EDGES = (0.0, 0.025, 0.05, 0.075, 0.1)
HTML_STYLE = """
body { font: 17px/1.55 Palatino, "Palatino Linotype", serif; margin: 2rem auto; max-width: 980px; color: #1b1b1b; }
h1, h2, h3 { font-weight: 600; }
h1 { font-size: 1.85rem; }
h2 { margin-top: 2.4rem; }
p.lead { font-size: 1.08rem; }
figure { margin: 1.4rem 0 2rem; }
figcaption { font-size: 0.92rem; color: #444; margin-top: 0.45rem; }
img { max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 1rem 0 1.6rem; }
th, td { border: 1px solid #ccc; padding: 0.35rem 0.55rem; }
th { background: #f4f4f4; text-align: left; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.note { color: #444; }
"""


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def shear_columns(names: tuple[str, ...]) -> tuple[int, int]:
    names = tuple(names)
    if "g1" not in names or "g2" not in names:
        raise ValueError(f"feature names must include g1 and g2, got {names!r}")
    return names.index("g1"), names.index("g2")


def abs_g(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return np.hypot(
        np.asarray(g1, dtype=np.float64),
        np.asarray(g2, dtype=np.float64),
    )


def shrinkage_along_truth(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Fractional residual along true shear: (ĝ-g)·g / |g|². Negative is shrinkage."""

    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if estimate.shape != truth.shape or estimate.ndim != 2 or estimate.shape[-1] != 2:
        raise ValueError("estimate and truth must have shape (N, 2)")
    denom = np.sum(truth * truth, axis=1)
    numer = np.sum((estimate - truth) * truth, axis=1)
    out = np.full(len(truth), np.nan, dtype=np.float64)
    np.divide(numer, denom, out=out, where=denom > 0.0)
    return out


def abs_g_split_masks(
    truth_g: np.ndarray,
    *,
    split: float = ABS_G_SPLIT,
) -> dict[str, np.ndarray]:
    """Inner |g| < split, outer |g| > split. Equality is in the full catalog only."""

    truth_g = np.asarray(truth_g, dtype=np.float64)
    if truth_g.ndim != 2 or truth_g.shape[-1] != 2:
        raise ValueError("truth_g must have shape (N, 2)")
    amplitude = abs_g(truth_g[:, 0], truth_g[:, 1])
    finite = np.isfinite(amplitude)
    return {
        "full": finite,
        "inner": finite & (amplitude < float(split)),
        "outer": finite & (amplitude > float(split)),
    }


def interval_coverage(
    truth: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float = NOMINAL_COVERAGE,
) -> dict[str, float]:
    """Equal-mass fraction of truth inside [lower, upper], with binomial SE."""

    truth = np.asarray(truth, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if truth.shape != lower.shape or truth.shape != upper.shape:
        raise ValueError("truth, lower, and upper must have the same shape")
    finite = np.isfinite(truth) & np.isfinite(lower) & np.isfinite(upper)
    n = int(np.count_nonzero(finite))
    if n == 0:
        return {
            "coverage": float("nan"),
            "coverage_se": float("nan"),
            "delta": float("nan"),
            "n": 0,
        }
    inside = (truth[finite] >= lower[finite]) & (truth[finite] <= upper[finite])
    coverage = float(np.mean(inside))
    se = float(np.sqrt(coverage * (1.0 - coverage) / n))
    return {
        "coverage": coverage,
        "coverage_se": se,
        "delta": coverage - float(nominal),
        "n": n,
    }


def coverage_takeaway(
    rows: list[dict],
    *,
    posterior: str = "proposal",
    nominal: float = NOMINAL_COVERAGE,
) -> str:
    """One-sentence reading of inner vs outer 16–84% coverage."""

    def mean_coverage(slice_name: str) -> float:
        match = [
            row
            for row in rows
            if row.get("posterior") == posterior and row.get("slice") == slice_name
        ]
        if not match:
            return float("nan")
        return 0.5 * (float(match[0]["g1_coverage"]) + float(match[0]["g2_coverage"]))

    inner = mean_coverage("inner")
    outer = mean_coverage("outer")
    if not np.isfinite(inner) or not np.isfinite(outer):
        return "The inner/outer coverage split did not produce a finite fraction."
    low = float(nominal) - 0.10
    high = float(nominal) + 0.05
    near = 0.05
    if inner < low and outer > high:
        return (
            "The inner half sits well below 68% while the outer half overcovers. "
            "The global interval can look honest because the two halves average."
        )
    if inner > high + 0.05 and outer < float(nominal) - 0.03:
        return (
            "The inner half overcovers while the outer half sits below 68%. "
            "The global interval is an average of a too-wide interior and a "
            "slightly too-narrow outer half."
        )
    if abs(inner - float(nominal)) < near and abs(outer - float(nominal)) < near:
        return (
            "Both halves sit near 68%. The overconfident interior seen on the "
            "wider-shear network does not appear as an inner/outer split here."
        )
    return (
        "The two halves do not separate cleanly. Inner coverage is "
        f"{100 * inner:.1f}% and outer coverage is {100 * outer:.1f}%. "
        "Look at the table before calling the posterior locally calibrated."
    )


def eligible_mask(truth: np.ndarray, *, abs_g_min: float = ABS_G_MIN) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.float64)
    if truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth must have shape (N, 2)")
    amplitude = abs_g(truth[:, 0], truth[:, 1])
    return np.isfinite(amplitude) & (amplitude > float(abs_g_min))


def shrinkage_pool_mask(
    shrinkage: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    shrinkage = np.asarray(shrinkage, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if shrinkage.shape != eligible.shape:
        raise ValueError("shrinkage and eligible must match")
    values = shrinkage[eligible]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("no finite shrinkage values in the eligible set")
    median = float(np.median(finite))
    return eligible & np.isfinite(shrinkage) & (shrinkage <= median)


def draw_selected_indices(
    pool: np.ndarray,
    n: int = N_SELECTED,
    *,
    seed: int = 42,
) -> np.ndarray:
    pool = np.unique(np.asarray(pool, dtype=np.int64))
    if n <= 0:
        raise ValueError("n must be positive")
    if pool.size < n:
        raise ValueError(f"pool has {pool.size} galaxies; need {n}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n, replace=False)
    return np.sort(chosen)


def select_large_shrinkage(
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    n: int = N_SELECTED,
    abs_g_min: float = ABS_G_MIN,
    seed: int = 42,
) -> dict:
    """Pick n galaxies from the more-shrunk half of |g| > abs_g_min."""

    shrinkage = shrinkage_along_truth(estimate, truth)
    eligible = eligible_mask(truth, abs_g_min=abs_g_min)
    pool_mask = shrinkage_pool_mask(shrinkage, eligible)
    pool = np.flatnonzero(pool_mask)
    selected = draw_selected_indices(pool, n, seed=seed)
    return {
        "shrinkage": shrinkage,
        "eligible": eligible,
        "pool": pool_mask,
        "selected": selected,
        "median_eligible_shrinkage": float(
            np.median(shrinkage[eligible][np.isfinite(shrinkage[eligible])])
        ),
        "n_eligible": int(np.count_nonzero(eligible)),
        "n_pool": int(pool.size),
        "n_selected": int(selected.size),
        "abs_g_min": float(abs_g_min),
        "seed": int(seed),
    }


def galaxy_shape_variance(var_g1: np.ndarray, var_g2: np.ndarray) -> np.ndarray:
    var_g1 = np.asarray(var_g1, dtype=np.float64)
    var_g2 = np.asarray(var_g2, dtype=np.float64)
    if var_g1.shape != var_g2.shape:
        raise ValueError("g1 and g2 variances must match")
    return 0.5 * (var_g1 + var_g2)


def inverse_variance_weights(variance: np.ndarray) -> np.ndarray:
    variance = np.asarray(variance, dtype=np.float64)
    weights = np.zeros_like(variance)
    np.divide(1.0, variance, out=weights, where=np.isfinite(variance) & (variance > 0.0))
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != values.shape[: weights.ndim]:
        raise ValueError("weights must match the leading axes of values")
    while weights.ndim < values.ndim:
        weights = weights[..., None]
    finite = np.isfinite(values) & (weights > 0.0)
    safe_values = np.where(finite, values, 0.0)
    safe_weights = np.where(finite, weights, 0.0)
    numer = np.sum(safe_weights * safe_values, axis=axis)
    denom = np.sum(safe_weights, axis=axis)
    out = np.full(numer.shape, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=out, where=denom > 0.0)
    return out


def prefix_ivw(
    means: np.ndarray,
    variances: np.ndarray,
    prefixes: tuple[int, ...] = PREFIXES,
) -> np.ndarray:
    """IVW of realization Means. means (G, R, 2), variances (G, R) -> (G, P, 2)."""

    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    if means.ndim != 3 or means.shape[-1] != 2:
        raise ValueError("means must have shape (n_galaxies, n_realizations, 2)")
    n_gal, n_real, _ = means.shape
    if variances.shape != (n_gal, n_real):
        raise ValueError("variances must have shape (n_galaxies, n_realizations)")
    weights = inverse_variance_weights(variances)
    out = np.full((n_gal, len(prefixes), 2), np.nan, dtype=np.float64)
    for index, count in enumerate(prefixes):
        if count < 1 or count > n_real:
            raise ValueError(f"prefix {count} is outside 1..{n_real}")
        out[:, index] = weighted_mean(
            means[:, :count], weights[:, :count], axis=1
        )
    return out


def prefix_mean_of_means(
    means: np.ndarray,
    prefixes: tuple[int, ...] = PREFIXES,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    if means.ndim != 3 or means.shape[-1] != 2:
        raise ValueError("means must have shape (n_galaxies, n_realizations, 2)")
    n_gal, n_real, _ = means.shape
    out = np.full((n_gal, len(prefixes), 2), np.nan, dtype=np.float64)
    for index, count in enumerate(prefixes):
        if count < 1 or count > n_real:
            raise ValueError(f"prefix {count} is outside 1..{n_real}")
        out[:, index] = np.mean(means[:, :count], axis=1)
    return out


def stacked_log_prob(realization_log_probs: np.ndarray) -> np.ndarray:
    scores = np.asarray(realization_log_probs, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("realization log-probs must have shape (n_realizations, n_candidates)")
    return np.sum(scores, axis=0)


def map_index(stacked: np.ndarray) -> int:
    stacked = np.asarray(stacked, dtype=np.float64)
    if stacked.ndim != 1 or stacked.size == 0:
        raise ValueError("stacked scores must be a non-empty vector")
    finite = np.isfinite(stacked)
    if not np.any(finite):
        raise ValueError("stacked scores are all non-finite")
    return int(np.nanargmax(stacked))


def prefix_stack_map(
    bank: np.ndarray,
    log_probs: np.ndarray,
    prefixes: tuple[int, ...] = PREFIXES,
    *,
    proposal_per_realization: int = PROPOSAL_PER_REALIZATION,
) -> np.ndarray:
    """MAP over a concat-per-realization bank. bank (R*S, D), log_probs (R, R*S).

    Prefix N uses the first N realizations as both observations and proposals.
    """

    bank = np.asarray(bank, dtype=np.float64)
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if bank.ndim != 2:
        raise ValueError("bank must have shape (n_candidates, n_features)")
    n_real_expected, n_candidates = log_probs.shape
    if bank.shape[0] != n_candidates:
        raise ValueError("bank and log_probs candidate counts differ")
    if n_candidates != n_real_expected * proposal_per_realization:
        raise ValueError(
            "bank must be proposal_per_realization samples from each realization"
        )
    n_features = bank.shape[1]
    out = np.full((len(prefixes), n_features), np.nan, dtype=np.float64)
    for index, count in enumerate(prefixes):
        if count < 1 or count > n_real_expected:
            raise ValueError(f"prefix {count} is outside 1..{n_real_expected}")
        stop = count * proposal_per_realization
        stacked = stacked_log_prob(log_probs[:count, :stop])
        out[index] = bank[map_index(stacked)]
    return out


def linear_slope(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if truth.shape != prediction.shape or truth.size < 2 or np.allclose(truth, truth[0]):
        return float("nan"), float("nan")
    design = np.column_stack((truth, np.ones_like(truth)))
    slope, intercept = np.linalg.lstsq(design, prediction, rcond=None)[0]
    return float(slope), float(intercept)


def residual_calibration(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """ĝ = (1+m) g + c, reported as multiplicative m and additive c."""

    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    slope, intercept = linear_slope(truth, estimate)
    residual = estimate - truth
    return {
        "m": slope - 1.0,
        "c": intercept,
        "mean_residual": float(np.mean(residual)) if residual.size else float("nan"),
        "median_abs_residual": float(np.median(np.abs(residual)))
        if residual.size
        else float("nan"),
        "n": int(residual.size),
    }


def vector_median_abs_residual(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[-1] != 2:
        raise ValueError("truth and estimate must have shape (N, 2)")
    return float(np.median(abs_g(estimate[:, 0] - truth[:, 0], estimate[:, 1] - truth[:, 1])))


def noise_seeds(seed: int, realization: int, index: int) -> tuple[int, int]:
    image_seed = int(seed) + NOISE_STRIDE * int(realization) + int(index)
    return image_seed, image_seed + SPEC_SEED_OFFSET


def component_table(
    truth_g: np.ndarray,
    estimates: dict[str, np.ndarray],
    prefixes: tuple[int, ...] = PREFIXES,
) -> list[dict]:
    """Per-prefix, per-estimator, per-component calibration rows."""

    truth_g = np.asarray(truth_g, dtype=np.float64)
    rows = []
    for name, values in estimates.items():
        stacked = np.asarray(values, dtype=np.float64)
        if stacked.shape != (len(truth_g), len(prefixes), 2):
            raise ValueError(f"{name} must have shape (n, n_prefixes, 2)")
        for prefix_index, count in enumerate(prefixes):
            hat = stacked[:, prefix_index]
            row = {
                "estimator": name,
                "n_realizations": int(count),
                "median_abs_residual": vector_median_abs_residual(truth_g, hat),
            }
            for axis, label in enumerate(("g1", "g2")):
                metrics = residual_calibration(truth_g[:, axis], hat[:, axis])
                row[f"{label}_m"] = metrics["m"]
                row[f"{label}_c"] = metrics["c"]
            rows.append(row)
    return rows


def takeaway(rows: list[dict], *, n_realizations: int = N_REALIZATIONS) -> str:
    def mean_m(estimator: str) -> float:
        match = [
            row
            for row in rows
            if row["estimator"] == estimator and row["n_realizations"] == n_realizations
        ]
        if not match:
            return float("nan")
        return 0.5 * (match[0]["g1_m"] + match[0]["g2_m"])

    stack_m = mean_m("stack_map")
    ivw_m = mean_m("ivw")
    if not np.isfinite(stack_m) or not np.isfinite(ivw_m):
        return "The stacked estimators did not produce a finite slope."
    if abs(stack_m) < 0.15 and abs(ivw_m) > 0.25:
        return (
            "The shared-parameter stack sits near the true shear, while the "
            "inverse-variance mean of Means stays pulled toward zero. The "
            "catalog slope is then a property of the Mean as a point estimator, "
            "not a wrong likelihood."
        )
    if abs(stack_m) > 0.25 and abs(ivw_m) > 0.25:
        return (
            "Both the shared-parameter stack and the inverse-variance mean of "
            "Means stay pulled toward zero. Combining many noises of the same "
            "galaxy does not recover the true shear, so the network's likelihood "
            "is miscalibrated, not merely shrunk by a wide posterior Mean."
        )
    return (
        "The two combination rules do not separate cleanly. The stack slope "
        f"is {stack_m:.2f} and the inverse-variance Mean slope is {ivw_m:.2f}. "
        "That is a mixed verdict: look at the versus-N residual curve before "
        "blaming the Mean or the likelihood."
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def galaxies_npz(report_dir: Path) -> Path:
    return Path(report_dir) / "galaxies.npz"


def dumps_meta(meta: dict) -> np.ndarray:
    return np.asarray(json.dumps(json_safe(meta)))


def loads_meta(raw) -> dict:
    text = raw.item() if getattr(raw, "shape", None) == () else raw.tolist()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return json.loads(str(text))
