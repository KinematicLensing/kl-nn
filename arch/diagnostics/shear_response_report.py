#!/usr/bin/env python3
"""Measure a held-out matched finite-shear response from current caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

from cache_contract import (
    CURRENT_FEATURE_NAMES,
    load_cache_partitions,
    load_partitioned_array,
)


SHEAR_STENCIL = np.asarray(
    [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
)
STATE_ORDER = ("zero", "g1_plus", "g1_minus", "g2_plus", "g2_minus")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--posterior-source",
        choices=("proposal", "tf_target"),
        required=True,
        help="Select the population and its matching posterior summaries.",
    )
    parser.add_argument("--estimator", choices=("map", "mean"), default="mean")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=31415)
    return parser.parse_args(argv)



def normalize_log_weights(log_weight: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weight, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("population log ratios must be a finite vector")
    shifted = values - np.max(values)
    weight = np.exp(shifted)
    return weight / np.sum(weight)


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if values.shape[0] != len(weight):
        raise ValueError("one ensemble weight is required per base galaxy")
    weight = weight / np.sum(weight)
    return np.tensordot(weight, values, axes=(0, 0))


def weighted_mean_se(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    weight = weight / np.sum(weight)
    mean = weighted_mean(values, weight)
    ess = 1.0 / np.sum(weight**2)
    variance = np.tensordot(
        weight, np.square(values - mean), axes=(0, 0)
    )
    return np.sqrt(variance / max(ess - 1.0, 1.0))


def summarize_importance_sampling(
    manifest: pd.DataFrame,
    base_weight: np.ndarray,
    posterior_ess: np.ndarray,
    posterior_ess_fraction: np.ndarray,
    posterior_max_weight: np.ndarray,
) -> dict:
    """Summarize candidate and population IS health over matched states."""

    ordered = manifest.sort_values("ID").reset_index(drop=True)
    identifiers = pd.to_numeric(ordered["ID"], errors="raise").to_numpy()
    if not np.array_equal(identifiers, np.arange(len(ordered))):
        raise ValueError("manifest IDs must match cache row order for IS diagnostics")
    base_ids = pd.to_numeric(ordered["base_id"], errors="raise").to_numpy()
    if not np.all(base_ids == np.floor(base_ids)):
        raise ValueError("manifest base_id must be integral for IS diagnostics")
    base_ids = base_ids.astype(np.int64)
    base_weight = np.asarray(base_weight, dtype=np.float64)
    if (
        base_weight.ndim != 1
        or len(base_weight) == 0
        or np.any(base_ids < 0)
        or np.any(base_ids >= len(base_weight))
        or not np.all(np.isfinite(base_weight))
        or np.any(base_weight < 0.0)
        or np.sum(base_weight) <= 0.0
    ):
        raise ValueError("invalid base-galaxy population weights")
    base_weight = base_weight / np.sum(base_weight)
    row_weight = base_weight[base_ids]
    row_weight = row_weight / np.sum(row_weight)

    def summarize(name: str, values: np.ndarray, *, upper: float | None = None):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(ordered),) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite with one value per cache row")
        if np.any(values <= 0.0) or (upper is not None and np.any(values > upper)):
            raise ValueError(f"{name} is outside its valid range")
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_weight = row_weight[order]
        coordinate = np.cumsum(sorted_weight) - 0.5 * sorted_weight
        quantiles = np.interp(
            (0.16, 0.5, 0.84),
            coordinate,
            sorted_values,
            left=sorted_values[0],
            right=sorted_values[-1],
        )
        return {
            "minimum": float(np.min(values)),
            "population_weighted_mean": float(np.sum(row_weight * values)),
            "population_weighted_16th": float(quantiles[0]),
            "population_weighted_median": float(quantiles[1]),
            "population_weighted_84th": float(quantiles[2]),
            "maximum": float(np.max(values)),
        }

    return {
        "population_effective_sample_size": float(
            1.0 / np.sum(base_weight**2)
        ),
        "matched_base_galaxies": int(len(base_weight)),
        "posterior_rows": int(len(ordered)),
        "posterior_candidate_ess": summarize(
            "posterior_tf_ess", posterior_ess
        ),
        "posterior_candidate_ess_fraction": summarize(
            "posterior_tf_ess_fraction", posterior_ess_fraction, upper=1.0
        ),
        "posterior_candidate_max_weight": summarize(
            "posterior_tf_max_weight", posterior_max_weight, upper=1.0
        ),
    }


def validate_matched_truth_cube(
    truth_cube: np.ndarray, rmag_cube: np.ndarray
) -> float:
    """Validate the complete five-state finite-shear truth contract."""

    truth_cube = np.asarray(truth_cube, dtype=np.float64)
    rmag_cube = np.asarray(rmag_cube, dtype=np.float64)
    expected_truth_shape = (
        truth_cube.shape[0],
        len(STATE_ORDER),
        len(CURRENT_FEATURE_NAMES),
    )
    if truth_cube.shape != expected_truth_shape:
        raise ValueError(
            f"matched truth cube has shape {truth_cube.shape}; "
            f"expected {expected_truth_shape}"
        )
    if rmag_cube.shape != truth_cube.shape[:2]:
        raise ValueError("matched rmag_true cube shape differs from truth groups")
    if not np.all(np.isfinite(truth_cube)) or not np.all(np.isfinite(rmag_cube)):
        raise ValueError("matched truth and rmag_true values must be finite")

    nuisance = truth_cube[:, :, 2:8]
    nuisance_equal = np.all(
        np.isclose(nuisance, nuisance[:, :1], rtol=1e-6, atol=1e-7),
        axis=(1, 2),
    )
    if not np.all(nuisance_equal):
        failing = np.flatnonzero(~nuisance_equal).tolist()
        names = CURRENT_FEATURE_NAMES[2:8]
        raise ValueError(
            f"non-shear nuisance truths {names!r} differ within five-state groups "
            f"at base rows {failing}"
        )

    halpha_index = CURRENT_FEATURE_NAMES.index("halpha_flux_true")
    halpha = truth_cube[:, :, halpha_index]
    halpha_equal = np.all(
        np.isclose(halpha, halpha[:, :1], rtol=2e-6, atol=0.0), axis=1
    )
    if not np.all(halpha_equal):
        failing = np.flatnonzero(~halpha_equal).tolist()
        raise ValueError(
            "halpha_flux_true differs within five-state groups at base rows "
            f"{failing}"
        )
    rmag_equal = np.all(
        np.isclose(rmag_cube, rmag_cube[:, :1], rtol=0.0, atol=1e-4), axis=1
    )
    if not np.all(rmag_equal):
        failing = np.flatnonzero(~rmag_equal).tolist()
        raise ValueError(
            f"rmag_true differs within five-state groups at base rows {failing}"
        )

    delta = truth_cube[:, STATE_ORDER.index("g1_plus"), 0]
    expected_shear = delta[:, None, None] * SHEAR_STENCIL[None, :, :]
    shear_equal = np.all(
        np.isclose(
            truth_cube[:, :, :2], expected_shear, rtol=1e-6, atol=1e-7
        ),
        axis=(1, 2),
    )
    valid_delta = np.isfinite(delta) & (delta > 0.0)
    if not np.all(shear_equal & valid_delta):
        failing = np.flatnonzero(~(shear_equal & valid_delta)).tolist()
        raise ValueError(
            "cached truths do not follow the expected zero/g1+/g1-/g2+/g2- "
            f"shear stencil at base rows {failing}"
        )
    if not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-7):
        raise ValueError("matched truths do not have one common positive shear step")
    return float(delta[0])


def build_matched_cubes(
    manifest: pd.DataFrame,
    truth: np.ndarray,
    estimate: np.ndarray,
    rmag_true: np.ndarray,
    population_log_ratio: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    required_columns = {"ID", "base_id", "state", "g1", "g2"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError(
            f"manifest must contain {sorted(required_columns)}"
        )
    manifest = manifest.sort_values("ID").reset_index(drop=True)
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    rmag_true = np.asarray(rmag_true, dtype=np.float64)
    if population_log_ratio is not None:
        population_log_ratio = np.asarray(
            population_log_ratio, dtype=np.float64
        )
    nrows = len(manifest)
    if truth.shape != (nrows, len(CURRENT_FEATURE_NAMES)):
        raise ValueError(
            f"truth must have shape {(nrows, len(CURRENT_FEATURE_NAMES))}, "
            f"got {truth.shape}"
        )
    if estimate.ndim != 2 or estimate.shape[0] != nrows or estimate.shape[1] < 2:
        raise ValueError("estimate must contain at least g1 and g2 for every row")
    if rmag_true.shape != (nrows,):
        raise ValueError("rmag_true must contain one value per cache row")
    if not np.all(np.isfinite(estimate[:, :2])):
        raise ValueError("shear estimates must be finite")
    if population_log_ratio is not None and population_log_ratio.shape != (len(truth),):
        raise ValueError("population log-ratio length differs from cache")

    try:
        identifiers = pd.to_numeric(manifest["ID"], errors="raise").to_numpy(
            dtype=np.float64
        )
        base_column_numeric = pd.to_numeric(
            manifest["base_id"], errors="raise"
        ).to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest ID and base_id must be integers") from exc
    if (
        not np.array_equal(identifiers, np.arange(nrows, dtype=np.float64))
        or not np.all(np.isfinite(base_column_numeric))
        or not np.all(base_column_numeric == np.floor(base_column_numeric))
    ):
        raise ValueError(
            "manifest IDs must be contiguous cache row indices and base_id must be integer"
        )
    base_column = base_column_numeric.astype(np.int64)

    try:
        manifest_shear = manifest[["g1", "g2"]].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest g1 and g2 must be numeric") from exc
    if not np.all(np.isfinite(manifest_shear)) or not np.allclose(
        manifest_shear, truth[:, :2], rtol=1e-6, atol=1e-7
    ):
        raise ValueError("manifest g1/g2 do not match cached shear truth rows")

    state_code = {name: index for index, name in enumerate(STATE_ORDER)}
    unknown = sorted(set(manifest["state"]) - set(state_code))
    if unknown:
        raise ValueError(f"Unknown matched states: {unknown}")
    base_ids = np.unique(base_column)
    if not np.array_equal(base_ids, np.arange(len(base_ids), dtype=np.int64)):
        raise ValueError("manifest base_id values must be contiguous from zero")
    estimate_cube = np.empty((len(base_ids), len(STATE_ORDER), 2), dtype=np.float64)
    truth_cube = np.empty(
        (len(base_ids), len(STATE_ORDER), len(CURRENT_FEATURE_NAMES)),
        dtype=np.float64,
    )
    rmag_cube = np.empty((len(base_ids), len(STATE_ORDER)), dtype=np.float64)
    ratio_cube = (
        np.empty((len(base_ids), len(STATE_ORDER)), dtype=np.float64)
        if population_log_ratio is not None
        else None
    )
    state_column = manifest["state"].map(state_code).to_numpy()
    for row, base_id in enumerate(base_ids):
        for state_name, state_index in state_code.items():
            where = np.flatnonzero(
                (base_column == base_id) & (state_column == state_index)
            )
            if len(where) != 1:
                raise ValueError(
                    f"Expected one {state_name} row for base {base_id}, found {len(where)}"
                )
            cache_index = where[0]
            estimate_cube[row, state_index] = estimate[cache_index, :2]
            truth_cube[row, state_index] = truth[cache_index]
            rmag_cube[row, state_index] = rmag_true[cache_index]
            if ratio_cube is not None:
                ratio_cube[row, state_index] = population_log_ratio[cache_index]
    validate_matched_truth_cube(truth_cube, rmag_cube)

    base_log_ratio = None
    if ratio_cube is not None:
        if not np.all(ratio_cube == ratio_cube[:, :1]):
            failing = np.flatnonzero(np.any(ratio_cube != ratio_cube[:, :1], axis=1))
            raise ValueError(
                "TF population log ratio must be identical within every five-state "
                f"matched group; failures at base rows {failing.tolist()}"
            )
        base_log_ratio = ratio_cube[:, 0]
    return estimate_cube, truth_cube, base_log_ratio, base_ids


def analyze_response(
    estimate_cube: np.ndarray,
    truth_cube: np.ndarray,
    base_weight: np.ndarray,
    *,
    calibration_fraction: float,
    seed: int,
) -> dict:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie strictly between 0 and 1")
    nbase = len(estimate_cube)
    if nbase < 4:
        raise ValueError("at least four matched base galaxies are required")
    code = {name: index for index, name in enumerate(STATE_ORDER)}
    delta_g1 = 0.5 * (
        truth_cube[:, code["g1_plus"], 0]
        - truth_cube[:, code["g1_minus"], 0]
    )
    delta_g2 = 0.5 * (
        truth_cube[:, code["g2_plus"], 1]
        - truth_cube[:, code["g2_minus"], 1]
    )
    if (
        np.any(delta_g1 <= 0.0)
        or np.any(delta_g2 <= 0.0)
        or not np.allclose(delta_g1, delta_g1[0], rtol=1e-6, atol=1e-7)
        or not np.allclose(delta_g2, delta_g2[0], rtol=1e-6, atol=1e-7)
        or not np.isclose(delta_g1[0], delta_g2[0], rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("matched truths do not have one common positive shear step")
    delta = float(delta_g1[0])
    response = np.empty((nbase, 2, 2), dtype=np.float64)
    response[:, :, 0] = (
        estimate_cube[:, code["g1_plus"]]
        - estimate_cube[:, code["g1_minus"]]
    ) / (2.0 * delta)
    response[:, :, 1] = (
        estimate_cube[:, code["g2_plus"]]
        - estimate_cube[:, code["g2_minus"]]
    ) / (2.0 * delta)

    rng = np.random.default_rng(seed)
    order = rng.permutation(nbase)
    split = int(round(calibration_fraction * nbase))
    split = min(max(split, 2), nbase - 2)
    calibration, holdout = order[:split], order[split:]
    calibration_weight = base_weight[calibration]
    calibration_weight = calibration_weight / np.sum(calibration_weight)
    holdout_weight = base_weight[holdout]
    holdout_weight = holdout_weight / np.sum(holdout_weight)

    calibration_response = weighted_mean(response[calibration], calibration_weight)
    calibration_additive = weighted_mean(
        estimate_cube[calibration, code["zero"]], calibration_weight
    )
    inverse_response = np.linalg.inv(calibration_response)
    corrected = np.einsum(
        "ij,bsj->bsi", inverse_response, estimate_cube - calibration_additive
    )
    raw_holdout_response = weighted_mean(response[holdout], holdout_weight)
    corrected_response = inverse_response @ raw_holdout_response
    holdout_additive = weighted_mean(
        corrected[holdout, code["zero"]], holdout_weight
    )
    additive_se = weighted_mean_se(
        corrected[holdout, code["zero"]], holdout_weight
    )
    return {
        "nbase": nbase,
        "n_calibration": len(calibration),
        "n_holdout": len(holdout),
        "calibration_weight_ess": float(1.0 / np.sum(calibration_weight**2)),
        "holdout_weight_ess": float(1.0 / np.sum(holdout_weight**2)),
        "delta_g": delta,
        "calibration_additive": calibration_additive.tolist(),
        "calibration_response": calibration_response.tolist(),
        "raw_holdout_response": raw_holdout_response.tolist(),
        "corrected_holdout_response": corrected_response.tolist(),
        "corrected_holdout_additive": holdout_additive.tolist(),
        "corrected_holdout_additive_se": additive_se.tolist(),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest = pd.read_csv(args.manifest)
    cache_partitions = load_cache_partitions(args.cache_root)
    if cache_partitions.observation_provenance["matched_group_size"] != len(
        STATE_ORDER
    ):
        raise ValueError(
            "shear response requires cache noise shared within five-state groups"
        )
    truth = np.asarray(load_partitioned_array(cache_partitions, "truth"))
    rmag_true = np.asarray(
        load_partitioned_array(cache_partitions, "rmag_true"), dtype=np.float64
    )
    prefix = "proposal" if args.posterior_source == "proposal" else "tf_target"
    if args.estimator == "map":
        estimate = np.asarray(
            load_partitioned_array(
                cache_partitions, f"{prefix}_map_estimates"
            )
        )
    else:
        summary = np.asarray(
            load_partitioned_array(
                cache_partitions, f"{prefix}_mean_estimates"
            )
        )
        if summary.ndim != 3 or summary.shape[1] != 3:
            raise ValueError("mean summary cache must have shape (galaxy, 3, feature)")
        estimate = summary[:, 1]
    population_log_ratio = None
    posterior_importance_arrays = None
    if args.posterior_source == "tf_target":
        population_log_ratio = np.asarray(
            load_partitioned_array(
                cache_partitions, "population_tf_log_ratio"
            ),
            dtype=np.float64,
        )
        posterior_importance_arrays = tuple(
            np.asarray(load_partitioned_array(cache_partitions, name), dtype=np.float64)
            for name in (
                "posterior_tf_ess",
                "posterior_tf_ess_fraction",
                "posterior_tf_max_weight",
            )
        )
    cube, true_cube, base_log_ratio, base_ids = build_matched_cubes(
        manifest, truth, estimate, rmag_true, population_log_ratio
    )
    base_weight = (
        np.full(len(base_ids), 1.0 / len(base_ids), dtype=np.float64)
        if base_log_ratio is None
        else normalize_log_weights(base_log_ratio)
    )
    result = analyze_response(
        cube,
        true_cube,
        base_weight,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    result.update(
        {
            "posterior_source": args.posterior_source,
            "estimator": args.estimator,
            "population_weighting": (
                "equal across matched base galaxies"
                if args.posterior_source == "proposal"
                else "globally normalized TF log ratios across matched base galaxies"
            ),
            "targets": {"abs_c": 1e-4, "abs_m": 1e-2},
            "note": "Calibration and validation base galaxies are disjoint.",
        }
    )
    result["population_effective_sample_size"] = float(
        1.0 / np.sum(base_weight**2)
    )
    if posterior_importance_arrays is not None:
        result["tf_importance_sampling"] = summarize_importance_sampling(
            manifest,
            base_weight,
            *posterior_importance_arrays,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    if args.output.suffix.lower() == ".html":
        args.output.write_text(
            "<!doctype html><meta charset=\"utf-8\"><title>Shear response pilot</title>"
            "<style>body{font-family:system-ui;max-width:900px;margin:2rem auto}"
            "pre{background:#f5f5f5;padding:1rem;overflow:auto}</style>"
            "<h1>Matched finite-shear response pilot</h1>"
            "<p>The selected posterior and galaxy population are stated explicitly. "
            "Calibration and holdout base-galaxy sets are disjoint; response uses "
            "central finite differences from matched simulations.</p>"
            f"<pre>{payload}</pre>",
            encoding="utf-8",
        )
    else:
        args.output.write_text(payload, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
