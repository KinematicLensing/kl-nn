#!/usr/bin/env python3
"""Cache posterior products for standard or TF-conformed test analyses.

The sampling adapter in :mod:`train` constructs the canonical original/R90
ensemble and inverse-aligns every rotated parameter row. This script adds no
second observation, resampling, or alternate network path. ``--test-set``
keeps a compact candidate bank (physical shear plus normalized TF log weights)
and TF-weighted Mean summaries. It does not request density scores, compute
MAP, or apply a second truth-population TF weight.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pyxis.torch as pxt
import torch
from torch.utils.data import Subset

try:
    from . import config
    from .cache_contract import (
        CACHE_SCHEMA,
        COMBINED_TEST_SET_CACHE_SCHEMA,
        STANDARD_ANALYSIS_MODE,
        TEST_SET_ANALYSIS_MODE,
        TEST_SET_CACHE_SCHEMA,
    )
    from .inclination_prior import (
        InclinationPrior,
        isotropic_inclination_log_prior_ratio,
    )
    from .model_registry import load_model_config
    from .networks import KLNPE
    from .tf_prior import (
        TFPrior,
        population_log_importance_ratio,
        posterior_importance_from_log_ratio,
        posterior_importance_weights,
        tf_log_prior_ratio,
    )
    from .train import load_model, sample_density, seed_everything
    from .utils import (
        denormalization_logabsdet,
        denormalize,
        resolve_feature_index,
    )
except ImportError:  # Direct execution from arch/.
    import config
    from cache_contract import (
        CACHE_SCHEMA,
        COMBINED_TEST_SET_CACHE_SCHEMA,
        STANDARD_ANALYSIS_MODE,
        TEST_SET_ANALYSIS_MODE,
        TEST_SET_CACHE_SCHEMA,
    )
    from inclination_prior import (
        InclinationPrior,
        isotropic_inclination_log_prior_ratio,
    )
    from model_registry import load_model_config
    from networks import KLNPE
    from tf_prior import (
        TFPrior,
        population_log_importance_ratio,
        posterior_importance_from_log_ratio,
        posterior_importance_weights,
        tf_log_prior_ratio,
    )
    from train import load_model, sample_density, seed_everything
    from utils import denormalization_logabsdet, denormalize, resolve_feature_index


DEFAULT_SHARED_ROOT = Path("/ocean/projects/phy250048p/shared")

STANDARD_CACHE_ARRAY_TYPES = (
    "sample",
    "base_log_prob",
    "posterior_tf_log_ratio",
    "posterior_tf_log_weight",
    "posterior_tf_weight",
    "posterior_tf_ess",
    "posterior_tf_ess_fraction",
    "posterior_tf_max_weight",
    "posterior_tf_log_mean_ratio",
    "population_tf_log_ratio",
    "truth",
    "rmag_true",
    "image_snr",
    "central_halpha_snr",
    "image_noise_sigma",
    "central_spectral_noise_sigma",
    "proposal_map_estimates",
    "proposal_mean_estimates",
    "tf_target_map_estimates",
    "tf_target_mean_estimates",
)
CACHE_ARRAY_TYPES = STANDARD_CACHE_ARRAY_TYPES
TEST_SET_CACHE_ARRAY_TYPES = (
    "shear_sample",
    "posterior_tf_log_weight",
    "posterior_tf_ess",
    "posterior_tf_ess_fraction",
    "posterior_tf_max_weight",
    "truth",
    "rmag_true",
    "image_snr",
    "central_halpha_snr",
    "image_noise_sigma",
    "central_spectral_noise_sigma",
    "proposal_mean_estimates",
    "tf_target_mean_estimates",
)
COMBINED_TEST_SET_CACHE_ARRAY_TYPES = (
    "shear_sample",
    "posterior_target_log_weight",
    "posterior_target_ess",
    "posterior_target_ess_fraction",
    "posterior_target_max_weight",
    "truth",
    "rmag_true",
    "image_snr",
    "central_halpha_snr",
    "image_noise_sigma",
    "central_spectral_noise_sigma",
    "proposal_mean_estimates",
    "target_mean_estimates",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--partition-index", type=int, required=True)
    parser.add_argument("--nparts", type=int, required=True)
    parser.add_argument("--ngals", type=int, required=True)
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model-root", type=Path, default=DEFAULT_SHARED_ROOT / "models"
    )
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_SHARED_ROOT / "datasets"
    )
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_SHARED_ROOT / "cache"
    )
    parser.add_argument("--cache-tag", default="posterior_candidates")
    parser.add_argument(
        "--test-set",
        action="store_true",
        help=(
            "Write a compact TF-conformed test cache with TF-weighted "
            "posterior Mean summaries, normalized candidate log weights, "
            "and physical shear candidates."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help=(
            "Generation manifest; test-set mode defaults to "
            "DATASET/manifest.json and requires it."
        ),
    )
    parser.add_argument(
        "--isotropic-inclination-prior",
        action="store_true",
        help=(
            "In compact test-set mode, additionally replace the uniform-sini "
            "training prior with the isotropic uniform-cosi prior. The output "
            "cache name receives a _tf_iso_inclination suffix."
        ),
    )
    parser.add_argument("--matched-group-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--tf-slope", type=float, default=-7.22)
    parser.add_argument("--tf-intercept", type=float, default=36.0)
    parser.add_argument("--tf-scatter-dex", type=float, default=0.1)
    parser.add_argument("--warn-ess-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_dataset_manifest(
    dataset_path: Path,
    data_root: Path,
    explicit_path: Path | None,
) -> Path:
    """Resolve the required generation sidecar for a compact test cache."""

    if explicit_path is None:
        return dataset_path / "manifest.json"
    return (
        explicit_path
        if explicit_path.is_absolute()
        else data_root / explicit_path
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_test_set_generation_manifest(
    path: Path,
    *,
    dataset_size: int,
    tf_prior: TFPrior,
    hlr_bounds: tuple[float, float] | list[float],
    require_isotropic_inclination: bool = False,
) -> dict:
    """Fail closed on the generation provenance asserted by test-set mode."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Test-set generation manifest not found: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Cannot read test-set generation manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("test-set generation manifest must be a JSON object")
    expected = {
        "schema": "klnn-generation-manifest-v1",
        "analysis_mode": TEST_SET_ANALYSIS_MODE,
        "population": "tf_conformed_catalog",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(
                f"generation manifest {name} must equal {value!r}"
            )
    sample_count = payload.get("sample_count")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count != dataset_size
    ):
        raise ValueError(
            "generation manifest sample_count must equal the dataset size "
            f"{dataset_size}"
        )
    for name in ("redshift", "simulation_redshift"):
        redshift = payload.get(name)
        if (
            isinstance(redshift, bool)
            or not isinstance(redshift, (int, float))
            or not math.isclose(
                float(redshift), 0.3, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(f"generation manifest {name} must be 0.3")
    for name in (
        "source_catalog",
        "catalog_sampling",
        "parameter_sampling",
        "sample_table",
    ):
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"generation manifest {name} must be an object")

    bounds = np.asarray(hlr_bounds, dtype=np.float64)
    if (
        bounds.shape != (2,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("hlr_bounds must contain two increasing finite values")
    catalog_sampling = payload["catalog_sampling"]
    eligibility = catalog_sampling.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError(
            "generation manifest catalog_sampling.eligibility must be an object"
        )
    hlr_eligibility = eligibility.get("hlr")
    expected_hlr_keys = {"finite", "minimum", "maximum", "bounds"}
    if (
        not isinstance(hlr_eligibility, dict)
        or set(hlr_eligibility) != expected_hlr_keys
    ):
        raise ValueError(
            "generation manifest catalog_sampling.eligibility.hlr must "
            "contain exactly finite, minimum, maximum, and bounds"
        )
    if hlr_eligibility.get("finite") is not True:
        raise ValueError(
            "generation manifest catalog_sampling.eligibility.hlr.finite "
            "must be true"
        )
    if hlr_eligibility.get("bounds") != "inclusive":
        raise ValueError(
            "generation manifest catalog_sampling.eligibility.hlr.bounds "
            "must be 'inclusive'"
        )
    for name, expected_bound in zip(("minimum", "maximum"), bounds):
        actual = hlr_eligibility.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual),
                float(expected_bound),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "generation manifest catalog_sampling.eligibility.hlr."
                f"{name} must equal {float(expected_bound)!r}"
            )
    sample_table = payload["sample_table"]
    for name in ("path", "sha256", "id_policy"):
        if not isinstance(sample_table.get(name), str) or not sample_table[name]:
            raise ValueError(
                f"generation manifest sample_table.{name} must be non-empty"
            )
    if len(sample_table["sha256"]) != 64 or any(
        character not in "0123456789abcdef"
        for character in sample_table["sha256"]
    ):
        raise ValueError(
            "generation manifest sample_table.sha256 must be lowercase hex"
        )
    table_row_count = sample_table.get("row_count")
    if (
        isinstance(table_row_count, bool)
        or not isinstance(table_row_count, int)
        or table_row_count != dataset_size
    ):
        raise ValueError(
            "generation manifest sample_table.row_count must equal the "
            f"dataset size {dataset_size}"
        )

    generated_tf = payload.get("tf")
    if not isinstance(generated_tf, dict):
        raise ValueError("generation manifest tf must be an object")
    expected_tf = tf_prior.to_dict()
    if set(generated_tf) != set(expected_tf):
        raise ValueError(
            "generation manifest tf must contain exactly "
            f"{sorted(expected_tf)}"
        )
    for name, value in expected_tf.items():
        actual = generated_tf.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isclose(
                float(actual), float(value), rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(
                f"generation manifest tf.{name} must equal {value!r}"
            )

    if require_isotropic_inclination:
        inclination = payload["parameter_sampling"].get("inclination")
        expected_inclination = {
            "distribution": "cosi_uniform_0_1_latin_hypercube",
            "transform": "sini=sqrt(1-cosi**2)",
        }
        if inclination != expected_inclination:
            raise ValueError(
                "isotropic inclination prior requires generation manifest "
                f"parameter_sampling.inclination={expected_inclination!r}"
            )

    embedded = dict(payload)
    embedded["path"] = str(path.resolve())
    embedded["sha256"] = _sha256(path)
    return embedded


def resolve_checkpoint(args: argparse.Namespace, model_path: str | Path) -> Path:
    if args.checkpoint is not None:
        path = args.checkpoint
    elif args.epoch is not None:
        path = (
            Path(model_path)
            / args.model_name
            / f"{args.model_name}{args.epoch}"
        )
    else:
        path = (
            Path(model_path)
            / args.model_name
            / f"{args.model_name}best"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def resolve_partition_range(
    partition_index: int, galaxies_per_partition: int, dataset_size: int
) -> tuple[int, int]:
    if partition_index < 0:
        raise ValueError("partition index must be non-negative")
    if galaxies_per_partition <= 0:
        raise ValueError("ngals must be positive")
    start = partition_index * galaxies_per_partition
    end = start + galaxies_per_partition
    if end > dataset_size:
        raise ValueError(
            f"partition [{start}, {end}) exceeds dataset size {dataset_size}"
        )
    return start, end


def validate_partition_coverage(
    nparts: int, galaxies_per_partition: int, dataset_size: int
) -> None:
    """Require cache partitions to cover the dataset exactly once."""

    if nparts <= 0 or galaxies_per_partition <= 0 or dataset_size <= 0:
        raise ValueError("nparts, ngals, and dataset size must be positive")
    covered = int(nparts) * int(galaxies_per_partition)
    if covered != int(dataset_size):
        raise ValueError(
            "cache partition coverage must equal the complete dataset: "
            f"nparts*ngals={covered}, dataset_size={dataset_size}"
        )


def partition_label(partition_index: int, nparts: int) -> str:
    if nparts <= 0 or not 0 <= partition_index < nparts:
        raise ValueError("partition index must lie in [0, nparts)")
    return f"part{partition_index}of{nparts}"


def _to_numpy(value, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype == object:
        raise ValueError(f"{name} cannot be an object array")
    return result


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray | tuple[float, ...],
    weight: np.ndarray,
) -> np.ndarray:
    """Inverse weighted ECDF for finite one-dimensional samples."""

    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if values.ndim != 1 or weight.shape != values.shape:
        raise ValueError("values and weight must be matching vectors")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("quantiles must lie in [0, 1]")
    finite = np.isfinite(values) & np.isfinite(weight) & (weight >= 0.0)
    if not np.any(finite):
        return np.full(quantiles.shape, np.nan, dtype=np.float64)
    values = values[finite]
    weight = weight[finite]
    total = np.sum(weight)
    if not np.isfinite(total) or total <= 0.0:
        return np.full(quantiles.shape, np.nan, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weight = weight[order] / total
    coordinates = np.cumsum(weight) - 0.5 * weight
    return np.interp(
        quantiles, coordinates, values, left=values[0], right=values[-1]
    )


def summarize_posterior_samples(
    samples: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """Return 16th, mean, and 84th summaries, including circular theta."""

    values = np.asarray(samples, dtype=np.float64)
    feature_names = tuple(feature_names)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("samples must have shape (draw, len(feature_names))")
    if values.shape[0] == 0:
        raise ValueError("samples must contain at least one draw")
    supplied_weight = weight is not None
    if weight is None:
        weight = np.full(values.shape[0], 1.0 / values.shape[0])
    else:
        weight = np.asarray(weight, dtype=np.float64)
        if weight.shape != (values.shape[0],):
            raise ValueError("weight must contain one value per draw")
        if (
            np.any(~np.isfinite(weight))
            or np.any(weight < 0.0)
            or np.sum(weight) <= 0.0
        ):
            raise ValueError("weight must be finite, non-negative, and non-zero")
        weight = weight / np.sum(weight)

    summary = np.empty((3, values.shape[1]), dtype=np.float64)
    for index in range(values.shape[1]):
        if supplied_weight:
            lower, upper = weighted_quantile(
                values[:, index], (0.16, 0.84), weight
            )
        else:
            lower, upper = np.percentile(values[:, index], (16.0, 84.0))
        summary[:, index] = (
            lower,
            np.sum(weight * values[:, index]),
            upper,
        )

    if "theta_int" in feature_names:
        theta_index = feature_names.index("theta_int")
        theta = values[:, theta_index]
        center = math.atan2(
            np.sum(weight * np.sin(theta)),
            np.sum(weight * np.cos(theta)),
        )
        residual = (theta - center + np.pi) % (2.0 * np.pi) - np.pi
        if supplied_weight:
            bounds = weighted_quantile(residual, (0.16, 0.84), weight)
        else:
            bounds = np.percentile(residual, (16.0, 84.0))
        circular = np.array((center + bounds[0], center, center + bounds[1]))
        summary[:, theta_index] = (
            circular + np.pi
        ) % (2.0 * np.pi) - np.pi
    return summary


def proposal_mean_summaries(
    samples: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Return equal-candidate 16th, Mean, and 84th summaries per galaxy."""

    values = np.asarray(samples)
    feature_names = tuple(feature_names)
    if values.ndim != 3 or values.shape[2] != len(feature_names):
        raise ValueError(
            "samples must have shape (galaxy, draw, len(feature_names))"
        )
    return np.stack(
        [
            summarize_posterior_samples(row, feature_names)
            for row in values
        ],
        axis=0,
    )


def target_mean_summaries(
    samples: np.ndarray,
    target_weight: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Return target-weighted 16th, Mean, and 84th summaries per galaxy."""

    values = np.asarray(samples)
    weights = np.asarray(target_weight)
    feature_names = tuple(feature_names)
    if values.ndim != 3 or values.shape[2] != len(feature_names):
        raise ValueError(
            "samples must have shape (galaxy, draw, len(feature_names))"
        )
    if weights.shape != values.shape[:2]:
        raise ValueError(
            "target_weight must have shape (galaxy, draw)"
        )
    return np.stack(
        [
            summarize_posterior_samples(row, feature_names, row_weight)
            for row, row_weight in zip(values, weights)
        ],
        axis=0,
    )


def tf_target_mean_summaries(
    samples: np.ndarray,
    tf_weight: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
) -> np.ndarray:
    """Backward-compatible name for TF-only target summaries."""

    return target_mean_summaries(samples, tf_weight, feature_names)


def posterior_summaries(
    samples: np.ndarray,
    base_log_prob: np.ndarray,
    tf_log_ratio: np.ndarray,
    tf_weight: np.ndarray,
    feature_names: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Build explicitly named proposal and TF-target point summaries."""

    samples = np.asarray(samples)
    base_log_prob = np.asarray(base_log_prob)
    tf_log_ratio = np.asarray(tf_log_ratio)
    tf_weight = np.asarray(tf_weight)
    if samples.ndim != 3:
        raise ValueError("samples must have shape (galaxy, draw, feature)")
    expected = samples.shape[:2]
    for name, value in (
        ("base_log_prob", base_log_prob),
        ("tf_log_ratio", tf_log_ratio),
        ("tf_weight", tf_weight),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")

    n_galaxies, _, n_features = samples.shape
    proposal_map = np.empty((n_galaxies, n_features), dtype=np.float64)
    target_map = np.empty_like(proposal_map)
    proposal_mean = np.empty(
        (n_galaxies, 3, n_features), dtype=np.float64
    )
    target_mean = np.empty_like(proposal_mean)
    for index in range(n_galaxies):
        proposal_map[index] = samples[
            index, np.nanargmax(base_log_prob[index])
        ]
        target_map[index] = samples[
            index, np.nanargmax(base_log_prob[index] + tf_log_ratio[index])
        ]
        proposal_mean[index] = summarize_posterior_samples(
            samples[index], feature_names
        )
        target_mean[index] = summarize_posterior_samples(
            samples[index], feature_names, tf_weight[index]
        )
    return {
        "proposal_map_estimates": proposal_map,
        "proposal_mean_estimates": proposal_mean,
        "tf_target_map_estimates": target_map,
        "tf_target_mean_estimates": target_mean,
    }


def physical_log_prob_from_normalized(
    samples_normalized,
    normalized_log_prob,
    *,
    par_ranges,
    feature_names,
    target_transforms,
):
    """Convert density scores to physical coordinates for MAP selection."""

    samples_normalized = np.asarray(samples_normalized)
    normalized_log_prob = np.asarray(normalized_log_prob)
    if normalized_log_prob.shape != samples_normalized.shape[:-1]:
        raise ValueError(
            "normalized_log_prob must match every non-feature sample dimension"
        )
    return normalized_log_prob - denormalization_logabsdet(
        samples_normalized,
        par_ranges=par_ranges,
        feature_names=feature_names,
        target_transforms=target_transforms,
    )


def _save_array(
    root: Path, name: str, label: str, value: np.ndarray
) -> str:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{label}.npy"
    np.save(path, value)
    return str(path.relative_to(root))


def _config_value(section, name):
    return section[name] if isinstance(section, dict) else getattr(section, name)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.nparts <= 0 or not 0 <= args.partition_index < args.nparts:
        raise ValueError("partition-index must lie in [0, nparts)")
    if args.nsamples <= 0 or args.nsamples % 2:
        raise ValueError(
            "nsamples must be positive and even for the R90 ensemble"
        )
    if (
        args.matched_group_size not in (1, 5)
        or args.ngals % args.matched_group_size
    ):
        raise ValueError(
            "matched-group-size must be 1 or a divisor-aligned 5"
        )
    if args.test_set and args.matched_group_size != 1:
        raise ValueError("test-set caches require matched-group-size=1")
    if args.isotropic_inclination_prior and not args.test_set:
        raise ValueError(
            "--isotropic-inclination-prior is only valid with --test-set"
        )
    if not args.test_set and args.dataset_manifest is not None:
        raise ValueError("--dataset-manifest is only valid with --test-set")
    if not 0.0 <= args.warn_ess_fraction <= 1.0:
        raise ValueError("warn-ess-fraction must lie in [0, 1]")

    artifact_root = args.model_root.parent
    model_config = load_model_config(
        args.model_name,
        configs_root=str(artifact_root / "configs"),
    )
    config.set_model_config(model_config)
    train_config = config.train
    feature_names = tuple(_config_value(train_config, "feature_names"))
    if len(feature_names) != 9:
        raise ValueError(
            f"current posterior schema requires 9 targets, got {feature_names}"
        )
    vcirc_index = resolve_feature_index(feature_names, "vcirc")
    sini_index = resolve_feature_index(feature_names, "sini")
    vcirc_bounds = config.par_ranges["vcirc"]
    tf_prior = TFPrior(
        slope=args.tf_slope,
        intercept=args.tf_intercept,
        scatter_dex=args.tf_scatter_dex,
        vcirc_min=float(vcirc_bounds[0]),
        vcirc_max=float(vcirc_bounds[1]),
    )
    sini_bounds = config.par_ranges["sini"]
    inclination_prior = InclinationPrior(
        sini_min=float(sini_bounds[0]),
        sini_max=float(sini_bounds[1]),
    )

    device = resolve_device(args.device)
    partition_seed = args.seed + 1_000_003 * args.partition_index
    seed_everything(
        partition_seed,
        deterministic=bool(_config_value(train_config, "deterministic")),
    )
    dataset_path = resolve_path(args.data_root, args.dataset)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    dataset = pxt.TorchDataset(str(dataset_path))
    validate_partition_coverage(args.nparts, args.ngals, len(dataset))
    generation_manifest = None
    if args.test_set:
        manifest_path = resolve_dataset_manifest(
            dataset_path,
            args.data_root,
            args.dataset_manifest,
        )
        generation_manifest = load_test_set_generation_manifest(
            manifest_path,
            dataset_size=len(dataset),
            tf_prior=tf_prior,
            hlr_bounds=config.par_ranges["hlr"],
            require_isotropic_inclination=(
                args.isotropic_inclination_prior
            ),
        )
    start, end = resolve_partition_range(
        args.partition_index, args.ngals, len(dataset)
    )
    subset = Subset(dataset, np.arange(start, end).tolist())
    label = partition_label(args.partition_index, args.nparts)

    checkpoint = resolve_checkpoint(args, args.model_root)
    model = load_model(
        KLNPE,
        path=str(checkpoint),
        model_name=args.model_name,
        strict=True,
        device=str(device),
        networks_root=str(artifact_root / "networks"),
    )
    # Loading instantiates the snapshotted network before restoring its state,
    # which consumes random numbers. Reset here so candidate streams depend
    # only on the documented partition seed.
    seed_everything(
        partition_seed,
        deterministic=bool(_config_value(train_config, "deterministic")),
    )
    sampled = sample_density(
        model,
        subset,
        args.nsamples,
        device=str(device),
        matched_group_size=args.matched_group_size,
        noise_seed=partition_seed,
        spectral_noise_seed=partition_seed + 101,
        return_log_prob=not args.test_set,
        return_observation_metadata=True,
    )
    if args.test_set:
        samples_normalized, metadata = sampled
        base_log_prob = None
    else:
        samples_normalized, base_log_prob, metadata = sampled
    samples_normalized = _to_numpy(samples_normalized, name="samples")
    if base_log_prob is not None:
        base_log_prob = _to_numpy(base_log_prob, name="base_log_prob")
    expected_shape = (
        args.ngals,
        args.nsamples,
        len(feature_names),
    )
    if samples_normalized.shape != expected_shape:
        raise RuntimeError(
            f"unexpected sample shape {samples_normalized.shape}; "
            f"expected {expected_shape}"
        )
    if (
        base_log_prob is not None
        and base_log_prob.shape != samples_normalized.shape[:2]
    ):
        raise RuntimeError(
            f"unexpected base-log-probability shape {base_log_prob.shape}"
        )
    if not np.all(np.isfinite(samples_normalized)):
        raise RuntimeError("base posterior returned non-finite samples")
    if base_log_prob is not None and not np.all(np.isfinite(base_log_prob)):
        raise RuntimeError("base posterior returned non-finite scores")
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "sample_density must return observation metadata"
        )

    required_metadata = (
        "truth",
        "rmag_true",
        "image_snr",
        "central_halpha_snr",
        "image_noise_sigma",
        "central_spectral_noise_sigma",
    )
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise RuntimeError(f"sample_density metadata is missing {missing}")
    truth_normalized = _to_numpy(metadata["truth"], name="truth")
    rmag_true = _to_numpy(
        metadata["rmag_true"], name="rmag_true"
    ).astype(np.float64)
    image_snr = _to_numpy(metadata["image_snr"], name="image_snr").astype(
        np.float64
    )
    central_halpha_snr = _to_numpy(
        metadata["central_halpha_snr"], name="central_halpha_snr"
    ).astype(np.float64)
    image_noise_sigma = _to_numpy(
        metadata["image_noise_sigma"], name="image_noise_sigma"
    ).astype(np.float64)
    central_spectral_noise_sigma = _to_numpy(
        metadata["central_spectral_noise_sigma"],
        name="central_spectral_noise_sigma",
    ).astype(np.float64)
    if truth_normalized.shape != (args.ngals, len(feature_names)):
        raise RuntimeError(
            f"unexpected truth shape {truth_normalized.shape}"
        )
    if (
        any(
            value.shape != (args.ngals,)
            for value in (
                rmag_true,
                image_snr,
                central_halpha_snr,
                image_noise_sigma,
                central_spectral_noise_sigma,
            )
        )
    ):
        raise RuntimeError(
            "observation metadata must contain one scalar per galaxy"
        )
    if (
        any(
            not np.all(np.isfinite(value))
            or np.any(value <= 0.0)
            for value in (
                image_snr,
                central_halpha_snr,
                image_noise_sigma,
                central_spectral_noise_sigma,
            )
        )
        or not np.all(np.isfinite(rmag_true))
    ):
        raise RuntimeError("observation metadata contains non-finite values")

    samples = denormalize(
        samples_normalized,
        par_ranges=config.par_ranges,
        feature_names=feature_names,
        target_transforms=config.TARGET_TRANSFORMS,
    ).astype(np.float32, copy=False)
    truth = denormalize(
        truth_normalized,
        par_ranges=config.par_ranges,
        feature_names=feature_names,
        target_transforms=config.TARGET_TRANSFORMS,
    ).astype(np.float32, copy=False)
    if args.test_set:
        if args.isotropic_inclination_prior:
            tf_log_ratio = tf_log_prior_ratio(
                np.asarray(samples[..., vcirc_index], dtype=np.float64),
                rmag_true[:, None],
                tf_prior,
            )
            inclination_log_ratio = isotropic_inclination_log_prior_ratio(
                np.asarray(samples[..., sini_index], dtype=np.float64),
                inclination_prior,
            )
            importance = posterior_importance_from_log_ratio(
                tf_log_ratio + inclination_log_ratio
            )
            target_summary_name = "target_mean_estimates"
        else:
            importance = posterior_importance_weights(
                samples[..., vcirc_index], rmag_true, tf_prior
            )
            target_summary_name = "tf_target_mean_estimates"
        summaries = {
            "proposal_mean_estimates": proposal_mean_summaries(
                samples, feature_names
            ),
            target_summary_name: target_mean_summaries(
                samples, importance.weight, feature_names
            ),
        }
        population_log_ratio = None
    else:
        assert base_log_prob is not None
        physical_base_log_prob = physical_log_prob_from_normalized(
            samples_normalized,
            base_log_prob,
            par_ranges=config.par_ranges,
            feature_names=feature_names,
            target_transforms=config.TARGET_TRANSFORMS,
        )
        importance = posterior_importance_weights(
            samples[..., vcirc_index], rmag_true, tf_prior
        )
        population_log_ratio = population_log_importance_ratio(
            truth[:, vcirc_index].astype(np.float64),
            rmag_true,
            tf_prior,
        )
        summaries = posterior_summaries(
            samples,
            physical_base_log_prob,
            importance.log_ratio,
            importance.weight,
            feature_names,
        )

    assert importance is not None
    low_ess = importance.effective_sample_fraction < args.warn_ess_fraction
    if np.any(low_ess):
        weighting_label = (
            "combined TF + isotropic-inclination"
            if args.isotropic_inclination_prior
            else "TF"
        )
        logging.warning(
            "%d/%d galaxies have posterior %s ESS fraction below %.3f; "
            "minimum %.5f",
            int(np.count_nonzero(low_ess)),
            len(low_ess),
            weighting_label,
            args.warn_ess_fraction,
            float(np.min(importance.effective_sample_fraction)),
        )

    dataset_name = dataset_path.name
    if args.cache_tag:
        dataset_name += f"_{args.cache_tag.strip('_')}"
    if args.isotropic_inclination_prior:
        dataset_name += "_tf_iso_inclination"
    output_root = args.cache_root / args.model_name / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    common_arrays = {
        "truth": truth,
        "rmag_true": rmag_true.astype(np.float32),
        "image_snr": image_snr.astype(np.float32),
        "central_halpha_snr": central_halpha_snr.astype(np.float32),
        "image_noise_sigma": image_noise_sigma.astype(np.float32),
        "central_spectral_noise_sigma": central_spectral_noise_sigma.astype(
            np.float32
        ),
    }
    if args.test_set:
        assert importance is not None
        shear_indices = [
            resolve_feature_index(feature_names, name)
            for name in ("g1", "g2")
        ]
        test_set_common_arrays = {
            "shear_sample": samples[..., shear_indices].astype(
                np.float32, copy=False
            ),
            **common_arrays,
            "proposal_mean_estimates": summaries[
                "proposal_mean_estimates"
            ].astype(np.float32),
        }
        if args.isotropic_inclination_prior:
            arrays = {
                **test_set_common_arrays,
                "posterior_target_log_weight": importance.log_weight.astype(
                    np.float32
                ),
                "posterior_target_ess": (
                    importance.effective_sample_size.astype(np.float64)
                ),
                "posterior_target_ess_fraction": (
                    importance.effective_sample_fraction.astype(np.float64)
                ),
                "posterior_target_max_weight": importance.max_weight.astype(
                    np.float64
                ),
                "target_mean_estimates": summaries[
                    "target_mean_estimates"
                ].astype(np.float32),
            }
            array_types = COMBINED_TEST_SET_CACHE_ARRAY_TYPES
        else:
            arrays = {
                **test_set_common_arrays,
                "posterior_tf_log_weight": importance.log_weight.astype(
                    np.float32
                ),
                "posterior_tf_ess": importance.effective_sample_size.astype(
                    np.float64
                ),
                "posterior_tf_ess_fraction": (
                    importance.effective_sample_fraction.astype(np.float64)
                ),
                "posterior_tf_max_weight": importance.max_weight.astype(
                    np.float64
                ),
                "tf_target_mean_estimates": summaries[
                    "tf_target_mean_estimates"
                ].astype(np.float32),
            }
            array_types = TEST_SET_CACHE_ARRAY_TYPES
    else:
        assert base_log_prob is not None
        assert importance is not None
        assert population_log_ratio is not None
        arrays = {
            "sample": samples,
            "base_log_prob": base_log_prob.astype(np.float32),
            "posterior_tf_log_ratio": importance.log_ratio.astype(np.float32),
            "posterior_tf_log_weight": importance.log_weight.astype(np.float32),
            "posterior_tf_weight": importance.weight.astype(np.float32),
            "posterior_tf_ess": importance.effective_sample_size.astype(
                np.float64
            ),
            "posterior_tf_ess_fraction": (
                importance.effective_sample_fraction.astype(np.float64)
            ),
            "posterior_tf_max_weight": importance.max_weight.astype(
                np.float64
            ),
            "posterior_tf_log_mean_ratio": importance.log_mean_ratio.astype(
                np.float64
            ),
            "population_tf_log_ratio": population_log_ratio.astype(
                np.float64
            ),
            **common_arrays,
            **{
                name: value.astype(np.float32)
                for name, value in summaries.items()
            },
        }
        array_types = STANDARD_CACHE_ARRAY_TYPES
    saved = {
        name: _save_array(output_root, name, label, arrays[name])
        for name in array_types
    }

    provenance = {}
    for key, value in metadata.items():
        if key in required_metadata or not np.isscalar(value):
            continue
        provenance[key] = value.item() if isinstance(value, np.generic) else value
    partition_noise_seed = partition_seed
    provenance.update(
        {
            "matched_group_size": args.matched_group_size,
            "posterior_sample_seed": partition_seed,
            "image_noise_seed": partition_noise_seed,
            "spectral_noise_seed": partition_noise_seed + 101,
        }
    )
    if args.test_set:
        provenance.update(
            {
                "analysis_mode": TEST_SET_ANALYSIS_MODE,
                "snr_source": "dataset_record",
                "snr_redraw": False,
                "snr_clipping": False,
            }
        )
    analysis_mode = (
        TEST_SET_ANALYSIS_MODE if args.test_set else STANDARD_ANALYSIS_MODE
    )
    if args.isotropic_inclination_prior:
        cache_schema = COMBINED_TEST_SET_CACHE_SCHEMA
    elif args.test_set:
        cache_schema = TEST_SET_CACHE_SCHEMA
    else:
        cache_schema = CACHE_SCHEMA
    manifest = {
        "schema": cache_schema,
        "analysis_mode": analysis_mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_path),
        "dataset_size": len(dataset),
        "partition": {
            "index": args.partition_index,
            "total": args.nparts,
            "label": label,
            "galaxy_start": start,
            "galaxy_end": end,
        },
        "feature_names": list(feature_names),
        "physical_parameter_ranges": {
            name: [float(value) for value in config.par_ranges[name]]
            for name in feature_names
        },
        "target_transforms": {
            name: config.TARGET_TRANSFORMS[name] for name in feature_names
        },
        "density_coordinates": (
            {
                "stored_shear_samples": "physical_target_coordinates",
                "posterior_summary": "physical_target_coordinates",
                "map_selection": "not_computed",
            }
            if args.test_set
            else {
                "stored_base_log_prob": "normalized_target_coordinates",
                "map_selection": "physical_target_coordinates",
                "map_jacobian": (
                    "subtract_logabsdet_dphysical_dnormalized"
                ),
            }
        ),
        "observation_model": {
            "schema_version": int(config.observation["schema_version"]),
            "context_fields": list(config.observation["context_fields"]),
            "halpha_flux_semantics": config.observation[
                "halpha_flux_semantics"
            ],
            "image_snr_distribution": config.observation[
                "image_snr_distribution"
            ],
            "central_halpha_snr_distribution": config.observation[
                "central_halpha_snr_distribution"
            ],
            "center_exposure_s": float(
                config.observation["center_exposure_s"]
            ),
            "offset_exposure_s": float(
                config.observation["offset_exposure_s"]
            ),
        },
        "sample_shape": list(samples.shape),
        "symmetry": {
            "policy": "original_plus_r90_equal_mixture",
            "rotated_joint_rows_inverse_aligned": True,
        },
        "observation_provenance": provenance,
        "files": saved,
    }
    if args.test_set:
        assert generation_manifest is not None
        assert importance is not None
        test_set_metadata = {
            "population": "tf_conformed_catalog",
            "posterior_candidate_weighting": "tf_importance",
            "population_weighting": "uniform",
            "point_estimator": "mean",
            "map_computed": False,
            "tf_importance_weighting": True,
            "shape_noise_regularization": "report_time",
            "snr_source": "dataset_record",
            "snr_policy": "used_as_stored_without_redraw_or_clipping",
            "stored_candidate_parameters": ["g1", "g2"],
            "tf": tf_prior.to_dict(),
            "generation_manifest": generation_manifest,
        }
        posterior_description = (
            "TF-conformed catalog truth / TF-weighted posterior"
        )
        posterior_log_weight_description = (
            "stored within-galaxy log-softmax of posterior_log_ratio"
        )
        ess_manifest_name = "posterior_tf_ess"
        if args.isotropic_inclination_prior:
            inclination_metadata = {
                "training": "uniform_sini",
                "target": "uniform_cosi_0_1",
                "parameter": "sini",
                "composition": (
                    "added_to_tf_log_ratio_before_within_galaxy_log_softmax"
                ),
                "resampling": False,
                "bounds": [
                    float(inclination_prior.sini_min),
                    float(inclination_prior.sini_max),
                ],
            }
            test_set_metadata.update(
                posterior_candidate_weighting=(
                    "tf_x_isotropic_inclination_importance"
                ),
                inclination_importance_weighting=True,
                inclination_prior=inclination_metadata,
            )
            posterior_description = (
                "TF-conformed isotropic-inclination catalog truth / "
                "TF + isotropic-inclination-weighted posterior"
            )
            posterior_log_weight_description = (
                "TF component is not stored separately; the compact cache "
                "stores the within-galaxy log-softmax after adding the "
                "isotropic-inclination log-ratio"
            )
            ess_manifest_name = "posterior_target_ess"
        manifest.update(
            {
                "tf": {
                    **tf_prior.to_dict(),
                    "magnitude": "rmag_true",
                    "magnitude_measurement_error": 0.0,
                    "posterior_log_ratio": (
                        "computed log[p_TF(vcirc|rmag_true)/p0(vcirc)] "
                        "per candidate; not stored in the compact cache"
                    ),
                    "posterior_log_weight": (
                        posterior_log_weight_description
                    ),
                    "posterior_weight_normalization": "within_galaxy",
                    "population_log_ratio_normalization": (
                        "not_applicable_already_tf_conformed"
                    ),
                    "resampling": False,
                },
                "posterior_populations": {
                    "test_set": posterior_description
                },
                "test_set": test_set_metadata,
                ess_manifest_name: {
                    "minimum": float(
                        np.min(importance.effective_sample_size)
                    ),
                    "median": float(
                        np.median(importance.effective_sample_size)
                    ),
                    "mean": float(
                        np.mean(importance.effective_sample_size)
                    ),
                    "maximum": float(
                        np.max(importance.effective_sample_size)
                    ),
                },
            }
        )
    else:
        assert importance is not None
        manifest.update(
            {
                "tf": {
                    **tf_prior.to_dict(),
                    "magnitude": "rmag_true",
                    "magnitude_measurement_error": 0.0,
                    "posterior_log_ratio": (
                        "raw log[p_TF(vcirc|rmag_true)/p0(vcirc)] per candidate"
                    ),
                    "posterior_log_weight": (
                        "within-galaxy log-softmax of posterior_log_ratio"
                    ),
                    "posterior_weight_normalization": "within_galaxy",
                    "population_log_ratio_normalization": (
                        "global_after_partition_concat"
                    ),
                    "resampling": False,
                },
                "posterior_populations": {
                    "proposal": (
                        "base NPE trained under independent uniform vcirc prior"
                    ),
                    "tf_target": (
                        "same joint candidates with "
                        "p_TF(vcirc|rmag_true)/p0(vcirc) weights"
                    ),
                },
                "posterior_tf_ess": {
                    "minimum": float(
                        np.min(importance.effective_sample_size)
                    ),
                    "median": float(
                        np.median(importance.effective_sample_size)
                    ),
                    "mean": float(
                        np.mean(importance.effective_sample_size)
                    ),
                    "maximum": float(
                        np.max(importance.effective_sample_size)
                    ),
                },
            }
        )
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / f"{label}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        # Preserve feature order in physical_parameter_ranges; the fail-closed
        # cache contract requires it to match feature_names exactly.
        json.dump(manifest, handle, indent=2)
    logging.info("Saved %s posterior cache to %s", label, output_root)


if __name__ == "__main__":
    main()
