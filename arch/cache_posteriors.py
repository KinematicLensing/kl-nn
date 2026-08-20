#!/usr/bin/env python3
"""Cache one base NPE candidate bank and its post-training TF weights.

The sampling adapter in :mod:`train` constructs the canonical original/R90
ensemble and inverse-aligns every rotated parameter row. This script adds no
second observation, resampling, or alternate network path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    from .model_registry import load_model_config
    from .networks import KLNPE
    from .tf_prior import (
        TFPrior,
        population_log_importance_ratio,
        posterior_importance_weights,
    )
    from .train import load_model, sample_density, seed_everything
    from .utils import denormalize, resolve_feature_index
except ImportError:  # Direct execution from arch/.
    import config
    from model_registry import load_model_config
    from networks import KLNPE
    from tf_prior import (
        TFPrior,
        population_log_importance_ratio,
        posterior_importance_weights,
    )
    from train import load_model, sample_density, seed_everything
    from utils import denormalize, resolve_feature_index


DEFAULT_SHARED_ROOT = Path("/ocean/projects/phy250048p/shared")

CACHE_ARRAY_TYPES = (
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
    "spectral_reference_quality",
    "proposal_map_estimates",
    "proposal_mean_estimates",
    "tf_target_map_estimates",
    "tf_target_mean_estimates",
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
    vcirc_bounds = config.par_ranges["vcirc"]
    tf_prior = TFPrior(
        slope=args.tf_slope,
        intercept=args.tf_intercept,
        scatter_dex=args.tf_scatter_dex,
        vcirc_min=float(vcirc_bounds[0]),
        vcirc_max=float(vcirc_bounds[1]),
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
    samples_normalized, base_log_prob, metadata = sample_density(
        model,
        subset,
        args.nsamples,
        device=str(device),
        matched_group_size=args.matched_group_size,
        noise_seed=partition_seed,
        spectral_noise_seed=(
            partition_seed + 101
        ),
        spectral_quality_seed=(
            partition_seed + 307
        ),
        return_log_prob=True,
        return_observation_metadata=True,
    )
    samples_normalized = _to_numpy(samples_normalized, name="samples")
    base_log_prob = _to_numpy(
        base_log_prob, name="base_log_prob"
    )
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
    if base_log_prob.shape != samples_normalized.shape[:2]:
        raise RuntimeError(
            f"unexpected base-log-probability shape {base_log_prob.shape}"
        )
    if (
        not np.all(np.isfinite(samples_normalized))
        or not np.all(np.isfinite(base_log_prob))
    ):
        raise RuntimeError(
            "base posterior returned non-finite samples or scores"
        )
    if not isinstance(metadata, dict):
        raise RuntimeError(
            "sample_density must return observation metadata"
        )

    required_metadata = (
        "truth",
        "rmag_true",
        "spectral_reference_quality",
    )
    missing = [name for name in required_metadata if name not in metadata]
    if missing:
        raise RuntimeError(f"sample_density metadata is missing {missing}")
    truth_normalized = _to_numpy(metadata["truth"], name="truth")
    rmag_true = _to_numpy(
        metadata["rmag_true"], name="rmag_true"
    ).astype(np.float64)
    spectral_quality = _to_numpy(
        metadata["spectral_reference_quality"],
        name="spectral_reference_quality",
    ).astype(np.float64)
    if truth_normalized.shape != (args.ngals, len(feature_names)):
        raise RuntimeError(
            f"unexpected truth shape {truth_normalized.shape}"
        )
    if (
        rmag_true.shape != (args.ngals,)
        or spectral_quality.shape != (args.ngals,)
    ):
        raise RuntimeError(
            "observation metadata must contain one scalar per galaxy"
        )
    if (
        not np.all(np.isfinite(rmag_true))
        or not np.all(np.isfinite(spectral_quality))
    ):
        raise RuntimeError("observation metadata contains non-finite values")

    samples = denormalize(
        samples_normalized,
        par_ranges=config.par_ranges,
        feature_names=feature_names,
    ).astype(np.float32, copy=False)
    truth = denormalize(
        truth_normalized,
        par_ranges=config.par_ranges,
        feature_names=feature_names,
    ).astype(np.float32, copy=False)
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
        base_log_prob,
        importance.log_ratio,
        importance.weight,
        feature_names,
    )

    low_ess = (
        importance.effective_sample_fraction < args.warn_ess_fraction
    )
    if np.any(low_ess):
        logging.warning(
            "%d/%d galaxies have posterior TF ESS fraction below %.3f; "
            "minimum %.5f",
            int(np.count_nonzero(low_ess)),
            len(low_ess),
            args.warn_ess_fraction,
            float(np.min(importance.effective_sample_fraction)),
        )

    dataset_name = dataset_path.name
    if args.cache_tag:
        dataset_name += f"_{args.cache_tag.strip('_')}"
    output_root = args.cache_root / args.model_name / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sample": samples,
        "base_log_prob": base_log_prob.astype(np.float32),
        "posterior_tf_log_ratio": (
            importance.log_ratio.astype(np.float32)
        ),
        "posterior_tf_log_weight": importance.log_weight.astype(np.float32),
        "posterior_tf_weight": importance.weight.astype(np.float32),
        "posterior_tf_ess": (
            importance.effective_sample_size.astype(np.float64)
        ),
        "posterior_tf_ess_fraction": (
            importance.effective_sample_fraction.astype(np.float64)
        ),
        "posterior_tf_max_weight": (
            importance.max_weight.astype(np.float64)
        ),
        "posterior_tf_log_mean_ratio": (
            importance.log_mean_ratio.astype(np.float64)
        ),
        "population_tf_log_ratio": (
            population_log_ratio.astype(np.float64)
        ),
        "truth": truth,
        "rmag_true": rmag_true.astype(np.float32),
        "spectral_reference_quality": spectral_quality.astype(np.float32),
        **{
            name: value.astype(np.float32)
            for name, value in summaries.items()
        },
    }
    saved = {
        name: _save_array(output_root, name, label, arrays[name])
        for name in CACHE_ARRAY_TYPES
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
            "spectral_quality_seed": partition_noise_seed + 307,
        }
    )
    manifest = {
        "schema": "klnn-posterior-cache-v1",
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
        "sample_shape": list(samples.shape),
        "symmetry": {
            "policy": "original_plus_r90_equal_mixture",
            "rotated_joint_rows_inverse_aligned": True,
        },
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
        "observation_provenance": provenance,
        "files": saved,
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
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with (meta_dir / f"{label}.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    logging.info("Saved %s posterior cache to %s", label, output_root)


if __name__ == "__main__":
    main()
