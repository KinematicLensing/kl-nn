#!/usr/bin/env python3
"""Nautilus inference for the bounded factorized nine-dimensional NRE."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from diagnostics.likelihood_stack_core import (
    CACHE_ROOT,
    DATA_ROOT,
    G01_NPE,
    MODEL_ROOT,
    NRE9D_DIR,
    NRE9D_NAME,
    NRE_CONTEXT_DIM,
    NRE_VALID_DATASET,
    json_safe,
    write_json,
)
from diagnostics.likelihood_stack_nre_9d_infer import (
    apply_identity_noise,
    bounded_nre_log_density,
    chain_run_dir,
    encode_flow_context,
    load_frozen_parent,
    load_observation_batch,
    load_selection_indices,
    nre_checkpoint_path,
)
from diagnostics.likelihood_stack_nre_train import load_ratio_head
from tf_prior import TFPrior
from train import seed_everything
from utils import denormalize


DEFAULT_RUN_NAME = "nautilus_factorized_n100_s42"
DEFAULT_N_LIVE = 256
DEFAULT_N_BATCH = 256
DEFAULT_N_EFF = 5000
DEFAULT_N_LIKE_MAX = 200000


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_ROOT / NRE_VALID_DATASET
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--selection-file", type=Path, default=None)
    parser.add_argument("--n-galaxies", type=int, default=100)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--n-live", type=int, default=DEFAULT_N_LIVE)
    parser.add_argument("--n-batch", type=int, default=DEFAULT_N_BATCH)
    parser.add_argument("--n-eff", type=int, default=DEFAULT_N_EFF)
    parser.add_argument("--n-like-max", type=int, default=DEFAULT_N_LIKE_MAX)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_uniform_prior(parameter_names: tuple[str, ...]):
    try:
        from nautilus import Prior
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "nautilus-sampler is required; install it in the kl-nn environment"
        ) from exc
    prior = Prior()
    for name in parameter_names:
        prior.add_parameter(name, (-1.0, 1.0))
    return prior


def select_indices(
    dataset_size: int,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    dataset_size = int(dataset_size)
    count = int(count)
    if count <= 0 or count > dataset_size:
        raise ValueError("n-galaxies must be in [1, dataset_size]")
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(dataset_size, size=count, replace=False).astype(np.int64)
    )


def save_selection(
    path: Path,
    *,
    indices: np.ndarray,
    dataset: Path,
    dataset_size: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        index=np.asarray(indices, dtype=np.int64),
        dataset=np.asarray(str(dataset)),
        dataset_size=np.asarray(int(dataset_size)),
        selection_seed=np.asarray(int(seed)),
    )


def load_or_create_selection(
    path: Path,
    *,
    dataset: Path,
    dataset_size: int,
    count: int,
    seed: int,
) -> np.ndarray:
    if path.is_file():
        data = np.load(path, allow_pickle=False)
        if "index" not in data.files:
            raise ValueError(f"{path} does not contain an index array")
        indices = np.asarray(data["index"], dtype=np.int64)
        if indices.size != count:
            raise ValueError(
                f"{path} contains {indices.size} indices; expected {count}"
            )
        if np.any(indices < 0) or np.any(indices >= dataset_size):
            raise ValueError(f"{path} contains out-of-range indices")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{path} contains duplicate indices")
        return indices
    indices = select_indices(dataset_size, count, seed=seed)
    save_selection(
        path,
        indices=indices,
        dataset=dataset,
        dataset_size=dataset_size,
        seed=seed,
    )
    return indices


def artifact_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"galaxy_{int(index):06d}.npz"


def checkpoint_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"galaxy_{int(index):06d}.hdf5"


def save_artifact(
    path: Path,
    *,
    index: int,
    points: np.ndarray,
    log_weights: np.ndarray,
    log_likelihood: np.ndarray,
    truth_physical: np.ndarray,
    summary: dict,
    parameter_names: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            index=np.asarray(index, dtype=np.int64),
            points=np.asarray(points, dtype=np.float64),
            log_weights=np.asarray(log_weights, dtype=np.float64),
            log_likelihood=np.asarray(log_likelihood, dtype=np.float64),
            truth_physical=np.asarray(truth_physical, dtype=np.float64),
            parameter_names=np.asarray(parameter_names),
            summary_json=np.asarray(json.dumps(json_safe(summary))),
        )
    temporary.replace(path)


def load_artifact(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    required = {
        "index",
        "points",
        "log_weights",
        "log_likelihood",
        "truth_physical",
        "parameter_names",
        "summary_json",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    return {
        "index": int(data["index"].item()),
        "points": np.asarray(data["points"], dtype=np.float64),
        "log_weights": np.asarray(data["log_weights"], dtype=np.float64),
        "log_likelihood": np.asarray(data["log_likelihood"], dtype=np.float64),
        "truth": np.asarray(data["truth_physical"], dtype=np.float64),
        "parameter_names": tuple(str(name) for name in data["parameter_names"]),
        "summary": json.loads(str(data["summary_json"].item())),
    }


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_weights must be a non-empty one-dimensional array")
    finite = np.isfinite(values)
    if not np.all(finite):
        raise ValueError("Nautilus returned non-finite posterior weights")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Nautilus weights do not yield a finite posterior")
    return weights / total


def weighted_quantile(
    values: np.ndarray,
    quantiles: tuple[float, ...],
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
        raise ValueError("weights must be non-negative and normalized")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    return np.asarray(
        np.interp(np.asarray(quantiles, dtype=np.float64), cumulative, sorted_values),
        dtype=np.float64,
    )


def summarize_posterior(
    points: np.ndarray,
    log_weights: np.ndarray,
    log_likelihood: np.ndarray,
    *,
    parameter_names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    target_transforms,
    sampler_n_eff: float,
    sampler_n_like: int,
    sampler_log_z: float,
    sampler_eta: float,
    runtime_s: float,
) -> dict:
    points = np.asarray(points, dtype=np.float64)
    log_likelihood = np.asarray(log_likelihood, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != len(parameter_names):
        raise ValueError("points have the wrong parameter dimension")
    if log_likelihood.shape != (len(points),):
        raise ValueError("log_likelihood must have one value per point")
    weights = normalized_weights(log_weights)
    physical = denormalize(
        points,
        par_ranges,
        feature_names=parameter_names,
        target_transforms=target_transforms,
    )
    means = np.sum(physical * weights[:, None], axis=0)
    intervals = np.stack(
        [
            weighted_quantile(physical[:, column], (0.16, 0.84), weights)
            for column in range(physical.shape[1])
        ],
        axis=0,
    )
    finite_likelihood = np.isfinite(log_likelihood)
    if not np.any(finite_likelihood):
        raise ValueError("Nautilus returned no finite likelihood values")
    map_index = int(np.nanargmax(np.where(finite_likelihood, log_likelihood, -np.inf)))
    effective_sample_size = float(1.0 / np.sum(weights**2))
    g1_index = parameter_names.index("g1")
    g2_index = parameter_names.index("g2")
    return {
        "g1_mean": float(means[g1_index]),
        "g2_mean": float(means[g2_index]),
        "g1_lower": float(intervals[g1_index, 0]),
        "g1_upper": float(intervals[g1_index, 1]),
        "g2_lower": float(intervals[g2_index, 0]),
        "g2_upper": float(intervals[g2_index, 1]),
        "g1_map_like": float(physical[map_index, g1_index]),
        "g2_map_like": float(physical[map_index, g2_index]),
        "map_like_log_density": float(log_likelihood[map_index]),
        "n_samples": int(len(points)),
        "effective_sample_size": effective_sample_size,
        "nautilus_n_eff": float(sampler_n_eff),
        "n_like": int(sampler_n_like),
        "log_z_relative": float(sampler_log_z),
        "eta": float(sampler_eta),
        "runtime_s": float(runtime_s),
    }


def _load_nautilus():
    try:
        from nautilus import Sampler
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "nautilus-sampler is required; install it in the kl-nn environment"
        ) from exc
    return Sampler


def run_galaxy(
    *,
    run_dir: Path,
    index: int,
    context: torch.Tensor,
    rmag_true: float,
    truth_physical: np.ndarray,
    head,
    prior,
    parameter_names: tuple[str, ...],
    par_ranges: dict[str, list[float]],
    target_transforms,
    n_live: int,
    n_batch: int,
    n_eff: int,
    n_like_max: int,
    seed: int,
    resume: bool,
    tf_prior: TFPrior,
) -> dict:
    Sampler = _load_nautilus()
    checkpoint = checkpoint_path(run_dir, index)
    artifact = artifact_path(run_dir, index)
    likelihood = lambda points: bounded_nre_log_density(
        points,
        head=head,
        context=context,
        rmag_true=rmag_true,
        parameter_names=parameter_names,
        par_ranges=par_ranges,
        prior=tf_prior,
    )
    sampler = Sampler(
        prior,
        likelihood,
        n_live=n_live,
        n_batch=n_batch,
        vectorized=True,
        pass_dict=False,
        seed=int(seed),
        filepath=str(checkpoint),
        resume=bool(resume),
    )
    start = time.perf_counter()
    sampler.run(n_eff=n_eff, n_like_max=n_like_max, verbose=False)
    runtime_s = time.perf_counter() - start
    points, log_weights, log_likelihood = sampler.posterior()
    summary = summarize_posterior(
        points,
        log_weights,
        log_likelihood,
        parameter_names=parameter_names,
        par_ranges=par_ranges,
        target_transforms=target_transforms,
        sampler_n_eff=float(sampler.n_eff),
        sampler_n_like=int(sampler.n_like),
        sampler_log_z=float(sampler.log_z),
        sampler_eta=float(sampler.eta),
        runtime_s=runtime_s,
    )
    summary.update(
        {
            "index": int(index),
            "truth_g1": float(truth_physical[parameter_names.index("g1")]),
            "truth_g2": float(truth_physical[parameter_names.index("g2")]),
            "truth_parameters": np.asarray(truth_physical).tolist(),
            "rmag_true": float(rmag_true),
            "status": "completed",
        }
    )
    save_artifact(
        artifact,
        index=index,
        points=points,
        log_weights=log_weights,
        log_likelihood=log_likelihood,
        truth_physical=truth_physical,
        summary=summary,
        parameter_names=parameter_names,
    )
    return summary


def _write_run_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def main(argv=None) -> None:
    args = parse_args(argv)
    names = tuple(config.TARGET_NAMES)
    if len(names) != 9:
        raise ValueError("the Nautilus campaign requires the nine-target schema")
    if args.n_live < 2 * len(names):
        raise ValueError("n-live must be at least twice the parameter dimension")
    if args.n_batch < 1 or args.n_eff <= 0 or args.n_like_max <= 0:
        raise ValueError("n-batch, n-eff, and n-like-max must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    config.require_matching_dataset_par_ranges(args.data_dir, config.par_ranges)
    dataset = pxt.TorchDataset(str(args.data_dir))
    dataset_size = len(dataset)
    run_dir = chain_run_dir(args.cache_root, args.nre_name, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.selection_file is None:
        selection_file = run_dir / "selection.npz"
    else:
        selection_file = Path(args.selection_file)
    indices = load_or_create_selection(
        selection_file,
        dataset=args.data_dir,
        dataset_size=dataset_size,
        count=args.n_galaxies,
        seed=args.selection_seed,
    )
    if args.max_galaxies is not None:
        if args.max_galaxies <= 0:
            raise ValueError("max-galaxies must be positive")
        indices = indices[: args.max_galaxies]

    if args.overwrite and not args.resume:
        for index in indices:
            for path in (artifact_path(run_dir, int(index)), checkpoint_path(run_dir, int(index))):
                if path.exists():
                    path.unlink()
    pending = [
        int(index)
        for index in indices
        if not (args.resume and artifact_path(run_dir, int(index)).is_file())
    ]
    report_dir = Path(args.report_dir or NRE9D_DIR / args.run_name)
    metadata_path = run_dir / "metadata.json"
    metadata = {
        "status": "running",
        "sampler": "nautilus",
        "run_name": args.run_name,
        "chain_cache": str(run_dir),
        "report_dir": str(report_dir),
        "parent_npe": args.parent_npe,
        "nre_name": args.nre_name,
        "data_dir": str(args.data_dir),
        "selection_file": str(selection_file),
        "dataset_size": int(dataset_size),
        "indices": [int(index) for index in indices],
        "n_expected": int(len(indices)),
        "n_cached_before": int(len(indices) - len(pending)),
        "n_live": int(args.n_live),
        "n_batch": int(args.n_batch),
        "n_eff_target": int(args.n_eff),
        "n_like_max": int(args.n_like_max),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "selection_seed": int(args.selection_seed),
        "parameter_names": list(names),
    }
    _write_run_metadata(metadata_path, metadata)

    encoder, model_config, parent_checkpoint = load_frozen_parent(
        parent_npe=args.parent_npe,
        model_root=args.model_root,
        checkpoint_suffix="best",
        device=device,
    )
    checkpoint = nre_checkpoint_path(args.model_root, args.nre_name)
    head, nre_meta = load_ratio_head(
        checkpoint,
        device=device,
        expected_parent=args.parent_npe,
    )
    if tuple(nre_meta.get("target_names") or ()) != names:
        raise ValueError("9D checkpoint target_names do not match the current schema")
    if int(nre_meta.get("parameter_dim", 0)) != len(names):
        raise ValueError("9D checkpoint parameter_dim is not nine")
    prior = build_uniform_prior(names)
    tf_prior = TFPrior()
    # Construct once so malformed TF configuration fails before the first galaxy.
    tf_prior.to_dict()
    channels_last = bool(model_config.train.channels_last)

    try:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = np.asarray(
                pending[start : start + args.batch_size], dtype=np.int64
            )
            batch = load_observation_batch(dataset, batch_indices, device)
            image, spec = apply_identity_noise(
                batch["img"],
                batch["spec"],
                batch["image_snr"],
                batch["spec_snr"],
                batch_indices,
                seed=args.seed,
                device=device,
            )
            if channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            context = encode_flow_context(
                encoder,
                image,
                spec,
                batch["fib_pos"],
                batch["observation_context"],
            )
            if context.shape != (len(batch_indices), NRE_CONTEXT_DIM):
                raise ValueError(f"unexpected context shape {tuple(context.shape)}")
            truth_physical = denormalize(
                batch["fid_pars"].cpu().numpy(),
                config.par_ranges,
                feature_names=names,
                target_transforms=config.TARGET_TRANSFORMS,
            )
            for row, index in enumerate(batch_indices):
                if artifact_path(run_dir, int(index)).is_file():
                    continue
                row_context = context[row].detach()
                rmag_true = float(batch["observation_context"]["rmag_true"][row].item())
                summary = run_galaxy(
                    run_dir=run_dir,
                    index=int(index),
                    context=row_context,
                    rmag_true=rmag_true,
                    truth_physical=truth_physical[row],
                    head=head,
                    prior=prior,
                    parameter_names=names,
                    par_ranges=config.par_ranges,
                    target_transforms=config.TARGET_TRANSFORMS,
                    n_live=args.n_live,
                    n_batch=args.n_batch,
                    n_eff=args.n_eff,
                    n_like_max=args.n_like_max,
                    seed=args.seed + int(index),
                    resume=args.resume,
                    tf_prior=tf_prior,
                )
                print(
                    f"completed galaxy {int(index)} "
                    f"({summary['n_like']} likelihood calls, "
                    f"n_eff={summary['nautilus_n_eff']:.1f})",
                    flush=True,
                )
            metadata["n_cached"] = len(
                list(run_dir.glob("galaxy_*.npz"))
            )
            _write_run_metadata(metadata_path, metadata)
    except Exception as exc:
        metadata.update({"status": "failed", "error": repr(exc)})
        _write_run_metadata(metadata_path, metadata)
        raise
    finally:
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata.update(
        {
            "status": "completed",
            "parent_checkpoint": str(parent_checkpoint),
            "nre_checkpoint": str(checkpoint),
            "n_cached": len(list(run_dir.glob("galaxy_*.npz"))),
        }
    )
    _write_run_metadata(metadata_path, metadata)
    print(json.dumps(json_safe(metadata), indent=2))


if __name__ == "__main__":
    main()
