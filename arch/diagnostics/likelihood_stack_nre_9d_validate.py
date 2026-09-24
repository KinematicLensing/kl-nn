#!/usr/bin/env python3
"""Compare legacy and factorized 9D NRE permutation sensitivities."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pyxis.torch as pxt
import torch

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from diagnostics.likelihood_stack_core import (
    DATA_ROOT,
    G01_NPE,
    MODEL_ROOT,
    NRE9D_LEGACY_NAME,
    NRE9D_NAME,
    NRE_CONTEXT_DIM,
    NRE_VALID_DATASET,
    json_safe,
    write_json,
)
from diagnostics.likelihood_stack_nre_train import (
    extract_contexts,
    load_frozen_parent,
    load_ratio_head,
    nre_checkpoint_path,
)
from train import seed_everything


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--legacy-name", default=NRE9D_LEGACY_NAME)
    parser.add_argument("--factorized-name", default=NRE9D_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument(
        "--valid-data", type=Path, default=DATA_ROOT / NRE_VALID_DATASET
    )
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def permutation(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    indices = torch.randperm(values.shape[0], device=values.device, generator=generator)
    identity = torch.arange(values.shape[0], device=values.device)
    if torch.equal(indices, identity):
        indices = torch.roll(identity, shifts=1)
    return values.index_select(0, indices)


def replace_columns(
    parameters: torch.Tensor,
    columns: slice,
    generator: torch.Generator,
) -> torch.Tensor:
    result = parameters.clone()
    result[:, columns] = permutation(result[:, columns], generator)
    return result


def sensitivity(delta: torch.Tensor) -> dict[str, float]:
    values = delta.detach().abs().cpu().numpy().astype(np.float64)
    return {
        "mean_abs_logit_change": float(np.mean(values)),
        "median_abs_logit_change": float(np.median(values)),
        "p95_abs_logit_change": float(np.quantile(values, 0.95)),
    }


def evaluate_head(
    head,
    context: torch.Tensor,
    parameters: torch.Tensor,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = torch.Generator(device=context.device).manual_seed(seed)
    shear_parameters = replace_columns(parameters, slice(0, 2), generator)
    nuisance_parameters = replace_columns(parameters, slice(2, None), generator)
    all_parameters = permutation(parameters, generator)
    with torch.inference_mode():
        baseline = head(context, parameters)
        shear_changed = head(context, shear_parameters)
        nuisance_changed = head(context, nuisance_parameters)
        all_changed = head(context, all_parameters)
    return {
        "shear_only_shuffle": sensitivity(shear_changed - baseline),
        "nuisance_only_shuffle": sensitivity(nuisance_changed - baseline),
        "all_parameter_shuffle": sensitivity(all_changed - baseline),
    }


def evaluate_factorized_components(
    head,
    context: torch.Tensor,
    parameters: torch.Tensor,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    generator = torch.Generator(device=context.device).manual_seed(seed)
    shear = parameters[:, :2]
    nuisance = parameters[:, 2:]
    shuffled_shear = permutation(shear, generator)
    shuffled_nuisance = permutation(nuisance, generator)
    with torch.inference_mode():
        shear_base = head.forward_shear(context, shear)
        shear_changed = head.forward_shear(context, shuffled_shear)
        nuisance_base = head.forward_nuisance(context, shear, nuisance)
        nuisance_changed = head.forward_nuisance(
            context, shear, shuffled_nuisance
        )
    return {
        "shear_factor_shear_shuffle": sensitivity(shear_changed - shear_base),
        "conditional_nuisance_factor_nuisance_shuffle": sensitivity(
            nuisance_changed - nuisance_base
        ),
    }


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.max_samples < 2:
        raise ValueError("max-samples must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    encoder, model_config, _ = load_frozen_parent(
        parent_npe=args.parent_npe,
        model_root=args.model_root,
        checkpoint_suffix="best",
        device=device,
    )
    dataset = pxt.TorchDataset(str(args.valid_data))
    n_samples = min(int(args.max_samples), len(dataset))
    indices = np.arange(n_samples, dtype=np.int64)
    context, parameters = extract_contexts(
        encoder,
        dataset,
        indices,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        channels_last=bool(model_config.train.channels_last),
        names=tuple(config.TARGET_NAMES),
        split_name="diagnostic",
        target_names=tuple(config.TARGET_NAMES),
    )
    if context.shape != (n_samples, NRE_CONTEXT_DIM):
        raise ValueError(f"unexpected context shape {tuple(context.shape)}")
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    context = context.to(device)
    parameters = parameters.to(device)

    results = {
        "n_samples": n_samples,
        "valid_data": str(args.valid_data),
        "legacy_name": args.legacy_name,
        "factorized_name": args.factorized_name,
        "legacy": evaluate_head(
            load_ratio_head(
                nre_checkpoint_path(
                    args.model_root, args.legacy_name, suffix="best"
                ),
                device=device,
                expected_parent=args.parent_npe,
            )[0],
            context,
            parameters,
            seed=args.seed + 1,
        ),
        "factorized": evaluate_head(
            load_ratio_head(
                nre_checkpoint_path(
                    args.model_root, args.factorized_name, suffix="best"
                ),
                device=device,
                expected_parent=args.parent_npe,
            )[0],
            context,
            parameters,
            seed=args.seed + 1,
        ),
    }
    factorized_head = load_ratio_head(
        nre_checkpoint_path(args.model_root, args.factorized_name, suffix="best"),
        device=device,
        expected_parent=args.parent_npe,
    )[0]
    results["factorized_components"] = evaluate_factorized_components(
        factorized_head,
        context,
        parameters,
        seed=args.seed + 2,
    )
    results = json_safe(results)
    if args.output is not None:
        write_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
