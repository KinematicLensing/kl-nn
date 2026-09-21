#!/usr/bin/env python3
"""Train a frozen-encoder 2D NRE head on g01 valid_100k. New nre2d dir only."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import numpy as np
import pyxis.torch as pxt
import torch
from torch.optim import AdamW

ARCH_DIR = Path(__file__).resolve().parents[1]
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from data import apply_central_halpha_snr_noise, apply_image_noise_for_snr
from diagnostics.likelihood_stack_core import (
    DATA_ROOT,
    G01_NPE,
    MODEL_ROOT,
    NRE_BATCH_SIZE,
    NRE_CONTEXT_DIM,
    NRE_EPOCHS,
    NRE_HIDDEN_DIMS,
    NRE_LR,
    NRE_NAME,
    NRE_TRAIN_DATASET,
    NRE_VALID_DATASET,
    json_safe,
    noise_seeds,
    shear_columns,
    write_json,
)
from diagnostics.likelihood_stack_nre_core import (
    RatioHead,
    freeze_encoder,
    nre_bce_loss,
)
from model_registry import load_model_config
from networks import FEATURE_DIM, KLNPE
from train import (
    _seeded_generator,
    build_observation_levels,
    load_model,
    seed_everything,
    validate_observation_record,
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--parent-npe", default=G01_NPE)
    parser.add_argument("--nre-name", default=NRE_NAME)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--parent-checkpoint-suffix", default="best")
    parser.add_argument(
        "--train-data", type=Path, default=DATA_ROOT / NRE_TRAIN_DATASET
    )
    parser.add_argument(
        "--valid-data", type=Path, default=DATA_ROOT / NRE_VALID_DATASET
    )
    parser.add_argument("--epochs", type=int, default=NRE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=NRE_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=NRE_LR)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=list(NRE_HIDDEN_DIMS)
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def nre_model_dir(model_root: Path, nre_name: str) -> Path:
    return Path(model_root) / nre_name


def nre_checkpoint_path(model_root: Path, nre_name: str, suffix: str = "best") -> Path:
    return nre_model_dir(model_root, nre_name) / f"{nre_name}{suffix}"


def parent_json_path(model_root: Path, nre_name: str) -> Path:
    return nre_model_dir(model_root, nre_name) / "parent.json"


def parent_checkpoint_path(
    model_root: Path, parent_npe: str, suffix: str = "best"
) -> Path:
    return Path(model_root) / parent_npe / f"{parent_npe}{suffix}"


def refuse_parent_overwrite(model_root: Path, parent_npe: str, nre_name: str) -> Path:
    nre_dir = nre_model_dir(model_root, nre_name).resolve()
    parent_dir = (Path(model_root) / parent_npe).resolve()
    if nre_dir == parent_dir:
        raise RuntimeError(
            "refusing to write the NRE head into the parent NPE directory "
            f"{parent_dir}"
        )
    return nre_dir


def load_frozen_parent(
    *,
    parent_npe: str,
    model_root: Path,
    checkpoint_suffix: str,
    device: torch.device,
):
    configs_root = Path(model_root).parent / "configs"
    model_config = load_model_config(parent_npe, configs_root=str(configs_root))
    config.set_model_config(model_config)
    checkpoint = parent_checkpoint_path(model_root, parent_npe, checkpoint_suffix)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    encoder = load_model(
        KLNPE,
        path=str(checkpoint),
        model_name=parent_npe,
        device=str(device),
        strict=True,
        networks_root=str(Path(model_root).parent / "networks"),
    )
    freeze_encoder(encoder)
    channels_last = bool(model_config.train.channels_last)
    if channels_last:
        encoder = encoder.to(memory_format=torch.channels_last)
    if encoder.flow_context_features != FEATURE_DIM:
        raise ValueError(
            "frozen flow context width "
            f"{encoder.flow_context_features} != {FEATURE_DIM}"
        )
    return encoder, model_config, checkpoint


@torch.inference_mode()
def encode_flow_context(encoder, image, spectra, fiber_positions, observation_context):
    raw = encoder._raw_features(
        image, spectra, fiber_positions, observation_context
    )
    return encoder._flow_context(raw)


def apply_identity_noise(
    image: torch.Tensor,
    spec: torch.Tensor,
    image_snr: torch.Tensor,
    spec_snr: torch.Tensor,
    indices: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = int(config.observation["center_fiber_index"])
    noisy_images = []
    noisy_spectra = []
    for row, index in enumerate(indices):
        image_seed, spec_seed = noise_seeds(seed, 0, int(index))
        noisy_images.append(
            apply_image_noise_for_snr(
                image[row : row + 1],
                image_snr[row : row + 1],
                randgen=_seeded_generator(device, image_seed),
            )
        )
        noisy_spectra.append(
            apply_central_halpha_snr_noise(
                spec[row : row + 1],
                spec_snr[row : row + 1],
                center_fiber_index=center,
                center_exposure_s=config.observation["center_exposure_s"],
                offset_exposure_s=config.observation["offset_exposure_s"],
                spectral_units=config.observation["spectral_units"],
                randgen=_seeded_generator(device, spec_seed),
                device=device,
            )
        )
    return torch.cat(noisy_images, dim=0), torch.cat(noisy_spectra, dim=0)


def load_observation_batch(dataset, indices, device: torch.device):
    rows = [dataset[int(index)] for index in indices]
    image = torch.stack([torch.as_tensor(row["img"]) for row in rows]).float().to(device)
    spec = torch.stack([torch.as_tensor(row["spec"]) for row in rows]).float().to(device)
    positions = (
        torch.stack([torch.as_tensor(row["fib_pos"]) for row in rows]).float().to(device)
    )
    labels = torch.stack([torch.as_tensor(row["fid_pars"]) for row in rows]).float()
    metadata = [
        validate_observation_record(row, location=f"nre record {int(index)}")
        for row, index in zip(rows, indices)
    ]
    rmag = torch.tensor(
        [item[0] for item in metadata], device=device, dtype=torch.float32
    )
    image_snr, spec_snr = build_observation_levels(
        torch.tensor(
            [item[2] for item in metadata], device=device, dtype=torch.float32
        ),
        torch.tensor(
            [item[3] for item in metadata], device=device, dtype=torch.float32
        ),
    )
    observation_context = {
        "rmag_true": rmag,
        "image_snr": image_snr,
        "central_halpha_snr": spec_snr,
    }
    return {
        "img": image,
        "spec": spec,
        "fib_pos": positions,
        "fid_pars": labels,
        "image_snr": image_snr,
        "spec_snr": spec_snr,
        "observation_context": observation_context,
    }


@torch.inference_mode()
def extract_contexts(
    encoder,
    dataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
    channels_last: bool,
    names: tuple[str, ...],
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 so g can be shuffled")
    g1_idx, g2_idx = shear_columns(names)
    contexts = []
    shears = []
    total = math.ceil(len(indices) / batch_size)
    encoder.eval()
    for batch_number, start in enumerate(range(0, len(indices), batch_size), start=1):
        batch_indices = np.asarray(indices[start : start + batch_size], dtype=np.int64)
        batch = load_observation_batch(dataset, batch_indices, device)
        image, spec = apply_identity_noise(
            batch["img"],
            batch["spec"],
            batch["image_snr"],
            batch["spec_snr"],
            batch_indices,
            seed=seed,
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
            raise ValueError(
                "flow context must have shape "
                f"({len(batch_indices)}, {NRE_CONTEXT_DIM}); got {tuple(context.shape)}"
            )
        shear = batch["fid_pars"][:, [g1_idx, g2_idx]]
        contexts.append(context.float().cpu())
        shears.append(shear)
        if batch_number == 1 or batch_number == total or batch_number % 20 == 0:
            print(
                f"{split_name}: encoded batch {batch_number}/{total}",
                flush=True,
            )
    return torch.cat(contexts), torch.cat(shears)


def epoch_loss(
    head: RatioHead,
    context: torch.Tensor,
    shear: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
    train: bool,
    optimizer: AdamW | None = None,
) -> float:
    n = int(context.shape[0])
    order = torch.randperm(n, device=context.device, generator=generator)
    summed = 0.0
    counted = 0
    head.train(train)
    for start in range(0, n, batch_size):
        batch_index = order[start : start + batch_size]
        if int(batch_index.numel()) < 2:
            continue
        if train:
            if optimizer is None:
                raise ValueError("optimizer is required when train=True")
            optimizer.zero_grad(set_to_none=True)
        loss = nre_bce_loss(
            head, context[batch_index], shear[batch_index]
        )
        if train:
            loss.backward()
            optimizer.step()
        summed += float(loss.detach().item()) * int(batch_index.numel())
        counted += int(batch_index.numel())
    if counted == 0:
        raise RuntimeError("no NRE batches with size >= 2")
    return summed / counted


def load_ratio_head(
    path: Path,
    *,
    device: torch.device,
    expected_parent: str | None = None,
) -> tuple[RatioHead, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "head" not in payload:
        raise ValueError(f"{path} is not an NRE head checkpoint")
    meta = dict(payload.get("meta") or {})
    if expected_parent is not None and meta.get("parent_npe") != expected_parent:
        raise ValueError(
            f"{path} parent_npe={meta.get('parent_npe')!r}; expected {expected_parent!r}"
        )
    hidden = tuple(meta.get("hidden_dims") or NRE_HIDDEN_DIMS)
    context_dim = int(meta.get("context_dim") or NRE_CONTEXT_DIM)
    head = RatioHead(context_dim=context_dim, hidden_dims=hidden).to(device)
    head.load_state_dict(payload["head"])
    head.eval()
    return head, meta


def train_head(
    train_context: torch.Tensor,
    train_shear: torch.Tensor,
    valid_context: torch.Tensor,
    valid_shear: torch.Tensor,
    *,
    hidden_dims: tuple[int, ...],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[RatioHead, dict]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    seed_everything(seed, deterministic=True)
    head = RatioHead(
        context_dim=int(train_context.shape[1]),
        hidden_dims=hidden_dims,
    ).to(device)
    optimizer = AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_gen = torch.Generator(device=device).manual_seed(seed + 11)
    valid_gen = torch.Generator(device=device).manual_seed(seed + 29)
    history = []
    best_state = None
    best_valid = float("inf")
    best_epoch = -1
    report_every = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        train_bce = epoch_loss(
            head,
            train_context,
            train_shear,
            batch_size=batch_size,
            generator=train_gen,
            train=True,
            optimizer=optimizer,
        )
        with torch.inference_mode():
            valid_bce = epoch_loss(
                head,
                valid_context,
                valid_shear,
                batch_size=batch_size,
                generator=valid_gen,
                train=False,
            )
        history.append(
            {"epoch": epoch, "train_bce": train_bce, "valid_bce": valid_bce}
        )
        if valid_bce < best_valid:
            best_valid = valid_bce
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % report_every == 0:
            print(
                f"nre epoch {epoch:4d}/{epochs}: "
                f"train_bce={train_bce:.6f} valid_bce={valid_bce:.6f} "
                f"best={best_valid:.6f}@{best_epoch}",
                flush=True,
            )
    if best_state is None:
        raise RuntimeError("NRE training produced no finite validation BCE")
    head.load_state_dict(best_state)
    head.eval()
    return head, {
        "best_epoch": best_epoch,
        "best_valid_bce": best_valid,
        "history": history,
    }


def save_nre_checkpoint(
    *,
    nre_dir: Path,
    nre_name: str,
    head: RatioHead,
    meta: dict,
    overwrite: bool,
) -> Path:
    nre_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = nre_dir / f"{nre_name}best"
    sidecar = nre_dir / "parent.json"
    if (checkpoint.exists() or sidecar.exists()) and not overwrite:
        raise FileExistsError(f"{checkpoint} exists; use --overwrite")
    payload = {"head": head.state_dict(), "meta": json_safe(meta)}
    torch.save(payload, checkpoint)
    write_json(sidecar, meta)
    return checkpoint


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.nre_name == args.parent_npe:
        raise ValueError("nre-name must differ from the parent NPE")
    nre_dir = refuse_parent_overwrite(args.model_root, args.parent_npe, args.nre_name)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    seed_everything(args.seed, deterministic=True)

    encoder, model_config, parent_ckpt = load_frozen_parent(
        parent_npe=args.parent_npe,
        model_root=args.model_root,
        checkpoint_suffix=args.parent_checkpoint_suffix,
        device=device,
    )
    names = tuple(config.TARGET_NAMES)
    config.require_matching_dataset_par_ranges(args.train_data, config.par_ranges)
    config.require_matching_dataset_par_ranges(args.valid_data, config.par_ranges)
    trainable = [
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(f"encoder still trainable: {trainable[:8]}")

    train_ds = pxt.TorchDataset(str(args.train_data))
    valid_ds = pxt.TorchDataset(str(args.valid_data))
    train_indices = np.arange(len(train_ds), dtype=np.int64)
    valid_indices = np.arange(len(valid_ds), dtype=np.int64)
    channels_last = bool(model_config.train.channels_last)
    print(
        f"encoding train={args.train_data} n={len(train_indices)} "
        f"valid={args.valid_data} n={len(valid_indices)}",
        flush=True,
    )
    train_context, train_shear = extract_contexts(
        encoder,
        train_ds,
        train_indices,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        channels_last=channels_last,
        names=names,
        split_name="train",
    )
    valid_context, valid_shear = extract_contexts(
        encoder,
        valid_ds,
        valid_indices,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed + 17,
        channels_last=channels_last,
        names=names,
        split_name="valid",
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head, fit = train_head(
        train_context.to(device),
        train_shear.to(device),
        valid_context.to(device),
        valid_shear.to(device),
        hidden_dims=tuple(args.hidden_dims),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        seed=args.seed,
    )
    meta = {
        "parent_npe": args.parent_npe,
        "parent_checkpoint": str(parent_ckpt),
        "nre_name": args.nre_name,
        "hidden_dims": list(args.hidden_dims),
        "context_dim": NRE_CONTEXT_DIM,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "train_data": str(args.train_data),
        "valid_data": str(args.valid_data),
        "n_train": int(len(train_indices)),
        "n_valid": int(len(valid_indices)),
        **fit,
    }
    checkpoint = save_nre_checkpoint(
        nre_dir=nre_dir,
        nre_name=args.nre_name,
        head=head,
        meta=meta,
        overwrite=args.overwrite,
    )
    print(json.dumps(json_safe({
        "checkpoint": str(checkpoint),
        "best_epoch": fit["best_epoch"],
        "best_valid_bce": fit["best_valid_bce"],
    }), indent=2))


if __name__ == "__main__":
    main()
