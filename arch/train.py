from __future__ import annotations

import json
import logging
import os
import random
import time
from os.path import join

import numpy as np
import pandas as pd
import pyxis.torch as pxt
import torch
from torch import nn, optim
from torch.utils.data import Subset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)

try:
    from . import config
    from .data import (
        apply_central_halpha_snr_noise,
        apply_image_noise_for_snr,
        central_halpha_line_norm,
        image_matched_filter_norm,
        rotate_90_datavector,
        rotate_90_parameters,
    )
    from .model_registry import (
        infer_model_name_from_checkpoint_path,
        load_networks_module_for_model,
    )
    from .networks import CCLPretrain
except ImportError:  # Direct execution with arch/ on sys.path.
    import config
    from data import (
        apply_central_halpha_snr_noise,
        apply_image_noise_for_snr,
        central_halpha_line_norm,
        image_matched_filter_norm,
        rotate_90_datavector,
        rotate_90_parameters,
    )
    from model_registry import (
        infer_model_name_from_checkpoint_path,
        load_networks_module_for_model,
    )
    from networks import CCLPretrain


RNG_STREAM_IDS = {
    "ambient": 0,
    "train_order": 1,
    "valid_order": 2,
    "train_img_noise": 3,
    "train_spec_noise": 4,
    "valid_img_noise": 5,
    "valid_spec_noise": 6,
    "train_spec_quality": 7,
    "valid_spec_quality": 8,
    "train_npe_view": 9,
    "valid_npe_view": 10,
    "train_image_snr": 11,
    "train_central_halpha_snr": 12,
    "valid_image_snr": 13,
    "valid_central_halpha_snr": 14,
}

GALAXY_AXIS_FIBER_LAYOUT_CODE = 1
INSTRUMENT_METADATA = {
    "image_band_code": 0,
    "target_line_code": 0,
    "spectral_units_code": 0,
    "center_fiber_index": 2,
}
INSTRUMENT_FLOAT_METADATA = (
    "center_exposure_s",
    "offset_exposure_s",
    "image_reference_psf_fwhm_arcsec",
    "image_pixel_scale_arcsec",
)


def _format_elapsed_time(seconds):
    """Format an elapsed wall-clock duration as ``HH:MM:SS.s``."""
    total_tenths = max(0, round(float(seconds) * 10))
    hours, remainder = divmod(total_tenths, 36_000)
    minutes, remainder = divmod(remainder, 600)
    return f"{hours:02d}:{minutes:02d}:{remainder / 10:04.1f}"


def derive_stream_seed(base_seed, rank=0, epoch=0, stream="ambient"):
    if stream not in RNG_STREAM_IDS:
        raise ValueError(f"unknown RNG stream {stream!r}")
    return int(
        (
            int(base_seed)
            + 1_000_003 * int(rank)
            + 10_007 * int(epoch)
            + 97 * RNG_STREAM_IDS[stream]
        )
        % (2**63 - 1)
    )


def seed_everything(seed, deterministic=False):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def _scalar(record, name, location):
    if name not in record:
        raise ValueError(f"{location} is missing required metadata {name!r}")
    value = torch.as_tensor(record[name])
    if value.numel() != 1:
        raise ValueError(f"{location} metadata {name!r} must be scalar")
    return value.reshape(())


def validate_observation_record(record, *, location="record"):
    """Validate the sole supported simulator-v3 LMDB record."""
    observation = config.observation
    version = int(_scalar(record, "observation_model_version", location).item())
    if version != int(observation["schema_version"]):
        raise ValueError(
            f"{location} has schema version {version}; "
            f"expected {observation['schema_version']}"
        )
    layout = int(_scalar(record, "fiber_layout", location).item())
    expected_layout = GALAXY_AXIS_FIBER_LAYOUT_CODE
    if layout != expected_layout:
        raise ValueError(
            f"{location} has fiber-layout code {layout}; expected {expected_layout}"
        )
    for name, expected in INSTRUMENT_METADATA.items():
        actual = int(_scalar(record, name, location).item())
        configured = (
            int(observation["center_fiber_index"])
            if name == "center_fiber_index"
            else expected
        )
        if actual != configured:
            raise ValueError(
                f"{location} has {name}={actual}; expected {configured}"
            )
    for name in INSTRUMENT_FLOAT_METADATA:
        actual = float(_scalar(record, name, location).item())
        expected = float(observation[name])
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"{location} has {name}={actual}; expected {expected}"
            )
    rmag = float(_scalar(record, "rmag_true", location).item())
    halpha = float(_scalar(record, "halpha_flux_true", location).item())
    if not np.isfinite(rmag) or not (
        observation["rmag_min"] - 1e-4
        <= rmag
        <= observation["rmag_max"] + 1e-4
    ):
        raise ValueError(f"{location} has invalid rmag_true={rmag}")
    tolerance = 2e-6 * observation["halpha_flux_max"]
    if not np.isfinite(halpha) or not (
        observation["halpha_flux_min"] - tolerance
        <= halpha
        <= observation["halpha_flux_max"] + tolerance
    ):
        raise ValueError(f"{location} has invalid halpha_flux_true={halpha}")
    fid = torch.as_tensor(record["fid_pars"])
    if fid.numel() != len(config.TARGET_NAMES):
        raise ValueError(
            f"{location} must contain exactly {len(config.TARGET_NAMES)} targets"
        )
    image_snr = float(_scalar(record, "image_snr", location).item())
    central_halpha_snr = float(
        _scalar(record, "central_halpha_snr", location).item()
    )
    for name, value, lower, upper in (
        (
            "image_snr",
            image_snr,
            observation["image_snr_min"],
            observation["image_snr_max"],
        ),
        (
            "central_halpha_snr",
            central_halpha_snr,
            observation["central_halpha_snr_min"],
            observation["central_halpha_snr_max"],
        ),
    ):
        if not np.isfinite(value) or not (
            float(lower) - 1e-5 <= value <= float(upper) + 1e-5
        ):
            raise ValueError(f"{location} has invalid {name}={value}")
    return rmag, halpha, image_snr, central_halpha_snr


def build_observation_levels(image_snr, central_halpha_snr):
    """Validate and return two aligned observation S/N controls."""

    image = torch.as_tensor(image_snr)
    line = torch.as_tensor(
        central_halpha_snr, device=image.device, dtype=image.dtype
    )
    if not image.is_floating_point():
        image = image.to(torch.get_default_dtype())
        line = line.to(image)
    if image.shape != line.shape:
        raise ValueError("image_snr and central_halpha_snr must have matching shapes")
    observation = config.observation
    for name, values, lower, upper in (
        (
            "image_snr",
            image,
            observation["image_snr_min"],
            observation["image_snr_max"],
        ),
        (
            "central_halpha_snr",
            line,
            observation["central_halpha_snr_min"],
            observation["central_halpha_snr_max"],
        ),
    ):
        if bool(
            (~torch.isfinite(values)
             | (values < float(lower))
             | (values > float(upper))).any()
        ):
            raise ValueError(f"{name} must lie within [{lower}, {upper}]")
    return image, line


def draw_uniform_observation_levels(
    count,
    *,
    image_generator,
    central_halpha_generator,
    device,
    dtype=torch.float32,
):
    """Draw independent image and central-H-alpha S/N levels for one epoch."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    observation = config.observation
    if observation["image_snr_distribution"] != "uniform" or (
        observation["central_halpha_snr_distribution"] != "uniform"
    ):
        raise ValueError("epoch-level S/N draws currently require uniform bounds")

    def draw(name, generator):
        lower = float(observation[f"{name}_min"])
        upper = float(observation[f"{name}_max"])
        values = torch.rand(
            count,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return lower + values * (upper - lower)

    return build_observation_levels(
        draw("image_snr", image_generator),
        draw("central_halpha_snr", central_halpha_generator),
    )


def make_ccl_training_batch(image, spectra, targets, positions, context):
    """Concatenate identity and R90 views for one averaged CCL objective."""
    rotated = rotate_90_datavector(
        image, spectra, targets, positions
    )
    image_r, spectra_r, targets_r, positions_r = rotated
    context_r = {
        name: torch.cat((value, value), dim=0)
        for name, value in context.items()
    }
    return (
        torch.cat((image, image_r), dim=0),
        torch.cat((spectra, spectra_r), dim=0),
        torch.cat((targets, targets_r), dim=0),
        torch.cat((positions, positions_r), dim=0),
        context_r,
    )


def make_npe_training_batch(
    image, spectra, targets, positions, context, *, rotate_mask
):
    """Select one identity or R90 view per row for the density objective.

    The NPE loss is additive over observations, so a balanced random view is
    an unbiased estimate of the former identity-plus-R90 batch average without
    evaluating the frozen feature extractor twice. The caller owns the RNG so
    training and fixed validation streams remain reproducible.
    """
    rotate_mask = torch.as_tensor(rotate_mask, device=image.device)
    if rotate_mask.dtype != torch.bool or rotate_mask.shape != (image.shape[0],):
        raise ValueError("rotate_mask must be bool with shape (batch,)")
    image_r, spectra_r, targets_r, positions_r = rotate_90_datavector(
        image, spectra, targets, positions
    )

    def choose(identity, rotated):
        shape = (rotate_mask.shape[0],) + (1,) * (identity.ndim - 1)
        return torch.where(rotate_mask.reshape(shape), rotated, identity)

    return (
        choose(image, image_r),
        choose(spectra, spectra_r),
        choose(targets, targets_r),
        choose(positions, positions_r),
        context,
    )


def _resolve_amp_dtype(value):
    value = str(value).lower()
    if value in ("float16", "fp16", "half"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"unsupported AMP dtype {value!r}")


def _owner(model):
    owner = model.module if hasattr(model, "module") else model
    while hasattr(owner, "_orig_mod"):
        owner = owner._orig_mod
    return owner


def _maybe_compile_model(model, stage_config, logger=None):
    if not stage_config.get("use_compile", False):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "compilation was requested, but torch.compile is unavailable"
        )
    if not hasattr(model, "compile"):
        raise RuntimeError(
            "compilation was requested, but nn.Module.compile is unavailable"
        )
    kwargs = {"mode": stage_config.get("compile_mode", "default")}
    backend = stage_config.get("compile_backend")
    if backend is not None:
        kwargs["backend"] = backend
    try:
        # Compile in place so the module hierarchy, parameter identities, and
        # state-dict names remain unchanged. This is important for the frozen
        # NPE feature extractor loaded from a pretraining snapshot and for the
        # optimizer constructed before compilation.
        model.compile(**kwargs)
    except Exception as exc:
        raise RuntimeError(
            "torch.compile setup failed; refusing to silently run eager"
        ) from exc
    if logger:
        logger.info("Enabled torch.compile with %s", kwargs)
    return model


def _synchronize_pretrain_batch_norm(model, *, train_mode, world_size):
    """Use global batch statistics for multi-GPU CCL pretraining.

    The image backbone contains BatchNorm layers and its running buffers are
    transferred to the frozen NPE feature extractor. Ordinary DDP does not
    reduce those buffers, so leaving them as BatchNorm would make every rank
    validate a different model and would save only rank zero's shard-local
    statistics. SyncBatchNorm keeps the pretraining forward pass and running
    buffers identical across ranks. Its state-dict keys remain compatible with
    the ordinary BatchNorm modules used by single-process inference.
    """

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if train_mode == "pretrain" and world_size > 1:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


class Trainer:
    def __init__(
        self,
        world_size,
        model,
        train_ds,
        valid_ds,
        optimizer,
        gpu_id,
        save_every,
        batch_size,
        *,
        stage_config,
    ):
        self.world_size = int(world_size)
        self.model = model
        self.train_data = train_ds
        self.valid_data = valid_ds
        self.optimizer = optimizer
        self.gpu_id = int(gpu_id)
        self.device = torch.device(f"cuda:{gpu_id}")
        self.log_rank = 0
        self.save_every = int(save_every)
        self.batch_size = int(batch_size)
        self.stage_config = dict(stage_config)
        self.model_name = self.stage_config["model_name"]
        self.base_seed = int(self.stage_config["seed"])
        self.deterministic = bool(self.stage_config["deterministic"])
        self.nfeatures = len(config.TARGET_NAMES)
        self.use_amp = bool(self.stage_config["use_amp"])
        self.amp_dtype = _resolve_amp_dtype(self.stage_config["amp_dtype"])
        self.channels_last = bool(self.stage_config["channels_last"])
        self.fixed_validation_streams = bool(
            self.stage_config["fixed_validation_streams"]
        )
        self.gradient_clip_norm = float(
            self.stage_config.get("gradient_clip_norm", 1.0)
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.use_amp
        )
        self.logger = logging.getLogger(type(self).__name__)
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.preclip_grad_norm_history = []
        self.training_diagnostic_history = []
        self.epoch_diagnostics = {"train": {}, "valid": {}}

    @staticmethod
    def _distributed():
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _generator(self, epoch, stream, *, validation=False):
        effective_epoch = 0 if validation and self.fixed_validation_streams else epoch
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            derive_stream_seed(
                self.base_seed, self.gpu_id, effective_epoch, stream
            )
        )
        return generator

    def _all_ranks_true(self, value):
        flag = torch.tensor(
            [1 if bool(value) else 0], dtype=torch.int32, device=self.device
        )
        if self._distributed():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _global_mean(self, total, count):
        pair = torch.tensor(
            [float(total), float(count)], dtype=torch.float64, device=self.device
        )
        if self._distributed():
            torch.distributed.all_reduce(pair, op=torch.distributed.ReduceOp.SUM)
        return float(pair[0] / pair[1]) if pair[1] > 0 else float("nan")

    @staticmethod
    def _scalar_diagnostics(diagnostics):
        result = {}
        for name, value in diagnostics.items():
            tensor = torch.as_tensor(value).detach()
            if tensor.numel() == 1:
                result[name] = float(tensor.cpu())
        return result

    def _finish_epoch_diagnostics(
        self, split, totals, counts, *, valid_batches, total_batches
    ):
        diagnostics = {
            name: self._global_mean(totals[name], counts[name])
            for name in sorted(totals)
        }
        valid_fraction = self._global_mean(valid_batches, total_batches)
        diagnostics["invalid_batch_fraction"] = 1.0 - valid_fraction
        self.epoch_diagnostics[split] = diagnostics

    def _copy_dataset(self, dataset, split):
        total = len(dataset)
        local = total // self.world_size
        if local * self.world_size != total:
            raise ValueError(f"{split} dataset size must divide world_size")
        start = self.gpu_id * local
        first = dataset[start]
        image = torch.empty(
            (local, *torch.as_tensor(first["img"]).shape),
            dtype=torch.float32,
            device=self.device,
        )
        spectra = torch.empty(
            (local, *torch.as_tensor(first["spec"]).shape),
            dtype=torch.float32,
            device=self.device,
        )
        targets = torch.empty(
            (local, self.nfeatures), dtype=torch.float32, device=self.device
        )
        positions = torch.empty(
            (local, *torch.as_tensor(first["fib_pos"]).shape),
            dtype=torch.float32,
            device=self.device,
        )
        rmag = torch.empty(local, dtype=torch.float32, device=self.device)
        image_snr = torch.empty_like(rmag)
        central_halpha_snr = torch.empty_like(rmag)
        for local_index in range(local):
            record_index = start + local_index
            record = dataset[record_index]
            (
                rmag_value,
                _,
                image_snr_value,
                central_halpha_snr_value,
            ) = validate_observation_record(
                record, location=f"{split} record {record_index}"
            )
            image[local_index] = torch.as_tensor(record["img"], device=self.device)
            spectra[local_index] = torch.as_tensor(record["spec"], device=self.device)
            fid = torch.as_tensor(
                record["fid_pars"], dtype=torch.float32, device=self.device
            ).reshape(-1)
            if fid.shape != (self.nfeatures,):
                raise ValueError(
                    f"{split} record {record_index} has target shape {tuple(fid.shape)}"
                )
            targets[local_index] = fid
            positions[local_index] = torch.as_tensor(
                record["fib_pos"], device=self.device
            )
            rmag[local_index] = rmag_value
            image_snr[local_index] = image_snr_value
            central_halpha_snr[local_index] = central_halpha_snr_value
        return (
            image,
            spectra,
            targets,
            positions,
            rmag,
            image_snr,
            central_halpha_snr,
        )

    def _set_tensors(self):
        (
            self.img_train,
            self.spec_train,
            self.fid_train,
            self.fibpos_train,
            self.rmag_train,
            self.image_snr_train,
            self.central_halpha_snr_train,
        ) = self._copy_dataset(self.train_data, "training")
        (
            self.img_valid,
            self.spec_valid,
            self.fid_valid,
            self.fibpos_valid,
            self.rmag_valid,
            self.image_snr_valid,
            self.central_halpha_snr_valid,
        ) = self._copy_dataset(self.valid_data, "validation")
        self.ntrain = self.img_train.shape[0]
        self.nvalid = self.img_valid.shape[0]
        self.nbatch_train = self.ntrain // self.batch_size
        self.nbatch_valid = self.nvalid // self.batch_size
        if not self.nbatch_train or not self.nbatch_valid:
            raise ValueError("batch size exceeds a local dataset shard")

        self.image_norm_train = image_matched_filter_norm(self.img_train)
        self.image_norm_valid = image_matched_filter_norm(self.img_valid)
        self.central_line_norm_train = central_halpha_line_norm(
            self.spec_train,
            center_fiber_index=config.observation["center_fiber_index"],
        )
        self.central_line_norm_valid = central_halpha_line_norm(
            self.spec_valid,
            center_fiber_index=config.observation["center_fiber_index"],
        )

    def _prepare_epoch(self, epoch):
        (
            self.image_snr_train,
            self.central_halpha_snr_train,
        ) = draw_uniform_observation_levels(
            self.ntrain,
            image_generator=self._generator(epoch, "train_image_snr"),
            central_halpha_generator=self._generator(
                epoch, "train_central_halpha_snr"
            ),
            device=self.device,
            dtype=self.img_train.dtype,
        )
        (
            self.image_snr_valid,
            self.central_halpha_snr_valid,
        ) = draw_uniform_observation_levels(
            self.nvalid,
            image_generator=self._generator(
                epoch, "valid_image_snr", validation=True
            ),
            central_halpha_generator=self._generator(
                epoch, "valid_central_halpha_snr", validation=True
            ),
            device=self.device,
            dtype=self.img_valid.dtype,
        )
        self.train_order = torch.randperm(
            self.ntrain,
            device=self.device,
            generator=self._generator(epoch, "train_order"),
        )
        self.valid_order = torch.randperm(
            self.nvalid,
            device=self.device,
            generator=self._generator(
                epoch, "valid_order", validation=True
            ),
        )
        self.train_img_generator = self._generator(epoch, "train_img_noise")
        self.train_spec_generator = self._generator(epoch, "train_spec_noise")
        self.valid_img_generator = self._generator(
            epoch, "valid_img_noise", validation=True
        )
        self.valid_spec_generator = self._generator(
            epoch, "valid_spec_noise", validation=True
        )
        self.train_npe_view_generator = self._generator(
            epoch, "train_npe_view"
        )
        self.valid_npe_view_generator = self._generator(
            epoch, "valid_npe_view", validation=True
        )

    def _noisy_batch(self, indices, split):
        if split == "train":
            image, spectra = self.img_train[indices], self.spec_train[indices]
            image_snr = self.image_snr_train[indices]
            central_halpha_snr = self.central_halpha_snr_train[indices]
            image_norm = self.image_norm_train[indices]
            central_line_norm = self.central_line_norm_train[indices]
            image_generator, spec_generator = (
                self.train_img_generator,
                self.train_spec_generator,
            )
        else:
            image, spectra = self.img_valid[indices], self.spec_valid[indices]
            image_snr = self.image_snr_valid[indices]
            central_halpha_snr = self.central_halpha_snr_valid[indices]
            image_norm = self.image_norm_valid[indices]
            central_line_norm = self.central_line_norm_valid[indices]
            image_generator, spec_generator = (
                self.valid_img_generator,
                self.valid_spec_generator,
            )
        image = apply_image_noise_for_snr(
            image,
            image_snr,
            clean_norm=image_norm,
            randgen=image_generator,
        )
        spectra = apply_central_halpha_snr_noise(
            spectra,
            central_halpha_snr,
            clean_central_line_norm=central_line_norm,
            center_fiber_index=config.observation["center_fiber_index"],
            center_exposure_s=config.observation["center_exposure_s"],
            offset_exposure_s=config.observation["offset_exposure_s"],
            spectral_units=config.observation["spectral_units"],
            randgen=spec_generator,
            device=self.device,
        )
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spectra = spectra.contiguous(memory_format=torch.channels_last)
        return image, spectra

    def _context(self, indices, split):
        if split == "train":
            rmag = self.rmag_train[indices]
            image_snr = self.image_snr_train[indices]
            central_halpha_snr = self.central_halpha_snr_train[indices]
        else:
            rmag = self.rmag_valid[indices]
            image_snr = self.image_snr_valid[indices]
            central_halpha_snr = self.central_halpha_snr_valid[indices]
        return {
            "rmag_true": rmag,
            "image_snr": image_snr,
            "central_halpha_snr": central_halpha_snr,
        }

    def _step_scheduler(self, valid_loss):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()

    def _save_checkpoint(self, epoch, *, best=False, train_loss=None, valid_loss=None):
        directory = join(self.stage_config["model_path"], self.model_name)
        os.makedirs(directory, exist_ok=True)
        suffix = "best" if best else str(epoch)
        path = join(directory, self.model_name + suffix)
        torch.save(_owner(self.model).state_dict(), path)
        if best:
            metadata = {
                "model_name": self.model_name,
                "epoch": int(epoch + 1),
                "epoch_index": int(epoch),
                "checkpoint_path": path,
                "train_loss": float(train_loss),
                "validation_loss": float(valid_loss),
                "next_epoch_learning_rates": [
                    float(group["lr"]) for group in self.optimizer.param_groups
                ],
            }
            metadata_path = join(directory, "best.json")
            temporary = metadata_path + f".tmp.{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
            os.replace(temporary, metadata_path)

    def train(self, max_epochs):
        self._set_tensors()
        history = []
        patience = self.stage_config.get("early_stopping_patience")
        min_delta = float(self.stage_config.get("early_stopping_min_delta", 0.0))
        for epoch in range(int(max_epochs)):
            epoch_start = time.perf_counter()
            self._prepare_epoch(epoch)
            train_loss = self._train_epoch(epoch)
            if self._distributed():
                torch.distributed.barrier()
            valid_loss = self._valid_epoch(epoch)
            self._step_scheduler(valid_loss)
            row = {"train": train_loss, "valid": valid_loss}
            for split in ("train", "valid"):
                for name, value in self.epoch_diagnostics.get(split, {}).items():
                    row[f"{split}_{name}"] = value
            history.append(row)
            if self.gpu_id == self.log_rank:
                elapsed = _format_elapsed_time(time.perf_counter() - epoch_start)
                lines = [
                    f"Epoch {epoch + 1}/{int(max_epochs)}",
                    f"  elapsed: {elapsed}",
                    "  loss:",
                    f"    train: {train_loss:.8g}",
                    f"    valid: {valid_loss:.8g}",
                ]
                for split in ("train", "valid"):
                    diagnostics = self.epoch_diagnostics.get(split, {})
                    if diagnostics:
                        lines.append(f"  {split} diagnostics:")
                        lines.extend(
                            f"    {name}: {value:.6g}"
                            for name, value in diagnostics.items()
                        )
                self.logger.info("\n".join(lines))
            if self.gpu_id == self.log_rank and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
            improved = np.isfinite(valid_loss) and (
                valid_loss < self.best_validation_loss - min_delta
            )
            if improved:
                self.best_validation_loss = float(valid_loss)
                self.epochs_without_improvement = 0
                if self.gpu_id == self.log_rank:
                    self._save_checkpoint(
                        epoch,
                        best=True,
                        train_loss=train_loss,
                        valid_loss=valid_loss,
                    )
            else:
                self.epochs_without_improvement += 1
            if patience is not None and self.epochs_without_improvement >= int(patience):
                break
        if self.gpu_id == self.log_rank:
            loss_dir = join(self.stage_config["model_path"], "losses")
            os.makedirs(loss_dir, exist_ok=True)
            pd.DataFrame(history).to_csv(
                join(loss_dir, f"losses_{self.model_name}.csv"), index=False
            )


class FETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        kwargs["stage_config"] = config.pretrain
        super().__init__(*args, **kwargs)
        total = int(config.pretrain["epoch_number"])
        warmup = min(5, max(0, total - 1))
        if warmup:
            self.scheduler = SequentialLR(
                self.optimizer,
                [
                    LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup),
                    CosineAnnealingLR(
                        self.optimizer, T_max=max(1, total - warmup), eta_min=1e-6
                    ),
                ],
                milestones=[warmup],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1, eta_min=1e-6)

    def _batch_loss(self, indices, split, *, train):
        image, spectra = self._noisy_batch(indices, split)
        targets = self.fid_train[indices] if split == "train" else self.fid_valid[indices]
        positions = (
            self.fibpos_train[indices]
            if split == "train"
            else self.fibpos_valid[indices]
        )
        context = self._context(indices, split)
        image, spectra, targets, positions, context = make_ccl_training_batch(
            image, spectra, targets, positions, context
        )
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spectra = spectra.contiguous(memory_format=torch.channels_last)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            output = self.model(
                image,
                spectra,
                positions,
                labels=targets,
                observation_context=context,
                return_diagnostics=True,
            )
            loss, diagnostics = output if isinstance(output, tuple) else (output, {})
        self._last_batch_diagnostics = self._scalar_diagnostics(diagnostics)
        valid_loss = self._all_ranks_true(torch.isfinite(loss).item())
        gradients_valid = valid_loss
        if train and valid_loss:
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            norm = nn.utils.clip_grad_norm_(
                _owner(self.model).parameters(),
                1.0,
                error_if_nonfinite=False,
            )
            gradients_valid = self._all_ranks_true(torch.isfinite(norm).item())
            if gradients_valid:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                else:
                    self.optimizer.step()
            if self.use_amp:
                self.scaler.update()
        self._last_batch_valid = gradients_valid
        return loss, diagnostics

    def _run(self, split, *, train):
        order = self.train_order if split == "train" else self.valid_order
        batches = self.nbatch_train if split == "train" else self.nbatch_valid
        total, count = 0.0, 0
        diagnostic_totals, diagnostic_counts = {}, {}
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for index in range(batches):
                ids = order[index * self.batch_size:(index + 1) * self.batch_size]
                loss, _ = self._batch_loss(ids, split, train=train)
                if self._last_batch_valid:
                    total += float(loss.detach())
                    count += 1
                    for name, value in self._last_batch_diagnostics.items():
                        diagnostic_totals.setdefault(name, 0.0)
                        diagnostic_counts.setdefault(name, 0)
                        if np.isfinite(value):
                            diagnostic_totals[name] += value
                            diagnostic_counts[name] += 1
        self._finish_epoch_diagnostics(
            split,
            diagnostic_totals,
            diagnostic_counts,
            valid_batches=count,
            total_batches=batches,
        )
        return self._global_mean(total, count)

    def _train_epoch(self, epoch):
        self.model.train()
        return self._run("train", train=True)

    def _valid_epoch(self, epoch):
        self.model.eval()
        return self._run("valid", train=False)


class NPETrainer(Trainer):
    def __init__(self, *args, **kwargs):
        kwargs["stage_config"] = config.train
        super().__init__(*args, **kwargs)
        scheduler_type = config.train["scheduler_type"]
        total = int(config.train["epoch_number"])
        warmup = int(config.train["warmup_epochs"])
        minimum = float(config.train["min_learning_rate"])
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, "min", factor=0.5, patience=10
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=max(1, total), eta_min=minimum
            )
        elif scheduler_type == "warmup_cosine":
            if not 0 <= warmup < total:
                raise ValueError("warmup_epochs must be smaller than epoch_number")
            if warmup:
                self.scheduler = SequentialLR(
                    self.optimizer,
                    [
                        LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup),
                        CosineAnnealingLR(
                            self.optimizer, T_max=total - warmup, eta_min=minimum
                        ),
                    ],
                    milestones=[warmup],
                )
            else:
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=total, eta_min=minimum
                )
        else:
            raise ValueError(f"unknown scheduler_type {scheduler_type!r}")

    def _batch_loss(self, indices, split, *, train):
        image, spectra = self._noisy_batch(indices, split)
        targets = self.fid_train[indices] if split == "train" else self.fid_valid[indices]
        positions = (
            self.fibpos_train[indices]
            if split == "train"
            else self.fibpos_valid[indices]
        )
        context = self._context(indices, split)
        view_generator = (
            self.train_npe_view_generator
            if split == "train"
            else self.valid_npe_view_generator
        )
        rotate_mask = torch.rand(
            image.shape[0], device=self.device, generator=view_generator
        ) < 0.5
        image, spectra, targets, positions, context = make_npe_training_batch(
            image,
            spectra,
            targets,
            positions,
            context,
            rotate_mask=rotate_mask,
        )
        if self.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
            spectra = spectra.contiguous(memory_format=torch.channels_last)
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss = self.model(
                image,
                spectra,
                targets,
                fiber_positions=positions,
                observation_context=context,
            )
        diagnostics = self._scalar_diagnostics(
            getattr(_owner(self.model), "last_training_diagnostics", {})
        )
        self._last_batch_diagnostics = diagnostics
        valid_loss = self._all_ranks_true(torch.isfinite(loss).item())
        if not train or not valid_loss:
            return loss, valid_loss
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        parameters = [
            parameter
            for parameter in _owner(self.model).parameters()
            if parameter.grad is not None
        ]
        norm = nn.utils.clip_grad_norm_(
            parameters,
            self.gradient_clip_norm,
            error_if_nonfinite=False,
        )
        gradients_valid = self._all_ranks_true(torch.isfinite(norm).item())
        if gradients_valid:
            if self.use_amp:
                self.scaler.step(self.optimizer)
            else:
                self.optimizer.step()
        if self.use_amp:
            self.scaler.update()
        self.preclip_grad_norm_history.append(float(norm.detach()))
        if diagnostics:
            self.training_diagnostic_history.append(diagnostics)
        return loss, gradients_valid

    def _run(self, split, *, train):
        order = self.train_order if split == "train" else self.valid_order
        batches = self.nbatch_train if split == "train" else self.nbatch_valid
        total, count = 0.0, 0
        diagnostic_totals, diagnostic_counts = {}, {}
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for index in range(batches):
                ids = order[index * self.batch_size:(index + 1) * self.batch_size]
                loss, valid = self._batch_loss(ids, split, train=train)
                if valid:
                    total += float(loss.detach())
                    count += 1
                    for name, value in self._last_batch_diagnostics.items():
                        diagnostic_totals.setdefault(name, 0.0)
                        diagnostic_counts.setdefault(name, 0)
                        if np.isfinite(value):
                            diagnostic_totals[name] += value
                            diagnostic_counts[name] += 1
        self._finish_epoch_diagnostics(
            split,
            diagnostic_totals,
            diagnostic_counts,
            valid_batches=count,
            total_batches=batches,
        )
        return self._global_mean(total, count)

    def _train_epoch(self, epoch):
        self.model.train()
        owner = _owner(self.model)
        owner.feature_extractor.eval()
        return self._run("train", train=True)

    def _valid_epoch(self, epoch):
        self.model.eval()
        return self._run("valid", train=False)


def _npe_optimizer_parameters(model, train_config):
    owner = _owner(model)
    flow = owner.flow
    non_theta = [
        parameter
        for parameter in flow.non_theta_flow.parameters()
        if parameter.requires_grad
    ]
    theta = [
        parameter
        for parameter in flow.theta_transform.parameters()
        if parameter.requires_grad
    ]
    branch_ids = {id(value) for value in non_theta + theta}
    remaining = [
        parameter
        for parameter in owner.parameters()
        if parameter.requires_grad and id(parameter) not in branch_ids
    ]
    if len({id(value) for value in remaining + non_theta + theta}) != len(
        remaining + non_theta + theta
    ):
        raise RuntimeError("optimizer groups overlap")
    base_lr = float(train_config["initial_learning_rate"])
    return [
        {"params": remaining, "lr": base_lr, "group_name": "shared"},
        {
            "params": non_theta,
            "lr": float(train_config["non_theta_learning_rate"] or base_lr),
            "group_name": "non_theta_flow",
        },
        {
            "params": theta,
            "lr": float(train_config["theta_learning_rate"] or base_lr),
            "group_name": "theta_transform",
        },
    ]


def load_model(
    model_class,
    *,
    path,
    model_name=None,
    device="cpu",
    networks_root=None,
    strict=True,
    model_kwargs=None,
):
    """Load a current artifact strictly; no live-code or schema fallback."""
    resolved_name = model_name or infer_model_name_from_checkpoint_path(path)
    if not resolved_name:
        raise ValueError("model_name is required for strict artifact loading")
    kwargs = {} if model_kwargs is None else dict(model_kwargs)
    snapshot = load_networks_module_for_model(
        resolved_name, networks_root=networks_root
    ) if networks_root is not None else load_networks_module_for_model(resolved_name)
    snapshot_class = getattr(snapshot, model_class.__name__)
    model = snapshot_class(**kwargs).to(device)
    state = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(state, strict=strict)
    return model


def load_train_objs(
    model_class,
    rank,
    stage_config,
    *,
    train_mode,
    world_size=1,
    epoch=None,
    logger=None,
    device=None,
):
    train_ds = pxt.TorchDataset(config.data["data_dir"])
    valid_ds = pxt.TorchDataset(config.test["data_dir"])
    train_ds = limit_dataset(
        train_ds, config.data["size"], split="training"
    )
    valid_ds = limit_dataset(
        valid_ds, config.test["size"], split="validation"
    )
    if train_mode == "pretrain":
        model = model_class()
    elif train_mode == "train":
        if epoch is None:
            raise ValueError("NPE training requires a pretraining checkpoint")
        checkpoint = join(
            stage_config["model_path"],
            stage_config["pretrained_name"],
            stage_config["pretrained_name"] + str(epoch),
        )
        pretrained = load_model(
            CCLPretrain,
            path=checkpoint,
            model_name=stage_config["pretrained_name"],
            device=device,
            networks_root=join(
                os.path.dirname(
                    os.path.abspath(stage_config["model_path"]).rstrip(os.sep)
                ),
                "networks",
            ),
        )
        model = model_class(feature_extractor=pretrained.backbone)
        for parameter in model.feature_extractor.parameters():
            parameter.requires_grad = False
    else:
        raise ValueError(f"unknown train_mode {train_mode!r}")
    model = _synchronize_pretrain_batch_norm(
        model, train_mode=train_mode, world_size=world_size
    )
    model = model.to(device)
    parameters = (
        model.parameters()
        if train_mode == "pretrain"
        else _npe_optimizer_parameters(model, stage_config)
    )
    kwargs = {
        "lr": stage_config["initial_learning_rate"],
        "weight_decay": stage_config["weight_decay"],
    }
    if train_mode == "pretrain":
        kwargs["eps"] = stage_config["eps"]
    use_fused = bool(stage_config.get("use_fused_adamw")) and torch.cuda.is_available()
    if use_fused:
        kwargs["fused"] = True
    try:
        optimizer = optim.AdamW(parameters, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        optimizer = optim.AdamW(parameters, **kwargs)
    if logger and rank == 0:
        logger.info("Loaded current %s model", train_mode)
    return train_ds, valid_ds, model, optimizer


def limit_dataset(dataset, requested_size, *, split):
    """Use the configured deterministic prefix and reject oversized requests."""
    requested_size = int(requested_size)
    available = len(dataset)
    if requested_size <= 0:
        raise ValueError(f"{split} dataset size must be positive")
    if requested_size > available:
        raise ValueError(
            f"requested {requested_size} {split} rows, but the dataset contains "
            f"only {available}"
        )
    if requested_size == available:
        return dataset
    return Subset(dataset, range(requested_size))


def ddp_setup(rank, world_size):
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    torch.cuda.set_device(rank)
    init_process_group("nccl", rank=rank, world_size=world_size)


def train_nn(
    rank,
    world_size,
    Model=CCLPretrain,
    TrainerClass=FETrainer,
    save_every=1,
    train_mode="pretrain",
    model_config_payload=None,
):
    if model_config_payload is not None:
        config.set_model_config(config.ModelConfig.from_dict(model_config_payload))
    stage = config.pretrain if train_mode == "pretrain" else config.train
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("setup")
    ddp_setup(rank, world_size)
    try:
        seed_everything(stage["seed"], stage["deterministic"])
        device = torch.device(f"cuda:{rank}")
        train_ds, valid_ds, model, optimizer = load_train_objs(
            Model,
            rank,
            stage,
            train_mode=train_mode,
            world_size=world_size,
            epoch=stage.get("pretrain_from"),
            logger=logger,
            device=device,
        )
        if stage["channels_last"]:
            model = model.to(memory_format=torch.channels_last)
        model = _maybe_compile_model(model, stage, logger)
        ddp_kwargs = {
            "device_ids": [rank],
            "find_unused_parameters": stage["ddp_find_unused_parameters"],
            "broadcast_buffers": stage["ddp_broadcast_buffers"],
            "gradient_as_bucket_view": stage["ddp_gradient_as_bucket_view"],
        }
        if stage["ddp_static_graph"]:
            ddp_kwargs["static_graph"] = True
        model = DDP(model, **ddp_kwargs)
        trainer = TrainerClass(
            world_size,
            model,
            train_ds,
            valid_ds,
            optimizer,
            rank,
            save_every,
            stage["batch_size"],
        )
        trainer.train(stage["epoch_number"])
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            destroy_process_group()


def checkpoint_scalar(model, name):
    owner = _owner(model)
    if not hasattr(owner, name):
        raise RuntimeError(f"checkpoint is missing {name}")
    value = getattr(owner, name).detach().reshape(())
    if not bool(torch.isfinite(value) & (value > 0)):
        raise RuntimeError(f"checkpoint has invalid {name}")
    return value


def _seeded_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))


def _sample_bank(value, nfeatures):
    value = torch.as_tensor(value)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2 or value.shape[-1] != nfeatures:
        raise RuntimeError(f"unexpected posterior sample shape {tuple(value.shape)}")
    return value


def sample_density(
    model,
    dataset,
    nsamples,
    *,
    device="cpu",
    matched_group_size=1,
    noise_seed=42,
    spectral_noise_seed=None,
    return_log_prob=True,
    return_observation_metadata=True,
    progress=None,
):
    """Draw the sole current identity/R90 posterior ensemble."""
    if nsamples <= 0 or nsamples % 2:
        raise ValueError("nsamples must be positive and even")
    if matched_group_size not in (1, 5) or len(dataset) % matched_group_size:
        raise ValueError("matched_group_size must be 1 or a divisor-aligned 5")
    if spectral_noise_seed is None:
        spectral_noise_seed = noise_seed + 101
    model.eval()
    nfeatures = len(config.TARGET_NAMES)

    rmag = torch.empty(len(dataset), dtype=torch.float32, device=device)
    halpha = torch.empty_like(rmag)
    image_snr = torch.empty_like(rmag)
    central_halpha_snr = torch.empty_like(rmag)
    truths = torch.empty(
        len(dataset), nfeatures, dtype=torch.float32, device=device
    )
    for index in range(len(dataset)):
        record = dataset[index]
        rmag_i, halpha_i, image_snr_i, central_halpha_snr_i = (
            validate_observation_record(
            record, location=f"analysis record {index}"
            )
        )
        rmag[index], halpha[index] = rmag_i, halpha_i
        image_snr[index] = image_snr_i
        central_halpha_snr[index] = central_halpha_snr_i
        truths[index] = torch.as_tensor(
            record["fid_pars"], dtype=torch.float32, device=device
        )
    grouped_rmag = rmag.reshape(-1, matched_group_size)
    grouped_halpha = halpha.reshape(-1, matched_group_size)
    if not torch.allclose(grouped_rmag, grouped_rmag[:, :1], atol=1e-4, rtol=0):
        raise ValueError("matched groups must share rmag_true")
    if not torch.allclose(grouped_halpha, grouped_halpha[:, :1], atol=0, rtol=2e-6):
        raise ValueError("matched groups must share halpha_flux_true")
    grouped_image_snr = image_snr.reshape(-1, matched_group_size)
    grouped_central_halpha_snr = central_halpha_snr.reshape(
        -1, matched_group_size
    )
    if not torch.allclose(
        grouped_image_snr,
        grouped_image_snr[:, :1],
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("matched groups must share image_snr")
    if not torch.allclose(
        grouped_central_halpha_snr,
        grouped_central_halpha_snr[:, :1],
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("matched groups must share central_halpha_snr")
    if matched_group_size == 5:
        grouped_truth = truths.reshape(-1, 5, nfeatures)
        if not torch.allclose(
            grouped_truth[:, :, 2:],
            grouped_truth[:, :1, 2:].expand_as(grouped_truth[:, :, 2:]),
            atol=2e-6,
            rtol=0,
        ):
            raise ValueError("matched groups must share every non-shear truth")
        shear = grouped_truth[:, :, :2]
        zeros = torch.zeros_like(shear[:, 0])
        valid_stencil = (
            torch.allclose(shear[:, 0], zeros, atol=2e-6, rtol=0)
            and torch.allclose(shear[:, 1, 1], zeros[:, 1], atol=2e-6, rtol=0)
            and torch.allclose(shear[:, 2, 1], zeros[:, 1], atol=2e-6, rtol=0)
            and torch.allclose(shear[:, 3, 0], zeros[:, 0], atol=2e-6, rtol=0)
            and torch.allclose(shear[:, 4, 0], zeros[:, 0], atol=2e-6, rtol=0)
            and torch.all(shear[:, 1, 0] > 0)
            and torch.allclose(shear[:, 2, 0], -shear[:, 1, 0], atol=2e-6, rtol=0)
            and torch.all(shear[:, 3, 1] > 0)
            and torch.allclose(shear[:, 4, 1], -shear[:, 3, 1], atol=2e-6, rtol=0)
        )
        if not bool(valid_stencil):
            raise ValueError(
                "matched groups must follow zero,g1+,g1-,g2+,g2- shear order"
            )
    group_count = len(dataset) // matched_group_size
    base_image_norm = torch.empty(group_count, dtype=torch.float32, device=device)
    base_line_norm = torch.empty_like(base_image_norm)
    for group in range(group_count):
        record = dataset[group * matched_group_size]
        clean_image = torch.as_tensor(
            record["img"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        clean_spectra = torch.as_tensor(
            record["spec"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        base_image_norm[group] = image_matched_filter_norm(clean_image)[0]
        base_line_norm[group] = central_halpha_line_norm(
            clean_spectra,
            center_fiber_index=config.observation["center_fiber_index"],
        )[0]
    group_image_sigma = base_image_norm / grouped_image_snr[:, 0]
    group_center_spectral_sigma = (
        base_line_norm / grouped_central_halpha_snr[:, 0]
    )

    samples, scores = [], []
    iterator = range(len(dataset))
    if progress is not None:
        iterator = progress(iterator, total=len(dataset), desc="Sampling")
    with torch.no_grad():
        for index in iterator:
            group = index // matched_group_size
            image_generator = _seeded_generator(device, noise_seed + group)
            spectrum_generator = _seeded_generator(
                device, spectral_noise_seed + group
            )
            record = dataset[index]
            image = apply_image_noise_for_snr(
                torch.as_tensor(record["img"], dtype=torch.float32, device=device).unsqueeze(0),
                image_snr[index:index + 1],
                clean_norm=base_image_norm[group:group + 1],
                randgen=image_generator,
            )
            spectra = apply_central_halpha_snr_noise(
                torch.as_tensor(record["spec"], dtype=torch.float32, device=device).unsqueeze(0),
                central_halpha_snr[index:index + 1],
                clean_central_line_norm=base_line_norm[group:group + 1],
                center_fiber_index=config.observation["center_fiber_index"],
                center_exposure_s=config.observation["center_exposure_s"],
                offset_exposure_s=config.observation["offset_exposure_s"],
                spectral_units=config.observation["spectral_units"],
                randgen=spectrum_generator,
                device=device,
            )
            positions = torch.as_tensor(
                record["fib_pos"], dtype=torch.float32, device=device
            ).unsqueeze(0)
            context = {
                "rmag_true": rmag[index:index + 1],
                "image_snr": image_snr[index:index + 1],
                "central_halpha_snr": central_halpha_snr[index:index + 1],
            }
            if config.train["channels_last"]:
                image = image.contiguous(memory_format=torch.channels_last)
                spectra = spectra.contiguous(memory_format=torch.channels_last)
            half = nsamples // 2
            original = _sample_bank(
                model.sample(
                    image,
                    spectra,
                    half,
                    fiber_positions=positions,
                    observation_context=context,
                ),
                nfeatures,
            )
            image_r, spectra_r, _, positions_r = rotate_90_datavector(
                image, spectra, fiber_positions=positions
            )
            rotated = _sample_bank(
                model.sample(
                    image_r,
                    spectra_r,
                    half,
                    fiber_positions=positions_r,
                    observation_context=context,
                ),
                nfeatures,
            )
            rotated = rotate_90_parameters(rotated, inverse=True)
            bank = torch.cat((original, rotated), dim=0)
            samples.append(bank.cpu().numpy())
            if return_log_prob:
                log_original = model.posterior_log_prob(
                    image,
                    spectra,
                    bank,
                    fiber_positions=positions,
                    observation_context=context,
                )
                log_rotated = model.posterior_log_prob(
                    image_r,
                    spectra_r,
                    rotate_90_parameters(bank),
                    fiber_positions=positions_r,
                    observation_context=context,
                )
                mixture = torch.logsumexp(
                    torch.stack((log_original, log_rotated)), dim=0
                ) - np.log(2.0)
                scores.append(mixture.detach().cpu().numpy())
    sample_array = np.stack(samples)
    metadata = {
        "truth": truths.cpu().numpy(),
        "rmag_true": rmag.cpu().numpy(),
        "halpha_flux_true": halpha.cpu().numpy(),
        "image_snr": image_snr.cpu().numpy(),
        "central_halpha_snr": central_halpha_snr.cpu().numpy(),
        "image_noise_sigma": torch.repeat_interleave(
            group_image_sigma, matched_group_size
        ).cpu().numpy(),
        "central_spectral_noise_sigma": torch.repeat_interleave(
            group_center_spectral_sigma, matched_group_size
        ).cpu().numpy(),
    }
    if return_log_prob:
        score_array = np.stack(scores)
        if return_observation_metadata:
            return sample_array, score_array, metadata
        return sample_array, score_array
    if return_observation_metadata:
        return sample_array, metadata
    return sample_array
