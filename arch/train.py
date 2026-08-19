import sys,time,os
import json
import random
from os.path import join
import logging
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.optimize import curve_fit

import torch
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader, Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
import pyxis.torch as pxt
from timm.optim.lars import Lars

from networks import *
from dataset import *
from utils import *
from data import (
    TFCalculator,
    _load_rmag_snr_relation,
    apply_views,
    apply_noise,
    apply_fixed_gaussian_image_noise,
    apply_spectral_noise,
    depth_scaled_total_image_flux,
    deterministic_lower_median,
    estimate_spectral_reference_line_norm,
    fixed_image_noise_sigma_from_depth_fluxes,
    gaussian_psf_noise_equivalent_pixels,
    sample_observed_magnitude,
    apply_d4_to_datavector,
    sample_magnitudes,
    app_mag_to_snr,
    snr_to_app_mag,
    rotate_90_degrees
)
import config
from model_registry import infer_model_name_from_checkpoint_path, load_networks_module_for_model

RNG_STREAM_IDS = {
    "ambient": 0,
    "train_order": 1,
    "valid_order": 2,
    "train_snr": 3,
    "valid_snr": 4,
    "train_img_noise": 5,
    "train_spec_noise": 6,
    "valid_img_noise": 7,
    "valid_spec_noise": 8,
    "train_numpy": 9,
    "valid_numpy": 10,
    "train_spec_quality": 11,
    "valid_spec_quality": 12,
    "train_mag_observation": 13,
    "valid_mag_observation": 14,
}

FIBER_LAYOUT_CODES = {
    "image_axis": 0,
    "galaxy_axis": 1,
}
V2_INSTRUMENT_METADATA_FIELDS = (
    "image_band_code",
    "target_line_code",
    "spectral_units_code",
    "center_fiber_index",
    "center_exposure_s",
    "offset_exposure_s",
    "image_reference_psf_fwhm_arcsec",
    "image_pixel_scale_arcsec",
)
V2_IMAGE_BAND_CODES = {"r": 0}
V2_TARGET_LINE_CODES = {"Ha": 0}
V2_SPECTRAL_UNITS_CODES = {"counts": 0}


def validate_v2_observation_record(
    record,
    *,
    observation,
    expected_fiber_layout,
    location,
):
    """Validate and return the latent magnitude from one archived v2 record."""
    required = (
        "rmag_true",
        "halpha_flux_true",
        "observation_model_version",
        "fiber_layout",
        *V2_INSTRUMENT_METADATA_FIELDS,
    )
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(
            f"Observation-model-v2 {location} is missing LMDB metadata: "
            f"{missing}"
        )

    scalars = {}
    for name in required:
        value = torch.as_tensor(record[name])
        if value.numel() != 1:
            raise ValueError(
                f"Observation metadata {name!r} must be scalar in {location}; "
                f"got shape {tuple(value.shape)}"
            )
        scalars[name] = value.reshape(())

    expected_version = int(observation.get("model_version", 1))
    version = int(scalars["observation_model_version"].item())
    if version != expected_version:
        raise ValueError(
            f"Configured observation model v{expected_version} does not "
            f"match {location} (v{version})"
        )
    if expected_fiber_layout not in FIBER_LAYOUT_CODES:
        raise ValueError(
            f"Unsupported configured fiber layout {expected_fiber_layout!r}"
        )
    expected_layout_code = FIBER_LAYOUT_CODES[expected_fiber_layout]
    layout_code = int(scalars["fiber_layout"].item())
    if layout_code != expected_layout_code:
        raise ValueError(
            f"Configured fiber layout {expected_fiber_layout!r} (code "
            f"{expected_layout_code}) does not match {location} "
            f"(code {layout_code})"
        )

    categorical_expectations = {
        "image_band_code": V2_IMAGE_BAND_CODES.get(observation["image_band"]),
        "target_line_code": V2_TARGET_LINE_CODES.get(observation["target_line"]),
        "spectral_units_code": V2_SPECTRAL_UNITS_CODES.get(
            observation["spectral_units"]
        ),
        "center_fiber_index": int(observation["center_fiber_index"]),
    }
    unsupported = [
        name for name, value in categorical_expectations.items()
        if value is None
    ]
    if unsupported:
        raise ValueError(
            "Configured v2 instrument schema is unsupported for "
            + ", ".join(unsupported)
        )
    for name, expected in categorical_expectations.items():
        actual = int(scalars[name].item())
        if actual != expected:
            raise ValueError(
                f"Configured {name}={expected} does not match {location} "
                f"({actual})"
            )

    continuous_expectations = {
        "center_exposure_s": float(observation["center_exposure_s"]),
        "offset_exposure_s": float(observation["offset_exposure_s"]),
        "image_reference_psf_fwhm_arcsec": float(
            observation["image_reference_psf_fwhm_arcsec"]
        ),
        "image_pixel_scale_arcsec": float(
            observation["image_pixel_scale_arcsec"]
        ),
    }
    for name, expected in continuous_expectations.items():
        actual = float(scalars[name].item())
        if (
            not np.isfinite(actual)
            or not np.isclose(actual, expected, rtol=1e-6, atol=1e-6)
        ):
            raise ValueError(
                f"Configured {name}={expected} does not match {location} "
                f"({actual})"
            )

    rmag_true = float(scalars["rmag_true"].item())
    rmag_min = float(observation["rmag_min"])
    rmag_max = float(observation["rmag_max"])
    if not np.isfinite(rmag_true):
        raise ValueError(f"Observation-model-v2 {location} has non-finite rmag_true")
    tolerance = 1e-4
    if rmag_true < rmag_min - tolerance or rmag_true > rmag_max + tolerance:
        raise ValueError(
            f"Observation-model-v2 {location} has rmag_true={rmag_true}, "
            f"outside configured [{rmag_min}, {rmag_max}]"
        )
    halpha_flux_true = float(scalars["halpha_flux_true"].item())
    halpha_flux_min = float(observation["halpha_flux_min"])
    halpha_flux_max = float(observation["halpha_flux_max"])
    flux_tolerance = 2e-6 * halpha_flux_max
    if not np.isfinite(halpha_flux_true):
        raise ValueError(
            f"Observation-model-v2 {location} has non-finite halpha_flux_true"
        )
    if (
        halpha_flux_true < halpha_flux_min - flux_tolerance
        or halpha_flux_true > halpha_flux_max + flux_tolerance
    ):
        raise ValueError(
            f"Observation-model-v2 {location} has "
            f"halpha_flux_true={halpha_flux_true}, outside configured "
            f"[{halpha_flux_min}, {halpha_flux_max}]"
        )
    return rmag_true


def build_v2_observation_levels(
    rmag_true,
    *,
    image_band="r",
    image_depth_5sigma_mag=23.4,
    spectral_quality_min=3.0,
    spectral_quality_max=100.0,
    spectral_quality_distribution="log_uniform",
    spectral_generator=None,
):
    """Build independent image and spectral quality levels for simulator v2."""
    rmag_true = torch.as_tensor(rmag_true)
    if not rmag_true.is_floating_point():
        rmag_true = rmag_true.to(torch.get_default_dtype())
    if bool((~torch.isfinite(rmag_true)).any()):
        raise ValueError("rmag_true must contain finite values")
    quality_min = float(spectral_quality_min)
    quality_max = float(spectral_quality_max)
    if not np.isfinite(quality_min) or not np.isfinite(quality_max):
        raise ValueError("spectral-quality bounds must be finite")
    if quality_min <= 0 or quality_min >= quality_max:
        raise ValueError("spectral-quality bounds must be positive and increasing")
    if spectral_quality_distribution not in ("log_uniform", "uniform"):
        raise ValueError("unsupported spectral-quality distribution")

    image_snr = app_mag_to_snr(
        rmag_true,
        band=image_band,
        depth_5sigma_mag=image_depth_5sigma_mag,
    )
    unit_draw = torch.rand(
        rmag_true.shape,
        device=rmag_true.device,
        dtype=rmag_true.dtype,
        generator=spectral_generator,
    )
    if spectral_quality_distribution == "log_uniform":
        log_min = np.log10(quality_min)
        log_max = np.log10(quality_max)
        spectral_quality = 10 ** (unit_draw * (log_max - log_min) + log_min)
    else:
        spectral_quality = unit_draw * (quality_max - quality_min) + quality_min
    return image_snr, spectral_quality


def derive_stream_seed(base_seed, rank=0, epoch=0, stream="ambient"):
    """Derive a stable seed without relying on randomized Python hashes."""
    if stream not in RNG_STREAM_IDS:
        raise ValueError(f"Unknown RNG stream '{stream}'")
    modulus = 2**63 - 1
    return int(
        (
            int(base_seed)
            + 1_000_003 * int(rank)
            + 10_007 * int(epoch)
            + 97 * RNG_STREAM_IDS[stream]
        )
        % modulus
    )


def seed_everything(seed, deterministic=False):
    """Seed ambient Python, NumPy, Torch CPU, and Torch CUDA RNGs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))


def make_ccl_pretrain_views(
    img,
    spec,
    fid,
    fp,
    *,
    use_rot90_counterpart,
):
    """Return the canonical CCL training branches for one noisy batch."""
    original = (img, spec, fid, fp)
    if not use_rot90_counterpart:
        return (original,)
    img_90, fid_90, fp_90 = rotate_90_degrees(img, fid, fp)
    rotated = (
        img_90,
        spec,
        fid_90.contiguous(),
        fp_90.contiguous(),
    )
    return original, rotated


def make_npe_training_batch(
    img,
    spec,
    fid,
    fp,
    snr,
    *,
    use_rot90_counterpart,
):
    """Build the legacy NPE batch or keep the exact-D4 batch unduplicated."""
    if not use_rot90_counterpart:
        return img, spec, fid, fp, snr
    img_90, spec_90, fid_90, fp_90 = apply_d4_to_datavector(
        img,
        spec,
        fid,
        fp,
        element="r90",
    )
    return (
        torch.cat((img, img_90), dim=0),
        torch.cat((spec, spec_90), dim=0),
        torch.cat((fid, fid_90), dim=0),
        torch.cat((fp, fp_90), dim=0),
        torch.cat((snr, snr), dim=0) if snr is not None else None,
    )


def _resolve_amp_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in ("float16", "fp16", "half"):
        return torch.float16
    if normalized in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype '{value}'. Use 'float16' or 'bfloat16'.")

def _patch_networkx_entry_points() -> None:
    try:
        import importlib.metadata as importlib_metadata
    except Exception:
        return
    entry_points = getattr(importlib_metadata, "entry_points", None)
    if entry_points is None or getattr(entry_points, "_kl_nn_filtered", False):
        return

    def _filtered_entry_points(*args, **kwargs):
        eps = entry_points(*args, **kwargs)
        group = kwargs.get("group")
        if group in ("networkx.backends", "networkx.backend_info"):
            try:
                return [ep for ep in eps if ep.name != "nx-loopback"]
            except TypeError:
                return eps
        return eps

    _filtered_entry_points._kl_nn_filtered = True  # type: ignore[attr-defined]
    importlib_metadata.entry_points = _filtered_entry_points


def _maybe_compile_model(
    model: torch.nn.Module,
    log: logging.Logger | None = None,
    use_compile: bool | None = None,
    compile_mode: str | None = None,
    compile_backend: str | None = None,
) -> torch.nn.Module:
    if log is not None:
        log.info(type(model))
    if use_compile is None:
        use_compile = bool(config.train.get('use_compile', False))
    if not use_compile:
        return model
    if not hasattr(torch, "compile"):
        if log is not None:
            log.warning("torch.compile is not available in this Torch build; skipping.")
        return model
    try:
        _patch_networkx_entry_points()
        mode = compile_mode if compile_mode is not None else config.train.get('compile_mode', 'default')
        backend = compile_backend if compile_backend is not None else config.train.get('compile_backend', 'inductor')
        if backend is None:
            return torch.compile(model, mode=mode)
        return torch.compile(model, mode=mode, backend=backend)
    except Exception as exc:
        if log is not None:
            log.warning("torch.compile failed; continuing without compilation: %s", exc)
        return model


def _maybe_strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict.keys()):
        return state_dict
    stripped = {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }
    model_keys = set(model.state_dict().keys())
    raw_matches = sum(1 for key in state_dict.keys() if key in model_keys)
    stripped_matches = sum(1 for key in stripped.keys() if key in model_keys)
    return stripped if stripped_matches >= raw_matches else state_dict

#--------------------#
# Trainer Base Class #
#--------------------#

class Trainer:
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
        *,
        seed: int | None = None,
        deterministic: bool | None = None,
    ) -> None:
        
        self.model_name = config.train['model_name']
        self.world_size = world_size
        self.gpu_id = gpu_id
        self.log_rank = 0
        self.device = torch.device(f"cuda:{gpu_id}")
        self.train_data = train_ds
        self.valid_data = valid_ds
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = model
        self.batch_size = batch_size
        self.base_seed = int(config.train.get('seed', 20260810) if seed is None else seed)
        self.deterministic = bool(
            config.train.get('deterministic', False)
            if deterministic is None
            else deterministic
        )
        self.ntrain = len(train_ds)//world_size
        self.nvalid = len(valid_ds)//world_size
        self.nbatch_train = self.ntrain//self.batch_size
        self.nbatch_valid = self.nvalid//self.batch_size
        self.nfeatures = config.train['feature_number']
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=10)
        self.logger = logging.getLogger('Trainer')
        self.use_amp = bool(config.train.get('use_amp', False)) and torch.cuda.is_available()
        self.amp_dtype = _resolve_amp_dtype(config.train.get('amp_dtype', 'float16'))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_channels_last = bool(config.train.get('channels_last', False))
        self.observation_model_version = int(
            config.observation.get("model_version", 1)
        )
        self.expected_fiber_layout = str(
            config.observation.get("fiber_layout", "galaxy_axis")
        )
        if self.expected_fiber_layout not in FIBER_LAYOUT_CODES:
            raise ValueError(
                f"Unsupported configured fiber layout {self.expected_fiber_layout!r}"
            )
        self.use_noise_cache_maxs = bool(config.train.get('noise_cache_maxs', False))
        self.fixed_validation_streams = bool(
            config.train.get('fixed_validation_streams', False)
        )
        self.gradient_clip_norm = config.train.get('gradient_clip_norm', 1.0)
        if self.gradient_clip_norm is not None:
            self.gradient_clip_norm = float(self.gradient_clip_norm)
        # NPETrainer enables these controls explicitly. Keeping them disabled in
        # the base class preserves the historical feature-pretraining loop.
        self.enable_best_checkpoint = False
        self.early_stopping_patience = None
        self.early_stopping_min_delta = 0.0
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.preclip_grad_norm_history = []
        self.training_diagnostic_history = []

    @staticmethod
    def _distributed_is_initialized():
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    def _noise_cache_for_batch(self, cache, batch_ids):
        """Select a legacy max cache when it exists.

        Observation model v2 uses fixed image noise and independent spectral
        noise, so it intentionally does not allocate the legacy per-object
        maximum caches even when an archived pretraining config leaves
        ``noise_cache_maxs`` enabled.
        """
        if not self.use_noise_cache_maxs or cache is None:
            return None
        return cache[batch_ids]

    def _all_ranks_true(self, value):
        """Return ``True`` only when every distributed rank reports true."""
        flag = torch.tensor(
            [1 if bool(value) else 0],
            dtype=torch.int32,
            device=self.device,
        )
        if self._distributed_is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _global_mean_from_sum_count(self, local_sum, local_count):
        """Reduce a metric as a float64 sum/count pair and return its mean."""
        totals = torch.tensor(
            [float(local_sum), float(local_count)],
            dtype=torch.float64,
            device=self.device,
        )
        if self._distributed_is_initialized():
            torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        total_sum, total_count = totals.tolist()
        if total_count == 0:
            return float("nan")
        return total_sum / total_count

    def _global_max(self, value):
        reduced = torch.tensor(
            [float(value)], dtype=torch.float64, device=self.device
        )
        if self._distributed_is_initialized():
            torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.MAX)
        return float(reduced.item())

    def _global_min(self, value):
        reduced = torch.tensor(
            [float(value)], dtype=torch.float64, device=self.device
        )
        if self._distributed_is_initialized():
            torch.distributed.all_reduce(
                reduced, op=torch.distributed.ReduceOp.MIN
            )
        return float(reduced.item())

    def _synchronize_counter(self, value):
        """Synchronize counters which already represent global batch decisions."""
        return int(self._global_max(value))

    def _make_epoch_generator(self, epoch, stream):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(
            derive_stream_seed(self.base_seed, self.gpu_id, epoch, stream)
        )
        return generator

    def _reset_epoch_rngs(self, epoch):
        """Reset independent rank/epoch streams used by the training loop."""
        seed_everything(
            derive_stream_seed(
                self.base_seed,
                rank=self.gpu_id,
                epoch=epoch,
                stream="ambient",
            ),
            deterministic=self.deterministic,
        )
        streams = (
            "train_order",
            "valid_order",
            "train_snr",
            "valid_snr",
            "train_spec_quality",
            "valid_spec_quality",
            "train_mag_observation",
            "valid_mag_observation",
            "train_img_noise",
            "train_spec_noise",
            "valid_img_noise",
            "valid_spec_noise",
        )
        self.epoch_generators = {}
        for stream in streams:
            stream_epoch = (
                0
                if self.fixed_validation_streams and stream.startswith("valid_")
                else epoch
            )
            self.epoch_generators[stream] = self._make_epoch_generator(
                stream_epoch, stream
            )
        self.epoch_numpy_generators = {
            split: np.random.default_rng(
                derive_stream_seed(
                    self.base_seed,
                    rank=self.gpu_id,
                    epoch=(
                        0
                        if self.fixed_validation_streams and split == "valid"
                        else epoch
                    ),
                    stream=f"{split}_numpy",
                )
            )
            for split in ("train", "valid")
        }

    def _load_v2_record_metadata(self, record, *, split, record_index):
        observation = dict(config.observation)
        observation["model_version"] = self.observation_model_version
        return validate_v2_observation_record(
            record,
            observation=observation,
            expected_fiber_layout=self.expected_fiber_layout,
            location=f"{split} record {record_index}",
        )
    
    def _set_tensors(self):
        '''
        Put dataset arrays on GPU for direct access in training
        '''
        # Initialize large arrays on GPU
        if self.gpu_id == self.log_rank:
            self.logger.info("Setting up tensors on GPU")
        img_format = torch.channels_last if self.use_channels_last else torch.contiguous_format
        self.img_train = torch.empty(
            (self.ntrain, 1, 48, 48),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.img_valid = torch.empty(
            (self.nvalid, 1, 48, 48),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.spec_train = torch.empty(
            (self.ntrain, 1, 5, 64),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.spec_valid = torch.empty(
            (self.nvalid, 1, 5, 64),
            dtype=torch.float,
            device=self.device,
            memory_format=img_format,
        )
        self.fid_train = torch.empty((self.ntrain, self.nfeatures), dtype=torch.float, device=self.device)
        self.fid_valid = torch.empty((self.nvalid, self.nfeatures), dtype=torch.float, device=self.device)
        self.fibpos_train = torch.empty((self.ntrain, 5, 2), dtype=torch.float, device=self.device)
        self.fibpos_valid = torch.empty((self.nvalid, 5, 2), dtype=torch.float, device=self.device)
        if self.observation_model_version == 2:
            self.rmag_train = torch.empty(
                self.ntrain, dtype=torch.float, device=self.device
            )
            self.rmag_valid = torch.empty(
                self.nvalid, dtype=torch.float, device=self.device
            )
        else:
            self.rmag_train = None
            self.rmag_valid = None
        
        # Fill arrays with values
        start = self.gpu_id*self.ntrain
        if self.gpu_id == self.log_rank:
            self.logger.info("Uploading training set to GPU...")
        prev_prog = 0
        for i in range(self.ntrain):
            i_db = start+i
            record = self.train_data[i_db]
            self.img_train[i] = record['img']
            self.spec_train[i] = record['spec']
            self.fid_train[i] = record['fid_pars'][:self.nfeatures]
            self.fibpos_train[i] = record['fib_pos']
            if self.observation_model_version == 2:
                self.rmag_train[i] = self._load_v2_record_metadata(
                    record, split="training", record_index=i_db
                )

            prog = 100*i//self.ntrain
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == self.log_rank:
                prev_prog = prog
                self.logger.info(f"{prog}% complete")
        
        start = self.gpu_id*self.nvalid
        if self.gpu_id == self.log_rank:
            self.logger.info("Uploading validation set to GPU...")
        prev_prog = 0
        for i in range(self.nvalid):
            i_db = start+i
            record = self.valid_data[i_db]
            self.img_valid[i] = record['img']
            self.spec_valid[i] = record['spec']
            self.fid_valid[i] = record['fid_pars'][:self.nfeatures]
            self.fibpos_valid[i] = record['fib_pos']
            if self.observation_model_version == 2:
                self.rmag_valid[i] = self._load_v2_record_metadata(
                    record, split="validation", record_index=i_db
                )

            prog = 100*i//self.nvalid
            if prog % 10 == 0 and prog > prev_prog and self.gpu_id == self.log_rank:
                prev_prog = prog
                self.logger.info(f"{prog}% complete")

        self.image_noise_sigma = None
        self.spectral_reference_line_norm = None
        if self.observation_model_version == 2:
            local_depth_fluxes = depth_scaled_total_image_flux(
                self.img_train,
                self.rmag_train,
                config.observation["image_depth_5sigma_mag"],
            )
            if self._distributed_is_initialized():
                gathered_depth_fluxes = [
                    torch.empty_like(local_depth_fluxes)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(
                    gathered_depth_fluxes, local_depth_fluxes
                )
                global_depth_fluxes = torch.cat(gathered_depth_fluxes)
            else:
                global_depth_fluxes = local_depth_fluxes
            reference_noise_pixels = gaussian_psf_noise_equivalent_pixels(
                config.observation["image_reference_psf_fwhm_arcsec"],
                config.observation["image_pixel_scale_arcsec"],
            )
            self.image_noise_sigma = (
                fixed_image_noise_sigma_from_depth_fluxes(
                    global_depth_fluxes, reference_noise_pixels
                ).reshape(())
            )

            local_reference = estimate_spectral_reference_line_norm(
                self.spec_train,
                center_fiber_index=int(config.observation["center_fiber_index"]),
            ).reshape(1)
            if self._distributed_is_initialized():
                gathered = [
                    torch.empty_like(local_reference)
                    for _ in range(torch.distributed.get_world_size())
                ]
                torch.distributed.all_gather(gathered, local_reference)
                local_reference = deterministic_lower_median(
                    torch.cat(gathered)
                ).reshape(1)
            self.spectral_reference_line_norm = local_reference.reshape(())
            model_owner = self.model.module
            while hasattr(model_owner, "_orig_mod"):
                model_owner = model_owner._orig_mod
            checkpoint_values = {
                "image_noise_sigma": self.image_noise_sigma,
                "spectral_reference_line_norm": (
                    self.spectral_reference_line_norm
                ),
            }
            for buffer_name, value in checkpoint_values.items():
                if hasattr(model_owner, buffer_name):
                    buffer = getattr(model_owner, buffer_name)
                    with torch.no_grad():
                        buffer.copy_(
                            value.to(device=buffer.device, dtype=buffer.dtype)
                        )
                elif hasattr(model_owner, "_prepare_observation_context"):
                    raise RuntimeError(
                        "Observation-model-v2 NPE is missing its archived "
                        f"{buffer_name} buffer"
                    )
            if self.gpu_id == self.log_rank:
                self.logger.info(
                    "Observation v2 fixed image noise sigma: %.8g "
                    "(Gaussian-equivalent N_eff=%.6g)",
                    float(self.image_noise_sigma.item()),
                    reference_noise_pixels,
                )
                self.logger.info(
                    "Observation v2 spectral reference line norm: %.8g",
                    float(self.spectral_reference_line_norm.item()),
                )

        self.img_train_maxs = None
        self.img_valid_maxs = None
        self.spec_train_maxs = None
        self.spec_valid_maxs = None
        if self.use_noise_cache_maxs and self.observation_model_version != 2:
            if self.gpu_id == self.log_rank:
                self.logger.info("Precomputing noise max caches")
            self.img_train_maxs = torch.amax(self.img_train, dim=(-1, -2, -3))
            self.img_valid_maxs = torch.amax(self.img_valid, dim=(-1, -2, -3))
            self.spec_train_maxs = torch.amax(self.spec_train, dim=(-1, -2, -3))
            self.spec_valid_maxs = torch.amax(self.spec_valid, dim=(-1, -2, -3))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            
    def _apply_noise(self, data, snr, maxs=None, randgen=None):
        if snr is None:
            return data
        if self.observation_model_version == 2:
            if self.image_noise_sigma is None:
                raise RuntimeError(
                    "Observation model v2 fixed image noise sigma is unset"
                )
            output = apply_fixed_gaussian_image_noise(
                data, self.image_noise_sigma, randgen=randgen
            )
        else:
            output = apply_noise(
                data,
                snr,
                device=self.device,
                use_iterative=True,
                maxs=maxs,
                randgen=randgen,
            )
        if self.use_channels_last:
            output = output.contiguous(memory_format=torch.channels_last)
        return output

    def _apply_spectrum_noise(
        self,
        data,
        image_snr,
        *,
        spectral_quality=None,
        maxs=None,
        randgen=None,
    ):
        if self.observation_model_version != 2:
            return self._apply_noise(
                data, image_snr, maxs=maxs, randgen=randgen
            )
        if spectral_quality is None:
            raise ValueError("Observation model v2 requires spectral quality")
        if self.spectral_reference_line_norm is None:
            raise RuntimeError("Observation model v2 spectral reference is unset")
        output = apply_spectral_noise(
            data,
            spectral_quality,
            self.spectral_reference_line_norm,
            center_fiber_index=int(config.observation["center_fiber_index"]),
            center_exposure_s=float(config.observation["center_exposure_s"]),
            offset_exposure_s=float(config.observation["offset_exposure_s"]),
            spectral_units=config.observation["spectral_units"],
            randgen=randgen,
            device=self.device,
        )
        if self.use_channels_last:
            output = output.contiguous(memory_format=torch.channels_last)
        return output

    def _observation_context_for_batch(
        self, batch_ids, *, split, duplicate=False
    ):
        """Return observed, D4-invariant v2 scalars for one density batch."""
        if self.observation_model_version != 2:
            return None
        if split not in ("train", "valid"):
            raise ValueError("split must be 'train' or 'valid'")
        suffix = "train" if split == "train" else "valid"
        source_names = {
            "rmag_obs": f"RMAG_OBS_{suffix}",
            "rmag_sigma": f"RMAG_SIGMA_{suffix}",
            "image_snr": f"IMAGE_SNR_OBS_{suffix}",
            "spectral_reference_quality": f"SPEC_QUALITY_{suffix}",
            "spectral_noise_scale": f"SPEC_NOISE_SCALE_{suffix}",
        }
        context = {
            name: getattr(self, attribute)[batch_ids]
            for name, attribute in source_names.items()
        }
        if duplicate:
            context = {
                name: torch.cat((value, value), dim=0)
                for name, value in context.items()
            }
        return context
    
    def _run_epoch(self, epoch, show_log=True):
        self._reset_epoch_rngs(epoch)

        if self.gpu_id == self.log_rank:
            self.logger.info(f'Starting epoch {epoch+1}')
            
        # Generate the permutation arrays BEFORE building the corresponding SNRs
        self.train_order = torch.randperm(
            self.ntrain,
            device=self.device,
            generator=self.epoch_generators["train_order"],
        )
        self.valid_order = torch.randperm(
            self.nvalid,
            device=self.device,
            generator=self.epoch_generators["valid_order"],
        )
            
        if self.observation_model_version == 2:
            observation = config.observation
            level_kwargs = {
                "image_band": observation["image_band"],
                "image_depth_5sigma_mag": observation["image_depth_5sigma_mag"],
                "spectral_quality_min": observation["spectral_quality_min"],
                "spectral_quality_max": observation["spectral_quality_max"],
                "spectral_quality_distribution": observation[
                    "spectral_quality_distribution"
                ],
            }
            self.SNR_train, self.SPEC_QUALITY_train = build_v2_observation_levels(
                self.rmag_train,
                spectral_generator=self.epoch_generators["train_spec_quality"],
                **level_kwargs,
            )
            self.SNR_valid, self.SPEC_QUALITY_valid = build_v2_observation_levels(
                self.rmag_valid,
                spectral_generator=self.epoch_generators["valid_spec_quality"],
                **level_kwargs,
            )
            train_mag_observation = sample_observed_magnitude(
                self.rmag_train,
                self.SNR_train,
                randgen=self.epoch_generators["train_mag_observation"],
            )
            valid_mag_observation = sample_observed_magnitude(
                self.rmag_valid,
                self.SNR_valid,
                randgen=self.epoch_generators["valid_mag_observation"],
            )
            self.RMAG_OBS_train = train_mag_observation["rmag_obs"]
            self.RMAG_SIGMA_train = train_mag_observation["rmag_sigma"]
            self.IMAGE_SNR_OBS_train = train_mag_observation[
                "image_flux_snr"
            ]
            self.RMAG_OBS_valid = valid_mag_observation["rmag_obs"]
            self.RMAG_SIGMA_valid = valid_mag_observation["rmag_sigma"]
            self.IMAGE_SNR_OBS_valid = valid_mag_observation[
                "image_flux_snr"
            ]
            self.SPEC_NOISE_SCALE_train = (
                self.spectral_reference_line_norm / self.SPEC_QUALITY_train
            )
            self.SPEC_NOISE_SCALE_valid = (
                self.spectral_reference_line_norm / self.SPEC_QUALITY_valid
            )
            if self.gpu_id == self.log_rank:
                self.logger.info(
                    "Observation v2 epoch %d: target image SNR %.5g--%.5g; "
                    "observed flux SNR %.5g--%.5g; independent spectral "
                    "quality %.5g--%.5g",
                    epoch + 1,
                    float(self.SNR_train.min().item()),
                    float(self.SNR_train.max().item()),
                    float(self.IMAGE_SNR_OBS_train.min().item()),
                    float(self.IMAGE_SNR_OBS_train.max().item()),
                    float(self.SPEC_QUALITY_train.min().item()),
                    float(self.SPEC_QUALITY_train.max().item()),
                )
        else:
            self.SNR_train = self.generate_snr(
                size=self.ntrain,
                mode='log_uniform',
                generator=self.epoch_generators["train_snr"],
                np_rng=self.epoch_numpy_generators["train"],
            )
            self.SNR_valid = self.generate_snr(
                size=self.nvalid,
                mode='log_uniform',
                generator=self.epoch_generators["valid_snr"],
                np_rng=self.epoch_numpy_generators["valid"],
            )
            self.SPEC_QUALITY_train = None
            self.SPEC_QUALITY_valid = None
            self.RMAG_OBS_train = None
            self.RMAG_SIGMA_train = None
            self.IMAGE_SNR_OBS_train = None
            self.RMAG_OBS_valid = None
            self.RMAG_SIGMA_valid = None
            self.IMAGE_SNR_OBS_valid = None
            self.SPEC_NOISE_SCALE_train = None
            self.SPEC_NOISE_SCALE_valid = None
            if self.gpu_id == self.log_rank:
                self.logger.info(f'Randomized SNR and noise for epoch {epoch+1}')

        train_loss = self._trainFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        valid_loss = self._validFunc(epoch)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return train_loss, valid_loss
    
    def _trainFunc(self, epoch, show_log=True):
        raise NotImplementedError("Subclasses must implement _trainFunc")
    
    def _validFunc(self, epoch, show_log=True):
        raise NotImplementedError("Subclasses must implement _validFunc")

    def _save_checkpoint(self, epoch):
        raise NotImplementedError("Subclasses must implement _save_checkpoint")

    def _save_best_checkpoint(self, epoch, train_loss, valid_loss):
        raise NotImplementedError(
            "Subclasses which enable best checkpoints must implement "
            "_save_best_checkpoint"
        )

    def _update_training_control(self, valid_loss):
        """Synchronize best-metric and early-stopping state across all ranks."""
        distributed = self._distributed_is_initialized()
        controller = (not distributed) or self.gpu_id == self.log_rank
        if controller:
            improved = (
                np.isfinite(valid_loss)
                and valid_loss
                < self.best_validation_loss - self.early_stopping_min_delta
            )
            if improved:
                self.best_validation_loss = float(valid_loss)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            should_stop = (
                self.early_stopping_patience is not None
                and self.epochs_without_improvement
                >= self.early_stopping_patience
            )
            state_values = (
                float(improved),
                float(should_stop),
                self.best_validation_loss,
                float(self.epochs_without_improvement),
            )
        else:
            state_values = (0.0, 0.0, float("inf"), 0.0)

        state = torch.tensor(
            state_values, dtype=torch.float64, device=self.device
        )
        if distributed:
            torch.distributed.broadcast(state, src=self.log_rank)
        improved = bool(state[0].item())
        should_stop = bool(state[1].item())
        self.best_validation_loss = float(state[2].item())
        self.epochs_without_improvement = int(state[3].item())
        return improved, should_stop

    def _step_scheduler(self, valid_loss):
        if isinstance(self.scheduler, ReduceLROnPlateau):
            # NPE validation loss is a globally reduced metric, so every rank
            # advances the plateau scheduler from the same observation.
            self.scheduler.step(valid_loss)
        else:
            self.scheduler.step()
    
    def train(self, max_epochs: int):
        self._set_tensors()
        train_losses = []
        valid_losses = []
        if self.gpu_id == self.log_rank:
            self.logger.info("Training start")
        for epoch in range(max_epochs):
            train_loss, valid_loss = self._run_epoch(epoch)
            self._step_scheduler(valid_loss)
            if self.gpu_id == self.log_rank:
                self.logger.info(f"Current LR is {self.scheduler.get_last_lr()}")
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            if self.gpu_id == self.log_rank and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

            should_stop = False
            if self.enable_best_checkpoint:
                improved, should_stop = self._update_training_control(valid_loss)
                if improved and self.gpu_id == self.log_rank:
                    self._save_best_checkpoint(epoch, train_loss, valid_loss)
            if should_stop:
                if self.gpu_id == self.log_rank:
                    self.logger.info(
                        "Early stopping after epoch %d: validation loss did not "
                        "improve by %.6g for %d epochs. Best loss: %.8g",
                        epoch + 1,
                        self.early_stopping_min_delta,
                        self.epochs_without_improvement,
                        self.best_validation_loss,
                    )
                break

        if self.gpu_id == self.log_rank:
            losses = pd.DataFrame(np.vstack([train_losses, valid_losses]))
            losses_dir = join(config.train['model_path'], 'losses')
            os.makedirs(losses_dir, exist_ok=True)
            losses.to_csv(join(losses_dir, f'losses_{self.model_name}.csv'), index=False)
            self.logger.info("Loss history saved successfully. Training cycle complete.")
    
    def generate_snr(
        self,
        size,
        mode='uniform',
        generator=None,
        np_rng=None,
        **kwargs,
    ):
        if mode == 'none':
            return None
        if mode == 'uniform':
            min_snr = kwargs.get('min', 5)
            max_snr = kwargs.get('max', 1000)
            return torch.rand(
                (size,),
                device=self.device,
                generator=generator,
            ) * (max_snr - min_snr) + min_snr
        elif mode == 'log_uniform':
            min_log_snr = kwargs.get('min', 0)
            max_log_snr = kwargs.get('max', 4)
            log_snr = torch.rand(
                (size,),
                device=self.device,
                generator=generator,
            ) * (max_log_snr - min_log_snr) + min_log_snr
            return 10**log_snr
        elif mode == 'tf':
            tf_calc = TFCalculator(slope=config.tf['slope'], intercept=config.tf['intercept'], scatter=config.tf['scatter'])
            if size == self.ntrain:
                vcirc_norm = self.fid_train[:, 5]
            elif size == self.nvalid:
                vcirc_norm = self.fid_valid[:, 5]
            vcirc_min = config.par_ranges['vcirc'][0]
            vcirc_max = config.par_ranges['vcirc'][1]
            vcirc_mu = ((vcirc_norm + 1)/2 * (vcirc_max - vcirc_min) + vcirc_min).cpu().numpy()
            mag = tf_calc.sample_mag_from_vcirc(vcirc_mu, rng=np_rng)
            snr = app_mag_to_snr(mag)
            return torch.from_numpy(snr).float().to(self.device)
        elif mode == 'rmag':
            min_rmag = kwargs.get('min', 15)
            max_rmag = kwargs.get('max', 23)
            rmag = sample_magnitudes(size, m_min=min_rmag, m_max=max_rmag, rng=np_rng)
            a, b = _load_rmag_snr_relation()
            log_snr = (rmag - b) / a
            snr = 10**log_snr
            return torch.from_numpy(snr).float().to(self.device)
        else:
            raise ValueError("Invalid SNR generation mode")

#---------------------------#
# Feature Extractor Trainer #
#---------------------------#

class FETrainer(Trainer):
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
        *,
        seed: int | None = None,
        deterministic: bool | None = None,
    ) -> None:
        
        super().__init__(
            world_size, model, train_ds, valid_ds, optimizer, gpu_id,
            save_every, batch_size,
            seed=config.pretrain.get("seed", 20260810) if seed is None else seed,
            deterministic=(
                config.pretrain.get("deterministic", False)
                if deterministic is None
                else deterministic
            ),
        )
        
        self.model_name = config.pretrain['model_name']
        self.return_components = False
        total_epochs = int(config.pretrain['epoch_number'])
        warmup_epochs = min(5, max(0, total_epochs - 1))
        if warmup_epochs:
            lin_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cos_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_epochs - warmup_epochs),
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[lin_scheduler, cos_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=1, eta_min=1e-6)
        self.use_amp = bool(config.pretrain.get('use_amp', False)) and torch.cuda.is_available()
        self.amp_dtype = _resolve_amp_dtype(config.pretrain.get('amp_dtype', 'float16'))
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.use_channels_last = bool(config.pretrain.get('channels_last', False))
        self.use_noise_cache_maxs = bool(config.pretrain.get('noise_cache_maxs', False))
        self.fixed_validation_streams = bool(
            config.pretrain.get('fixed_validation_streams', False)
        )
        self.use_rot90_counterpart = bool(
            config.pretrain.get('use_rot90_counterpart', True)
        )
    
    def _run_batch(self, img, spec, fp, img2=None, spec2=None, fp2=None, fid=None):
        self.optimizer.zero_grad(set_to_none=True)
        diagnostics = {}
        if img2 is None or spec2 is None or fp2 is None:
            output = self.model(
                img, spec, fp, labels=fid, return_diagnostics=True
            )
            if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
                all_loss, diagnostics = output
            else:
                all_loss, diagnostics = output, {}
            loss = all_loss
        else:
            all_loss = self.model(img, spec, fp, img2, spec2, fp2, return_components=self.return_components)
            loss = all_loss[0] if self.return_components else all_loss
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)
            self.optimizer.step()
        return all_loss, diagnostics

    @staticmethod
    def _accumulate_ccl_diagnostics(totals, *diagnostics):
        valid = [item for item in diagnostics if item]
        if not valid:
            return False
        common_keys = set.intersection(*(set(item) for item in valid))
        if not common_keys:
            return False
        for key in common_keys:
            value = torch.stack([item[key] for item in valid]).mean()
            totals[key] = totals.get(key, torch.zeros_like(value)) + value
        return True

    def _log_ccl_diagnostics(self, split, epoch, totals, count, show_log):
        if not count or not show_log or self.gpu_id != self.log_rank:
            return
        means = {key: (value / count).item() for key, value in totals.items()}
        self.logger.info(
            "[%s_CCL] Epoch: %d TargetEntropy: %.6f UniformBaseline: %.6f "
            "ExcessLoss: %.6f EffectivePositives: %.3f TargetMass: %.6f",
            split,
            epoch + 1,
            means["target_entropy"],
            means["uniform_baseline"],
            means["excess_loss"],
            means["effective_positives"],
            means["target_mass"],
        )
        if "shear_loss" in means:
            self.logger.info(
                "[%s_SHEAR] Epoch: %d ShearLoss: %.6f "
                "WeightedShearLoss: %.6f TotalLoss: %.6f",
                split,
                epoch + 1,
                means["shear_loss"],
                means["weighted_shear_loss"],
                means["total_loss"],
            )
    
    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        losses = []
        ccl_diagnostic_totals = {}
        ccl_diagnostic_batches = 0
        if self.return_components:
            sim_losses = []
            var_losses = []
            cov_losses = []
            eff_dim_img_list = []
            eff_dim_spec_list = []
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids] if self.SNR_train is not None else None
            spectral_quality = (
                self.SPEC_QUALITY_train[batch_ids]
                if self.SPEC_QUALITY_train is not None
                else None
            )
            img_maxs = self._noise_cache_for_batch(
                self.img_train_maxs, batch_ids
            )
            spec_maxs = self._noise_cache_for_batch(
                self.spec_train_maxs, batch_ids
            )
            img = self._apply_noise(
                self.img_train[batch_ids],
                snr,
                maxs=img_maxs,
                randgen=self.epoch_generators["train_img_noise"],
            )
            spec = self._apply_spectrum_noise(
                self.spec_train[batch_ids],
                snr,
                spectral_quality=spectral_quality,
                maxs=spec_maxs,
                randgen=self.epoch_generators["train_spec_noise"],
            )
            fid = self.fid_train[batch_ids]
            fp = self.fibpos_train[batch_ids]
            branches = make_ccl_pretrain_views(
                img,
                spec,
                fid,
                fp,
                use_rot90_counterpart=self.use_rot90_counterpart,
            )
            branch_losses = []
            branch_diagnostics = []
            for view_img, view_spec, view_fid, view_fp in branches:
                if self.use_channels_last:
                    view_img = view_img.contiguous(
                        memory_format=torch.channels_last
                    )
                    view_spec = view_spec.contiguous(
                        memory_format=torch.channels_last
                    )
                branch_loss, diagnostics = self._run_batch(
                    view_img,
                    view_spec,
                    view_fp,
                    fid=view_fid,
                )
                branch_losses.append(branch_loss)
                branch_diagnostics.append(diagnostics)
            loss = sum(branch_losses) / len(branch_losses)
            if torch.isfinite(loss):
                losses.append(loss.item())
            if self._accumulate_ccl_diagnostics(
                ccl_diagnostic_totals, *branch_diagnostics
            ):
                ccl_diagnostic_batches += 1

            if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = sum(losses) / len(losses)
        self._log_ccl_diagnostics(
            "TRAIN",
            epoch,
            ccl_diagnostic_totals,
            ccl_diagnostic_batches,
            show_log,
        )
        if self.return_components:
            epoch_sim_loss = sum(sim_losses) / len(sim_losses)
            epoch_var_loss = sum(var_losses) / len(var_losses)
            epoch_cov_loss = sum(cov_losses) / len(cov_losses)
            epoch_eff_dim_img = sum(eff_dim_img_list) / len(eff_dim_img_list)
            epoch_eff_dim_spec = sum(eff_dim_spec_list) / len(eff_dim_spec_list)
            if show_log and self.gpu_id == self.log_rank:
                self.logger.info(f"[TRAIN] Epoch: {epoch+1} Loss: {epoch_loss} Sim: {epoch_sim_loss} Var: {epoch_var_loss} Cov: {epoch_cov_loss}")
                self.logger.info(f"[TRAIN] Epoch: {epoch+1} Effective Dimensionality - Image: {epoch_eff_dim_img} Spectrum: {epoch_eff_dim_spec}")
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
            if torch.cuda.is_available():
                gib = 1024 ** 3
                self.logger.info(
                    "[TRAIN] Epoch: %d Peak CUDA allocated: %.2f GiB; reserved: %.2f GiB",
                    epoch + 1,
                    torch.cuda.max_memory_allocated(self.device) / gib,
                    torch.cuda.max_memory_reserved(self.device) / gib,
                )
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        losses = []
        ccl_diagnostic_totals = {}
        ccl_diagnostic_batches = 0
        if self.return_components:
            sim_losses = []
            var_losses = []
            cov_losses = []
            eff_dim_img_list = []
            eff_dim_spec_list = []
        epoch_start = time.time()

        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids] if self.SNR_valid is not None else None
                spectral_quality = (
                    self.SPEC_QUALITY_valid[batch_ids]
                    if self.SPEC_QUALITY_valid is not None
                    else None
                )
                img_maxs = self._noise_cache_for_batch(
                    self.img_valid_maxs, batch_ids
                )
                spec_maxs = self._noise_cache_for_batch(
                    self.spec_valid_maxs, batch_ids
                )
                img = self._apply_noise(
                    self.img_valid[batch_ids],
                    snr,
                    maxs=img_maxs,
                    randgen=self.epoch_generators["valid_img_noise"],
                )
                spec = self._apply_spectrum_noise(
                    self.spec_valid[batch_ids],
                    snr,
                    spectral_quality=spectral_quality,
                    maxs=spec_maxs,
                    randgen=self.epoch_generators["valid_spec_noise"],
                )
                fid = self.fid_valid[batch_ids]
                fp = self.fibpos_valid[batch_ids]
                branches = make_ccl_pretrain_views(
                    img,
                    spec,
                    fid,
                    fp,
                    use_rot90_counterpart=self.use_rot90_counterpart,
                )
                branch_losses = []
                branch_diagnostics = []
                for view_img, view_spec, view_fid, view_fp in branches:
                    if self.use_channels_last:
                        view_img = view_img.contiguous(
                            memory_format=torch.channels_last
                        )
                        view_spec = view_spec.contiguous(
                            memory_format=torch.channels_last
                        )
                    branch_loss, diagnostics = self.model(
                        view_img,
                        view_spec,
                        view_fp,
                        labels=view_fid,
                        return_diagnostics=True,
                    )
                    branch_losses.append(branch_loss)
                    branch_diagnostics.append(diagnostics)
                loss = sum(branch_losses) / len(branch_losses)
                if torch.isfinite(loss):
                    losses.append(loss.item())

                if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                    self.logger.info(f"Batch {i} complete")
                if self._accumulate_ccl_diagnostics(
                    ccl_diagnostic_totals, *branch_diagnostics
                ):
                    ccl_diagnostic_batches += 1

        epoch_loss = sum(losses) / len(losses)
        self._log_ccl_diagnostics(
            "VALID",
            epoch,
            ccl_diagnostic_totals,
            ccl_diagnostic_batches,
            show_log,
        )
        if self.return_components:
            epoch_sim_loss = sum(sim_losses) / len(sim_losses)
            epoch_var_loss = sum(var_losses) / len(var_losses)
            epoch_cov_loss = sum(cov_losses) / len(cov_losses)
            epoch_eff_dim_img = sum(eff_dim_img_list) / len(eff_dim_img_list)
            epoch_eff_dim_spec = sum(eff_dim_spec_list) / len(eff_dim_spec_list)
            if show_log and self.gpu_id == self.log_rank:
                self.logger.info(f"[VALID] Epoch: {epoch+1} Loss: {epoch_loss} Sim: {epoch_sim_loss} Var: {epoch_var_loss} Cov: {epoch_cov_loss}")
                self.logger.info(f"[VALID] Epoch: {epoch+1} Effective Dimensionality - Image: {epoch_eff_dim_img} Spectrum: {epoch_eff_dim_spec}")
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.pretrain['model_path'], config.pretrain['model_name'], config.pretrain['model_name']+str(epoch))
        torch.save(ckp, PATH)

#-------------#
# NPE Trainer #
#-------------#

class NPETrainer(Trainer):
    def __init__(
        self,
        world_size: int,
        model: torch.nn.Module,
        train_ds: FiberDataset,
        valid_ds: FiberDataset,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        batch_size: int,
        *,
        seed: int | None = None,
        deterministic: bool | None = None,
    ) -> None:
        
        super().__init__(
            world_size, model, train_ds, valid_ds, optimizer, gpu_id,
            save_every, batch_size,
            seed=seed,
            deterministic=deterministic,
        )
        if (
            self.observation_model_version == 2
            and int(getattr(self.model.module, "mode", -1)) != 1
        ):
            raise ValueError(
                "Observation model v2 requires an unweighted mode-1 base posterior; "
                "apply TF information explicitly at inference"
            )
        
        self.model_name = config.train['model_name']
        self.scheduler_type = str(
            config.train.get('scheduler_type', 'plateau')
        ).lower()
        self._configure_scheduler(int(config.train['epoch_number']))
        self.enable_best_checkpoint = True
        patience = config.train.get('early_stopping_patience', None)
        self.early_stopping_patience = (
            None if patience is None else int(patience)
        )
        self.early_stopping_min_delta = float(
            config.train.get('early_stopping_min_delta', 0.0)
        )
        self.use_rot90_counterpart = bool(
            config.train.get('use_rot90_counterpart', True)
        )

    def _configure_scheduler(self, total_epochs):
        if self.scheduler_type == 'plateau':
            # Preserve the historical scheduler exactly when no new scheduler
            # type is requested.
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 'min', factor=0.5, patience=10
            )
            return
        if self.scheduler_type not in ('warmup_cosine', 'cosine'):
            raise ValueError(
                "Unsupported scheduler_type "
                f"'{self.scheduler_type}'. Use 'plateau' or 'warmup_cosine'."
            )

        warmup_epochs = max(0, int(config.train.get('warmup_epochs', 0)))
        if self.scheduler_type == 'cosine':
            warmup_epochs = 0
        elif warmup_epochs >= total_epochs:
            raise ValueError(
                "warmup_epochs must be smaller than epoch_number for "
                "scheduler_type='warmup_cosine'"
            )
        min_lr = float(config.train.get('min_learning_rate', 1e-6))
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        if warmup_epochs == 0:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=cosine_epochs, eta_min=min_lr
            )
            return

        warmup = LinearLR(
            self.optimizer,
            start_factor=float(config.train.get('warmup_start_factor', 0.1)),
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=cosine_epochs, eta_min=min_lr
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    _DIAGNOSTIC_MEAN_KEYS = (
        'raw_feature_rms',
        'effective_sample_size',
        'effective_sample_fraction',
        'affine_log_prob_mean',
        'bounded_log_prob_mean',
        'theta_log_prob_mean',
    )
    _DIAGNOSTIC_MAX_KEYS = (
        'max_normalized_weight',
        'theta_raw_logit_abs_max',
        'theta_bounded_logit_abs_max',
        'theta_logdet_max',
        'theta_derivative_max',
        'theta_wrap_count',
        'theta_max_wrap_excursion',
        'bounded_support_violation_count',
        'bounded_raw_logit_abs_max',
        'bounded_logit_abs_max',
        'bounded_logdet_max',
        'bounded_derivative_max',
    )
    _DIAGNOSTIC_MIN_KEYS = (
        'theta_logdet_min',
        'theta_derivative_min',
        'bounded_logdet_min',
        'bounded_derivative_min',
    )

    def _reset_training_diagnostics(self):
        self._diagnostic_sums = {
            key: 0.0 for key in self._DIAGNOSTIC_MEAN_KEYS
        }
        self._diagnostic_counts = {
            key: 0 for key in self._DIAGNOSTIC_MEAN_KEYS
        }
        self._diagnostic_maxima = {
            key: -float('inf') for key in self._DIAGNOSTIC_MAX_KEYS
        }
        self._diagnostic_minima = {
            key: float('inf') for key in self._DIAGNOSTIC_MIN_KEYS
        }

    def _capture_training_diagnostics(self):
        diagnostics = getattr(
            self.model.module, 'last_training_diagnostics', None
        )
        if not diagnostics:
            return
        keys = (
            *self._DIAGNOSTIC_MEAN_KEYS,
            *self._DIAGNOSTIC_MAX_KEYS,
            *self._DIAGNOSTIC_MIN_KEYS,
        )
        stacked_values = []
        for key in keys:
            value = diagnostics.get(key)
            if value is None:
                value = torch.full(
                    (), float('nan'), dtype=torch.float64, device=self.device
                )
            else:
                value = torch.as_tensor(
                    value, dtype=torch.float64, device=self.device
                ).reshape(())
            stacked_values.append(value)
        # One stacked device-to-host transfer avoids one CUDA synchronization
        # for every individual diagnostic scalar.
        scalars = torch.stack(stacked_values).detach().cpu().tolist()
        values_by_key = dict(zip(keys, scalars))

        for key in self._DIAGNOSTIC_MEAN_KEYS:
            scalar = values_by_key[key]
            if np.isfinite(scalar):
                self._diagnostic_sums[key] += scalar
                self._diagnostic_counts[key] += 1
        for key in self._DIAGNOSTIC_MAX_KEYS:
            scalar = values_by_key[key]
            if np.isfinite(scalar):
                self._diagnostic_maxima[key] = max(
                    self._diagnostic_maxima[key], scalar
                )
        for key in self._DIAGNOSTIC_MIN_KEYS:
            scalar = values_by_key[key]
            if np.isfinite(scalar):
                self._diagnostic_minima[key] = min(
                    self._diagnostic_minima[key], scalar
                )

    def _finalize_training_diagnostics(self):
        diagnostics = {}
        for key in self._DIAGNOSTIC_MEAN_KEYS:
            diagnostics[key] = self._global_mean_from_sum_count(
                self._diagnostic_sums[key], self._diagnostic_counts[key]
            )
        for key in self._DIAGNOSTIC_MAX_KEYS:
            value = self._global_max(self._diagnostic_maxima[key])
            diagnostics[key] = value if np.isfinite(value) else float('nan')
        for key in self._DIAGNOSTIC_MIN_KEYS:
            value = self._global_min(self._diagnostic_minima[key])
            diagnostics[key] = value if np.isfinite(value) else float('nan')
        return diagnostics

    def _run_batch(
        self,
        img,
        spec,
        fid,
        mode,
        fp=None,
        snr=None,
        observation_context=None,
    ):
        if mode == 'train':
            self.optimizer.zero_grad(set_to_none=True)

        if self.model.module.mode == 2:
            self.mag = snr_to_app_mag(snr) if snr is not None else None
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                model_kwargs = {
                    "fp": fp,
                    "mag": self.mag,
                    "snr": snr,
                }
                if observation_context is not None:
                    model_kwargs["observation_context"] = observation_context
                loss = self.model(img, spec, fid, **model_kwargs)
        else:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                model_kwargs = {"fp": fp}
                if observation_context is not None:
                    model_kwargs["observation_context"] = observation_context
                loss = self.model(img, spec, fid, **model_kwargs)
        if mode == 'train':
            self._capture_training_diagnostics()

        # All ranks must make the same backward/step decision; otherwise DDP can
        # deadlock or silently let the replicas diverge.
        loss_is_valid = self._all_ranks_true(torch.isfinite(loss).item())
        self.last_batch_globally_valid = loss_is_valid
        self.last_batch_step_valid = loss_is_valid
        self.last_preclip_grad_norm = float("nan")
        if not loss_is_valid:
            self.invalid_loss_count += 1
            self.last_batch_step_valid = False
            return loss

        if mode != 'train':
            return loss

        if self.use_amp:
            self.scaler.scale(loss).backward()
            # clip_grad_norm_ must see unscaled gradients. This also registers
            # AMP's non-finite-gradient check before scaler.step().
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        parameters = [
            parameter
            for parameter in self.model.module.parameters()
            if parameter.grad is not None
        ]
        max_norm = (
            self.gradient_clip_norm
            if self.gradient_clip_norm is not None
            and self.gradient_clip_norm > 0
            else float("inf")
        )
        preclip_norm = nn.utils.clip_grad_norm_(
            parameters,
            max_norm=max_norm,
            error_if_nonfinite=False,
        )
        self.last_preclip_grad_norm = float(preclip_norm.detach().item())
        gradients_are_valid = self._all_ranks_true(
            torch.isfinite(preclip_norm).item()
        )
        self.last_batch_step_valid = gradients_are_valid
        if gradients_are_valid:
            self._preclip_grad_norm_sum += self.last_preclip_grad_norm
            self._preclip_grad_norm_count += 1
            self._preclip_grad_norm_max = max(
                self._preclip_grad_norm_max, self.last_preclip_grad_norm
            )
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        else:
            self.invalid_gradient_count += 1
            if self.use_amp:
                # A globally rejected step must also leave every rank with the
                # same scale. Explicit backoff avoids rank-local scaler drift.
                current_scale = float(self.scaler.get_scale())
                backoff = float(self.scaler.get_backoff_factor())
                self.scaler.update(new_scale=current_scale * backoff)

        return loss

    def _trainFunc(self, epoch, show_log=True):
        self.model.train()
        if hasattr(self.model.module, 'feature_extractor'):
            self.model.module.feature_extractor.eval()
        local_loss_sum = 0.0
        local_loss_count = 0
        epoch_start = time.time()
        self.invalid_loss_count = 0
        self.invalid_gradient_count = 0
        self._preclip_grad_norm_sum = 0.0
        self._preclip_grad_norm_count = 0
        self._preclip_grad_norm_max = 0.0
        self._reset_training_diagnostics()
        
        for i in range(self.nbatch_train):
            start = i*self.batch_size
            batch_ids = self.train_order[start:start+self.batch_size]
            snr = self.SNR_train[batch_ids] if self.SNR_train is not None else None
            spectral_quality = (
                self.SPEC_QUALITY_train[batch_ids]
                if self.SPEC_QUALITY_train is not None
                else None
            )
            img_maxs = self._noise_cache_for_batch(
                self.img_train_maxs, batch_ids
            )
            spec_maxs = self._noise_cache_for_batch(
                self.spec_train_maxs, batch_ids
            )
            img = self._apply_noise(
                self.img_train[batch_ids],
                snr,
                maxs=img_maxs,
                randgen=self.epoch_generators["train_img_noise"],
            )
            spec = self._apply_spectrum_noise(
                self.spec_train[batch_ids],
                snr,
                spectral_quality=spectral_quality,
                maxs=spec_maxs,
                randgen=self.epoch_generators["train_spec_noise"],
            )
            fid = self.fid_train[batch_ids]
            fp = self.fibpos_train[batch_ids]
            observation_context = self._observation_context_for_batch(
                batch_ids,
                split="train",
                duplicate=self.use_rot90_counterpart,
            )
            model_snr = None if self.observation_model_version == 2 else snr
            img, spec, fid, fp, snr = make_npe_training_batch(
                img,
                spec,
                fid,
                fp,
                model_snr,
                use_rot90_counterpart=self.use_rot90_counterpart,
            )
            if self.use_channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            loss = self._run_batch(
                img,
                spec,
                fid,
                'train',
                fp=fp,
                snr=snr,
                observation_context=observation_context,
            )
            if self.last_batch_globally_valid:
                local_loss_sum += float(loss.detach().item())
                local_loss_count += 1

            if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                self.logger.info(f"Batch {i} complete")

        epoch_loss = self._global_mean_from_sum_count(
            local_loss_sum, local_loss_count
        )
        self.invalid_loss_count = self._synchronize_counter(
            self.invalid_loss_count
        )
        self.invalid_gradient_count = self._synchronize_counter(
            self.invalid_gradient_count
        )
        mean_grad_norm = self._global_mean_from_sum_count(
            self._preclip_grad_norm_sum, self._preclip_grad_norm_count
        )
        max_grad_norm = self._global_max(self._preclip_grad_norm_max)
        if not np.isfinite(mean_grad_norm):
            max_grad_norm = float("nan")
        self.preclip_grad_norm_history.append(
            {"epoch": epoch + 1, "mean": mean_grad_norm, "max": max_grad_norm}
        )
        training_diagnostics = self._finalize_training_diagnostics()
        self.training_diagnostic_history.append(
            {"epoch": epoch + 1, **training_diagnostics}
        )
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[TRAIN] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
            self.logger.info(
                "[TRAIN] Epoch: %d InvalidLossSteps: %d "
                "InvalidGradientSteps: %d PreClipGradNormMean: %.8g "
                "PreClipGradNormMax: %.8g",
                epoch + 1,
                self.invalid_loss_count,
                self.invalid_gradient_count,
                mean_grad_norm,
                max_grad_norm,
            )
            if np.isfinite(training_diagnostics['raw_feature_rms']):
                self.logger.info(
                    "[TRAIN_FEATURE] Epoch: %d RawFeatureRMS: %.8g",
                    epoch + 1,
                    training_diagnostics['raw_feature_rms'],
                )
            if np.isfinite(training_diagnostics['effective_sample_size']):
                self.logger.info(
                    "[TRAIN_TF] Epoch: %d EffectiveSampleSize: %.6g "
                    "EffectiveSampleFraction: %.6g MaxNormalizedWeight: %.6g",
                    epoch + 1,
                    training_diagnostics['effective_sample_size'],
                    training_diagnostics['effective_sample_fraction'],
                    training_diagnostics['max_normalized_weight'],
                )
            if np.isfinite(training_diagnostics['affine_log_prob_mean']):
                self.logger.info(
                    "[TRAIN_FLOW] Epoch: %d AffineLogProbMean: %.8g "
                    "ThetaLogProbMean: %.8g",
                    epoch + 1,
                    training_diagnostics['affine_log_prob_mean'],
                    training_diagnostics['theta_log_prob_mean'],
                )
            if np.isfinite(
                training_diagnostics['theta_raw_logit_abs_max']
            ):
                self.logger.info(
                    "[TRAIN_THETA] Epoch: %d RawLogitAbsMax: %.8g "
                    "BoundedLogitAbsMax: %.8g LogDetMin: %.8g LogDetMax: %.8g "
                    "DerivativeMin: %.8g DerivativeMax: %.8g "
                    "WrapCountMax: %.8g MaxWrapExcursion: %.8g",
                    epoch + 1,
                    training_diagnostics['theta_raw_logit_abs_max'],
                    training_diagnostics['theta_bounded_logit_abs_max'],
                    training_diagnostics['theta_logdet_min'],
                    training_diagnostics['theta_logdet_max'],
                    training_diagnostics['theta_derivative_min'],
                    training_diagnostics['theta_derivative_max'],
                    training_diagnostics['theta_wrap_count'],
                    training_diagnostics['theta_max_wrap_excursion'],
                )
            if np.isfinite(
                training_diagnostics['bounded_raw_logit_abs_max']
            ):
                self.logger.info(
                    "[TRAIN_BOUNDED] Epoch: %d NonThetaLogProbMean: %.8g "
                    "SupportViolationCountMax: %.8g RawLogitAbsMax: %.8g "
                    "BoundedLogitAbsMax: %.8g LogDetMin: %.8g "
                    "LogDetMax: %.8g DerivativeMin: %.8g "
                    "DerivativeMax: %.8g",
                    epoch + 1,
                    training_diagnostics['bounded_log_prob_mean'],
                    training_diagnostics[
                        'bounded_support_violation_count'
                    ],
                    training_diagnostics['bounded_raw_logit_abs_max'],
                    training_diagnostics['bounded_logit_abs_max'],
                    training_diagnostics['bounded_logdet_min'],
                    training_diagnostics['bounded_logdet_max'],
                    training_diagnostics['bounded_derivative_min'],
                    training_diagnostics['bounded_derivative_max'],
                )
        return epoch_loss

    def _validFunc(self,epoch,show_log=True):
        self.model.eval()
        local_loss_sum = 0.0
        local_loss_count = 0
        epoch_start = time.time()
        self.invalid_loss_count = 0

        with torch.no_grad():
            for i in range(self.nbatch_valid):
                start = i*self.batch_size
                batch_ids = self.valid_order[start:start+self.batch_size]
                snr = self.SNR_valid[batch_ids] if self.SNR_valid is not None else None
                spectral_quality = (
                    self.SPEC_QUALITY_valid[batch_ids]
                    if self.SPEC_QUALITY_valid is not None
                    else None
                )
                img_maxs = self._noise_cache_for_batch(
                    self.img_valid_maxs, batch_ids
                )
                spec_maxs = self._noise_cache_for_batch(
                    self.spec_valid_maxs, batch_ids
                )
                img = self._apply_noise(
                    self.img_valid[batch_ids],
                    snr,
                    maxs=img_maxs,
                    randgen=self.epoch_generators["valid_img_noise"],
                )
                spec = self._apply_spectrum_noise(
                    self.spec_valid[batch_ids],
                    snr,
                    spectral_quality=spectral_quality,
                    maxs=spec_maxs,
                    randgen=self.epoch_generators["valid_spec_noise"],
                )
                fid = self.fid_valid[batch_ids]
                fp = self.fibpos_valid[batch_ids]
                observation_context = self._observation_context_for_batch(
                    batch_ids,
                    split="valid",
                    duplicate=self.use_rot90_counterpart,
                )
                model_snr = None if self.observation_model_version == 2 else snr
                img, spec, fid, fp, snr = make_npe_training_batch(
                    img,
                    spec,
                    fid,
                    fp,
                    model_snr,
                    use_rot90_counterpart=self.use_rot90_counterpart,
                )
                if self.use_channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                    spec = spec.contiguous(memory_format=torch.channels_last)
                loss = self._run_batch(
                    img,
                    spec,
                    fid,
                    'valid',
                    fp=fp,
                    snr=snr,
                    observation_context=observation_context,
                )
                if self.last_batch_globally_valid:
                    local_loss_sum += float(loss.detach().item())
                    local_loss_count += 1

                if show_log and self.gpu_id == self.log_rank and i%100 == 0:
                    self.logger.info(f"Batch {i} complete")

        epoch_loss = self._global_mean_from_sum_count(
            local_loss_sum, local_loss_count
        )
        self.invalid_loss_count = self._synchronize_counter(
            self.invalid_loss_count
        )
        epoch_time = time.time() - epoch_start
        if show_log and self.gpu_id == self.log_rank:
            self.logger.info("[VALID] Epoch: {} Loss: {} Time: {:.0f}:{:.0f}".format(epoch+1, epoch_loss,
                                                                                    epoch_time // 60, 
                                                                                    epoch_time % 60))
            self.logger.info(
                "[VALID] Epoch: %d InvalidLossSteps: %d",
                epoch + 1,
                self.invalid_loss_count,
            )
        return epoch_loss
    
    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = join(config.train['model_path'], config.train['model_name'], config.train['model_name']+str(epoch))
        torch.save(ckp, PATH)

    def _save_best_checkpoint(self, epoch, train_loss, valid_loss):
        model_dir = join(config.train['model_path'], config.train['model_name'])
        os.makedirs(model_dir, exist_ok=True)
        checkpoint_name = f"{config.train['model_name']}best"
        checkpoint_path = join(model_dir, checkpoint_name)
        metadata_path = join(model_dir, 'best.json')
        torch.save(self.model.module.state_dict(), checkpoint_path)

        numeric_checkpoint = f"{config.train['model_name']}{epoch}"
        metadata = {
            "model_name": config.train['model_name'],
            "checkpoint": numeric_checkpoint,
            "checkpoint_path": join(model_dir, numeric_checkpoint),
            "checkpoint_suffix": str(epoch),
            "named_best_checkpoint": checkpoint_name,
            "named_best_checkpoint_path": checkpoint_path,
            "named_best_checkpoint_suffix": "best",
            "epoch": int(epoch + 1),
            "epoch_index": int(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(valid_loss),
            "scheduler_type": self.scheduler_type,
            # Scheduler stepping precedes checkpoint selection in the legacy
            # loop, so these are the rates prepared for the following epoch.
            "next_epoch_learning_rates": [
                float(group['lr']) for group in self.optimizer.param_groups
            ],
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "early_stopping_patience": self.early_stopping_patience,
        }
        temporary_path = f"{metadata_path}.tmp.{os.getpid()}"
        with open(temporary_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        os.replace(temporary_path, metadata_path)
        self.logger.info(
            "Saved best checkpoint from epoch %d with validation loss %.8g",
            epoch + 1,
            valid_loss,
        )

#------------------#
# Global functions #
#------------------#

def train_nn(rank: int, world_size: int, Model=VICRegPretrain, Trainer=FETrainer,
             save_every=1, train_mode='pretrain', model_config_payload=None):
    '''
    Main function to train any network.
    '''
    if model_config_payload is not None:
        config.set_model_config(config.ModelConfig.from_dict(model_config_payload))

    if train_mode == 'pretrain':
        train_config = config.pretrain
    elif train_mode == 'train':
        train_config = config.train
    else:
        raise ValueError(f"Invalid train_mode '{train_mode}'. Must be 'pretrain' or 'train'.")

    base_seed = int(train_config.get('seed', 20260810))
    deterministic = bool(train_config.get('deterministic', False))
    os.environ["PYTHONHASHSEED"] = str(base_seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Set parameters based on stage
    total_epochs = train_config['epoch_number']
    batch_size = train_config['batch_size']
    pretrain_epoch = train_config.get('pretrain_from', None)
    
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('Setup')
    if rank == 0:
        log.info('Initializing')
        log.info(f'Run seed={base_seed}; deterministic={deterministic}')
    
    try:
        ddp_setup(rank, world_size)
        log.info(f'[rank: {rank}] Successfully set up device')
        torch.backends.cudnn.benchmark = bool(train_config.get('cudnn_benchmark', False)) and not deterministic
        seed_everything(base_seed, deterministic=deterministic)

        device = torch.device(f"cuda:{rank}")
        train_ds, valid_ds, model, optimizer = load_train_objs(
            Model,
            rank,
            train_config,
            train_mode=train_mode,
            epoch=pretrain_epoch,
            log=log,
            device=device,
        )
        if train_config.get('channels_last', False):
            model = model.to(memory_format=torch.channels_last)
        model = _maybe_compile_model(model, log=log, use_compile=train_config.get('use_compile', False),)
        ddp_kwargs = {
            "device_ids": [rank],
            "find_unused_parameters": bool(train_config.get('ddp_find_unused_parameters', False)),
            "broadcast_buffers": bool(train_config.get('ddp_broadcast_buffers', False)),
            "gradient_as_bucket_view": bool(train_config.get('ddp_gradient_as_bucket_view', True)),
        }
        if train_config.get('ddp_static_graph', False):
            ddp_kwargs["static_graph"] = True
        try:
            model = DDP(model, **ddp_kwargs)
        except TypeError:
            ddp_kwargs.pop("static_graph", None)
            ddp_kwargs.pop("gradient_as_bucket_view", None)
            model = DDP(model, **ddp_kwargs)
        seed_everything(
            derive_stream_seed(base_seed, rank=rank, epoch=0, stream="ambient"),
            deterministic=deterministic,
        )
        log.info(f'[rank: {rank}] Successfully loaded training objects')
    
        os.makedirs(join(train_config['model_path'], train_config['model_name']), exist_ok=True)
        trainer = Trainer(
            world_size, model, train_ds, valid_ds, optimizer, rank, save_every,
            batch_size, seed=base_seed, deterministic=deterministic,
        )
        log.info(f'[rank: {rank}] Successfully initialized Trainer')
        torch.distributed.barrier()
        trainer.train(total_epochs)
        torch.distributed.barrier()
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            destroy_process_group()
    
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.synchronize()


def _npe_optimizer_parameters(model, train_config):
    """Build non-overlapping LR groups for the hybrid posterior.

    Legacy models keep their historical single parameter group. The hybrid
    factorization gets a lower LR for its circular conditioner while every
    trainable parameter still appears exactly once.
    """
    flow = getattr(model, 'flow', None)
    affine_flow = getattr(flow, 'affine_flow', None)
    theta_transform = getattr(flow, 'theta_transform', None)
    if affine_flow is None or theta_transform is None:
        return model.parameters()

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    affine = [
        parameter
        for parameter in affine_flow.parameters()
        if parameter.requires_grad
    ]
    theta = [
        parameter
        for parameter in theta_transform.parameters()
        if parameter.requires_grad
    ]
    affine_ids = {id(parameter) for parameter in affine}
    theta_ids = {id(parameter) for parameter in theta}
    if affine_ids & theta_ids:
        raise RuntimeError(
            "Hybrid affine_flow and theta_transform share optimizer parameters"
        )

    branch_ids = affine_ids | theta_ids
    remaining = [
        parameter for parameter in trainable if id(parameter) not in branch_ids
    ]
    grouped_ids = {id(parameter) for parameter in remaining} | branch_ids
    trainable_ids = {id(parameter) for parameter in trainable}
    if grouped_ids != trainable_ids or (
        len(remaining) + len(affine) + len(theta) != len(trainable)
    ):
        raise RuntimeError(
            "Hybrid optimizer parameter grouping omitted or duplicated parameters"
        )

    base_lr = float(train_config['initial_learning_rate'])
    affine_lr = train_config.get('affine_learning_rate', None)
    theta_lr = train_config.get('theta_learning_rate', None)
    groups = []
    if remaining:
        groups.append(
            {'params': remaining, 'lr': base_lr, 'group_name': 'shared'}
        )
    if affine:
        groups.append({
            'params': affine,
            'lr': base_lr if affine_lr is None else float(affine_lr),
            'group_name': 'affine_flow',
        })
    if theta:
        groups.append({
            'params': theta,
            'lr': base_lr if theta_lr is None else float(theta_lr),
            'group_name': 'theta_transform',
        })
    return groups


def load_train_objs(
    Model,
    rank,
    train_config,
    train_mode='pretrain',
    epoch=None,
    log=None,
    device=None,
    **kwargs,
):
    # Create dataset objects
    train_ds = pxt.TorchDataset(config.data['data_dir'])
    valid_ds = pxt.TorchDataset(config.test['data_dir'])
    model_kwargs = dict(kwargs)
    if isinstance(Model, type) and issubclass(Model, KLNPE):
        # Never rely on constructor defaults for values supplied by a loaded
        # run config.  This also protects worker processes using archived
        # configurations from module-import-time defaults.
        model_kwargs.setdefault('mode', int(train_config['mode']))
        model_kwargs.setdefault('batch_size', int(train_config['batch_size']))
        model_kwargs.setdefault(
            'nfeatures', int(train_config['feature_number'])
        )
        model_kwargs.setdefault('nspec', int(config.data['nspec']))
        model_kwargs.setdefault(
            'backbone_type', train_config.get('backbone_type', None)
        )
        model_kwargs.setdefault(
            'posterior_symmetry',
            train_config.get('posterior_symmetry', None),
        )
    # Initialize model and optimizer
    if epoch is not None: # if epoch is specified, load pretrained model
        strict = True
        model_dir = train_config['model_path'] + train_config['pretrained_name'] + '/' + train_config['pretrained_name'] + str(epoch)
        pretrained_model = load_model(train_config, Model=CCLPretrain, path=model_dir, strict=strict, assign=True)
        model = Model(pretrained_model.backbone, **model_kwargs)
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
        if rank == 0:
            if log is not None:
                log.info(f"Loaded model {train_config['pretrained_name']} at epoch {epoch}")
    else:
        model = Model(**model_kwargs)  # initialize new model
        if rank == 0:
            if log is not None:
                log.info(f"Loaded new model {train_config['model_name']}")
    if device is not None:
        model = model.to(device)
    if train_mode == 'pretrain':
        # optimizer = Lars(
        #     model.parameters(), 
        #     lr=train_config['initial_learning_rate'], 
        #     weight_decay=train_config['weight_decay'], 
        #     momentum=0.9,
        #     eps=train_config.get('eps', 1e-8),
        # )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config['initial_learning_rate'],
            weight_decay=train_config['weight_decay'],
            eps=train_config.get('eps', 1e-8),
        )
    else:
        use_fused = (
            bool(train_config.get('use_fused_adamw', False))
            and torch.cuda.is_available()
            and next(model.parameters()).is_cuda
        )
        optimizer_kwargs = dict(
            lr=train_config['initial_learning_rate'],
            weight_decay=train_config['weight_decay'],
            eps=train_config.get('eps', 1e-8)
        )
        # optimizer_diff = [{'params': model.feature_extractor.parameters(), 'lr': train_config['initial_learning_rate']*0.1},
        #                   {'params': model.flow.parameters(), 'lr': train_config['initial_learning_rate']*10}]
        if use_fused:
            optimizer_kwargs["fused"] = True
        try:
            optimizer = optim.AdamW(
                _npe_optimizer_parameters(model, train_config),
                **optimizer_kwargs,
            )
        except TypeError:
            if use_fused and log is not None:
                log.warning("Fused AdamW not supported in this Torch build; falling back to standard AdamW.")
            optimizer_kwargs.pop("fused", None)
            optimizer = optim.AdamW(
                _npe_optimizer_parameters(model, train_config),
                **optimizer_kwargs,
            )

    return train_ds, valid_ds, model, optimizer

def load_model(
    train_config,
    Model=FeatureExtractor,
    path=None,
    strict=True,
    assign=False,
    GPUs=1,
    device='cpu',
    model_name=None,
    networks_root=None,
    use_compile: bool | None = None,
    compile_mode: str | None = None,
    compile_backend: str | None = None,
    channels_last: bool | None = None,
    use_archived_networks: bool = True,
):

    model_cls = Model
    resolved_name = None
    if path is not None and use_archived_networks:
        resolved_name = model_name or infer_model_name_from_checkpoint_path(path)
        if resolved_name:
            try:
                if networks_root is None:
                    archived_module = load_networks_module_for_model(resolved_name)
                else:
                    archived_module = load_networks_module_for_model(
                        resolved_name, networks_root=networks_root
                    )
            except FileNotFoundError:
                archived_module = None

            if archived_module is not None:
                try:
                    model_cls = getattr(archived_module, Model.__name__)
                except AttributeError as exc:
                    raise AttributeError(
                        f"Archived networks module for '{resolved_name}' "
                        f"does not define {Model.__name__}."
                    ) from exc
    model = model_cls()
    model.to(device)
    if channels_last is None:
        channels_last = bool(train_config.get('channels_last', False))
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    if GPUs > 1:
        model = DDP(model, device_ids=None)

    if path != None:
        state_dict = torch.load(path, weights_only=False, map_location=torch.device(device))
        state_dict = _maybe_strip_state_dict_prefix(state_dict, model, "module.")
        state_dict = _maybe_strip_state_dict_prefix(state_dict, model, "_orig_mod.")
        model.load_state_dict(state_dict, strict=strict, assign=assign)

    if use_compile is None:
        use_compile = bool(train_config.get('use_compile', False))
    if use_compile and GPUs <= 1:
        model = _maybe_compile_model(
            model,
            log=None,
            use_compile=use_compile,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
        )

    return model

def point_estimate(
    model,
    test_ds,
    snr=None,
    randgen=None,
    device='cpu',
    progress=None,
):
    '''
    Run this function to get point estimates from trained density estimation models
    '''
    model.eval()
    preds = []
    trues = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*995 + 5
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Estimating")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            fids = torch.as_tensor(test_ds[i]['fid_pars'][:model.nfeatures], dtype=torch.float32, device=device).unsqueeze(0)
            trues.append(fids.detach().cpu().numpy())
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            pred = model.point_estimate(img, spec, fp)
            preds.append(pred.detach().cpu().numpy())
    
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    snrs = snrs.cpu().numpy()
    return preds, trues, snrs

def pair_rotation_branches(values):
    """Pair interleaved original/rotated outputs along a branch axis.

    ``sample_density`` appends ``original, rotated`` for each galaxy. This
    converts ``(2 * N, S, ...)`` to ``(N, S, 2, ...)``, with branch 0 original
    and branch 1 rotated.
    """
    values = np.asarray(values)
    if values.ndim < 2:
        raise ValueError("rotation-paired values need at least two dimensions")
    if values.shape[0] % 2:
        raise ValueError("rotation-paired values need an even leading dimension")
    paired = values.reshape(values.shape[0] // 2, 2, *values.shape[1:])
    return np.moveaxis(paired, 1, 2)

def _unwrap_posterior_model(model):
    """Return the checkpoint-owning module through optional compile wrappers."""
    owner = model
    while hasattr(owner, "_orig_mod"):
        owner = owner._orig_mod
    return owner


def load_v2_observation_metadata(
    dataset,
    *,
    expected_fiber_layout=None,
    device="cpu",
    return_halpha_flux: bool = False,
):
    """Load validated v2 magnitudes and optionally integrated H-alpha fluxes."""
    observation = config.observation
    expected_version = int(observation.get("model_version", 1))
    if expected_version != 2:
        raise ValueError(
            "load_v2_observation_metadata requires observation model_version=2"
        )
    if expected_fiber_layout is None:
        expected_fiber_layout = observation.get("fiber_layout", "galaxy_axis")
    expected_fiber_layout = str(expected_fiber_layout)
    if expected_fiber_layout not in FIBER_LAYOUT_CODES:
        raise ValueError(
            f"Unsupported configured fiber layout {expected_fiber_layout!r}"
        )
    magnitudes = torch.empty(len(dataset), dtype=torch.float32, device=device)
    halpha_fluxes = (
        torch.empty(len(dataset), dtype=torch.float32, device=device)
        if return_halpha_flux
        else None
    )

    for index in range(len(dataset)):
        record = dataset[index]
        magnitudes[index] = validate_v2_observation_record(
            record,
            observation=observation,
            expected_fiber_layout=expected_fiber_layout,
            location=f"analysis record {index}",
        )
        if halpha_fluxes is not None:
            halpha_fluxes[index] = torch.as_tensor(
                record["halpha_flux_true"], dtype=torch.float32, device=device
            ).reshape(())
    if halpha_fluxes is not None:
        return magnitudes, halpha_fluxes
    return magnitudes


def checkpoint_spectral_reference_line_norm(model):
    """Read the positive spectral reference persisted in a v2 checkpoint."""
    owner = _unwrap_posterior_model(model)
    if not hasattr(owner, "spectral_reference_line_norm"):
        raise RuntimeError(
            "Observation-model-v2 checkpoint is missing "
            "spectral_reference_line_norm"
        )
    reference = owner.spectral_reference_line_norm.detach().reshape(())
    if not bool(torch.isfinite(reference) & (reference > 0)):
        raise RuntimeError(
            "Observation-model-v2 checkpoint has an invalid "
            "spectral_reference_line_norm"
        )
    return reference


def checkpoint_image_noise_sigma(model):
    """Read the positive fixed image-pixel RMS persisted in a v2 checkpoint."""
    owner = _unwrap_posterior_model(model)
    if not hasattr(owner, "image_noise_sigma"):
        raise RuntimeError(
            "Observation-model-v2 checkpoint is missing image_noise_sigma"
        )
    sigma = owner.image_noise_sigma.detach().reshape(())
    if not bool(torch.isfinite(sigma) & (sigma > 0)):
        raise RuntimeError(
            "Observation-model-v2 checkpoint has an invalid image_noise_sigma"
        )
    return sigma


def _seeded_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed) % (2**63 - 1))


def sample_density(
    model,
    test_ds,
    nsamples,
    snr=None,
    mag=None,
    vcirc_mu=None,
    randgen=None,
    apply_add_noise_cancellation=False,
    return_log_prob=False,
    device='cpu',
    channels_last: bool | None = None,
    matched_group_size: int = 1,
    noise_seed: int = 42,
    spectral_noise_seed: int | None = None,
    magnitude_seed: int | None = None,
    spectral_quality_seed: int | None = None,
    image_randgen=None,
    spectral_randgen=None,
    magnitude_randgen=None,
    spectral_quality_randgen=None,
    spectral_quality=None,
    rmag_true=None,
    tf_inference=None,
    return_observation_metadata: bool = False,
    progress=None,
):
    """Sample posteriors with legacy-v1 or versioned-v2 observations."""
    del vcirc_mu  # Retained in the public signature for legacy callers.
    posterior_owner = _unwrap_posterior_model(model)
    posterior_symmetry = getattr(posterior_owner, "posterior_symmetry", "none")
    observation_model_version = int(
        getattr(posterior_owner, "observation_model_version", 1)
    )
    if posterior_symmetry == "d4" and apply_add_noise_cancellation:
        raise ValueError(
            "apply_add_noise_cancellation is redundant and incompatible with "
            "an exactly D4-symmetrized posterior"
        )
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if matched_group_size < 1 or len(test_ds) % matched_group_size:
        raise ValueError(
            "matched_group_size must be positive and divide the dataset size"
        )

    observation_metadata = None
    if observation_model_version == 2:
        if getattr(posterior_owner, "mode", None) != 1:
            raise ValueError(
                "Observation model v2 requires a mode-1 base posterior"
            )
        if tf_inference not in (None, "prior_replacement"):
            raise ValueError(
                "Observation model v2 supports only no TF inference or "
                "tf_inference='prior_replacement'"
            )
        if (
            tf_inference == "prior_replacement"
            and apply_add_noise_cancellation
        ):
            raise ValueError(
                "TF prior replacement does not support the two-branch "
                "additive-noise cancellation diagnostic"
            )
        if mag is not None:
            raise ValueError(
                "Do not pass mag for observation model v2; the noisy catalog "
                "magnitude is generated from archived rmag_true"
            )
        archived_rmag, archived_halpha_flux = load_v2_observation_metadata(
            test_ds,
            expected_fiber_layout=config.observation.get(
                "fiber_layout", "galaxy_axis"
            ),
            device=device,
            return_halpha_flux=True,
        )
        if rmag_true is not None:
            supplied_rmag = torch.as_tensor(
                rmag_true,
                dtype=archived_rmag.dtype,
                device=archived_rmag.device,
            )
            if supplied_rmag.shape != archived_rmag.shape or not torch.allclose(
                supplied_rmag, archived_rmag, rtol=0.0, atol=1e-4
            ):
                raise ValueError(
                    "Supplied rmag_true does not match archived LMDB metadata"
                )
        rmag_true = archived_rmag

        grouped_rmag = rmag_true.reshape(-1, matched_group_size)
        group_rmag = grouped_rmag[:, 0]
        if not torch.allclose(
            grouped_rmag,
            group_rmag[:, None].expand_as(grouped_rmag),
            rtol=0.0,
            atol=1e-4,
        ):
            raise ValueError(
                "Every matched observation group must share the same "
                "archived rmag_true"
            )
        grouped_halpha_flux = archived_halpha_flux.reshape(
            -1, matched_group_size
        )
        if not torch.allclose(
            grouped_halpha_flux,
            grouped_halpha_flux[:, :1].expand_as(grouped_halpha_flux),
            rtol=2e-6,
            atol=0.0,
        ):
            raise ValueError(
                "Every matched observation group must share the same "
                "archived halpha_flux_true"
            )

        if spectral_noise_seed is None:
            spectral_noise_seed = noise_seed + 101
        if magnitude_seed is None:
            magnitude_seed = noise_seed + 211
        if spectral_quality_seed is None:
            spectral_quality_seed = noise_seed + 307
        if image_randgen is None:
            image_randgen = randgen
        if image_randgen is None:
            image_randgen = _seeded_generator(device, noise_seed)
        if spectral_randgen is None:
            spectral_randgen = _seeded_generator(device, spectral_noise_seed)
        if magnitude_randgen is None:
            magnitude_randgen = _seeded_generator(device, magnitude_seed)
        if spectral_quality_randgen is None:
            spectral_quality_randgen = _seeded_generator(
                device, spectral_quality_seed
            )

        observation = config.observation
        group_expected_snrs, group_generated_quality = (
            build_v2_observation_levels(
                group_rmag,
                image_band=observation["image_band"],
                image_depth_5sigma_mag=observation["image_depth_5sigma_mag"],
                spectral_quality_min=observation["spectral_quality_min"],
                spectral_quality_max=observation["spectral_quality_max"],
                spectral_quality_distribution=observation[
                    "spectral_quality_distribution"
                ],
                spectral_generator=spectral_quality_randgen,
            )
        )
        expected_snrs = torch.repeat_interleave(
            group_expected_snrs, matched_group_size
        )
        generated_quality = torch.repeat_interleave(
            group_generated_quality, matched_group_size
        )
        if snr is not None:
            supplied_snrs = torch.as_tensor(
                snr, dtype=expected_snrs.dtype, device=expected_snrs.device
            )
            if supplied_snrs.shape != expected_snrs.shape or not torch.allclose(
                supplied_snrs, expected_snrs, rtol=2e-6, atol=1e-6
            ):
                raise ValueError(
                    "Observation-model-v2 image SNR must be derived from "
                    "archived rmag_true and the configured survey depth"
                )
        snrs = expected_snrs
        if spectral_quality is None:
            spectral_quality = generated_quality
        else:
            spectral_quality = torch.as_tensor(
                spectral_quality,
                dtype=snrs.dtype,
                device=snrs.device,
            )
            if spectral_quality.shape != snrs.shape:
                raise ValueError(
                    "spectral_quality must have one value per galaxy"
                )
            if bool(
                (~torch.isfinite(spectral_quality)
                 | (spectral_quality <= 0)).any()
            ):
                raise ValueError(
                    "spectral_quality must contain finite positive values"
                )
            grouped_quality = spectral_quality.reshape(
                -1, matched_group_size
            )
            if not torch.allclose(
                grouped_quality,
                grouped_quality[:, :1].expand_as(grouped_quality),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Every matched observation group must share one "
                    "spectral_quality"
                )

        image_noise_sigma = checkpoint_image_noise_sigma(model).to(
            device=device, dtype=snrs.dtype
        )
        reference_line_norm = checkpoint_spectral_reference_line_norm(model).to(
            device=device, dtype=snrs.dtype
        )
        group_mag_measurement = sample_observed_magnitude(
            group_rmag,
            group_expected_snrs,
            randgen=magnitude_randgen,
        )
        mag_measurement = {
            name: torch.repeat_interleave(value, matched_group_size)
            for name, value in group_mag_measurement.items()
        }
        mags = mag_measurement["rmag_obs"]
        mag_sigmas = mag_measurement["rmag_sigma"]
        spectral_noise_scales = reference_line_norm / spectral_quality
        observation_metadata = {
            "image_snr": mag_measurement["image_flux_snr"].detach().cpu().numpy(),
            "spectral_quality": spectral_quality.detach().cpu().numpy(),
            "spectral_noise_scale": spectral_noise_scales.detach().cpu().numpy(),
            "rmag_obs": mags.detach().cpu().numpy(),
            "rmag_sigma": mag_sigmas.detach().cpu().numpy(),
            "image_noise_sigma": float(
                image_noise_sigma.detach().cpu().item()
            ),
            "spectral_reference_line_norm": float(
                reference_line_norm.detach().cpu().item()
            ),
        }
        if tf_inference == "prior_replacement":
            diagnostic_fields = {
                "tf_effective_sample_size": "effective_sample_size",
                "tf_effective_sample_fraction": (
                    "effective_sample_fraction"
                ),
                "tf_max_normalized_weight": "max_normalized_weight",
                "tf_candidate_log_normalizer": (
                    "candidate_log_normalizer"
                ),
            }
            for cache_name in diagnostic_fields:
                observation_metadata[cache_name] = np.full(
                    len(test_ds), np.nan, dtype=np.float64
                )
    else:
        if tf_inference is not None:
            raise ValueError(
                "Explicit tf_inference is reserved for observation model v2"
            )
        if snr is None and randgen is not None:
            snr = (
                torch.rand(
                    len(test_ds), generator=randgen, device=device
                ) * 995 + 5
            )
        snrs = (
            torch.as_tensor(snr, device=device)
            if snr is not None
            else torch.rand((len(test_ds),), device=device) * 995 + 5
        )
        mags = (
            torch.as_tensor(mag, device=device)
            if mag is not None
            else snr_to_app_mag(snrs)
            if getattr(posterior_owner, "mode", None) == 2
            else None
        )
        mag_sigmas = None

    model.eval()
    samples = []
    if return_log_prob:
        log_probs = []
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Sampling")
    with torch.no_grad():
        for i in iterator:
            image_snr_i = snrs[i]
            mag_i = mags[i] if mags is not None else None
            image_noise_gen = (
                image_randgen if observation_model_version == 2 else randgen
            )
            spectrum_noise_gen = spectral_randgen
            if matched_group_size > 1:
                image_noise_gen = _seeded_generator(
                    device, noise_seed + i // matched_group_size
                )
                if observation_model_version == 2:
                    spectrum_noise_gen = _seeded_generator(
                        device,
                        spectral_noise_seed + i // matched_group_size,
                    )
            record = test_ds[i]
            if observation_model_version == 2:
                img = apply_fixed_gaussian_image_noise(
                    record['img'].unsqueeze(0).float().to(device),
                    image_noise_sigma,
                    randgen=image_noise_gen,
                )
                spec = apply_spectral_noise(
                    record['spec'].unsqueeze(0).float().to(device),
                    spectral_quality[i],
                    reference_line_norm,
                    center_fiber_index=int(
                        config.observation["center_fiber_index"]
                    ),
                    center_exposure_s=float(
                        config.observation["center_exposure_s"]
                    ),
                    offset_exposure_s=float(
                        config.observation["offset_exposure_s"]
                    ),
                    spectral_units=config.observation["spectral_units"],
                    randgen=spectrum_noise_gen,
                    device=device,
                )
                observation_context = {
                    "rmag_obs": mag_i,
                    "rmag_sigma": mag_sigmas[i],
                    "image_snr": mag_measurement["image_flux_snr"][i],
                    "spectral_reference_quality": spectral_quality[i],
                    "spectral_noise_scale": spectral_noise_scales[i],
                }
                sample_kwargs = {
                    "mag": mag_i if tf_inference is not None else None,
                    "mag_sigma": (
                        mag_sigmas[i] if tf_inference is not None else None
                    ),
                    "snr": None,
                    "tf_inference": tf_inference,
                    "observation_context": observation_context,
                }
            else:
                img = apply_noise(
                    record['img'].unsqueeze(0).float().to(device),
                    image_snr_i,
                    randgen=image_noise_gen,
                    device=device,
                )
                spec = apply_noise(
                    record['spec'].unsqueeze(0).float().to(device),
                    image_snr_i,
                    randgen=image_noise_gen,
                    device=device,
                )
                sample_kwargs = {
                    "mag": mag_i,
                    "snr": image_snr_i,
                }
            fp = (
                record['fib_pos'].unsqueeze(0).float().to(device)
                if 'fib_pos' in record
                else None
            )
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            if return_log_prob:
                sample, log_prob = model.sample(
                    img,
                    spec,
                    nsamples,
                    fp=fp,
                    return_log_prob=True,
                    sample_id=i,
                    **sample_kwargs,
                )
                log_probs.append(log_prob.detach().cpu().numpy())
            else:
                sample = model.sample(
                    img,
                    spec,
                    nsamples,
                    fp=fp,
                    sample_id=i,
                    **sample_kwargs,
                )
            if tf_inference == "prior_replacement":
                diagnostics = getattr(
                    posterior_owner, "last_tf_inference_diagnostics", None
                )
                if not isinstance(diagnostics, dict):
                    raise RuntimeError(
                        "TF prior replacement did not publish diagnostics"
                    )
                for cache_name, diagnostic_name in diagnostic_fields.items():
                    value = torch.as_tensor(diagnostics[diagnostic_name])
                    if value.numel() != 1:
                        raise RuntimeError(
                            f"Unexpected TF diagnostic {diagnostic_name!r} "
                            f"shape {tuple(value.shape)}"
                        )
                    scalar = float(value.detach().cpu().reshape(()).item())
                    if not np.isfinite(scalar):
                        raise RuntimeError(
                            f"Non-finite TF diagnostic {diagnostic_name!r}"
                        )
                    observation_metadata[cache_name][i] = scalar
            samples.append(sample.detach().cpu().numpy())
            if apply_add_noise_cancellation:
                img, _, fp = rotate_90_degrees(img, fp=fp)
                if channels_last:
                    img = img.contiguous(memory_format=torch.channels_last)
                    spec = spec.contiguous(memory_format=torch.channels_last)
                if return_log_prob:
                    sample, log_prob = model.sample(
                        img,
                        spec,
                        nsamples,
                        fp=fp,
                        return_log_prob=True,
                        sample_id=i,
                        **sample_kwargs,
                    )
                    log_probs.append(log_prob.detach().cpu().numpy())
                else:
                    sample = model.sample(
                        img,
                        spec,
                        nsamples,
                        fp=fp,
                        sample_id=i,
                        **sample_kwargs,
                    )
                samples.append(sample.detach().cpu().numpy())
    samples = np.vstack(samples)
    if apply_add_noise_cancellation:
        samples = pair_rotation_branches(samples)
    snrs = snrs.detach().cpu().numpy()
    if return_log_prob:
        log_probs = np.vstack(log_probs)
        if apply_add_noise_cancellation:
            log_probs = pair_rotation_branches(log_probs)
        if return_observation_metadata:
            return samples, log_probs, snrs, observation_metadata
        return samples, log_probs, snrs
    if return_observation_metadata:
        return samples, snrs, observation_metadata
    return samples, snrs


def evaluate_conditional_2d(
    model,
    test_ds,
    snr=None,
    randgen=None,
    device='cpu',
    channels_last: bool | None = None,
    progress=None,
):
    if channels_last is None:
        channels_last = bool(config.train.get('channels_last', False))
    if snr is None and randgen is not None:
        snr = torch.rand(len(test_ds), generator=randgen, device=device)*995 + 5
    model.eval()
    probs = []
    snrs = snr if snr is not None else torch.rand((len(test_ds),), device=device)*995 + 5
    # snrs = torch.full((len(test_ds),), 1000.0, device=device)
    iterator = range(len(test_ds))
    if progress is not None:
        iterator = progress(iterator, total=len(test_ds), desc="Evaluating")
    with torch.no_grad():
        for i in iterator:
            snr = snrs[i]
            img = apply_noise(test_ds[i]['img'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            spec = apply_noise(test_ds[i]['spec'].unsqueeze(0).float().to(device), snr, randgen=randgen, device=device)
            fids = torch.as_tensor(test_ds[i]['fid_pars'][:model.nfeatures], dtype=torch.float32, device=device).unsqueeze(0)
            fp = test_ds[i]['fib_pos'].unsqueeze(0).float().to(device) if 'fib_pos' in test_ds[i] else None
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            if i == 0:
                prob, g1_vals, g2_vals = model.evaluate_conditional_2d(img, spec, fids, 0, 1, fp=fp, grid_bins=200)
            else:
                prob, _, _ = model.evaluate_conditional_2d(img, spec, fids, 0, 1, fp=fp, grid_bins=200)
            probs.append(prob.unsqueeze(0).detach().cpu().numpy())
    probs = np.vstack(probs)
    g1_vals = g1_vals.cpu().numpy()
    g2_vals = g2_vals.cpu().numpy()
    snrs = snrs.cpu().numpy()
    return probs, g1_vals, g2_vals, snrs
