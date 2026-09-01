from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np


TARGET_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "cosi",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
    "halpha_flux_true",
)

CANONICAL_PARAMETER_RANGES = {
    "g1": [-0.1, 0.1],
    "g2": [-0.1, 0.1],
    "theta_int": [-float(np.pi), float(np.pi)],
    "cosi": [0.0, 1.0],
    "v0": [-30.0, 30.0],
    "vcirc": [60.0, 540.0],
    "rscale": [0.1, 5.0],
    "hlr": [0.1, 5.0],
    "halpha_flux_true": [1.0e-17, 1.0e-14],
}

TARGET_TRANSFORMS = {
    name: "log10" if name == "halpha_flux_true" else "identity"
    for name in TARGET_NAMES
}

ORACLE_CONTEXT_FIELDS = (
    "rmag_true",
    "image_snr",
    "central_halpha_snr",
)


@dataclass
class DatasetConfig:
    size: int
    nspec: int
    data_dir: str


@dataclass
class PretrainConfig:
    epoch_number: int = 100
    initial_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    eps: float = 1e-8
    batch_size: int = 100
    model_path: str = "/ocean/projects/phy250048p/shared/models/"
    model_name: str = "CNN-CNN-Meta-CCL-simv3-r90"
    use_amp: bool = False
    amp_dtype: str = "float16"
    use_compile: bool = False
    compile_mode: str = "default"
    compile_backend: str | None = None
    use_fused_adamw: bool = True
    channels_last: bool = True
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_broadcast_buffers: bool = False
    ccl_temperature: float = 0.1
    ccl_sigma_label: float = 0.15
    ccl_d_cutoff: float = 0.40
    ccl_label_scales: dict[str, float] = field(default_factory=dict)
    ccl_distance_reduction: str = "mean"
    seed: int = 42
    deterministic: bool = False
    fixed_validation_streams: bool = True


@dataclass
class TrainConfig:
    epoch_number: int = 200
    initial_learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 50
    feature_names: list[str] = field(default_factory=lambda: list(TARGET_NAMES))
    model_path: str = "/ocean/projects/phy250048p/shared/models/"
    model_name: str = "CNN-CNN-Meta-bounded-hybrid-simv3-r90"
    pretrained_name: str = "CNN-CNN-Meta-CCL-simv3-r90"
    pretrain_from: int | str = "best"
    use_amp: bool = False
    amp_dtype: str = "float16"
    use_compile: bool = False
    compile_mode: str = "default"
    compile_backend: str | None = None
    use_fused_adamw: bool = True
    channels_last: bool = True
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = False
    ddp_broadcast_buffers: bool = False
    seed: int = 42
    deterministic: bool = False
    scheduler_type: str = "warmup_cosine"
    warmup_epochs: int = 2
    min_learning_rate: float = 1e-6
    fixed_validation_streams: bool = True
    feature_norm_trainable: bool = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: float = 1.0
    non_theta_learning_rate: float | None = None
    theta_learning_rate: float | None = None


@dataclass
class FlowConfig:
    num_layers: int = 4
    num_bins: int = 8
    theta_num_layers: int = 1
    theta_logit_limit: float = 10.0
    bounded_logit_limit: float = 10.0


@dataclass
class ObservationConfig:
    """The sole supported simulator-v3 observation and metadata schema."""

    schema_version: int = 3
    fiber_layout: str = "galaxy_axis"
    context_fields: list[str] = field(
        default_factory=lambda: list(ORACLE_CONTEXT_FIELDS)
    )
    rmag_min: float = 15.0
    rmag_max: float = 23.4
    halpha_flux_min: float = 1.0e-17
    halpha_flux_max: float = 1.0e-14
    halpha_flux_distribution: str = "log_uniform"
    halpha_flux_units: str = "erg s^-1 cm^-2"
    halpha_flux_semantics: str = (
        "central_fiber_integrated_after_seeing_before_instrument"
    )
    halpha_flux_transform: str = "log10"
    image_band: str = "r"
    target_line: str = "Ha"
    image_reference_psf_fwhm_arcsec: float = 1.0
    image_pixel_scale_arcsec: float = 0.2637
    image_snr_min: float = 10.0
    image_snr_max: float = 1000.0
    image_snr_distribution: str = "uniform"
    central_halpha_snr_min: float = 1.0
    central_halpha_snr_max: float = 150.0
    central_halpha_snr_distribution: str = "uniform"
    spectral_units: str = "counts"
    center_fiber_index: int = 2
    center_exposure_s: float = 180.0
    offset_exposure_s: float = 600.0


@dataclass
class ModelConfig:
    data: DatasetConfig
    test: DatasetConfig
    par_ranges: dict[str, list[float]]
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def __post_init__(self) -> None:
        names = tuple(self.train.feature_names)
        if names != TARGET_NAMES:
            raise ValueError(
                "feature_names must exactly match the current nine-target schema: "
                f"{TARGET_NAMES!r}"
            )
        if tuple(self.par_ranges) != TARGET_NAMES:
            raise ValueError(
                "par_ranges must use the current target order: "
                f"{TARGET_NAMES!r}"
            )
        canonical_ranges = {
            name: [float(value) for value in bounds]
            for name, bounds in CANONICAL_PARAMETER_RANGES.items()
        }
        if self.par_ranges != canonical_ranges:
            raise ValueError(
                "par_ranges are immutable for the current normalized LMDB schema; "
                f"expected {canonical_ranges!r}"
            )
        if tuple(self.observation.context_fields) != ORACLE_CONTEXT_FIELDS:
            raise ValueError(
                "context_fields must contain only the independent oracle fields: "
                f"{ORACLE_CONTEXT_FIELDS!r}"
            )
        if self.data.nspec != 5 or self.test.nspec != 5:
            raise ValueError(
                "the current simulator schema requires five spectra per galaxy"
            )
        observation = self.observation
        if observation.schema_version != 3:
            raise ValueError("only simulator schema version 3 is supported")
        if observation.fiber_layout != "galaxy_axis":
            raise ValueError("the current pipeline requires galaxy_axis fibers")
        if observation.image_band != "r" or observation.target_line != "Ha":
            raise ValueError("the current observation schema requires r band and H-alpha")
        if observation.spectral_units != "counts":
            raise ValueError("the current spectral schema requires count units")
        if observation.halpha_flux_distribution != "log_uniform":
            raise ValueError("the H-alpha proposal must be log-uniform")
        if observation.halpha_flux_units != "erg s^-1 cm^-2":
            raise ValueError("H-alpha must use integrated flux units")
        if observation.halpha_flux_semantics != (
            "central_fiber_integrated_after_seeing_before_instrument"
        ):
            raise ValueError("H-alpha must use central-fiber flux semantics")
        if observation.halpha_flux_transform != "log10":
            raise ValueError("H-alpha target normalization must use log10 flux")
        if observation.image_snr_distribution != "uniform":
            raise ValueError("the image-S/N proposal must be uniform")
        if observation.central_halpha_snr_distribution != "uniform":
            raise ValueError("the central H-alpha S/N proposal must be uniform")
        fixed_simulator_metadata = {
            "rmag_min": 15.0,
            "rmag_max": 23.4,
            "image_reference_psf_fwhm_arcsec": 1.0,
            "image_pixel_scale_arcsec": 0.2637,
            "center_fiber_index": 2,
            "center_exposure_s": 180.0,
            "offset_exposure_s": 600.0,
        }
        for name, expected in fixed_simulator_metadata.items():
            if getattr(observation, name) != expected:
                raise ValueError(
                    f"observation {name} is fixed by simulator schema v3 at "
                    f"{expected!r}"
                )
        if (
            observation.halpha_flux_min
            != canonical_ranges["halpha_flux_true"][0]
            or observation.halpha_flux_max
            != canonical_ranges["halpha_flux_true"][1]
        ):
            raise ValueError(
                "observation H-alpha bounds must match the immutable target bounds"
            )
        for name in (
            "rmag_min",
            "rmag_max",
            "image_reference_psf_fwhm_arcsec",
            "image_pixel_scale_arcsec",
            "image_snr_min",
            "image_snr_max",
            "central_halpha_snr_min",
            "central_halpha_snr_max",
            "center_exposure_s",
            "offset_exposure_s",
        ):
            if not np.isfinite(getattr(observation, name)):
                raise ValueError(f"observation {name} must be finite")
        if observation.image_snr_min <= 0 or (
            observation.image_snr_min >= observation.image_snr_max
        ):
            raise ValueError("image-S/N bounds must be positive and increasing")
        if observation.central_halpha_snr_min <= 0 or (
            observation.central_halpha_snr_min
            >= observation.central_halpha_snr_max
        ):
            raise ValueError(
                "central H-alpha S/N bounds must be positive and increasing"
            )
        if not (
            self.train.pretrain_from == "best"
            or (
                isinstance(self.train.pretrain_from, int)
                and not isinstance(self.train.pretrain_from, bool)
                and self.train.pretrain_from >= 0
            )
        ):
            raise ValueError(
                "train.pretrain_from must be 'best' or a non-negative integer"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        """Load only the current schema; archived compatibility is absent."""
        expected = {
            "data",
            "test",
            "par_ranges",
            "pretrain",
            "train",
            "flow",
            "observation",
        }
        supplied = set(payload)
        if supplied != expected:
            raise ValueError(
                "ModelConfig keys must exactly match the current schema; "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}"
            )
        nested_types = {
            "data": DatasetConfig,
            "test": DatasetConfig,
            "pretrain": PretrainConfig,
            "train": TrainConfig,
            "flow": FlowConfig,
            "observation": ObservationConfig,
        }
        for name, nested_type in nested_types.items():
            value = payload[name]
            if not isinstance(value, dict):
                raise ValueError(f"ModelConfig {name!r} must be an object")
            expected_fields = {item.name for item in fields(nested_type)}
            supplied_fields = set(value)
            if supplied_fields != expected_fields:
                raise ValueError(
                    f"ModelConfig {name!r} keys must exactly match the current "
                    f"schema; missing={sorted(expected_fields - supplied_fields)}, "
                    f"extra={sorted(supplied_fields - expected_fields)}"
                )
        for name, bounds in payload["par_ranges"].items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError(f"par_ranges[{name!r}] must contain exactly two bounds")
        return cls(
            data=DatasetConfig(**payload["data"]),
            test=DatasetConfig(**payload["test"]),
            par_ranges={
                name: [float(bounds[0]), float(bounds[1])]
                for name, bounds in payload["par_ranges"].items()
            },
            pretrain=PretrainConfig(**payload["pretrain"]),
            train=TrainConfig(**payload["train"]),
            flow=FlowConfig(**payload["flow"]),
            observation=ObservationConfig(**payload["observation"]),
        )

    def to_json(self, path: str, *, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=indent)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


def _default_model_config() -> ModelConfig:
    return ModelConfig(
        data=DatasetConfig(
            size=1_000_000,
            nspec=5,
            data_dir=(
                "/ocean/projects/phy250048p/shared/datasets/"
                "train_1m_simv3_galaxyaxis_central_halpha/"
            ),
        ),
        test=DatasetConfig(
            size=100_000,
            nspec=5,
            data_dir=(
                "/ocean/projects/phy250048p/shared/datasets/"
                "valid_100k_simv3_galaxyaxis_central_halpha/"
            ),
        ),
        par_ranges={
            name: list(bounds)
            for name, bounds in CANONICAL_PARAMETER_RANGES.items()
        },
    )


MODEL_CONFIG: ModelConfig = _default_model_config()


def _publish_runtime_views(model_config: ModelConfig) -> None:
    """Publish current dataclass values as runtime dictionaries."""
    global data, test, par_ranges, target_transforms, pretrain, train, flow, observation
    data = asdict(model_config.data)
    test = asdict(model_config.test)
    par_ranges = {
        name: list(bounds) for name, bounds in model_config.par_ranges.items()
    }
    target_transforms = dict(TARGET_TRANSFORMS)
    pretrain = asdict(model_config.pretrain)
    train = asdict(model_config.train)
    train["feature_number"] = len(model_config.train.feature_names)
    flow = asdict(model_config.flow)
    observation = asdict(model_config.observation)


def set_model_config(model_config: ModelConfig) -> None:
    global MODEL_CONFIG
    MODEL_CONFIG = model_config
    _publish_runtime_views(model_config)


def load_model_config_from_json(path: str) -> ModelConfig:
    model_config = ModelConfig.from_json(path)
    set_model_config(model_config)
    return model_config


_publish_runtime_views(MODEL_CONFIG)
