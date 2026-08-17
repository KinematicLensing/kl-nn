from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from os.path import dirname, join
from typing import Any

import numpy as np


@dataclass
class DatasetConfig:
    size: int
    nimg: int
    nspec: int
    data_dir: str
    data_stem: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class PretrainConfig:
    epoch_number: int
    initial_learning_rate: float
    weight_decay: float
    eps: float
    batch_size: int
    save_model: bool
    model_path: str
    model_name: str
    backbone_type: str = "legacy"
    use_rot90_counterpart: bool = True
    use_amp: bool = True
    amp_dtype: str = "float16"
    use_compile: bool = True
    compile_mode: str = "default"
    compile_backend: str | None = None
    use_fused_adamw: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_broadcast_buffers: bool = False
    noise_cache_maxs: bool = True
    ccl_temperature: float = 0.1
    ccl_sigma_label: float = 0.15
    ccl_d_cutoff: float = 0.40
    ccl_label_scales: dict[str, float] = field(default_factory=dict)
    ccl_distance_reduction: str = "mean"
    seed: int = 42
    deterministic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class TrainConfig:
    mode: int
    epoch_number: int
    initial_learning_rate: float
    weight_decay: float
    batch_size: int
    feature_number: int
    feature_names: list[str]
    save_model: bool
    model_path: str
    model_name: str
    use_pretrain: bool
    pretrained_name: str
    pretrain_from: int
    backbone_type: str = "legacy"
    posterior_symmetry: str = "none"
    use_rot90_counterpart: bool = True
    use_amp: bool = True
    amp_dtype: str = "float16"
    use_compile: bool = True
    compile_mode: str = "default"
    compile_backend: str | None = None
    use_fused_adamw: bool = True
    cudnn_benchmark: bool = True
    channels_last: bool = True
    ddp_static_graph: bool = True
    ddp_find_unused_parameters: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_broadcast_buffers: bool = False
    noise_cache_maxs: bool = True
    seed: int = 42
    deterministic: bool = False
    # Keep the historical optimizer/scheduler behavior as defaults so archived
    # JSON configs load without silently changing their training semantics.
    scheduler_type: str = "plateau"
    warmup_epochs: int = 0
    min_learning_rate: float = 1e-6
    fixed_validation_streams: bool = False
    context_norm_trainable: bool = True
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: float = 1.0
    affine_learning_rate: float | None = None
    theta_learning_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlowConfig:
    num_layers: int
    mlp: list[int] = field(default_factory=lambda: [1, 128, 64, 2])
    # ``affine`` preserves the historical Euclidean NPE. ``circular_rqs``
    # applies RQS transforms to the joint vector. ``hybrid_circular`` retains
    # the affine seven-dimensional flow and adds a conditional circular theta
    # factor. ``bounded_hybrid_circular`` replaces that affine marginal with a
    # compact RQS marginal while retaining the correlated circular factor.
    flow_type: str = "affine"
    num_bins: int = 8
    theta_num_layers: int = 1
    theta_logit_limit: float = 10.0
    bounded_logit_limit: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
@dataclass
class TFConfig:
    slope: float = -7.22
    intercept: float = 36.0
    scatter: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class ModelConfig:
    rmag_snr_source_path: str
    rmag_snr_fit_path: str
    data: DatasetConfig
    test: DatasetConfig
    par_ranges: dict[str, list[float]]
    pretrain: PretrainConfig
    train: TrainConfig
    flow: FlowConfig
    tf: TFConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        pretrain_payload = dict(payload["pretrain"])
        archived_objective = pretrain_payload.pop("ccl_objective", None)
        archived_shear_weight = pretrain_payload.pop("ccl_shear_loss_weight", None)
        pretrain_config = PretrainConfig(**pretrain_payload)
        # Archived network snapshots for the retired auxiliary experiment still
        # read these legacy dictionary keys. Dynamic attributes keep those old
        # checkpoints loadable without serializing retired options in new configs.
        if archived_objective not in (None, "ccl"):
            pretrain_config._archived_ccl_objective = archived_objective
            pretrain_config._archived_ccl_shear_loss_weight = (
                1.0 if archived_shear_weight is None else archived_shear_weight
            )

        return cls(
            rmag_snr_source_path=payload["rmag_snr_source_path"],
            rmag_snr_fit_path=payload["rmag_snr_fit_path"],
            data=DatasetConfig(**payload["data"]),
            test=DatasetConfig(**payload["test"]),
            par_ranges={k: [float(v[0]), float(v[1])] for k, v in payload["par_ranges"].items()},
            pretrain=pretrain_config,
            train=TrainConfig(**payload["train"]),
            flow=FlowConfig(**payload["flow"]),
            tf=TFConfig(**payload["tf"])
        )

    def to_json(self, path: str, *, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(self.to_dict(), fobj, indent=indent)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as fobj:
            payload = json.load(fobj)
        return cls.from_dict(payload)


def _default_model_config() -> ModelConfig:
    return ModelConfig(
        rmag_snr_source_path="/ocean/projects/phy250048p/shared/temp/rmag_snr_pv.npz",
        rmag_snr_fit_path=join(dirname(__file__), "rmag_snr_fit.npz"),
        data=DatasetConfig(
            size=1000000,
            nimg=1,
            nspec=5,
            data_dir="/ocean/projects/phy250048p/shared/datasets/train_1m/",
            data_stem="gal_",
        ),
        test=DatasetConfig(
            size=100000,
            nimg=1,
            nspec=5,
            data_dir="/ocean/projects/phy250048p/shared/datasets/valid_1m/",
            data_stem="gal_",
        ),
        par_ranges={
            "g1": [-0.1, 0.1],
            "g2": [-0.1, 0.1],
            "theta_int": [-float(np.pi), float(np.pi)],
            "sini": [0.0, 1.0],
            "v0": [-30.0, 30.0],
            "vcirc": [60.0, 540.0],
            "rscale": [0.1, 2.0],
            "hlr": [0.1, 3.0],
            # "dx_disk": [-0.5, 0.5],
            # "dy_disk": [-0.5, 0.5],
            # "dx_spec": [-0.5, 0.5],
            # "dy_spec": [-0.5, 0.5],
        },
        pretrain=PretrainConfig(
            epoch_number=100,
            initial_learning_rate=1e-3,
            weight_decay=1e-4,
            eps=1e-8,
            batch_size=100,
            save_model=True,
            model_path="/ocean/projects/phy250048p/shared/models/",
            model_name="CNN-CNN_CCL_rot90",
            use_amp=False,
            amp_dtype="float16",
            use_compile=False,
            compile_mode="default",
            compile_backend=None,
            use_fused_adamw=True,
            cudnn_benchmark=True,
            channels_last=True,
            ddp_static_graph=True,
            ddp_find_unused_parameters=False,
            ddp_gradient_as_bucket_view=True,
            ddp_broadcast_buffers=False,
            noise_cache_maxs=True
        ),
        train=TrainConfig(
            mode=2,  # 0: point estimate; 1: density estimate; 2: density estimate with TF prior
            epoch_number=200,
            initial_learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=50,
            feature_number=8,
            feature_names=["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            save_model=True,
            model_path="/ocean/projects/phy250048p/shared/models/",
            model_name="CNN-CNN-flow_CCL_rot90",
            use_pretrain=True,
            pretrained_name="CNN-CNN_CCL_rot90",
            pretrain_from=73,
            use_amp=False,
            amp_dtype="float16",
            use_compile=False,
            compile_mode="default",
            compile_backend=None,
            use_fused_adamw=True,
            cudnn_benchmark=True,
            channels_last=True,
            ddp_static_graph=True,
            ddp_find_unused_parameters=False,
            ddp_gradient_as_bucket_view=False,
            ddp_broadcast_buffers=False,
            noise_cache_maxs=True,
        ),
        flow=FlowConfig(num_layers=12),
        tf=TFConfig(slope=-7.22, intercept=36.0, scatter=0.1)
    )


MODEL_CONFIG: ModelConfig = _default_model_config()


def _sync_legacy_globals(model_config: ModelConfig) -> None:
    global rmag_snr_source_path, rmag_snr_fit_path, data, test, par_ranges, pretrain, train, flow, tf

    rmag_snr_source_path = model_config.rmag_snr_source_path
    rmag_snr_fit_path = model_config.rmag_snr_fit_path
    data = model_config.data.to_dict()
    test = model_config.test.to_dict()
    par_ranges = model_config.par_ranges.copy()
    pretrain = model_config.pretrain.to_dict()
    archived_objective = getattr(
        model_config.pretrain, "_archived_ccl_objective", None
    )
    if archived_objective is not None:
        pretrain["ccl_objective"] = archived_objective
        pretrain["ccl_shear_loss_weight"] = getattr(
            model_config.pretrain,
            "_archived_ccl_shear_loss_weight",
            1.0,
        )
    train = model_config.train.to_dict()
    flow = model_config.flow.to_dict()
    tf = model_config.tf.to_dict()


def set_model_config(model_config: ModelConfig) -> None:
    global MODEL_CONFIG
    MODEL_CONFIG = model_config
    _sync_legacy_globals(MODEL_CONFIG)


def load_model_config_from_json(path: str) -> ModelConfig:
    model_config = ModelConfig.from_json(path)
    set_model_config(model_config)
    return model_config


_sync_legacy_globals(MODEL_CONFIG)
