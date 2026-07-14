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
    enable_handedness_flip: bool = False
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlowConfig:
    num_layers: int
    mlp: list[int] = field(default_factory=lambda: [1, 128, 64, 2])

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
    train: TrainConfig
    flow: FlowConfig
    tf: TFConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            rmag_snr_source_path=payload["rmag_snr_source_path"],
            rmag_snr_fit_path=payload["rmag_snr_fit_path"],
            data=DatasetConfig(**payload["data"]),
            test=DatasetConfig(**payload["test"]),
            par_ranges={k: [float(v[0]), float(v[1])] for k, v in payload["par_ranges"].items()},
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
        },
        train=TrainConfig(
            mode=2,  # 0: point estimate; 1: density estimate; 2: density estimate with TF prior
            epoch_number=200,
            initial_learning_rate=1e-4,
            weight_decay=1e-5,
            batch_size=100,
            feature_number=8,
            feature_names=["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            save_model=True,
            model_path="/ocean/projects/phy250048p/shared/models/",
            model_name="ViT-CNN-flow_tf_train",
            enable_handedness_flip=False,
            use_pretrain=False,
            pretrained_name="CNN-CNN-flow_all_params",
            pretrain_from=99,
            use_amp=True,
            amp_dtype="float16",
            use_compile=True,
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
    global rmag_snr_source_path, rmag_snr_fit_path, data, test, par_ranges, train, flow, tf

    rmag_snr_source_path = model_config.rmag_snr_source_path
    rmag_snr_fit_path = model_config.rmag_snr_fit_path
    data = model_config.data.to_dict()
    test = model_config.test.to_dict()
    par_ranges = model_config.par_ranges.copy()
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
