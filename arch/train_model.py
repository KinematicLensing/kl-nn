from __future__ import annotations

import argparse
import math
import os
from os.path import join
from pathlib import Path

import torch
import torch.multiprocessing as mp

try:
    from . import config
    from .model_registry import save_model_artifacts
    from .networks import CCLPretrain, KLNPE
    from .train import FETrainer, NPETrainer, train_nn
except ImportError:  # Direct execution from arch/.
    import config
    from model_registry import save_model_artifacts
    from networks import CCLPretrain, KLNPE
    from train import FETrainer, NPETrainer, train_nn


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the simulator-v3 CNN-CNN-Meta CCL or bounded-hybrid "
            "KL-NN model"
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--stage", choices=("pretrain", "npe"), required=True)
    parser.add_argument("--config", help="Current-schema ModelConfig JSON")
    parser.add_argument("--train-data")
    parser.add_argument("--valid-data")
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--valid-size", type=int)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--model-root",
        help="Shared models directory; sibling configs/ and networks/ hold artifacts",
    )
    parser.add_argument("--pretrained-name")
    parser.add_argument(
        "--pretrain-from",
        help="CCL checkpoint suffix: a non-negative epoch index or 'best'",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--compile", dest="use_compile",
        action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--amp", dest="use_amp",
        action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16")
    )
    parser.add_argument(
        "--fixed-validation-streams",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--initial-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--ccl-sigma-label", type=float)
    parser.add_argument("--ccl-d-cutoff", type=float)
    parser.add_argument("--flow-num-layers", type=int)
    parser.add_argument("--flow-num-bins", type=int)
    parser.add_argument("--theta-num-layers", type=int)
    parser.add_argument("--theta-logit-limit", type=float)
    parser.add_argument("--bounded-logit-limit", type=float)
    parser.add_argument("--non-theta-learning-rate", type=float)
    parser.add_argument("--theta-learning-rate", type=float)
    parser.add_argument(
        "--scheduler-type", choices=("plateau", "cosine", "warmup_cosine")
    )
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument(
        "--feature-norm-trainable",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Freeze the pretrained CNN backbone during NPE (default). "
            "Pass --no-freeze-feature-extractor to fine-tune it with the flow."
        ),
    )
    parser.add_argument(
        "--image-spectrum-fusion",
        dest="use_image_spectrum_fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply bidirectional FiLM to the concatenated image and spectral "
            "512-d branches during NPE (default). Pass "
            "--no-image-spectrum-fusion for concat-only NPE."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--early-stopping-min-delta", type=float)
    parser.add_argument("--gradient-clip-norm", type=float)
    return parser.parse_args(argv)


def _set_if_not_none(owner, name, value):
    if value is not None:
        setattr(owner, name, value)


def _parse_checkpoint_suffix(value):
    if value is None:
        return None
    if str(value).lower() == "best":
        return "best"
    try:
        epoch = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("pretrain-from must be 'best' or a non-negative integer") from error
    if epoch < 0:
        raise ValueError("pretrain-from must be 'best' or a non-negative integer")
    return epoch


def apply_overrides(args):
    if args.config:
        config.load_model_config_from_json(args.config)

    stage = (
        config.MODEL_CONFIG.pretrain
        if args.stage == "pretrain"
        else config.MODEL_CONFIG.train
    )
    _set_if_not_none(config.MODEL_CONFIG.data, "data_dir", args.train_data)
    _set_if_not_none(config.MODEL_CONFIG.test, "data_dir", args.valid_data)
    _set_if_not_none(config.MODEL_CONFIG.data, "size", args.train_size)
    _set_if_not_none(config.MODEL_CONFIG.test, "size", args.valid_size)
    _set_if_not_none(stage, "model_name", args.model_name)
    if args.model_root is not None:
        model_root = str(Path(args.model_root).expanduser().resolve())
        config.MODEL_CONFIG.pretrain.model_path = model_root
        config.MODEL_CONFIG.train.model_path = model_root
    _set_if_not_none(stage, "epoch_number", args.epochs)
    _set_if_not_none(stage, "batch_size", args.batch_size)
    _set_if_not_none(stage, "seed", args.seed)
    _set_if_not_none(stage, "deterministic", args.deterministic)
    _set_if_not_none(stage, "use_compile", args.use_compile)
    _set_if_not_none(stage, "use_amp", args.use_amp)
    _set_if_not_none(stage, "amp_dtype", args.amp_dtype)
    _set_if_not_none(
        stage, "fixed_validation_streams", args.fixed_validation_streams
    )
    _set_if_not_none(
        stage, "initial_learning_rate", args.initial_learning_rate
    )
    _set_if_not_none(stage, "weight_decay", args.weight_decay)

    if args.stage == "pretrain":
        invalid_npe = any(
            value is not None
            for value in (
                args.pretrained_name,
                args.pretrain_from,
                args.flow_num_layers,
                args.flow_num_bins,
                args.theta_num_layers,
                args.theta_logit_limit,
                args.bounded_logit_limit,
                args.non_theta_learning_rate,
                args.theta_learning_rate,
                args.scheduler_type,
                args.warmup_epochs,
                args.min_learning_rate,
                args.feature_norm_trainable,
                args.freeze_feature_extractor,
                args.use_image_spectrum_fusion,
                args.early_stopping_patience,
                args.early_stopping_min_delta,
                args.gradient_clip_norm,
            )
        )
        if invalid_npe:
            raise ValueError("NPE-only options cannot be used during pretraining")
        _set_if_not_none(stage, "ccl_sigma_label", args.ccl_sigma_label)
        _set_if_not_none(stage, "ccl_d_cutoff", args.ccl_d_cutoff)
    else:
        if args.ccl_sigma_label is not None or args.ccl_d_cutoff is not None:
            raise ValueError("CCL options cannot be used during NPE training")
        _set_if_not_none(stage, "pretrained_name", args.pretrained_name)
        _set_if_not_none(
            stage, "pretrain_from", _parse_checkpoint_suffix(args.pretrain_from)
        )
        flow = config.MODEL_CONFIG.flow
        _set_if_not_none(flow, "num_layers", args.flow_num_layers)
        _set_if_not_none(flow, "num_bins", args.flow_num_bins)
        _set_if_not_none(flow, "theta_num_layers", args.theta_num_layers)
        _set_if_not_none(flow, "theta_logit_limit", args.theta_logit_limit)
        _set_if_not_none(flow, "bounded_logit_limit", args.bounded_logit_limit)
        for name in (
            "non_theta_learning_rate",
            "theta_learning_rate",
            "scheduler_type",
            "warmup_epochs",
            "min_learning_rate",
            "feature_norm_trainable",
            "freeze_feature_extractor",
            "use_image_spectrum_fusion",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "gradient_clip_norm",
        ):
            _set_if_not_none(stage, name, getattr(args, name))

    _validate_current_config(args.stage)
    config.set_model_config(config.MODEL_CONFIG)
    return stage


def _positive_finite(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _validate_current_config(stage_name):
    model = config.MODEL_CONFIG
    stage = model.pretrain if stage_name == "pretrain" else model.train
    if model.data.size <= 0 or model.test.size <= 0:
        raise ValueError("dataset sizes must be positive")
    if stage.epoch_number <= 0 or stage.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
    if stage.amp_dtype not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
    if not 0 <= stage.seed <= 2**32 - 1:
        raise ValueError("seed must lie in [0, 2**32-1]")
    _positive_finite(stage.initial_learning_rate, "initial learning rate")
    if not math.isfinite(stage.weight_decay) or stage.weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")

    observation = model.observation
    if observation.schema_version != 3:
        raise ValueError("only simulator schema version 3 is supported")
    if observation.fiber_layout != "galaxy_axis":
        raise ValueError("the current pipeline requires galaxy_axis fibers")
    if tuple(observation.context_fields) != config.ORACLE_CONTEXT_FIELDS:
        raise ValueError("the current pipeline requires the three oracle contexts")
    if observation.halpha_flux_distribution != "log_uniform":
        raise ValueError("H-alpha proposal must be log-uniform")
    if observation.halpha_flux_transform != "log10":
        raise ValueError("H-alpha target transform must be log10")
    _positive_finite(observation.halpha_flux_min, "H-alpha minimum")
    if observation.halpha_flux_max <= observation.halpha_flux_min:
        raise ValueError("H-alpha bounds must be increasing")
    for name in ("image_snr", "central_halpha_snr"):
        if getattr(observation, f"{name}_distribution") != "uniform":
            raise ValueError(f"{name} proposal must be uniform")
        minimum = getattr(observation, f"{name}_min")
        maximum = getattr(observation, f"{name}_max")
        _positive_finite(minimum, f"{name} minimum")
        if maximum <= minimum:
            raise ValueError(f"{name} bounds must be increasing")

    if stage_name == "pretrain":
        _positive_finite(stage.ccl_sigma_label, "CCL sigma")
        _positive_finite(stage.ccl_d_cutoff, "CCL cutoff")
        return

    if not (
        stage.pretrain_from == "best"
        or (
            isinstance(stage.pretrain_from, int)
            and not isinstance(stage.pretrain_from, bool)
            and stage.pretrain_from >= 0
        )
    ):
        raise ValueError("pretrain_from must be 'best' or a non-negative integer")

    flow = model.flow
    if flow.num_layers < 1 or flow.num_bins < 2 or flow.theta_num_layers < 1:
        raise ValueError("flow layers must be positive and bins at least two")
    _positive_finite(flow.theta_logit_limit, "theta logit limit")
    _positive_finite(flow.bounded_logit_limit, "bounded logit limit")
    for name in ("non_theta_learning_rate", "theta_learning_rate"):
        value = getattr(stage, name)
        if value is not None:
            _positive_finite(value, name)
    if stage.warmup_epochs < 0:
        raise ValueError("warmup epochs must be non-negative")
    if stage.early_stopping_patience is not None and stage.early_stopping_patience < 1:
        raise ValueError("early stopping patience must be positive")
    _positive_finite(stage.gradient_clip_norm, "gradient clip norm")


def main(argv=None):
    args = parse_args(argv)
    stage_config = apply_overrides(args)
    seed = int(stage_config.seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if stage_config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    shared_root = Path(stage_config.model_path).expanduser().resolve().parent
    artifacts = save_model_artifacts(
        config.MODEL_CONFIG,
        train_type="pretrain" if args.stage == "pretrain" else "train",
        configs_root=str(shared_root / "configs"),
        networks_root=str(shared_root / "networks"),
        overwrite=True,
    )
    print(f"Saved model config: {artifacts['config_path']}")
    print(f"Saved network source: {artifacts['network_path']}")
    view_mode = (
        "identity+r90" if args.stage == "pretrain" else "random(identity,r90)"
    )
    print(
        "Current pipeline: "
        f"stage={args.stage}, targets={len(config.TARGET_NAMES)}, "
        f"views={view_mode}, contexts={list(config.ORACLE_CONTEXT_FIELDS)}, "
        f"train={config.MODEL_CONFIG.data.data_dir}, "
        f"valid={config.MODEL_CONFIG.test.data_dir}"
    )

    os.makedirs(join(stage_config.model_path, stage_config.model_name), exist_ok=True)
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Training requires at least one visible CUDA device")
    model_class = CCLPretrain if args.stage == "pretrain" else KLNPE
    trainer_class = FETrainer if args.stage == "pretrain" else NPETrainer
    mp.spawn(
        train_nn,
        args=(
            world_size,
            model_class,
            trainer_class,
            1,
            "pretrain" if args.stage == "pretrain" else "train",
            config.MODEL_CONFIG.to_dict(),
        ),
        nprocs=world_size,
    )


if __name__ == "__main__":
    main()
