import argparse
import math
import os
from os.path import join

import torch
import torch.multiprocessing as mp

from networks import *
from train import *
import config
from model_registry import save_model_artifacts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument("--model", default="VICRegPretrain", help="Model architecture to use")
    parser.add_argument("--trainer", default="FETrainer", help="Trainer class to use")
    parser.add_argument(
        "--backbone-type",
        choices=BACKBONE_TYPES,
        help="Feature-extractor architecture (recorded for pretraining and NPE)",
    )
    parser.add_argument(
        "--rot90-counterpart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train a separate rot90 counterpart (disabled for exact D4 backbones)",
    )
    parser.add_argument(
        "--posterior-symmetry",
        choices=("none", "d4"),
        help="Posterior symmetry used during NPE training and inference",
    )
    parser.add_argument(
        "--train-type",
        "--train_type",
        dest="train_type",
        choices=("pretrain", "train"),
        default="pretrain",
        help="Training stage",
    )
    parser.add_argument(
        "--config",
        help="Optional ModelConfig JSON to load before applying command-line overrides",
    )
    parser.add_argument("--seed", type=int, help="Base seed for all random streams")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request deterministic Torch/cuDNN behavior",
    )
    parser.add_argument("--train-data", help="Training LMDB directory")
    parser.add_argument("--valid-data", help="Validation LMDB directory")
    parser.add_argument("--train-size", type=int, help="Recorded training-set size")
    parser.add_argument("--valid-size", type=int, help="Recorded validation-set size")
    parser.add_argument("--model-name", help="Output model and artifact name")
    parser.add_argument(
        "--pretrained-name",
        help="Pretraining model name recorded for downstream NPE construction",
    )
    parser.add_argument(
        "--pretrain-from",
        type=int,
        help="Pretraining checkpoint epoch recorded for downstream NPE construction",
    )
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Per-rank batch size")
    parser.add_argument("--mode", type=int, choices=(0, 1, 2), help="NPE mode")
    parser.add_argument(
        "--flow-type",
        choices=("affine", "circular_rqs"),
        help="NPE flow family (legacy affine or periodic theta_int RQS)",
    )
    parser.add_argument(
        "--flow-num-bins",
        type=int,
        help="Number of rational-quadratic spline bins for circular_rqs",
    )
    parser.add_argument("--initial-learning-rate", type=float, help="Optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, help="Optimizer weight decay")
    parser.add_argument(
        "--ccl-sigma-label",
        type=float,
        help="CCL label-kernel bandwidth in normalized RMS parameter distance",
    )
    parser.add_argument(
        "--ccl-d-cutoff",
        type=float,
        help="CCL background cutoff in normalized RMS parameter distance",
    )
    parser.add_argument(
        "--compile",
        dest="use_compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable torch.compile",
    )
    return parser.parse_args(argv)


def apply_overrides(args):
    if args.config:
        config.load_model_config_from_json(args.config)

    if args.train_type == "pretrain":
        stage_config = config.MODEL_CONFIG.pretrain
    else:
        stage_config = config.MODEL_CONFIG.train

    if args.train_data is not None:
        config.MODEL_CONFIG.data.data_dir = args.train_data
    if args.valid_data is not None:
        config.MODEL_CONFIG.test.data_dir = args.valid_data
    if args.train_size is not None:
        config.MODEL_CONFIG.data.size = args.train_size
    if args.valid_size is not None:
        config.MODEL_CONFIG.test.size = args.valid_size
    if args.backbone_type is not None:
        config.MODEL_CONFIG.pretrain.backbone_type = args.backbone_type
        config.MODEL_CONFIG.train.backbone_type = args.backbone_type
    if args.posterior_symmetry is not None:
        if args.train_type != "train":
            raise ValueError("--posterior-symmetry is only valid for NPE training")
        config.MODEL_CONFIG.train.posterior_symmetry = args.posterior_symmetry
    elif args.train_type == "train" and stage_config.backbone_type == "stage4_d4":
        config.MODEL_CONFIG.train.posterior_symmetry = "d4"

    if args.rot90_counterpart is not None:
        stage_config.use_rot90_counterpart = args.rot90_counterpart
    elif stage_config.backbone_type == "stage4_d4":
        stage_config.use_rot90_counterpart = False
    if args.pretrained_name is not None:
        config.MODEL_CONFIG.train.pretrained_name = args.pretrained_name
    if args.pretrain_from is not None:
        config.MODEL_CONFIG.train.pretrain_from = args.pretrain_from
    if args.flow_type is not None or args.flow_num_bins is not None:
        if args.train_type != "train":
            raise ValueError("flow options are only valid for NPE training")
        if args.flow_type is not None:
            config.MODEL_CONFIG.flow.flow_type = args.flow_type
        if args.flow_num_bins is not None:
            config.MODEL_CONFIG.flow.num_bins = args.flow_num_bins

    overrides = {
        "model_name": args.model_name,
        "epoch_number": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "use_compile": args.use_compile,
        "mode": args.mode,
        "initial_learning_rate": args.initial_learning_rate,
        "weight_decay": args.weight_decay,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(stage_config, name, value)

    ccl_overrides = {
        "ccl_sigma_label": args.ccl_sigma_label,
        "ccl_d_cutoff": args.ccl_d_cutoff,
    }
    if any(value is not None for value in ccl_overrides.values()):
        if args.train_type != "pretrain":
            raise ValueError("CCL options are only valid for --train-type pretrain")
        for name, value in ccl_overrides.items():
            if value is not None:
                setattr(stage_config, name, value)
    if stage_config.epoch_number <= 0:
        raise ValueError("--epochs must be positive")
    if stage_config.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if config.MODEL_CONFIG.data.size <= 0:
        raise ValueError("--train-size must be positive")
    if config.MODEL_CONFIG.test.size <= 0:
        raise ValueError("--valid-size must be positive")
    if config.MODEL_CONFIG.train.pretrain_from < 0:
        raise ValueError("--pretrain-from must be non-negative")
    if not 0 <= stage_config.seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 2**32 - 1")
    if stage_config.initial_learning_rate <= 0 or not math.isfinite(stage_config.initial_learning_rate):
        raise ValueError("--initial-learning-rate must be positive and finite")
    if stage_config.weight_decay < 0 or not math.isfinite(stage_config.weight_decay):
        raise ValueError("--weight-decay must be non-negative and finite")

    if args.train_type == "pretrain":
        is_d4_backbone = stage_config.backbone_type == "stage4_d4"
        if is_d4_backbone == bool(stage_config.use_rot90_counterpart):
            expected = "disabled" if is_d4_backbone else "enabled"
            raise ValueError(
                "rot90 counterpart must be " + expected
                + f" for backbone_type={stage_config.backbone_type!r}"
            )
        if not math.isfinite(stage_config.ccl_sigma_label) or stage_config.ccl_sigma_label <= 0:
            raise ValueError("--ccl-sigma-label must be positive and finite")
        if not math.isfinite(stage_config.ccl_d_cutoff) or stage_config.ccl_d_cutoff <= 0:
            raise ValueError("--ccl-d-cutoff must be positive and finite")
    else:
        if config.MODEL_CONFIG.flow.flow_type not in ("affine", "circular_rqs"):
            raise ValueError("--flow-type must be 'affine' or 'circular_rqs'")
        if config.MODEL_CONFIG.flow.num_bins < 2:
            raise ValueError("--flow-num-bins must be at least 2")
        is_d4_backbone = stage_config.backbone_type == "stage4_d4"
        is_d4_posterior = stage_config.posterior_symmetry == "d4"
        if stage_config.posterior_symmetry not in ("none", "d4"):
            raise ValueError("posterior symmetry must be 'none' or 'd4'")
        if is_d4_backbone != is_d4_posterior:
            raise ValueError(
                "D4 posterior symmetry requires backbone_type='stage4_d4', "
                "and the Stage 4 backbone requires D4 posterior symmetry"
            )
        if is_d4_posterior == bool(stage_config.use_rot90_counterpart):
            expected = "disabled" if is_d4_posterior else "enabled"
            raise ValueError(
                "rot90 counterpart must be " + expected
                + f" for posterior_symmetry={stage_config.posterior_symmetry!r}"
            )

    config.set_model_config(config.MODEL_CONFIG)

    return stage_config


if __name__ == "__main__":
    args = parse_args()
    train_config = apply_overrides(args)

    effective_seed = int(train_config.seed)
    deterministic = bool(train_config.deterministic)
    os.environ["PYTHONHASHSEED"] = str(effective_seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    artifacts = save_model_artifacts(
        config.MODEL_CONFIG,
        train_type=args.train_type,
        overwrite=True,
    )
    print(f"Saved model config JSON: {artifacts['config_path']}")
    print(f"Saved networks snapshot: {artifacts['network_path']}")
    print(
        "Effective run: "
        f"seed={effective_seed}, deterministic={deterministic}, "
        f"compile={train_config.use_compile}, "
        f"backbone={train_config.backbone_type}, "
        f"posterior_symmetry={getattr(train_config, 'posterior_symmetry', None)}, "
        f"rot90_counterpart={getattr(train_config, 'use_rot90_counterpart', None)}, "
        f"flow_type={config.MODEL_CONFIG.flow.flow_type}, "
        f"flow_num_bins={config.MODEL_CONFIG.flow.num_bins}, "
        f"train={config.MODEL_CONFIG.data.data_dir} "
        f"(size={config.MODEL_CONFIG.data.size}), "
        f"valid={config.MODEL_CONFIG.test.data_dir} "
        f"(size={config.MODEL_CONFIG.test.size}), "
        f"ccl_sigma={getattr(train_config, 'ccl_sigma_label', None)}, "
        f"ccl_cutoff={getattr(train_config, 'ccl_d_cutoff', None)}"
    )

    os.makedirs(join(train_config.model_path, train_config.model_name), exist_ok=True)
    try:
        model_class = globals()[args.model]
        trainer_class = globals()[args.trainer]
    except KeyError as exc:
        raise ValueError(f"Unknown model or trainer class: {exc.args[0]}") from exc

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("Training requires at least one visible CUDA device")
    print(f"Training with {world_size} GPUs")

    model_config_payload = config.MODEL_CONFIG.to_dict()
    mp.spawn(
        train_nn,
        args=(
            world_size,
            model_class,
            trainer_class,
            1,
            args.train_type,
            model_config_payload,
        ),
        nprocs=world_size,
    )
