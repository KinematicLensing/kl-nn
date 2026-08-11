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
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Per-rank batch size")
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
        "--ccl-objective",
        choices=("ccl", "ccl_shear"),
        help="Use the original CCL objective or CCL plus a backbone shear head",
    )
    parser.add_argument(
        "--ccl-shear-loss-weight",
        type=float,
        help="Weight of the normalized g1/g2 MSE in the ccl_shear objective",
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

    overrides = {
        "model_name": args.model_name,
        "epoch_number": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "use_compile": args.use_compile,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(stage_config, name, value)

    ccl_overrides = {
        "ccl_sigma_label": args.ccl_sigma_label,
        "ccl_d_cutoff": args.ccl_d_cutoff,
        "ccl_objective": args.ccl_objective,
        "ccl_shear_loss_weight": args.ccl_shear_loss_weight,
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
    if not 0 <= stage_config.seed <= 2**32 - 1:
        raise ValueError("--seed must be between 0 and 2**32 - 1")

    if args.train_type == "pretrain":
        if not math.isfinite(stage_config.ccl_sigma_label) or stage_config.ccl_sigma_label <= 0:
            raise ValueError("--ccl-sigma-label must be positive and finite")
        if not math.isfinite(stage_config.ccl_d_cutoff) or stage_config.ccl_d_cutoff <= 0:
            raise ValueError("--ccl-d-cutoff must be positive and finite")
        if stage_config.ccl_objective not in ("ccl", "ccl_shear"):
            raise ValueError("--ccl-objective must be 'ccl' or 'ccl_shear'")
        if (
            not math.isfinite(stage_config.ccl_shear_loss_weight)
            or stage_config.ccl_shear_loss_weight < 0
        ):
            raise ValueError("--ccl-shear-loss-weight must be non-negative and finite")

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
        f"train={config.MODEL_CONFIG.data.data_dir} "
        f"(size={config.MODEL_CONFIG.data.size}), "
        f"valid={config.MODEL_CONFIG.test.data_dir} "
        f"(size={config.MODEL_CONFIG.test.size}), "
        f"ccl_sigma={getattr(train_config, 'ccl_sigma_label', None)}, "
        f"ccl_cutoff={getattr(train_config, 'ccl_d_cutoff', None)}, "
        f"ccl_objective={getattr(train_config, 'ccl_objective', None)}, "
        f"ccl_shear_weight={getattr(train_config, 'ccl_shear_loss_weight', None)}"
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
