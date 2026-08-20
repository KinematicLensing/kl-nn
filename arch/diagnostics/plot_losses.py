#!/usr/bin/env python3
"""Plot training and validation losses for a trained model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import join

# Add parent directory (arch/) to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

import config


DEFAULT_FIGURE_ROOT = "/ocean/projects/phy250048p/shared/figures"
DEFAULT_CONFIGS_ROOT = "/ocean/projects/phy250048p/shared/configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/valid loss curves from training CSV output.")
    parser.add_argument(
        "--loss-csv",
        default=None,
        help=(
            "Explicit path to a losses CSV. If omitted, uses "
            "<model-root>/losses/losses_<model-name>.csv."
        ),
    )
    parser.add_argument(
        "--stage", choices=("pretrain", "npe"), default="npe",
        help="Configuration stage whose loss history should be plotted.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Model name. When provided, runtime settings are loaded from "
            "a JSON config in <configs-root> (cfg_<model-name>.json by default)."
        ),
    )
    parser.add_argument(
        "--model-root",
        default=None,
        help=(
            "Model root path containing losses/ directory. Ignored when "
            "--model-name is provided."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory to save the figure "
            "(default: <figure-root>/<model-name>)."
        ),
    )
    parser.add_argument(
        "--figure-root",
        default=DEFAULT_FIGURE_ROOT,
        help="Root directory for default output when --output-dir is not provided.",
    )
    parser.add_argument(
        "--configs-root",
        default=DEFAULT_CONFIGS_ROOT,
        help="Directory containing saved training configs named cfg_<model>.json.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename (default: <model-name>_loss.png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def _get_saved_config_candidates(configs_root: str, model_name: str) -> tuple[str, ...]:
    return (join(configs_root, f"cfg_{model_name}.json"),)


def load_saved_train_config(configs_root: str, model_name: str, stage: str, *, required: bool = True) -> dict | None:
    cfg_path = next(
        (path for path in _get_saved_config_candidates(configs_root, model_name) if os.path.exists(path)),
        None,
    )
    if cfg_path is None:
        if required:
            candidates = ", ".join(_get_saved_config_candidates(configs_root, model_name))
            raise FileNotFoundError(f"Saved training config not found. Tried: {candidates}")
        return None

    with open(cfg_path, "r", encoding="utf-8") as fobj:
        payload = json.load(fobj)

    config_key = "pretrain" if stage == "pretrain" else "train"
    train_cfg = payload.get(config_key)
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Saved config missing '{config_key}' dictionary: {cfg_path}")
    return train_cfg


def resolve_runtime_settings(args: argparse.Namespace) -> tuple[str, str]:
    train_config = config.pretrain if args.stage == "pretrain" else config.train
    requested_model_name = args.model_name if args.model_name is not None else train_config["model_name"]
    train_cfg = load_saved_train_config(
        args.configs_root,
        requested_model_name,
        args.stage,
        required=args.model_name is not None,
    )
    if train_cfg is not None:
        model_root = train_cfg.get("model_path")
        if model_root is None:
            raise ValueError("Saved train config is missing train['model_path'].")
        model_name = train_cfg.get("model_name", requested_model_name)
        return model_name, model_root

    model_name = train_config["model_name"]
    model_root = args.model_root if args.model_root is not None else train_config["model_path"]
    return model_name, model_root


def resolve_loss_csv(loss_csv: str | None, model_root: str, model_name: str) -> str:
    if loss_csv is not None:
        return loss_csv
    return join(model_root, "losses", f"losses_{model_name}.csv")


def load_losses(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Loss CSV not found: {csv_path}")

    losses = pd.read_csv(csv_path)
    required = {"train", "valid"}
    missing = required - set(losses.columns)
    if missing or losses.empty:
        raise ValueError(
            "Loss CSV must contain one row per epoch and the columns "
            f"'train' and 'valid'; missing={sorted(missing)}."
        )
    try:
        numeric = losses.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Loss CSV columns must all be numeric") from error
    values = numeric.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Loss CSV must contain only finite values")
    train_losses = numeric["train"].to_numpy(dtype=np.float64)
    valid_losses = numeric["valid"].to_numpy(dtype=np.float64)
    return train_losses, valid_losses


def plot_losses(train_losses: np.ndarray, valid_losses: np.ndarray, model_name: str) -> Figure:
    epochs = np.arange(1, train_losses.shape[0] + 1)

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "figure.dpi": 300})
    fig = plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, "r.-", label="Training Set")
    plt.plot(epochs, valid_losses, "b.-", label="Validation Set")
    plt.text(0.95, 0.75, f"Epoch with lowest validation loss: {np.argmin(valid_losses)}", transform=plt.gca().transAxes, fontsize=12, ha="right", va="top")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.ylim((min(min(train_losses), min(valid_losses)), max(train_losses[0], valid_losses[0]) * 1.05))
    plt.tick_params(axis="both", which="major", labelsize=14)
    # plt.xticks([])
    # plt.yticks([])
    plt.legend(fontsize=14)
    plt.title(model_name, fontsize=14)
    plt.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    model_name, model_root = resolve_runtime_settings(args)
    csv_path = resolve_loss_csv(args.loss_csv, model_root, model_name)

    train_losses, valid_losses = load_losses(csv_path)
    fig = plot_losses(train_losses, valid_losses, model_name)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else join(args.figure_root, model_name)
    )
    os.makedirs(output_dir, exist_ok=True)
    output_name = args.output_name if args.output_name is not None else f"{model_name}_loss.png"
    output_path = join(output_dir, output_name)

    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved loss figure: {output_path}")
    print(f"Detected epochs: {train_losses.shape[0]}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
