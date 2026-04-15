#!/usr/bin/env python3
"""Plot training and validation losses for a trained model."""

from __future__ import annotations

import argparse
import importlib.util
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
        "--model-name",
        default=None,
        help=(
            "Model name. When provided, runtime settings are loaded from "
            "<configs-root>/cfg_train_model_<model-name>.py."
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
        help="Directory containing saved training configs named cfg_train_model_<model>.py.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output filename (default: <model-name>_loss.png).",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def load_saved_train_config(configs_root: str, model_name: str) -> dict:
    cfg_path = join(configs_root, f"cfg_train_model_{model_name}.py")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Saved training config not found: {cfg_path}")

    module_name = f"cfg_train_model_{model_name}"
    spec = importlib.util.spec_from_file_location(module_name, cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config module from: {cfg_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    train_cfg = getattr(module, "train", None)
    if not isinstance(train_cfg, dict):
        raise ValueError(f"Saved config missing 'train' dictionary: {cfg_path}")
    return train_cfg


def resolve_runtime_settings(args: argparse.Namespace) -> tuple[str, str]:
    if args.model_name is not None:
        train_cfg = load_saved_train_config(args.configs_root, args.model_name)
        model_root = train_cfg.get("model_path")
        if model_root is None:
            raise ValueError("Saved train config is missing train['model_path'].")
        model_name = train_cfg.get("model_name", args.model_name)
        return model_name, model_root

    model_name = config.train["model_name"]
    model_root = args.model_root if args.model_root is not None else config.train["model_path"]
    return model_name, model_root


def resolve_loss_csv(loss_csv: str | None, model_root: str, model_name: str) -> str:
    if loss_csv is not None:
        return loss_csv
    return join(model_root, "losses", f"losses_{model_name}.csv")


def load_losses(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Loss CSV not found: {csv_path}")

    losses = pd.read_csv(csv_path)
    if losses.shape[0] != 2:
        raise ValueError(
            "Expected exactly 2 rows in losses CSV "
            f"(train row + valid row), got shape={losses.shape}."
        )
    if losses.shape[1] < 1:
        raise ValueError("Loss CSV has no epoch columns.")

    train_losses = losses.iloc[0, :].to_numpy(dtype=np.float64)
    valid_losses = losses.iloc[1, :].to_numpy(dtype=np.float64)
    return train_losses, valid_losses


def plot_losses(train_losses: np.ndarray, valid_losses: np.ndarray, model_name: str) -> Figure:
    epochs = np.arange(1, train_losses.shape[0] + 1)

    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "figure.dpi": 300})
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, "r.-", label="Training Set")
    plt.plot(epochs, valid_losses, "b.-", label="Validation Set")
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=14)
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
