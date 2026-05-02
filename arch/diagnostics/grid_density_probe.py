#!/usr/bin/env python3
"""Load a flow model, estimate density on a grid, and make example corner plots.

This script is intentionally conservative because the 2D->3D feature transition
introduced known breakpoints in older checkpoints/workflows.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from os.path import join

# Add parent directory (arch/) to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pyxis.torch as pxt  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from torch.utils.data import DataLoader, Subset  # type: ignore[import-not-found]

import config
from train import estimate_density, load_model
from utils import denormalize, make_corner_plot


FEATURE_INDEX_BY_NAME = {
    "g1": 0,
    "g2": 1,
    "theta_int": 2,
    "sini": 3,
    "v0": 4,
    "vcirc": 5,
    "rscale": 6,
    "hlr": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe flow density estimation on a configurable grid.")
    parser.add_argument("--model-name", default=config.train["model_name"], help="Model directory name.")
    parser.add_argument(
        "--model-root", default=config.train["model_path"], help="Root path containing model directories."
    )
    parser.add_argument(
        "--checkpoint-suffix",
        default="latest",
        help="Checkpoint suffix after model name, or 'latest'.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=config.test["data_dir"],
        help="Directory containing test dataset files for pyxis.torch.TorchDataset.",
    )
    parser.add_argument("--mode", type=int, default=2, choices=[1, 2], help="Inference mode.")
    parser.add_argument("--grid-size", type=int, default=40, help="Grid resolution per feature axis.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for estimate_density.")
    parser.add_argument("--n-galaxies", type=int, default=10, help="Number of galaxies to evaluate.")
    parser.add_argument(
        "--posterior-draws",
        type=int,
        default=2000,
        help="Number of weighted posterior draws for corner examples.",
    )
    parser.add_argument(
        "--n-corner",
        type=int,
        default=3,
        help="Number of galaxies for which corner plots are generated.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for generated figures (default: /ocean/.../figures/<model>/density_probe).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise exceptions instead of exiting gracefully on known migration blockers.",
    )
    return parser.parse_args()


def resolve_checkpoint(model_root: str, model_name: str, checkpoint_suffix: str) -> str:
    model_dir = join(model_root, model_name)
    if checkpoint_suffix != "latest":
        ckpt_name = checkpoint_suffix if checkpoint_suffix.startswith(model_name) else f"{model_name}{checkpoint_suffix}"
        path = join(model_dir, ckpt_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pat = re.compile(rf"^{re.escape(model_name)}(\d+)$")
    candidates = []
    for fname in os.listdir(model_dir):
        match = pat.match(fname)
        if match is None:
            continue
        candidates.append((int(match.group(1)), join(model_dir, fname)))

    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching '{model_name}<number>' found in {model_dir}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def build_grid(grid_size: int, nfeatures: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    axes = [torch.linspace(-1, 1, grid_size, device=device) for _ in range(nfeatures)]
    mesh = torch.meshgrid(*axes, indexing="ij")
    stacked = torch.stack(mesh, dim=-1)
    zz = stacked.reshape(-1, nfeatures)

    flat_axes = [m.detach().cpu().numpy().reshape(-1) for m in mesh]
    grid_np = np.vstack(flat_axes).T
    return zz, grid_np


def get_vcirc_mu(subset: Subset, device: torch.device) -> torch.Tensor:
    vcirc_true = torch.zeros((len(subset),), dtype=torch.float32, device=device)
    for i in range(len(subset)):
        vcirc_true[i] = subset[i]["fid_pars"][5]
    # fid_pars stores normalized vcirc in [-1, 1]
    return 0.5 * (vcirc_true + 1.0) * 480.0 + 60.0


def weighted_draws_from_grid(
    log_prob_1d: np.ndarray, grid_phys: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    log_prob = log_prob_1d - np.max(log_prob_1d)
    prob = np.exp(log_prob)
    prob_sum = np.sum(prob)
    if not np.isfinite(prob_sum) or prob_sum <= 0:
        raise ValueError("Invalid log_prob map encountered while drawing weighted posterior samples.")
    prob /= prob_sum
    draw_idx = rng.choice(grid_phys.shape[0], size=n_draws, replace=True, p=prob)
    return grid_phys[draw_idx]


def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = resolve_checkpoint(args.model_root, args.model_name, args.checkpoint_suffix)

    print(f"Loading checkpoint: {checkpoint}")
    model = load_model(mode=args.mode, path=checkpoint, strict=True, assign=True, device=device)

    nfeatures = int(config.train["feature_number"])
    if nfeatures not in (2, 3):
        raise ValueError(f"Unsupported feature_number={nfeatures}; expected 2 or 3.")

    feature_names = ["g1", "g2"] if nfeatures == 2 else ["g1", "g2", "vcirc"]

    zz, grid_np = build_grid(args.grid_size, nfeatures, device)
    grid_phys = denormalize(grid_np.copy(), config.par_ranges, feature_names=feature_names)

    test_ds = pxt.TorchDataset(args.dataset_dir)
    n_eval = min(args.n_galaxies, len(test_ds))
    subset = Subset(test_ds, np.arange(0, n_eval))
    test_dl = DataLoader(subset, batch_size=args.batch_size, pin_memory=False, shuffle=False)

    vcirc_mu = get_vcirc_mu(subset, device) if args.mode == 2 else None

    print(
        f"Running estimate_density for mode={args.mode}, nfeatures={nfeatures}, "
        f"grid_size={args.grid_size}, galaxies={n_eval}"
    )

    log_probs, _, snr = estimate_density(
        zz,
        test_dl,
        model,
        batch_size=args.batch_size,
        vcirc_mu=vcirc_mu,
        device=device,
    )
    print(f"Completed density estimation. log_probs shape={log_probs.shape}, snr shape={snr.shape}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = join("/ocean/projects/phy250048p/shared/figures", args.model_name, "density_probe")
    os.makedirs(output_dir, exist_ok=True)

    labels = ["g_1", "g_2", "v_{circ}"][:nfeatures]
    n_plot = min(args.n_corner, n_eval)

    for i in range(n_plot):
        posterior_draws = weighted_draws_from_grid(log_probs[i], grid_phys, args.posterior_draws, rng)

        fid_pars = subset[i]["fid_pars"]
        if torch.is_tensor(fid_pars):
            fid_pars = fid_pars.detach().cpu().numpy()
        true_norm = np.array([[fid_pars[FEATURE_INDEX_BY_NAME[name]] for name in feature_names]], dtype=np.float32)
        true_phys = denormalize(true_norm.copy(), config.par_ranges, feature_names=feature_names)[0]

        fig, _, _ = make_corner_plot(
            posterior_draws,
            truth=true_phys,
            labels=labels,
            sample_label=f"mode {args.mode}",
        )
        fig.savefig(join(output_dir, f"density_corner_mode{args.mode}_idx{i}.png"), dpi=300)
        plt.close(fig)

    print(f"Saved {n_plot} corner plots to: {output_dir}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as exc:
        msg = (
            "grid_density_probe hit a known blocker in the current 2D->3D migration path. "
            "This script is parked for follow-up debugging. "
            f"Error: {exc}"
        )
        print(msg)
        if args.strict:
            raise


if __name__ == "__main__":
    main()
