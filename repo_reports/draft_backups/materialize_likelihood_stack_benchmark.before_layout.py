#!/usr/bin/env python3
"""Materialize the 16 deterministic identity-noise views for benchmark 04."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pyxis.torch as pxt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCH_DIR = REPO_ROOT / "arch"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ARCH_DIR) not in sys.path:
    sys.path.insert(0, str(ARCH_DIR))

import config
from data import apply_central_halpha_snr_noise, apply_image_noise_for_snr
from diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_DATASET,
    BENCHMARK_N_GALAXIES,
    BENCHMARK_N_REALIZATIONS,
    benchmark_noise_seeds,
)
from train import _seeded_generator, build_observation_levels, validate_observation_record


DEFAULT_DATA_DIR = Path(
    "/ocean/projects/phy250048p/shared/datasets"
) / BENCHMARK_DATASET
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / "benchmark_noisy.npz"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "benchmark_noisy.json"


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=420042)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.output.exists() or args.manifest.exists()) and not args.overwrite:
        raise FileExistsError("benchmark noisy outputs exist; use --overwrite")
    dataset = pxt.TorchDataset(str(args.data_dir))
    n_rows = len(dataset)
    expected_rows = 25 * BENCHMARK_N_GALAXIES
    if n_rows != expected_rows:
        raise ValueError(f"expected {expected_rows} clean rows, got {n_rows}")
    images = np.empty(
        (n_rows, BENCHMARK_N_REALIZATIONS, 1, 48, 48), dtype=np.float32
    )
    spectra = np.empty(
        (n_rows, BENCHMARK_N_REALIZATIONS, 1, 5, 64), dtype=np.float32
    )
    for index in range(n_rows):
        record = dataset[index]
        rmag, _, image_snr, spec_snr = validate_observation_record(
            record, location=f"benchmark materialization {index}"
        )
        image = torch.as_tensor(record["img"]).float()
        spec = torch.as_tensor(record["spec"]).float()
        image_level, spec_level = build_observation_levels(
            torch.tensor([image_snr]), torch.tensor([spec_snr])
        )
        cell, galaxy = divmod(index, BENCHMARK_N_GALAXIES)
        for realization in range(BENCHMARK_N_REALIZATIONS):
            image_seed, spec_seed = benchmark_noise_seeds(
                args.seed, cell, galaxy, realization
            )
            noisy_image = apply_image_noise_for_snr(
                image.unsqueeze(0),
                image_level,
                randgen=_seeded_generator(torch.device("cpu"), image_seed),
            )
            noisy_spec = apply_central_halpha_snr_noise(
                spec.unsqueeze(0),
                spec_level,
                center_fiber_index=int(config.observation["center_fiber_index"]),
                center_exposure_s=config.observation["center_exposure_s"],
                offset_exposure_s=config.observation["offset_exposure_s"],
                spectral_units=config.observation["spectral_units"],
                randgen=_seeded_generator(torch.device("cpu"), spec_seed),
                device=torch.device("cpu"),
            )
            images[index, realization] = noisy_image[0].numpy()
            spectra[index, realization] = noisy_spec[0].numpy()
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == n_rows:
            print(f"materialized {index + 1}/{n_rows}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, images=images, spectra=spectra)
    payload = {
        "schema": "likelihood-stack-fixed-shear-noise-v1",
        "data_dir": str(args.data_dir),
        "output": str(args.output),
        "seed": int(args.seed),
        "n_rows": n_rows,
        "n_cells": 25,
        "n_galaxies_per_cell": BENCHMARK_N_GALAXIES,
        "n_realizations": BENCHMARK_N_REALIZATIONS,
        "stream_key": "(seed, shear_cell, latent_galaxy, realization)",
        "identity_noise_only": True,
    }
    args.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
