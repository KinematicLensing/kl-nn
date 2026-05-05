# Copilot instructions for `kl-nn`

## Build, test, and lint commands

This repository does not define a dedicated build or lint toolchain (`pyproject.toml`, `setup.cfg`, `tox.ini`, and lint configs are not present).

- **Run all tests:** `pytest -q tests`
- **Run one test file:** `pytest -q tests/test_noise_apply.py`
- **Run one test case:** `pytest -q tests/test_noise_apply.py -k test_apply_noise_preserves_shape_dtype`

Tests require the project ML stack (notably `torch`) installed in the active environment.

## High-level architecture

`kl-nn` is split into two main pipelines:

1. **Synthetic data generation** (`data_generate/`)
   - `latin_hypercube.py` creates parameter CSVs.
   - `generate_fits.py` renders per-galaxy FITS files using `kl_tools`.
   - `make_database.py` converts FITS + fiducial parameters into a Pyxis dataset (`img`, `spec`, `fid_pars`, `id`) used by training.

2. **Model training / inference** (`arch/`)
   - `config.py` is the central runtime config (dataset paths, model name/path, feature set, training mode).
   - `[scr]_train_model.py` launches distributed training with `mp.spawn`.
   - `train.py` handles DDP setup, training loops, noise injection, model loading, and posterior sampling helpers.
   - `networks.py` defines `ForkCNN` (image + spectra branches) and mode-specific heads:
     - mode `0`: point-estimate regression
     - mode `1`: conditional flow density estimation
     - mode `2`: flow density estimation with TF-prior adjustment for `vcirc`

Operationally, Slurm scripts in `arch/*.slurm` and `data_generate/*.slurm` are the standard entrypoints on the cluster.

## Key conventions in this codebase

- **Shared filesystem assumptions are hard-coded**: many scripts default to `/ocean/projects/phy250048p/shared/...`. Keep `config.py` and Slurm script paths aligned when changing datasets/models.
- **Feature ordering is contract-critical**: `config.train["feature_names"]` and `config.train["feature_number"]` must stay consistent across training, sampling, denormalization, and TF-prior logic (`vcirc` index resolution uses feature names).
- **Training data normalization convention**: parameters are normalized to `[-1, 1]` (see `make_database.py`) and later denormalized via `utils.denormalize`.
- **Fixed tensor shape expectations**: training/inference code assumes image shape `(1, 48, 48)` and spectra shape `(1, nspec, 64)`; these are baked into dataset loading and network definitions.
- **Script naming pattern**: primary executable scripts in `arch/` use the literal prefix `[scr]_` (for example `[scr]_train_model.py`), and Slurm wrappers invoke those names directly.
- **Test import path setup**: `tests/conftest.py` prepends repo root and `arch/` to `sys.path`, so tests import modules like `train` directly from `arch/train.py`.
