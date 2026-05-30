# Copilot instructions for `kl-nn`

## Build, test, and lint commands

There is no dedicated build/lint config in this repo (`pyproject.toml`, `setup.cfg`, `tox.ini`, `ruff.toml`, etc. are not present).

- **Before running tests:** `conda activate kl-nn`
- **Run all tests:** `pytest -q tests`
- **Run one test file:** `pytest -q tests/test_model_config_registry.py`
- **Run one test case:** `pytest -q tests/test_noise_apply.py -k test_apply_noise_preserves_shape_dtype`

Tests depend on the ML runtime stack (notably `torch`, `pyxis`, and training dependencies imported by `arch/train.py`).

## Recent updates (2026-05-25)

- `train.load_model` now prefers archived `networks_<model>.py` from `/ocean/projects/phy250048p/shared/networks`, falling back to repo `networks.py` if missing.
- Training optimizations are default-on (AMP, `torch.compile`, fused AdamW with fallback, cuDNN benchmark, channels_last, GPU-side shuffling, noise max caching, DDP efficiency flags).
- Added `tests/test_training_optimizations.py` for config and short training-path coverage (GPU tests auto-skip when CUDA is unavailable).
- `train._maybe_compile_model` now filters the NetworkX `nx-loopback` entry point to avoid `torch.compile` failures in this environment (invalid syntax during NetworkX backend import) and handles `compile_backend=None` explicitly.
- `ForkCNN` pads missing `fib_pos` features when absent so the concatenated feature vector remains 1024-d, preventing training shape mismatches when datasets omit fiber positions.
- `circular_spline.unconstrained_rational_quadratic_spline` was refactored to avoid in-place masked indexing; uses clamped inputs + `torch.where` to eliminate `torch.compile` graph breaks and CopySlices anomaly warnings.
- Test suite currently fails on `tests/test_rmag_calibration.py::test_generate_snr_rmag_uses_fitted_relation` (pre-existing mismatch).

## High-level architecture

`kl-nn` has two primary surfaces:

1. **Data generation** (`data_generate/`)
   - Classic synthetic pipeline: `latin_hypercube.py` → `generate_fits.py` / wrappers → `make_database.py`.
   - TNG pipeline: `generate_tng_sample.py`, `tng_rotation_fit.py`, `backfill_tng_sample_csvs.py`, and `make_tng_database.py`.
   - Database writers produce Pyxis datasets with keys such as `img`, `spec`, `fid_pars`, and `id` (plus `fib_pos` in some flows).

2. **Training, loading, and inference** (`arch/`)
   - `config.py` defines typed dataclasses (`ModelConfig`, `DatasetConfig`, `TrainConfig`, `FlowConfig`) and also mirrors legacy dict globals (`config.train`, `config.data`, etc.) via `set_model_config`.
   - `model_registry.py` snapshots/loads model artifacts (`cfg_<model>.json`, `networks_<model>.py`) under shared roots.
   - `[scr]_train_model.py` snapshots config + network source, then launches DDP training via `mp.spawn(train_nn, ...)`.
   - `train.py` owns model loading, training loops, SNR/noise generation, density estimation, and sampling.
   - `networks.py` defines `ForkCNN`:
     - mode `0`: point estimate
     - mode `1`: conditional flow posterior
     - mode `2`: flow posterior with TF-prior adjustment for `vcirc`

## Key conventions in this codebase

- **Shared filesystem defaults are intentional:** many scripts default to `/ocean/projects/phy250048p/shared/...`; keep script/config paths aligned when changing datasets, cache roots, or model roots.
- **Config loading pattern matters:** for archived models, load config by model name (`model_registry.load_model_config`) and call `config.set_model_config(...)` before using `config.train`/`config.par_ranges`.
- **Feature-name order is contract-critical:** `config.train["feature_names"]` and `feature_number` must match training targets, denormalization, TF-prior logic, and feature index lookups (`resolve_feature_index`).
- **Mode 2 requires `vcirc_mu`:** density APIs in `train.py` and TF adjustments in `networks.py` assert per-galaxy `vcirc_mu`; callers must provide it.
- **Noise injection is centralized:** use `train.apply_noise(...)` (iterative mask refinement by default). `CNNTrainer._apply_noise` is intentionally a thin delegate.
- **R-mag ↔ SNR calibration is persisted:** `train._load_rmag_snr_relation` reads `config.rmag_snr_fit_path` and auto-fits from `rmag_snr_source_path` if missing.
- **Tensor shape expectations are fixed in core paths:** image `(1, 48, 48)`, spectra `(1, nspec, 64)`, with fiber-position support where present.
- **Model compatibility fallback exists:** `train.load_model` first loads current `ForkCNN`; on state mismatch it retries with archived `networks_<model>.py` inferred from checkpoint path.
- **Test imports rely on `tests/conftest.py`:** repo root and `arch/` are prepended to `sys.path`, so tests import modules like `train` and `config` directly.
- **Fiber ordering and geometry conventions (D4/diagnostics):**
  - Fibers are always arranged in a cross centered on the galaxy and aligned with the galaxy’s major axis; array order is **(+major, −major, center, +minor, −minor)**.
  - “Positive” is defined as toward +x (rightward) in the image when `theta_int == 0`.
  - D4 transforms rotate/reflect images; rotations keep spectra order fixed, while reflections swap the ±minor spectra (see `D4_SPEC_PERM`).
  - `fib_pos` is always transformed to match the image transform (see `d4_fib_pos`).
  - Reflections/handedness flips **swap the ±minor fibers** and negate `g2` and `theta_int` to reflect the spin-2 ambiguity.
  - The simulation uses the **opposite sign convention from NumPy** for `theta_int`, so correcting `theta_int` in the opposite direction of the image transform is intentional.
  - Diagonal flips are treated as **flip across y** followed by **±90° rotation**; this is the intended convention in the diagnostics.

## Diagnostic and analysis notebooks

These notebooks are ad-hoc diagnostics (ignore `.ipynb_checkpoints`); most assume the shared `/ocean/projects/phy250048p/shared/...` paths and large cached datasets.

- `arch/point_est_diagnostic.ipynb` — tests trained point-estimate models: loss curves, per-parameter residual plots (`g1/g2`, `theta_int`, `sini`, `v0`, `vcirc`, `rscale`, `hlr`), and includes D4 symmetry checks on outputs.
- `arch/flow_diagnostic.ipynb` — flow model diagnostics: loss plots, cached-sample loading, grid density estimation, stress tests (Sersic index, TNG samples, blob injection), plus saliency, rmag/SNR, and contour diagnostics.
- `arch/compare_model.ipynb` — compares two trained models via loss curves and parameter error plots for the same targets.
- `arch/diagnostics/sample_diagnostics.ipynb` — reads hierarchical cached samples produced by `[scr]_tf_analysis.py` and generates summary diagnostics across partitions.
- `arch/diagnostics/d4_analysis.ipynb` — regenerates datavectors under D4 transforms and measures symmetry violations/average diffs.
- `arch/diagnostics/latent_analysis.ipynb` — extracts latents (with `apply_noise` + `extract_latent`) over a subset and visualizes latent distributions vs truth/SNR.
- `arch/diagnostics/resample_test.ipynb` — toy resampling sanity check with `torch.multinomial` and scatter plots.

Supporting notebooks:
- `arch/train_model.ipynb` — in-notebook training/DDP debug run (inefficient; use for testing only).
- `arch/conditional_flow.ipynb` — standalone conditional normalizing flow demo on synthetic Gaussian data.
