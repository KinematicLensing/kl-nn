# KL-NN

KL-NN is a proof-of-concept neural posterior estimator for kinematic
lensing. The supported pipeline uses simulator-v2 galaxy-axis observations,
continuous-contrastive feature pretraining, a bounded hybrid posterior, a
two-view identity/R90 ensemble, and optional post-training Tully--Fisher (TF)
importance weights.

This repository intentionally supports one current configuration. Historical
architectures, point-estimator training, training-time TF variants, and
compatibility loaders are not part of the maintained interface.

This is a breaking schema. Eight-target LMDBs, previous model checkpoints,
and earlier posterior caches cannot be loaded by the current pipeline. Rebuild
the LMDB from strict current-schema FITS (or regenerate FITS when those headers
or H-alpha truth are absent), then pretrain and train new checkpoints.

## Current statistical contract

The broad simulation proposal does not impose a TF relation. The nine
posterior targets are:

```text
g1, g2, theta_int, cosi, v0, vcirc, rscale, hlr, halpha_flux_true
```

The only scalar observation context is perfect knowledge of:

```text
rmag_true, spectral_reference_quality
```

H-alpha flux is a posterior target, not context. The simulator produces one
r-band image and five H-alpha fiber spectra. Image and spectral noise remain
Gaussian; their levels are controlled independently and cover the configured
ground-based survey range.

The trained base posterior uses the independent uniform proposal. TF
information enters only after training through importance ratios. Candidate
weights are normalized within each galaxy for TF-adjusted posterior summaries.
For rectangular proposal datasets, truth-level population ratios are normalized
globally after all cache partitions are joined. Already-TF-conformed catalog
test sets instead retain uniform truth-population mass before posterior-
precision weighting, avoiding a second TF population correction.

## Workflow

Run the commands below from the repository root. All machine-specific paths
(shared data, models, caches, logs, the kl-tools checkout) live in
[`paths.json`](paths.json); see [Paths and job submission](#paths-and-job-submission).
Each launcher documents the environment-variable overrides it supports for
run-specific values. Submit launchers with `./submit.sh` (same arguments as
`sbatch`), e.g. `./submit.sh arch/train_npe.slurm`.

1. Generate the canonical 100,000-row training sample table with
   `python data_generate/latin_hypercube.py --seed 42`.
   Generate validation and final-test tables independently with different
   seeds, `--nsamples`, and `--output` values. The final-test table must not be
   used for checkpoint selection.
2. Render FITS with `data_generate/generate_simulator_v2.slurm`.
3. Build and merge the database with
   `data_generate/make_database_simulator_v2.slurm` and
   `data_generate/merge_database_simulator_v2.slurm`.
4. Pretrain the feature extractor with `arch/pretrain_ccl.slurm`.
5. Pass the reported `PRETRAINED_NAME` and `PRETRAIN_FROM` to
   `arch/train_npe.slurm`.
6. Set `DATASET` to the independent final-test database, then cache base
   candidates and TF weights with `arch/cache_posteriors.slurm`.
7. Analyse the partitioned cache with the diagnostics in the companion
   repository [kl-nn-analysis](https://github.com/KinematicLensing/kl-nn-analysis),
   which also holds the paper tests, paper figures and reports.

Runbooks and design reports live in kl-nn-analysis under `reports/`:
`SHEAR_RESPONSE_RUNBOOK.md` (matched finite-shear response calibration),
`DESI_TF_TEST_SET_RUNBOOK.md` (catalog-backed, TF-conformed Xu-cut evaluation
sets) and `KL-NN_SIMV2_ARCH.md` / `KL-NN_SIMV3_ARCH.md` (architecture contract).
Their paths still refer to the pre-split layout, where these files lived in
`repo_reports/` and the diagnostics in `arch/diagnostics/`. The full history of
the analysis code is on branch `nre` of this repository.

## Paths and job submission

- `paths.json` maps keys (`datasets`, `samples`, `fits`, `models`, `configs`,
  `networks`, `cache`, `reports`, `figures`, `logs`, `shared_tmp`,
  `desi_cut_catalogs`, `kl_tools_root`, `analysis_root`, `paper_repo`,
  `conda_env`) to paths. Values may reference other keys as `{key}`;
  `{repo_root}` is this checkout. To move to another machine, edit
  `shared_root` (and anything else that moved), or put per-machine overrides in
  a gitignored `paths.local.json`. A single key can also be overridden with an
  environment variable `KLNN_PATH_<KEY>`, e.g. `KLNN_PATH_MODELS=/scratch/models`.
- Python reads them with `from klnn_paths import path; path("models")`.
  `python klnn_paths.py --all` prints every resolved value.
- Shell and slurm launchers `source klnn_paths.sh`, which exports one
  `KLNN_<KEY>` variable per key (`KLNN_MODELS`, `KLNN_DATASETS`, ...) and
  defines `klnn_path KEY`, `klnn_activate` and `klnn_sbatch`.
- Launchers have no `#SBATCH --output` line. `./submit.sh [sbatch options]
  launcher.slurm [launcher arguments]` writes the log to
  `<logs>/<launcher name>_%j.out` (`_%A_%a.out` for array jobs) and passes
  `KLNN_REPO_ROOT` to the job. Plain `sbatch` also works when run from the
  repository root, but the log then goes to `slurm-<id>.out` in that directory.

## Environment and tests

Create the Conda environment with `conda env create -f environment.yml`. It
installs KL-tools from the `data_generate` branch of
[wxs0703/kl-tools](https://github.com/wxs0703/kl-tools); for development,
clone kl-tools next to this repository and `pip install -e ../kl-tools`
instead. `environment.full.yml` is the exact export of the Bridges-2 env.
Cluster launchers assume the Bridges-2 module environment (`module load
anaconda3`, `cuda`) and should be reviewed if run elsewhere.

Run the maintained tests with:

```bash
conda activate kl-nn
pytest -q tests
```

Model configuration and the network source are snapshotted at training time
under the configured shared roots. Treat the saved current-schema JSON and
checkpoint as one artifact; archived-schema fallback is intentionally absent.
