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
g1, g2, theta_int, sini, v0, vcirc, rscale, hlr, halpha_flux_true
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
weights are normalized within each galaxy for TF-adjusted posterior summaries;
truth-level population ratios must be normalized globally after all cache
partitions are joined for ensemble statistics.

## Workflow

Run the commands below from the repository root. Default shared-data paths
point at `/ocean/projects/phy250048p/shared`; each launcher documents the
environment-variable overrides it supports for run-specific values.

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
7. Generate diagnostics from the partitioned cache under `arch/diagnostics/`.

For matched finite-shear response calibration, follow
[`repo_reports/SHEAR_RESPONSE_RUNBOOK.md`](repo_reports/SHEAR_RESPONSE_RUNBOOK.md).
The detailed architecture contract is
[`repo_reports/KL-NN_SIMV2_ARCH.md`](repo_reports/KL-NN_SIMV2_ARCH.md).

## Environment and tests

Install KL-tools from its simulator branch, then install this repository's
Python dependencies and `ml-pyxis` in the `kl-nn` Conda environment. Cluster
launchers assume the Bridges-2 module environment and should be reviewed if
run elsewhere.

Run the maintained tests with:

```bash
conda activate kl-nn
pytest -q tests
```

Model configuration and the network source are snapshotted at training time
under the configured shared roots. Treat the saved current-schema JSON and
checkpoint as one artifact; archived-schema fallback is intentionally absent.
