# `kl-nn` repository report

Reviewed 2026-08-07 on branch `vicreg-pretrain`, at commit
`6634d4cacb64487337ec987e032f68d907b06725` (`some model tests`, 2026-07-21).

This is a private working map of the repository as it exists now. It is based on
the checked-in Python, shell/Slurm scripts, tests, README, repository guidance,
and the source/markdown cells of the notebooks. Notebook outputs and
`.ipynb_checkpoints` were not treated as authoritative. No existing file was
changed as part of this review.

## Executive summary

The scientific goal is simulation-based inference of galaxy parameters—most
importantly weak-lensing shear `(g1, g2)`—from a 48×48 photometric image, five
fiber spectra, and the five fiber positions. The current default model predicts
an eight-dimensional posterior over:

`g1, g2, theta_int, sini, v0, vcirc, rscale, hlr`.

The active architecture is `KLNPE`, not `ForkCNN`. It combines a deep residual
image CNN with a spectral CNN and fiber coordinates, yielding a 1024-element
context. Mode 0 produces an MSE point estimate; mode 1 uses a 12-layer
conditional masked autoregressive flow; mode 2 additionally incorporates a
Tully–Fisher (TF) prior on `vcirc` during training and importance-resamples flow
draws during inference.

There are two data-generation families:

1. A parametric synthetic-galaxy pipeline built on `kl_tools`, which currently
   writes all fields needed by training.
2. A TNG50 rendering pipeline built on `kl_pipe`/JAX, which writes image and
   spectra FITS files and then a Pyxis database. Its database writer currently
   omits `fib_pos`, so its output is not compatible with the active
   `FeatureExtractor` or the GPU training loader without further work.

The repository is strongly coupled to PSC Bridges/Ocean paths, Slurm, CUDA,
Pyxis, and adjacent `kl-tools`/`kl_pipe` installations. It is not packaged and
has no committed environment or dependency lockfile. The README describes the
broad intent but is behind the implementation in several important places.

## End-to-end mental model

```text
parameter samples (CSV)
        |
        +-- parametric: latin_hypercube.py -> generate_fits.py
        |                                      |
        |                                      v
        |                               per-object FITS
        |                                      |
        |                               make_database.py
        |                                      |
        +-- TNG: per-galaxy LHS -> generate_tng_sample.py
                                               |
                                        per-object FITS
                                               |
                                      make_tng_database.py
                                               |
                                               v
                                 Pyxis/LMDB-style dataset
                            {img, spec, fid_pars, id, fib_pos?}
                                               |
                             noise + symmetry augmentation
                                               |
                       ImgCNN ----+---- SpecCNN + fiber positions
                                  v
                          1024-d context vector
                                  |
                +-----------------+------------------+
                |                                    |
             mode 0                              mode 1/2
          MLP point estimate               conditional normalizing flow
                                                     |
                                      posterior samples / log density
                                                     |
                                   cached diagnostics and D4/TF analysis
```

## Repository layout and responsibilities

### Root

- `README.md`: setup and broad workflow; partly stale.
- `.github/copilot-instructions.md`: the most detailed existing architecture
  note, but it contains some claims from an older implementation state.
- `tests/`: focused unit/smoke tests plus two manual plot-generation scripts.
- There is no `pyproject.toml`, `requirements.txt`, `environment.yml`, Makefile,
  lint configuration, or CI workflow.

### `data_generate/`

- `latin_hypercube.py`: creates the main 12-parameter synthetic samples. The
  physical ranges align with `arch/config.py`.
- `generate_fits_wrapper.py`: reads a CSV slice and invokes
  `generate_fits.py` once per sample.
- `generate_fits.py`: constructs a noiseless `kl_tools` datavector. It uses a
  DECam r-band 48×48 image and five DESI fiber observations of one selected
  emission-line window. Fibers are positioned along the apparent major/minor
  axes derived from shear, inclination, and position angle.
- `make_database.py`: reads FITS files, normalizes physical labels to `[-1,1]`,
  and writes Pyxis records. It supports parallel shards and a merge mode.
- `latin_hypercube_tng.py`: creates shear/orientation realizations per TNG
  galaxy and includes fitted `v0`, `vcirc`, and `rscale`.
- `tng_rotation_fit.py`: fits an arctangent rotation curve to a rendered TNG
  velocity map using weighted radial bins.
- `generate_tng_sample.py`: renders a dusted TNG50 image and H-alpha spectral
  cube at redshift 0.3, observes it with five fibers, and writes a compact FITS
  file containing `IMAGE`, `FLUX`, and a `FIBERS` table.
- `make_tng_database.py`: builds a seven-label TNG Pyxis dataset. It converts
  inclination into an angle and normalizes it to `[0, pi]`; this differs from
  the main model's fourth feature name/range (`sini`, `[0,1]`). It also omits
  fiber positions and `hlr`.
- `backfill_tng_sample_csvs.py` and `patch_tng_fid_pars.py`: migration/repair
  utilities for fitted rotation labels and existing databases.
- `generate_noise.py`: a variation of the legacy simulator for generating a
  noise realization/model; it is not used by the active training noise path.
- `sample_variations.py`, `gen_diagnostic_samples.py`, and
  `transform_samples.py`: special-purpose sample-set generation/transforms.
- Slurm and shell files orchestrate large arrays, shard merging, completeness
  checking, and TNG generation. Most defaults are experiment-specific.
- `notebooks/`: exploratory checks for database generation, distributions,
  fiber placement, TNG datavectors, stacking, and training-data quality.

### `arch/`

- `config.py`: typed nested dataclasses plus synchronized legacy dictionaries.
  The default dataset is 1M train / 100k validation, five fibers, and eight
  targets. Configuration snapshots are JSON-serializable.
- `model_registry.py`: saves `cfg_<model>.json` and a copy of
  `networks_<model>.py` under shared storage and dynamically imports archived
  network code for checkpoint compatibility.
- `networks.py`: active model, feature extractor, self-supervised pretraining
  methods, flow, and alternative/experimental backbones.
- `data.py`: TF conversions, r-magnitude/SNR calibration, rotation and
  handedness augmentations, masking, and image/spectrum noise injection.
- `train.py`: GPU-resident training data, DDP orchestration, trainers, optimizer
  setup, checkpoint loading, and inference helpers.
- `[scr]_train_model.py`: CLI entrypoint that snapshots model artifacts and
  launches DDP with a chosen model/trainer class.
- `[scr]_sample_trained_model.py` and `[scr]_tf_analysis.py`: checkpoint
  sampling/caching and TF-oriented partitioned analysis. Their interfaces are
  not fully synchronized with current `train.py`/`KLNPE` (see risks below).
- `d4_diffs.py`: D4 transformation construction and cached symmetry-difference
  analysis.
- `circular_spline.py`: compile-friendly rational-quadratic circular spline
  transforms. It is tested, but the default flow currently uses affine
  autoregressive transforms instead.
- `utils.py`: denormalization, shear coordinate transforms, saliency, density
  contours, corner plots, clipping, and plotting helpers.
- `plots.py`: reusable parameter-contour plotting and cached-array loading.
- `dataset.py`: an older `FiberDataset` wrapper. Active training uses
  `pyxis.torch.TorchDataset` directly.
- `diagnostics/`: exploratory flow, latent, feature, D4, conditional, and cached
  sample analyses. `sample_diagnostics.ipynb` is the cleanest current cached
  posterior analysis notebook.

## Data contracts

### Synthetic database record

| Key | Expected shape | Meaning |
|---|---:|---|
| `img` | `(1, 48, 48)` | noiseless r-band galaxy image |
| `spec` | `(1, 5, 64)` | five fiber spectra |
| `fib_pos` | `(5, 2)` | fiber `(dx,dy)` in arcsec |
| `fid_pars` | `(12,)` | normalized physical parameters |
| `id` | scalar | unsigned sample ID |

Training slices the first `feature_number` columns from `fid_pars`; the default
therefore trains on the first eight parameters and ignores the four offsets.
Column order is an implicit, critical contract shared by CSV generation,
database construction, configuration, augmentation, denormalization, and TF
logic.

### Normalization

Labels are linearly mapped to `[-1,1]`. Default physical ranges are:

| Feature | Range |
|---|---:|
| `g1`, `g2` | `[-0.1, 0.1]` |
| `theta_int` | `[-pi, pi]` |
| `sini` | `[0, 1]` |
| `v0` | `[-30, 30]` km/s |
| `vcirc` | `[60, 540]` km/s |
| `rscale` | `[0.1, 2.0]` arcsec |
| `hlr` | `[0.1, 3.0]` arcsec |
| four centroid offsets | `[-0.5, 0.5]` arcsec |

The TNG writer instead stores seven values with inclination angle in column 4
and uses `rscale` up to 10. This dataset must have its own archived
configuration/schema or be transformed before use with the default model.

### Fiber and symmetry conventions

The intended fiber order is:

`(+major, -major, center, +minor, -minor)`.

For the identity orientation, positive major points toward image +x. A 90-degree
augmentation rotates the image and fiber coordinates, negates both shear
components, and wraps the normalized position angle. Reflections negate `g2`
and `theta_int` and swap the two minor-axis spectra. The D4 scripts encode eight
image transforms and explicit fiber-coordinate/spectrum behavior.

There are multiple implementations of rotation/permutation logic
(`make_database.py`, `data.py`, `d4_diffs.py`, and notebooks), and they are not
obviously identical. This is a high-value area for a single canonical API and
property tests.

## Model architecture

### Feature extraction

- Images are L2-normalized spatially, then processed by `ImgCNN`: initial
  convolutions followed by deep residual stages from 64 to 512 channels and
  average pooling. The result flattens to 512 features for a 48×48 input.
- Spectra are L2-normalized across fiber/wavelength dimensions and processed as
  a 2D tensor by `SpecCNN`. Four wavelength downsamplings reduce 64 bins to 4;
  the final convolution spans all five fibers and four remaining bins, yielding
  `512 - 2*nspec = 502` values.
- The ten normalized fiber coordinates are concatenated to the spectral output,
  restoring 512 spectral/geometry features.
- Image and spectral branches concatenate to a 1024-element context.

`FeatureExtractor.forward` requires a non-null fiber-position tensor. Despite
some callers checking whether `fib_pos` exists, passing `None` will fail at
`fp / 1.5`.

### Prediction heads

- Mode 0: `1024 -> 512 -> 256 -> nfeatures`, trained with MSE.
- Modes 1/2: layer-normalized context conditions a diagonal Gaussian base and
  12 repetitions of reverse permutation plus masked affine autoregressive
  transformation. Training minimizes negative conditional log likelihood.
- Mode 2 training multiplies per-example log likelihood by normalized TF prior
  density weights. Inference draws candidates from the flow, estimates their
  marginal `vcirc` density with a batch KDE, and importance-resamples toward a
  magnitude/SNR-dependent lognormal TF prior.

The mode-2 procedure is a custom reweighting scheme rather than a flow trained
directly against a formally derived posterior objective. Its statistical target
and calibration should be documented and validated explicitly.

### Pretraining

- `VICRegPretrain` creates two augmented views, projects 1024 features to 128,
  and combines invariance, variance, and covariance losses.
- `CCLPretrain` uses continuous labels to create Gaussian pairwise target
  weights and contrasts projected embeddings.
- Default configuration names CCL pretraining and freezes the pretrained
  feature extractor for downstream `KLNPE` training.
- Alternative ViT and spectral RNN classes exist but are not active defaults.

## Training behavior

`train_nn` initializes NCCL DDP, creates Pyxis datasets, builds or restores a
model, optionally uses channels-last layout and `torch.compile`, wraps in DDP,
and trains on one node with one process per GPU.

The trainer copies each process's dataset partition into preallocated GPU
tensors. Every epoch it performs GPU-side shuffling, generates fresh SNRs,
injects Gaussian noise into image and spectra, and validates. The active NPE
path doubles batches with a 90-degree image/label/fiber augmentation while
leaving spectra fixed. It uses AMP optionally, gradient clipping, AdamW,
ReduceLROnPlateau, per-epoch checkpoints, and a final CSV loss history.

Noise amplitude is estimated from a thresholded source segmentation and the
requested total SNR. A coarse pass estimates background RMS, then a refined
segmentation determines the final scale. Per-sample maxima can be cached.

Operationally, the entire training set is expected to fit in aggregate GPU
memory, and shapes are hard-coded to one 48×48 image, five 64-bin spectra, and
five 2D positions even though some of these values also appear in config.

## Inference and diagnostics

- `point_estimate`: adds noise and returns predictions, truths, and SNRs.
- `sample_density`: adds noise, derives apparent magnitude from SNR for mode 2,
  samples per object, and can return log probabilities or paired 90-degree
  samples for additive-noise cancellation.
- `evaluate_conditional_2d`: evaluates a two-parameter grid while fixing all
  other parameters to their truth values; useful diagnostically but not an
  unconditional shear posterior.
- `[scr]_tf_analysis.py`: partitions a dataset, creates TF-conformed or other
  magnitude/SNR regimes, samples, and writes hierarchical cache arrays plus
  metadata.
- `sample_diagnostics.ipynb`: examines shear residuals versus SNR, orientation,
  inclination, and true shear; makes corner plots and polynomial bias fits.
- `d4_diffs.py` and `d4_analysis.ipynb`: measure equivariance violations across
  the D4 group and, in the notebook, compare transformed arrays to regenerated
  simulations.
- `feature_extraction_analysis.ipynb`: linear/nonlinear probing, R², and mutual
  information of learned features.
- `latent_analysis.ipynb`: flow-latent distribution checks versus truth/SNR.

Several notebooks still call old APIs (`load_model(mode=...)`, `ForkCNN`, or
`model.sample(..., vcirc_mu=...)`) and should be considered historical until
updated.

## Tests and current coverage

The suite covers:

- circular spline invertibility and boundary continuity;
- handedness transformation and exact-half masking;
- noise shape/dtype, determinism, finite behavior, and approximate SNR scaling;
- model-config JSON and artifact snapshots;
- trainer delegation to shared noise logic;
- optimization flags, cached maxima, fused optimizer selection, AMP/compile,
  and a short training-path smoke test;
- r-magnitude/SNR calibration.

The repository guidance says the r-magnitude calibration test was already
failing. I did not run the suite because this review was requested to avoid
changing the repository, and Python/pytest execution would update bytecode or
cache artifacts unless isolated. More importantly, current tests import and
instantiate `ForkCNN`, whose checked-in implementation is only `pass`, so the
current working tree and tests appear out of sync.

Coverage gaps include end-to-end FITS-to-database-to-model tests, real
checkpoint loading, mode-2 posterior calibration, DDP/multi-GPU correctness,
TNG schema compatibility, D4 group properties, and inference CLI smoke tests.

## High-priority consistency and correctness risks

These are observations, not changes.

1. **Active class/test mismatch.** `ForkCNN` is an empty stub, while tests and
   some notebooks still expect it to be the configurable main network.
   `KLNPE` is the actual implementation and the training Slurm file selects it.

2. **Checkpoint loading can reference an unbound variable.** In
   `train.load_model`, when `path` is supplied but no archived networks module
   is found, `model_cls` is never assigned before `model_cls()` is called. This
   contradicts the documented fallback to the live network class.

3. **TNG records are incompatible with the active network.** The TNG database
   omits `fib_pos`, while `FeatureExtractor` requires it. Its label schema also
   uses inclination angle rather than `sini`, has seven rather than eight
   fields, and normalizes `rscale` with a different range.

4. **Sampling scripts use stale APIs.** `[scr]_sample_trained_model.py` calls
   `load_model(mode=...)`, although `load_model` accepts a config and `Model`,
   not `mode`. It passes `vcirc_mu` to `sample_density`, but that parameter is
   currently unused. Several notebooks repeat the old interface.

5. **Optimizer fallback is broken.** If fused AdamW raises `TypeError`, the
   fallback uses `optimizer_diff`, whose definition is commented out.

6. **Package import is broken.** `arch/__init__.py` imports `.simulation`, which
   does not exist. The project mostly avoids this by adding `arch/` to
   `sys.path` and importing modules as top-level names.

7. **Calibration scripts reference a missing module.** Both calibration entry
   scripts import `train_cali`, but no such file exists in the repository.

8. **Configuration is captured in default arguments.** Several network
   constructors use `config.*` values as function defaults, which are evaluated
   at module import. Calling `config.set_model_config` later does not update
   those defaults. This is especially risky when loading archived configs.

9. **Feature order remains partly positional.** Although
   `resolve_feature_index` is used in newer TF/flip code, training slices labels
   by position, rotation assumes shear/angle are columns 0/1/2, and TF SNR
   generation assumes `vcirc` is column 5.

10. **Mode-2 documentation and interface disagree.** Docstrings mention a
    required `vcirc_mu`, but current sampling derives the prior from `mag` and
    `snr`; the public helper still accepts an unused `vcirc_mu` argument.

11. **Destructive shard merge.** `make_database.py --merge` deletes every shard
    after copying it. It logs but skips missing shards, so an incomplete master
    database can be finalized and the available shards then deleted.

12. **Scientific transforms are duplicated.** Rotation, reflection, angle-sign,
    and fiber-permutation conventions are distributed across scripts and
    notebooks. Comments already note that the simulator's angle sign differs
    from NumPy, making silent drift particularly hazardous.

13. **Hard-coded infrastructure and personal paths.** Shared roots, account
    paths, email addresses, model names, dataset sizes, GPU type/count, and
    environment names are embedded in code and jobs. Reproducibility outside
    the current cluster is low.

14. **No dependency/environment record.** The code depends on PyTorch,
    torchvision, nflows, normflows, timm, Pyxis, Astropy, SciPy, pandas,
    matplotlib, JAX, `kl_tools`, and `kl_pipe`, but compatible versions are not
    recorded here.

15. **Dirty working tree.** At review time many source, notebook, job, and
    bytecode files were already modified. Those changes were preserved and may
    explain some mismatches between guidance, tests, and implementation.

## Scientific questions that need an owner answer

1. Is the intended production target shear only, the current eight-parameter
   joint posterior, or both as separately supported configurations?
2. For TNG, should the fourth label be inclination `i`, `sin(i)`, or `cos(i)`?
   The current synthetic and TNG schemas disagree.
3. Is TNG intended only for out-of-distribution evaluation, or should it train
   the same network? The answer determines whether `hlr` and exact fiber
   coordinates must be added.
4. What posterior is mode 2 intended to represent mathematically? In
   particular, is the TF relation a training-population reweighting, an
   observational likelihood/prior applied only at inference, or both?
5. Should the five spectra remain fixed under 90-degree rotation because their
   array slots represent major/minor roles, or should slots follow physical
   fiber coordinates? The codebase contains multiple permutation conventions.
6. Which `theta_int` sign/range is canonical at each boundary: simulator, CSV,
   normalized database, augmentation, and plotting?
7. Are image and spectral noise meant to share the same nominal SNR and
   independent draws, despite different pixel counts and signal morphology?
8. Is freezing the complete pretrained feature extractor during flow training
   intentional, or should it eventually be fine-tuned?
9. Which existing model/checkpoint and dataset pair should be treated as the
   reference baseline for regression tests?
10. Are the calibration scripts intentionally retired, or is `train_cali.py`
    stored elsewhere?

## Suggested stabilization sequence (for a later change request)

1. Declare one canonical model class and repair checkpoint loading plus tests.
2. Define versioned dataset schemas, including names, order, units, ranges, and
   required/optional fields; validate them on dataset open.
3. Decide and test the exact D4/fiber/angle convention in one module.
4. Make TNG output conform to a declared schema or explicitly mark it as an
   evaluation-only adapter.
5. Update inference scripts and notebooks to the current config/model APIs.
6. Add a small local fixture that exercises CSV/FITS-like input through model
   loss and posterior sampling without cluster storage.
7. Add a reproducible environment and package/import structure.
8. Parameterize cluster paths and jobs only after the scientific/data contracts
   are stable.

## Practical entrypoints

- Synthetic sample generation: `data_generate/latin_hypercube.py`
- Synthetic FITS: `data_generate/generate_fits_wrapper.py`
- Synthetic database: `data_generate/make_database.py`
- TNG sample/FITS: `data_generate/latin_hypercube_tng.py` and
  `data_generate/generate_tng_sample.py`
- TNG database: `data_generate/make_tng_database.py`
- Model configuration: `arch/config.py`
- Training CLI: `arch/[scr]_train_model.py`
- Training implementation: `arch/train.py`
- Architecture: `arch/networks.py`
- Posterior batch analysis: `arch/[scr]_tf_analysis.py`
- D4 batch analysis: `arch/d4_diffs.py`
- Primary cached diagnostic notebook:
  `arch/diagnostics/sample_diagnostics.ipynb`

## Confidence and review boundary

Confidence is high for static structure, shapes, call contracts, configured
defaults, and the inconsistencies listed above. Confidence is lower for the
intended scientific semantics of TF reweighting, inclination representation,
and fiber transforms because those require domain decisions not recoverable
from code alone. No large shared datasets, saved checkpoints, external
`kl_tools`/`kl_pipe` sources, or Slurm jobs were inspected or executed.
