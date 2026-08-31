# KL-NN simulator-v3 observation contract

Simulator v3 is a retraining boundary. V2 FITS, LMDBs, model configs, and
checkpoints are not rewritten; new artifacts must use names containing
`simv3` and the dedicated v3 launchers.

## Physical proposal

The nine inference targets keep their existing order. `hlr` and `rscale` now
span `[0.1, 5.0]` arcsec. `halpha_flux_true` means the clean, integrated
central-fiber H-alpha flux after seeing/aperture losses and before instrument
throughput, exposure, or noise. It is sampled log-uniformly from
`1e-17` to `1e-14 erg s^-1 cm^-2` and normalized through `log10(flux)`.
All other target transforms remain identity.

`kl-tools` converts the requested central-fiber flux to the total line flux
once, using the central PSF-convolved aperture fraction, then builds the model
cube once. FITS provenance records the requested central flux, derived total
flux, aperture fraction, semantics, transform, units, and calibration API
version (`HAFAPI=1`).

Training draws the observation controls independently for every galaxy in
every epoch:

- `image_snr ~ Uniform(5, 1000)`
- `central_halpha_snr ~ Uniform(1, 200)`

The same draw controls noise amplitude and the metadata supplied to the model.
Validation uses deterministic epoch-zero draws so validation losses remain
comparable across epochs. Existing v3 sample, FITS, and LMDB S/N fields are
retained as legacy compatibility metadata but are ignored by CCL and NPE
training. Together with `rmag_true`, the active S/N draws are the model's
three oracle context fields. Both S/N fields use linear min-max normalization;
no target truth is admitted through the context path.

## Feature-extraction architecture

The simulator-v3 model uses three deliberately simple branches:

- The unchanged image CNN produces 512 features.
- A fixed-order, joint 2D spectral CNN produces 512 features from the complete
  `5 x 64` fiber-by-wavelength array. Spectra receive one global L2
  normalization per galaxy, preserving relative amplitudes between fibers.
- The metadata MLP maps 13 inputs through `13 -> 64 -> 128`: the ten flattened
  fiber-position coordinates followed by exactly `rmag_true`, `image_snr`, and
  `central_halpha_snr`.

The three branch outputs are concatenated directly into a 1152-dimensional
feature vector. There is no Set Attention block and no per-fiber spectral
normalization. This replacement changes parameter names and tensor shapes, so
older feature-extractor and CCL checkpoints are intentionally incompatible and
must not be partially loaded into this architecture.

## Runtime noise

FITS/LMDB data remain clean. Every training epoch draws fresh S/N controls and
fresh Gaussian noise realizations independently. The epoch S/N draw is reused
consistently by noise scaling and the metadata context.

- Image white-noise RMS is `||I_clean||_2 / image_snr`.
- Central spectral white-noise RMS is the continuum-subtracted central-line
  `L2` norm divided by `central_halpha_snr`; LMDB wavelength padding is
  excluded from both the norm and the added noise.
- Spectra are in counts. Offset-fiber RMS is central RMS times
  `sqrt(600 / 180)`, retaining the Xu-2024 180 s central / 600 s offset
  exposure convention.

For the five-row shear stencil, S/N values, base-record signal norms, and the
standard-normal noise realization are shared across
`zero, g1+, g1-, g2+, g2-`. This gives matched noise while allowing the clean
sheared signals to differ.

The retired v2 global image-noise scalar, spectral reference-line norm, and
epoch-drawn spectral-quality variable are not model buffers or runtime inputs.

## Posterior cache and reports

Cache schema v2 stores the two requested S/N values and the realized image and
central-spectrum RMS values per galaxy. It also records physical parameter
ranges, target transforms, observation semantics, and density coordinates.
Because H-alpha uses a nonlinear transform, cached MAP rows maximize physical
density by subtracting `log|d physical / d normalized|`; posterior means and
quantiles are computed after physical denormalization. The cache reader and
report retain explicit legacy-v1 read support.

`shear_bias_report.py --weighted` composes population weights with the
spin-symmetric shear precision
`2 / [Var(g1) + Var(g2)]`. It reports 90th-, 95th-, 99th-percentile, and
uncapped precision sweeps; downstream weighted panels use the 95th-percentile
cap. The cap clips precision values rather than removing galaxies. Raw TF
importance-sampling ESS remains separately reported. One 2x4 nuisance figure
overlays proposal and TF posterior-Mean residuals in common truth bins.

## CCL boundary

No auxiliary shear loss, CCL formula change, or manual label-scale change is
part of v3. `ccl_label_scales` is declared in `PretrainConfig` in
`arch/config.py`; `CCLPretrain.__init__` in `arch/networks.py` reads it in
feature order and passes the resulting vector to `ContinuousContrastiveLoss`.
The default empty mapping therefore leaves every normalized label at scale 1.

## Launch sequence

Use `latin_hypercube.py` to create the v3 sample table, then submit:

1. `data_generate/generate_simulator_v3.slurm`
2. `data_generate/make_database_simulator_v3.slurm`
3. `data_generate/merge_database_simulator_v3.slurm`
4. `arch/pretrain_ccl.slurm`
5. `arch/train_npe.slurm`

Run a small pilot and inspect S/N reconstruction, central/total H-alpha
provenance, 5-arcsec edge cases, and generation throughput before launching
the full one-million-object array.
