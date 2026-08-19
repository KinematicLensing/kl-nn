# Simulator v2: independent photometric and spectroscopic quality

## Scope

Simulator v2 remains a proof-of-concept forward model with one r-band image
and one H-alpha spectral window. It changes the observation model only where
the earlier choices were difficult to justify:

- apparent r magnitude is an independently sampled latent observable, not
  inferred from an injected SNR and not generated from circular velocity;
- integrated H-alpha line flux is sampled independently of magnitude and all
  eight inference parameters;
- the magnitude sets the rendered continuum flux;
- image and spectral noise remain Gaussian but have separate, reproducible
  quality levels;
- the NPE receives the known observation-quality metadata;
- Tully--Fisher information is supplied as an external prior replacement,
  never baked into the base simulations.

The first controlled model uses the galaxy-axis fiber layout and the historical
12-layer affine MAF with exact D4 symmetrization. Bounded/spline flows and
image-axis fibers are separate later ablations.

## Base population

The eight inference parameters retain their broad Latin-hypercube proposal.
For observation-model v2,

```
rmag_true ~ Uniform(15, 23.4)
F_Halpha  ~ Uniform(1.2e-16, 3.0143e-14) erg s^-1 cm^-2
vcirc     ~ Uniform(60, 540 km/s)
```

are drawn from independent randomized Latin hypercubes. The H-alpha quantity
is integrated observed-frame line flux, not flux density per nm, and is passed
directly to KL-tools as `em_Ha_flux`. The numerical interval is the DESI-KL
paper's fiducial grid range; it is not the much broader formal likelihood
prior. This v2 proposal is linear-uniform and independent, intentionally not
the paper grid's logarithmic, continuum-correlated construction. No
Tully--Fisher relation is used to generate training or validation galaxies.

KL-tools renders the continuum with `cont_norm_method='mag'` in DECam r.
Only H-alpha has nonzero line flux in v2. Magnitude, H-alpha flux,
observation-model version, and fiber layout are carried from CSV to FITS to
LMDB, while `fid_pars` remains exactly the eight inference targets. Neither
latent magnitude nor latent H-alpha flux is passed to the NPE as context.

The FITS-to-LMDB archive also requires the r-band/H-alpha codes, spectral
units, center-fiber index, center/offset exposures, rendered-image PSF FWHM,
and image pixel scale. Version-2 packaging and training fail on a missing or
mismatched field; missing-field fallback is reserved for historical v1 data.

Sampling line and continuum brightness independently removes the earlier
deterministic equivalent-width--magnitude relation while retaining a broad,
uninformative proof-of-concept population. It is not intended as an empirical
equivalent-width distribution.

## Gaussian observation levels

The image flux SNR follows the documented fixed-depth relation

```
rho_img = 5 * 10**[-0.4 * (rmag_true - m5)]
```

with `m5=23.4` by default. Because the rendered signal itself now scales with
magnitude, this relation is used for the catalog-flux likelihood, not as a
per-object pixel-noise control.

One dataset-global image pixel RMS is calibrated from the clean training set.
Each image total is first scaled from its rendered magnitude to `m5`, then the
median is taken across every training rank:

```
F5,i    = sum_pixels(I_i) * 10**[-0.4 * (m5 - rmag_true,i)]
F5,ref  = median_training_set(F5,i)
```

The depth reference is a circular Gaussian-PSF-equivalent matched-filter
template, with default FWHM `1.0 arcsec` and pixel scale `0.2637 arcsec`:

```
sigma_PSF,pix = FWHM / [2 sqrt(2 ln 2) pixel_scale]
N_eff         = 4 pi sigma_PSF,pix**2
sigma_img,pix = F5,ref / [5 sqrt(N_eff)]
```

`sigma_img,pix` is a single scalar shared by every galaxy and pixel, is
rank-synchronized during training, and is persisted in the v2 NPE checkpoint
for analysis. The simulator renders an `airy_fwhm` PSF; “Gaussian-equivalent”
describes only this fixed depth-calibration convention, not the exact rendered
PSF. No clean per-object segmentation, morphology, shear, or latent magnitude
is used to alter the pixel RMS.

The catalog summary is generated in linear flux at the same fixed depth. In
units of the survey flux uncertainty,

```
rho_obs = rho_img + Normal(0, 1)
rmag_obs = rmag_true - 2.5 log10(rho_obs / rho_img)
sigma_r = (2.5 / ln 10) / rho_obs
```

The negligible non-positive-flux tail at the configured `rho_img >= 5` is
redrawn, making the logarithmic-magnitude catalog explicitly positive-flux
selected. This remains a versioned Gaussian-flux summary, not a CCD/Tractor
model. Inference sees `rmag_obs`, its reported `sigma_r`, and the observed
`rho_obs`; the expected `rho_img` is private to the catalog likelihood and
observation provenance, and `rmag_true` is never model context.

Spectral reference quality is drawn independently:

```
rho_spec,ref ~ LogUniform(3, 100)
```

A fixed reference H-alpha norm is measured from the clean training-set offset
fibers and persisted in the NPE checkpoint. All four offset fibers receive the
same Gaussian noise sigma. Stored spectra are counts, so the 180-second center
fiber has

```
sigma_center / sigma_offset = sqrt(180 / 600).
```

Actual per-fiber line SNR varies naturally with the independently sampled line
flux and captured aperture fraction and is a diagnostic, not an inference
input. Consequently, the configured 3--100 reference-quality interval is not
the final achieved-H-alpha-SNR interval; its quantiles must be checked on a
pilot sample. Separate RNG streams control image noise,
spectral noise, spectral quality, and the catalog-magnitude measurement.
For matched shear-response groups, all states must carry the same `rmag_true`
and `halpha_flux_true`; the catalog-flux realization, spectral quality, and
image/spectral noise draws are then shared within the group.

For v2 analysis the legacy-compatible `snr/` cache contains the observed
catalog flux SNR, exactly matching the explicit `image_snr/` cache. The latent
expected catalog SNR is not stored under the ambiguous legacy name. Each
partition manifest records this interpretation and the checkpoint pixel RMS.

## NPE conditioning

For v2 the flow context appends five D4-invariant observed scalars to the
1024-dimensional frozen-backbone feature:

```
rmag_obs
rmag_sigma
image_snr  # observed catalog flux SNR, not target/true SNR
spectral_reference_quality
spectral_noise_scale
```

The scalar order is archived in the model config. Every D4 branch receives the
same standardized values. The latent `rmag_true` and `halpha_flux_true` are
explicitly rejected at the model boundary. Legacy-v1 checkpoints retain their
exact 1024-wide context and state-dict layout.

## External Tully--Fisher prior

The v2 NPE is trained in mode 1 under the uniform physical `vcirc` base prior.
At inference, the optional prior replacement evaluates a truncated Gaussian in
`log10(vcirc)`, including magnitude measurement uncertainty, the
`1 / (vcirc ln 10)` Jacobian, and normalization on 60--540 km/s. Complete
joint posterior draws are reweighted by

```
w(theta) proportional to pi_TF(vcirc | rmag_obs, sigma_r) / pi_0(vcirc).
```

The denominator is the known uniform simulation prior, not a KDE of the
posterior marginal. Resampling preserves all shear--nuisance correlations and
reports per-galaxy effective sample size. The code rejects applying this
correction to a mode-2/TF-weighted model, which would count the prior twice.

## Joint-likelihood comparison contract

The NPE marginalizes over the independently drawn H-alpha flux; it is neither
an inference target nor supplied as truth context. A fair joint-likelihood
comparison must therefore include integrated `em_Ha_flux` as a sampled SED
nuisance with the same linear-uniform density on
`[1.2e-16, 3.0143e-14] erg s^-1 cm^-2`, and marginalize it. Fixing the
likelihood fit to the injected flux or to the legacy `1.2e-16` value would give
it extra information and invalidate the comparison. The current external
KL-tools example script does not yet encode this exact comparison prior and
needs to be updated before the paper benchmark.

## Reproducible run sequence

Earlier v2 CSV/FITS/LMDB/checkpoints do not contain `halpha_flux_true` and are
not compatible with this in-place schema revision. The dedicated launchers use
new `_halpha` artifact basenames while keeping `OBSMODV=2`; regenerate both
proposal tables and every downstream artifact rather than appending to the old
directories.

Create independent training and validation proposal tables first:

```bash
python data_generate/latin_hypercube.py --nsamples 100000 --seed 42 \
  --observation-model-version 2 --fiber-layout galaxy_axis \
  --halpha-flux-min 1.2e-16 --halpha-flux-max 301.43e-16 \
  --output /ocean/projects/phy250048p/shared/samples/samples_valid_1m_simv2_galaxyaxis.csv
python data_generate/latin_hypercube.py --nsamples 10000 --seed 43 \
  --observation-model-version 2 --fiber-layout galaxy_axis \
  --halpha-flux-min 1.2e-16 --halpha-flux-max 301.43e-16 \
  --output /ocean/projects/phy250048p/shared/samples/samples_small_1m_simv2_galaxyaxis.csv
```

Generate and package the 100k set with the default 50-way arrays:

```bash
FITS_SUBMISSION=$(sbatch --parsable data_generate/generate_simulator_v2.slurm)
FITS_JOB_ID=${FITS_SUBMISSION%%;*}
DB_SUBMISSION=$(sbatch --parsable --dependency=afterok:${FITS_JOB_ID} \
  data_generate/make_database_simulator_v2.slurm)
DB_JOB_ID=${DB_SUBMISSION%%;*}
sbatch --dependency=afterok:${DB_JOB_ID} data_generate/merge_database_simulator_v2.slurm
```

For the 10k set, use five-task arrays with identical environment overrides and
preserve the dependencies explicitly:

```bash
SMALL_EXPORT=ALL,SAMPLE_NAME=small_1m_simv2_galaxyaxis,DATASET_NAME=small_1m_simv2_galaxyaxis,TOTAL=10000,CHUNK_SIZE=2000
FITS_SUBMISSION=$(sbatch --parsable --array=1-5 --export=${SMALL_EXPORT} data_generate/generate_simulator_v2.slurm)
FITS_JOB_ID=${FITS_SUBMISSION%%;*}
DB_SUBMISSION=$(sbatch --parsable --array=1-5 --dependency=afterok:${FITS_JOB_ID} --export=${SMALL_EXPORT} data_generate/make_database_simulator_v2.slurm)
DB_JOB_ID=${DB_SUBMISSION%%;*}
sbatch --dependency=afterok:${DB_JOB_ID} --export=${SMALL_EXPORT} data_generate/merge_database_simulator_v2.slurm
```

The merger refuses to open an output database until every expected shard is
present. After both LMDBs exist, run
`arch/pretrain_ccl_simulator_v2.slurm`, then pass its model name as
`PRETRAINED_NAME` to `arch/train_npe_simulator_v2_affine.slurm`. Both launchers
use `--fixed-validation-streams`, so checkpoint comparisons reuse identical
validation observation draws.

## Known approximations

- The quoted `m5=23.4` is mapped to pixel RMS through the fixed
  Gaussian-PSF-equivalent template and the training-set median rendered flux.
  It is not an exact reproduction of Legacy Survey Tractor depth or its
  morphology-dependent selection/completeness.
- The Gaussian-flux catalog draw is not derived from the exact image-noise
  realization. It represents a catalog summary paired with a shape image.
- The broad spectral-quality range is a documented proof-of-concept design
  range, not claimed as an empirical DESI-PV distribution.
- Independent linear-uniform continuum magnitude and H-alpha flux are a broad
  design proposal, not a calibrated galaxy equivalent-width population.
- Galaxy-axis fibers reproduce the historical noiseless observed-axis SVD
  convention. Image-axis placement remains a separate controlled ablation.
- Selection effects, multiple bands, multiple emission lines, CCD/sky noise,
  and a physical luminosity population are deliberately out of scope.
