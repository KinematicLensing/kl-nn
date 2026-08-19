# KL-NN simulator-v2 architecture and pipeline audit

**Audited run:** `CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568`  
**Audit date:** 2026-08-19  
**Scope:** proposal generation, forward simulation, FITS/LMDB construction, runtime observations, CCL pretraining, neural posterior estimation, D4 symmetrization, TF prior replacement, and the 10k shear-bias analysis.

This document is an audit of the concrete model named above, not a generic description of every architecture that exists in the repository. The exact archived run configuration and network snapshot are authoritative where available. Current source is used for the remainder, with provenance limitations called out explicitly.

## Executive summary

The run implements this pipeline:

```text
independent Latin-hypercube proposal
    8 physical inference parameters + true r magnitude + true H-alpha flux
                         |
                         v
KL-tools clean forward model at z=0.3
    one 48x48 DECam-r image + five 64-bin H-alpha fiber spectra
    galaxy/observed-axis fiber cross; no pixel or spectral noise in FITS
                         |
                         v
versioned FITS -> normalized eight-target LMDB + observation metadata
                         |
                         v
on-the-fly independent Gaussian observations each training epoch
    fixed-depth image pixels + catalog magnitude measurement
    independent spectral quality and per-fiber spectral noise
                         |
                         v
Stage-4 D4 feature extractor
    image ResNet-like CNN + shared spectral CNN + fiber set attention
    exact eight-view D4 orbit average -> 1024 equivariant features
                         |
                 CCL pretraining
                         |
                         v
frozen feature extractor + 5 observed scalar context values
                         |
                         v
12-layer conditional affine MAF over 8 normalized parameters
    exact balanced D4 mixture at posterior evaluation/sampling
                         |
                         v
optional inference-only TF prior replacement on vcirc
    q_target(theta|d,m) proportional to q_base(theta|d) pi_TF(v|m)/pi_0(v)
                         |
                         v
5,000 posterior draws per galaxy -> 10k shear-bias report
```

The implementation is internally coherent in its main statistical choice: the simulator does **not** impose a Tully--Fisher relation, mode-1 NPE training learns a broad base posterior, and TF information is applied afterward as an explicit prior ratio. The true magnitude and true H-alpha flux are rejected from neural context, so neither is directly leaked to the posterior.

The main audit findings are:

| Priority | Finding | Consequence |
|---|---|---|
| High | The affine MAF has support on all of R^8 although all training priors are bounded. In the actual TF-reweighted 10k cache, **27.85%** of joint draws are outside at least one physical prior bound. | Posterior means, intervals, MAP selection, and calibration include impossible parameter values. |
| High | `small_1m_simv2_galaxyaxis` is both the 10k validation set used to select the checkpoint and the 10k analysis set. | The HTML is a development/validation result, not a final held-out test. |
| High | The NPE's best validation loss occurs at the final epoch, the LR never falls from `1e-3`, and validation was still improving. | The 20-epoch NPE should be treated as not demonstrably converged. |
| Medium | Training maximizes the **mean of eight D4 branch log densities**, while `posterior_log_prob` evaluates the **log of the arithmetic eight-component mixture**. | Training optimizes a Jensen lower bound/geometric branch average, not the exact likelihood of the posterior density used at inference. |
| Medium | The affine flow treats directed `theta_int` as Euclidean. Returned D4 samples are wrapped into the canonical interval, but the learned density has no circular seam constraint. | Seam behavior and theta modality require explicit monitoring. |
| Medium | Five scalar context fields contain redundancy: `rmag_sigma` is determined by `image_snr`, and `spectral_noise_scale` is determined by spectral quality and a fixed checkpoint buffer. | This is not leakage, but the model effectively receives repeated representations of the same observation-quality information. |
| Medium | A catalog-flux observation is drawn independently of the noisy image pixels rather than measured from that same realization. | The pipeline represents a separate idealized catalog measurement; it should not be described as a measurement extracted from the supplied image. |
| Medium | The exact network source is archived, but the trainer, generator, KL-tools worktree, dependency lockfile, and dataset checksums were not snapshotted with this run. | Exact end-to-end reproduction cannot currently be proven from artifacts alone. |
| Observed | NPE epoch 9 had a very large finite loss/gradient excursion and recovered under clipping. | Numerical health was imperfect even though there were no invalid loss or gradient steps. |

## 1. Exact artifact identity and provenance

### 1.1 Authoritative run artifacts

| Artifact | Path or value |
|---|---|
| Archived configuration | `/ocean/projects/phy250048p/shared/configs/cfg_CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568.json` |
| Config SHA-256 | `336ef48eab30343a3729cd125ac2b351dfd9135cc5e90e9d4ccec91cbd7364b5` |
| Archived network source | `/ocean/projects/phy250048p/shared/networks/networks_CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568.py` |
| Network SHA-256 | `60e3f6b1b8fab5adc4be055facbeb0c041ca8a3a2d6eae3364d6f479882c4c6f` |
| NPE model directory | `/ocean/projects/phy250048p/shared/models/CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568/` |
| NPE log | `/ocean/projects/phy250048p/shared/logs/train_npe_simulator_v2_affine_43901568.out` |
| Pretrained model | `CNN-SetAttn-D4_CCL_simv2_galaxyaxis_s42_43861749`, suffix `19` |
| Pretraining log | `/ocean/projects/phy250048p/shared/logs/pretrain_ccl_simulator_v2_43861749.out` |
| Selected NPE checkpoint | numeric suffix `19`, human epoch 20; also copied to the named `best` checkpoint |
| Best train / validation NLL | `-8.89586823` / `-8.93842512` |
| Analysis report | [`model_report/shear_bias_simv2_10k.html`](../model_report/shear_bias_simv2_10k.html) |

The archived network source is byte-identical to the current [`arch/networks.py`](../arch/networks.py) at audit time. That is strong evidence for the architecture definition. It does not establish that the current trainer and simulator files are byte-identical to those used on August 19, because those files were not archived per run.

At audit time, the KL-NN Git `HEAD` is `74ed6dfb739d9d4b2cfb8f5432ea42a9d9ffa809`, but the worktree contains uncommitted simulator-v2 changes. The KL-tools `HEAD` is `42e06b69e11f0282521c53aac8f5093ce52e3bf9`, also with a dirty worktree. These commit IDs alone are therefore insufficient to reproduce the rendered FITS exactly.

Observed environment versions were PyTorch `2.9.0+cu128`, CUDA `12.8`, normflows `1.7.3`, NumPy `1.26.4`, SciPy `1.13.1`, and Astropy `6.1.0`. The installed nflows package is the 0.14-era API, but it does not expose `__version__` at module level.

### 1.2 Dataset identities

Despite their historical names, the configured datasets are:

| Role | LMDB | Rows |
|---|---|---:|
| NPE and CCL training | `valid_1m_simv2_galaxyaxis` | 100,000 |
| validation and current analysis | `small_1m_simv2_galaxyaxis` | 10,000 |

Both sample CSVs contain exactly 13 columns: `ID`, the eight inference parameters, `rmag_true`, `halpha_flux_true`, `fiber_layout`, and `observation_model_version`. There are no sampled subpixel-position columns in v2.

The exact 100k proposal has correlations `corr(rmag_true, vcirc)=0.00573`, `corr(Halpha, vcirc)=0.00183`, and `corr(rmag_true, Halpha)=0.00610`. The 10k proposal gives `0.00719`, `0.00329`, and `-0.00388`, respectively. These are consistent with independent randomized Latin hypercubes; no TFR is present in data generation.

## 2. Proposal generation

The proposal is implemented in [`data_generate/latin_hypercube.py`](../data_generate/latin_hypercube.py), especially `PARAMETER_LIMITS` at line 35 and `generate_samples` at line 54.

### 2.1 Inference parameters

Each physical parameter is sampled independently and uniformly over its range through one eight-dimensional scrambled Latin hypercube:

| Index | Parameter | Physical range | Meaning |
|---:|---|---|---|
| 0 | `g1` | `[-0.1, 0.1]` | detector-frame shear component |
| 1 | `g2` | `[-0.1, 0.1]` | detector-frame shear component |
| 2 | `theta_int` | `[-pi, pi]` | directed intrinsic position angle |
| 3 | `sini` | `[0, 1]` | sine of disk inclination |
| 4 | `v0` | `[-30, 30] km/s` | velocity zero point |
| 5 | `vcirc` | `[60, 540] km/s` | asymptotic circular velocity |
| 6 | `rscale` | `[0.1, 2.0] arcsec` | arctan velocity scale radius |
| 7 | `hlr` | `[0.1, 3.0] arcsec` | image half-light radius |

The target table is normalized before LMDB storage as

```text
x = 2 (p - p_min) / (p_max - p_min) - 1,
```

so all eight neural targets lie in `[-1, 1]`. `theta_int` is consequently represented as `theta/pi`.

`sini` is uniform, not an isotropic-orientation prior. This is a deliberate broad design proposal, but it should be stated as such; an isotropic disk population would be uniform in `cos(i)`.

### 2.2 Photometric magnitude and line flux

Two separate one-dimensional scrambled Latin hypercubes sample:

- `rmag_true ~ Uniform(15, 23.4)`;
- integrated observer-frame `halpha_flux_true ~ Uniform(1.2e-16, 3.0143e-14) erg s^-1 cm^-2`.

The H-alpha interval is the requested DESI-KL fiducial-grid range. It is **not** the broad likelihood-prior range printed in the DESI-KL paper. The distribution here is linear-uniform, which strongly favors the upper part of a factor-251 interval relative to a log-uniform distribution.

Neither magnitude nor H-alpha flux is an inference target. They define the simulated observation distribution and are retained as truth metadata for auditing.

### 2.3 The apparent “12 parameters” in the simulator

[`data_generate/generate_fits.py`](../data_generate/generate_fits.py) still constructs a KL-tools `Pars` object with 12 internal sampled-parameter slots: the eight targets plus `dx_disk`, `dy_disk`, `dx_spec`, and `dy_spec`. This preserves the historical simulator API. For v2, [`data_generate/generate_fits_wrapper.py`](../data_generate/generate_fits_wrapper.py) forwards only the original eight parameters; all four offset arguments therefore retain their CLI defaults of zero.

The effective v2 proposal is thus **8 inference parameters + magnitude + H-alpha flux**. The four subpixel offsets are present in the simulator object but fixed, not sampled and not passed to the NPE target.

## 3. Forward simulator

The forward model lives in [`data_generate/generate_fits.py`](../data_generate/generate_fits.py), with versioned SED and fiber-layout helpers in [`data_generate/observation_schema.py`](../data_generate/observation_schema.py).

### 3.1 Fixed observing setup

| Quantity | v2 setting |
|---|---|
| Redshift | `z = 0.3` |
| Photometry | one DECam `r` image |
| Image shape | `1 x 48 x 48` |
| Pixel scale | `0.2637 arcsec/pixel` |
| Rendered PSF | Airy profile specified by FWHM `1.0 arcsec` |
| Image exposure metadata | 60 s |
| Spectroscopy | H-alpha only |
| Fiber count | five: two major, center, two minor |
| Fiber radius | `0.75 arcsec` |
| Offset radius | `1.5 arcsec` |
| Center exposure | 180 s |
| Offset exposure | 600 s |
| Spectrum shape | `1 x 5 x 64` |
| H-alpha wavelength window | 851.0--855.81 nm |
| Wavelength sampling | 0.08 nm |

The simulator renders **clean expected data** because `ADD_NOISE=False`. Noise is added later on GPU during pretraining, NPE training, and analysis. The detector noise fields still present in the KL-tools observation config are therefore not the noise model used by this run.

### 3.2 Source model

The image is an inclined exponential profile with `hlr` and a unit spatial normalization. In v2, the SED continuum is then normalized to `rmag_true` through KL-tools' magnitude normalization, so `intensity.flux=1` is a profile normalization rather than a constant observed galaxy brightness.

Only H-alpha remains non-zero. OII, H-beta, and both OIII lines are explicitly set to zero. The sampled integrated H-alpha flux is passed directly to KL-tools as `em_Ha_flux`; its Gaussian wavelength profile has observed sigma 0.065 nm.

The velocity model uses `v0`, `vcirc`, and `rscale` in the existing default/arctan KL-tools velocity field. `sini`, `theta_int`, `g1`, and `g2` enter the projected and sheared disk geometry.

### 3.3 Galaxy-axis fiber convention

The five canonical offsets are `(+-1.5,0)`, `(0,0)`, and `(0,+-1.5)` arcsec. For `fiber_layout=galaxy_axis`, the code forms

```text
T = A(g1,g2) R(theta_int) P(sini),
```

takes the left singular vectors of `T`, fixes their global sign using the transformed intrinsic major axis, and rotates the cross by those observed principal axes. The positions stored beside the spectra are permuted together, so token-to-spectrum pairing is preserved.

An important terminology detail is that these are the **observed/sheared principal axes**, not an oracle unsheared intrinsic-axis placement. This reproduces the historical galaxy-axis convention used by the project.

### 3.4 FITS provenance

Every v2 FITS primary header records the observation-model version, true magnitude, true H-alpha flux, photometric band, target line, fiber layout, spectral units, center-fiber index, exposure times, image PSF FWHM, and image pixel scale. The schema fails closed for missing v2 keys rather than silently substituting v1 defaults.

## 4. FITS-to-LMDB packaging

[`data_generate/make_database.py`](../data_generate/make_database.py) packages batches into Pyxis LMDB records containing:

```text
img             float64 (1, 48, 48)   clean image
spec            float64 (1, 5, 64)    clean spectra
fib_pos         float64 (5, 2)        matching fiber coordinates
fid_pars        float32 (8,)          normalized inference targets
id              uint64
rmag_true       float32               latent simulation metadata
halpha_flux_true float32              latent simulation metadata
observation/instrument schema fields
```

The trainer validates v2 version, fiber layout, magnitude and H-alpha ranges, band, target line, units, exposure convention, PSF, and pixel scale on load. `fid_pars` remains exactly eight-dimensional.

The database normalizer imports parameter ranges from the repository configuration at packaging time. A future audit should archive the normalized sample table and checksums of the CSV, FITS manifest, and final LMDB alongside the model so this dependency cannot drift silently.

## 5. Runtime observation model

The clean LMDB data are converted into noisy observations independently each epoch. Seed 42, deterministic Torch algorithms, separate named RNG streams, and fixed validation streams are enabled.

### 5.1 Image pixels

The code calibrates one scalar pixel RMS from the entire training population:

```text
N_eff = 4 pi [FWHM / (2 sqrt(2 ln 2) pixel_scale)]^2 = 32.5892

F_5,ref = lower_median_i [ sum_pixels(image_i)
                           * 10^(-0.4 * (m_5 - rmag_true_i)) ]

sigma_image = F_5,ref / (5 sqrt(N_eff)).
```

For this checkpoint, `sigma_image = 24.982618` in rendered image units. The same homoscedastic Gaussian pixel RMS is used for every galaxy. The calibration uses a Gaussian PSF noise-equivalent area while the forward image uses an Airy PSF parameterized by FWHM; this is an explicit depth approximation, not an assertion that the rendered PSF is Gaussian.

### 5.2 Catalog magnitude measurement

True magnitude determines only the expected flux SNR at the configured depth:

```text
rho_expected = 5 * 10^[0.4 (23.4 - rmag_true)].
```

A separate noisy catalog measurement is drawn in linear-flux SNR units:

```text
rho_obs = rho_expected + Normal(0,1), conditioned on rho_obs > 0
rmag_obs = rmag_true - 2.5 log10(rho_obs / rho_expected)
rmag_sigma = (2.5 / ln 10) / rho_obs.
```

Only `rmag_obs`, `rmag_sigma`, and `rho_obs` are available to the NPE. `rmag_true` is explicitly rejected by the model API. The expected SNR spans approximately 5 to 11,454 over the proposal.

This catalog-flux deviate and the image-pixel noise are independent streams. This is defensible if the context is interpreted as an external catalog measurement, but it is not a self-consistent extraction from the supplied noisy image. A likelihood comparison must use the same factorization.

### 5.3 Spectral noise

A reference spectral quality is drawn independently for each galaxy:

```text
q_spec ~ LogUniform(3, 100).
```

The checkpoint stores a training-population median continuum-subtracted offset-fiber line norm, `L_ref = 3484.974365`. Offset-fiber Gaussian count noise is

```text
sigma_offset = L_ref / q_spec.
```

Because arrays are in counts and the center exposure is 180 s versus 600 s for offsets,

```text
sigma_center = sigma_offset * sqrt(180/600).
```

All four offset fibers have equal noise. Achieved H-alpha SNR is not restricted to 3--100: it additionally depends on sampled H-alpha flux, aperture throughput, geometry, and line shape. The label `spectral_reference_quality` is therefore correct; it should not be described as the measured per-galaxy line SNR.

### 5.4 Five scalar context fields

The v2 flow context appends these D4-invariant observed values to the visual feature vector:

1. `rmag_obs`;
2. `rmag_sigma`;
3. observed catalog `image_snr`;
4. `spectral_reference_quality`;
5. `spectral_noise_scale`.

They are standardized using the archived magnitude range, SNR 5 as a reference, the log quality range, and the checkpoint line-norm buffer. The mapping API rejects `rmag_true` and `halpha_flux_true` by name.

There are only three independent scalar degrees of freedom here: `rmag_sigma` and image SNR are exact inverses, while spectral noise scale is `L_ref/q_spec`. Redundant representations are not invalid, but an ablation should determine whether all five help.

## 6. Feature architecture

The exact model source is [`arch/networks.py`](../arch/networks.py). Total KLNPE parameter count is **44,139,376**:

| Component | Parameters | Trained during NPE? |
|---|---:|---|
| D4 multimodal feature extractor | 31,268,704 | no; loaded from CCL suffix 19 and frozen |
| 1024-channel LayerNorm | 2,048 | yes |
| conditional diagonal Gaussian base encoder | 141,136 | yes |
| 12 affine autoregressive transforms | 12,727,488 | yes |
| **NPE-trainable total** | **12,870,672** | yes |

### 6.1 Image branch

Before encoding, each image is L2-normalized over its two spatial axes. `ImgCNN` then applies two initial 3x3 convolutions followed by residual stages at 128, 256, and 512 channels with downsampling and final average pooling. Its flattened output is 512 features.

Image L2 normalization deliberately removes total image amplitude from the CNN. Brightness information enters through catalog context, while image morphology and noise texture remain in the normalized pixels.

### 6.2 Spectral branch and fiber tokens

Each of the five spectra is processed by the same `SharedSpecCNN`:

```text
Conv1d 1->32, k=7 -> GroupNorm/GELU -> max pool
Conv1d 32->64, k=5 -> GroupNorm/GELU -> max pool
Conv1d 64->128, k=3 -> GroupNorm/GELU
deterministic mean pool from 16 to 8 wavelength positions
flatten 128*8 -> Linear -> 128-dimensional fiber embedding.
```

Each individual fiber spectrum is L2-normalized along wavelength. The network separately retains each fiber's norm relative to the full five-fiber spectral datavector. A token concatenates:

- 128 spectral features;
- five coordinate features `(x, y, r^2, x^2-y^2, 2xy)`;
- one relative spectral-strength value;
- one observed-mask flag.

The 135 values are projected to a 128-dimensional token. Four-head self-attention operates over the unordered fiber set without a storage-index embedding. A 128-dimensional image-derived query attends to the fiber tokens, producing a 512-dimensional fiber summary.

The 512 image and 512 fiber summaries are concatenated, passed through a residual fusion MLP, and normalized to a 1024-dimensional joint representation.

### 6.3 Exact D4 orbit wrapper

`D4OrbitFeatureExtractor` constructs all eight rotations/reflections of the complete datavector: image, spectra, fiber coordinates, masks, and parameter actions remain synchronized. Every view uses the shared Stage-3 backbone. Output channels are interpreted as:

```text
512 D4 scalars | 256 directed spin-1 channels | 256 spin-2 channels.
```

Each view's raw representation is inverse-aligned to the input frame and the eight aligned features are averaged. This makes the returned 1024-vector exactly equivariant under the implemented discrete D4 action, up to floating-point arithmetic.

For parameters, 90-degree image rotations flip both shear components, directed theta is shifted and wrapped, reflections flip handedness (`g2` and theta), and all remaining parameters are scalars. Reflections also swap the two directed minor-axis fiber entries.

Because every ordinary forward call evaluates eight backbone views, exact D4 symmetry has a substantial compute and memory cost.

## 7. Continuous contrastive pretraining

The pretraining model has **31,421,088** parameters: the 31,268,704-parameter backbone plus a 152,384-parameter equivariant projection head.

The 128-dimensional projection is partitioned into 64 scalar, 32 spin-1, and 32 spin-2 channels. Projected embeddings are L2-normalized. Across the global four-GPU batch, soft target neighbors are defined from the normalized eight-parameter distance

```text
d_ij^2 = mean_k [(theta_ik - theta_jk) / scale_k]^2,
```

where the theta difference is circular and every configured scale is 1. Soft-positive weights are Gaussian with `sigma_label=0.15`; the background mass is set by `d_cutoff=0.40`. Similarity logits use temperature `0.1`. Features and labels are gathered across DDP ranks before evaluating the loss.

Pretraining used four V100-32 GB GPUs, batch 100 per rank, AdamW (`lr=1e-3`, weight decay `1e-4`), deterministic operations, channels-last memory, no AMP, and no compile. A five-epoch linear warmup is followed by cosine decay to `1e-6` at epoch 20. Fixed validation streams make epoch-to-epoch validation directly comparable.

At epoch 20:

- training CCL loss = `4.08856`, excess over target entropy = `1.36517`;
- validation CCL loss = `4.09822`, excess = `1.38409`;
- effective soft positives were about 10.9 per row;
- validation target mass was `0.93933`.

The final pretraining checkpoint, rather than a separately selected best checkpoint, initialized the NPE.

## 8. Conditional affine posterior

### 8.1 Context and base density

The frozen 1024-dimensional backbone feature is passed through a trainable LayerNorm and concatenated with the five standardized observation scalars, giving context dimension 1029.

The base distribution is an eight-dimensional conditional diagonal Gaussian. Its mean and log-scale are produced by an MLP:

```text
1029 -> 128 -> 64 -> 16.
```

### 8.2 Transform stack

The flow applies 12 repetitions of:

```text
ReversePermutation(8)
MaskedAffineAutoregressiveTransform(
    features=8,
    hidden_features=256,
    num_blocks=2,
    context_features=1029)
```

This is a conditional MAF density. It is not a bounded flow and it does not use a circular theta transform. All eight normalized coordinates are modeled on the real line.

### 8.3 D4 posterior objective and inference density

For each training galaxy, the code transforms both context features and target parameters through all eight D4 elements. It then computes eight branch log densities and uses

```text
training score = (1/8) sum_h log q_h(theta | d).
```

At posterior evaluation, however, it returns

```text
log q_D4(theta | d) = log[(1/8) sum_h q_h(theta | d)].
```

The first is the log of a geometric branch average; the second is the log of an arithmetic mixture. Jensen's inequality makes the training score a lower bound on the inference-mixture log density. This can encourage all branches to assign density to the target, but it is not exact maximum likelihood for the density later reported. The distinction should be intentional and tested against direct mixture-NLL training.

Sampling is a balanced mixture: the requested sample count must be divisible by eight, an equal number is drawn from each branch, and all samples are inverse-mapped to the original frame.

### 8.4 Sampling boundary policy

For this affine model, each returned normalized coordinate is clamped to `[-1.5, 1.5]`, not to its physical prior `[-1, 1]`. In D4 sampling, theta is instead wrapped modulo two into `[-1,1)`. Non-finite draws may be replaced by finite draws from the same context, with up to five attempts.

This policy explains the exact hard limits observed in the cache: for example, shear reaches `+-0.15`, `sini` reaches `[-0.25,1.25]`, `v0` reaches `+-45`, `rscale` reaches `[-0.375,2.475]`, and `hlr` reaches `[-0.625,3.725]`.

## 9. NPE training execution

The NPE launcher is [`arch/train_npe_simulator_v2_affine.slurm`](../arch/train_npe_simulator_v2_affine.slurm).

| Setting | Value |
|---|---|
| GPUs | four V100-32 GB, DDP |
| Seed | 42, deterministic algorithms |
| Training rows | 100,000 |
| Validation rows | 10,000 |
| Epochs | 20 |
| Per-rank batch | 50 |
| Optimizer | fused AdamW |
| Initial LR | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 10 |
| Gradient clipping | global norm 1.0 |
| AMP / compile | off / off |
| Feature backbone | frozen and kept in evaluation mode |
| Validation observations | fixed across epochs |
| Objective | unweighted mode-1 negative conditional log density |

No TFR is used during NPE training. The trainer explicitly rejects observation-v2 with mode 2 to prevent TF information from being applied both during training and inference.

The LR remained `1e-3` throughout. The final epoch was the best validation epoch, so the plateau scheduler had no chance to demonstrate convergence. No invalid loss or gradient steps were logged.

There was nevertheless one severe finite excursion at epoch 9: training loss rose to `1.352641e6`, the mean pre-clipping gradient norm to `3.002269e9`, and the maximum to `1.499365e12`; validation remained finite at `-7.68920`, and training recovered on the following epoch under clipping. Later epochs were stable except for a maximum pre-clipping norm of 2748 at epoch 18. This should be retained in any numerical-health summary rather than reporting only the absence of NaNs.

The process emitted NCCL/TCPStore shutdown warnings after saving all artifacts. They appear post-completion rather than evidence of a failed checkpoint, but clean `destroy_process_group` behavior should be confirmed in a future run.

## 10. Inference and TF prior replacement

### 10.1 Broad base posterior

Mode 1 estimates the posterior under the independent uniform training proposal. The 10k launcher loads the archived network source, checkpoint suffix 19, and the same 10k LMDB used for validation. It generates one deterministic noisy observation per galaxy using separate seed streams, then draws 5,000 balanced-D4 candidates.

### 10.2 Prior-replacement formula

For observed magnitude `m_obs` and uncertainty `sigma_m`, the external TF prior is Gaussian in `y=log10(vcirc)`:

```text
mu_y = (m_obs - 36) / (-7.22)
sigma_y^2 = 0.1^2 + [sigma_m / (-7.22)]^2.
```

It is normalized as a truncated base-10 lognormal on `vcirc in [60,540] km/s`, including the `1/(v ln 10)` Jacobian. Since the training prior is uniform in physical velocity,

```text
log w_s = log pi_TF(v_s | m_obs, sigma_m) + log(540 - 60).
```

Weights are normalized within each galaxy, and multinomial sampling with replacement selects complete eight-dimensional posterior rows. This preserves learned correlations between `vcirc`, shear, inclination, sizes, and all other parameters. The code correctly refuses TF prior replacement for a mode-2 model.

The returned log score is the base D4-mixture log density plus the selected prior-ratio log weight. It is proportional to the target posterior density but omits the per-galaxy evidence constant.

### 10.3 Importance-sampling health

Across the full 10k analysis, each galaxy started with 5,000 candidates. TF ESS quantiles were:

| Quantile | ESS | ESS fraction |
|---:|---:|---:|
| min | 1.0 | 0.00020 |
| 1% | 2.47 | 0.00049 |
| 10% | 57.2 | 0.0114 |
| median | 1,452 | 0.290 |
| 90% | 3,725 | 0.745 |
| 99% | 4,872 | 0.974 |
| max | 4,997 | 0.999 |

The median is acceptable for a diagnostic, but the lowest tail is not. The maximum normalized weight is 1.0 in the worst case and exceeds 0.61 at the 99th percentile. Those galaxies are effectively represented by only a handful of unique candidates after resampling and should be flagged or rerun with a larger/adaptive candidate bank.

## 11. Actual cache and report snapshot

The analyzed cache is:

```text
/ocean/projects/phy250048p/shared/cache/
  CNN-SetAttn-D4-affine_simv2_galaxyaxis_s42_43901568/
  small_1m_simv2_galaxyaxis_tf_prior_replacement_
  simv2_prior_replacement_10k_s42/
```

It contains ten partitions of 1,000 galaxies, 5,000 samples per galaxy, exact D4 sampling, mode 1, TF prior replacement, and no counter-rotated-noise cancellation. All 50 million cached parameter vectors and all cached log scores are finite.

### 11.1 Physical support audit

After TF replacement, physical out-of-support fractions are:

| Parameter | Outside prior |
|---|---:|
| `g1` | 9.51% |
| `g2` | 9.80% |
| `theta_int` | 0% after wrapping |
| `sini` | 1.27% |
| `v0` | 2.65% |
| `vcirc` | 0% because the TF factor is truncated to support |
| `rscale` | 9.06% |
| `hlr` | 2.04% |
| **any coordinate** | **27.85%** |

The cache is therefore numerically finite but not a posterior on the declared physical parameter space.

### 11.2 Current shear-bias summary

For the primary posterior mean, the current HTML reports:

- `10^2 m = -35.95 ± 6.87` for g1 and `-38.76 ± 7.15` for g2 in the low-absolute-shear fit;
- cubic `10^2 m = -33.86 ± 1.71` and `-32.53 ± 1.72`;
- `10^4 c = 2.92 ± 4.58` and `5.38 ± 4.55`;
- 16th--84th coverage of 52.5% for g1 and 53.0% for g2.

Coverage for `sini` and `vcirc` is 41.8% and 23.7%, respectively; theta coverage is 72.6%. These are large conditional-posterior failures in this v2 run, not merely point-estimator shrinkage.

The directed-theta diagnostic reports median true-branch mass 0.999, opposite-branch mass 0.000, and middle mass 0.001. It classifies 89.3% of galaxies as one mode, 2.7% as two modes, 7.4% as three or more modes, and 0.7% unresolved/flat. The aggregate posterior is one-mode, but a nontrivial per-galaxy multimodal tail remains.

These results are included only to connect architecture to observed behavior. They are not a held-out performance claim because the same 10k rows selected the checkpoint.

## 12. Items I would audit or change next

### 12.1 Required before a paper-quality performance claim

1. **Create a disjoint test set.** Keep `small_1m_simv2_galaxyaxis` for model selection, generate a new frozen 10k or larger test LMDB with a separately seeded proposal, and never use it for checkpoint choice.
2. **Resolve posterior support.** The controlled comparison should either use a genuinely bounded density for all bounded scalars or explicitly define and validate a mathematically consistent support transform. Post-hoc all-parameter filtering changes the posterior and is not a substitute.
3. **Train to a convergence criterion.** Extend the affine run until validation plateaus or use a scheduler that actually decays within the budget. Compare architectures at matched examples and optimizer steps.
4. **Snapshot the whole experiment.** Archive generator, schema, database packer, data/training/analysis modules, KL-tools revision/diff, environment lock, exact launcher, sample CSV checksums, LMDB manifest/checksum, and Slurm job environment.
5. **Use the same H-alpha prior in likelihood sampling.** H-alpha flux must be a likelihood nuisance with the same linear-uniform `[1.2e-16,3.0143e-14]` proposal; fixing it at truth or at the old constant value makes the NPE/MCMC comparison unfair.

### 12.2 High-value controlled diagnostics

1. Compare broad-prior inference and TF replacement on identical noisy observations, with ESS-binned results.
2. Report calibration and bias both before and after removing galaxies below documented ESS thresholds; do not silently drop them.
3. Compare the present D4 geometric-average training objective with direct arithmetic-mixture NLL on the same frozen backbone and data.
4. Add support-violation rates to every analysis manifest and HTML, including componentwise and joint rates.
5. Measure achieved H-alpha SNR from each clean spectrum and its known noise, then plot its distribution against sampled line flux, `hlr`, inclination, and reference quality.
6. Ablate redundant scalar context: `(rmag_obs, flux_ivar)` or an equivalent two-scalar photometric representation, plus one known spectral-noise variable, is sufficient in principle.
7. Verify the NPE and joint-likelihood comparator use exactly the same catalog-flux likelihood, image-pixel likelihood, spectral variance, priors, and selection.
8. Repeat the 10k posterior analysis on the checkpoint both with and without physical-support enforcement only as a diagnostic of how much current summaries are driven by leakage.

### 12.3 Modeling choices to disclose rather than necessarily change

- redshift is fixed at 0.3;
- one image band and one emission line are used;
- magnitude, H-alpha flux, and the eight physical targets are mutually independent in the base proposal;
- H-alpha is linear-uniform over a factor-251 interval;
- `sini` is uniform rather than isotropic orientation;
- subpixel image/spectrum offsets are fixed to zero;
- fiber positions follow observed/sheared principal axes;
- image and spectral noise are homoscedastic Gaussian models with independently drawn quality;
- the catalog magnitude observation is an independent summary, not extracted from the noisy pixels;
- the image CNN sees an L2-normalized image and each spectral CNN sees an L2-normalized fiber spectrum;
- the external TF relation is an inference prior, not part of the simulated population.

## 13. Personal audit checklist

Use this as a short reproducibility walk-through.

### Proposal and data

- [ ] Confirm the two sample CSVs have exactly the 13 expected columns and no subpixel columns.
- [ ] Recompute every proposal min/max and all correlations with magnitude and H-alpha.
- [ ] Randomly select IDs and verify CSV -> FITS header -> LMDB metadata equality.
- [ ] Confirm `fid_pars` contains exactly the eight normalized targets in the archived order.
- [ ] Confirm every v2 FITS has `OBSMODV`, `RMAGTRUE`, `HAFLUX`, `FIBLAY`, band/line, units, exposure, PSF, and pixel-scale headers.
- [ ] Render a few controlled galaxies and verify magnitude normalization and integrated H-alpha flux numerically.
- [ ] Check fiber positions and spectra remain paired under the galaxy-axis transform and any permutation.

### Observation model

- [ ] Recompute `N_eff`, global image sigma, and spectral reference line norm from the training LMDB.
- [ ] Confirm checkpoint buffers equal `24.982618` and `3484.974365`.
- [ ] Verify fixed validation streams reproduce identical noisy validation tensors across epochs.
- [ ] Check observed-flux pulls and positive-flux redraw rate.
- [ ] Check actual per-fiber H-alpha SNR rather than relying on reference quality.
- [ ] Confirm no latent magnitude or H-alpha value enters `observation_context`.

### Architecture and training

- [ ] Load the archived config and archived network module, not current defaults.
- [ ] Verify pretrained suffix 19 loads strictly and all backbone parameters are frozen.
- [ ] Check a D4 orbit numerically for image, spectra, coordinates, targets, and feature blocks.
- [ ] Verify CCL theta distance wraps at the seam.
- [ ] Confirm all NPE trainable parameters appear exactly once in the optimizer.
- [ ] Review epoch 9 and all gradient-norm spikes from the NPE log.
- [ ] Confirm checkpoint suffix 19 and named `best` have identical state dictionaries.

### Inference

- [ ] Confirm 5,000 is divisible by eight and every D4 branch contributes 625 candidates.
- [ ] Re-evaluate a sample under the eight-component arithmetic mixture.
- [ ] Numerically integrate the truncated TF density over `[60,540]` to one.
- [ ] Verify prior replacement divides by the known uniform physical `vcirc` prior, not by a posterior KDE.
- [ ] Verify resampling gathers complete joint rows.
- [ ] Inspect low-ESS galaxies individually.
- [ ] Recompute finite-value and physical-support statistics from all sample partitions.
- [ ] Run final bias/coverage analysis only on a disjoint held-out test set.

## 14. Source map

| Pipeline stage | Primary implementation |
|---|---|
| Proposal generation | [`data_generate/latin_hypercube.py`](../data_generate/latin_hypercube.py) |
| Versioned SED, fibers, FITS/LMDB schema | [`data_generate/observation_schema.py`](../data_generate/observation_schema.py) |
| CSV-to-simulator forwarding | [`data_generate/generate_fits_wrapper.py`](../data_generate/generate_fits_wrapper.py) |
| KL-tools forward model and FITS writing | [`data_generate/generate_fits.py`](../data_generate/generate_fits.py) |
| FITS-to-LMDB normalization and packaging | [`data_generate/make_database.py`](../data_generate/make_database.py) |
| D4 transforms and runtime noise helpers | [`arch/data.py`](../arch/data.py) |
| Feature extractor, CCL, flow, D4 posterior, TF ratio | [`arch/networks.py`](../arch/networks.py) |
| DDP loading, RNG streams, augmentation, training, inference helpers | [`arch/train.py`](../arch/train.py) |
| Configuration schema | [`arch/config.py`](../arch/config.py) |
| Training CLI | [`arch/[scr]_train_model.py`](<../arch/[scr]_train_model.py>) |
| CCL launcher | [`arch/pretrain_ccl_simulator_v2.slurm`](../arch/pretrain_ccl_simulator_v2.slurm) |
| NPE launcher | [`arch/train_npe_simulator_v2_affine.slurm`](../arch/train_npe_simulator_v2_affine.slurm) |
| TF analysis driver | [`arch/[scr]_tf_analysis.py`](<../arch/[scr]_tf_analysis.py>) |
| 10k inference launcher | [`arch/tf_analysis_simulator_v2_10k.slurm`](../arch/tf_analysis_simulator_v2_10k.slurm) |
| HTML report generator | [`arch/diagnostics/shear_bias_report.py`](../arch/diagnostics/shear_bias_report.py) |

## Bottom line

This run is a valid proof-of-concept implementation of a broad-prior, multimodal image-plus-fiber NPE with exact discrete D4 handling and an external inference-time TF prior. Its most defensible current successes are the explicit separation of simulator proposal from TF information, strict v2 metadata validation, latent-context guards, deterministic observation streams, complete-row TF reweighting, and archived network/config artifacts.

It is not yet a paper-quality demonstration that the NPE matches joint likelihood sampling. The current result is validation-set performance from a still-improving 20-epoch model, its affine posterior leaks substantially outside physical support, and the low-ESS tail makes TF resampling unreliable for some galaxies. Those issues should be addressed or explicitly controlled before interpreting the present m/c and coverage numbers as properties of the NPE approach itself.
