# KL-NN simulator-v2 architecture

**Updated:** 2026-08-20

**Status:** current design contract

**Scope:** data generation, observation model, feature pretraining, neural
posterior estimation, posterior caching, and post-training TF weighting

This document describes the one configuration maintained by the repository
after the breaking cleanup. It is a design and audit checklist, not a claim
that a new model trained after the cleanup has passed scientific validation.
Numerical performance must be established from a fresh held-out run.

## 1. Pipeline at a glance

```text
independent broad proposal
  nine posterior targets + true r magnitude
                    |
                    v
simulator-v2 clean expectation
  one r-band image + five H-alpha fiber spectra
                    |
                    v
runtime Gaussian observations
  photometric depth and independently sampled spectral quality
                    |
                    v
image CNN + shared spectral CNN + fiber set attention
  identity/R90 pair, CCL-pretrained representation
                    |
                    v
bounded hybrid conditional posterior
  eight bounded non-angular targets + circular theta_int
                    |
                    v
base posterior candidate cache
  equal identity/R90 mixture, returned in one coordinate frame
                    |
                    v
post-training TF importance ratios
  within-galaxy posterior weights + across-galaxy population ratios
```

There is no TF relation in proposal generation or model training. There is no
alternative point-estimator path or training-time TF posterior.

## 2. Proposal and target schema

The proposal is deliberately uninformative for the proof of concept. Each
field is drawn independently over its configured interval; randomized Latin
hypercubes improve coverage without introducing a population relation.

| Index | Target | Range | Neural support |
|---:|---|---:|---|
| 0 | `g1` | `[-0.1, 0.1]` | bounded |
| 1 | `g2` | `[-0.1, 0.1]` | bounded |
| 2 | `theta_int` | `[-pi, pi)` | circular |
| 3 | `sini` | `[0, 1]` | bounded |
| 4 | `v0` | `[-30, 30] km/s` | bounded |
| 5 | `vcirc` | `[60, 540] km/s` | bounded |
| 6 | `rscale` | `[0.1, 2.0] arcsec` | bounded |
| 7 | `hlr` | `[0.1, 3.0] arcsec` | bounded |
| 8 | `halpha_flux_true` | `[1.2, 301.43] x 10^-16 erg/s/cm^2` | bounded |

`halpha_flux_true` is both a forward-simulator input and a posterior target.
It must be present in the database `fid_pars`, the CCL label distance, the NPE
feature list, normalization, denormalization, and cached truth arrays.

`rmag_true` is sampled independently over `[15, 23.4]`. It controls image
brightness and is supplied as perfectly known scalar context, but it is not a
posterior target. In particular, no `vcirc`--magnitude relation is imposed by
the simulator.

The order in `config.TARGET_NAMES` is the only supported target order. Code
must resolve named indices rather than repeat numeric indices at call sites.

## 3. Forward observation

The minimal observation is sufficient for the stated proof-of-concept goal:

- one 48 by 48 ground-based r-band image;
- one H-alpha line observed through five fibers;
- fibers ordered as the two major-axis offsets, center, then the two
  minor-axis offsets;
- galaxy-axis placement with the coordinates stored beside each spectrum;
- clean expected image and spectra saved to FITS/LMDB, with stochastic noise
  applied at training and inference time.

The image flux is set from `rmag_true`. The integrated H-alpha flux is drawn
independently and uniformly in linear flux over the range above, then passed
into the spectral forward model.
This removes both constant-galaxy-flux and constant-line-flux assumptions while
avoiding an unrequested star-formation or TF population model.

The database must retain, at minimum, `img`, `spec`, `fib_pos`, `fid_pars`,
`rmag_true`, `halpha_flux_true`, the galaxy identifier, and the versioned
instrument metadata needed to reject incompatible data.

## 4. Noise levels and oracle context

The noise family remains intentionally simple: additive Gaussian noise for
image pixels and spectra. Realism is introduced through noise *levels*, not a
detector-level stochastic model.

Photometric brightness and spectral quality are independent controls:

- `rmag_true` sets the clean image signal, while the configured survey depth
  calibrates one fixed pixel RMS; expected image SNR is therefore a
  deterministic diagnostic of true magnitude;
- `spectral_reference_quality` is drawn independently over the configured
  generous range, currently log-uniform from 3 to 100;
- the requested spectral quality is converted to an RMS using the clean
  continuum-subtracted reference-line norm;
- image SNR is never reused as spectral SNR.

The only scalar context fields are, in this exact order:

```text
rmag_true, spectral_reference_quality
```

These fields are treated as perfect simulator controls. The model does not
receive an observed magnitude, magnitude uncertainty, achieved noisy SNR,
spectral noise scale, or redundant transformations of either context. This
keeps measurement-method bias outside the proof-of-concept model.

The context normalizer maps magnitude linearly over its configured range and
maps spectral quality linearly in log space. Schema validation must reject
missing, extra, non-finite, or non-positive fields.

See `repo_reports/SPECTRAL_NOISE_SCOPE.md` for the explicit noise-model
boundary.

## 5. Feature extractor and CCL pretraining

The feature extractor contains most of the trainable capacity, so a useful
pretrained representation is a prerequisite rather than an optional warm
start. Its supported structure is:

1. an image CNN for the single photometric cutout;
2. a shared spectral CNN applied to each fiber spectrum;
3. fiber-coordinate embeddings paired with their corresponding spectra;
4. permutation-aware set attention over the five spectral tokens;
5. fusion of image and spectral representations into the fixed feature vector.

Before per-fiber spectrum normalization, the extractor retains a
continuum-subtracted absolute line-norm scalar for each fiber, alongside the
relative fiber strength. This is necessary for independently varying H-alpha
flux to remain learnable after normalization; it is computed from the observed
spectrum and is not latent-truth context.

The CCL projection head concatenates that feature vector with the same two
oracle context scalars used by the NPE. The projection head is a pretraining
objective component and is not transferred as part of the frozen backbone.

CCL pretraining uses all nine normalized targets in its label distance. The
H-alpha label scale should be monitored because its raw physical interval is
large even though the target stored for training is normalized.

Every noisy batch has exactly two deterministic geometry branches: the
original observation and one 90-degree rotation. The rotated branch must
transform image pixels, `g1`, `g2`, `theta_int`, and fiber coordinates
consistently while preserving spectrum-to-coordinate pairing. Both views form
one doubled CCL batch, so the pairwise loss can include cross-view pairs; that
single loss is followed by one optimizer step. Taking one optimizer step per
view would change the objective and effective learning rate and is not the
intended contract.

For multi-GPU CCL, image-backbone BatchNorm layers are converted to
SyncBatchNorm before optimizer construction. Consequently every rank validates
the same running statistics and the rank-zero best checkpoint represents the
globally trained feature extractor. The synchronized state remains strictly
loadable into ordinary BatchNorm for frozen single-process inference.

The CCL logs should include target entropy, the uniform baseline, excess loss,
effective positives, and target mass. A falling excess loss and downstream
frozen-feature probes are the practical checks that pretraining is learning
the needed simulator information.

Canonical launcher: `arch/pretrain_ccl.slurm`.

## 6. Bounded hybrid posterior

NPE training initializes the feature extractor from a CCL checkpoint and
freezes it. The learned conditional density has the factorization

```text
q(x, theta_int | context)
  = q_box(x | context) q_circle(theta_int | x, context),
```

where `x` contains all eight non-angular targets. The non-angular density uses
compact rational-quadratic autoregressive transforms with a conditional unit
box base. The angular factor uses conditional circular splines and is
conditioned on both the fused observation context and the sampled non-angular
parameters. This retains correlations between orientation and the rest of the
posterior while respecting the angular seam.

The public density must return `-inf` outside the configured non-angular
support. Sampling must fail on a support violation rather than silently clamp
an invalid draw. Training diagnostics should separately track non-angular and
angular log probability, spline derivative extrema, transform logits,
gradient norms, invalid batches, and support violations.

The identity and R90 observations are included during NPE training. At
inference, candidate counts must be even: half are drawn from each observation
branch, and samples from the rotated branch are inverse-transformed into the
original coordinate system before concatenation. The result is one equal
mixture, not two galaxy observations.

Canonical launcher: `arch/train_npe.slurm`.

## 7. Post-training TF importance sampling

TF information is isolated in `arch/tf_prior.py` and enters only after joint
base-posterior candidates have been drawn. The default relation is a truncated
normal in `log10(vcirc)` conditional on perfect `rmag_true`, with configurable
slope, intercept, intrinsic scatter, and the same velocity bounds as the
proposal.

Two distinct importance calculations are required.

### 7.1 Candidate weights within one galaxy

For each posterior candidate `j` of galaxy `i`, compute

```text
log r_ij = log p_TF(vcirc_ij | rmag_true_i)
           - log p_uniform(vcirc_ij).
```

Normalize these ratios across candidates for that galaxy only. Those weights
produce TF-target posterior means, quantiles, and other within-galaxy
summaries. Save ESS, ESS fraction, maximum weight, and the log mean ratio so a
poor candidate proposal is visible. Do not resample and discard the weights.

### 7.2 Population weights across galaxies

The broad simulator proposal also overrepresents galaxies far from the assumed
TF population. For each simulated galaxy, evaluate the same log ratio at its
*true* `vcirc` and `rmag_true`. Cache that unnormalized scalar. After all
partitions used by a report are concatenated, normalize the full vector once
and use it in every ensemble statistic, fit, bin average, bootstrap, coverage
estimate, and selection summary intended to represent the TF population.

Normalizing separately inside each partition would give partitions equal
mass regardless of their TF likelihood and is therefore incorrect. Candidate
weights and population weights solve different problems and must never be
substituted for each other.

This conditional TF ratio preserves the simulator's uniform magnitude
marginal: its target is `q_sim(rmag) p_TF(vcirc | rmag)`. A Tully--Fisher
relation alone does not define a luminosity function or survey selection. If a
future analysis targets a different magnitude distribution, it must add the
separate documented ratio `p_target(rmag) / q_sim(rmag)` rather than silently
calling the conditional TF weighting a complete galaxy population model.

Canonical cache entry points: `arch/cache_posteriors.py` and
`arch/cache_posteriors.slurm`.

## 8. Cache contract

The current cache stores one physical-unit candidate bank plus enough metadata
to reproduce both proposal-posterior and TF-target summaries. The stable core
includes:

- `sample` and `base_log_prob`;
- `posterior_tf_log_ratio`, normalized `posterior_tf_log_weight`, and its
  linear-space counterpart `posterior_tf_weight`;
- posterior TF ESS, ESS fraction, maximum weight, and log mean ratio;
- unnormalized `population_tf_log_ratio`;
- `truth`, `rmag_true`, and `spectral_reference_quality`;
- proposal and TF-target MAP/mean summaries;
- a manifest containing model, checkpoint, dataset, partition, target order,
  R90 alignment policy, and TF hyperparameters.

Reports require the complete manifest set and fail closed if partitions differ
in checkpoint, dataset, target order, TF definition, posterior semantics, or
non-seed observation provenance. Array filenames, row ranges, and shapes must
exactly match those manifests.

Reports should consume the named proposal or TF-target products. A numeric
model selector is not part of the cache schema.

## 9. Required validation before scientific use

Before interpreting shear bias, record the following checks for the exact
checkpoint and held-out dataset:

1. Validate the nine-target database schema and target order.
2. Confirm proposal correlations between magnitude, `vcirc`, and H-alpha flux
   are consistent with independent sampling.
3. Confirm requested spectral quality is independent of photometric magnitude
   unless a scientifically justified selection is deliberately added.
4. Unit-test the forward and inverse R90 parameter and fiber-coordinate maps.
5. Verify one CCL optimizer update per original/R90 batch pair.
6. Examine CCL excess loss and frozen-feature probe performance.
7. Verify the feature extractor is frozen during NPE optimization.
8. Check bounded-support and circular-seam diagnostics on held-out data.
9. Check NPE convergence with learning curves and a checkpoint not selected on
   the final scientific test set.
10. Inspect posterior TF ESS before using weighted summaries.
11. Concatenate all report partitions and normalize population weights once.
12. Report proposal-posterior and TF-target results with explicit labels.
13. Keep response-calibration and response-validation base galaxies disjoint.

## 10. Deliberate non-goals

This proof of concept does not attempt to model Poisson counting statistics,
sky lines, wavelength-dependent variance, image resampling covariance,
catalog extraction errors, multiple emission lines, or a realistic joint
galaxy population. Those may be valuable later, but they are not required to
test whether NPE can reproduce the corresponding simple likelihood analysis.

The scientifically defensible claim is narrower: within the declared
simulator, proposal, noise-level ranges, and post-training population prior,
compare NPE with joint likelihood sampling on genuinely held-out data.
