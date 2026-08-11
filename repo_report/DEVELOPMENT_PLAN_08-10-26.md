# Development Plan — 2026-08-10

## Scientific objective

Build a joint neural posterior estimator for the eight KL parameters
`[g1, g2, theta_int, sini, v0, vcirc, rscale, hlr]`, with the final shear
measurements meeting approximately `|c| < 1e-4` and `|m| < 1e-2`. Kinematic
information and the Tully–Fisher (TF) prior must constrain intrinsic shape
rather than merely calibrating shear after inference.

This plan prioritizes a physically consistent datavector, correct parameter
geometry, and symmetry-aware feature extraction before revisiting TF
application choices or simulation-derived response calibration.

## Fixed conventions

- Images use array coordinates. `torch.rot90(..., k=1)` maps a fiber position
  `(x, y) -> (y, -x)`.
- Positive `theta_int` rotates clockwise relative to a normal displayed-array
  counter-clockwise rotation.
- Fiber order is `[major+, major-, center, minor+, minor-]`.
- The sign on the major fibers denotes positive/negative radial velocity.
- `minor+` and `minor-` are 90 degrees counter-clockwise from the
  corresponding directed major fibers.
- Simulator, CSV, and plotting values are physical. Every parameter presented
  to the model is normalized to `[-1, 1]`.
- `theta_int` is a directed, periodic `2 pi` parameter: angles separated by
  `pi` reverse the velocity field and are not physically equivalent.
- D4 symmetry is sufficient; no larger discrete group is required.

## Stage 0 — Correct the known 90-degree transform

Status: completed before this plan was recorded.

The fiber-position action accompanying `torch.rot90(k=1)` was corrected from
`(-y, x)` to `(y, -x)`. The old action placed every directed fiber on the
opposite side of the rotated galaxy and produced the observed
`theta_int +/- pi` posterior branch.

Regression tests cover the one-step action and identity after four rotations.

## Stage 1 — One authoritative D4 datavector action

Put all D4 transformations in `arch/data.py`. Diagnostic scripts may consume
that API but must not maintain independent image, spectrum, fiber-position, or
parameter transformation tables.

The generators are:

### 90-degree array rotation

- image: `torch.rot90(k=1)`
- fiber position: `(x, y) -> (y, -x)`
- shear: `(g1, g2) -> (-g1, -g2)`
- normalized angle: `theta_int -> wrap(theta_int - 0.5)`
- spectra: unchanged at the same fiber indices
- remaining parameters: unchanged

### Row-axis reflection

- image: flip the image row axis
- fiber position: `(x, y) -> (x, -y)`
- shear: `(g1, g2) -> (g1, -g2)`
- normalized angle: `theta_int -> -theta_int`
- spectra and positions: swap `minor+` and `minor-`
- remaining parameters: unchanged

All eight elements are compositions of these generators.

Required tests:

- `r^4 = e`
- `s^2 = e`
- `s r s = r^-1`
- closure and inverse behavior for every element
- complete image/spectrum/fiber/parameter association
- correct periodic angle wrapping
- backward compatibility of the existing 90-degree helper

The old `arch/d4_diffs.py` names and diagnostics can remain, but its
layout-specific transformation implementation is replaced by calls into
`data.py`.

## Stage 2 — Correct CCL parameter geometry

Retain separate original and rotated CCL similarity matrices. Do not
concatenate the two label sets into one contrastive matrix.

Replace batch-standardized Euclidean label distance with a fixed-scale metric.
Because all model parameters are already normalized, a default scale of one is
one half of each parameter's physical prior span. Aggregate featurewise squared
distances with a mean so the kernel width does not shrink exponentially merely
because more KL parameters are included.

For `theta_int`, use the directed circular difference

```text
delta_theta = atan2(sin(pi * delta_normalized),
                    cos(pi * delta_normalized)) / pi
```

so values adjacent across the `-pi/pi` seam are close, while angles separated
by `pi` remain maximally separated.

The initial implementation preserves the intentional two-pass FETrainer
behavior. A later option is to calculate two independent CCL losses and average
them in one optimizer step; that would still prohibit cross-pairs while
removing optimizer-order dependence.

The first training run occurs after this stage.

## Stage 3 — Shared spectral encoder and permutation-aware fusion

Replace 2D convolution across the semantic fiber axis with a shared 1D
spectral encoder:

```text
each spectrum_i -> shared Conv1d encoder -> spectral embedding_i
```

Construct a token for each observed fiber from its spectrum embedding,
coordinates, D4-useful coordinate features, and observation mask. Process the
fiber tokens with a permutation-equivariant set-attention block with no
learned storage-index positional encoding. Simultaneously permuting spectra,
positions, and masks must leave the pooled representation unchanged.

Use the global image feature as an attention query over fiber-token keys and
values. Fuse the attended fiber summary and image feature through a residual
MLP. If needed in a later iteration, sample the spatial image feature map at
each fiber coordinate and append that local feature to its fiber token.

Run the small train/validation experiment after this stage.

## Stage 4 — Full multimodal D4 equivariance

D4 must act on the complete multimodal datavector, not only `ImgCNN`.
Following the full-orbit idea in D4CNN x AnaCal, evaluate a shared multimodal
backbone over all eight transformed views and map representation blocks back
to a common frame.

Useful representation blocks are:

- invariant scalar channels for `sini, v0, vcirc, rscale, hlr`;
- spin-1 channel pairs for the directed angle representation
  `(cos(theta_int), sin(theta_int))`;
- spin-2 channel pairs for shear-like information.

Once exact D4 model tests pass, remove the rot90 training counterpart. A
hard-coded orbit construction should provide the discrete symmetry instead of
requiring the model to learn it from duplicated examples.

For posterior training, define for datavector `d`, parameters `y`, and group
element `g`:

```text
ell_g = log p_phi(rho_g(y) | T_g(d))
L_orbit = -mean_g ell_g
```

At inference, use the symmetrized density

```text
p_D4(y | d) = (1/8) sum_g p_phi(rho_g(y) | T_g(d)).
```

This guarantees posterior equivariance. If the eight-branch cost proves
impractical, test a sampled posterior log-density consistency penalty as the
fallback, recognizing that it encourages rather than guarantees equivariance.

Run the small experiment after this stage.

## Stage 5 — Periodic theta flow

Implement a genuinely periodic circular spline for `theta_int` only after the
feature extractor and D4 action are trustworthy.

Required tests include:

- continuity across the `-pi/pi` seam;
- forward/inverse agreement;
- correct log-Jacobian;
- valid sampling and log probability near the seam;
- compatibility with every D4 angle action.

Run the small experiment after this stage.

## Stage 6 — Controlled staged evaluation

For each training stage:

- train on `valid_1m` (100,000 samples);
- validate on `small_1m` (10,000 samples);
- keep model settings, SNR law, TF treatment, seeds, GPU count, and evaluation
  sample fixed unless the stage explicitly changes one;
- record the full configuration and source snapshot with the checkpoint.

Evaluate at minimum:

- shear additive, multiplicative, and cubic bias;
- theta residual contours and the frequency of the `+/- pi` branch;
- D4 posterior discrepancies;
- all eight parameter residuals;
- posterior coverage;
- shear bias versus theta, inclination, and SNR.

A seed is a run configuration, not just a Torch call. The launch path must seed
Python, NumPy, Torch CPU, and every CUDA process before model initialization,
then derive deterministic rank- and stream-specific seeds for sample order,
SNR, image noise, spectral noise, augmentation, and validation. Strict runs
also disable cuDNN benchmarking, request deterministic Torch algorithms, set
`PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`, and keep the GPU count and
software stack fixed.

This guarantees a repeatable fresh run under the fixed environment above.
Current checkpoints contain model weights only, so exact mid-run continuation
would additionally require optimizer, scheduler, scaler, Python/NumPy/Torch,
CUDA, and explicit generator states; resumable RNG state is deferred.

## Stage 7 — Hold TF fixed, then revisit it

Use the current training-time TF weighting and inference-time TF reweighting
for the main staged models. Removing TF would change the physical inference
problem and confound architecture comparisons.

TF quantities are invariant under D4, so every orbit branch for one galaxy
must receive the same TF weight.

At major milestones, inspect the raw unreweighted flow posterior as a
diagnostic. Do not run the full TF-application factorial after every
architecture change. Once the feature extractor, D4 behavior, and periodic
theta flow are stable, compare TF strategies on that single trusted
architecture.

Response calibration remains a last resort rather than a target part of this
development path.

## First-run handoff criterion

The first training run is ready when:

1. the centralized D4 group/action tests pass;
2. the CCL seam, fixed-scale, and gradient tests pass;
3. the corrected rot90 path still passes its regression tests;
4. the training CLI accepts and records a seed;
5. the small-run Slurm script fixes train/validation paths, GPU count,
   deterministic settings, and all effective configuration overrides;
6. the focused CPU test suite passes.

