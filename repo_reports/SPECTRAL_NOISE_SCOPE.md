# Spectral noise scope for simulator-v2

> Archived contract. Simulator v3 replaces `spectral_reference_quality` with
> explicit linear `central_halpha_snr`; see `KL-NN_SIMV3_ARCH.md`.

The proof-of-concept keeps a Gaussian observation model. The goal of the
current update is to make the requested noise *level* scientifically
defensible without adding detector physics that is unnecessary for the NPE
versus likelihood-sampling comparison.

## Current contract

- FITS and LMDB store the clean expected H-alpha spectra.
- Training, validation, and inference add zero-mean Gaussian spectral noise at
  runtime.
- `spectral_reference_quality` is sampled independently of photometric
  magnitude and image noise.
- The configured range is deliberately generous, currently log-uniform from 3
  to 100, so it covers plausible galaxies at the fixed redshift rather than
  encoding a narrow selection.
- The noise RMS is calibrated from a robust continuum-subtracted norm of the
  clean reference line, so H-alpha flux still controls the absolute signal.
- The exact sampled quality is supplied to the network as perfect scalar
  context. An achieved SNR estimated from a noisy realization is diagnostic
  output, not model context.
- Center and offset fibers retain their configured exposure-time convention.
  Spectrum-to-fiber-coordinate pairing is never changed by noise injection.

This is independent of the image-noise control. In particular, the code must
not set spectral SNR equal to image SNR or infer spectral quality from
`rmag_true`.

## Deferred realism

The following are intentionally outside the present project scope:

- Poisson source and sky counts;
- read noise and detector gain;
- sky-line or wavelength-dependent variance;
- correlated spectral bins;
- continuum-model uncertainty;
- line-fitting or redshift-pipeline failure;
- a physical joint model of continuum magnitude, star formation, and line
  flux.

These effects can be introduced later as a versioned observation model if the
scientific claim expands. They should not be mixed into the current dataset
without updating both the likelihood baseline and NPE inputs.

## Minimum audit checks

For every generated dataset and training run:

1. Confirm the requested spectral-quality samples cover the configured range
   and are approximately log-uniform.
2. Confirm correlations with `rmag_true`, image SNR, `vcirc`, and H-alpha flux
   are consistent with independent draws.
3. Verify the recovered noise RMS scales inversely with requested quality for
   fixed clean spectra.
4. Check that no NaN or zero reference norm silently becomes a finite noise
   scale.
5. Plot achieved line-measurement precision against requested quality as a
   diagnostic, while retaining the requested value as the oracle input.
6. Apply exactly the same runtime noise construction in validation, posterior
   caching, and the likelihood comparison.
