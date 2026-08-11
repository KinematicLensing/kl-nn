# DESI Spectral-Noise Notes

## Scope and current decision

For now, the model will continue to use scalar spectral SNR. No spectral-noise code changes are proposed at this stage. These notes preserve the DESI noise research for later work, when a more realistic noise model becomes a higher priority.

The main conclusion of the research is that there is no single physically complete “DESI spectral SNR.” A realistic DESI-like simulation should eventually use wavelength- and fiber-dependent variance. The existing instrument settings in `data_generate/generate_fits.py` already reproduce the Xu et al. setup surprisingly closely; the main physical simplification comes later, when the spectra are assigned one scalar Gaussian noise level.

## What Xu et al. assume

The fiducial mock observations in Xu et al. use:

| Quantity | Fiducial value |
|---|---:|
| Conditions | Typical dark time |
| Sky model | DESI `specsim` dark sky |
| Airmass | 1 |
| Extinction | None |
| Seeing FWHM | 1.0 arcsec |
| Sky brightness | $r=20.61\ {\rm mag\,arcsec^{-2}}$ |
| Central exposure | 180 s |
| Each offset exposure | 600 s |
| Alternative offset exposure | 900 s |
| Fiber radius | 0.75 arcsec |
| Spectral pixel width | 0.08 nm |
| Window around each line | 5 nm |
| Effective telescope diameter | 332.42 cm |
| z-camera read noise | 2.6 electrons/pixel |
| Gain | 1 electron/ADU |

They simulate H-alpha, the [O II] doublet, and both [O III] lines, with an intrinsic line-width standard deviation of 0.05 nm, approximately 23 km/s. Their reported baseline is the 600-second offset exposure. See [Xu et al. (2024)](https://arxiv.org/html/2407.20867v1).

This closely matches `data_generate/generate_fits.py`. An important difference is that the current simulations retain only H-alpha, whereas the paper's forecasts combine four emission lines.

## Recommended future noise model

For each fiber $f$ and wavelength pixel $i$, generate an expected electron count and use a variance of the form

$$
\sigma^2_{fi}
=
N_{{\rm source},fi}
+
N_{{\rm sky},fi}
+
n_{{\rm CCD},fi}\,{\rm RN}^2
+
\sigma^2_{{\rm sky-model},fi}.
$$

Then sample

$$
d_{fi}\sim \mathcal{N}\!\left(\mu_{fi},\,\sigma^2_{fi}\right),
$$

or retain $\mu_{fi}$ and the inverse variance separately and draw noise during training.

The DESI pipeline produces flux, inverse variance, masks, and a resolution matrix for every spectrum. Its variance propagation includes extraction noise, detector noise, sky subtraction, and additional sky-model uncertainty. That provides a much better template than uniform white noise. See the [DESI spectroscopic pipeline paper](https://arxiv.org/abs/2209.14482) and [DESI data-product glossary](https://data.desi.lbl.gov/doc/glossary/).

In particular:

- Central and offset fibers should not share a noise amplitude. Their exposure times and collected source flux differ.
- Noise should vary strongly with wavelength. H-alpha at $z=0.3$ lies near 853 nm, where structured night-sky emission matters.
- Bad or sky-dominated pixels should have masks or very low inverse variance.
- The resolution matrix should act on the noiseless spectral model before noise is drawn.
- Image SNR and spectral SNR should be sampled separately, although both can depend on galaxy magnitude and observing conditions.

The current scalar-noise implementation assigns one standard deviation to the whole five-fiber tensor. Its requested SNR is effectively a total segmented-array SNR, not a per-pixel, continuum, or emission-line SNR, so it is difficult to compare directly with DESI quantities.

## Scalar-SNR fallback

While retaining scalar SNR, a useful definition would be the integrated H-alpha line SNR for each fiber:

$$
{\rm SNR}_{\rm H\alpha}
=
\frac{F_{\rm H\alpha}}{\sigma(F_{\rm H\alpha})}.
$$

DESI's `TSNR2` should not be used directly for this purpose. It is a target-class-weighted survey metric based on spectrum noise, rather than an H-alpha measurement SNR. The distinction is described in the [DESI glossary](https://data.desi.lbl.gov/doc/glossary/).

As a provisional engineering distribution, not a published DESI result, use:

- Broad coverage: per-fiber H-alpha SNR of roughly 1--100.
- Most training probability: approximately 3--30 for offset fibers.
- A deliberately difficult population below 5.
- Values above 100 only as a bright tail.
- Values of $10^3$--$10^4$ only for effectively noiseless diagnostics.

The current log-uniform interval from 1 to $10^4$ puts too much training support in regimes unlikely to resemble actual offset-fiber observations.

For intuition, at 853 nm and $R\sim4500$, the instrumental line width is about 28 km/s in Gaussian-$\sigma$ units. Combining that with the paper's 23 km/s intrinsic width gives approximately 36 km/s. The idealized centroid uncertainty is then

$$
\sigma_v \approx \frac{36\ {\rm km\,s^{-1}}}{{\rm SNR}_{\rm line}}.
$$

Line SNRs of 5, 10, and 20 therefore correspond approximately to 7, 4, and 2 km/s centroid precision. Since shear-induced minor-axis velocities can be only a few km/s, SNR 10--20 per useful line is a sensible desirable range, while SNR around 5 is marginal but scientifically valuable. This is an inference from DESI's resolution and the line model in Xu et al., not a sensitivity claim made by the paper.

Under sky-dominated conditions, changing an offset exposure from 180 to 600 seconds improves SNR by only

$$
\sqrt{600/180}\approx1.83.
$$

The offset fiber generally collects less galaxy light, so its longer exposure does not automatically imply better SNR than the central fiber.

## Empirical calibration route

DESI DR1 provides a way to measure the relevant distribution rather than guessing it:

1. Select BGS disk galaxies matched to the simulated redshift, apparent magnitude, size, and fiber magnitude.
2. Use the DR1 emission-line catalog's H-alpha flux and inverse variance, or refit the spectra using their per-pixel inverse variance and resolution matrices.
3. Measure $F_{\rm H\alpha}\sqrt{{\rm IVAR}_{\rm H\alpha}}$.
4. Model that distribution conditional on magnitude, redshift, size, and possibly color.
5. Use the simulator's surface-brightness model to predict how much line flux enters each offset fiber.
6. Recompute the 600- or 900-second variance with `specsim`, rather than scaling only the final SNR.

The official DR1 emission-line products expose H-alpha flux and inverse-variance fields. See the [DESI DR1 emission-line catalog documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/emfit/).

DESI BGS normally uses an effective 180-second exposure and obtains high redshift success, but redshift success is a substantially weaker requirement than measuring several-km/s spatial velocity differences. It should therefore not be used as evidence that 180 seconds is adequate for kinematic lensing. See the [DESI BGS validation paper](https://arxiv.org/abs/2208.08512).

## Deferred recommendation

When realistic spectral noise becomes a priority, preserve the Xu et al. 180/600-second configuration, introduce per-pixel inverse variance from `specsim`, and treat H-alpha-only versus four-line spectra as an explicit ablation. Until then, a per-fiber H-alpha SNR distribution concentrated around 3--30 is a more defensible placeholder than a shared 1--$10^4$ scalar.
