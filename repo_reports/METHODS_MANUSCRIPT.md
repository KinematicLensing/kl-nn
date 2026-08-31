# KL-NN simulator-v3 methods manuscript

<!-- klnn-methods-source-sha256: 0df40b67c1af44a267307ca29b78257d393c4b5e60a3fd19245deb788dd08584 -->

> **Living-document status.** This manuscript describes the simulator-v3
> working tree on 2026-08-28. The implementation, rather than this prose, is
> normative. Repository-specific statements below therefore cite the source
> files that define them; empirical or methodological statements cite the
> literature. The source fingerprint above is checked by
> [`check_methods_manuscript.py`](check_methods_manuscript.py), so a change to
> the data-generation, architecture, or training surface requires a review and
> corresponding update of this manuscript. The checked-out `kl-nn` tree is
> based on commit `2832e974fbf75f87191d27e4e09e4ef40ec20ff6` but contains
> simulator-v3 changes beyond that commit, so the present working tree is the
> relevant implementation snapshot.

## Generating DESI-like image and spectra

### Scope of the forward model

The scientific setting is **kinematic weak lensing**: imaging constrains the
apparent morphology of a disk galaxy, while spatially resolved emission-line
spectroscopy constrains its projected velocity field. Their joint use can
separate intrinsic disk orientation from weak gravitational shear in a way
that imaging alone cannot. This is the inference problem developed for
resolved velocity maps by [Pranjal et al. (2023)](#pranjal2023) and specialized
to a five-fiber DESI observing concept by [Xu et al. (2024)](#xu2024). It is not
a strong-lensing simulation: the repository contains no foreground lens-mass
model, multiple-image mapping, caustics, or magnification calculation.

Following the notation of the primary paper draft, the physical simulator
parameters are collected into

\[
\begin{aligned}
\bm{\theta}={}&\bigl(g_1,g_2,\theta_\mathrm{int},\sin i,v_0,v_\mathrm{circ},\\
&R_\mathrm{vscale},R_h,F_{\mathrm{H}\alpha,\mathrm{cen}}\bigr).
\end{aligned}
\]

The simulator produces a data vector
\(\bm{D}(\bm{\theta})\) whose learned observational components are the model
image \(M_I\), the ordered set of five model spectra \(M_S\), and the fiber
coordinates. Symbols in the scientific text follow this convention, while
literal code fields such as `rscale`, `hlr`, and `halpha_flux_true` are retained
when identifying implementation objects.

Here, “DESI-like” has a deliberately narrow operational meaning. The simulator
uses a DESI bandpass/instrument configuration, five 1.5-arcsec fibers, and a
narrow redshifted H-alpha window, while the image uses a DECam-like pixel scale
and `r`-band response. DESI itself is a 5,020-fiber, broad-wavelength survey
spectrograph ([DESI Collaboration 2022](#desi2022)); standard DESI survey
observations do not place five fibers on every galaxy. That five-fiber strategy
comes from the kinematic-lensing experiment of [Xu et al. (2024)](#xu2024).
Likewise, the simulated image is a single idealized cutout, not a Legacy
Surveys coadd or reduction product. The precise approximation is defined by
[`generate_fits.py`](../data_generate/generate_fits.py) and
[`observation_schema.py`](../data_generate/observation_schema.py).

### Space-filling proposal

The simulator begins from a scrambled Latin-hypercube design (LHS), a
stratified construction introduced by [McKay, Beckman, and Conover
(1979)](#mckay1979). Eight physical coordinates are generated together in an
eight-dimensional LHS; separate one-dimensional LHS streams generate
`rmag_true`, log10 H-alpha flux, image S/N, and central-fiber H-alpha S/N. The
five random-number streams are spawned from one `SeedSequence`. This is a
space-filling experimental design over a rectangular domain, not a draw from a
measured galaxy population. In particular, it imposes no Tully--Fisher
relation or other astrophysical covariance among size, luminosity, rotation
speed, inclination, line flux, and shear. A uniform proposal in `sini` also
must not be interpreted as the isotropic-orientation distribution, for which
cosine inclination is uniform. These facts follow directly from
[`generate_samples`](../data_generate/latin_hypercube.py) and the immutable
schema in [`config.py`](../arch/config.py).

The LHS command has no fixed seed by default: unless `--seed` is supplied,
`SeedSequence(None)` obtains fresh entropy and the exact table cannot be
reconstructed from the remaining command-line arguments. Reproducible
production therefore requires recording an explicit seed as well as the table
itself ([`latin_hypercube.py`](../data_generate/latin_hypercube.py)).

### Catalog-backed TF-conformed evaluation populations

The rectangular LHS above remains the training and validation proposal.
Evaluation additionally supports a distinct `--test-set` mode that constructs
an empirical population from one DESI selection-cut catalog. It scans the FITS
table in bounded-memory blocks and samples jointly eligible rows uniformly
without replacement. True r magnitude, half-light radius, image S/N, and
central H-alpha S/N are copied from the same selected row, preserving their
catalog dependence. The catalog's `xu_effective_weight` is recorded only as
provenance and is used neither for row selection nor for downstream ensemble
weighting.

Eligibility is restricted to the trained model's support: (15leq rleq
23.4), (5leq
ho_Ileq1000), (1leq
ho_{mathrm{H}alpha}leq200),
and finite (0.1leq R_hleq5) arcsec, with inclusive bounds. Rows outside
any support bound are excluded before the uniform joint-row draw; HLR is not
clamped after selection. Catalog redshift is retained as source metadata but
all images and spectra are still rendered at the model's fixed (z=0.3).

The remaining coordinates use the usual ranges. The shear components,
intrinsic angle, systemic velocity, and velocity scale radius use uniform LHS
coverage; the synthetic true H-alpha flux remains log-uniform. Inclination is
isotropic rather than uniform in sine: a one-dimensional LHS covers
(cos isimmathcal U(0,1)), followed by
(sin i=sqrt{1-cos^2 i}). Circular speed is drawn by inverse CDF from the
same truncated TF conditional used for post-training prior replacement,

[
log_{10}v_mathrm{circ}mid r
sim mathcal N!left(rac{r-36}{-7.22},,0.1^2
ight),
qquad 60leq v_mathrm{circ}leq540 {
m km,s^{-1}}.
]

Each generated CSV has a JSON sidecar that records the catalog identity,
eligibility and support audit, random seed, parameter laws, TF configuration,
fixed redshift, row-ID policy, and CSV SHA-256. Database construction validates
that digest and row count and installs the sidecar as `manifest.json` only
after successful LMDB finalization. Every generated FITS also records the row
ID and a versioned SHA-256 over all simulator parameters, observation controls,
fiber layout, and observation-model version. Resume mode skips an existing
file only when this full-row identity and the structural FITS checks agree;
legacy or mismatched files are regenerated. These behaviors are defined by
[`desi_test_set_sampling.py`](../data_generate/desi_test_set_sampling.py),
[`generate_desi_test_sets.slurm`](../data_generate/generate_desi_test_sets.slurm),
[`generation_integrity.py`](../data_generate/generation_integrity.py), and
[`make_database.py`](../data_generate/make_database.py).

Every batch-array task uses an isolated scratch child beneath
`/ocean/projects/phy250048p/shared/tmp`, exports that child through the standard
temporary-directory environment variables, and removes only that child on
normal exit, failure, or a catchable termination signal. The persistent shared
root is never used directly as task scratch and is never cleared globally.

At posterior-cache time, test-set mode consumes the S/N values stored in each
record without redrawing or clipping them. Although the truth population is
already TF-conformed, the neural posterior was trained under an independent
uniform circular-speed prior. The cache therefore applies the usual
within-galaxy candidate ratio
(p_mathrm{TF}(v_mathrm{circ}mid r)/p_0(v_mathrm{circ})) and normalizes it
separately for every galaxy. It does not apply a second, truth-level population
TF ratio across the already-conformed rows. Test-set mode omits MAP
optimization and stores TF-weighted 16th/Mean/84th summaries for all targets,
physical ((g_1,g_2)) candidates, normalized candidate log weights, and
candidate-weight diagnostics. The shear-bias report uses those candidate
weights for posterior intervals, ranks, and shear variance. It starts with
equal truth-galaxy mass. The unweighted test-set report retains that mass,
whereas the weighted test-set report composes it with posterior precision
regularized by a fixed ensemble shape-noise floor. Both report the Mean only.
As an independent production audit, the report maps every cached
((r,v_mathrm{circ})) truth pair through the embedded truncated-TF
conditional CDF and reports its uniform-KS/DKW status and residual/PIT
quantiles. Invalid TF configuration or out-of-support truth aborts the report;
statistical nonconformance is displayed as a failed audit without changing any
analysis weight. The compact schema, cache writer, and report behavior are defined by
[`cache_contract.py`](../arch/cache_contract.py),
[`cache_posteriors.py`](../arch/cache_posteriors.py), and
[`shear_bias_report.py`](../arch/diagnostics/shear_bias_report.py).

The `meta_pars["priors"]` block inside the FITS generator is metadata for
the downstream `kl-tools` likelihood interface; it does not draw the simulated
rows because the LHS values are passed directly as the fiducial parameter
vector. Likewise, `sampled_pars_std` is constructed but is not used to perturb
that vector. Only the proposal-table code and ranges above define the generated
training proposal ([`generate_fits.py`](../data_generate/generate_fits.py)).

The nine inference targets, their proposal bounds, and their stored
transformations are listed below. Except for H-alpha flux, each LHS coordinate
is linear in the displayed physical quantity. The units are those supplied to
the forward renderer by [`generate_fits.py`](../data_generate/generate_fits.py);
the ordering, bounds, and transforms are fixed by `TARGET_NAMES`,
`CANONICAL_PARAMETER_RANGES`, and `TARGET_TRANSFORMS` in
[`config.py`](../arch/config.py).

| Stored target | Physical interval | Proposal coordinate | Forward-model role |
|---|---:|---|---|
| \(g_1\) | \([-0.1,0.1]\) | linear | Cartesian reduced-shear component |
| \(g_2\) | \([-0.1,0.1]\) | linear | Cartesian reduced-shear component |
| \(\theta_\mathrm{int}\) | \([-\pi,\pi]\) rad | linear and periodic | intrinsic disk position angle |
| \(\sin i\) | \([0,1]\) | linear | sine of disk inclination |
| \(v_0\) | \([-30,30]\) km s\(^{-1}\) | linear | systemic line-of-sight velocity |
| \(v_\mathrm{circ}\) | \([60,540]\) km s\(^{-1}\) | linear | asymptotic speed of the arctangent rotation curve |
| \(R_\mathrm{vscale}\) (`rscale`) | \([0.1,5]\) arcsec | linear | turnover radius of the rotation curve |
| \(R_h\) (`hlr`) | \([0.1,5]\) arcsec | linear | half-light radius of the exponential disk |
| \(F_{\mathrm{H}\alpha,\mathrm{cen}}\) | \([10^{-17},10^{-14}]\) erg s\(^{-1}\) cm\(^{-2}\) | uniform in \(\log_{10}F\) | clean central-fiber integrated H-alpha flux after seeing/aperture and before instrument response |

The proposal table also carries three non-target observation descriptors.
`rmag_true` is stratified over \([15,23.4]\); the upper bound coincides with
the nominal 5-sigma `r`-band depth quoted for the Legacy Surveys by [Dey et al.
(2019)](#dey2019), but the full interval is a repository choice rather than a
catalog selection function. `image_snr` is stratified over \([5,1000]\), and
`central_halpha_snr` over \([1,200]\). The latter two values survive into FITS
and LMDB metadata but are replaced by new uniform draws during every training
epoch; `rmag_true` is retained. At held-out inference, the stored S/N fields are
used instead. The generation definitions are in
[`latin_hypercube.py`](../data_generate/latin_hypercube.py), whereas the
training replacement occurs in [`Trainer._prepare_epoch`](../arch/train.py).

After FITS conversion, each physical target \(\theta_j\) is mapped to a
dimensionless coordinate \(\widetilde{\theta}_j\in[-1,1]\). If \(Q_j\)
is the identity except \(Q_F(F)=\log_{10}F\), and
\(a_j=Q_j(\theta_{j,\min})\),
\(b_j=Q_j(\theta_{j,\max})\), then the database stores

\[
\widetilde{\theta}_j =
\frac{2Q_j(\theta_j)-(a_j+b_j)}{b_j-a_j}.
\]

This is an implementation identity, not an empirical scaling relation; it is
defined by [`normalize_targets`](../arch/utils.py) and invoked by
[`make_database.py`](../data_generate/make_database.py).

### Disk, velocity field, and spectral-energy distribution

The source is a centered, infinitesimally thin inclined exponential disk
parameterized by `hlr`. Exponential disks are the classical model of [Freeman
(1970)](#freeman1970), and the actual analytic profile, rotation, shear,
inclination, PSF convolution, and pixel rendering are performed with GalSim
([Rowe et al. 2015](#rowe2015)). The simulator fixes the photometric and
spectroscopic centroids to zero. It does not sample a Sersic index, bulge,
spiral structure, dust field, clumps, bars, warps, or non-circular motions;
these absences are visible in the `inclined_exp` configuration in
[`generate_fits.py`](../data_generate/generate_fits.py).

The installed `kl-tools` dependency evaluates the rotation law

\[
v_\mathrm{rot}(R)=\frac{2v_\mathrm{circ}}{\pi}
\tan^{-1}\!\left(\frac{R}{R_\mathrm{vscale}}\right),
\qquad
v_\mathrm{LoS}(R,\phi)=v_0+\sin i\cos\phi\,v_\mathrm{rot}(R),
\]

and Doppler-shifts the local spectrum with this line-of-sight field. The
arctangent form is the empirical parameterization used by [Courteau
(1997)](#courteau1997); it is a simplified rotation curve, not a dynamical mass
model. The exact equations are also auditable in the pinned dependency source
[`kl_tools/velocity.py`](https://github.com/wxs0703/kl-tools/blob/42e06b69e11f0282521c53aac8f5093ce52e3bf9/kl_tools/velocity.py).

All galaxies are placed at fixed redshift \(z=0.3\). The continuum uses the
local `GSB2.spec` starburst template and is normalized to `rmag_true` through
the DECam `r` response. The template belongs to the SB2 family of [Kinney et
al. (1996)](#kinney1996), but the local GSB2 file is a resampled/extrapolated
product listed by [Gwyn](#gwyn). The repository does not embed a checksum or a
complete provenance record for that file, so it should not be described as an
unchanged Kinney spectrum or assumed reproducible from the citation alone.
H-alpha is represented by a Gaussian component with
configured width 0.065 nm. The [O II], H-beta, and [O III] amplitudes are set
exactly to zero, so the current spectral observation contains only continuum
plus H-alpha. These choices are made by `base_sed` in
[`generate_fits.py`](../data_generate/generate_fits.py) and enforced by
[`configure_sed`](../data_generate/observation_schema.py).

The flux target is not the total H-alpha flux of the galaxy. It denotes the
clean H-alpha flux integrated by the on-axis fiber after PSF convolution and
aperture loss, but before throughput, exposure, or noise. The simulator first
renders a unit-total spatial profile, measures its central-fiber aperture
fraction \(f_{\rm ap}\), and supplies the total line flux
\(F_\mathrm{total}=F_{\mathrm{H}\alpha,\mathrm{cen}}/f_\mathrm{ap}\) to the spectral cube.
It refuses a result unless
\(F_\mathrm{total}f_\mathrm{ap}=F_{\mathrm{H}\alpha,\mathrm{cen}}\) within the configured
numerical tolerance, and it records all three values in the FITS header. This
semantics and dependency API contract are defined and checked in
[`observation_schema.py`](../data_generate/observation_schema.py) and
[`generate_fits.py`](../data_generate/generate_fits.py). No conversion to
H-alpha luminosity or star-formation rate is performed.

The installed dependency adopts a vacuum H-alpha rest wavelength of 6564.589
Angstrom, which the fixed redshift places at 853.3966 nm before peculiar
velocity shifts. At each spatial sample it evaluates the spectrum at

\[
\lambda_\mathrm{emit}=\lambda_\mathrm{obs}/(1+v_\mathrm{LoS}/c)
\]

and includes the corresponding wavelength Jacobian in the model cube. Four
high-resolution wavelength subsamples are summed into each 0.08-nm output bin.
Each monochromatic plane is weighted by the PSF-convolved circular-fiber mask
and integrated spatially; the resulting spectrum is multiplied by the DESI
throughput and by collecting-area, exposure-time, and gain factors to produce
counts. Finally, the configured `FIBRBLUR=3.4` constructs an 11-sample Gaussian
resolution kernel with sigma `3.4/4 = 0.85` output bins. This fixed kernel is
not a wavelength-dependent DESI extraction matrix. The construction is
auditable in the pinned dependency's
[`emission.py`](https://github.com/wxs0703/kl-tools/blob/42e06b69e11f0282521c53aac8f5093ce52e3bf9/kl_tools/emission.py),
[`likelihood.py`](https://github.com/wxs0703/kl-tools/blob/42e06b69e11f0282521c53aac8f5093ce52e3bf9/kl_tools/likelihood.py),
and
[`cube.py`](https://github.com/wxs0703/kl-tools/blob/42e06b69e11f0282521c53aac8f5093ce52e3bf9/kl_tools/cube.py),
with the active wrapper settings in
[`generate_fits.py`](../data_generate/generate_fits.py).

### Fiber placement and instrumental approximation

Five circular fibers are ordered as positive first axis, negative first axis,
center, positive second axis, and negative second axis. Before alignment their
centers are \((\pm1.5,0)\), \((0,0)\), and \((0,\pm1.5)\) arcsec. For each
galaxy, define the primary manuscript's shear, rotation, and projection
matrices as

\[
\mathbf G(\bm g)=
\begin{pmatrix}1-g_1&-g_2\\-g_2&1+g_1\end{pmatrix},
\quad
\mathbf R(\theta_\mathrm{int})=
\begin{pmatrix}
\cos\theta_\mathrm{int}&-\sin\theta_\mathrm{int}\\\sin\theta_\mathrm{int}&\cos\theta_\mathrm{int}
\end{pmatrix},
\quad
\mathbf P(i)=
\begin{pmatrix}1&0\\0&\cos i\end{pmatrix}.
\]

With those definitions, the fiber-placement implementation uses the
image-coordinate convention

\[
\mathbf T_\mathrm{fib}
=\mathbf G(-\bm g)\mathbf R(\theta_\mathrm{int})\mathbf P(i)
=\begin{pmatrix}1+g_1&g_2\\g_2&1-g_1\end{pmatrix}
\mathbf R(\theta_\mathrm{int})\mathbf P(i).
\]

It uses the left singular vectors of \(\mathbf T_\mathrm{fib}\) as the observed principal
axes, fixes the sign of the first axis against the transformed intrinsic
major-axis direction, and rotates the five centers into that basis. The
equation above is a transcription of
[`compute_fiber_offsets`](../data_generate/observation_schema.py). It makes the
fiber order galaxy-axis-relative rather than a fixed sky-axis order. The
central-plus-four-offset geometry and asymmetric exposure strategy follow the
DESI kinematic-lensing concept in [Xu et al. (2024)](#xu2024), rather than the
standard survey tiling described by [DESI Collaboration (2022)](#desi2022).

The complete clean observation is summarized below. All numerical entries are
implementation definitions from
[`generate_fits.py`](../data_generate/generate_fits.py), with schema constants
cross-checked by [`observation_schema.py`](../data_generate/observation_schema.py).
The DECam plate scale is independently documented by [Flaugher et al.
(2015)](#flaugher2015), and the broader DESI instrument by [DESI Collaboration
(2022)](#desi2022).

| Component | Current simulator-v3 definition |
|---|---|
| Image | One `r`-band \(48\times48\) cutout; 0.2637 arcsec pixel\(^{-1}\), hence 12.66 arcsec on a side; 60 s configured exposure; DECam response |
| Image PSF | GalSim/instrument `airy_fwhm` kernel forced to 1.0-arcsec FWHM; this is a numerical PSF choice, not a full atmospheric-seeing model |
| Spatial model grid | \(64\times64\) samples at 0.11 arcsec sample\(^{-1}\), fourfold supersampling |
| Fibers | Five circular apertures of radius 0.75 arcsec (1.5-arcsec diameter), with 3.4-pixel configured fiber blur |
| Fiber exposures | 180 s for the center and 600 s for each offset fiber |
| Spectrum | DESI `z` response; H-alpha window 851--855.81 nm at fixed \(z=0.3\); 0.08-nm output sampling; 61 stored samples per fiber |
| Internal wavelength grid | `resolution=500000` is the simulator's high-resolution cube setting; it must not be quoted as DESI's instrumental resolving power |
| Detector noise at generation | Disabled (`ADD_NOISE=False`); FITS output is explicitly written with `write_noise=False` |

The spatial model spans only \(64(0.11)=7.04\) arcsec, whereas the proposal
allows \(R_h=5\) arcsec and an exponential profile has nonzero support
beyond any finite stamp. The active aperture calculation sums the rendered
finite image without renormalizing flux that lies outside that grid. Large-size
proposal points are therefore a finite-stamp edge regime; this manuscript makes
no convergence claim for them. The grid follows
[`generate_fits.py`](../data_generate/generate_fits.py), while the aperture
sum is implemented by the external working-tree dependency
`/jet/home/xwang30/kl-tools/kl_tools/cube.py`.

The presence of a sky-spectrum path, CCD gain, and read-noise numbers in the
configuration does not imply that those stochastic processes enter the saved
data: noise is disabled at this stage. The generated data therefore do not
model sky-line residuals, Poisson counting statistics, read-noise covariance,
wavelength-calibration error, spectrophotometric calibration, cosmic rays,
fiber failures, image resampling covariance, or survey selection. The later
Gaussian noise model is described below. This boundary follows from the active
flags in [`generate_fits.py`](../data_generate/generate_fits.py), not from a
claim that omitted effects are negligible.

> **Suggested observation figure.** Show the noiseless `r`-band cutout beside
> the projected velocity field. Overlay the five 1.5-arcsec fiber apertures on
> both panels, with the center fiber and the two signed positions on each
> observed principal axis distinctly labeled. Beneath them, plot the five
> H-alpha windows in the fixed storage order using a shared wavelength axis.
> A caption should state explicitly that the different line centroids arise
> from the projected rotation field and that the plotted samples precede
> runtime noise.

### FITS production, validation, and LMDB assembly

One proposal row is rendered into one FITS file containing a primary header,
five spectral HDUs of shape `(61,)`, and one image HDU of shape `(48,48)`.
The header carries the observation-model version, target line and band, fiber
layout, exposure times, image scale and PSF, apparent magnitude, both nominal
S/N controls, and the central/total H-alpha flux audit trail. Each file is
written to a temporary path, fully reopened and checked, and then atomically
published. The validator requires six observation HDUs, the exact extension
shapes, and a 46,080-byte complete FITS file. These are integrity contracts,
not scientific properties, and are defined in
[`generation_integrity.py`](../data_generate/generation_integrity.py) and used
by [`generate_fits.py`](../data_generate/generate_fits.py).

The production launcher partitions a nominal one-million-row table into 2,000
objects per task, implying 500 parts. Its checked-in Slurm array is currently a
sparse resume list and sets `ALLOW_PARTIAL_ARRAY=1`; a fresh complete run must
submit all 500 task indices. Each row is rendered in an isolated Python
subprocess; the current launcher allows 12 hours because repeated interpreter
startup and shared-filesystem throughput are node dependent. Production
submissions cap each array at ten concurrent tasks. The wrapper can skip a
final-path file only after validating it against the corresponding proposal
row, so interrupted runs retain their 2,000-row part mapping and resume only
validator-reported incomplete parts. These operational facts are defined by
[`generate_simulator_v3.slurm`](../data_generate/generate_simulator_v3.slurm)
and [`generate_fits_wrapper.py`](../data_generate/generate_fits_wrapper.py).

The database stage preflights every expected FITS before opening a Pyxis/LMDB
writer. It stacks images as `(N,1,48,48)`, spectra as `(N,1,5,64)`, and fiber
positions as `(N,5,2)`. The 61 physical spectral samples occupy the beginning
of the length-64 axis and the remaining three entries are exact-zero padding.
The record also stores the ordered normalized targets, object identifier, and
all validated observation metadata. Production uses 500 isolated shards of
2,000 objects and merges them sequentially; after a successful merge the
temporary shard directories are deleted. This behavior is implemented by
[`make_database.py`](../data_generate/make_database.py) and its
[`shard`](../data_generate/make_database_simulator_v3.slurm) and
[`merge`](../data_generate/merge_database_simulator_v3.slurm) launchers.

The command-line defaults are not a self-contained provenance record. In
particular, `latin_hypercube.py` defaults to 100,000 rows while its default
output filename contains `1m`, and the current sample/dataset path defaults in
the generator, training configuration, and Slurm launchers are not all
identical. A production run must therefore supply and record the requested row
count and explicit sample/dataset names; filenames alone do not establish the
sample size or identity. This follows from the defaults in
[`latin_hypercube.py`](../data_generate/latin_hypercube.py),
[`config.py`](../arch/config.py), and the simulator-v3 Slurm launchers.

> **Suggested pipeline diagram.** Use a left-to-right flow with five boxes:
> scrambled LHS proposal table; `kl-tools` galaxy and velocity cube; atomic
> clean FITS files; sharded then merged LMDB; and epoch-specific noisy network
> tensors. Annotate the edges with the transformations that occur there:
> central-to-total H-alpha aperture calibration before cube rendering,
> nine-target normalization and 61-to-64 padding during LMDB construction,
> and S/N redraw plus Gaussian noise only after loading for training.

### Runtime observation model

Training loads the clean model image \(M_I\) and clean ordered five-fiber
spectra \(M_S\) into GPU memory. At the beginning of every training epoch, it
independently draws one effective image S/N per object from
\(\mathcal U(5,1000)\) and one effective central-fiber H-alpha S/N from
\(\mathcal U(1,200)\). The S/N columns originally stratified into the
proposal table therefore remain schema and provenance metadata, but they do
not set the noise in CCL or NPE training. The newly drawn values are supplied
both to the noise operator and to the network's metadata branch. Held-out
inference instead uses the S/N values stored with each record. These operations
are implemented in [`Trainer._prepare_epoch`](../arch/train.py).

This on-the-fly construction has two practical motivations. First, retaining
one clean realization avoids storing many noisy copies of each galaxy and
allows fresh noise levels and random deviates to be drawn across training
epochs. The resulting stochastic augmentation samples more of the adopted
noise distribution and reduces dependence on any one fixed realization.
Second, the clean norms are precomputed once, after which a vector of
object-specific RMS values can be broadcast across an entire GPU batch.
Galaxies with different requested noise levels can therefore be perturbed in
parallel without serial per-object rendering. Validation is intentionally
different: its epoch-zero S/N and noise streams are reused at every epoch so
that checkpoint comparisons are made against one fixed realization.

For the image, the repository defines the white-noise matched-filter norm by
flattening every pixel of the clean model image,

\[
A_I=\lVert M_I\rVert_2
=\left(\sum_p M_{I,p}^2\right)^{1/2},
\]

and sets

\[
\sigma_I=\frac{A_I}{\rho_I},
\qquad
M_I^\mathrm{noisy}=M_I+\sigma_I\,\epsilon_I,
\qquad
\epsilon_I\sim\mathcal N(0,\mathbf 1).
\]

For a known image template under independent, equal-variance Gaussian pixel
noise, the corresponding matched-filter S/N is
\(\sqrt{M_I^\mathsf{T}C^{-1}M_I}=A_I/\sigma_I\) when
\(C=\sigma_I^2\mathbf 1\). This motivates the norm as a controlled
detectability measure; matched filtering is standard for optimal astronomical
source detection under an adopted noise model
([Zackay and Ofek 2017](#zackay2017)). Here the exact noiseless galaxy is the
template, so the construction is an oracle matched-filter convention rather
than a catalog photometric S/N estimator. It includes all cutout pixels and
does not subtract a sky level, fit a PSF template, or use a variance map.

For the central fiber, let \(s_\ell\) be the clean spectral counts and let
\(\mathcal V=\{\ell:s_\ell\ne0\}\) exclude the exact-zero LMDB padding.
After sorting the \(N_\mathcal V\) valid values, the code takes the lower
median

\[
c_\mathrm{med}
=s_{(\lfloor(N_\mathcal V-1)/2\rfloor)},\qquad
A_{\mathrm{H}\alpha}
=\left[\sum_{\ell\in\mathcal V}
(s_\ell-c_\mathrm{med})^2\right]^{1/2}.
\]

The usual 61-sample spectrum has odd length, so this is its ordinary median;
the lower-median convention matters only if the valid count is even. A median
is a robust constant-location estimator ([Rousseeuw 2018](#rousseeuw2018)),
and over this narrow window it limits the influence of a line occupying a
minority of the samples. The implementation nevertheless includes H-alpha
samples in the order statistic and fits neither line-free sidebands nor a
wavelength-dependent stellar continuum. It is therefore a computational
baseline used to define the line norm, not a physical continuum measurement.

The central and offset-fiber RMS values are

\[
\sigma_\mathrm{cen}
=\frac{A_{\mathrm{H}\alpha}}{\rho_{\mathrm{H}\alpha}},
\qquad
\sigma_\mathrm{off}
=\sigma_\mathrm{cen}\sqrt{600/180},
\]

for the repository's integrated-count convention. Independent standard-normal
deviates are added with the appropriate per-fiber RMS only where the clean
spectrum is nonzero. A legitimate physical sample that is exactly zero is
consequently treated as padding. The formulas and mask are defined by
[`image_matched_filter_norm`, `central_halpha_line_norm`,
`apply_image_noise_for_snr`, and
`apply_central_halpha_snr_noise`](../arch/data.py).

The two quantities denoted S/N are intentionally simulation controls and
diagnostics of example difficulty. They are defined from unavailable clean
truth and need not reproduce the catalog-specific estimators used on real
images or spectra. This choice permits a simple, reproducible noise coordinate
that can be varied independently of \(m_{r,\mathrm{true}}\) and
\(F_{\mathrm{H}\alpha,\mathrm{cen}}\). It also means that the simulated
magnitude--noise and line-flux--noise relations are not survey selection
functions: a faint proposal object may be assigned high effective S/N and a
bright object low effective S/N.

The adopted likelihood is independent Gaussian pixel/bin noise with one RMS
per image and one exposure-scaled RMS per fiber. This is a useful controlled
augmentation but not the complete DECam or DESI detector likelihood. The DESI
pipeline instead propagates wavelength-dependent variance, masks, calibration,
sky subtraction, and a spectral resolution matrix
([Bolton and Schlegel 2010](#bolton2010);
[Guy et al. 2023](#guy2023)). The present simulator does not include source or
sky Poisson variance, read noise, sky-line residuals, wavelength-dependent
throughput uncertainty, bad pixels, or calibration error. Consequently, a
claim of robustness to real survey noise requires validation on more realistic
mocks or data even though exact agreement with a survey's reported S/N
diagnostic is not an objective here. Training S/N, noise, object order, and
rotation choices use separate deterministic streams; validation reuses its
epoch-zero streams as described above
([`derive_stream_seed` and `Trainer._generator`](../arch/train.py)).

## Neural Network Architecture

### Inputs, normalization, and rotational action

One object enters the feature extractor as a noisy single-channel
`48x48` model image \(M_I^\mathrm{noisy}\), an ordered tensor of five noisy
64-bin spectra \(M_S^\mathrm{noisy}\) with shape `1x5x64`, five
two-dimensional fiber centers, and three scalar observation descriptors

\[
\mathbf o=
(m_{r,\mathrm{true}},\rho_I,\rho_{\mathrm{H}\alpha}).
\]

The observation descriptors are linearly mapped to \([-1,1]\) with the
proposal bounds given above. They are simulator truths—hence the code's term
“oracle context”—and are not estimated from the noisy tensors. No inference
target is passed through this branch. These validation and scaling rules are
enforced by [`OracleContextNormalizer`](../arch/networks.py) and the exact
field list in [`config.py`](../arch/config.py).

Immediately before convolution, the feature extractor applies

\[
\widehat M_I
=\frac{M_I^\mathrm{noisy}}{\lVert M_I^\mathrm{noisy}\rVert_2},
\qquad
\widehat M_S
=\frac{M_S^\mathrm{noisy}}{\lVert M_S^\mathrm{noisy}\rVert_2},
\]

where the image norm spans both spatial axes and the spectral norm spans the
complete fiber--wavelength plane. The ten fiber-coordinate values are divided
by 1.5 arcsec. These operations are implemented in
[`SimpleFusionFeatureExtractor.forward`](../arch/networks.py).

Noise is applied before these normalizations, so the morphology, spectral
shape, relative amplitudes among fibers, and effective S/N remain in the
normalized tensors. A single common multiplicative amplitude for each modality
does not: neither CNN directly receives the absolute image count scale or the
absolute common scale of all five spectra. Absolute photometric information is
instead available explicitly through \(m_{r,\mathrm{true}}\); together with
the retained line-to-continuum and inter-fiber ratios, it can inform the
H-alpha-flux target. This separation makes the encoders invariant to a common
multiplicative calibration or exposure factor and bounds each modality's
overall scale before convolution, while the supplied S/N values tell the model which noise regime
generated each normalized tensor. Batch Normalization inside the CNNs acts on
learned activation channels and is distinct from these per-object input
normalizations.

The only geometric augmentation is a discrete 90-degree rotation. In the
repository's array convention, it maps

\[
M_I\mapsto R_{90}M_I,\quad
(x,y)\mapsto(y,-x),\quad
(g_1,g_2)\mapsto(-g_1,-g_2),\quad
\widetilde{\theta}_\mathrm{int}
\mapsto\operatorname{wrap}_{[-1,1)}(\widetilde{\theta}_\mathrm{int}-0.5),
\]

while copying the spectral array unchanged because its rows are stored in
galaxy-axis order. The shear sign reversal is the spin-2 transformation under
a 90-degree rotation (see the weak-lensing conventions reviewed by [Mandelbaum
2018](#mandelbaum2018)); the exact array, coordinate, and angle conventions are
defined by [`rotate_90_datavector`](../arch/data.py). Astronomy-specific use of
rotated galaxy views has precedent in [Dieleman et al. (2015)](#dieleman2015),
but this target-equivariant action is specific to the repository.

### Multimodal encoder

The encoder consists of three independent branches whose outputs are directly
concatenated into the condition vector

\[
\mathbf c=
[\mathbf c_I,\mathbf c_S,\mathbf c_\mathrm{meta}]
\in\mathbb R^{512+512+128}=\mathbb R^{1152}.
\]

There is no attention module or learned fusion layer after concatenation. The
topology below is defined by [`ImgCNN`, `JointSpecCNN`, `MetadataMLP`, and
`SimpleFusionFeatureExtractor`](../arch/networks.py). Residual blocks follow
the construction of [He et al. (2016)](#he2016), Batch Normalization follows
[Ioffe and Szegedy (2015)](#ioffe2015), Layer Normalization follows [Ba, Kiros,
and Hinton (2016)](#ba2016), and GELU follows [Hendrycks and Gimpel
(2016)](#hendrycks2016).

| Branch | Exact current topology | Output |
|---|---|---:|
| Image | Two `3x3` convolutions at 64 channels, each with BatchNorm and ReLU; `3x3` stride-2 max pool; four 128-channel residual blocks with downsampling in the fourth; five 256-channel residual blocks with downsampling in the fifth; five 512-channel residual blocks with downsampling in the fifth; final `3x3` average pool | 512 |
| Spectrum | Conv pairs at 16, 32, 64, and 128 channels, each convolution followed by BatchNorm and ReLU and each pair followed by wavelength-only `1x2` max pooling; two 256-channel convolutions; final `5x4` convolution spanning every fiber and remaining wavelength bin | 512 |
| Metadata | Concatenate ten scaled ordered fiber coordinates and three normalized contexts; `13 -> 64 -> 128`, GELU after both linear maps, then LayerNorm | 128 |

The image branch reduces the spatial grid from 48 to 24, 12, 6, and 3 pixels
before the final average pool. Each residual block contains two `3x3`
convolutions; a `1x1` projection and BatchNorm are used on the shortcut when
stride or channel count changes. The spectral branch pools only wavelength,
reducing 64 bins to 4 before the final `5x4` kernel. It therefore models the
five ordered fibers jointly and is not permutation invariant. These output
shapes are checked at runtime by [`networks.py`](../arch/networks.py).

> **Suggested architecture figure.** Draw three parallel inputs—a `48x48`
> image, a `5x64` ordered spectral plane, and 13 metadata scalars—feeding
> encoders labeled 512, 512, and 128. Concatenate them into condition vector \(\mathbf c\) and
> split the figure into two stages: above, the temporary 128-dimensional CCL
> projection head; below, the trainable LayerNorm and hybrid box/circle
> posterior flow. Mark the CNN backbone as trainable during CCL and frozen
> during NPE.

### Continuous-label contrastive pretraining

Pretraining appends a projection MLP
`1152 -> 2048 -> 512 -> 128`. The first two linear maps are followed by ReLU
and non-affine BatchNorm; the loss L2-normalizes the final embedding. Each
object contributes both its identity and R90-transformed observation and the
correspondingly transformed label. The augmentation is therefore
label-equivariant, not a demand that the two embeddings be identical.
Temperature-scaled normalized embeddings and augmented views descend from
contrastive methods such as SimCLR ([Chen et al. 2020](#chen2020)) and
supervised contrastive learning ([Khosla et al. 2020](#khosla2020)); the
continuous soft-label objective below is repository-specific, with related
kernel-based integration of side information discussed by [Dufumier et al. (2023)](#dufumier2023).
The implementation is in [`CCLPretrain` and
`ContinuousContrastiveLoss`](../arch/networks.py), and view construction is in
[`make_ccl_training_batch`](../arch/train.py).

For normalized parameters \(\widetilde{\bm\theta}_i\) and \(\widetilde{\bm\theta}_j\), the code
computes

\[
d_{ij}^2=\frac{1}{9}\sum_{k=1}^9\Delta_{ijk}^2,
\qquad
\Delta_{ij,\mathrm{int}}=
\frac{1}{\pi}\operatorname{atan2}
\{\sin[\pi(\widetilde\theta_{i,\mathrm{int}}-\widetilde\theta_{j,\mathrm{int}})],
  \cos[\pi(\widetilde\theta_{i,\mathrm{int}}-\widetilde\theta_{j,\mathrm{int}})]\},
\]

with ordinary differences for the other eight coordinates. All default label
scales are one. Candidate weights and the explicit background weight are

\[
w_{ij}=\exp[-d_{ij}^2/(2\sigma_{\widetilde\theta}^2)],\qquad
w_\mathrm{bg}=\exp[-d_\mathrm{cut}^2/(2\sigma_{\widetilde\theta}^2)],
\]

where \(\sigma_{\widetilde\theta}=0.15\) and \(d_\mathrm{cut}=0.40\). With anchor self-pairs
removed, the target coefficient is

\[
t_{ij}=\frac{w_{ij}}{\sum_{k\ne i}w_{ik}+w_{\rm bg}}.
\]

If \(\mathbf u\) denotes the normalized CCL projection and \(T=0.1\), the
optimized loss is

\[
\mathcal L_{\rm CCL}=-\frac{1}{N}\sum_i\sum_{j\ne i}t_{ij}
\log\frac{\exp(\mathbf u_i^\mathsf T\mathbf u_j/T)}
{\sum_{k\ne i}\exp(\mathbf u_i^\mathsf T\mathbf u_k/T)}.
\]

Because \(\sum_j t_{ij}<1\) when the background term is nonzero, this is not
the standard supervised-contrastive objective. The background mass and exact
normalization should not be omitted when reporting the method. Under DDP, the
projected vectors and labels are gathered with gradients across ranks, so the
operational four-GPU, 100-object-per-rank job produces 200 views per rank and
an 800-view global candidate set. Synchronized BatchNorm is enabled for CCL.
These details are defined in [`networks.py`](../arch/networks.py),
[`train.py`](../arch/train.py), and
[`pretrain_ccl.slurm`](../arch/pretrain_ccl.slurm).

### Bounded and circular neural posterior

The second stage is neural posterior estimation (NPE): simulator pairs are used
to minimize

\[
\mathcal L_{\rm NPE}=-\frac{1}{B}\sum_{n=1}^B
\log q_\psi(\widetilde{\bm\theta}_n\mid\mathbf c_n).
\]

Under the training proposal and simulator, this cross-entropy targets the
corresponding conditional posterior, not a proposal-independent likelihood;
this is the neural conditional-density argument of [Papamakarios and Murray
(2016)](#papamakarios2016) and the broader cosmological SBI setting discussed
by [Alsing et al. (2019)](#alsing2019). Consequently, the model's posterior
claim is conditional on this proposal, forward model, runtime noise process,
and oracle context.

The selected CCL checkpoint contributes only its 1,152-dimensional backbone.
Every backbone parameter is frozen, and the feature extractor is forced to
evaluation mode during NPE; the CNN is not jointly fine-tuned with the flow. A
trainable LayerNorm acts on the frozen representation. Flow density arithmetic
is evaluated with autocasting disabled, even when the CNN is evaluated under
mixed precision. These operations are implemented in
[`load_train_objs`, `NPETrainer`, and `KLNPE`](../arch/train.py).

The posterior respects the different supports of the targets through

\[
q_\psi(\widetilde{\bm\theta}\mid\mathbf c)=
q_{\psi,\mathrm{box}}(\widetilde{\bm\theta}_{-\mathrm{int}}\mid\mathbf c)\,
q_{\psi,\mathrm{circ}}(\widetilde\theta_\mathrm{int}\mid
\widetilde{\bm\theta}_{-\mathrm{int}},\mathbf c).
\]

The eight non-angular coordinates have hard support on \([-1,1]^8\). They are
first mapped pointwise to \([0,1]^8\), then passed through four conditional
masked autoregressive rational-quadratic spline transforms, each preceded by a
reverse permutation. Every transform uses eight bins, 256 hidden features, and
two autoregressive residual blocks, with a uniform unit-box base. Autoregressive
flows follow [Papamakarios, Pavlakou, and Murray (2017)](#papamakarios2017), and
rational-quadratic spline transforms follow [Durkan et al. (2019)](#durkan2019).
The implementation returns negative-infinite density outside the box rather
than silently clipping the inputs
([`BoundedHybridCircularFlow`](../arch/networks.py)).

The normalized intrinsic angle \(\widetilde\theta_\mathrm{int}=\theta_\mathrm{int}/\pi\) lives on the
half-open circle \([-1,1)\). Its default transform is one periodic
rational-quadratic spline layer with eight bins. The conditioner receives the
1,152 representation values plus the eight realized non-angular coordinates
and has topology `1160 -> 128 -> 128 -> 24`, with SiLU activations. The base is
uniform on a circumference-two circle, contributing \(-\log 2\) to the angle
log density. Left- and right-boundary derivatives are tied so that the seam has
one derivative. Treating angular variables on their circular rather than
Euclidean topology follows the construction discussed by [Rezende et al.
(2020)](#rezende2020); the exact kernel is in
[`circular_spline.py`](../arch/circular_spline.py).

Both box and circular spline conditioners begin at the identity map. Raw
spline logits are smoothly limited as \(10\tanh(r/10)\). The custom inverse
for the bounded autoregressive spline promotes common lower-precision inputs
to FP64 internally and casts the result back. These are numerical safeguards
defined by [`networks.py`](../arch/networks.py), not claims inherited from the
flow literature.

During NPE training, each object is assigned independently to the identity or
R90 view with probability one half. At posterior sampling, an even sample bank
is split equally between the identity observation and the rotated observation;
the rotated samples are inverse-aligned before concatenation, and density
evaluation uses the corresponding equal two-view mixture. This symmetrization
is implemented by [`make_npe_training_batch` and
`sample_density`](../arch/train.py); it should not be confused with a
mixture-density-network output layer.

> **Suggested posterior figure.** Depict eight bounded coordinates moving
> through four reverse-permutation/autoregressive-spline blocks inside a unit
> cube. Feed the realized eight-vector, together with the 1,152-dimensional
> observation condition, into a separate spline drawn as a closed ring for
> `theta_int`. Label the two arrows with the factorization
> \(q_{\rm box}q_{\rm circ}\), and show the identified circular endpoints.

### Optimization, validation, and checkpoint selection

Both stages use AdamW, the decoupled-weight-decay optimizer of [Loshchilov and
Hutter (2019)](#adamw), with linear warm-up followed by cosine annealing
([Goyal et al. 2017](#goyal2017); [Loshchilov and Hutter 2017](#sgdr)). The
operational settings below are the defaults of the checked-in Slurm launchers,
which materially override several Python dataclass defaults. They are sourced
from [`pretrain_ccl.slurm`](../arch/pretrain_ccl.slurm),
[`train_npe.slurm`](../arch/train_npe.slurm), and the scheduler/optimizer code
in [`train.py`](../arch/train.py).

| Setting | CCL pretraining | NPE training |
|---|---:|---:|
| Hardware launcher | 4 V100-32 GPUs, one node | 4 V100-32 GPUs, one node |
| Configured train / valid row limits | 1,000,000 / 100,000 separate LMDB rows | same |
| Maximum epochs | 100 | 200 |
| Local batch per rank | 100 objects; expanded to 200 views | 100 objects |
| Global batch/candidate set | 400 objects, 800 views | 400 objects |
| AdamW learning rate | \(10^{-3}\) | shared/box \(3\times10^{-4}\); circle \(10^{-4}\) |
| Weight decay | \(10^{-4}\) | \(10^{-5}\) |
| Warm-up | 5 epochs from 0.01 of base LR | 2 epochs from 0.1 of group LR |
| Cosine floor | \(10^{-6}\) | \(10^{-5}\) |
| Gradient-norm clip | 1 | 1 |
| Early stopping | none | patience 20, minimum improvement \(10^{-3}\) |
| Augmentation per row | identity and R90 both used | fair random choice of identity or R90 |
| Numerical modes | deterministic algorithms, fixed validation streams, FP16 AMP, channels-last tensors, `torch.compile` | same; flow density remains FP32 |
| Seed | 42 unless overridden | 42 unless overridden |

The row counts in this table are launcher requests for deterministic dataset
prefixes, not evidence that a complete million-object dataset or a finished
training run is versioned in this repository. The launchers reject missing
directories, but external storage was not used as evidence for this methods
description.

Automatic mixed precision follows the numerical strategy of [Micikevicius et
al. (2018)](#micikevicius2018); gradient clipping has the stability motivation
described by [Pascanu, Mikolov, and Bengio (2013)](#pascanu2013). Epoch counts,
batch sizes, learning rates, clip threshold, and early-stopping rule remain
repository choices, not values inferred from those papers.

Pyxis records are copied into preallocated GPU tensors rather than streamed by
a DataLoader. Each DDP rank receives a fixed contiguous shard, then permutes
its local order each epoch; dataset size must divide the world size. The number
of batches is the integer floor of local rows divided by local batch size, so a
remainder is not consumed. Independent random streams control order, image
noise, spectral noise, both S/N draws, and NPE view choice. The seed is

\[
s=(s_0+1{,}000{,}003r+10{,}007e+97k)\bmod(2^{63}-1),
\]

for rank \(r\), epoch \(e\), and stream identifier \(k\). Validation replaces
\(e\) by zero for all epochs, fixing its order, S/N, noise, and view choices.
This makes validation-loss comparisons deterministic conditional on one Monte
Carlo realization; it does not marginalize validation performance over the
noise distribution. The batching and seed behavior are defined in
[`Trainer`](../arch/train.py).

The scheduler advances after each validation pass. A numbered state dictionary
is written every epoch, and a separate `best` state dictionary is replaced
when validation loss improves by the configured threshold. `best.json` records
the selected epoch, train and validation loss, path, and subsequent learning
rates. Optimizer, scheduler, gradient scaler, RNG state, and within-epoch
position are not checkpointed, so a saved model is suitable for strict loading
but not exact interruption-and-resume training. The run also snapshots its
JSON model configuration plus `networks.py` and `circular_spline.py`; those
artifacts do not constitute a full environment lock or record the external
`kl-tools` working tree. These behaviors follow from
[`train.py`](../arch/train.py), [`train_model.py`](../arch/train_model.py), and
[`model_registry.py`](../arch/model_registry.py).

No performance, coverage, or real-survey validity claim follows from the
architecture or validation loss alone. A final assessment requires an
independent held-out dataset and simulation-based calibration or equivalent
coverage tests; this is standard for simulation-based inference ([Cook,
Gelman, and Rubin 2006](#cook2006); [Talts et al. 2018](#talts2018)). The
checked-in posterior-cache launcher explicitly requires a dataset not used for
training or checkpoint selection, but `train_model.py` itself performs no
final test.

### Change-control contract

This manuscript is intentionally coupled to the implementation and imported
notation files listed by
[`check_methods_manuscript.py`](check_methods_manuscript.py). The associated
test fails when any monitored source changes without a reviewed fingerprint
update. A relevant code or notation change must first be reflected in the
scientific prose, tables, equations, scope statements, and figure descriptions
that it affects; only then should the fingerprint be refreshed with
`python repo_reports/check_methods_manuscript.py --update --acknowledge-review`.
The fingerprint detects drift but cannot determine whether a prose edit is
scientifically adequate. Changes in the external `kl-tools` dependency also
require manual review because that repository lies outside the fingerprint's
scope.

## References

<a id="adamw"></a>**Loshchilov, I., & Hutter, F. (2019).** Decoupled weight
decay regularization. *ICLR*. [Paper](https://openreview.net/forum?id=Bkg6RiCqY7).

<a id="alsing2019"></a>**Alsing, J., Charnock, T., Feeney, S., & Wandelt, B.
(2019).** Fast likelihood-free cosmology with neural density estimators and
active learning. *MNRAS, 488*, 4440--4458.
[DOI](https://doi.org/10.1093/mnras/stz1960).

<a id="ba2016"></a>**Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** Layer
normalization. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).

<a id="bolton2010"></a>**Bolton, A. S., & Schlegel, D. J. (2010).**
Spectro-perfectionism: an algorithmic framework for photon noise-limited
extraction of optical fiber spectroscopy. *PASP, 122*, 248--257.
[DOI](https://doi.org/10.1086/651008).

<a id="chen2020"></a>**Chen, T., Kornblith, S., Norouzi, M., & Hinton, G.
(2020).** A simple framework for contrastive learning of visual
representations. *Proceedings of ICML, 119*, 1597--1607.
[Paper](https://proceedings.mlr.press/v119/chen20j.html).

<a id="cook2006"></a>**Cook, S. R., Gelman, A., & Rubin, D. B. (2006).**
Validation of software for Bayesian models using posterior quantiles.
*Journal of Computational and Graphical Statistics, 15*, 675--692.
[DOI](https://doi.org/10.1198/106186006X136976).

<a id="courteau1997"></a>**Courteau, S. (1997).** Optical rotation curves and
line widths for Tully--Fisher applications. *AJ, 114*, 2402.
[DOI](https://doi.org/10.1086/118656).

<a id="desi2022"></a>**DESI Collaboration (2022).** Overview of the
instrumentation for the Dark Energy Spectroscopic Instrument. *AJ, 164*, 207.
[DOI](https://doi.org/10.3847/1538-3881/ac882b).

<a id="dey2019"></a>**Dey, A., et al. (2019).** Overview of the DESI Legacy
Imaging Surveys. *AJ, 157*, 168.
[DOI](https://doi.org/10.3847/1538-3881/ab089d).

<a id="dieleman2015"></a>**Dieleman, S., Willett, K. W., & Dambre, J. (2015).**
Rotation-invariant convolutional neural networks for galaxy morphology
prediction. *MNRAS, 450*, 1441--1459.
[DOI](https://doi.org/10.1093/mnras/stv632).

<a id="dufumier2023"></a>**Dufumier, B., et al. (2023).** Integrating prior
knowledge in contrastive learning with kernel. *Proceedings of ICML, 202*,
8851--8878.
[Paper](https://proceedings.mlr.press/v202/dufumier23a.html).

<a id="durkan2019"></a>**Durkan, C., Bekasov, A., Murray, I., & Papamakarios,
G. (2019).** Neural spline flows. *NeurIPS, 32*.
[arXiv:1906.04032](https://arxiv.org/abs/1906.04032).

<a id="flaugher2015"></a>**Flaugher, B., et al. (2015).** The Dark Energy
Camera. *AJ, 150*, 150.
[DOI](https://doi.org/10.1088/0004-6256/150/5/150).

<a id="freeman1970"></a>**Freeman, K. C. (1970).** On the disks of spiral and
S0 galaxies. *ApJ, 160*, 811.
[DOI](https://doi.org/10.1086/150474).

<a id="goyal2017"></a>**Goyal, P., et al. (2017).** Accurate, large minibatch
SGD: training ImageNet in 1 hour.
[arXiv:1706.02677](https://arxiv.org/abs/1706.02677).

<a id="gwyn"></a>**Gwyn, S. D. J.** Extended Coleman, Wu & Weedman and Kinney
spectral templates. [Template documentation](https://www.astro.uvic.ca/~gwyn/pz/specc/index.html).

<a id="guy2023"></a>**Guy, J., et al. (2023).** The spectroscopic data
processing pipeline for the Dark Energy Spectroscopic Instrument. *AJ, 165*,
144. [DOI](https://doi.org/10.3847/1538-3881/acb212).

<a id="he2016"></a>**He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep
residual learning for image recognition. *CVPR*, 770--778.
[DOI](https://doi.org/10.1109/CVPR.2016.90).

<a id="hendrycks2016"></a>**Hendrycks, D., & Gimpel, K. (2016).** Gaussian
error linear units (GELUs).
[arXiv:1606.08415](https://arxiv.org/abs/1606.08415).

<a id="ioffe2015"></a>**Ioffe, S., & Szegedy, C. (2015).** Batch
normalization: accelerating deep network training by reducing internal
covariate shift. *Proceedings of ICML, 37*, 448--456.
[Paper](https://proceedings.mlr.press/v37/ioffe15.html).

<a id="khosla2020"></a>**Khosla, P., et al. (2020).** Supervised contrastive
learning. *NeurIPS, 33*.
[Paper](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html).

<a id="kinney1996"></a>**Kinney, A. L., Calzetti, D., Bohlin, R. C., McQuade,
K., Storchi-Bergmann, T., & Schmitt, H. R. (1996).** Template ultraviolet to
near-infrared spectra of star-forming galaxies and their application to
K-corrections. *ApJ, 467*, 38.
[DOI](https://doi.org/10.1086/177583).

<a id="mandelbaum2018"></a>**Mandelbaum, R. (2018).** Weak gravitational
lensing for precision cosmology. *Annual Review of Astronomy and Astrophysics,
56*, 393--433.
[DOI](https://doi.org/10.1146/annurev-astro-081817-051928).

<a id="mckay1979"></a>**McKay, M. D., Beckman, R. J., & Conover, W. J.
(1979).** A comparison of three methods for selecting values of input
variables in the analysis of output from a computer code. *Technometrics, 21*,
239--245. [DOI](https://doi.org/10.1080/00401706.1979.10489755).

<a id="micikevicius2018"></a>**Micikevicius, P., et al. (2018).** Mixed
precision training. *ICLR*.
[arXiv:1710.03740](https://arxiv.org/abs/1710.03740).

<a id="papamakarios2016"></a>**Papamakarios, G., & Murray, I. (2016).** Fast
epsilon-free inference of simulation models with Bayesian conditional density
estimation. *NeurIPS, 29*.
[arXiv:1605.06376](https://arxiv.org/abs/1605.06376).

<a id="papamakarios2017"></a>**Papamakarios, G., Pavlakou, T., & Murray, I.
(2017).** Masked autoregressive flow for density estimation. *NeurIPS, 30*.
[Paper](https://papers.nips.cc/paper_files/paper/2017/hash/6c1da886822c67822bcf3679d04369fa-Abstract.html).

<a id="pascanu2013"></a>**Pascanu, R., Mikolov, T., & Bengio, Y. (2013).** On
the difficulty of training recurrent neural networks. *Proceedings of ICML,
28*, 1310--1318. [Paper](https://proceedings.mlr.press/v28/pascanu13.html).

<a id="pranjal2023"></a>**Pranjal, R. S., et al. (2023).** Kinematic lensing
inference I: characterizing shape noise with simulated analyses. *MNRAS, 524*,
3324--3334. [DOI](https://doi.org/10.1093/mnras/stad2014).

<a id="rezende2020"></a>**Rezende, D. J., Papamakarios, G., Racaniere, S.,
Albergo, M., Kanwar, G., Shanahan, P., & Cranmer, K. (2020).** Normalizing
flows on tori and spheres. *Proceedings of ICML, 119*, 8083--8092.
[Paper](https://proceedings.mlr.press/v119/rezende20a.html).

<a id="rousseeuw2018"></a>**Rousseeuw, P. J. (2018).** Anomaly detection by
robust statistics. *WIREs Data Mining and Knowledge Discovery, 8*, e1236.
[DOI](https://doi.org/10.1002/widm.1236).

<a id="rowe2015"></a>**Rowe, B. T. P., et al. (2015).** GalSim: the modular
galaxy image simulation toolkit. *Astronomy and Computing, 10*, 121--150.
[DOI](https://doi.org/10.1016/j.ascom.2015.02.002).

<a id="sgdr"></a>**Loshchilov, I., & Hutter, F. (2017).** SGDR: stochastic
gradient descent with warm restarts. *ICLR*.
[arXiv:1608.03983](https://arxiv.org/abs/1608.03983).

<a id="talts2018"></a>**Talts, S., Betancourt, M., Simpson, D., Vehtari, A., &
Gelman, A. (2018).** Validating Bayesian inference algorithms with
simulation-based calibration.
[arXiv:1804.06788](https://arxiv.org/abs/1804.06788).

<a id="xu2024"></a>**Xu, X., et al. (2024).** Kinematic lensing with DESI:
probing structure formation at very low redshift.
[arXiv:2407.20867](https://arxiv.org/abs/2407.20867).

<a id="zackay2017"></a>**Zackay, B., & Ofek, E. O. (2017).** How to coadd
images? I. Optimal source detection and photometry using ensembles of images.
*ApJ, 836*, 187. [DOI](https://doi.org/10.3847/1538-4357/836/2/187).
