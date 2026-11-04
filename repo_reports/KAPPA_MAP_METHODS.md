# How we make κ maps for the cluster test: standard Kaiser–Squires vs. our likelihood-based map

Written 2026-10-23 for the cluster substructure test (`arch/diagnostics/paper_tests/h_cluster_analysis.py`).

> **Update 2026-10-24:** several details here are out of date. The tuned settings are σ_κ = 0.03, ℓ = 1 px,
> 0.5 Mpc (not σ_κ = 0.1). The residual is now the difference map(main + sub) − map(main only), not a Nautilus
> halo fit. The S/N maps use a Gauss–Newton noise covariance. Sec. 5's results are round 1. For the current,
> detailed explanation see `KAPPA_FITTING_EXPLAINED.md`; this note stays useful for the KS comparison (Secs. 2,
> 4) and earlier work using similar methods (Sec. 6).

## 1. What κ and γ are, and why they're linked

Everything in the map comes from one function, the lensing potential ψ(**θ**): a 2D, projected version of the
cluster's gravitational potential. Its second derivatives give the two quantities we care about:

- **Convergence**: κ = ½ (ψ₁₁ + ψ₂₂). This is the projected mass density in units of Σ_crit, the
  quantity we want to map.
- **Shear**: γ₁ = ½ (ψ₁₁ − ψ₂₂) and γ₂ = ψ₁₂. This is what stretches the background galaxies.

Here ψ₁₂ means ∂²ψ/∂θ₁∂θ₂, and so on (Bartelmann & Schneider 2001, Sec. 3).

Because both are built from the same ψ, **the shear map is completely fixed by the κ map**, and vice versa
(up to one constant, see Sec. 4). The link is simplest in Fourier space. A derivative ∂/∂θ becomes a
factor i k, so each Fourier mode of κ produces a shear mode

  γ̂(**k**) = D(**k**) κ̂(**k**),  with  D(**k**) = (k₁² − k₂² + 2i k₁k₂) / |**k**|²

(Kaiser & Squires 1993). D has magnitude 1. It doesn't change the size of a mode; it only rotates the shear
pattern according to the direction of **k**. We call D the **KS kernel**. In the code, D₁ and D₂ are its
real and imaginary parts (`ks_kernel`).

One more detail: galaxies actually respond to the **reduced shear** g = γ / (1 − κ). In most of the field κ is
small, so g ≈ γ, but near the cluster centre the two differ (κ ≈ 0.2 at 50 kpc for our cluster).

## 2. The standard method: Kaiser–Squires (KS)

The relation γ̂ = D κ̂ can be turned around, because D has magnitude 1:

  κ̂(**k**) = D*(**k**) γ̂(**k**)

(Kaiser & Squires 1993). So the recipe is:

1. **Bin.** Put the galaxies into cells on a regular grid, and estimate one shear per cell. In our code
   these are the stacked shears of the Cartesian cells.
2. **Fourier transform** the shear map, **multiply by D***, and **transform back**. That gives κ.
3. **Smooth** the result. Each galaxy's intrinsic shape adds noise to its shear estimate, and the
   inversion passes that noise straight into κ, so the raw map is very noisy (Kaiser & Squires 1993).

KS is fast, has no free parameters apart from the smoothing, and is still widely used. It has known
limitations, several of which bite at our source density:

- **It needs a full grid of shears.** Empty cells and the field edge have to be filled somehow. We set them
  to zero and zero-pad the grid, which creates edge artefacts.
- **It treats every cell the same.** A cell with 2 galaxies counts as much as a cell with 10, and the
  per-cell uncertainties aren't used. KS ignores both noise and masks (Jeffrey et al. 2018).
- **It assumes g ≈ γ.** Iterative versions fix this for reduced shear (Seitz & Schneider 1995), but plain
  KS does not.
- **It works with binned averages.** It never sees each galaxy's full likelihood: KL constrains one shear
  direction much better than the other, and the posteriors are non-Gaussian and bounded by the prior box.
  At ~5 galaxies per cell, the cells are also large (~1 Mpc at sample-3 density), so the map is coarse.

## 3. Our method: fit the κ map to the individual galaxies

Instead of going from shear to κ, we go **from κ to shear and compare with the data**. We look for the κ map
whose predicted shears best match what each galaxy's NPE likelihood says.

1. **Unknowns.** κ values on a pixel grid. The current setting is 12 × 12 pixels of 0.5 Mpc, covering the
   field.
2. **Predict the shear for a trial κ map**, using the same KS kernel in the *forward* direction:
   Fourier transform κ, multiply by D, transform back. This gives γ on the grid (again zero-padded, so κ = 0
   outside the field).
3. **Read off the prediction at every galaxy's exact position.** We interpolate γ and κ to each galaxy and
   form its reduced shear gᵢ = γᵢ / (1 − κᵢ). There is no binning, so every galaxy sits at its own position.
4. **Score the trial map** with the galaxies' own likelihoods: Σᵢ log Lᵢ(gᵢ). Lᵢ is the same KDE likelihood
   from the NPE posterior that we use for stacking (posterior divided by the training prior).
   - A galaxy with a tight likelihood pulls hard on the map and a vague one barely matters, so weighting is
     automatic.
   - Direction-dependent precision is handled automatically too.
5. **Add a prior that keeps the map sensible.** With ~140 galaxies and 144 pixels, the data alone can't pin
   down every pixel. We add a Gaussian prior that penalises:
   - large κ values (amplitude scale σ_κ = 0.1);
   - large differences between neighbouring pixels (a smoothness term).

   This replaces the smoothing step of KS. It is the same idea as a Wiener filter: a Gaussian prior on the
   field, used as a Bayesian map estimate (Jeffrey et al. 2018).
6. **Find the best map.** We maximise score + prior with a gradient optimiser (L-BFGS in PyTorch), which
   gives the most probable (MAP) κ map.

Earlier "direct" reconstructions work the same way: write down the mass map, predict the lensing data from
it, and regularise with a prior. Bridle et al. (1998) used a maximum-entropy prior. Marshall et al. (2002)
used each galaxy's shape as an independent estimate of the reduced shear, with a built-in smoothness.
Lanusse et al. (2016) used sparsity priors. What's new here is only the data term: each galaxy enters through
its **NPE likelihood**, the same object we stack, in the spirit of Bernstein & Armstrong (2014).

**The residual map** is the likelihood κ map of the with-subhalo data minus the κ of the best-fitting
main-halo-only model, from the Nautilus halo fit. A subhalo should show up as a positive blob at its position.

## 4. Side by side

| | Standard KS | Our likelihood-based map |
|---|---|---|
| Direction | shear → κ (inverse kernel D*) | κ → shear (forward kernel D), then compare with data |
| Input | one shear per cell (binned) | every galaxy's full NPE likelihood at its own position |
| Weighting | none: all cells equal | automatic: tight likelihoods count more, direction by direction |
| Reduced shear | assumes g ≈ γ | exact, g = γ/(1 − κ) |
| Empty regions | must be filled (zeros) | allowed; the prior fills them in smoothly |
| Noise control | smoothing after the fact | a prior inside the fit (Wiener-filter-like) |
| Free choices | smoothing scale | pixel size, σ_κ, smoothness strength |
| Output | a map, no uncertainty | the most probable map (uncertainty needs an extra step, e.g. a Laplace approximation) |
| Cost | one pair of FFTs | an optimisation, ~seconds at our size |

**Limitations both share:**

- **Mass-sheet degeneracy.** Adding a constant to κ (more precisely, κ → λκ + (1 − λ)) leaves the reduced
  shear unchanged, so the overall level of the map is not fixed by shear alone (Bartelmann & Schneider
  2001). KS sets the k = 0 mode to zero; our zero-padding and the amplitude prior do the same job.
- **Edge effects.** Both assume nothing outside the field.

**Limitation specific to our method:** the prior shrinks the map toward zero, just as a Wiener filter does,
so peak heights come out lower than the truth. That's why we compare the reconstruction to the truth through
the same pipeline on synthetic data, rather than reading κ values off the map at face value.

## 5. What we've seen so far (synthetic data, sample-3 density, one realisation)

- Both maps recover the main halo.
- Neither clearly shows the 30% subhalo: its signal is about as large as the pixel-to-pixel noise, as the
  S/N forecast predicted (best case ≈ 2 at this density).
- The likelihood map is noisy at 0.5 Mpc pixels with the current prior. Next is to try coarser pixels or a
  stronger smoothness prior, choosing the settings on synthetic data before looking at the NPE run.

## 6. Previous applications of similar methods

Building a mass map, predicting the lensing signal at each galaxy, and fitting it with a prior is an
established approach, especially for clusters. What changes from method to method is the data term (how each
galaxy enters) and the prior. Our map uses a standard reconstruction scheme; what's new is the data term.

**Closest to ours**

- **Bridle et al. (1998); Marshall et al. (2002)** ("LensEnt2").
  - They reconstruct a cluster's projected mass directly and regularise it with a maximum-entropy prior.
  - Marshall et al. treat each background galaxy's shape as an independent estimate of the reduced shear
    at its position, and build an intrinsic smoothness into the reconstruction.
  - This is essentially our method with a Gaussian per-galaxy ellipticity likelihood in place of our NPE
    likelihood, and an entropic prior in place of our Gaussian smoothness prior.
- **Schneider et al. (2017),** probabilistic mass mapping.
  - They infer shear and κ fields from galaxy ellipticity catalogs, under a spatial-process prior on the
    lensing potential.
  - In its general form the method makes no weak-shear or Gaussian-noise assumption.
  - They demonstrate it on simulated Gaussian shear fields and on the merging cluster Abell 781.
- **Schneider et al. (2015; arXiv:1411.2608),** the companion hierarchical framework.
  - It works from per-galaxy *interim posteriors*, reweighted by the interim prior they were drawn under.
  - That is the same step as our Lᵢ = qᵢ / π_train (NPE posterior divided by the training prior), so it's
    the closest precedent for using per-galaxy posteriors as likelihoods inside a larger model.

**Cluster methods with the same forward-fitting idea**

- **Bradač et al. (2005),** "Strong and weak lensing united".
  - It combines weak lensing, extended into the inner, non-weak parts of the cluster, with strong-lensing
    multiple images.
  - Combining the two can break the mass-sheet degeneracy.
  - To check before citing it for specifics: it is commonly described as working with the lensing
    potential on a grid. I haven't confirmed how it uses individual galaxy ellipticities.

**Bayesian mass mapping with binned shear (less close)**

These are fully Bayesian maps with physically motivated priors, but the data enter as pixelised, binned
shears with Gaussian noise rather than per-galaxy likelihoods.

- **Jeffrey et al. (2018).** Wiener filter (Gaussian prior) and GLIMPSE (sparsity prior), compared with KS
  on DES SV.
- **Fiedorowicz et al. (2022),** KaRMMa. A lognormal prior on κ; it samples the posterior distribution of
  maps.
- **Porqueres et al. (2021),** BORG-WL. Field-level forward modelling: a physical structure-formation model
  links the matter field to the shear data.

**What's new in ours:** each galaxy enters through a learned, non-Gaussian KL likelihood, which carries the
direction-dependent precision of the kinematics, the TF prior and the bounded prior box. For the paper we can
cite Marshall et al. (2002) and Schneider et al. (2017) for the reconstruction scheme, and present the NPE
likelihood as the plug-in.

## References

- Bartelmann, M. & Schneider, P. 2001, *Weak gravitational lensing*, Phys. Rep. 340, 291. arXiv:astro-ph/9912508.
- Bernstein, G. M. & Armstrong, R. 2014, *Bayesian lensing shear measurement*, MNRAS 438, 1880 (`BA14` in main.bib).
- Bradač, M., Schneider, P., Lombardi, M. & Erben, T. 2005, *Strong and weak lensing united. I. The combined
  strong and weak lensing cluster mass reconstruction method*, A&A 437, 39. arXiv:astro-ph/0410643.
- Bridle, S. L., Hobson, M. P., Lasenby, A. N. & Saunders, R. 1998, *A maximum-entropy method for
  reconstructing the projected mass distribution of gravitational lenses*, MNRAS. arXiv:astro-ph/9802159.
- Fiedorowicz, P. et al. 2022, *KaRMMa – Kappa reconstruction for mass mapping*, MNRAS 512, 73. arXiv:2105.14699.
- Jeffrey, N. et al. 2018, *Improving weak lensing mass map reconstructions using Gaussian and sparsity priors:
  application to DES SV*, MNRAS 479, 2871. arXiv:1801.08945.
- Kaiser, N. & Squires, G. 1993, *Mapping the dark matter with weak gravitational lensing*, ApJ 404, 441.
  doi:10.1086/172297.
- Lanusse, F., Starck, J.-L., Leonard, A. & Pires, S. 2016, *High resolution weak lensing mass mapping combining
  shear and flexion* (GLIMPSE), A&A 591, A2. arXiv:1603.01599.
- Marshall, P. J., Hobson, M. P., Gull, S. F. & Bridle, S. L. 2002, *Maximum-entropy weak lens reconstruction:
  improved methods and application to data*, MNRAS 335, 1037. arXiv:astro-ph/0112396.
- Porqueres, N., Heavens, A., Mortlock, D. & Lavaux, G. 2021, *Bayesian forward modelling of cosmic shear data*,
  MNRAS. arXiv:2011.07722.
- Schneider, M. D., Hogg, D. W., Marshall, P. J., Dawson, W. A., Meyers, J., Bard, D. J. & Lang, D. 2015,
  *Hierarchical probabilistic inference of cosmic shear*. arXiv:1411.2608.
- Schneider, M. D., Ng, K. Y., Dawson, W. A., Marshall, P. J., Meyers, J. E. & Bard, D. J. 2017, *Probabilistic
  cosmological mass mapping from weak lensing shear*, ApJ. arXiv:1610.06673.
- Seitz, C. & Schneider, P. 1995, *Steps towards nonlinear cluster inversion through gravitational distortions.
  II. Generalization of the Kaiser & Squires method*, A&A 297, 287. arXiv:astro-ph/9408050.
