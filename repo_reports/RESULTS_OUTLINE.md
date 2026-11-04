# KL-NN paper: Results / post-processing outline

Drafted 2026-09-29, revised the same day after Eason's feedback. It lists
figures and tables, what each claims, the data behind it, and fast tests to
run before writing. Figures are generated first, then text is written.

## REVISED STRUCTURE after Tim's feedback (2026-10-15) — supersedes §1, §8 where they differ

**Message.** Likelihood stacking is THE route from NPE posteriors to shear. KL's strength is
not only better per-galaxy precision for standard cluster lensing, but also *localized*
shear: substructure where the shear departs from the main-halo profile (cf. KL inference II,
cluster lensing with O(1) galaxies). The NPE cannot do single-galaxy shear (posterior means
are biased; stacking needs N ≳ 4 for percent-level m), but a localized stack — a sector of an
annulus — can still reveal substructure, at lower angular resolution.

**Sec. 5 Validating the NPE**
- 5.1 Training and example posteriors.
  - F1 loss curve (unchanged).
  - **F2 replaced (Tim):** heatmap of the shear posterior width
    σ_g = √(σ_g1² + σ_g2²) over image S/N (ρ_I) × Hα S/N (ρ_S). A single galaxy can't
    support the general claim "Hα S/N matters more than image S/N"; a population map can.
    Design to decide (see open questions). The single-galaxy panels and the 9D corner go to
    the appendix or are dropped.
  - Speed sentence; MCMC comparison (7-galaxy package with Lauren).
- 5.2 Posterior calibration: P–P (Eason rewrites in his own words) + coverage table.

**Sec. 6 From posteriors to shear: likelihood stacking**
- 6.1 Raw per-galaxy posterior means (F4) — kept as motivation: m ≈ −0.5 for point estimates.
- 6.2 Why posterior means shrink (F5, Gaussian toy) — kept; explains why we stack likelihoods.
- ~~6.3 Weighting and posterior-width cuts~~ — **removed** (and tab:selection).
- ~~6.4 Response calibration~~ — **removed** (and F7).
- 6.3 Likelihood stacking at constant shear (T8, F8): shows the method works; m vs N.
- 6.4 Cluster lensing application (NEW design, realistic):
  - Realistic source density and radial distribution (∝ annulus area), one cluster →
    a stack of N_cl clusters; m, c and errors vs N_cl.
  - Localized (sector) stacking to recover a substructure shear offset from the main-halo
    prediction, with the resolution set by the ≳4 galaxies a sector needs.
- Tab. 2 (estimator summary) shrinks to raw vs stacked, or is dropped.

**Sec. 7 Discussion**: KL + NPE for cluster substructure; per-galaxy limit (N ≳ 4);
limitations (single z — redshift appendix; Hα only; Gaussian noise; true-valued metadata;
TF known; magnification ignored); future work (NRE, non-constant-shear model fitting).

**Appendix**: redshift scan; 9D corner / nuisance P–P; (optionally) the removed
selection/response-calibration material if we want a record of it.

**Annulus test as run so far is NOT realistic (checked 2026-10-15):** galaxies were assigned
to annuli by |g|, ~6k per inner annulus → implied density ~2000 /arcmin² at R = 0.1–0.15 Mpc
falling to 0.5 /arcmin² at 2–3 Mpc (z_l = 0.1; 32k galaxies over 0.6 deg²). Equivalent to a
stack of hundreds of clusters with the inner annuli overweighted. Must be redesigned.

**Open questions to settle with Eason (2026-10-15):**
1. F2 heatmap: controlled grid (fixed galaxies, ρ_I × ρ_S overridden, new GPU runs) vs
   empirical binning of the Xu3 cache (free, but confounded by cos i, size and the
   catalog's ρ_I–ρ_S correlation, and sparse in the corners).
2. Removed sections: drop entirely, or keep a short appendix?
3. Cluster test: the DESI-like KL source density to assume; lens mass/redshift; how many
   clusters to stack; reuse Xu3 galaxies (shear direction set by g_i, as now) or simulate a
   dedicated cluster set with true 2D positions and per-position shear.
4. Substructure demo: subhalo mass/position, sector size, what we claim (detection vs
   profile recovery), single cluster vs stacked.

## Cluster test (Sec. 6.4) — design notes (2026-10-22; Tae's answers pending)

**Plan so far (Eason/Rachel):**
- Shear field from GalSim `NFWHalo` (Rachel). Two scenarios with the SAME background galaxies at
  z_s = 0.3 (same positions, nuisances, noise seeds): main NFW halo only, and main + subhalo.
  Superpose halos in γ and κ (`getShear(reduced=False)`, `getConvergence`), then
  g = γ_tot/(1 − κ_tot); never add reduced shears. No deflection/magnification (GalSim).
- 2D map in the spirit of Murray, Kilbinger & Payerne 2026 (arXiv:2609.24948): they use
  15 log annuli (1′–32′) × 10 sectors, tangential + cross, inner cut 1′, at 10–100 /arcmin².
  We need far fewer cells (≥4 galaxies per cell).
- Goal is qualitative: show the subhalo is visible by eye at lower resolution than KL-II
  (individual galaxies), not a formal detection or mass estimate.
- Presentation (Rachel): κ maps side by side, 2–3 panels (main only, with subhalo, and maybe the
  residual). Caution: at KL source density the reconstructed κ will need heavy smoothing.
- Outputs: side-by-side maps; per-cell pulls (ĝ − g_ref)/σ; overall m, c across cells.

**Cautions to design around:**
- Source density: Eason suggested DES (n_eff 5.6 /arcmin² Y3, 8.2 Y6), but those are photometric
  WL sources. KL needs DESI fiber + Hα + disk + R_h ≥ 1″ → ~0.1–0.2 /arcmin² (confirm from Xu+24 /
  KL-DESI). Proposed: KL density fiducial, DES density as an optimistic case. At z_l ≈ 0.1 a
  single cluster has only ~100–200 KL sources within 2 Mpc.
- Training prior |g_k| < 0.1: choose M, z_l and the inner radius so no galaxy exceeds it.
- Cell truth: the likelihood product is ≈ a precision-MATRIX-weighted mean,
  (Σ F_i)⁻¹ Σ F_i g_i, with F_i from each galaxy's NPE likelihood covariance (KL constrains one
  shear direction much better than the other). Compute in the same frame as the stack; validate
  with synthetic Gaussian likelihoods first.
- Frame: rotate each galaxy's likelihood to (g_t, g_×) about the main-halo centre for polar
  cells; the subhalo shows as excess g_t and non-zero g_×.
- Many independent realizations (positions/nuisances/noise) for the bias statistics; paired
  difference maps for the visual, single-map uncertainties for any significance statement.
- Cubic response (checked 2026-10-22 on existing stacks): constant-shear T8 (1M, N = 32)
  b₃ = −2.8 ± 1.9 (g1), 0.0 ± 1.8 (g2), not significant; adding b₃ absorbs the −1% linear g1
  residual (m → +0.005 ± 0.011). Annulus stacks (100k models, g_t 0.008–0.061): b₃ ≈ −6 to −10
  (±4–5), i.e. ~−1% response at g = 0.03 and ~−4% at 0.06; linear-only χ² 7.5–12.7 / 6. My
  weighted refit gives linear m ≈ −0.03 vs the summary's −0.011 ± 0.011 (bootstrap fit): reconcile
  when redoing. → Report per-cell bias vs |g|, include b₃ in the overall fit, keep inner cells at
  |g| ≲ 0.05 or show them separately.

**Option under consideration (Eason: do it if it moves us toward BA14-style non-constant shear
and isn't too hard; he wants to think more):** fit a parametric shear-field model directly to
the galaxies instead of comparing binned cells to a "cell truth". With g(x; θ) for NFW (M, c,
centre) [+ subhalo (M_sub, position)], L(θ) = Π_i L_i(g(x_i; θ)), L_i = each galaxy's KDE
likelihood (the one already used for stacking); sample θ (few parameters). Handles within-cell
gradients and matrix weights exactly. Cheap version for the visual: fit the main halo only, then
show binned/κ residuals relative to it.

**Verified parameters (2026-10-22; Eason: one ideal case, subhalo = 30% of host mass, ≥4 galaxies per
cell, model fitting to individual galaxies, try both polar and Cartesian cells):**
- Tae (via Eason): skeptical of detectability; 2′ cells ≈ 300 kpc; suggests κ reconstruction and a lens
  at half the source distance.
- Lens z_l = 0.13: D_l D_ls/D_s peaks at 0.134 for z_s = 0.3 (Planck18); z_l = 0.10 already gives 94%.
  1′ = 144 kpc; the R < 3 Mpc field is 1369 arcmin².
- Source density, no tomography (all sources behind the lens, simulated at z = 0.3), from the
  catalogs in $S/desi_cut_catalogs/:
  - Xu sample 3: 0.143 /arcmin² (Xu ref 0.18); 75.5% behind z_l = 0.13 → 0.11–0.14 /arcmin²
    → 150–190 galaxies, ≤ 37–47 cells of 0.8–0.9 Mpc.
  - Xu sample 1: 0.515 /arcmin² (Xu ref 0.527); 87.9% behind (2M-row subsample) → 0.45 /arcmin²
    → ~620 galaxies, ≤ ~155 cells of ~0.43 Mpc. Caveat: sample-1 galaxies mostly lack DESI Hα, so
    use the sample-1 density with the Xu3 nuisance population ("future survey targeting all
    eligible disks").
- GalSim 2.7.2 NFWHalo: mass = M200c in M☉/h, own Cosmology (Ωm, ΩΛ; set Ωm = Planck18 0.311),
  positions in arcsec, no c–M relation. Concentration from Duffy+08 (200c, full sample: A = 5.71,
  B = −0.084, C = −0.47, M_pivot = 2e12 M☉/h): c = 3.31 (1e15 M☉), 3.66 (3e14 M☉) at z_l = 0.13.
- Shear (M = 1e15, z_l = 0.13, z_s = 0.3): |g| = 0.063 at 50 kpc, 0.033 at 0.5 Mpc, 0.007 at 3 Mpc;
  κ = 0.22 at 50 kpc. Nothing exceeds |g_k| = 0.1 in the field, so the training prior is safe.
- Subhalo signal |Δg|: median 0.032 / 0.023 / 0.014 within 0.25 / 0.5 / 1 Mpc of the subhalo
  (almost the same for a subhalo at 0.5 or 1 Mpc).
- Matched-filter (best-case) subhalo S/N with σ_g ≈ 0.043 per component: ≈ 2 at sample-3 density,
  ≈ 4.3 at sample-1 density. A by-eye map will look weaker.
- Still to decide (Eason): which density is the main case; which model parameters to free and their
  priors; how to handle cells with < 4 galaxies (equal-count bins / merge / drop). Waiting on Tae:
  subhalo realism (truncation), κ-reconstruction method.

**Decisions 2026-10-23 (Eason):** residual κ map = reconstruction minus the TRUE main-halo κ (as if
known); no halo fit needed for the paper. The Nautilus halo fit stays in `h_cluster_analysis.py`
(`halo_fit`) for later if the advisor wants it. Polar cells are equal-area by construction (~1 Mpc² each:
expected count = density × area ≈ 5); Murray+26 use log annuli × fixed 10 sectors (area ∝ r²), which is not
workable at KL density. Synthetic validation (30 realizations) passed: pull mean 0.00–0.03, spread
0.93–0.94; 1–2 galaxy cells shrink by ~0.005. Report: https://claude.ai/artifact/CVoDEcr5VXciuNfCVfkC82
(copy at $S/reports/paper-tests-1m/H_cluster/s3_r0/report.html). Known issue: subhalo at 0.5 Mpc sits inside
the 0.57 Mpc central polar cell → move it to ~1 Mpc.

**Questions sent to Tae (via Eason, 2026-10-22; answers pending):**
1. Realistic yet visibly detectable subhalo: mass ratio, projected distance, concentration/truncation.
2. How to define a cell's "true" shear when the shear varies inside it.
3. Tangential/cross on polar cells vs Cartesian (g1, g2) cells; is g_× a good visual indicator.
4. Usual inner radius cut; which effects change the shear map itself (miscentering,
   magnification/source density, line-of-sight structure).
5. Realistic lens redshift and mass for sources at z ≈ 0.3.
6. Clearest presentation of a sparse 2D shear map (whiskers / aperture mass / κ map).

## 0. Scope for this draft (Eason, 2026-09-29)

- **Model:** the current 100k-trained, CCL-pretrained frozen-encoder model
  `CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair`.
- **Test set:** Xu3 only.
- **No new runs:** no 1M model, no Xu5 inference. An `\XW{}` note in
  `results.tex` says so.
- **Existing cache (no new inference needed):**
  `$S/cache/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair/test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded`
  (12 GB; 10^4 identity+R90 samples per galaxy, TF log weights, ESS).
- **Shape noise:** follows Xu+24, σ_ε = √(σ_g1² + σ_g2²) from the 1D
  marginal posterior widths of g1 and g2. Decide whether the width is the
  standard deviation or the 16–84% half-width.
- **Posterior-width cut:** on the same per-galaxy quantity,
  σ_ε,i = √(σ_g1,i² + σ_g2,i²), i.e. the individual galaxy's shape noise.
- **Story:** show the straight-from-inference results first. That means
  equal weights, the full sample, and no weighting, cuts, calibration or
  stacking. Then explain why they are bad, then show the mitigations.

## 1. Content allocation (revised after Eason's feedback, 2026-09-29)

Sec. 5 is about validating the NPE itself. The main results (shear) are in
Sec. 6. Sec. 5 is kept compact.

**Sec. 5 "Validating the Neural Posterior Estimator"**
- **5.1 Training and example posteriors.**
  - R1: loss curve. Train/valid NLL; best epoch 200; train −18.34, valid
    −17.80; `plot_losses.py`.
  - R2: (a) 9D corner plot of one Xu3 galaxy; (b) g1–g2 contours at two
    **image SNRs** (ρ_I = 50 vs 200, ρ_S fixed); (c) NPE vs MCMC for the 4
    handoff galaxies, a placeholder until the chains arrive.
  - **Speed: a brief mention in the text, no figure.** NPE ~0.33 s per galaxy
    for 10^4 samples plus TF weights on one V100, vs the MCMC time (pending
    from Jiachuan/Lauren; kl-pipe question).
- **5.2 Posterior calibration.** R3: NPE-only P–P curves for the 9
  parameters on Xu3, plus a coverage table (g1/g2 67.4/67.8%; nuisances
  within 2.5 pp).

**Sec. 6 "Towards an NPE-based Shear Estimator"** (Eason, 2026-09-30). This is
an overview of explorations, NOT a final calibrated, amortized estimator
tested on realistic data. Each subsection gives a short theory, then the
result. The Discussion and Conclusions keep the same exploratory framing.
- **6.1 Per-galaxy posterior means (raw).** Equal weights, full Xu3 sample,
  no post-processing. R5: binned ĝ − g vs g, low-|g| fit, m and c, plus the
  per-galaxy σ_ε,i distribution and the ensemble σ_ε (Xu24 definition).
  Expected: c ≈ 0, m ≈ −0.6.
- **6.2 Why posterior means shrink.** Gaussian toy model: m_i ≈
  −σ_i²/(σ_i² + σ_π²) with σ_π = 0.0577. D1: measured response vs σ_i² and
  this prediction; optional D4, response vs inclination.
- **6.3 Weighting and posterior-width cuts.**
  - Inverse-variance weights with a shape-noise floor.
  - R6: cuts on σ_ε,i, showing m, σ_ε and retained fraction vs threshold.
  - Include the selection-bias checks (T1/T2).
- **6.4 Response calibration.** R(σ) correction. The stencil exists only for
  the legacy model: either quote the legacy numbers with a note, or run a
  small stencil for the current model (T2b doubles as this).
- **6.5 Likelihood stacking with the NPE.** Eason, 2026-09-29: include
  NPE-based stacking, not NRE.
  - Convert the posterior to a likelihood (q/π_train) and combine galaxies
    that share a constant shear, as in Bernstein & Armstrong 2014.
  - Results come from the fixed-shear benchmark (T8).
  - No NRE results.

**Sec. 7 Discussion**
- Shape noise vs Xu+24.
- Local calibration (D2).
- Simulation limitations.
- Future work: 1M model, Xu5, pretraining, R90, logit limits, TF/m_r
  uncertainty, NRE.

**Sec. 8 Conclusions**

## 4. Appendix candidates

- A. Test-set construction details: catalog counts, support cuts, SNR
  column definitions.
- B. Full architecture and hyperparameter tables, if the main text gets
  shortened.
- C. Per-parameter P–P plots and coverage for all 9 parameters (the main text
  keeps g1, g2 and a summary).
- D. Shrinkage derivation (the Gaussian toy model of 6.1) in full.
- E. Response-calibration fit details (the stencil, the R(σ) fit).
- F. Pretraining (CCL), R90 and other ablations, once they are run.
- G. Galaxy-frame (g+, g×) bias results, if the main text uses g1, g2 only.

## 5. Fast tests before writing Sec. 5–6 (full plan in `PAPER_TESTS_PLAN.md`)

Most use only the existing Xu3 cache (CPU, minutes).

- **T1: does σ_i depend on g?** Bin the per-galaxy σ_ε,i by the true |g|,
  g+ and g× (in the galaxy frame), and fit dσ/dg. If it isn't flat,
  weighting or cutting on σ_i introduces a selection bias in m. (cache)
- **T2: selection bias from the width cut.**
  - (a) Compare the width cut with cuts on true quantities that don't depend
    on g (true cos i, ρ_I, ρ_S) at matched retained fractions. If m follows
    the same curve vs mean σ_i, the σ-cut adds no extra bias. (cache)
  - (b) A metacal-style selection response: on a matched-shear stencil (same
    galaxy, several g), measure how the selected fraction and mean
    ĝ change with g. Needs a small new inference run on the stencil
    (~1000 galaxies × 5 shears, minutes on one GPU).
- **T3: shrinkage prediction check.** In bins of σ_i, compare the measured m
  with −⟨σ_i²/(σ_i² + σ_π²)⟩. Confirms 6.1 quantitatively for the fibrepair
  model. (cache)
- **T4: linear regime.** m from the |g| < 0.02 fit vs the full range (or a
  cubic fit), to justify the low-|g| fit and show the prior-edge effects.
  (cache)
- **T5: bootstrap errors.** Bootstrap m, c and σ_ε for the raw, weighted and
  cut estimators, instead of relying only on fit errors. (cache)
- **T6: TF-weight health.** Distribution of the TF importance ESS fraction
  per galaxy (reference mean 0.60). Check that no subset (e.g. faint or
  edge-on galaxies) has a degenerate ESS. (cache)
- **T7: σ convention.** σ_ε from the standard deviation vs the 16–84%
  half-width, to check that the choice doesn't change conclusions. (cache)

## 6. Not included (decided)

NRE results (NPE stacking *is* included), the fiber hop, the Fisher bound, MAP/Laplace estimators, the
g-prior-width study, architecture ablations, pretraining results (until the
comparison is run), and Xu1/Xu5.

## 7. Eason's decisions after reading the per-test pages (2026-09-30)

- **Example posteriors.** Mention in Sec. 5 that Hα SNR, not image SNR,
  narrows the shear posterior. The 9D corner plot goes to an **appendix**.
  Paper figures need getdist-style smoothing: smooth 1D marginals and smooth,
  non-blobby 2D contours.
- **TF weights.** Not a paper result. Add one sentence in Sec. 4.3 ("verified
  on the test set that the reweighting keeps enough samples for almost all
  galaxies"). Optional later check: are low-ESS galaxies edge-on or wide, and
  would the cuts remove them anyway?
- **Calibration.** Keep it. Emphasize that what matters in the end is that g
  is calibrated. The θ_int under-coverage may be an artefact of intervals on
  a circular coordinate (to check).
- **Error bars.** Use the cheapest accurate ones (the fit errors).
- **Width convention.** Use the posterior standard deviation. How to present
  it is decided while writing.
- **Linearity.** Fit over |g| < 0.02 because it covers the shear of the vast
  majority of real galaxies. Beyond that, nonlinear effects appear, as other
  methods have also reported (find citations).
- **Training.** Show the loss figure and mention the result briefly.
- **Shrinkage.** A paper result (Sec. 6.2): m = −σ_post²/σ_π², derived from
  the response.
- **Response calibration.** Describe the method; show m, shape noise and the
  effective sample size only. More investigation after the draft.
- **Stacking.** Show the realistic case: one noise realization per galaxy,
  with realistic error bars. Non-constant-shear stacking (Bernstein &
  Armstrong 2014) is future work, not this draft.
- **Open questions from Eason:**
  1. Why does selection bias from width cuts or weighting matter, given that
     traditional WL also applies cuts?
  2. What does the local, shear-dependent miscalibration do to the results?

## 8. Paper figure and table plan (2026-09-30)

**Output locations:**
- Final figures go to `$S/reports/paper-tests/paper_figs_final/`. After
  Eason approves them, copy them into the Overleaf `figs/`.
- One script, `kl-nn/arch/diagnostics/paper_tests/make_paper_figures.py`,
  with a shared style module.

**Style, applied to every figure:**
- Single-column width 3.4 in, double-column 7.0 in; vector PDF plus PNG.
- Serif font with Computer Modern math (`mathtext.fontset="cm"`).
- Consistent colours: raw vs weighted vs calibrated vs stacked.
- **Posterior panels use getdist 1.7.6** (installed in `kl-nn`), with
  TF-weighted samples and smooth 1D and 2D KDEs (smooth_scale_1D/2D ~0.3–0.5;
  tune visually). This replaces the current histogram / blobby contour plots.
- Axis labels use physical symbols (g₁, g₂, cos i, v_circ, …). No internal
  codes appear anywhere.

| # | Sec. | Figure / table | Content | Data source | Status |
|---|---|---|---|---|---|
| F1 | 5.1 | Training loss | Train and validation NLL vs epoch; best epoch marked | losses CSV (A_npe) | Exists; restyle |
| F2 | 5.1 | Example posterior vs SNR | g₁–g₂ getdist contours of one galaxy (6447): (a) ρ_I = 50 vs 200 at fixed ρ_S; (b) ρ_S = 8 vs 60 at fixed ρ_I; truth marked. Shows that Hα SNR, not image SNR, narrows the shear posterior | saved samples in `$S/cache/paper_tests_B_inference/r2b/` | Regenerate (smooth) |
| F3 | 5.2 | Calibration | P–P curves for g₁ and g₂ (main text) with the 95% band; the 7 nuisances in lighter lines, or appendix | `A_npe/a_npe_stream.npz` | Exists; restyle |
| Tab. 1 | 5.2 | Coverage | 50/68/95/99% coverage for g₁, g₂ (plus 68% for the nuisances) | A_npe report.json | New table |
| F4 | 6.1 | Raw bias | Binned ĝ−g vs g for g₁, g₂ with the \|g\|<0.02 fit and m, c in the legend | A_shear per_galaxy.npz | Exists; restyle |
| F5 | 6.2 | Shrinkage | Measured m vs ⟨σ_post²⟩ in bins of predicted σ (selection-free), with the line m = −σ_post²/σ_π² | A_shear supplement.json | Exists; restyle |
| F6 | 6.3 | Weighting and width cuts | m₁, m₂ and kept fraction vs σ_ε,i threshold for raw and weighted; mark where selection response matters; **fix the kept-count axis bug** | A_shear per_galaxy.npz; B stencil R_sel | Regenerate |
| F7 | 6.4 | Response calibration | R vs σ_post² on the Xu3-like stencil with the a+bσ² fit and the Gaussian prediction (optionally a panel of R vs cos i) | B_inference t2b_xu3like.json / cubes | Exists; restyle |
| F8 | 6.5 | NPE stacking | m vs number of galaxies at **one noise realization per galaxy** (realistic), bootstrap error bars, per-galaxy mean for comparison; optionally stacked ĝ−g vs true g per cell | C_stacking cache `analysis_v1.npz` | Regenerate (R=1 version) |
| Tab. 2 | 6 | Estimator summary | Rows: raw mean; weighted; width cut (t = 0.05, with R_sel quoted); ensemble response calibration; stacking (benchmark, R=1, N=32). Columns: m₁, m₂, c, effective shape noise, effective number / kept fraction, dataset used | A_shear, B, C JSONs | New. **Needs the calibrated shape noise** (ĝ/R scatter), not yet computed |
| FA1 | App. | 9D corner plot | getdist triangle plot of all 9 parameters for galaxy 6447 with truths | B saved 9D samples | Regenerate (smooth) |
| FD1 | Disc. | Local calibration | Measured vs shrinkage-predicted coverage vs true g | D_local_calibration (running) | Depends on the test |

**Small computations still needed:**
1. Calibrated effective shape noise and effective sample size for Tab. 2.
2. Stacking at one realization per galaxy with bootstrap errors vs N, if the
   existing `m_vs_N_R1` numbers lack them.
3. Kept counts for F6 (from the T5 `ess_low` values).

**Eason's answers (2026-09-30):**
- Nuisance P–P goes in the appendix.
- Stacking stays OUT of Tab. 2 and is discussed separately.
- F2 has two side-by-side panels (image SNR and Hα SNR).

Generation launched with two agents:
- Sec. 5: `make_paper_figures_sec5.py`.
- Sec. 6: `make_paper_figures_sec6.py`.
- Shared style: `paper_style.py`.
- Outputs in `paper_figs_final/`, with draft captions in
  `captions_sec{5,6}.md`.
