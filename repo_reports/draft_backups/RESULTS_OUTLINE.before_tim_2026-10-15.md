# KL-NN paper: Results / post-processing outline

Drafted 2026-09-29, revised the same day after Eason's feedback. It lists
figures and tables, what each claims, the data behind it, and fast tests to
run before writing. Figures are generated first, then text is written.

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
