# KL-NN paper: gaps found while drafting

Things we realized we still need to do while writing the first draft.
Newest entries at the bottom of each list. Mark done items with [x].

## Open

- [ ] **MCMC run time for DESI datavectors** (Introduction, cost paragraph).
  Ask Jiachuan / Lauren for a typical wall-clock time per galaxy for the
  DESI-KL MCMC runs.
- [ ] **Train the paper model on 1M / validate on 100k** (Sec. 3, 4, 5).
  Before training, repair the fiber gauge of `datasets/train_1m_simv3_cosi`.
  It is not in `reports/fiber-gauge/01_repair_audit/verify.json`, so it is
  probably still in the legacy (left-handed minor-axis) convention.
- [ ] **Evaluate Xu5 with the current model.** The cos i Xu5 test set was
  also not in the fiber repair audit. Xu1 is dropped from the paper (see below).
- [ ] **Run the 4-galaxy MCMC comparison** (`/jet/home/xwang30/mcmc_xu1_four_fits`).
  Those FITS come from the old sin i Xu1 set; match them to the cos i model.
- [ ] **Ask Jiachuan about the continuum template** (Sec. 3.1). kl-tools
  uses `data/Simulation/GSB2.spec` (no header). The paper describes it
  generically, without a citation, until confirmed. METHODS_MANUSCRIPT.md
  guesses Kinney+96 SB2 (starburst), while Xu+24 calls it a "barred spiral"
  continuum.
- [ ] **Software versions in the Acknowledgements** (advisor suggestion). An
  `\XW{}` note is in `main.tex`: GalSim 2.7.2, NumPy 1.26.4, SciPy 1.13.1,
  PyTorch 2.9.0, nflows 0.14, kl-tools commit 42e06b6 (conda env `kl-nn`).
  Re-check them before submission.
- [ ] **Test the model on samples at other redshifts** (Sec. 3.1). The paper
  justifies the single training redshift (z = 0.3) by assuming spectra are cut
  out centered on the redshifted Hα line, with apparent size and brightness
  covered by the R_h, m_r and F_Hα ranges. This is not demonstrated yet. At a
  fixed 0.08 nm/pixel, the velocity per pixel scales as 1/(1+z), so rotation
  curves span a different number of pixels at other z.
- [ ] **True-valued metadata** (Sec. 3.2). The model receives the true m_r,
  ρ_I and ρ_S. For real data these carry measurement noise; discuss in
  limitations or test with noisy metadata.
- [ ] **Xu1 test set** (Sec. 3.3). Xu1 is excluded because only ~7% of its
  catalog has a DESI Hα measurement (2.42M eligible out of 36.5M, vs 2.32M out
  of 10.4M for Xu3), so it collapses onto Xu3. Possible workaround: assign Hα
  SNR to galaxies without spectra, e.g. following however Xu+24 assigned
  line fluxes to sample 1. That would extrapolate beyond BGS depth.
- [ ] **Test-set distribution figure** (Sec. 3.3). Confirm
  `figs/xu_samples_3_5_distributions.png` is made after the support cuts.
  No image-SNR panel is needed, since it is highly correlated with m_r.
- [x] **Redraw the architecture figure** (done 2026-10-02: TikZ `figures_src/nn_arch_v2.tex` → `figs/nn_arch_v2.pdf`, now used in the paper) (`figs/nn_arch.eps`, Sec. 4). Add
  the metadata branch. Replace the FC-1024 layer and MLP-4 with
  concat(1152) + LayerNorm. Update the flow (4 RQ-spline MAF layers plus a
  circular spline, not 12 affine layers). Change the loss from D_KL to NLL.
  The spectra branch pools only after the 16/32-channel pairs.
- [ ] **Magnitude uncertainty in the TF prior** (Sec. 4.3). m_r is treated
  as exact; the conservative 0.1 dex scatter partly absorbs the m_r
  uncertainty (Eason). DESI/LS catalogs provide the r-band flux ivar, so a
  per-galaxy m_r-uncertainty treatment could be added later. The test sets
  use the same TFR as the prior, i.e. a perfectly known TFR.
- [ ] **METHODS_MANUSCRIPT.md says the disk is "infinitesimally thin"**. That
  is true only for the velocity field (as in Xu+24). The image uses GalSim's
  default h/r = 0.1. Fix the manuscript if it is kept.

## After this draft

- [ ] **Pretrain vs no-pretrain comparison.** Train the same NPE end-to-end
  from scratch and compare it with CCL pretraining plus a frozen encoder: m,
  σ_shape, coverage, validation NLL, training stability and time. Decided
  2026-09-29: the paper is written **as if there is no pretraining** until
  this shows a benefit. The shelved subsection is in
  `sections/pretrain-ccl.tex` (not included). All current results (fibrepair
  model) use a CCL-pretrained frozen encoder, so they don't yet match the
  paper's description. Related: the frozen vs fine-tuned ablation
  (45255702 vs 45255704) showed no difference.

- [ ] **R90 ablation** (Sec. 4.2). Eason recalls no significant improvement
  from R90 in training or R90 at inference. Compare the four combinations of
  R90 on/off in training and at inference on m, c (especially), σ_shape and
  coverage, to justify keeping the R90 paragraph in the paper.

- [ ] **Flow logit saturation** (Sec. 4.2/4.4; found 2026-09-29). In the
  canonical model's training log, the bounded spline logits of both the box
  flow and the θ spline sit at the tanh ceiling (10.0) from ~epoch 20; raw
  logits grow to ~50–56 and max spline derivatives are pinned at the cap
  (10). The flow may be unable to become as sharp as it wants, which could
  cap posterior sharpness and relate to local miscalibration. Test a higher
  logit limit and/or more θ spline layers.
- [ ] **θ_int learning-rate split** (Sec. 4.4). lr 1e-4 for the θ spline vs
  3e-4 elsewhere came from an agent-written Stage 5 script (commit 74ed6df,
  2026-08-17) and was never ablated. A plausible rationale: θ is the sharpest
  posterior dimension (final-epoch θ log-prob 4.39 vs ~1.7 per non-θ
  dimension) and is carried by a single spline layer. Optionally compare
  with a single lr.

- [ ] **Fiber placement uses the noiseless true sheared axes** (found
  2026-09-29, paper-tests B). `compute_fiber_offsets` takes the true g, so
  the fiber coordinates in the metadata encode the noiseless observed axis
  direction. For nearly face-on disks that direction is set by the shear, so
  the model may get a noise-free shear direction. Symptom: shear response
  R ≈ 1.4 (> 1) for the narrowest-posterior, cos i > 0.9 galaxies. Real
  fibers would be placed from a noisy image. Test by placing fibers from a
  noisy measured axis (or fixed sky axes) and comparing R for face-on
  galaxies. Mention in limitations at least.
- [ ] **σ-cut selection bias** (paper-tests A T2a, B T2b). Cutting on
  posterior width keeps galaxies scattered outward: R_sel/R_γ = +12% at
  σ_ε,i < 0.05 and +3% at < 0.075. Weighting shows no such bias. Sec. 6.3
  must present cuts with this caveat or correct them with R_sel.
  **Update 2026-10-01:** weights are NOT selection-free. On the stencil, the
  floor weights give R_sel/R_γ = +3.0 ± 0.2% and the no-floor weights
  +5.8 ± 1.2% (`paper_tests/e_sel_weights.py`). The earlier "weighting is
  safe" came from the noisier Xu3-cache comparison. Posterior-cos i cuts:
  |R_sel/R_γ| ≲ 3% (`e_sel_cosi.py`).

- [ ] **Shrinkage model details** (Sec. 6.2; from paper-tests T3/S4).
  - The paper uses the Gaussian-prior formula m = −σ_post²/σ_π² with the
    uniform prior replaced by a Gaussian of equal variance. A 1D toy with the
    real box prior fits worse (χ² 37.5 / 17.3 vs 18.4 / 15.8 for 12 bins),
    and nobody knows why.
  - The formula fails per component in the galaxy frame (m+ more negative,
    m× less negative than predicted; χ² 128 / 90). The two errors cancel in
    the image frame. The paper stays in the image frame and does not mention
    this.

- [ ] **Response-calibration transfer** (Sec. 6.4; found 2026-10-01). The
  stencil R(σ) (a = 1.19) overpredicts the Xu3 response by 10–20%: the
  selection-free ⟨ĝ⟩/⟨R⟩ gives m −0.17 / −0.10. The R²-weighted calibration
  reaches m ≈ 0 only because its selection response (+0.087) cancels this.
  Needs a larger calibration sample matched to the data (Xu3 has 9% with
  σ_ε < 0.03 vs 3% on the stencil) plus an R_sel term. Check after the
  fiber-placement fix (R > 1 drives a > 1).

- [ ] **Use a package KDE for stacking in production** (Sec. 6.5; Eason,
  2026-10-01). The current KDE is custom PyTorch code (`c_stack_analysis.py:
  kde_density`, per-axis Scott bandwidth, reflection, floor). Switch to an
  existing tool (e.g. `scipy.stats.gaussian_kde`, whose default is Scott; in
  2D Silverman = Scott), using mirrored samples for the box edges, and check
  that the results are unchanged. Scipy uses the full covariance kernel, not
  per-axis widths.

- [x] **Local (shear-dependent) miscalibration** (tested 2026-09-30,
  `paper-tests/D_local_calibration/`). **Explained by shrinkage:** the
  posteriors' own self-consistent prediction reproduces 88% / 62%
  (|g| < / > 0.05) to within 1 pp. The effect comes from wide,
  prior-dominated posteriors. Small genuine errors remain:
  - mass misplaced near the prior edges (+8% at 0.09 < |g1| < 0.1, −14% in
    the next bin inward);
  - corners covered 15% vs 8% predicted;
  - coverage in bins of ĝ off by up to 5.8 pp at |ĝ| ≳ 0.08.

  Possibly the bounded flow's edge handling; check on the 1M model.
- [ ] **θ_int coverage on a circle** (Eason's hypothesis). The under-coverage
  (64.5%) may come from defining 16–84% intervals on a wrapped coordinate
  when the posterior is wide. Recompute with circular ranks or HPD intervals
  and split by cos i (face-on galaxies have broad θ_int).
- [ ] **Low-ESS galaxies vs selection** (optional). Are low TF-ESS galaxies
  edge-on or wide-posterior ones that cuts would remove anyway?
- [ ] **Paper figure style.** Regenerate the posterior figures with
  getdist-like smoothing (smooth 1D marginals, smooth 2D contours).
- [x] **Citations for nonlinear shear bias** beyond |g| ~ 0.02 in other
  methods (for the linear-fit justification). Resolved 2026-09-30 with
  Anacal (Li & Mandelbaum 2023; negative third-order bias without blending)
  and AnaCalCluster (Zhou+26, arXiv 2606.21141, Eq. 29, cubic term up to
  |g| ~ 0.15).
- [ ] **Matched targeting/inference noise test** (Sec. 3.1; added 2026-10-09;
  run after the 1M model, ablations and paper reruns finish). In the
  measured-axis data, the fiber PA is measured on a targeting image made at
  generation, while the network sees a different noise realization drawn at
  train/valid/cache time. This holds in validation and test too.
  - Targeting image: GalSim CCDNoise (LS DR9 sky + read + source Poisson),
    seed 9e8+ID (`data_generate/fiber_targeting.py`).
  - Network image: white Gaussian at the catalog S/N, torch-seeded
    (`arch/data.py: apply_image_noise_for_snr`; `arch/train.py: sample_density`).
  - The model also takes the fiber positions as an input (`fib_pos`), so it
    can compare β with the apparent image axis.
  - If DESI-KL infers from the same LS image used for targeting, the matched
    case is the realistic one.

  Decided: we will not re-place fibers per epoch (too costly), so training
  stays independent-noise. Test plan:
  - Build a matched Xu3 test variant. The network image is the stored
    targeting image, reproduced from the clean image and TGTSEED, and the
    image-S/N input is TGTSNR (= ‖I‖/σ_sky, same definition the network uses).
    Spectra and fibers are unchanged.
  - Cache it with the measured-axis model and compare coverage, m/c, stencil
    face-on R and the annulus stack against the independent-noise Xu3 cache.

  Caveat: the targeting noise includes source Poisson noise, which training
  noise lacks; it raises the central-pixel variance by tens of percent for
  bright galaxies. If the comparison differs, the options are to freeze the
  image noise per galaxy in the measured-axis datasets, or to state the
  independence assumption in the paper.

## Checked, not a gap

- **Disk thickness q_z (2026-09-28).** The simulator is not an infinitely
  thin disk. `kl_tools/intensity.py` calls `galsim.InclinedExponential`
  without a scale height, so GalSim uses its default `scale_h_over_r = 0.1`.
  That equals the q_z = h_s/R_s = 0.1 of Xu et al. 2024. The earlier claim of
  an infinitely thin disk was wrong.
- **θ_int sign convention (2026-09-28).** The paper uses the standard
  counterclockwise-positive θ_int, so g+ = g1 cos2θ + g2 sin2θ and
  g× = −g1 sin2θ + g2 cos2θ. kl-tools and `arch/utils.py:img_to_gal_axis` use
  clockwise-positive θ; the two are equivalent under θ → −θ. The paper does
  not need to match the implementation.
- **WL shape noise ≈ 0.3 (2026-09-28).** Eason: this is an accepted
  approximate value summarizing results up to Stage-III, so it needs no fix. A
  stronger citation than Bernstein & Jarvis 2002 would be optional.
- **Stamp size (2026-09-29).** The "smallest stamp containing almost all
  flux" justification was dropped from the paper; Eason removed the note.
  For reference, a 48x48 stamp keeps ~63% of a face-on exponential disk's
  flux at R_h = 5".
- **Data generation cost (2026-09-29).** "~2000 CPU hours" is deliberately
  approximate (probably 1000-3000, depending on hardware) and holds for
  simulator v3. No fix needed.
- **R_h ≥ 1" vs 1.5" (2026-09-29).** Eason verified there is no significant
  change in m or shape noise with R_h above 1". Effects appear only below
  about one fiber radius. This is stated in Sec. 3.3.
- **Catalog flux SNR vs matched-filter SNR (2026-09-29).** Eason is aware of
  the difference; the note was removed from Sec. 3.3. The real-data metadata
  caveat stays in Sec. 3.2.
- **Population shift in the coverage test (2026-09-30).** Xu3 draws R_h (and
  m_r, ρ_I, ρ_S) from the catalog instead of the training prior, so the NPE
  posteriors are not exactly the Bayesian posteriors for that population.
  Eason: this is accepted for NPE, since no fully realistic population can be
  simulated. If it turns out to matter more, that motivates exploring NRE.
  No note in the paper.
