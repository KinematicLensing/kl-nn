# KL-NN paper: first-draft handoff (written 2026-09-28)

## RESUME HERE (last updated 2026-09-29, before a possible disconnect)

**Workflow Eason set.** Write one subsection at a time. Eason reviews style and
content before the next one is started. Never include claims he cannot back
up. For each subsection, list in chat the source of every factual claim and
every change to his existing text. Style rules are in memory
(`kl-nn-paper-writing-style.md`): plain, slightly informal voice;
hand-written "Name et al.~\citep{}"; name authors only at first mention; no
"KL group"; moderate spec detail. Gaps go in `\XW{}` comments phrased as
Eason's notes-to-self, and are also logged in `kl-paper-gaps.md` (same
folder, with an "After this draft" section for post-draft experiments).
Eason often hand-edits after approving, so always re-read a file before
editing it.

**Progress (files in the Overleaf clone `sections/`).**
- [x] Introduction (`introduction.tex`): final. Eason removed the timing
  sentence and added his own notes (S8 tension; DESI MCMC run time from
  Jiachuan/Lauren; kl-pipe speed-ups).
- [x] Sec. 2 KL basics (`kl-basics.tex`): approved. Counterclockwise θ_int
  (the paper doesn't follow kl-tools' convention), q_z = 0.1, WL shape noise
  ≈ 0.3.
- [x] Sec. 3 (`data-generate.tex`), titled "Simulating DESI-like Images and
  Spectra": 3.1 Galaxy and Observation Model (the DESI-PV paragraph mentions
  Willett et al. in prep.; the SED template is described generically,
  pending Jiachuan; single-z justification), 3.2 Training and Validation
  Data, 3.3 DESI-like Test Sets (**Xu3 and Xu5 only; Xu1 dropped**). All
  approved.
- [x] Sec. 4.1 Feature Extraction (`nn-arch.tex`): approved. No FiLM, no
  pretraining sentence. The figure caption has Eason's shortened redraw note.
- [x] Pretraining is **shelved**: `sections/pretrain-ccl.tex`, not included.
  The paper is written as if there is no pretraining. Pretrain vs
  no-pretrain is under "After this draft".
- [x] Sec. 4.2 Posterior Estimation with Normalizing Flows: approved. Bounded
  RQ-spline MAF (4 layers, 8 bins, 256 hidden, 2 blocks, uniform box base) +
  circular spline for θ_int, and the R90 paragraph (kept; R90 ablation under
  "After this draft"). The estimator is the posterior mean.
- [~] Sec. 4.3 Incorporating the TF Prior (label `ssec:tf-prior`): revised
  after Eason's feedback (0.1 dex partly accounts for m_r uncertainty; note
  on using LS flux ivar later). Awaiting a final look. Post-hoc importance reweighting,
  mean (36 − m_r)/7.22, 0.1 dex, truncated to 60–540 km/s, m_r exact.
- [x] Sec. 4.3 TF prior approved (2026-09-29).
- [~] Sec. 4.4 Training Configurations: drafted 2026-09-29 and awaiting
  review. Pretraining is mentioned only in Eason's requested `\XW{}` comment.
  Verified from the config/code: AdamW, wd 1e-5, lr 3e-4 (shared and non-θ
  flow) and 1e-4 (θ flow), 2-epoch linear warmup from 0.1×, cosine to 1e-5,
  grad clip 1, batch 400 on 4 V100s, ≤200 epochs, early stopping (patience
  20, min_delta 1e-3), best = epoch 200, 58 min (sacct 46075749). Notes: why
  θ has a lower lr; update for the 1M model. Bib: AdamW.
- [x] Sec. 4.4 approved (2026-09-29). The θ-lr note was dropped; the lr split
  and flow logit saturation are under "After this draft".
- [~] **Sec. 5–6 plan** (2026-09-29, end of session). See
  `RESULTS_OUTLINE.md`:
  - **Sec. 5 "Validating the Neural Posterior Estimator"**, compact: 5.1
    training + example posteriors (speed only as a brief mention, no
    figure); 5.2 calibration.
  - **Sec. 6 "Shear Estimation"**, the main results: 6.1 raw posterior
    means; 6.2 shrinkage theory; 6.3 weighting + width cuts; 6.4 response
    calibration; 6.5 **NPE likelihood stacking (included; NRE excluded)**.
  - Then Discussion, Conclusions, appendices.
  - Scope: the 100k pretrained fibrepair model on Xu3 only, no new runs
    beyond small tests (a note in `results.tex`).
- [x] **Tests DONE 2026-09-29.** Combined report:
  `$S/reports/paper-tests/index.html` (regenerate with
  `kl-nn/arch/diagnostics/paper_tests/write_index.py`); README row added.
  Headlines:
  - raw m = −0.616/−0.592, c ≈ 0;
  - shrinkage is m = −σ_post²/σ_π²;
  - width cuts are selection-biased (R_sel/R = +12% at 0.05), weighting is
    safe;
  - ensemble R(σ) calibration gives m ≈ (−0.05, 0.00);
  - NPE stacking gives m ≈ (−0.013 ± 0.006, 0.000 ± 0.005) on the benchmark;
  - calibration fine except θ_int at 64.5%;
  - image SNR barely changes the shear posterior, ρ_S does;
  - possible fiber-placement leak (R ≈ 1.4 for face-on).

  Explanatory per-test pages (Eason's request, 2026-09-29):
  - 14 pages, one per test, in `paper-tests/tests/<slug>.html`, each with
    question / why / theory / procedure / results / interpretation /
    caveats / provenance, and MathJax equations.
  - Generator: `paper_tests/write_test_pages.py`, built on
    `test_pages_common.py` and three content modules (`pages_npe.py`,
    `pages_shear.py`, `pages_stencil_stack.py`), drafted by 3 subagents.
  - Numbers must come from JSON via `J()`.
  - The index is rebuilt with descriptive names plus a code → page glossary.
  - Plan: `~/.claude/plans/there-were-so-many-fuzzy-snowglobe.md`.
  - **DONE 2026-09-29.** 14 pages + index generated and checked (key numbers,
    no codes in headings, balanced MathJax).

  Reviewed with Eason 2026-09-30:
  - Decisions are in RESULTS_OUTLINE §7.
  - **Sec. 6 retitled "Towards an NPE-based Shear Estimator"** (exploratory
    scope, no final calibrated estimator claimed).
  - The figure plan is in RESULTS_OUTLINE §8.
  - A conditional-coverage (local miscalibration) test is running as a
    subagent; it writes to `paper-tests/D_local_calibration/` and
    `pages_localcal.py`. Add slug "local-calibration" to `ORDER` in
    `write_test_pages.py` when it finishes.

  Figures DONE 2026-09-30 in `paper_figs_final/` (Eason: "look good for
  now"); scripts `make_paper_figures_sec{5,6}.py` plus `paper_style.py`.
  Local-calibration test DONE: shrinkage explains it; page
  `tests/local-calibration.html`.
- [~] **Sec. 5.1 drafted 2026-09-30** (`sections/results.tex`, now titled
  "Validating the Neural Posterior Estimator").
  - Old takeaway list replaced; backup in scratchpad `results.orig.tex`.
  - Copied `fig_training_loss.pdf` and `fig_posterior_snr.pdf` into
    Overleaf `figs/`.
  - References `sec:shear` (Sec. 6, not yet written) and `app:corner`
    (appendix, not yet written).
  - The intro's layout paragraph still needs a Sec. 6 mention.

  5.1 feedback applied (2026-09-30):
  - loss y-axis is now "NLL loss" (figure regenerated and copied);
  - Eason rewrote the image-SNR sentence himself (no 0.026/0.027 numbers);
  - the unsupported g2/g+ "only constrained by TF" sentences were removed.

  Eason approved 5.1 (he merged the 9D-posterior sentence into the SNR
  paragraph).
- [~] **Sec. 5.2 "Posterior Calibration" drafted 2026-09-30**
  (`ssec:calibration`, in `results.tex`).
  - `fig:pp-shear` plus an inline `tab:coverage`. The KS/DKW/PIT jargon was
    removed: the figure axes now read "fraction of galaxies with u_i ≤ u",
    "u" and "95% range if calibrated", and the table's KS row is gone (the
    generator `make_paper_figures_sec5.py` was updated to match).
  - Copied `fig_pp_shear.pdf` and `fig_pp_nuisance.pdf` into Overleaf
    `figs/`.
  - New bib entries: SBC (Talts+18) and TrustCrisis (Hermans+21).
  - `\XW` notes: the catalog R_h vs the uniform training prior; the θ_int
    circular-coverage check.
  - TF-health sentence added to Sec. 4.3 (`nn-arch.tex`): median ESS 60%;
    99.6% of galaxies keep ESS > 1000 out of 10^4.
  - New dangling references: `app:coverage` (nuisance appendix) and
    `sec:discussion` (not written yet).

  5.2 round-1 feedback applied (2026-09-30):
  - standard terms are back (coverage test, PIT, P–P plot, DKW band,
    binomial errors), each with a brief explanation;
  - new paragraph on what the P–P plot shows vs the coverage table;
  - both `\XW` notes removed (R_h population shift accepted; θ_int stays in
    the gaps file);
  - citations are now CookGelmanRubin06, SBC, TrustCrisis (TMLR 2022),
    DKW56, Massart90, plus FieldLevelWL (Zhou+24) and NPEShear (White+26) as
    WL examples;
  - figure legend is "95% DKW band" again.

  5.2 round 2: the P–P vs coverage comparison is cut to one "whereas"
  sentence at the end of paragraph 1. The results paragraph now just says
  "well calibrated globally"; the g1-slightly-narrow details were dropped.

  5.2 round 3 (trial, Eason checking how it reads): central-interval
  coverage test and `tab:coverage` removed; only the P–P plot remains.
  Nuisances are described as P–P checks at u = 0.16 / 0.84. If this sticks,
  drop the 68%-coverage panel from `fig_pp_nuisance` when writing
  `app:coverage`. Backups: scratchpad `results.v52{a,b,c,d}.tex`.

  Eason approved the P–P-only 5.2 (2026-09-30).
- [~] **Sec. 6 intro + 6.1 "Raw Posterior Means" drafted 2026-09-30**, in the
  new file `sections/shear.tex` (`\subfile` added to `main.tex`).
  - The intro's layout paragraph now mentions Sec. 6.
  - `fig_raw_bias.pdf` copied to Overleaf `figs/`.
  - Numbers: m1 = −0.616 ± 0.014, m2 = −0.592 ± 0.014, c ≈ 0; median
    σ_ε,i 0.066, 16–84% range 0.040–0.079; 31.5% of galaxies have
    σ_ε,i > 0.075 (prior-only value √2σ_π = 0.082).
  - Xu+24 checked: σ_ε = √(σ_g1² + σ_g2²) from the 1D posteriors (their
    Sec. IV B). Their sample-averaged values (weighted by density and inverse
    cosmic-shear variance) are 0.047 (sample 3) and 0.041 (sample 5), using
    four emission lines. Comparison deferred to the Discussion.
  - Dangling references: ssec:shrinkage, ssec:weights-cuts, ssec:response,
    ssec:stacking, sec:discussion.

  6.1 round 1 (2026-09-30):
  - Nonlinear-response sentence now cites Anacal and AnaCalCluster.
  - Added a sentence that uncalibrated WL estimators have m of tens of
    percent: DESY1Shape Fig. 14, im3shape −0.1 to −0.3; HSCY1Sims
    (Mandelbaum+18b, Fig. 12, blinded offset). It leads into 6.4.
  - σ_π explained: the uniform width w gives std w/√12, and √2 comes from
    the quadrature sum of the two components.

  Eason approved 6.1 (with small rewording).
- [~] **6.2 "Shrinkage of Posterior Means" drafted 2026-09-30**
  (`ssec:shrinkage`).
  - Gaussian derivation in Eqs. `eqn:gauss-posterior` and `eqn:shrinkage`.
  - A short "property of the posterior mean" paragraph:
    E[g | ĝ] = ĝ, but E[ĝ | g] is shrunk.
  - Selection-free binning on a cross-fitted GBM-predicted width.
  - fig_shrinkage copied to `figs/`. Numbers: χ² 18.4 / 15.8 for 12 bins;
    full-sample prediction −0.607 / −0.617.
  - The uniform-toy and galaxy-frame issues are left out of the paper and
    logged in the gaps file.

  6.2 round 1 (Eason): the GBM regression was too tedious for this result,
  so the test now bins on the measured σ_ε,i (15 bins, T3). The text says
  this is not the most stringent version and that the narrowest bins sit
  above the prediction (m > 0) because of noise selection, with a pointer
  to 6.3. χ² is not quoted (57.6 / 57.8 for 15 bins). `fig_shrinkage` was
  regenerated from T3 (no χ² in the legend). The selection-free numbers
  (χ² 18.4 / 15.8 for 12 bins) remain in A_shear/supplement.json as backup.

  6.2 round 2: selection is explained via the "measured shear" (no "noise
  realization" wording); "without any free parameters" removed from the text
  (kept in the caption); the closing sentence and the Sec. 6 intro now
  emphasize likelihood stacking (6.5) as the most natural approach.

  Eason approved 6.2 (2026-09-30) after his own edits.
  - He deleted the sentences on narrow-width bins having m above the
    prediction or above zero, and the selection caveat. He's unsure whether
    to discuss selection effects yet; they can be restored from scratchpad
    `shear.v62b.tex` if needed (e.g. when 6.3 discusses the width-cut
    selection bias).
  - He shortened the stacking sentence at the end of 6.2; the specifics
    belong in 6.5.

  **The scratchpad was wiped overnight (2026-10-01); the v52*/v62* backups
  are lost.** The 6.2 sentences Eason deleted were: "Only the bins with the
  narrowest posteriors lie above the prediction, and the two narrowest even
  have $m$ slightly above zero, which shrinkage alone cannot produce. This is
  the selection effect described above, which we come back to when we cut on
  posterior widths in Sec.~\ref{ssec:weights-cuts}." (He later kept the
  "Note that grouping galaxies by measured widths..." caveat paragraph.)
  Durable backups now go in `repo_reports/draft_backups/`.
  Eason also removed "which has no free parameters" from the fig_shrinkage
  caption.
- [~] **6.3 "Weighting and Cuts" drafted 2026-10-01** (`ssec:weights-cuts`).
  Three subsubsections: Weighting; Posterior Width and Inclination Cuts;
  Selection Bias (`sssec:selection`, deliberately pedantic for Eason to trim).
  - New figure `fig_cuts.pdf` (`make_paper_figures_sec6.py: fig_cuts`;
    numbers in `paper_figs_final/fig_cuts_numbers.json`). It plots m vs kept
    fraction for the width cut, the posterior-mean cos i cut and the true
    cos i cut, with the floor / no-floor weight points; bottom panel is
    σ_eff/√N_eff. It replaces fig_width_cut in the text.
  - New `tab:selection` (stencil R_sel/R_γ). Bib: MetacalPractical
    (Sheldon & Huff 17), HSCY1Shape (Mandelbaum+18; weights w = 1/(σ_SN² + σ_e²),
    their Eq. 35).
  - Key numbers:
    - floor weights m −0.512 / −0.486, N_eff 91%; no-floor −0.210 / −0.207,
      N_eff 10.5%;
    - kept 32%: width cut m −0.11, posterior cos i −0.27;
    - noise per component ×1e4: raw 1.79, floor 1.57, no-floor 1.08,
      width cuts 1.57–1.74, posterior cos i up to 1.98, true cos i up to 2.33;
    - R_sel/R_γ: width cut 0.05 +12 ± 3%, 0.075 +2.9 ± 0.8%, floor +3.0 ± 0.2%,
      no-floor +5.8 ± 1.2%, posterior cos i between −2.0 ± 1.0% and
      +2.8 ± 2.1%.
  - `\XW` notes: no-floor weights are the most efficient (story decision);
    trim the selection section; stencil caveats (size, R > 1 face-on).

  6.3 round 1 (2026-10-01; v1 backed up in `draft_backups/shear.2026-10-01-63v1.tex`):
  - Renamed "Applying Galaxy Weighting and Selection Cuts".
  - "A known m can be corrected for" removed (Eason: m on real data is never
    known).
  - The bottom panel of fig_cuts is now the UNCORRECTED propagated error
    σ_<ĝ> = sqrt(Σ w² σ_ε,i²)/Σw. Eason asked for an average of individual
    shape noise "like KL-I". The plain ⟨σ_ε,i⟩/√N_eff was rejected because it
    makes inverse-variance weights look worse (no-floor 3.28 vs raw 1.93).
    Values ×1e4: raw 2.02, floor 1.76, no-floor 1.30, width cuts up to 2.32,
    posterior cos i up to 2.86, true cos i up to 3.38.
  - Floor pros/cons kept, with Eason's comment for coauthors (he favours the
    floor).
  - Selection subsubsection shortened (eq., metacal link, method, table).
    Decide after 6.4 whether it moves to 6.4.

  6.3 round 2 (backup v2 in draft_backups):
  - The Xu+24 inclination claim is now attributed as their argument (they
    only show the shape-noise ratio in their Fig. 4; no number densities).
  - "Without correcting for bias" removed.
  - Cuts explained as w = 1/0, so σ_<ĝ> = σ_rms/√N_kept.
  - Floor-opinion comment moved to the end of 6.3.1.
  - Stencil-caveat comment deleted (kept between us: stencil small and not
    exactly Xu3; R > 1 face-on, see gaps). Still open: whether to move
    sssec:selection into 6.4.
  - S± and ⟨⟩_S± defined explicitly in the text.

  6.3 round 3: the fig_cuts bottom panel is back to the bias-corrected
  σ_eff/√N_eff (new Eq. `eqn:sigma-eff`), stated as "only meant to show
  trends", since m on real data is unknown. The uncorrected σ_<ĝ> version
  stays in the code as `sig_prop` for reference.

  6.3 round 4: Eason reworded parts (backup 63v4-eason). σ_eff is now framed as
  "correct each estimate with fitted m, c, then measure the spread around the
  truth" (mathematically identical, so numbers unchanged). The R_sel mechanism
  is rewritten as a concrete g1 = +Δ example. Eason: the rest of 6.3 is fine.

  Eason then shortened the R_sel > 0 explanation himself.
- [~] **6.4 "Response Calibration" drafted 2026-10-01** (`ssec:response`;
  backup of the pre-6.4 file in draft_backups).
  - fig_response copied to `figs/`. New Eq. `eqn:response`.
  - Stencil fit a = 1.19 ± 0.02, b = −350 ± 9 (Gaussian prediction 1, −300).
  - Per-galaxy division fails (m −0.39 / −0.40).
  - Ensemble calibration (R² weights): m −0.049 ± 0.029 / −0.000 ± 0.030,
    N_eff 42%, σ_eff/√N_eff 1.34e-4.
  - **New finding:** the R² weights carry a selection response,
    R_sel = +0.087 ± 0.005 on the stencil (`paper_tests/e_sel_calibrated.py`).
    The selection-free ⟨ĝ⟩/⟨R(σ)⟩ gives m −0.17 / −0.10 on Xu3
    (`e_cal_variants.py`), so the stencil R(σ) overpredicts the Xu3 response
    by 10–20%. "m ≈ 0" is therefore largely a cancellation. This is written
    into 6.4.
  - Not in the text: with the Gaussian R (no stencil), the ratio gives
    m +0.016 / +0.102 and the R²-weighted fit +0.15 / +0.21.
  - Recommendation for Eason: move sssec:selection into 6.4, since selection
    now matters there too.

  6.4 round 1 (Eason, 2026-10-01; backup 64v1):
  - Present the current calibration as is, with his `\XW` comment at the top
    ("done before I realized selection bias was a thing..."). No connection
    to selection, no hypotheses, minimal numbers.
  - The cancellation paragraph, the R > 1 cause note and the per-galaxy
    division numbers were removed; the stencil is re-described in 6.4 without
    referencing sssec:selection.
  - Plan (Eason): 6.3 mentions selection bias; a future 6.4 does the
    calibration properly with R_γ + R_sel. The cancellation finding is still
    logged in the gaps file.
  - The old codex/cursor scheme (σ/R < 0.2 cut, R clipped to [0.25, 1],
    per-galaxy division, floor weights) re-run on the current model gives
    m −0.12 / −0.09, N_eff ≈ 57% (`paper_tests/e_cal_oldscheme.py`), vs
    −0.05 / 0.00, N_eff 42% for the R²-weighted fit.

  6.4 round 2: the ensemble calibration is explained via Σĝ/ΣR (R≈0 galaxies add
  ~0 to both sums), plus new Eq. `eqn:ensemble-calibration`; the "R² weights"
  wording was dropped (backup 64v2).

  Eason decided to keep the fig_cuts bottom panel as σ_eff/√N_eff: it shows
  the shape-noise vs N_eff trade-off. Text should generally quote σ_eff.
- [~] **6.5 "Likelihood Stacking" drafted 2026-10-01** (`ssec:stacking`;
  opens with Eason's two sentences).
  - New Eqs. `eqn:stack` and `eqn:npe-likelihood`; Gaussian intuition
    σ_π²/(σ_π² + σ_L²/N).
  - KDE (Scott bandwidth, reflection, floor); benchmark of 25 cells × 32
    training-prior galaxies at R = 1, unweighted samples, paired bootstrap.
  - Result: per-galaxy average m ≈ −0.4 at every N; stacked nearly unbiased
    from N = 4; at N = 32, m1 −0.015 [−0.020, −0.010], m2 −0.001
    [−0.007, +0.004].
  - Bib: BA14 (Bernstein & Armstrong 2014) and Scott (1992 book).
    fig_stacking copied to `figs/`.
  - Closing: constant shear and the training prior are idealized; the BA14
    varying-shear extension with the TF prior is future work ("most
    promising direction").

  6.5 round 1 (backup 65v1):
  - The KDE is custom PyTorch code (`c_stack_analysis.py: kde_density`), so
    the Scott citation and bib entry were removed and the bandwidth
    σ_k n^{-1/6} is written out.
  - Bernstein & Armstrong named at the end of paragraph 1 and in the
    outlook.
  - New `tab:stacking` (N, m1, m2; bootstrap median with 16–84%); the text
    quotes m1 −0.015 ± 0.005 and m2 −0.001 (+0.005 / −0.006).
  - BA14 Sec. 2.2 explained to Eason in chat (P_i, Q_i, R_i Taylor
    coefficients; shear model g_i(a); Fourier example). The text keeps the
    vague wording at his request.

  NEXT: Eason reviews 6.5 → Discussion (sec:discussion: local calibration,
  Xu24 shape-noise comparison, limitations, future work) → Conclusions →
  appendices (app:corner, app:coverage).
- (history) Tests launched 2026-09-29 (four background subagents: A_shear,
  A_npe, B_inference, C_stacking), writing to `$S/reports/paper-tests/<group>/`
  and `paper-tests/paper_figs/`, with scripts in
  `kl-nn/arch/diagnostics/paper_tests/`. The main session compiles
  `paper-tests/index.html` and adds a row to `$S/reports/README.md` when
  they finish. If this session was lost, check each folder's `STATUS.txt`
  and `report.json` (and `squeue -u xwang30`) to see what finished.
- (original plan) run the tests in `PAPER_TESTS_PLAN.md`.
  - Four subagents (Group A shear tests; Group A NPE validation; Group B
    GPU R2b/T2b; Group C GPU T8 NPE stacking).
  - Output to `$S/reports/paper-tests/` with one `index.html` HTML report.
  - Eason approved running them with subagents; nothing has been launched
    yet.
  - After the report: review results with Eason, then write Sec. 5 and 6
    one subsection at a time.
- (The old Sec. 5–7 plan in "Planned draft structure" below is superseded
  by `RESULTS_OUTLINE.md`.)

**Before writing each subsection:**
1. **Verify every simulator/model fact in code or config yourself.** Don't
   trust agent summaries: an agent's "thin disk, no q_z" claim was wrong,
   since GalSim's default `scale_h_over_r = 0.1` applies.
2. **Check Eason's existing prose for text copied from earlier KL papers.**
   Download the arXiv LaTeX sources (`curl -L -A Mozilla/5.0
   https://arxiv.org/e-print/<id>`; ids 1311.1489, 2201.00739, 2209.11811,
   2404.00197, 2407.20867, 2409.08367, 2601.05322, 2604.26015). Strip math
   and compare 6-word shingles. Equations may match if cited; prose must be
   rephrased. Xu+24 (2407.20867) is the usual source, e.g. its Sec. 3
   simulator description and observation table, which the old
   `data-generate.tex` likely borrows from.
3. Fetch any new BibTeX from INSPIRE
   (`https://inspirehep.net/api/literature?q=arxiv:<id>&format=bibtex`).
   `main.bib` got 14 entries on 2026-09-28: KLCluster, D4CNN, NPEShear,
   SBIGuide, DESY6CS, DESI1/2, DESC_SRD, Roman_AFTA, HLIS-*, Euclid, spherex,
   DES_Y3_3x2pt.
4. REVTeX isn't installed locally, so the draft can't be compiled here.
   Check citation keys against `main.bib` by grep.

**Sec. 3.1 notes gathered so far.**
- Old `data-generate.tex` has an obs-config table (image 378.2856 cm
  diameter, 0.2637″/px, 1.0″ PSF, gain 4, 60 s; fiber 322.42 cm, gain 1,
  0.08 nm/px, 180/600 s, r = 0.75″, offset 1.5″). Verify each value against
  `data_generate/generate_fits.py` and Xu+24's table.
- Continuum is the GSB2 template (`generate_fits.py` ~L322), normalized to
  rmag_true via DECam r. Old text says "barred spiral"; check what GSB2 is.
- Only Hα is used (σ = 0.065 nm). Other lines are set to 0 in
  `configure_sed`; verify.
- The fiber-placement paragraph must be rewritten. Axes come from the
  observed map with a right-handed basis (`observation_schema.py:
  compute_fiber_offsets`), replacing the SVD sign-flip text. Eason decided
  not to discuss the fiber hop or the gauge bug in the paper.


Purpose: carry context between Claude sessions helping Eason write the first
complete draft of the KL-NN paper. The draft is for a discussion with advisors
(Tim Eifler et al.), not a final submission. **Gaps are marked with in-text
comments** (the Overleaf project defines `\XW{}` (red), `\tim{}`, `\JX{}`,
`\EK{}`, `\PRS{}`, `\rachel{}`; ask Eason which macro to use for
Claude-flagged gaps, defaulting to `\XW{}`).

- Overleaf clone: `/jet/home/xwang30/Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks`
  (`main.tex`, `sections/*.tex`, `main.bib`, `KL_NN.rtf` = extra reading list).
  REVTeX/PRD template is temporary, for preview only; no journal chosen.
- Best current methods description: `kl-nn/repo_reports/METHODS_MANUSCRIPT.md`
  (+ `.tex`). Mostly accurate for v3; stale S/N supports (says 5–1000 / 1–200;
  config is image 10–1000, Hα 1–150).
- Reports: `/ocean/projects/phy250048p/shared/reports/` (campaign folders; see
  its README). Latest status: `meeting-reports/advisor_summary.html`.

## Decisions Eason made (2026-09-28)

1. **Framing:** the NPE is presented as an alternative KL inference method that
   is actively being developed. No DESI-PV pilot measurement; real DESI data is
   out of scope. Drop `sections/desi.tex` and `sections/desi-kl.tex` content
   (already not included in `main.tex`); at most a sentence in the intro.
2. **Honors thesis** (UA 2025, same title) is Eason's; do not cite it.
3. **Paper model = trained on 1M, validated on 100k.** Not yet trained for the
   current architecture. Existing canonical numbers come from the 100k-train /
   10k-valid run; write the draft with these and flag with comments.
4. **Results must show under which constraints m is acceptable.** The
   posterior-width cut is the most promising; also inclination (cos i) cuts.
   Bayesian shrinkage goes in the **Discussion** as the explanation, not as an
   excuse. The current model is not usable for most galaxies without cuts.
5. **Likelihood stacking / NRE: leave out of this draft.** Eason will raise it
   with advisors. It may enter later (with the BA14 non-constant shear
   formulation and a much larger benchmark), possibly framed as a bias vs
   inference-time trade-off (NRE ~100x slower than NPE).
6. **Architecture:** ablations show no result differences, but justify the
   current design physically in Methods: sample cos i (uniform for isotropic
   orientations); bounded flow support matching the training prior; circular
   coordinate for θ_int; R90 symmetrization keeps additive bias stable.
   Ablations are at most a brief mention.
7. **Fiber hop: out of scope.** An earlier model with fibers fixed to the image
   axes gave similar m and shape noise. (It is an open question for the local
   miscalibration only.)
8. **Response calibration (R(σ)):** allowed in the paper, or briefly motivated in
   the Discussion.
9. **MCMC comparison:** only the 4 galaxies in
   `/jet/home/xwang30/mcmc_xu1_four_fits` (a collaborator runs them). Use them for
   corner-plot posterior comparisons. Not a statistical sample. Note: those FITS
   are from the old sin i xu1 set (`test_100k_simv3_xu1_tf`), which may need
   matching to the cos i model.
10. **Conventions in the paper follow standard usage, not kl-tools.** θ_int is
    counterclockwise-positive in the paper, while kl-tools and the kl-nn
    code are clockwise-positive. Disks have q_z = h_s/R_s = 0.1 (GalSim
    default, same as Xu+24). Progress and gaps are logged in
    `kl-paper-gaps.md` next to this file. Intro and Sec. 2 were drafted and
    reviewed on 2026-09-28.
11. Exclude: Fisher bound (almost certainly a normalization bug:
    σ_CR ~ 2e-5 at S/N=10), MAP/Laplace, g-prior-width (pre-repair, invalid),
    likelihood-stack diagnostics, fiber-gauge physics.

## Canonical model (use this, not 45255702)

`CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair`
(CCL parent `CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k_s42_fibrepair`).
Config: `$S/configs/cfg_<name>.json`, `$S = /ocean/projects/phy250048p/shared`.

- 9 targets: g1, g2 ~ U(±0.1); θ_int ~ U(−π, π) (circular); cos i ~ U(0, 1);
  v0 ~ U(±30) km/s; vcirc ~ U(60, 540); rscale, hlr ~ U(0.1, 5)″;
  Hα flux log-uniform 1e-17 to 1e-14. Fixed z = 0.3, thin disk (no q_z),
  arctan rotation curve, GSB2 continuum normalized to rmag, Hα only,
  σ_λ = 0.065 nm.
- Image 48×48 at 0.2637″/px, PSF 1.0″. 5 fibers (r = 0.75″, ±1.5″ offsets on
  observed major/minor axes, right-handed), DESI z-arm 851–855.81 nm,
  61 bins at 0.08 nm padded to 64. Exposures 180 s centre / 600 s offset.
- Noise added at runtime: image S/N ~ U(10, 1000), central Hα S/N ~ U(1, 150),
  offset fibers σ × √(600/180). Metadata = 10 fiber coords + rmag + 2 S/N (true values).
- Encoder: ResNet image branch (512), JointSpecCNN (512; pooling after the
  16/32 pairs only, final 5×16 conv), metadata MLP (128). Concat 1152 → LayerNorm.
  No FiLM in canonical (`use_image_spectrum_fusion: false`).
- CCL (continuous-label contrastive) pretraining, then frozen encoder.
- Flow: bounded box flow (4 masked autoregressive RQ-spline layers, uniform
  base on [0,1]^8) plus a circular RQ spline for θ_int. Unweighted NLL. R90 view
  augmentation; inference = equal mixture of identity and R90 draws.
- TF prior is **post-hoc importance reweighting** (not in the loss):
  log10 vcirc ~ N((r−36)/(−7.22), 0.1 dex), truncated to [60, 540]. No σ_m term.
- Training: AdamW, 3e-4 (box, shared) / 1e-4 (θ), wd 1e-5, warm-up + cosine,
  batch 400 global, 200 epochs, 4×V100. About 1 h CCL + 1 h NPE on the 100k set.
- Inference: ~0.33 s per galaxy (10k draws + TF reweight, 1 V100).

The draft's `data-generate.tex` and `nn-arch.tex` are outdated on nearly every
item above (8 params / sin i / fixed Hα flux / 12 affine MAF layers with a
Gaussian base / TF in the loss / 32 GPU-h / MAP estimator / figure caption).
Rewrite them from METHODS_MANUSCRIPT.md and the config.

## Key numbers (canonical fibrepair model, Xu3 TF test set, 100k)

Source: `$S/reports/fiber-gauge/03_g01_retrain/report.html`.
- Low-|g| (<0.02) m = −0.512 ± 0.016 / −0.486 ± 0.015; 10⁴c = 0.44 ± 1.41 / 0.81 ± 1.40.
- σ_shape (regularized) 0.0389; ESS 90,897. Coverage g1/g2 67.4/67.8%;
  all 9 params within 2.5 pp of 68%.
- Local coverage: 88% for |g| < 0.05 vs 62% for |g| > 0.05
  (`likelihood-stack/02_coverage`). A correct posterior under a bounded prior
  can also show this; treat it as an open "local miscalibration" question.

Cuts (legacy 45255702, `xu3-estimators/xu3_baseline100k_selection_m_ess_s42`),
to be **re-run on the fibrepair model**; units 10⁻²:

| Selection | m1 | m2 | σ_shape | low-\|g\| ESS |
|---|---|---|---|---|
| none (regularized weights) | −51.2 | −48.4 | 0.039 | — |
| cos i > 0.3 / 0.5 / 0.7 | −38.7 / −28.0 / −14.4 | −36.4 / −26.2 / −12.9 | 0.034 / 0.030 / 0.025 | 12.8k / 9.0k / 5.3k |
| σ⊥ < 0.075 | −36.2 | −33.6 | 0.033 | — |
| **σ⊥ < 0.05** | **−2.8 ± 3.0** | **+0.1 ± 2.9** | 0.022 | 4,135 (20.7%) |
| r < 19 & Hα S/N > 25 | −34.6 | −30.8 | 0.033 | — |

- Shrinkage evidence (Discussion): R(σ²) slope −297 vs −1/σ_π² = −300
  (σ_π = 0.0577); stencil response edge-on 0.36, face-on 0.80.
- R(σ) calibration after the σ cut: m ≈ −8.3 / −4.3 ×10⁻², σ 0.052, +9 pp
  over-coverage, keeps 68%. Not a production estimator.
- Xu1/3/5 three-sample numbers exist only for the old sin i model 44603007
  (`desi-tf-testset/`): m ≈ −0.50/−0.46, −0.48/−0.48, −0.41/−0.44.
- Shape-noise comparison with Xu+2024 (σ_ε = √(σ1² + σ2²): samples 1/3/5 =
  0.071 / 0.047 / 0.041). Our per-component posterior σ 0.039 gives 0.055 in
  quadrature; if prior-deconvolved (σ_L ≈ 0.053) it gives about 0.075. Which is
  fair depends on Xu's prior width. Flag it; don't claim.

## Provenance traps

- The `45255702` model folder was **overwritten in place** by the pairs
  continuation (job 45467143). Reports after about 9/8 labelled 45255702
  (e.g. map1d_identity) are actually the pairs weights.
- Legacy-gauge models (1m, better-spec, 45255702) must not be compared with
  repaired test sets.
- `datasets/train_1m_simv3_cosi` was **not** in the fiber repair audit
  (`reports/fiber-gauge/01_repair_audit/verify.json`), so it is probably still in
  the legacy gauge. Repair it before the 1M training. The Xu1/Xu5 cos i test sets
  were not audited either.
- Stored FITS were not repaired (`skip_fits=True`); only the LMDBs were.
- The NRE fixed-shear benchmark snaps to its grid (21×21 grid, truths on grid
  nodes). Irrelevant for this draft, but important when NRE comes back.

## Planned draft structure (first pass)

1. Abstract / Intro: rewrite. WL shape noise → KL (Huff+13, KL-I, KL-Roman,
   KL-DESI). MCMC cost (KL-I: ~3e6 likelihood calls per galaxy; KL-II: O(1)
   galaxies with nested sampling). NN shear methods need explicit calibration
   (FORKLENS, D4CNN×AnaCal, White+2607.09867 sees the same shrinkage). SBI
   refs (Deistler+2508.12939). To our knowledge no prior NN/SBI work for KL.
2. KL basics: keep and clean up.
3. Data generation: v3 simulator, runtime noise, normalization, 1M/100k
   datasets (comment: not yet trained), Xu1/3/5 TF-conformed test sets.
4. Architecture: encoder, CCL pretraining, bounded+circular flow, R90,
   post-hoc TF reweighting, training config.
5. Results: loss; example posteriors + 4-galaxy MCMC corner comparison
   (placeholder); coverage / P–P; inference timing vs MCMC; m, c, σ_shape for
   the full sample and under σ-width and cos i cuts; shape-noise comparison.
6. Discussion: Bayesian shrinkage explanation; local calibration;
   response-calibration option; limitations (z fixed, Hα only, thin disk,
   Gaussian noise, oracle metadata, TF assumed known).
7. Conclusions.

## Reference notes (for citations)

- BA14 (1304.1843): averaging per-galaxy posterior means is biased; ensemble
  likelihood product. Cite in the Discussion for shrinkage.
- KL-I (2209.11811): σ_ε^KL 0.022–0.041; emcee 100 walkers × 30k steps;
  m1 = −0.05 ± 0.06. KL-II (2409.08367): 3 galaxies behind A2261.
- Xu+2024 DESI-KL (2407.20867): samples above; assumes σ(m) = 4e-4.
- DESI PV TF (2512.03227): slope −7.22, scatter 0.466 mag (~0.065 dex).
  Reyes+11: intrinsic 0.036 dex.
- FORKLENS (2301.02986): raw m ≈ −2.2% → −0.04% with learned weights.
  D4CNN×AnaCal (2603.19046): |m| < 2e-3.
- White+ NPE shear (2607.09867): posterior-mean m ≈ −0.7 to −3.4%, attributed
  to prior shrinkage. Closest SBI-for-shear comparison; not yet in main.bib.
- Requirements: Stage-IV |m| ≲ 2e-3 (Massey+13); LSST SRD σ(m) 0.013 (Y1) /
  0.003 (Y10); DES Y6 m priors ~0.6%.

- **2026-10-02: figure naming in Overleaf.** The `fig_` prefix was dropped:
  `paper_figs_final/fig_X.pdf` → Overleaf `figs/X.pdf` (training_loss,
  posterior_snr, pp_shear, pp_nuisance, raw_bias, shrinkage, cuts, response,
  stacking). The generator scripts still write `fig_*`, so rename when
  copying. The architecture figure is `figs/nn_arch.pdf`, from the TikZ
  `figures_src/nn_arch.tex` (the v3 colour scheme Eason chose: darker
  blocks, lighter inner boxes); old variants are in `figures_src/old/`.

## 2026-10-03: post-meeting results campaign (plan: ~/.claude/plans/there-were-so-many-fuzzy-snowglobe.md)
The meeting agreed that likelihood stacking is the way forward and should get
a fuller exploration in the paper. There's no time pressure now.

**Step 1, pretraining test (Stage A).**
- New option: `PRETRAINED_NAME=none` trains from scratch (`config.py`,
  `train_model.py`, `train.py: load_train_objs`, `train_npe.slurm`). It
  requires `FREEZE_FEATURE_EXTRACTOR=0`.
- Jobs:

  | Run | Model | Train job | Xu3 cache job |
  |---|---|---|---|
  | S1 | `..._valid100k_scratch_s42` | 47390724 | 47390725 (chained) |
  | S2 | `..._valid100k_scratch_s43` | 47390726 | 47390727 (chained) |

- Cache seed is 42 for both, so the noise is paired with the canonical cache.
- Compare against the canonical frozen-pretrained model.

**Step 2, `measured_axis` fiber layout.**
- New `data_generate/fiber_targeting.py`:
  - targeting noise uses LS DR9 r-band `galsim.CCDNoise`, seeded with 9e8 + ID;
  - the position angle comes from HSM adaptive moments, with a random
    fallback flagged as TGTHSMOK = 0;
  - fibers go along the measured axes, with β ∈ (−π/2, π/2] and the minor axis
    90° counter-clockwise.
- `generate_fits.py` now uses `build_obs_conf(offsets)`.
- Two passes: pass 1 renders the centre fiber plus the image, then measure,
  then pass 2 renders everything. Cost is about 1.6× per galaxy, and the code
  asserts the image is unchanged between passes.
- Layout code 2 is wired through `observation_schema` (FIBER_LAYOUT_CODES),
  the wrapper (`--fiber-layout`), and the arch config/train
  (`FIBER_LAYOUT=measured_axis` in `train_npe.slurm`).
- Tests were updated and 130 pass.
- The 3-galaxy check passed: images bit-identical, measured PA within 0.3° of
  the true observed axis, targeting SNR close to the catalog SNR.
- New CSVs use the same rows: `samples/*_measaxis.csv`.

  | Dataset | Jobs (gen / db / merge) |
  |---|---|
  | `valid_100k_simv3_cosi_measaxis` (train) | 47390965 / 966 / 967 |
  | `small_10k_simv3_cosi_measaxis` | 47390968 / 47391077 / 078 |
  | `test_100k_simv3_cosi_xu3_tf_measaxis` | 47391079 / 080 / 081 |
  | `paper_tests_B_shear_response_xu3like_5k_measaxis` (stencil) | 47391082 / 083 / 084 |

- CHUNK_SIZE is 1000.
- Pilot FITS are in `fits/dev_fiberpilot_{galaxy_axis,measured_axis}`
  (3 galaxies; safe to delete).

**Next steps:**
- Step 3: train the Stage A winner on measaxis data with
  `FIBER_LAYOUT=measured_axis`, compare pair by pair, then decide with Eason.
- Step 4: annulus (cluster) stacking on the Xu3 cache, reinterpreting
  |g| as g_t.
- Step 5: robustness to misspecified m_r and SNR.
- **2026-10-03 storage cleanup (Eason approved).**
  - Deleted the per-epoch checkpoints of 34 finished models that have both
    `best` and `best.json`: 4982 files, 816 GB. Only `<name>best` and
    `best.json` are kept, so evaluations at specific non-best epochs are no
    longer possible for these models.
  - Every deleted file is listed in
    `repo_reports/model_checkpoint_cleanup_2026-10-03.log`.
  - Free space on /ocean went from 0.83 TB to 1.6 TB; the models folder went
    from 859 GB to 99 GB.
  - Round 2 (Eason approved): removed 23 old or empty folders (15 old
    simulator-v2, SetAttn and ViT experiments plus 8 empty folders; 99.8 GB).
    Kept the 3 NRE models and `losses/`. Also logged in the cleanup log.

### RESUME HERE (2026-10-03, end of session before /compact)
**Jobs to check:**
- **Stage A from-scratch trainings:** 47390724 (s42) and 47390726 (s43), still
  pending for GPUs. Their Xu3 caches 47390725 and 47390727 are chained, with
  tag `testset_tfweighted_v2_10k_s42_righthanded` and cache seed 42.
- **Measured-axis FITS generation:** 196/215 tasks done at 20:00.
  - DB and merge for `valid_100k_simv3_cosi_measaxis` (47390966/67) and
    `test_100k_simv3_cosi_xu3_tf_measaxis` (47391080/81) wait on generation.
  - `small_10k_simv3_cosi_measaxis`: shards 2, 3, 7, 8, 9 failed with a bus
    error (transient Lustre fault during the 900 GB deletion; RSS was only
    330 MB). They were rerun as individual `--wrap` jobs 47394771–75 (2 cpus),
    with merge 47394776 chained.
  - The stencil `paper_tests_B_shear_response_xu3like_5k_measaxis` merge failed
    the same way; the partial target was deleted and the merge rerun as
    47394765.
  - Note: `make_database_simulator_v3.slurm` refuses partial arrays, since it
    requires the array count to equal the part count. Retry single shards with
    `--wrap` as above. If the big DB jobs bus-error, do the same.

**Step 4, annulus stacking.** Code is in `arch/diagnostics/lensing_profiles.py`
(NFW following Wright & Brainerd 2000, Planck18) and
`paper_tests/f_annulus_stack.py` plus `.slurm`. Results are in
`$S/reports/annulus-stack/canonical_{synthetic,tf}_v2/summary.json`. The v1
cross test was invalid because its line went through g = 0.

- **Setup:** M200 = 1e15, c = 4, z_l = 0.1, z_s = 0.3, R = 0.1–3 Mpc in 8 log
  annuli. g_t runs from 0.061 to 0.008, with 568–6379 Xu3 galaxies per annulus
  (all available; counts are not area-weighted).

| Run | m | c | Notes |
|---|---|---|---|
| Synthetic check | −0.004 ± 0.017 | +0.0005 ± 0.0008 | — |
| Canonical model, TF | −0.025 ± 0.010 | +0.0015 ± 0.0005 | about +0.001 high at mid radii (annuli 2–5, 3–4σ each); innermost annulus slightly low |

- In both runs the posterior width matches the bootstrap scatter, and the
  cross null test is consistent with zero (|x̂| ≤ ~2σ).
- To check: whether the positive offset is the fiber leak (compare with the
  measured-axis model in Step 3).

**Not yet started:** Step 3 (needs the Stage A winner and the measaxis DBs)
and Step 5 (robustness to misspecified m_r and SNR).

### Update (2026-10-03, ~20:45): bus errors were systematic, now fixed
- **Cause:** not a one-off. LMDB maps `lock.mdb` as a shared writable mmap,
  and on /ocean (Lustre) page faults on it intermittently raise SIGBUS.
  Reproduced interactively: opening an intact shard with `lock=True` crashed
  about half the time, `lock=False` never did. Training (`pxt.TorchDataset`)
  already uses `lock=False`.
- **Fix:** `data_generate/make_database.py` opens shard readers with
  `lock=False` and creates both writers through `_open_writer`, which skips
  the lock file (each LMDB has one process at a time). Backup in
  `draft_backups/make_database.before_nolock.py`.
- Eason deleted the 28 crash leftovers with
  `data_generate/cleanup_measaxis_partials.sh`.
- `small_10k_simv3_cosi_measaxis` (10000) and
  `paper_tests_B_shear_response_xu3like_5k_measaxis` (5000) are merged and
  checked.
- Reran the failed shards as `--wrap` jobs: valid_100k shards 28, 29, 43, 46,
  47 (47395686–90), Xu3 test 21 shards (47395691–713). New merges chained:
  valid 47395714, test 47395715 (the old 47390967/47391081 were cancelled).
- **20:58: all four measured-axis LMDBs are complete and checked** (fiber_layout
  code 2): `valid_100k_simv3_cosi_measaxis` 100000,
  `small_10k_simv3_cosi_measaxis` 10000,
  `test_100k_simv3_cosi_xu3_tf_measaxis` 100000,
  `paper_tests_B_shear_response_xu3like_5k_measaxis` 5000. 13 valid_100k
  shards needed rebuilding in total (the first sacct check was truncated); the
  final merges ran in the foreground. Note: submit `merge_database_simulator_v3.slurm`
  from the repo root (it checks `SLURM_SUBMIT_DIR`).
- Stage A time limits lowered (Eason): training 47390724/26 to 5 h, cache
  arrays 47390725/27 to 1 h per task.

### Stage A result (2026-10-04)
S1/S2 trained in ~1h55m each; caches ~5 min/task. Reports:
`$S/reports/stageA/shear_bias_{weighted,unweighted}` (HTML, cases in order
canonical, S1, S2) and `$S/reports/annulus-stack/scratch_s4{2,3}_tf_v2`.

| | canonical (CCL frozen) | S1 scratch s42 | S2 scratch s43 |
|---|---|---|---|
| best valid NLL | −17.80 | −17.92 | −17.85 |
| σ_shape (weighted) | 0.0389 | 0.0397 | 0.0398 |
| raw m g1/g2 (cubic, weighted, %) | −49.7/−49.7 | −52.5/−51.8 | −51.9/−50.8 |
| 68% coverage g1/g2 | 67.4/67.8 | 67.6/68.2 | 67.9/68.1 |
| 68% coverage θ_int | 65.5 | 62.5 | 61.8 |
| 68% coverage v0 / Hα flux | 67.9/67.8 | 70.1/70.3 | 69.9/70.3 |
| annulus m | −0.025 ± 0.010 | −0.043 ± 0.011 | −0.028 ± 0.012 |
| annulus c | +0.0015 ± 0.0005 | +0.0019 ± 0.0005 | +0.0024 ± 0.0005 |

By the plan's rule the pretrained model wins (σ_shape and θ_int/nuisance
coverage better than the S1–S2 spread), although scratch has lower valid NLL.
Open question for Eason: Step 3 with the pretrained config needs a CCL encoder;
reuse the existing galaxy_axis CCL parent or retrain CCL on measaxis data.
- **Eason (2026-10-04): settled on CCL pretraining; option 2 for Step 3.** The
  paper (Sec. 4) must describe the CCL stage.

**Step 3 submitted (2026-10-04):** retrain CCL on measured-axis data, then
frozen NPE on top, all with canonical settings, seed 42.
- `arch/pretrain_ccl.slurm` now passes `FIBER_LAYOUT` (default galaxy_axis);
  backup in `draft_backups/pretrain_ccl.before_fiberlayout.slurm`.

| Stage | Model | Job | Limit |
|---|---|---|---|
| CCL | `CNN-CNN-Meta-CCL-simv3-cosi-r90_valid100k_s42_measaxis` | 47420661 | 3 h |
| frozen NPE | `CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_measaxis` | 47420662 | 5 h |
| Xu3 cache | dataset `test_100k_simv3_cosi_xu3_tf_measaxis`, tag `testset_tfweighted_v2_10k_s42_righthanded` | 47420663 (100 tasks) | 1 h each |

- Still to do after the NPE: stencil cache on
  `paper_tests_B_shear_response_xu3like_5k_measaxis` (b_t2b scripts) for face-on R;
  shear_bias_report vs canonical; annulus stack; validation NLL.
- **2026-10-04: CCL restored in `sections/nn-arch.tex`** as Sec. 4.2 "Pretraining
  the Feature Extractor" (from the shelved `pretrain-ccl.tex`), with the Stage A
  evidence replacing the old \XW, a CCL paragraph in Training Configurations
  (AdamW lr 1e-3, wd 1e-4, 5-epoch warmup from 1%, cosine to 1e-6, 100 epochs,
  batch 400, best validation epoch, ~1 h), and the figure caption updated.
  Backup: `draft_backups/nn-arch.2026-10-04-before-ccl.tex`. The shelved
  `pretrain-ccl.tex` is now redundant (not \input; duplicate labels only).

### Step 3 result (2026-10-05): measured-axis fibers vs canonical (true-axis), both CCL-frozen s42, 100k/10k
Reports: `$S/reports/stepB_fiber/` (shear_bias_weighted HTML, t2b_xu3like_measaxis.json + cubes),
`$S/reports/annulus-stack/measaxis_tf_v2`. Xu3 cache needed a manifest (written from the parent,
sha256 of the measaxis CSV; rows identical except fiber_layout).

| | canonical | measured-axis |
|---|---|---|
| CCL best valid loss | 4.352 | 4.396 |
| NPE best valid NLL | −17.80 | −16.64 |
| σ_shape (weighted) | 0.0389 | 0.0391 |
| raw m g1/g2 (cubic, weighted) | −49.7/−49.7% | −50.7/−50.4% |
| 68% coverage g1/g2 | 67.4/67.8 | 67.2/67.2 |
| 68% coverage θ_int / cos i | 65.5/67.6 | 71.2/66.5 |
| annulus m | −0.025 ± 0.010 | −0.011 ± 0.011 |
| annulus c | +0.0015 ± 0.0005 | +0.0004 ± 0.0005 |
| stencil R overall (R11/R22) | 0.42/0.41 | 0.41/0.39 |
| stencil R, cos i 0.8–1 | 0.93/0.97 | 0.86/0.87 |
| per-galaxy R, cos i > 0.9: mean / frac R > 1.5 | 1.13 / 14% | 0.92 / 4% |
| R(σ)=a+bσ² intercept a (pred. 1) | 1.19 ± 0.02 | 1.09 ± 0.02 |

Reading: the NLL rise is mostly θ_int (the true-axis fibers leaked orientation); shear
performance is unchanged; the annulus c offset and most of the face-on R > 1 anomaly go away,
supporting the fiber-leak hypothesis. Decision pending with Eason.

Also 2026-10-05: stacking-intuition figure `$S/reports/paper-tests/C_stacking_shear2d_v2/`
(`paper_tests/f_stack_shear2d.py`): T8 cell 15, galaxies 1–4, individual KDE contours + filled
product. Each likelihood is tight in a different direction and passes through the truth; means
are pulled by the box; product (0.0395, −0.0797) vs truth (0.04, −0.08). Eason liked it.

### 2026-10-05: measured-axis placement adopted (Eason); 1M generation + 100k ablations started
**Decision:** use `measured_axis` fiber placement from now on (paper model and numbers).
**Still open with Eason:** regenerate the T8 constant-shear benchmark with measured axes, or
lead 6.5 with the annulus test (T8 regenerated either way, for consistency).

**1M training set** (validation reuses `valid_100k_simv3_cosi_measaxis`, verified disjoint
from `train_1m_simv3_cosi`): `samples/train_1m_simv3_cosi_measaxis.csv` = the 1M rows with
fiber_layout=measured_axis (floats exact). CHUNK_SIZE 2000, 500 parts.
gen 47443563 (array 1-500) → db 47443565 (1-500) → merge 47443566. ~1,900 core-h expected;
~47 GB FITS + ~23 GB LMDB. If DB tasks fail, rebuild single shards with `--wrap` (see above).

**Ablations (100k measured-axis data, measured-axis CCL frozen, Xu3 caches seed 42):**
New option `r90_views` (config.train; `--r90-views/--no-r90-views`; `R90_VIEWS=0` in
train_npe.slurm) disables the random 90° view in NPE training/validation. Backups
`draft_backups/*.before_r90views.*`; new test `test_npe_r90_views_off_keeps_every_row_unrotated`.
Test suite: 491 pass, 5 pre-existing unrelated failures (NRE stage list, methods-manuscript
path, oracle-cache fixtures).

| Run | Model suffix | Train | Cache (tag) |
|---|---|---|---|
| seed replicate | `_frozen_s43_measaxis` | 47444071 | 47444072 (righthanded) |
| R90 train off | `_frozen_s42_measaxis_nor90` | 47444073 | 47444074 (righthanded), 47444075 (identity) |
| logit limit 20 | `_frozen_s42_measaxis_logit20` | 47444077 | 47444079 |
| single θ lr 3e-4 | `_frozen_s42_measaxis_thetalr3e-4` | 47444095 | 47444096 |
| R90 inference off | baseline `_frozen_s42_measaxis` | — | 47444101 (identity) |

Tags: `testset_tfweighted_v2_10k_s42_righthanded` (identity+R90) and
`testset_tfweighted_v2_10k_s42_identity`. Compare with Stage A metrics + annulus stack;
keep canonical unless a variant beats the s42/s43 spread. Check logit20's training log
for whether spline logits still saturate.

**6.5 plan (Eason, 2026-10-05):** the constant-shear T8 result shows that likelihood stacking
works; the annulus test shows a realistic application. Paper discussion waits until all
results are in.
- T8 benchmark regenerated with measured axes (same 800 rows, same noise seed 420042):
  `samples/likelihood_stack_04_benchmark_measaxis.csv` (+ `.benchmark.json`), dataset
  `likelihood_stack_04_benchmark_measaxis`. Jobs: fits 47444197 → db 47444198 → merge
  47444199 → noise 47444200. `materialize_likelihood_stack_benchmark.py` gained
  `--fiber-layout` (FIBER_LAYOUT env in the noise slurm; backups in draft_backups/).
- Inference/analysis (`paper_tests/c_stack_infer.py`, `c_stack_analysis.py`) hard-code the
  old MODEL/DATA/SAMPLES paths: parameterize them when the 1M measured-axis model is done.

### 2026-10-05 evening: 1M set done; two bugs found and fixed
- `train_1m_simv3_cosi_measaxis` complete: 1,000,000 records, layout code 2, 23 GB, no shards left.
- T8 measured-axis data complete: `likelihood_stack_04_benchmark_measaxis` (800 records) with
  `benchmark_noisy.npz` in its own folder.
- **Bug 1 (mine): `r90_views` broke loading of every older model config** (strict schema).
  Fixed in `model_registry._payload_for_current_schema` (missing → True, which is what all
  older runs used); test `test_load_model_config_defaults_missing_r90_views_to_on`. The
  baseline identity-only cache 47444101 had failed on this; resubmitted as 47454317.
- **Bug 2 (mine): the T8 noise step wrote into the OLD benchmark folder** (output path was
  hard-coded to the default dataset, and I passed --overwrite), overwriting the original
  galaxy-axis `benchmark_noisy.npz`. Fixed `materialize_likelihood_stack_benchmark.py` so
  --output/--manifest default to --data-dir. Recovered: regenerated the measured-axis file into
  its own folder (bit-identical to the misplaced one, proving the step is deterministic), then
  regenerated the original galaxy-axis file in place. Noise code is unchanged since the commit
  that produced the original, so the restored file equals the original. Old T8 results (sample
  cache) were never affected.
- Test suite: 492 pass, same 5 pre-existing unrelated failures.
- **1M CCL pretraining started (Eason, 2026-10-05):** `CNN-CNN-Meta-CCL-simv3-cosi-r90_1m_s42_measaxis`, job 47454366 (12 h limit; past 1M
  CCL runs took 7–9 h wall on 4 V100s). Train `train_1m_simv3_cosi_measaxis`, valid
  `valid_100k_simv3_cosi_measaxis`, measured_axis, seed 42, canonical CCL settings. R90 stays
  in pretraining regardless of the NPE-level R90 ablation (option 1).

### 2026-10-06: measured-axis placement written into Sec. 3.1 (`sections/data-generate.tex`)
Replaced the true-axis T=GRP/SVD text with the targeting procedure (SGA/DESI-PV motivation,
targeting image with LS r-band sky 44.54 e-/s/pix and read noise 2.6 e- added to Table obs-conf,
adaptive moments, β ∈ (−90°, 90°], minor axis 90° CCW, 0.3% HSM failures with random β).
Numbers from 10k Xu3 measured-axis galaxies: targeting SNR / catalog ρ_I median 1.00, 90% in
0.50–1.81; PA error median 0.21° (cos i < 0.5), 2.0–2.6° (cos i > 0.8). New bib keys SGA2020,
Mandelbaum2005. Backups: draft_backups/data-generate.2026-10-06-before-measaxis.tex,
main.2026-10-06-before-measaxis.bib.
- 2026-10-06: targeting failure modes (all 100k Xu3 measured-axis; table in scratch, recompute
  from FITS headers TGTPA/TGTHSMOK/TGTSNR): HSM fails for 0.34%, all cos i < 0.5, median
  R_h 4.0" (fail rate 0% for R_h<2", 2.4% at 3–4", 7.5% at 4–5"): large edge-on disks truncated by
  the 12.7" stamp. Errors >10° among successes: 2.0%, median cos i 0.91, median targeting SNR 93
  (cos i ≥ 0.8: 13% at SNR 50–150 vs 3% at SNR > 150). Both now stated in Sec. 3.1; nn-arch R90
  sentence now says "measured axes". Eason: no paper mention needed for the independent
  targeting/image noise, nor for the fiber-order convention under R90.
- **Correction (2026-10-06): HSM failures are NOT truncation.** Edge flux is only 1–2%, a
  GalSim analog on a 192-px stamp fails the same way, and even a pure elliptical Gaussian (no
  PSF) fails when large (hlr ≳ 2"), elongated (q ≲ 0.2) and oblique to the pixel grid (fine at
  0/45/90°). Cause: FindAdaptiveMom starts from a round 5-px weight (guess_sig default); each
  update is capped at 0.25 × the weight's minor-axis variance, which shrinks as the weight
  elongates, so the major axis can't grow ("too many iterations"/NaN). Pixel scale, tolerance,
  max_iter, max_amoment don't help; guess_sig ≈ major sigma converges in 18 iterations.
  Retrying the 335 Xu3 failures with guess_sig=10 px recovers all 335 (PA error median 0.07°,
  max 0.75°). Paper sentence corrected. Decision pending: fix `measure_position_angle` (retry
  with larger guess_sig) and regenerate the affected ~0.3% of galaxies, or keep as is.

### 2026-10-06: targeting fix in progress (Eason: fix before any training)
- Held all pending trainings: 1M CCL 47454366 and ablations 47444071/73/77/95 (`scontrol
  release` when the rebuilt LMDBs are in place). Their caches wait on them by dependency.
- `fiber_targeting.measure_position_angle`: default HSM start first (unchanged for galaxies that
  converged), then guess_sig 10/20/40 px, then (Eason's choice) unweighted second moments about
  the stamp center as a noisy fallback; random β only for a blank image. TGTHSMOK=1 means HSM
  converged, 0 means the unweighted fallback. Tests in `tests/test_fiber_targeting.py`.
  Backup `draft_backups/fiber_targeting.before_guess_retry.py`.
- Failure scan (`scan_targeting_failures_threaded.py`, 64 threads; 1M in 6 min): 1M 85,004,
  valid_100k 8,509, small_10k 855, Xu3 335, stencil 1, T8 46 → `$S/reports/targeting_fix/
  failures_all.csv` (94,750). Training failures are mostly faint (targeting SNR median 3.9; 65k of
  85k below 10; Xu3 minimum is 39); the guess retry recovers ~40% of them, all of Xu3's.
- Regeneration: `regen_targeting_failures.py` (moves old FITS to `fits/_targeting_fix_backup/`,
  reruns generate_fits, checks image bit-identical), array 47474887 (948 tasks × 100 galaxies).
  Pilot (3 galaxies) OK. Results per task in `$S/reports/targeting_fix/regen/`.
- Next: rebuild LMDBs for all six datasets (move old ones aside, not delete), regenerate T8
  benchmark_noisy.npz, release held jobs; then fix the paper sentence.
- **Targeting fix complete (2026-10-06 ~16:00).** Regeneration array 47474887: 94,750/94,750
  galaxies, all images bit-identical (task 836 hit the 30-min limit right after writing its
  results; complete). HSM now converges for all Xu3 + stencil failures, 80% of T8's, ~43% of
  the training/validation failures (rest: unweighted fallback, faint galaxies).
  Old LMDBs moved to `datasets/_pre_targeting_fix/` (Eason may delete later, ~30 GB); old FITS
  in `fits/_targeting_fix_backup/` (~4.5 GB). Rebuilt all six LMDBs (db/merge jobs
  47476758–47476777, no bus errors) and the T8 noise file. Verified vs old: counts and params
  identical; fib_pos identical for untouched galaxies, changed for every regenerated one.
- Released the 1M CCL (47454366) and the four ablation trainings. Added a fixed-data baseline
  `..._frozen_s42_measaxis_tgtfix` (47478506; caches 47478507 righthanded, 47478508 identity),
  because 8.5% of the 100k training galaxies changed: compare ablations to this, not to the
  pre-fix `_frozen_s42_measaxis`. The 100k CCL encoder used by all ablations is pre-fix (shared,
  so it cancels in the comparisons). Paper phrasing of the targeting fallback: discuss with Eason.

### 2026-10-08: paper folder about to be replaced by Eason's Overleaf edit
Pending (not in Overleaf) fiber-placement changes saved to
`repo_reports/draft_backups/2026-10-08_fiber_placement_pending/` (README.md lists the exact
text to re-insert: Sec. 3.1 measured-axis paragraphs, two obs-conf table rows + caption note,
nn-arch R90 "measured axes", bib keys SGA2020 + Mandelbaum2005; full file copies + diffs).
When Eason's new draft lands: re-read it, re-insert these, and update the failure sentence
(out of date after the targeting fix) after discussing phrasing with Eason.

### 2026-10-08: meeting report + corrections to the Step 3 reading
Report: `$S/reports/meeting-reports/2026-10-08/{report.html, background_notes.html, figs/}` (old
meeting report moved to `meeting-reports/2026-09-23/`). Figure scripts were throwaway (session scratchpad).
New findings (all on 100k models):
- Annulus face-on split: c from cos i > 0.9 only = +26±8, +35±9, +42±10 (e-4) for canonical,
  scratch s42, scratch s43 (true-axis data); cos i ≤ 0.9: +4±5, +4±6, +6±6. Measured-axis: +8±7 / +1±7.
  → the old-data annulus c offset is a face-on effect, robust across 3 trainings.
- Annulus per-annulus residuals are NOT noise: chi2 = 28.8/8 (measaxis), 49.9/8 (true-axis);
  synthetic Gaussian run chi2 = 4.9/8 → pattern lives in the NPE likelihoods. Open.
- Stencil: the R(σ²) linear-fit intercept is NOT evidence of R>1: a line fitted to the exact
  box-prior curve (R ≤ 1 everywhere, Brascamp–Lieb) at our bin variances gives a = 1.17. Direct
  evidence: old lowest-variance bin R = 1.41±0.07 (5.4σ above exact curve), face-on mean 1.13±0.05;
  measaxis 1.03±0.05 and 0.92±0.03. Draft Sec. 6.2 uses the Gaussian-prior line; fine for wide
  posteriors but not a proof of ideal posteriors at σ_post² ≲ 1e-3.
- "Fiber leak" reworded: old fibers turned exactly with shear; face-on stencil rotation median
  13.7° per Δg = 0.02 vs measured-β noise 12.8°. Earlier mechanism ("PA error swamps KL
  misalignment") is false: misalignment median 28° vs PA error 2.7° face-on.
- θ_int: only parameter whose width changes (+12.5% median), at all inclinations, largest at
  cos i 0.5–0.8 where PA error is 0.8° → not explained by layout; may be run-to-run (s43 pending).
- NLL +1.16 nats unexplained (θ_int width ≈ 0.17 nats).

### Ablation results (2026-10-12/13; 100k measured-axis, fixed data, shared pre-fix 100k CCL, Xu3 caches seed 42)
Reports: `$S/reports/ablations/ablations_weighted` (HTML; case order: baseline-fixed R90, s43,
noR90-train R90-inf, noR90-train id-inf, baseline-fixed id-inf, logit20, thetalr) and
`$S/reports/annulus-stack/ablation_*/summary.json`.

| Run | valid NLL | σ_shape | m g1/g2 % | 10⁴c g1; g2 (±1.4) | cov g1/g2/θ | annulus m | annulus c |
|---|---|---|---|---|---|---|---|
| baseline fixed s42 (R90 train+inf) | −16.75 | 0.0392 | −50.9/−51.4 | −1.1; 0.0 | 67.0/67.5/69.8 | −0.016±0.011 | +0.0002 |
| seed s43 | −16.77 | 0.0392 | −51.1/−50.6 | −1.1; 0.1 | 66.8/67.3/69.9 | −0.010±0.011 | 0.0000 |
| baseline, identity inference | — | 0.0391 | −50.8/−51.4 | **−10.1**; 0.5 | 66.7/67.2/68.5 | −0.016±0.012 | +0.0002 |
| no R90 train, identity inf | −16.46 | 0.0389 | −50.8/−50.5 | −4.9; −2.2 | 65.9/66.1/67.6 | −0.008±0.013 | +0.0004 |
| no R90 train, R90 inf | — | 0.0396 | −50.5/−51.5 | **−43.7; −12.4** | 66.8/67.2/70.3 | 0.000±0.012 | +0.0004 |
| logit limit 20 | −16.75 | 0.0392 | −50.3/−50.6 | −0.5; 1.1 | 67.4/67.2/70.5 | +0.002±0.012 | −0.0002 |
| θ lr 3e-4 (single lr) | **−17.23** | 0.0392 | −51.4/−50.9 | −0.4; 1.0 | 67.2/67.1/70.0 | −0.015±0.011 | +0.0004 |

Reading: R90 keeps c stable (identity inference leaves a training-dependent c1 ≈ −1e-3; R90
inference on a model not trained with R90 is badly biased) → keep R90 in training and inference.
Logit 20: logits saturate at the new cap, no gain → keep 10. Single θ lr: NLL −0.48 better (θ_int
log-prob 4.11 vs 3.73), shear metrics unchanged. 1M CCL done (best valid 4.332, ep 95).
Decision pending with Eason: θ lr for the 1M NPE.
- **Eason (2026-10-13): single θ lr (3e-4 everywhere), keep R90 (train+inference), logit limit 10.**
  1M CCL verified on the corrected data (rebuilt 2026-10-06 16:12; CCL ran 10-10 05:42–13:22).
  **1M paper-model NPE submitted:** `CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis`, job 49112712 (16 h; past 1M NPEs 8.5–9 h), train
  `train_1m_simv3_cosi_measaxis`, valid `valid_100k_simv3_cosi_measaxis`, frozen CCL
  `CNN-CNN-Meta-CCL-simv3-cosi-r90_1m_s42_measaxis`. Xu3 cache 49112713 (righthanded tag, seed 42) chained.

### Redshift-response test (2026-10-13; resolves the `\XW` note in data-generate.tex:43)
Design: 100 Xu3 measured-axis galaxies (random, seed 2026; hlr ≥ 1″, TF vcirc) rendered at
z = 0.14, 0.18, …, 0.50 with every other input fixed. Hα window moves with the line (61 px of
0.08 nm, [851, 855.81] nm at z = 0.3); line σ = 0.05(1+z) nm (rest frame fixed); camera = r at
z = 0.14, z otherwise; targeting seed = source row's, so image and fibers are the same at every z.
- Code: `generate_fits.py --redshift/--targeting-seed`, helpers in `observation_schema.py`
  (`halpha_window`, `halpha_observed_sigma`), wrapper forwards optional `redshift`/`targeting_seed`
  columns, fingerprint includes them only when non-default. Writer
  `data_generate/make_redshift_scan_samples.py`. Tests `tests/test_redshift_generation.py` (16).
  Full suite 512 pass / 5 pre-existing failures.
- Data: `samples/paper_tests_G_redshift_scan_xu3like_1k_measaxis{,_manifest}.csv`,
  `fits/` + `datasets/paper_tests_G_redshift_scan_xu3like_1k_measaxis` (ID = 100·k + base).
  Checks: z = 0.30 slice bit-identical to production Xu3 FITS (100/100); images same to 1e-7 at
  all z; fib_pos identical except one base at z ≥ 0.46 (2e-4″). Single-galaxy render: major-axis
  rotation split 5.54/6.34/7.33 px at z = 0.14/0.30/0.50 = (1+z)/1.3 to 0.4%.
- Inference: `arch/diagnostics/paper_tests/g_redshift_sample.{py,slurm}` (10 paired noise seeds,
  10⁴ identity+R90, TF weights) → `$S/cache/paper_tests_G_redshift/<model>/`.
  Analysis: `g_redshift_analysis.py` → `$S/reports/paper-tests/G_redshift/<model>/report.html`.
- **Dry run, 100k model `..._valid100k_frozen_s42_measaxis_tgtfix`** (1M model pending):
  - Shear robust: paired Δm(z) within ±0.004 (±0.006) over 0.14–0.50; c unchanged; per-galaxy
    RMS shift of ĝ ≈ 0.2σ at the ends; g coverage 0.67–0.71.
  - Velocities read in z = 0.3 pixel units: slope of v̂0 vs z = 0.3 is 0.89–1.12 and of v̂circ
    0.84–1.17 against the pixel-scale prediction (1+z)/1.3 = 0.88–1.15. RMS shift up to 2σ (v0),
    1σ (vcirc); v0 coverage 0.68 → 0.36–0.41 at the ends, vcirc → 0.58–0.61.
  - Hα flux RMS shift 2–2.8σ at the ends, coverage 0.26–0.29 (continuum under Hα changes with z
    at fixed m_r). cos i, rscale RMS 0.25–0.57σ.
  - So the draft's "redshift only changes size and brightness" is not true for velocities and
    Hα flux; it is true to Δm ≲ 0.004 for shear.
- **Eason (2026-10-14): keep the single-redshift (z = 0.3) training set; justify it briefly in
  the paper, with the full scan in an appendix.** Don't edit the draft yet; this is the plan.
  - Where: data-generate.tex:43 (the paragraph with the `\XW` note on redshift). Replace the
    unsupported claim "Redshift then only changes how large and how bright the galaxy appears"
    with a short, results-based justification:
    shear estimates are robust to source redshift; velocity and Hα-flux estimates come out as if
    the galaxy were at z = 0.3 (the network reads line-of-sight velocities in z = 0.3 pixel units,
    so v0 and vcirc scale by (1+z)/1.3; the Hα flux shifts because the continuum under the line
    changes at fixed m_r). Point to the appendix.
  - Appendix (new, "Redshift scan" or similar), contents:
    1. Setup: 100 Xu3 galaxies at 10 redshifts 0.14–0.50, all other inputs fixed; window centred
       on redshifted Hα with the same 61 × 0.08 nm pixels; rest-frame line width fixed; same image
       and fibers at every z; 10 paired noise draws per galaxy; z = 0.3 reference.
    2. What physically changes with z (velocity per pixel, line width in pixels, continuum level).
    3. Shear: paired Δm(z) and c(z) against z (figure), coverage.
    4. Velocities: slope of v̂ vs z = 0.3 against (1+z)/1.3 (figure), coverage drop.
    5. Hα flux: shift and coverage; cause is the continuum, not the velocity scale.
    6. One-line remark: real data would be rescaled to z = 0.3 velocity units (or the model
       trained over z) if velocities or flux are wanted; shear needs neither.
  - Numbers/figures: use the 1M paper-model run (job 49206985 →
    `$S/reports/paper-tests/G_redshift/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis/`),
    not the 100k dry run above. Paper figure candidates: `shear.png` (Δm panel), `slope.png`
    (v0, vcirc panels), example posteriors.

### 1M paper model results (2026-10-14)
`CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis`: NPE 49112712 (9.2 h), best
epoch 198/200, valid NLL −19.68 (train −19.73). Xu3 cache 49112713 (100/100).
Xu3 weighted report `$S/reports/paper-model-1m/xu3_weighted.html` (with the 100k θlr3e-4 and
tgtfix runs for comparison):

| Model | σ_shape | m g1/g2 % (full range) | 10⁴c g1; g2 (±1.4) | cov g1/g2/θ_int % |
|---|---|---|---|---|
| 1M paper model | 0.03913 | −49.3/−49.0 | +0.9; +0.3 | 68.2/67.7/69.6 |
| 100k θ lr 3e-4 | 0.03922 | −51.4/−50.9 | −0.4; +1.0 | 67.2/67.1/70.0 |
| 100k baseline (tgtfix) | 0.03924 | −50.9/−51.4 | −1.1; 0.0 | 67.0/67.5/69.8 |

Redshift scan on the 1M model (job 49206985; report
`$S/reports/paper-tests/G_redshift/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis/report.html`)
reproduces the dry run: Δm(z) within ±0.004 (errors 0.007–0.009), c unchanged; v0 slope
0.89–1.12 and vcirc 0.85–1.17 against (1+z)/1.3 = 0.88–1.15; v0 coverage 0.68 → 0.36 at both
ends, vcirc 0.71 → 0.57–0.59; Hα-flux coverage 0.65 → 0.25–0.26; g coverage 0.67–0.72.

### Paper reruns on the 1M model (started 2026-10-14; annulus stack deliberately NOT rerun yet —
Eason is still checking the annulus method)
All outputs go to a NEW tree `$S/reports/paper-tests-1m/` (same layout as `paper-tests/`), so the
100k fibrepair outputs stay intact. Scripts gained options whose defaults are the old values
(backups `draft_backups/*.before_1m.py`); figure scripts read env vars
`KLNN_PAPER_TESTS_ROOT`, `KLNN_PAPER_MODEL`, `KLNN_R2B_DIR`; selection scripts also
`KLNN_STENCIL_CACHE`.
| Paper item | Job | Producer → output |
|---|---|---|
| Sec. 5 loss, P-P, nuisance coverage, TF ESS (nn-arch :116) | 49208149 `a_1m_cpu_chain.slurm` | `a_npe_stream --case` → `a_npe_report --case` → `A_npe/` |
| Sec. 6.1–6.3 raw bias, shrinkage, weights, cuts | same chain | `a_shear_extract --model --dataset` → `a_shear_analysis` → `A_shear/` |
| Sec. 5 Fig. posterior_snr, appendix 9D corner (gal 6447, measaxis rows) | 49208130 `b_r2b_1m.slurm` | `b_r2b_sample --model --dataset --out` → `$S/cache/paper_tests_B_inference/r2b_1m_measaxis/` (image + Hα sweeps) |
| Sec. 6.3 tab:selection, 6.4 response/calibration | 49207673 stencil cache (done) | `b_t2b_analysis --label xu3like_1m_measaxis` → `B_inference/t2b_xu3like.json`; `e_sel_weights`, `e_sel_cosi` → `B_inference/*.txt` |
| Sec. 6.5 T8 stacking | 49208128 → 49208129 | `c_stack_infer --model --data likelihood_stack_04_benchmark_measaxis` → `c_stack_analysis` → `$S/cache/paper_tests_C_stacking/<1m>/` |
| Figures + `sec6_numbers.json`, `tab_*.tex` | 49208983 `paper_1m_figures.slurm` | `c_stack_report` → `make_paper_figures_sec5/6` → `paper_figs_final/` |
Text to update afterwards (not done; Eason's new Overleaf draft first): results.tex :10 caption
"lowest at the final epoch" (1M best is 198/200), the θ_int coverage wording (69.6%, over),
nn-arch :127 CCL setup (1M CCL ~7.7 h, best ep 95), :129 PE setup (single lr 3e-4; ~9.2 h),
results.tex :26 timing (re-time), data-generate :77 stale HSM failure sentence, all `\XW`
"100k" notes, shear.tex :70 (cos i widths, not in any json; recompute), e_sel_cosi thresholds
(0.436/0.312 were chosen to match width-cut kept fractions; re-choose if those change).

### MCMC comparison package for Lauren (2026-10-14)
7 Xu3 measured-axis galaxies (priority 6447, 1682, 96201, 40690, 62769; optional 13804, 26386;
inclination × Hα S/N grid + 6447). Package (send this folder): `$S/reports/paper-tests-1m/mcmc/package/`
(clean + noisy FITS, truth.csv, README for Lauren). Noisy data = exact 1M Xu3-cache noise
(`k_mcmc_export.py`/`.slurm`; σ matches the cache exactly; NPE rerun matches the cache within MC).
Our NPE samples on these exact data (not for Lauren): `$S/reports/paper-tests-1m/mcmc/npe/`.
Selection table: `$S/reports/paper-tests-1m/mcmc/selection.csv`. Supersedes the old Xu1 set in
`~/mcmc_xu1_four_fits` (sin i, galaxy-axis fibers).

## Cluster test round 2 (s3_sub1, subhalo at (1,0) Mpc), 2026-10-23
Report: https://claude.ai/artifact/P13C8d3J4NYQRc6XuasS7Z (copy: `$S/reports/paper-tests-1m/H_cluster/s3_sub1/report.html`).
- No halo fit; residual = map − TRUE main-halo κ (pixel-averaged, 4×4 subgrid; point κ diverges on the KS centre cell).
- Tuning (20 synthetic realizations, `h_cluster_tune.slurm` array + `--aggregate`): chosen likelihood map 0.5 Mpc /
  σ_κ 0.03 / smooth 1 px (contrast 3.0, false-peak 0.00); KS 0.7 cells (contrast 1.25, fp 0.07). Choice at grid edge.
- NPE single-map z at the subhalo (0.5 Mpc aperture vs synthetic null −0.0086 ± 0.0031): likelihood 3.1 (expected 3.0,
  16–84% 2.1–3.9, 85% > 2); KS 0.9 (expected 1.1). Paired sub−main: +0.0098 (expected +0.0095).
- NPE cells: no slope ≥ 2σ; Cartesian g1 c = +0.004/+0.006 ± 0.002 (shared noise; likely fluctuation); χ² low
  (16.8/27, 22.6/35), pull spread 0.72–0.86.
- Validation (30 real., realistic widths): pull mean |≤0.05|, spread 0.88–0.92; n=2 polar cells −0.010; a few cells
  3–4σ off ref (≤0.012) not removed by exact-F reference → stack-level (prior bound / KDE edge reflection?), TODO.
- FIXES: (1) synthetic() double-truncated widths (16% narrow) → `xu3_shear_covariance` now de-truncates (within 6% of
  NPE). (2) Laplace σ used exact Hessian (bilinear grid ⇒ no curvature, indefinite) → Gauss–Newton J^T F J; S/N uses
  noise cov H⁻¹H_dH⁻¹ (matches synthetic scatter to ~12%, conservative); posterior H⁻¹ is ~2× wider.
- Old outputs kept: `tuning_pointtruth_nanks/`, `tuning_narrowsynth/`; script backups in `repo_reports/draft_backups/`.
- 2026-10-23 (later): per Eason, the cluster "residual" = DIFFERENCE map(main+sub) − map(main only) (same galaxies and
  noise; goal is to visualise how visible the effect is, not real-data feasibility). NPE: likelihood peak 0.012 on the
  subhalo (true pixel κ 0.030), aperture 0.0098 (exp. 0.0095), background rms 0.0016; KS peak 0.008, aperture 0.0065.
  `h_cluster_analysis.py --replot` redraws kappa_maps.png from the npz. synthetic() now uses common random numbers
  per galaxy (before, independent MC draws made the synthetic difference map noisy); tuning/validation not rerun.

## Cluster ensemble, option B (50 independent galaxy sets, NPE), 2026-10-25
Tags `s3_sub1_n01..n50` (seeds in `$S/reports/paper-tests-1m/H_cluster/ensemble/ensemble_seeds.txt`; Poisson
N = 118–179, expected 148; inference SEED = 42 + 1000 r shared by main/sub). Map settings from s3_sub1 tuning.
Summary: `ensemble/summary.json`, figs `ensemble/{cells,detection,diffmaps}.png`; script `h_cluster_ensemble_summary.py`.
- Cells (pooled, linear WLS of stacked − ref on ref): polar g_t m = +0.04 ± 0.04 (main) / +0.01 ± 0.04 (sub);
  Cartesian g1 +0.02 ± 0.03, g2 +0.06 ± 0.03; c ≲ 3e-4. Pull means |≤ 0.04|, pull sd 0.81–0.86 (NPE errors ~15%
  conservative; synthetic 0.88–0.92). n = 1 / 2 polar cells −0.011 / −0.005; n ≥ 3 g_t: unweighted −0.0011 ± 0.0005,
  inverse-variance weighted +0.0004 ± 0.0003 (shrinkage only in broad cells).
- Detection (single-map z vs ensemble main-only null at 0.5 Mpc aperture; null −0.0124 ± 0.0038):
  likelihood map median z 2.4 (16–84: 0.9–3.4), z > 2 in 60%, z > 3 in 28%; KS median 1.1, z > 2 in 22%, z > 3 in 4%.
  Lower than the fixed-galaxy synthetic forecast (3.0) because configuration variance widens the null and the
  mean signal is 0.0084 (s3_sub1 single run 0.0098 was a favourable draw). corr(N_gal, z) = 0.32.
- Difference maps: likelihood peak within 0.5 Mpc of the subhalo in 38/50; peak median 0.014, background rms 0.0018
  (peak/rms ~ 7.6). Mean over 50 realizations is a clean blob on the subhalo; KS mean is a cell-sized blob.

## Cluster subsections drafted + halo fits (2026-10-27)
- **Paper text:** `KL-NN_old_draft/sections/shear-cluster.tex` (6.4 "Likelihood Stacking in a Cluster Shear Field",
  6.5 "Detecting Substructure"), `\input` at the end of `sections/shear.tex`; figs `figs/cluster_{maps,cells,detection}.pdf`;
  8 new verified bib entries (NFW97, Duffy08, Planck18, KS93, Zaroubi95, Jeffrey18, LBFGS, WtG3). Checked with a stub
  build (no REVTeX on Bridges-2). Halo-fit paragraph marked optional (`\XW`, Eason to ask advisors).
- **Version caveat:** `KL-NN_old_draft/` = the local copy (2026-10-16; post-Tim restructure, 1M numbers, citation fixes).
  Eason's Overleaf export (in `Accelerating_.../`, imported 2026-10-27) is OLDER in Sec. 6 (still has weighting/cuts) but
  has his own wording of the redshift sentences in Sec. 3.1. Needs a merge decision. Old pre-10-16 draft backed up to
  `draft_backups/KL-NN_old_draft_2026-10-06/`.
- **Figures:** `make_paper_figures_cluster.py` → `$S/reports/paper-tests-1m/paper_figs_final/fig_cluster_*` + `cluster_numbers.json`.
  Map example = realization with z closest to the ensemble median (n20). Cell figure uses the main+sub scenario only
  (main and sub share noise; pooling double-counts).
- **Halo fit (Tae's WtG-III-style suggestion):** `h_cluster_halo_fit.py` / `.slurm` (array 1-50), summary
  `h_cluster_halo_fit_summary.py` → `ensemble/halo_fit_summary.json`. 1-parameter model integrated on a grid (Nautilus
  needs ≥2 params; the old `halo_fit` in h_cluster_analysis.py would crash for "main"). Results (50 real.):
  main-only data M_main mean log 15.011 ± 0.010 (truth 15), pull sd 0.75; sub data, fixed position: ln B > 1 in 92%,
  > 3 in 64% (null: max 0.8); free position: ln B > 1 in 38%, offset < 0.5 Mpc in 80%; M_sub median of medians 14.40
  (truth 14.48), pull mean −0.23 ± 0.11, 68% coverage 0.74.
- Cells n ≥ 3 (sub scenario): unweighted mean g_t residual −0.0014 ± 0.0005, precision-weighted +0.0005 ± 0.0002;
  n = 1: −0.012 ± 0.003, n = 2: −0.005 ± 0.002.

## Paper folder swap (2026-10-27)
Eason confirmed the local copy is the current version. `~/Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/`
is now an exact copy of `~/KL-NN_old_draft/` (incl. cluster subsections). Eason chose not to restore his Overleaf Sec. 3.1
redshift wording: full replacement. The replaced Overleaf import is kept at
`draft_backups/Accelerating_overleaf_import_2026-10-27/`. Eason uploads the new folder to Overleaf.

## Halo-fit report, pull spread, source density (2026-10-27)
Working paper copy is now `~/Accelerating_Kinematic_Lensing_Inference_with_Neural_Networks/` (KL-NN_old_draft is stale).
- Source density (sec 6.4 text, XW removed): Xu sample 3 full-footprint catalog header WDENSITY = 0.1429/arcmin^2
  (completeness-weighted; raw 0.147; footprint 19721 deg^2); weighted fraction at z > 0.13 (spec-z else photo-z) 0.755
  -> 0.108 = config default. Xu et al. quote 0.18.
- Pull spread < 1 (sec 6.4 text, XW replaced by explanation). Diagnostics in
  `$S/reports/paper-tests-1m/H_cluster/diag_pulls/` (scripts + outputs; per-axis KDE results also in scratch). Pull sd of
  n>=3 cells by stacked sd bin [<0.015 / 0.015-0.03 / >0.03]:
  exact Gaussian likelihoods 0.97-1.02 / 0.98-1.07 / 0.64-0.75 (prior box);
  NPE full-covariance-bandwidth KDE 0.90-0.95 / 0.97-1.05 / 0.65-0.68;
  NPE per-axis Scott KDE (paper) 0.84-0.90 / 0.95-1.02 / 0.65-0.68.
  Cause: wide cells = prior box; narrow cells = per-component KDE bandwidth on tilted, elongated likelihoods
  (most precise galaxies' minor axis widened ~20%). TF-weight n_eff ~6000, so isotropic KDE inflation only ~3%.
  Possible fix (not done; Eason to decide): full-covariance bandwidth H = n_eff^(-1/3) Sigma, then rerun H_cluster
  (and Sec. 6.3 text describes per-component Scott's rule).
- Halo-fit significance: `h_cluster_halo_sigma.py` -> `H_cluster/ensemble/halo_sigma.json`. sigma = sqrt(2 dlnL_max),
  M_sub >= 0 at known position, main mass free. Sub data: median 3.46 (16-84: 2.48-5.03), >2 in 92%, >3 in 64%.
  Main data: 0 in most, max < 2. Map z for comparison: lik 2.36 (60% >2, 28% >3), KS 1.12 (22%, 4%).
- Report artifact: https://claude.ai/artifact/CFABynsHsQwFwrvZ861kzR (build: scratchpad halo_report/figs.py + build.py).

## Full-covariance KDE rerun (2026-10-27, Eason approved)
- New `arch/diagnostics/paper_tests/kde_grid.py`: scipy gaussian_kde Scott kernel (full weighted covariance,
  n_eff^(-1/3)), CIC binning + scipy.signal.fftconvolve, reflection by folding (edge nodes are their own mirror),
  H_MIN = 1e-3 kernel-sd floor. `kde_grid_validate.py`: binned vs exact gaussian_kde < 0.5% where density > 1% of
  peak (2% for the most precise galaxy, where the H_MIN floor applies); thin-axis width ratio 1.00-1.04.
- `h_cluster_analysis.py` and `c_stack_analysis.py` use it (pre-change copies in draft_backups/*.before_fullcov_kde.py;
  c_stack local exact-lattice check dropped; `c_stack_report.py` text adjusted).
- Old outputs kept: H_cluster/s3_sub1_nRR/npe_axiskde/, H_cluster/ensemble/axiskde_2026-10-27/,
  paper-tests-1m/C_stacking_axiskde_2026-10-27/, paper_figs_final/axiskde_2026-10-27/; C stacking analysis_v1.npz (old)
  vs analysis_v2.npz (new). Jobs: H_cluster/ensemble/fullcov_jobs.txt; stack 49953789, report 49956021.
- New numbers. Sec. 6.3 stacking N=32: m1 -0.010 (+0.004/-0.005), m2 0.001 +-0.004 (table + text updated).
  Cluster halo fit: sigma median 3.93 (16-84: 2.71-5.47), >2 in 94%, >3 in 76%; main-only data max 1.82.
  M_main (main data) 15.006 +- 0.010; M_sub median 14.42 (truth 14.48), pull -0.19 +- 0.12; coverage 0.72-0.80.
  ln B fixed median 5.8 (>1 94%, >3 74%); free: >1 52%, offset median 0.22 Mpc, < 0.5 Mpc in 82%.
  Cells n>=3: polar g_t m 0.011 +- 0.038 (sub); Cartesian 0.010 +- 0.023 / 0.041 +- 0.026; pull sd 0.83-0.90
  (wide cells, prior box, still < 1; Eason: prior-box explanation not in the paper for now).
- Paper: Sec. 6.3 KDE sentence now says full-covariance Scott's rule via SciPy (\citep{SciPy} added to main.bib).
  Sec. 6.4/6.5 still have the old text and numbers; to be rewritten per the 2026-10-27 restructure.

## Appendix restructure applied (2026-10-27, Eason approved layout)
- New `sections/appendix.tex` (\subfile'd in main.tex after acknowledgments; starts with \appendix):
  A "Further Tests of the NPE" = A.1 app:corner, A.2 app:coverage, A.3 app:redshift (headers + \XW to-write notes);
  B "Posterior Means" (app:posterior-means) = B.1 app:raw-bias, B.2 app:shrinkage: old 6.1/6.2 text near verbatim,
  raw_bias.pdf and the cubic-response sentences dropped (fig:shrinkage, eqn:gauss-posterior, eqn:shrinkage kept).
- Shape-noise paragraph moved verbatim to Sec. 5.1 (validation.tex, after fig:posterior-snr).
- Sec. 6 intro rewritten to the new flow (means biased -> App. B; stacking; cluster cells; substructure).
  Stacking first sentence and the Gaussian-model pointer adjusted; closing paragraph now introduces the two
  cluster routes (cells in small regions; likelihood of each galaxy at a shear-field model).
- Backups: draft_backups/shear.2026-10-27-before-appendix.tex, validation.2026-10-27-before-shapenoise.tex.
- Stub build (scratchpad build3: validation + shear + appendix) clean. Next: rewrite 6.4 (cells: m, c, residual vs n;
  no pull histogram) and 6.5 (halo fit only), with the full-cov-KDE numbers.

## Sec. 6.3 rewritten as halo fit only (2026-10-29, Eason approved schema)
- kappa maps / KS / single-map z deleted outright (figs cluster_maps, cluster_detection no longer used; KS93, Zaroubi95,
  Jeffrey18, LBFGS now uncited but left in main.bib). Detection statistic = likelihood-ratio sigma only (no ln B).
- New figs: figs/cluster_sigma.pdf (sigma histogram, both scenarios), figs/cluster_localization.pdf (free-position
  posterior medians; Eason to decide if it stays). From make_paper_figures_cluster.py (fig_sigma, fig_localization;
  main no longer draws maps/detection/halofit). Nautilus entry added to main.bib (from the package README, v1.0.6).
- Numbers: sigma median 3.9, >2 94%, >3 76%, main-only max 1.8; M_main 15.006 +- 0.010; M_sub medians ~14.42
  (truth 14.48), 68% coverage 0.80; free position < 0.5 Mpc in 82%, median offset 0.22 Mpc. Max |g| with sub 0.074.
- Backups: draft_backups/shear.2026-10-29-before-63-halofit.tex, make_paper_figures_cluster.before_sigma_loc.py,
  cluster_numbers.2026-10-29-before-63-halofit.json, main.2026-10-29-before-nautilus.bib.
- 2026-10-29 (later, after Eason's rewording): σ ≡ √Δχ², with Δχ² = −2 ln(L_max,main/L_max,sub); the figure axis
  and caption show σ, and the 2σ line is removed. Subhalo mass: mean of posterior medians 14.37 ± 0.04 over all 50
  (low because weak detections are prior-dominated), 14.47 ± 0.02 for the 38 above 3σ (cluster_numbers.json
  msub_mean). The localization figure is colored by σ (Spearman: offset vs σ −0.80; distance from main vs σ +0.37,
  p = 0.007); full column width. 6.2 now says "inverse covariance matrix" / "weighted by the inverse covariance";
  nn-arch has the reweight and \times fixes. Next session: Discussion (items in kl-paper-gaps.md: fast exploration,
  halo-fit model dependence as a strength).
