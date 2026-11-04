# KL-NN paper: pre-writing test plan (planned 2026-09-29, NOT yet run)

Goal: quick tests that decide how Sec. 5–6 of the paper are written (see
`RESULTS_OUTLINE.md`), plus the figures for those sections. Eason approved
running them with subagents and wants **one HTML report** at the end, in the
style of past campaign reports.

## Ground rules for every agent

- **Model:** `CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair`
  (100k-trained, CCL-pretrained frozen encoder). **Test set: Xu3 only.** No
  1M model, no Xu5.
- **Xu3 posterior cache (read-only, 12 GB):**
  `$S/cache/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair/test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded`,
  with `$S = /ocean/projects/phy250048p/shared`.
  - Holds 10^4 identity+R90 samples per galaxy, TF log weights, ESS and SNRs.
  - Truths: `$S/samples/test_100k_simv3_cosi_xu3_tf.csv`.
  - Read the loaders in `arch/diagnostics/shear_bias_report.py` to see the
    cache format; reuse its functions (`compose_shape_noise_regularized_weights`,
    `component_metrics`, `fit_design`, `galaxy_frame_components`) rather
    than reimplementing them.
- **Estimator:** the TF-weighted posterior mean of (g1, g2). Use the TF
  weights in all posterior summaries.
- **Shape noise (Xu24 definition):** σ_ε,i = √(σ_g1,i² + σ_g2,i²) from the
  TF-weighted 1D marginal standard deviations. The ensemble σ_ε is the mean
  (or RMS; report both) over galaxies.
- **Bias fit:** ĝ − g = m g + c on |g| < 0.02, per component. Report in the
  image frame (g1, g2) and the galaxy frame (g+, g×), with fit errors and
  bootstrap errors.
- **"Raw" means** equal galaxy weights, the full sample, and no cuts,
  weighting, calibration or stacking.
- **Never overwrite existing models, caches or reports.** Write only to the
  new campaign folder:
  `$S/reports/paper-tests/`, with an `index.html` plus one subfolder per
  test containing `report.html`, `report.json` and PNGs, and a `STATUS.txt`
  listing job IDs. Add a row for `paper-tests/` to `$S/reports/README.md`.
- Put new scripts in `kl-nn/arch/diagnostics/paper_tests/` (a new folder). Do
  not modify existing diagnostics. Don't commit unless Eason asks.
- CPU jobs can run on the login/interactive node if they're small; use SLURM
  (`sbatch`) for anything over ~10 min or needing a GPU (V100 partition as in
  `arch/*.slurm`).

## Tests

**Group A: from the Xu3 cache only (CPU, minutes each)**

| ID | Question | Method | Output |
|---|---|---|---|
| R-raw | Raw per-galaxy mean results (paper Sec. 6.1) | m, c (image and galaxy frame), σ_ε with equal weights on the full sample | Binned ĝ−g vs g plot (paper R5); histogram of σ_ε,i; table |
| T1 | Does σ_ε,i depend on the true shear? | Bin σ_ε,i by true \|g\|, g+, g× (galaxy frame); fit the slope dσ/dg; also split by cos i | Plots and slopes with errors. If not flat, σ-weighting/cuts can bias m |
| T3 | Does shrinkage explain m quantitatively? | In σ_ε,i bins, measured m vs predicted −⟨σ_i²/(σ_i²+σ_π²)⟩ with σ_π = 0.2/√12 = 0.0577. Use the per-component σ_g for the prediction | Plot (paper D1 candidate) |
| T4 | Is the low-\|g\| linear fit OK? | m from the \|g\|<0.02 linear fit vs the full range (linear and cubic); ĝ vs g over the full ±0.1 range | Plot and table |
| T5 | Robust errors | Bootstrap (≥1000) m, c, σ_ε for: raw; inverse-variance weights with a shape-noise floor (existing `compose_shape_noise_regularized_weights`); σ_ε,i cuts at a few thresholds (0.03, 0.04, 0.05, 0.075, 0.1) | Table |
| R6 | Width-cut curve (paper Sec. 6.3) | m1, m2, ensemble σ_ε, retained fraction and low-\|g\| count vs σ_ε,i threshold, for raw and weighted | Plot |
| T2a | Does the σ cut add selection bias? | Compare m vs mean σ_ε,i for (i) cuts on σ_ε,i and (ii) cuts on truth-level, shear-independent quantities (true cos i, ρ_I, ρ_S) at matched retained fractions | Plot. Same curve means no extra selection bias from cutting on σ |
| T6 | TF weight health | Distribution of the per-galaxy TF ESS fraction; ESS vs cos i, m_r, ρ_S; flag galaxies with ESS < 50 | Plots and summary |
| T7 | σ convention | σ_ε,i from the standard deviation vs the 16–84% half-width; rerun R6 with each | Plot and table |
| R3 | Calibration (paper Sec. 5.2) | NPE-only P–P curves for all 9 parameters (TF-weighted ranks) and a 68%/95% coverage table on the full Xu3 set. The existing `likelihood_stack_pp_comparison.py` includes NRE and uses only 1000 rows, so write an NPE-only version | Plot and table |
| R1 | Loss curve (paper Sec. 5.1) | From `$S/models/losses/losses_<model>.csv` (see `plot_losses.py`) | Plot |
| R2a | Corner plot (paper Sec. 5.1) | 9D corner plot for 1–2 Xu3 galaxies from the cached samples, with truth lines. Pick a moderately inclined, typical-SNR galaxy | Plot |

**Group B: small new inference (GPU, minutes)**

| ID | Question | Method | Output |
|---|---|---|---|
| R2b | Posterior vs image SNR (paper Sec. 5.1) | One Xu3 galaxy, sampled at ρ_I = 50 and 200 with ρ_S fixed at the catalog value, 10^4 samples each (`sample_density` in `arch/train.py`; see how `cache_posteriors.py` sets SNR) | g1–g2 contour plot |
| T2b | Selection and shear response for the current model | Build or reuse a matched-shear stencil: ~1000 galaxies × 5 shears (Δg = 0.01), shared noise. Measure the shear response R and the selection response of the σ cut (change in the selected fraction and in mean ĝ with g), metacal-style. The existing stencil (`$S/reports/shear-response/`, legacy model 45255702) shows how to build it (`data_generate/make_shear_response_samples.py`, `arch/diagnostics/shear_response_report.py`); regenerate the inference with the fibrepair model | Table. Also gives the R(σ) calibration for paper Sec. 6.4 |

**Group C: NPE likelihood stacking (paper Sec. 6.5; GPU, minutes to hours)**

| ID | Question | Method | Output |
|---|---|---|---|
| T8 | Does stacking NPE likelihoods remove the shrinkage? | Use the fixed-shear benchmark dataset `$S/datasets/likelihood_stack_04_benchmark` (25 shear cells × 32 nuisance galaxies × 16 noise realizations; see `arch/diagnostics/likelihood_stack_benchmark*.py` and `likelihood_stack/04_benchmark*/report.*`). Per observation, form the shear likelihood L_i(g) ∝ ∫ q(g,η\|D_i)/π_train(g,η) dη, i.e. the marginal posterior of g divided by the uniform shear prior. Multiply over the galaxies in a cell and estimate the cell shear | m vs number of galaxies N and noise realizations R; compare with the raw per-galaxy mean |

T8 must fix the known problems of the earlier benchmark:
- **Grid snapping.** The old 21×21 grid (0.01 spacing) contained every true
  shear. Use a continuous estimator, e.g. a KDE of the g marginal or a fine
  grid (≤0.001) with interpolation, and/or offset the cell shears from the
  grid.
- **Degenerate nuisance marginalization (ESS = 1).** Marginalize over η by
  using the 2D g marginal of the NPE (a KDE of the g samples divided by the
  uniform shear prior), not by importance-weighting 9D candidates.
- **Scale.** Report bootstrap errors, and use more galaxies per cell if
  possible.

**No NRE** anywhere (Eason's decision).

## Report

`$S/reports/paper-tests/index.html` holds:
- a summary table of every test (question → answer → implication for the
  paper);
- links to each test's `report.html`.

Figures intended for the paper should be saved as PNG/PDF with clean labels
in `paper-tests/paper_figs/`.

## Suggested agent split (for when this is resumed)

1. **Agent 1, Group A shear tests:** R-raw, T1, T3, T4, T5, R6, T2a, T7.
2. **Agent 2, Group A NPE validation:** R1, R2a, R3, T6.
3. **Agent 3, Group B:** R2b and T2b (GPU; SLURM).
4. **Agent 4, Group C:** T8 (GPU; SLURM).
5. **Then** the main session compiles `index.html` and reviews the results
   for claims before writing the text.

Each agent gets these ground rules, the cache paths, and the rule to report
exact numbers and anything that looks off.
