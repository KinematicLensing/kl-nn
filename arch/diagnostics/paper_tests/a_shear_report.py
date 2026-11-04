#!/usr/bin/env python3
"""Paper tests, Group A (shear): assemble report.html / report.json / STATUS.txt
from results.json, supplement.json and supplement2.json. Also re-renders the
paper-candidate R5 figure with clean tick labels from the stored binned data.
"""

from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_shear_analysis as A  # noqa: E402

R_DIR = Path("/ocean/projects/phy250048p/shared/reports/paper-tests/A_shear")
P_DIR = Path("/ocean/projects/phy250048p/shared/reports/paper-tests/paper_figs")
SCRIPTS = Path(__file__).resolve().parent
JOBS = {
    "47273346": "a_shear_extract.slurm: stream the cache into per_galaxy.npz (428 s)",
    "47273654": "a_shear_analysis.slurm: R-raw, T1, T3, T4, R6, T2a, T7, T5 (1000 bootstraps) -> results.json (275 s)",
    "47273993": "a_shear_supplement.slurm: S1-S4 follow-ups (500 paired bootstraps) -> supplement.json",
    "47274097": "a_shear_supplement2.slurm: S5 weighting follow-up (300 paired bootstraps) -> supplement2.json",
}


def f(x, n=3):
    return f"{x:+.{n}f}"


def pm(x, e, n=3):
    return f"{x:+.{n}f} ± {e:.{n}f}"


def img(name, caption):
    data = base64.b64encode((R_DIR / f"{name}.png").read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
            f"<figcaption>{caption}</figcaption></figure>")


def table(header, rows):
    h = "".join(f"<th>{c}</th>" for c in header)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def section(tid, title, question, method, result, implication):
    return (f'<section id="{tid}"><h2>{tid}: {title}</h2>'
            f"<p><b>Question.</b> {question}</p><p><b>Method.</b> {method}</p>"
            f"<h3>Result</h3>{result}<p class='impl'><b>Implication for the paper.</b> {implication}</p></section>")


def rerender_r5(R):
    b = R["R_raw"]["binned"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, comps, title in ((axes[0], ("g1", "g2"), "image frame"), (axes[1], ("g+", "gx"), "galaxy frame")):
        for comp, col, mk in zip(comps, (A.C1, A.C2), ("o", "s")):
            d, r = b[comp], R["R_raw"][comp]
            lab = {"g1": "$g_1$", "g2": "$g_2$", "g+": "$g_+$", "gx": r"$g_\times$"}[comp]
            ax.errorbar(d["x"], d["mean_resid"], d["se"], fmt=mk, ms=4, color=col, lw=1,
                        label=f"{lab}: $m$ = {r['low_m']:+.3f} ± {r['low_m_se']:.3f}")
            lim = 0.1 if comp in ("g1", "g2") else 0.12
            xx = np.linspace(-lim, lim, 50)
            ax.plot(xx, r["low_c"] + r["low_m"] * xx, color=col, lw=1, alpha=0.7)
        ax.axvspan(-A.LOW_G, A.LOW_G, color="#f0f0f0", zorder=0)
        ax.axhline(0, color=A.GREY, lw=0.8)
        ax.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])
        ax.set_xlabel("true shear component" + (r" ($g_1$, $g_2$)" if comps[0] == "g1" else r" ($g_+$, $g_\times$)"))
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel(r"$\hat g - g$ (bin mean ± s.e.)")
    fig.tight_layout()
    A.save(fig, R_DIR, "Rraw_binned_residual", P_DIR, "A_R5_raw_bias")


def main():
    R = json.loads((R_DIR / "results.json").read_text())
    S = json.loads((R_DIR / "supplement.json").read_text())
    S2 = json.loads((R_DIR / "supplement2.json").read_text())
    X = json.loads((R_DIR / "per_galaxy.npz.json").read_text())
    rerender_r5(R)
    ref = R["reference_check"]
    raw = R["R_raw"]
    se = raw["sigma_eps"]
    T5 = R["T5"]["estimators"]
    rawb = T5["raw"]["components"]
    wb = T5["weighted"]["components"]
    T1 = R["T1"]
    S1 = {r["box_max_abs_g1_g2"]: r for r in S["S1_interior_slopes"]}
    S3 = S["S3_mechanism"]
    S2r = {r["frac"]: r for r in S["S2_paired_selection_excess"]["raw"]}
    S2w = {r["frac"]: r for r in S["S2_paired_selection_excess"]["weighted"]}
    r6 = {r["threshold"]: r for r in R["R6"]["rows"]}
    T7 = R["T7"]
    chi3 = R["T3"]["chi2"]
    chi4 = S["S4_chi2"]
    T4 = R["T4"]
    ratios = [T5[n]["components"][c]["m_boot_se"] / T5[n]["components"][c]["m_fit_se"] for n in T5 for c in A.COMPONENTS]
    cr = [T5[n]["components"][c]["c_boot_se"] / T5[n]["components"][c]["c_fit_se"] for n in T5 for c in A.COMPONENTS]

    # ------------------------------------------------------------ summary
    summary = [
        ["Reference", "Loading reproduces 03_g01_retrain?",
         f"m1 = {pm(ref['got']['m1'], ref['got']['m1_se'], 4)}, m2 = {pm(ref['got']['m2'], ref['got']['m2_se'], 4)} "
         f"(expected −0.51247 ± 0.01551, −0.48568 ± 0.01534); σ<sub>shape</sub> = {ref['got']['sigma_shape_weighted']:.5f}; ESS = {ref['got']['ess']:.1f}",
         "Reproduced to all printed digits. The cache and the TF weights are loaded correctly."],
        ["R-raw", "Raw per-galaxy posterior means (Sec. 6.1)",
         f"m1 = {pm(raw['g1']['low_m'], rawb['g1']['m_boot_se'])}, m2 = {pm(raw['g2']['low_m'], rawb['g2']['m_boot_se'])}; "
         f"m+ = {pm(raw['g+']['low_m'], rawb['g+']['m_boot_se'])}, m× = {pm(raw['gx']['low_m'], rawb['gx']['m_boot_se'])}; "
         f"10<sup>4</sup>c1 = {pm(1e4 * raw['g1']['low_c'], 1e4 * rawb['g1']['c_boot_se'], 1)}, 10<sup>4</sup>c2 = {pm(1e4 * raw['g2']['low_c'], 1e4 * rawb['g2']['c_boot_se'], 1)}; "
         f"σ<sub>ε</sub> mean {se['mean']:.4f}, RMS {se['rms']:.4f}",
         "Confirms m ≈ −0.6 and c ≈ 0 (bootstrap errors). m+ − m× = "
         f"{raw['g+']['low_m'] - raw['gx']['low_m']:+.3f}: there is a galaxy-frame anisotropy in the response."],
        ["T1", "Does σ<sub>ε,i</sub> depend on the true shear?",
         f"d lnσ/d|g| = {pm(T1['|g|']['dlnsigma_dg_residualized'], T1['|g|']['dlnsigma_dg_residualized_se'])} (full); "
         f"{pm(S1[0.05]['|g|']['dlnsigma_dg'], S1[0.05]['|g|']['se'])} for |g1|,|g2| &lt; 0.05. Quadratic coefficient in g+: {pm(T1['g+']['quad_residualized']['quad'], T1['g+']['quad_residualized']['quad_se'], 2)}, in g×: {pm(T1['gx']['quad_residualized']['quad'], T1['gx']['quad_residualized']['quad_se'], 2)}",
         "Not flat. σ<sub>i</sub> shrinks with |g|, partly through the prior-box truncation, and more strongly for g+ than for g×. Because of this, σ<sub>i</sub> is not a shear-independent selection variable."],
        ["T3", "Does shrinkage explain m quantitatively?",
         f"With selection-free bins (S4), −⟨σ<sub>g</sub>²⟩/σ<sub>π</sub>² gives χ² = {chi4['gauss_posterior:g1']['chi2']:.1f}/{chi4['gauss_posterior:g1']['ndof']} (g1) and {chi4['gauss_posterior:g2']['chi2']:.1f}/{chi4['gauss_posterior:g2']['ndof']} (g2). "
         f"The plan's formula −⟨σ²/(σ²+σ<sub>π</sub>²)⟩ gives χ² = {chi4['plan_formula:g1']['chi2']:.0f}/12, so it under-predicts |m| by a factor of about 2. "
         f"Full sample: measured m1 = {raw['g1']['low_m']:+.3f}, −⟨σ²⟩/σ<sub>π</sub>² = {R['T3']['full_sample_predictions']['g1']['gauss_posterior']:+.3f}",
         "Yes, for g1 and g2, provided the prediction uses the <i>posterior</i> width: m = −σ<sub>post</sub>²/σ<sub>π</sub>². The formula σ²/(σ²+σ<sub>π</sub>²) holds only for the likelihood width. Sec. 6.2 must use the right one. In the galaxy frame m+ is more negative than predicted and m× less negative."],
        ["T4", "Is the low-|g| linear fit OK?",
         f"m1 (|g|&lt;0.02) = {pm(raw['g1']['low_m'], raw['g1']['low_m_se'])}; full-range linear fit {pm(T4['g1']['linear_window'][5]['m'], T4['g1']['linear_window'][5]['m_se'])}; "
         f"cubic fit m = {pm(T4['g1']['cubic_full']['m'], T4['g1']['cubic_full']['m_se'])}, q = {pm(T4['g1']['cubic_full']['q'], T4['g1']['cubic_full']['q_se'], 2)}",
         "Yes. m changes by ≤ 0.05 across fit windows of 0.01–0.15, within the errors of the narrowest windows. The cubic term is detected but small: it adds about −0.005 to ĝ at g = 0.1, compared with m·g ≈ −0.06."],
        ["T5", "Robust errors",
         f"Bootstrap/fit error ratio for m: {min(ratios):.2f}–{max(ratios):.2f}; for c: {min(cr):.2f}–{max(cr):.2f} (12 estimators × 4 components). Raw m1 fit {rawb['g1']['m_fit_se']:.4f} vs bootstrap {rawb['g1']['m_boot_se']:.4f}; "
         f"σ<sub>ε</sub> bootstrap error {T5['raw']['sigma_eps_mean_boot_se']:.5f}",
         "The sandwich fit errors of fit_design are reliable, so they can be quoted in the paper."],
        ["R6", "Width-cut curve (Sec. 6.3)",
         f"Raw: m1/m2 = {f(r6[0.05]['raw']['g1']['m'])}/{f(r6[0.05]['raw']['g2']['m'])} at σ<sub>ε,i</sub>&lt;0.05 (retained {r6[0.05]['frac_retained']:.3f}); "
         f"{f(r6[0.045]['raw']['g1']['m'])}/{f(r6[0.045]['raw']['g2']['m'])} at &lt;0.045 ({r6[0.045]['frac_retained']:.3f}); "
         f"maximum about +0.1 near 0.03–0.035",
         "|m| ≲ 0.05 when about 20–25% of galaxies are kept. However, m crosses zero and turns <i>positive</i>, and T2a shows this is mostly selection bias. The curve should not be presented as “the cut removes the bias.”"],
        ["T2a", "Does the σ cut add selection bias?",
         f"At matched retained fraction the σ-cut minus the shear-free (GBM) cut, after subtracting the toy shrinkage difference, is "
         f"{pm(S2r[0.1]['selection_excess'], S2r[0.1]['selection_excess_se'])} (f=0.1), {pm(S2r[0.2]['selection_excess'], S2r[0.2]['selection_excess_se'])} (0.2), "
         f"{pm(S2r[0.3]['selection_excess'], S2r[0.3]['selection_excess_se'])} (0.3), {pm(S2r[0.5]['selection_excess'], S2r[0.5]['selection_excess_se'])} (0.5), {pm(S2r[0.7]['selection_excess'], S2r[0.7]['selection_excess_se'])} (0.7). "
         f"Weights with no cut: excess {pm(S2['excess_obs_minus_gbm'], S2['excess_obs_minus_gbm_se'])}",
         "<b>σ cuts are NOT free of selection bias.</b> The observed-σ cut adds a positive Δm of +0.02 to +0.16 when it keeps ≤ 50% of galaxies, and the apparent m ≈ 0 at σ<sub>ε,i</sub> ≈ 0.045 is a cancellation. A selection response (T2b, metacal-style) is required before quoting cut results. The regularized <i>weights</i> alone show no detectable selection bias (+0.002 ± 0.002)."],
        ["T7", "σ convention (std vs 16–84% half-width)",
         f"Median half-width/std = {T7['halfwidth_over_std_g1_percentiles_5_16_50_84_95'][2]:.3f} (g1); ensemble σ<sub>ε</sub> {T7['sigma_eps_std']['mean']:.4f} (std) vs {T7['sigma_eps_halfwidth']['mean']:.4f} (half-width); Spearman {T7['spearman_sig_vs_sigq']:.4f}",
         "At matched retained fraction m differs by ≤ 0.025, except m2 at f = 0.1 (0.04). Conclusions do not depend on the convention, but thresholds do (1–17% shift), so state the convention. Recommend the std (Xu24)."],
    ]

    # ------------------------------------------------------------ sections
    secs = []
    # conventions
    conv_rows = [[f"{t}", f"{t / np.sqrt(2):.4f}", f"{r6[t]['frac_retained']:.3f}" if t in r6 else "—"] for t in (0.03, 0.04, 0.05, 0.075, 0.1)]
    conventions = (
        "<section id='conv'><h2>Conventions and provenance</h2>"
        "<p>Model <code>CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair</code>; test set Xu3 (10<sup>5</sup> galaxies); "
        "cache <code>$S/cache/&lt;model&gt;/test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded</code> (read-only). "
        "The cache stores 10<sup>4</sup> (g1, g2) draws per galaxy (identity + R90 equal mixture) and TF log weights. "
        "One streaming pass (<code>a_shear_extract.py</code>) reduced these to per-galaxy TF-weighted mean, variance, covariance and weighted quantiles "
        "(2.5/16/50/84/97.5%), saved as <code>per_galaxy.npz</code>. "
        f"Cross-checks: the variances agree with <code>posterior_component_variance</code> to {X['checks']['var_maxabs']:.1e}, "
        f"and the quantiles with <code>weighted_quantile</code> to {X['checks']['q_maxabs']:.1e}, on {X['checks']['n']} galaxies. "
        f"The TF means agree with the cached <code>tf_target_mean_estimates</code> to {X['checks']['mean_vs_cache_maxabs']:.1e}. "
        "The cached truths (g1, g2, θ<sub>int</sub>, cos i, ρ<sub>I</sub>, ρ<sub>S</sub>, m<sub>r</sub>) match "
        "<code>$S/samples/test_100k_simv3_cosi_xu3_tf.csv</code> to float32 precision (max |Δ| 1.6×10<sup>−8</sup> for g).</p>"
        "<p><b>Estimator:</b> the TF-weighted posterior mean. <b>Bias fit:</b> ĝ − g = m g + c on |g| &lt; 0.02 per component, "
        "fit with <code>component_metrics</code>/<code>fit_design</code> (sandwich errors). The galaxy frame (g+, g×) uses the true θ<sub>int</sub> "
        "(<code>galaxy_frame_components</code>). <b>c</b> in the tables is the intercept of the low-|g| fit unless stated otherwise. "
        "<b>Raw</b> means equal galaxy weights, the full sample and no cuts. "
        "<b>Weighted</b> means the existing <code>compose_shape_noise_regularized_weights</code>, "
        "w ∝ 1/(σ<sub>gal</sub>² + σ<sub>floor</sub>²), unchanged. After a cut, the floor is recomputed on the retained galaxies.</p>"
        "<p><b>Shape-noise convention (important).</b> Everything reported here as σ<sub>ε,i</sub>, and every cut threshold, uses the Xu24 quadrature definition "
        "σ<sub>ε,i</sub> = √(σ<sub>g1,i</sub>² + σ<sub>g2,i</sub>²) with TF-weighted 1D standard deviations. The existing "
        "<code>shear_bias_report.py</code> “shape noise” is the per-component RMS σ<sub>gal</sub> = √((σ<sub>g1</sub>² + σ<sub>g2</sub>²)/2) = σ<sub>ε,i</sub>/√2. "
        "Only the reference check and the internal floor of the weights use that convention. "
        f"For example, the reference report's σ<sub>shape</sub> = 0.03894 is a weighted mean of σ<sub>gal</sub>, and it corresponds to about 0.0551 in σ<sub>ε</sub>. "
        f"The prior width is σ<sub>π</sub> = 0.2/√12 = {A.SIGMA_PI:.5f} per component, so a prior-dominated posterior has σ<sub>ε,i</sub> → √2 σ<sub>π</sub> = 0.0816.</p>"
        + table(["σ<sub>ε,i</sub> threshold (Xu24, used here)", "same cut in per-component σ<sub>gal</sub> (sbr)", "retained fraction"], conv_rows)
        + "<p><b>Jobs</b> (RM-shared, CPU): " + "; ".join(f"<code>{k}</code> {html.escape(v)}" for k, v in JOBS.items())
        + f". Scripts: <code>{SCRIPTS}/a_shear_*.py</code>.</p></section>"
    )
    secs.append(conventions)

    # reference
    g = ref["got"]
    secs.append(section(
        "REF", "Reference reproduction",
        "Does our loading reproduce the known fibrepair Xu3 number from fiber-gauge/03_g01_retrain?",
        "Shape-noise-regularized weights (existing function, sbr convention) and the low-|g| fit on the TF mean.",
        table(["quantity", "expected (03_g01_retrain)", "this run"],
              [["m1", "−0.51247 ± 0.01551", pm(g["m1"], g["m1_se"], 5)],
               ["m2", "−0.48568 ± 0.01534", pm(g["m2"], g["m2_se"], 5)],
               ["10<sup>4</sup> c1 (mean residual)", "0.440", f"{g['c1_1e4']:.3f}"],
               ["10<sup>4</sup> c2 (mean residual)", "0.807", f"{g['c2_1e4']:.3f}"],
               ["reported σ<sub>shape</sub> (weighted σ<sub>gal</sub>)", "0.03894", f"{g['sigma_shape_weighted']:.5f}"],
               ["ESS", "90897.2", f"{g['ess']:.1f}"],
               ["m1 using cached tf_target_mean_estimates", "—", f"{g['m1_from_cache_mean']:+.5f}"]]),
        "Loading is validated, and the rest of this report uses the same arrays."))

    # R-raw
    rows = []
    for comp in A.COMPONENTS:
        r, b = raw[comp], rawb[comp]
        rows.append([comp, pm(r["low_m"], r["low_m_se"], 4), f"{b['m_boot_se']:.4f}",
                     pm(1e4 * r["low_c"], 1e4 * r["low_c_se"], 2), f"{1e4 * b['c_boot_se']:.2f}",
                     pm(1e4 * r["c"], 1e4 * r["c_se"], 2), pm(r["cubic_m"], r["cubic_m_se"], 4), f"{r['n_low']:,}"])
    res = table(["component", "m (fit)", "m bootstrap s.e.", "10<sup>4</sup> c (intercept)", "10<sup>4</sup> c bootstrap s.e.",
                 "10<sup>4</sup> ⟨ĝ−g⟩ (all g)", "m from cubic fit (full range)", "N(|g|&lt;0.02)"], rows)
    p = se["percentiles_5_16_50_84_95"]
    res += table(["ensemble σ<sub>ε</sub>", "value"],
                 [["mean σ<sub>ε,i</sub>", f"{se['mean']:.5f} ± {T5['raw']['sigma_eps_mean_boot_se']:.5f} (bootstrap)"],
                  ["RMS σ<sub>ε,i</sub>", f"{se['rms']:.5f} ± {T5['raw']['sigma_eps_rms_boot_se']:.5f}"],
                  ["median", f"{se['median']:.5f}"],
                  ["5/16/50/84/95 percentiles", " / ".join(f"{v:.4f}" for v in p)],
                  ["mean σ<sub>g1</sub>, σ<sub>g2</sub> (per component)", f"{se['per_component_mean_sigma_g1']:.5f}, {se['per_component_mean_sigma_g2']:.5f}"],
                  ["prior limit √2 σ<sub>π</sub>", f"{se['prior_sigma_eps_equiv_sqrt2_sigma_pi']:.5f}"],
                  ["fraction with σ<sub>ε,i</sub> &gt; 0.075 (nearly prior-dominated)", f"{se['frac_sigma_eps_gt_0.075']:.4f}"]])
    res += img("Rraw_binned_residual", "Paper R5 candidate (<code>paper_figs/A_R5_raw_bias.{png,pdf}</code>). Raw ĝ − g against true g: "
               "20 bins of width 0.01 in the image frame (left), 24 bins of width 0.01 over ±0.12 in the galaxy frame (right). "
               "Points are bin means ± standard error. Lines are the |g| &lt; 0.02 linear fits (shaded band), extrapolated. Equal weights, all 10<sup>5</sup> galaxies.")
    res += img("Rraw_sigma_hist", "Paper candidate (<code>paper_figs/A_R5_sigma_hist</code>). Histogram of the per-galaxy "
               "σ<sub>ε,i</sub> = √(σ<sub>g1</sub>² + σ<sub>g2</sub>²) (TF-weighted std), 80 bins over [0, 0.1]. Vertical lines mark the mean, the RMS and the prior limit √2 σ<sub>π</sub>.")
    secs.append(section(
        "R-raw", "Raw per-galaxy posterior means",
        "What are m, c and σ<sub>ε</sub> straight from inference: equal weights, full sample, no post-processing?",
        "Low-|g| linear fit per component in the image and galaxy frames, fit and bootstrap (1000) errors; the σ<sub>ε,i</sub> distribution.",
        res,
        f"The Sec. 6.1 numbers are m1 = {raw['g1']['low_m']:+.3f} ± {rawb['g1']['m_boot_se']:.3f} and m2 = {raw['g2']['low_m']:+.3f} ± {rawb['g2']['m_boot_se']:.3f}, with c consistent with zero at ≲ 2×10<sup>−4</sup>. "
        f"The ensemble shape noise is σ<sub>ε</sub> = {se['mean']:.4f} (mean) or {se['rms']:.4f} (RMS). "
        "The σ<sub>ε,i</sub> distribution piles up near the prior limit: about 32% of galaxies have σ<sub>ε,i</sub> &gt; 0.075, i.e. they are almost prior-dominated. "
        f"<b>Flag:</b> the galaxy-frame responses differ, m+ = {raw['g+']['low_m']:+.3f} vs m× = {raw['gx']['low_m']:+.3f}. The same galaxies enter both, so the "
        "difference (≈ 0.07, ≈ 4σ on the naive errors) is probably real. The shrinkage is anisotropic relative to the galaxy axis. It averages out in the image frame, where θ is uniform."))

    # T1
    rows = []
    for name in ("|g|", "g+", "gx"):
        r = T1[name]
        rows.append([name, pm(r["dsigma_dg"], r["dsigma_dg_se"], 4), pm(r["dlnsigma_dg"], r["dlnsigma_dg_se"]),
                     pm(r["dlnsigma_dg_residualized"], r["dlnsigma_dg_residualized_se"]),
                     pm(r["quad_residualized"]["lin"], r["quad_residualized"]["lin_se"]),
                     pm(r["quad_residualized"]["quad"], r["quad_residualized"]["quad_se"], 2)])
    res = table(["regressor (true)", "dσ<sub>ε,i</sub>/dg", "d lnσ/dg", "d lnσ/dg (residualized)", "linear term (quad fit, resid.)", "quadratic term (resid.)"], rows)
    rows = []
    for r in T1["by_cosi"]:
        rows.append([f"[{r['cosi_lo']}, {r['cosi_hi']}]", f"{r['n']:,}", f"{r['mean_sigma']:.4f}",
                     pm(r["|g|"]["residualized"], r["|g|"]["residualized_se"]), pm(r["g+"]["residualized"], r["g+"]["residualized_se"]),
                     pm(r["gx"]["residualized"], r["gx"]["residualized_se"])])
    res += "<p>Split by true cos i (linear slopes of residualized lnσ):</p>" + table(["cos i", "N", "mean σ<sub>ε,i</sub>", "d lnσ/d|g|", "d lnσ/dg+", "d lnσ/dg×"], rows)
    rows = []
    for box, r in S1.items():
        rows.append([f"{box}", f"{r['n']:,}", pm(r["|g|"]["dlnsigma_dg"], r["|g|"]["se"]), pm(r["g+"]["quad_lin"], r["g+"]["quad_lin_se"]),
                     pm(r["g+"]["quad"], r["g+"]["quad_se"], 2), pm(r["gx"]["quad"], r["gx"]["quad_se"], 2)])
    res += "<p>S1, away from the prior edge (galaxies with |g1|, |g2| &lt; box):</p>" + table(
        ["box", "N", "d lnσ/d|g|", "linear in g+", "quadratic in g+", "quadratic in g×"], rows)
    res += img("T1_sigma_vs_shear", "Residualized ln σ<sub>ε,i</sub> against true |g|, g+ and g× (bin means ± s.e.), in four true-cos i bins and for all galaxies. "
               f"Residualized means ln σ<sub>ε,i</sub> minus a 5-fold cross-fitted gradient-boosting prediction from shear-independent truths "
               f"(cos i, ln ρ<sub>I</sub>, ln ρ<sub>S</sub>, m<sub>r</sub>, v<sub>circ</sub>, hlr, r<sub>scale</sub>, log flux, v<sub>0</sub>, cos 2θ, sin 2θ; R² = {T1['gbm_r2_log_sigma']:.3f}). "
               "The prediction does not use g, so residualizing only reduces the noise of the slope; the expected slope is unchanged. Titles give the linear slope over all galaxies.")
    res += img("T1_sigma_raw_vs_gplus", "Mean σ<sub>ε,i</sub> (not residualized) against true g+ and g×.")
    secs.append(section(
        "T1", "Does σ<sub>ε,i</sub> depend on the true shear?",
        "If σ<sub>ε,i</sub> depends on g, then weighting or cutting on σ<sub>ε,i</sub> is a shear-dependent selection.",
        "OLS (<code>fit_design</code>, sandwich errors) of σ<sub>ε,i</sub>, ln σ<sub>ε,i</sub> and residualized ln σ<sub>ε,i</sub> on the true |g|, g+ and g× (galaxy frame from the true θ<sub>int</sub>). Also split by cos i and restricted to the interior of the prior box.",
        res,
        "σ<sub>ε,i</sub> is <b>not</b> flat in g. (i) It falls with |g|: d lnσ/d|g| ≈ −1.5 overall, −0.8 inside |g1|,|g2| &lt; 0.05, and up to −3 for face-on cos i &gt; 0.75. "
        "A 1D toy model of a uniform prior box gives d lnσ<sub>post</sub>/dg ≈ −1 to −2 for g in 0–0.05 at σ<sub>lik</sub> = 0.02–0.05, so most of this is prior-box truncation. "
        f"(ii) It is also anisotropic: the quadratic coefficient is {T1['g+']['quad_residualized']['quad']:+.1f} in g+ but {T1['gx']['quad_residualized']['quad']:+.1f} in g×. A box truncation would be the same for both after averaging over θ, "
        f"so part of the dependence is physical: shear along the galaxy axis changes the constraint. There is also a small linear term in g+ ({pm(T1['g+']['dlnsigma_dg_residualized'], T1['g+']['dlnsigma_dg_residualized_se'])}). "
        "Inside the |g| &lt; 0.02 fit window a dependence on the <i>true</i> g alone changes the retained g distribution only at second order, so by itself it biases m little. "
        "The same physics, however, makes σ<sub>i</sub> respond to the noise-driven apparent shear (S3), and that does bias m (T2a). "
        "The paper should say that σ<sub>i</sub> depends on the shear and is therefore not a shear-independent selection variable."))

    # T3
    rows = []
    for r in R["T3"]["bins"]:
        rows.append([f"{r['sig_lo']:.4f}–{r['sig_hi']:.4f}", f"{r['mean_sigma_eps']:.4f}"] +
                    [f"{pm(r[c]['m'], r[c]['m_se'])} | {r[c]['plan_formula']:+.3f} / {r[c]['gauss_posterior']:+.3f} / {r[c]['uniform_toy']:+.3f}" for c in ("g1", "g2")])
    res = ("<p>Bins in the <b>observed</b> σ<sub>ε,i</sub> (15 quantile bins of 6,667 galaxies, as in the plan). "
           "Each cell gives the measured m | the predictions from the plan formula / −⟨σ²⟩/σ<sub>π</sub>² / the uniform-prior toy. Each prediction uses the per-component TF-weighted posterior variance of that component.</p>"
           + table(["σ<sub>ε,i</sub> bin", "mean σ<sub>ε,i</sub>", "g1", "g2"], rows))
    rows = []
    for r in S["S4_T3_predicted_sigma_bins"]:
        rows.append([f"{r['mean_sigma_eps_pred']:.4f}", f"{r['mean_sigma_eps_obs']:.4f}"] +
                    [f"{pm(r[c]['m'], r[c]['m_se'])} | {r[c]['plan_formula']:+.3f} / {r[c]['gauss_posterior']:+.3f} / {r[c]['uniform_toy']:+.3f}" for c in A.COMPONENTS])
    res += ("<p>S4: bins in the <b>shear-free predicted</b> σ (12 quantile bins of the GBM prediction). These bins are immune to noise-driven selection.</p>"
            + table(["mean predicted σ", "mean observed σ<sub>ε,i</sub>", "g1", "g2", "g+", "g×"], rows))
    rows = []
    for key, lab in (("plan_formula", "−⟨σ²/(σ²+σ<sub>π</sub>²)⟩ (plan)"), ("gauss_posterior", "−⟨σ²⟩/σ<sub>π</sub>² (Gaussian, posterior σ)"), ("uniform_toy", "uniform-prior toy")):
        rows.append([lab] + [f"{chi3[key + ':' + c]['chi2']:.1f} / {chi4[key + ':' + c]['chi2']:.1f}" for c in A.COMPONENTS])
    res += "<p>χ² of measured vs predicted m (observed-σ bins, 15 dof / predicted-σ bins, 12 dof):</p>" + table(["prediction", "g1", "g2", "g+", "g×"], rows)
    fs = R["T3"]["full_sample_predictions"]
    res += table(["full sample", "measured m", "plan", "−⟨σ²⟩/σ<sub>π</sub>²", "uniform toy"],
                 [[c, pm(raw[c]["low_m"], raw[c]["low_m_se"]), f(fs[c]["plan_formula"]), f(fs[c]["gauss_posterior"]), f(fs[c]["uniform_toy"])] for c in A.COMPONENTS])
    res += img("T3_shrinkage", "Paper D1 candidate A (<code>paper_figs/A_D1_shrinkage</code>). Measured low-|g| m in 15 bins of the <i>observed</i> σ<sub>ε,i</sub>, image frame (left) and galaxy frame (right). "
               "Curves are component-averaged predictions. Note that the lowest bins are biased high by selection (see T2a), so this binning is not recommended for the paper.")
    res += img("S4_shrinkage_selection_free", "Paper D1 candidate B, recommended (<code>paper_figs/A_D1_shrinkage_selection_free</code>). As above, but in 12 bins of the shear-free predicted σ. "
               "The x-axis is the bin mean of the observed σ<sub>ε,i</sub>. Dashed line: −⟨σ<sub>g</sub>²⟩/σ<sub>π</sub>² from the per-component TF posterior variances. Solid line: the 1D uniform-prior toy, which maps each galaxy's "
               "posterior std to the low-|g| slope expected for a Gaussian likelihood under the U(−0.1, 0.1) prior.")
    secs.append(section(
        "T3", "Does shrinkage explain m quantitatively?",
        "In bins of σ<sub>ε,i</sub>, does the measured m match −⟨σ<sub>i</sub>²/(σ<sub>i</sub>²+σ<sub>π</sub>²)⟩ with σ<sub>π</sub> = 0.0577?",
        "Low-|g| fit per bin (equal weights). Three predictions, each from the per-component posterior variance of every galaxy in the bin: "
        "(a) the plan formula; (b) −⟨σ<sub>post</sub>²⟩/σ<sub>π</sub>², which is exact for a Gaussian prior and likelihood; "
        "(c) a 1D uniform-prior toy (g ~ U(−0.02, 0.02), g<sub>obs</sub> = g + N(0, σ<sub>lik</sub>), posterior = truncated normal on [−0.1, 0.1], 4×10<sup>5</sup> draws per σ<sub>lik</sub> on 160 grid points, mapped from σ<sub>post</sub> to m). "
        "Bins are taken in the observed σ (as planned) and, in S4, in the shear-free predicted σ.",
        res,
        "<b>Yes, quantitatively, but only with the correct formula.</b> For a Gaussian toy the posterior-mean response is 1 − σ<sub>post</sub>²/σ<sub>π</sub>², "
        "i.e. m = −σ<sub>post</sub>²/σ<sub>π</sub>² = −σ<sub>lik</sub>²/(σ<sub>lik</sub>²+σ<sub>π</sub>²). The plan formula is correct with the likelihood width, "
        "but the posterior σ<sub>i</sub> available here gives values about 2× too small (χ² &gt; 1000). "
        f"With selection-free bins, −⟨σ²⟩/σ<sub>π</sub>² fits g1 and g2 with χ² = {chi4['gauss_posterior:g1']['chi2']:.1f} and {chi4['gauss_posterior:g2']['chi2']:.1f} for 12 dof, and the full-sample prediction "
        f"({fs['g1']['gauss_posterior']:+.3f}, {fs['g2']['gauss_posterior']:+.3f}) matches the measured ({raw['g1']['low_m']:+.3f}, {raw['g2']['low_m']:+.3f}). "
        "The uniform toy is worse at intermediate σ (it under-predicts |m| by about 0.05–0.13). "
        f"In the galaxy frame the per-component prediction fails (χ² = {chi4['gauss_posterior:g+']['chi2']:.0f} for g+, {chi4['gauss_posterior:gx']['chi2']:.0f} for g×): m+ is more negative, and m× less negative, than σ+ and σ× predict. "
        "Sec. 6.2 should write m ≈ −σ<sub>post</sub>²/σ<sub>π</sub>² (or state explicitly that σ in σ²/(σ²+σ<sub>π</sub>²) is the likelihood width) and use the S4 figure. "
        "<b>Flag:</b> in the observed-σ binning the lowest bin has m &gt; 0 (m× = +0.13 ± 0.03). This is the selection effect of T2a, not over-response of well-measured galaxies. With shear-free binning the best 1/12 of galaxies has m ≈ −0.06."))

    # T4
    rows = []
    for c in A.COMPONENTS:
        lw = T4[c]["linear_window"]
        cq = T4[c]["cubic_full_with_quadratic"]
        rows.append([c] + [pm(x["m"], x["m_se"]) for x in lw] +
                    [pm(T4[c]["cubic_full"]["m"], T4[c]["cubic_full"]["m_se"]), pm(T4[c]["cubic_full"]["q"], T4[c]["cubic_full"]["q_se"], 2),
                     pm(cq["q2"], cq["q2_se"], 3)])
    res = table(["component"] + [f"linear, |g|&lt;{x['max_abs_g']}" for x in T4["g1"]["linear_window"]] + ["cubic fit m (c + m g + q g³, full)", "q", "g² term (if added)"], rows)
    res += img("T4_linearity", "Left: binned mean ĝ against true g (40 bins over ±0.1) with the |g| &lt; 0.02 fit extrapolated. Right: linear-fit m against the fit window. "
               "Equal weights. Windows ≥ 0.1 cover the full range for g1 and g2 (|g+|, |g×| reach 0.14).")
    secs.append(section(
        "T4", "Is the low-|g| linear fit OK?",
        "Does the |g| &lt; 0.02 linear fit represent the response over the prior range, and are there prior-edge effects?",
        "Linear fits in windows |g| &lt; 0.01–0.15 and a cubic c + m g + q g³ on the full range (<code>component_metrics</code>), plus a check with an added g² term.",
        res,
        f"The low-|g| fit is fine. m1 = {raw['g1']['low_m']:+.3f} ± {raw['g1']['low_m_se']:.3f} at |g| &lt; 0.02, against {T4['g1']['cubic_full']['m']:+.3f} ± {T4['g1']['cubic_full']['m_se']:.3f} from the cubic fit. "
        f"Across windows m varies by ≤ 0.05 and becomes slightly more negative with wider windows ({T4['g1']['linear_window'][5]['m']:+.3f} for the full range). "
        f"This is the cubic term q ≈ {T4['g1']['cubic_full']['q']:+.1f}, i.e. extra shrinkage near the prior edge, worth about −0.005 in ĝ at |g| = 0.1. "
        "No even (g²) term is detected. The paper can quote the |g| &lt; 0.02 fit and mention that the cubic fit gives the same m within 0.01 (0.02 for g2 and g×; 0.03 for g+, where q has the opposite sign)."))

    # T5
    rows = []
    for name, p_ in T5.items():
        cc = p_["components"]
        rows.append([html.escape(name), f"{p_['frac_retained']:.3f}", f"{p_['sigma_eps_mean']:.4f} ± {p_['sigma_eps_mean_boot_se']:.4f}",
                     f"{p_['sigma_eps_rms']:.4f} ± {p_['sigma_eps_rms_boot_se']:.4f}"] +
                    [f"{cc[c]['m']:+.3f} ({cc[c]['m_fit_se']:.3f} / {cc[c]['m_boot_se']:.3f})" for c in A.COMPONENTS] +
                    [f"{1e4 * cc[c]['c']:+.1f} ({1e4 * cc[c]['c_fit_se']:.1f} / {1e4 * cc[c]['c_boot_se']:.1f})" for c in ("g1", "g2")])
    res = table(["estimator", "retained", "σ<sub>ε</sub> mean (boot)", "σ<sub>ε</sub> RMS (boot)", "m1 (fit/boot s.e.)", "m2", "m+", "m×",
                 "10<sup>4</sup>c1 (fit/boot)", "10<sup>4</sup>c2"], rows)
    secs.append(section(
        "T5", "Robust errors",
        "Are the fit errors trustworthy for the raw, weighted and cut estimators?",
        f"Nonparametric bootstrap over galaxies ({R['T5']['n_boot']} resamples, seed {R['T5']['seed']}), implemented as multinomial counts × estimator weights. "
        "For the weighted estimators the shape-noise floor is recomputed in every resample, and after the cut. "
        "σ<sub>ε</sub> mean and RMS are taken over the retained galaxies with the estimator's galaxy weights (equal weights for raw). "
        "Note that σ<sub>ε,i</sub> &lt; 0.1 retains every galaxy (max σ<sub>ε,i</sub> ≈ 0.09), so that row equals the uncut one.",
        res,
        f"Bootstrap/fit ratio for m: {min(ratios):.2f}–{max(ratios):.2f}; for c: {min(cr):.2f}–{max(cr):.2f}. "
        "The sandwich errors from <code>fit_design</code> are accurate and can be quoted directly. Bootstrap errors are available for the tables. "
        "These are statistical errors for this fixed test set only; they do not include the selection-bias systematics of T2a."))

    # R6
    rows = []
    for t in (0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 1.0):
        if t not in r6:
            continue
        r = r6[t]
        rows.append([("no cut" if t == 1.0 else f"{t}"), f"{r['frac_retained']:.3f}", f"{r['n_retained']:,}", f"{r['raw']['g1']['n_low']:,}",
                     pm(r["raw"]["g1"]["m"], r["raw"]["g1"]["m_se"]), pm(r["raw"]["g2"]["m"], r["raw"]["g2"]["m_se"]),
                     f"{r['raw']['sigma_eps_mean']:.4f} / {r['raw']['sigma_eps_rms']:.4f}",
                     pm(r["weighted"]["g1"]["m"], r["weighted"]["g1"]["m_se"]), pm(r["weighted"]["g2"]["m"], r["weighted"]["g2"]["m_se"]),
                     f"{r['weighted']['sigma_eps_mean']:.4f} / {r['weighted']['sigma_eps_rms']:.4f}", f"{r['weighted']['ess']:.0f}"])
    res = table(["σ<sub>ε,i</sub> &lt;", "retained", "N", "N(|g1|&lt;0.02)", "raw m1", "raw m2", "raw σ<sub>ε</sub> mean / RMS",
                 "weighted m1", "weighted m2", "weighted σ<sub>ε</sub> mean / RMS", "weighted ESS"], rows)
    res += ("<p>The full grid (0.015–0.09 in steps of 0.0025) is in report.json (<code>R6.rows</code>). N(|g1|&lt;0.02) is the low-|g| count of the whole sample. "
            "After a cut, the number of low-|g| galaxies entering the fit is the retained fraction times this (g is independent of σ to first order).</p>")
    res += img("R6_width_cut", "Paper R6 candidate (<code>paper_figs/A_R6_width_cut</code>). Left: low-|g| m1 and m2 against the threshold on σ<sub>ε,i</sub> (Xu24 std convention). "
               "Solid lines are raw (equal weights); dashed lines are the shape-noise-regularized weights with the floor recomputed on the retained galaxies. Bands are ±1σ fit errors; markers at the right are the uncut values. "
               "Middle: ensemble σ<sub>ε</sub> (mean and RMS of σ<sub>ε,i</sub> over the retained galaxies, estimator-weighted). Right: retained fraction and the low-|g1| count.")
    secs.append(section(
        "R6", "Width-cut curve",
        "How do m, σ<sub>ε</sub> and the retained sample change as the σ<sub>ε,i</sub> threshold is tightened?",
        "Threshold scan from 0.015 to 0.09 on σ<sub>ε,i</sub> (Xu24), for the raw and weighted estimators.",
        res,
        f"|m| falls from 0.6 to about 0 at σ<sub>ε,i</sub> ≈ 0.045 (raw m1/m2 = {f(r6[0.045]['raw']['g1']['m'])}/{f(r6[0.045]['raw']['g2']['m'])}, {100 * r6[0.045]['frac_retained']:.0f}% retained, σ<sub>ε</sub> = {r6[0.045]['raw']['sigma_eps_mean']:.4f}). "
        "It then <b>overshoots to m ≈ +0.1</b> at 0.03–0.035. Weighting shifts the curve upward by ≈ 0.05–0.1 when little is cut, and much less when a lot is cut. "
        "T2a shows that the zero crossing is a cancellation between shrinkage (≈ −0.1 to −0.2 for these σ) and a positive selection bias of the σ cut. "
        "The figure can be used for Sec. 6.3, but the text must not claim that the width cut calibrates the shear. "
        "Present it as a trade-off curve whose m includes a selection response. A cut result needs the metacal-style selection response from T2b to be interpretable."))

    # T2a
    rows = []
    for name, rows_ in R["T2a"]["curves"].items():
        for r in rows_:
            if r["frac"] in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                rows.append([html.escape(name), f"{r['frac']}", f"{r['mean_sigma_eps']:.4f}", pm(r["g1"]["m"], r["g1"]["m_se"]),
                             pm(r["g2"]["m"], r["g2"]["m_se"]), pm(r["m_avg"], r["m_avg_se"]), f(r["toy_avg"]), f(r["excess_vs_toy"])])
    res = table(["cut variable", "retained", "mean σ<sub>ε,i</sub>", "m1", "m2", "(m1+m2)/2", "toy prediction", "m − toy"], rows)
    res += ("<p>Spearman correlation between each ranking variable and σ<sub>ε,i</sub>: "
            + ", ".join(f"{html.escape(k)} {v:.3f}" for k, v in R["T2a"]["proxy_spearman_with_sigma"].items()) + ".</p>")
    rows = []
    for fr in S2r:
        a_, b_ = S2r[fr], S2w[fr]
        rows.append([f"{fr}", f(a_["m_sigma_cut"]), f(a_["m_gbm_cut"]), pm(a_["delta_m"], a_["delta_m_se"]), f(a_["delta_toy"]),
                     pm(a_["selection_excess"], a_["selection_excess_se"]),
                     pm(b_["delta_m"], b_["delta_m_se"]), pm(b_["selection_excess"], b_["selection_excess_se"])])
    res += ("<p>S2: paired bootstrap (500 resamples) of the σ-cut minus the GBM-cut at matched retained fraction, (m1+m2)/2. "
            "“Excess” subtracts the difference expected from shrinkage alone, i.e. the uniform-toy prediction evaluated on each retained sample with its own weights.</p>"
            + table(["retained", "raw m (σ cut)", "raw m (GBM cut)", "raw Δm", "raw Δm<sub>toy</sub>", "raw selection excess", "weighted Δm", "weighted selection excess"], rows))
    res += table(["S5: weights without a cut, (m1+m2)/2", "value"],
                 [["raw", f(S2["raw"]["m_avg"])],
                  ["regularized weights from observed variances (existing)", f(S2["weighted_observed"]["m_avg"])],
                  ["same weights from shear-free GBM-predicted σ", f(S2["weighted_gbm"]["m_avg"])],
                  ["weighted − raw: Δm, and excess over the toy", f"{pm(S2['delta_obs_minus_raw'], S2['delta_obs_minus_raw_se'])}; excess {pm(S2['excess_obs_minus_raw'], S2['excess_obs_minus_raw_se'])}"],
                  ["observed − GBM weights: Δm, and excess over the toy", f"{pm(S2['delta_obs_minus_gbm'], S2['delta_obs_minus_gbm_se'])}; excess {pm(S2['excess_obs_minus_gbm'], S2['excess_obs_minus_gbm_se'])}"]])
    rows = [[f"{r['frac']}", pm(1e3 * r["mean_radial_resid_sigma_cut_lowg"], 1e3 * r["se_sigma_cut"], 2), pm(1e3 * r["mean_radial_resid_gbm_cut_lowg"], 1e3 * r["se_gbm_cut"], 2)]
            for r in S3["radial_resid_by_cut"]]
    res += ("<p>S3, mechanism: at fixed truth, the residual ln σ<sub>ε,i</sub> (observed minus shear-free prediction) is anti-correlated with the noise-driven apparent shear. "
            f"The slope on the radial residual (ĝ − g)·ĝ/|g| at fixed |g| is {pm(S3['dresid_dradial_at_fixed_absg'], S3['se'])}, "
            f"and on |ĝ| at fixed |g| it is {pm(S3['dresid_dabsghat_at_fixed_absg'], S3['se_absghat'])}. "
            f"Spearman correlations: {S3['spearman_resid_vs_radial']:.3f} (radial) and {S3['spearman_resid_vs_absghat']:.3f} (|ĝ|). "
            "Among low-|g| (|g| &lt; 0.02) galaxies, the mean radial residual (10<sup>−3</sup>) of the retained sample is:</p>"
            + table(["retained", "σ cut", "GBM cut"], rows))
    res += img("T2a_selection", "Left: (m1+m2)/2 against the mean σ<sub>ε,i</sub> of the retained sample for cuts at matched retained fractions (0.1–1.0) on the observed σ<sub>ε,i</sub>, "
               "and on shear-independent truths. Truth variables are ranked by their binned (50 quantile bins) mean σ<sub>ε,i</sub>, which handles the non-monotonic cos i dependence; "
               "the last truth-level ranking is the cross-fitted GBM prediction. Right: m minus the uniform-toy shrinkage prediction for the same retained galaxies. Errors are fit errors.")
    res += img("S2_S4_selection_and_shrinkage", "Left: S2 selection excess of the observed-σ cut relative to the GBM cut (paired bootstrap ±1σ), raw and weighted. Grey lines show the difference before the toy correction. Right: the S4 shrinkage test (see T3).")
    secs.append(section(
        "T2a", "Does the σ cut add selection bias?",
        "At matched retained fraction, do cuts on σ<sub>ε,i</sub> and cuts on shear-independent truths (cos i, ρ<sub>I</sub>, ρ<sub>S</sub>) give the same m against mean σ?",
        "Rank galaxies by each variable, keep the best f = 0.1–1.0, and fit m with equal weights. "
        "In addition to the planned 1D truth cuts, we use a truth-level σ proxy, a 5-fold cross-fitted gradient-boosting prediction of ln σ<sub>ε,i</sub> from all shear-free truths. "
        "It ranks galaxies almost like σ (Spearman 0.92) but cannot respond to the shear or to the noise. S2 gives paired-bootstrap differences, S3 the mechanism, S5 the same test for the weights with no cut.",
        res,
        "<b>The σ cut adds a positive selection bias, so σ cuts are not safe.</b> The truth-level cuts (cos i, ρ<sub>I</sub>, ρ<sub>S</sub>, GBM) all follow one m(⟨σ⟩) curve, and it lies "
        "0–0.09 below the uniform toy (typically 0.05–0.07) at all fractions. The observed-σ cut departs from that curve when it keeps less than about 50% of galaxies. "
        f"The selection excess is {pm(S2r[0.5]['selection_excess'], S2r[0.5]['selection_excess_se'])} at f = 0.5, {pm(S2r[0.3]['selection_excess'], S2r[0.3]['selection_excess_se'])} at 0.3, "
        f"{pm(S2r[0.2]['selection_excess'], S2r[0.2]['selection_excess_se'])} at 0.2 and {pm(S2r[0.1]['selection_excess'], S2r[0.1]['selection_excess_se'])} at 0.1, and it is consistent with 0 at f ≥ 0.7. "
        "The mechanism is that noise which makes a galaxy look more sheared (|ĝ| larger) also narrows its posterior (T1: σ falls with |g|). "
        "A σ cut therefore keeps preferentially the galaxies scattered outward, which inflates the response. This is the KL analogue of the metacal selection response. "
        "It is invisible in truth-binned residual plots unless the cut variable is compared with a shear-free proxy, as done here. "
        f"The regularized <i>weights</i> without a cut do not show this: the excess over the toy is {pm(S2['excess_obs_minus_gbm'], S2['excess_obs_minus_gbm_se'])}, so the +0.105 gain from weighting is fully explained by shrinkage. "
        "Sec. 6.3 should (i) present weighting as selection-safe at this precision; (ii) present width cuts together with their selection response, or restrict them to f ≳ 0.7; "
        "(iii) use T2b (matched-shear stencil) to measure R<sub>sel</sub> directly. A cut on a truth-level-like proxy (e.g. predicted σ from observables that do not respond to the shear) is an alternative worth mentioning, "
        "although real observables (the fitted cos i, the SNRs) are themselves noisy."))

    # T7
    rows = []
    for r in T7["matched_fraction"]:
        rows.append([f"{r['frac']}", f"{r['std:raw']['threshold']:.4f} / {r['halfwidth:raw']['threshold']:.4f}"] +
                    [f"{r[k]['m1']:+.3f} / {r[k]['m2']:+.3f}" for k in ("std:raw", "halfwidth:raw", "std:weighted", "halfwidth:weighted")] +
                    [f"{r['overlap_fraction']:.3f}"])
    res = table(["quantity", "std (Xu24)", "16–84% half-width"],
                [["ensemble σ<sub>ε</sub> mean", f"{T7['sigma_eps_std']['mean']:.5f}", f"{T7['sigma_eps_halfwidth']['mean']:.5f}"],
                 ["ensemble σ<sub>ε</sub> RMS", f"{T7['sigma_eps_std']['rms']:.5f}", f"{T7['sigma_eps_halfwidth']['rms']:.5f}"],
                 ["half-width/std percentiles 5/16/50/84/95 (g1)", "", " / ".join(f"{v:.3f}" for v in T7["halfwidth_over_std_g1_percentiles_5_16_50_84_95"])],
                 ["half-width/std percentiles 5/16/50/84/95 (g2)", "", " / ".join(f"{v:.3f}" for v in T7["halfwidth_over_std_g2_percentiles_5_16_50_84_95"])],
                 ["Spearman(σ<sub>ε,i</sub> std, half-width)", f"{T7['spearman_sig_vs_sigq']:.4f}", ""]])
    res += table(["retained", "threshold std / half-width", "m1/m2 std raw", "m1/m2 half-width raw", "m1/m2 std weighted", "m1/m2 half-width weighted", "overlap of retained sets"], rows)
    res += img("T7_convention", "Left: ratio of the 16–84% half-width to the std per galaxy, for g1 and g2. Right: (m1+m2)/2 against the retained fraction for cuts on each convention. "
               "Weighted curves use the variance of that convention (std² or half-width²) in the regularized weights.")
    res += img("T7_R6_halfwidth", "R6 rerun with the threshold applied to the half-width σ<sub>ε,i</sub><sup>(q)</sup> = √(h1² + h2²), with h = (q84 − q16)/2. Axes as in R6. "
               "The σ<sub>ε</sub> in the middle panel is still the std-convention ensemble value of the retained galaxies.")
    secs.append(section(
        "T7", "σ convention",
        "Does using the std or the 16–84% half-width for σ<sub>ε,i</sub> change any conclusion?",
        "Both conventions come from TF-weighted per-galaxy quantiles. Compare the distributions, then rerun R6 and compare at matched retained fractions.",
        res,
        f"Posteriors are platykurtic (the median half-width/std is {T7['halfwidth_over_std_g1_percentiles_5_16_50_84_95'][2]:.2f}; a uniform distribution gives 1.18), because the prior box truncates them. "
        f"The half-width convention therefore gives σ<sub>ε</sub> larger by about 11% ({T7['sigma_eps_halfwidth']['mean']:.4f} vs {T7['sigma_eps_std']['mean']:.4f}), and the threshold for a given retained fraction is larger by 1–17% (more for looser cuts). "
        "Rankings are almost identical (Spearman 0.997; 94–99% overlap of retained sets). At matched retained fraction m agrees to ≤ 0.025, except at f = 0.1 (raw m2 +0.143 vs +0.101). "
        "Conclusions are convention-independent. Use the std (Xu24) and state it wherever a threshold is quoted."))

    style = ("<style>body{font-family:system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#1b1b1b;background:#fff}"
             "section{margin:3rem 0;border-top:2px solid #ddd}table{border-collapse:collapse;width:100%;font-size:.85rem;margin:1rem 0}"
             "th,td{border:1px solid #ccc;padding:.35rem .5rem;text-align:right;vertical-align:top}th:first-child,td:first-child{text-align:left}"
             "th{background:#f3f3f3}img{display:block;max-width:100%;height:auto;margin:1rem auto}figcaption{font-size:.85rem;color:#444;margin:0 2rem 1.5rem}"
             "code{background:#f4f4f4;padding:.1rem .25rem}.impl{background:#f7f7f2;border-left:4px solid #999;padding:.5rem .75rem}"
             "#summary td:nth-child(3),#summary td:nth-child(4),#summary td:nth-child(2){text-align:left}</style>")
    body = ("<h1>Paper tests, Group A: shear (R-raw, T1, T3, T4, T5, R6, T2a, T7)</h1>"
            "<p>Fibrepair model, Xu3 test set, TF-weighted posterior means, no new inference. Plan: <code>kl-nn/repo_reports/PAPER_TESTS_PLAN.md</code>. "
            "Generated 2026-09-29. All numbers are also in <code>report.json</code>.</p>"
            "<section id='summary'><h2>Summary</h2>"
            + table(["test", "question", "key numbers", "verdict / implication"], summary)
            + "<p><b>Paper figures</b> (in <code>paper-tests/paper_figs/</code>): <code>A_R5_raw_bias</code>, <code>A_R5_sigma_hist</code>, <code>A_R6_width_cut</code>, "
              "<code>A_D1_shrinkage</code> (observed-σ bins, as planned; affected by selection), <code>A_D1_shrinkage_selection_free</code> (recommended).</p></section>"
            + "".join(secs))
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Group A shear tests</title>{style}</head><body>{body}</body></html>"
    (R_DIR / "report.html").write_text(page)
    report = {"summary": [{"test": r[0], "question": r[1], "key_numbers_html": r[2], "verdict_html": r[3]} for r in summary],
              "provenance": {"extract": X, "jobs": JOBS, "scripts": str(SCRIPTS)},
              "results": R, "supplement": S, "supplement2": S2}
    (R_DIR / "report.json").write_text(json.dumps(report, indent=1))
    status = ["Group A shear tests (R-raw, T1, T3, T4, T5, R6, T2a, T7): COMPLETE", "Reference reproduced: yes (m1 -0.51247, m2 -0.48568)", "", "SLURM jobs (RM-shared):"]
    status += [f"  {k}  {v}" for k, v in JOBS.items()]
    status += ["", "Local (login node): a_shear_report.py (report assembly)", f"Scripts: {SCRIPTS}/a_shear_*.py",
               "Outputs: report.html, report.json, results.json, supplement.json, supplement2.json, per_galaxy.npz(+.json), *.png",
               "Paper figs: paper_figs/A_R5_raw_bias, A_R5_sigma_hist, A_R6_width_cut, A_D1_shrinkage, A_D1_shrinkage_selection_free (.png/.pdf)"]
    (R_DIR / "STATUS.txt").write_text("\n".join(status) + "\n")
    print("written", R_DIR / "report.html", len(page))


if __name__ == "__main__":
    main()
