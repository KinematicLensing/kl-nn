#!/usr/bin/env python3
"""Paper figures and the estimator table for Sec. 6 ("Towards an NPE-based
Shear Estimator").

Outputs (paper_style.OUT = $S/reports/paper-tests/paper_figs_final/):
  fig_raw_bias.{pdf,png}     F4  binned ghat - g vs g, |g|<0.02 linear fit
  fig_shrinkage.{pdf,png}    F5  m vs <sigma_post^2> in selection-free width bins
  fig_width_cut.{pdf,png}    F6  m vs sigma_eps,i threshold, raw and weighted
  fig_response.{pdf,png}     F7  stencil response R vs sigma_post^2 and vs cos i
  fig_stacking.{pdf,png}     F8  m vs N per shear cell at one noise realization
  tab_estimators.tex         Tab. 2 (booktabs)
  sec6_numbers.json          every number shown in the figures / table
  captions_sec6.md           plain captions

Inputs (all read-only):
  A_shear/per_galaxy.npz        TF-weighted posterior mean / variance of (g1, g2),
                                 100k Xu3 galaxies (a_shear_extract.py)
  A_shear/supplement.json       S4 selection-free shrinkage bins (a_shear_supplement.py)
  B_inference/t2b_xu3like.json  Xu3-like matched-shear stencil: R(sigma) fit,
    + .tf_equal.cubes.npz        by-cos i blocks, selection response (b_t2b_analysis.py)
  C_stacking/report.json        stacking bias vs N at R=1 with paired galaxy
                                 bootstrap (c_stack_report.py)

Conventions: TF-weighted posterior mean; sigma_eps,i = sqrt(sg1^2 + sg2^2) from
TF-weighted posterior std; m, c from a linear fit of ghat - g on |g| < 0.02
(shear_bias_report.component_metrics, sandwich errors); image frame.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import shear_bias_report as sbr  # noqa: E402
from paper_style import COL, COLORS, DCOL, OUT, apply_style, savefig  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
PT = Path(os.environ.get("KLNN_PAPER_TESTS_ROOT", S / "reports/paper-tests"))
LOW_G = 0.02
SIGMA_PI2 = 0.2 ** 2 / 12.0
GREY = "#8c8c8c"
LIGHT = "#ededed"
NUM: dict = {}


# ------------------------------------------------------------------ helpers
def load_xu3():
    z = np.load(PT / "A_shear/per_galaxy.npz")
    g = z["truth"][:, :2].astype(float)
    e = z["tf_mean"].astype(float)
    var = z["tf_var"].astype(float)
    return dict(g=g, e=e, var=var, sd=np.sqrt(var), sig=np.sqrt(var.sum(1)), n=len(g))


def kish(w):
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def weights(D, kind, mask):
    """Galaxy weights: 'raw' = equal weights on the kept galaxies; 'weighted' =
    shape-noise-regularized inverse variance, floor recomputed on the kept set."""
    b = mask.astype(float)
    b /= b.sum()
    if kind == "raw":
        return b
    w, _ = sbr.compose_shape_noise_regularized_weights(b, D["var"][:, 0], D["var"][:, 1])
    return w


def fit(t, e, w):
    r = sbr.component_metrics(t, e, LOW_G, w)
    return r


def sigma_eff_component(t, e, w, r):
    """Noise per unit response of one component: sqrt(sum w^2 res^2 / sum w^2)/(1+m),
    res = e - c - (1+m) g, over the |g|<0.02 fit window. With this definition
    sigma_eff/sqrt(N_eff) is the standard error of the weighted mean shear after
    dividing by the response."""
    low = (np.abs(t) < LOW_G) & (w > 0)
    res = e[low] - r["low_c"] - (1 + r["low_m"]) * t[low]
    ww = w[low]
    return float(np.sqrt(np.sum(ww ** 2 * res ** 2) / np.sum(ww ** 2)) / (1 + r["low_m"]))


def binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    xc, mu, se = [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        xc.append(x[m].mean()); mu.append(y[m].mean()); se.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    return np.array(xc), np.array(mu), np.array(se)


def sci4(v, se):
    a = f"{1e4 * v:+.1f}"
    if a in ("+0.0", "-0.0"):
        a = "0.0"
    return f"{a} \\pm {1e4 * se:.1f}"


# ------------------------------------------------------------------ F4
def fig_raw_bias(D):
    edges = np.linspace(-0.1, 0.1, 21)
    w = np.full(D["n"], 1.0 / D["n"])
    fig, ax = plt.subplots(figsize=(COL, 0.78 * COL))
    ax.axvspan(-LOW_G, LOW_G, color=LIGHT, zorder=0, lw=0)
    ax.axhline(0, color=GREY, lw=0.6, zorder=1)
    out = {}
    for k, (name, col, mk, off) in enumerate((("g_1", COLORS["g1"], "o", -0.0012), ("g_2", COLORS["g2"], "s", 0.0012))):
        t, e = D["g"][:, k], D["e"][:, k]
        r = fit(t, e, w)
        xc, mu, se = binned(t, e - t, edges)
        xx_in = np.linspace(-LOW_G, LOW_G, 10)
        ax.plot(np.linspace(-0.1, 0.1, 50), r["low_c"] + r["low_m"] * np.linspace(-0.1, 0.1, 50),
                color=col, lw=0.8, ls=(0, (3, 2)), alpha=0.8, zorder=2)
        ax.plot(xx_in, r["low_c"] + r["low_m"] * xx_in, color=col, lw=1.6, zorder=3)
        lab = (rf"${name}$: $m={r['low_m']:.3f}\pm{r['low_m_se']:.3f}$" "\n"
               rf"      $10^4c={1e4 * r['low_c']:+.1f}\pm{1e4 * r['low_c_se']:.1f}$")
        ax.errorbar(xc + off, mu, se, fmt=mk, ms=3.2, color=col, mec=col, mfc="white", mew=0.9,
                    elinewidth=0.8, capsize=0, zorder=4, label=lab)
        out[name] = dict(m=r["low_m"], m_se=r["low_m_se"], c=r["low_c"], c_se=r["low_c_se"],
                         n_low=r["n_low"], bin_x=xc.tolist(), bin_mean=mu.tolist(), bin_se=se.tolist())
    ax.text(0, -0.071, r"fit window $|g|<0.02$", ha="center", va="bottom", fontsize=6.5, color="#555555")
    ax.set_xlim(-0.105, 0.105); ax.set_ylim(-0.075, 0.075)
    ax.set_xlabel(r"true shear $g_k$")
    ax.set_ylabel(r"$\langle\hat g_k - g_k\rangle$")
    ax.legend(loc="upper right", fontsize=6.8, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.5)
    savefig(fig, "fig_raw_bias")
    NUM["F4_raw_bias"] = out


# ------------------------------------------------------------------ F5
def fig_shrinkage():
    # Eason (2026-09-30): bin on the measured posterior width (T3), not the
    # GBM-predicted width (S4); the narrowest bins carry the noise-selection effect.
    t3 = json.load(open(PT / "A_shear/results.json"))["T3"]
    bins = t3["bins"]
    chi2 = t3["chi2"]
    fig, ax = plt.subplots(figsize=(COL, 0.8 * COL))
    xx = np.linspace(0, 1.08 * SIGMA_PI2, 50)
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(1e3 * SIGMA_PI2, color=GREY, lw=0.7, ls=":")
    ax.text(1e3 * SIGMA_PI2 - 0.04, 0.02, r"$\sigma_\pi^2$", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.plot(1e3 * xx, -xx / SIGMA_PI2, color=COLORS["prediction"], lw=1.1, ls="--",
            label=r"prediction $m=-\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2$")
    out = {}
    for comp, name, col, mk in (("g1", "g_1", COLORS["g1"], "o"), ("g2", "g_2", COLORS["g2"], "s")):
        # the stored Gaussian prediction is -<sigma_gk^2>/sigma_pi^2 over the bin's galaxies
        x = np.array([-b[comp]["gauss_posterior"] * SIGMA_PI2 for b in bins])
        m = np.array([b[comp]["m"] for b in bins]); se = np.array([b[comp]["m_se"] for b in bins])
        c2 = chi2[f"gauss_posterior:{comp}"]
        ax.errorbar(1e3 * x, m, se, fmt=mk, ms=3.4, color=col, mfc="white", mew=0.9, elinewidth=0.8,
                    capsize=0, label=rf"${name}$")
        out[comp] = dict(mean_sigma_post_sq=x.tolist(), m=m.tolist(), m_se=se.tolist(), chi2=c2["chi2"], nbins=c2["ndof"])
    out["n_per_bin"] = [b["n"] for b in bins]
    ax.set_xlim(0, 1e3 * xx[-1]); ax.set_ylim(-1.1, 0.25)
    ax.set_xlabel(r"$\langle\sigma_{{\rm post},k}^2\rangle$ in bin $[10^{-3}]$")
    ax.set_ylabel(r"multiplicative bias $m_k$")
    ax.legend(loc="lower left", fontsize=6.8, handletextpad=0.4)
    savefig(fig, "fig_shrinkage")
    NUM["F5_shrinkage"] = out


# ------------------------------------------------------------------ F6
def width_cut_scan(D, thresholds):
    rows = []
    for t in list(thresholds) + [np.inf]:
        mask = D["sig"] < t
        row = dict(threshold=float(t) if np.isfinite(t) else None, n_kept=int(mask.sum()),
                   frac_kept=float(mask.mean()))
        for k, comp in enumerate(("g1", "g2")):
            low = np.abs(D["g"][:, k]) < LOW_G
            row[f"n_kept_low_{comp}"] = int((mask & low).sum())
            row[f"n_low_{comp}"] = int(low.sum())
        for kind in ("raw", "weighted"):
            w = weights(D, kind, mask)
            row[kind] = {}
            for k, comp in enumerate(("g1", "g2")):
                r = fit(D["g"][:, k], D["e"][:, k], w)
                row[kind][comp] = dict(m=r["low_m"], m_se=r["low_m_se"])
            row[kind]["ess"] = kish(w[w > 0])
        rows.append(row)
    return rows


def fig_width_cut(D):
    thr = np.round(np.arange(0.02, 0.0851, 0.0025), 4)
    rows = width_cut_scan(D, thr)
    nocut = rows[-1]; rows = rows[:-1]
    sel = json.load(open(PT / "B_inference/t2b_xu3like.json"))["tf_equal"]["selection"]
    selv = {round(s["threshold"], 4): s for s in sel}
    x = np.array([r["threshold"] for r in rows])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL, 1.22 * COL), sharex=True,
                                 gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.08))
    ax.axhline(0, color=GREY, lw=0.6)
    for kind, col in (("raw", COLORS["raw"]), ("weighted", COLORS["weighted"])):
        for comp, ls, mk in (("g1", "-", "o"), ("g2", (0, (4, 1.5)), "s")):
            m = np.array([r[kind][comp]["m"] for r in rows]); se = np.array([r[kind][comp]["m_se"] for r in rows])
            lab = {"raw": "equal weights", "weighted": "inverse-variance weights"}[kind]
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.13, lw=0)
            ax.plot(x, m, color=col, ls=ls, lw=1.1, label=rf"$m_{comp[1]}$, {lab}")
    # selection response from the matched-shear stencil
    ymark = {0.05: 0.36, 0.075: 0.36}
    for t in (0.05, 0.075):
        s = selv[t]
        ratio = s["R_sel_over_R_gamma"]; ratio_se = s["R_sel_diag_se"] / s["R_gamma_diag"]
        ax.axvline(t, color=GREY, lw=0.7, ls=":", zorder=0)
        ax.text(t, ymark[t], rf"$R_{{\rm sel}}/R_\gamma$" "\n" rf"$={ratio:+.2f}\pm{ratio_se:.2f}$",
                ha="center", va="top", fontsize=6.4, color="#333333",
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_ylim(-0.72, 0.38)
    ax.set_ylabel(r"multiplicative bias $m_k$")
    ax.legend(loc="lower left", fontsize=6.6, ncol=1, handlelength=2.2, labelspacing=0.25)
    # no-cut values
    for kind, col in (("raw", COLORS["raw"]), ("weighted", COLORS["weighted"])):
        for comp, mk in (("g1", "o"), ("g2", "s")):
            ax.plot([0.0885], [nocut[kind][comp]["m"]], marker=mk, ms=3.2, color=col, mfc="white", mew=0.9, ls="none")
    ax.text(0.0885, -0.43, "no\ncut", ha="center", va="bottom", fontsize=6.2, color="#555555")
    # top axis: kept fraction
    frac = np.array([r["frac_kept"] for r in rows])
    top = ax.secondary_xaxis("top", functions=(lambda t: np.interp(t, x, frac), lambda f: np.interp(f, frac, x)))
    top.set_xticks([0.1, 0.25, 0.5, 0.75])
    top.set_xticklabels(["10%", "25%", "50%", "75%"])
    top.set_xlabel("kept fraction of galaxies", fontsize=7.5, labelpad=3)
    top.tick_params(labelsize=7, direction="in")
    ax.tick_params(top=False)
    # bottom: kept counts (one axis, log)
    bx.plot(x, [r["n_kept"] for r in rows], color="#333333", lw=1.1, label=r"all galaxies (of $10^5$)")
    nl = np.array([0.5 * (r["n_kept_low_g1"] + r["n_kept_low_g2"]) for r in rows])
    bx.plot(x, nl, color="#333333", lw=1.1, ls=(0, (4, 1.5)),
            label=rf"in fit window $|g_k|<0.02$ (of ${nocut['n_low_g1']:,}$)".replace(",", r"{,}"))
    bx.set_yscale("log"); bx.set_ylim(400, 1.6e5)
    bx.set_ylabel("kept galaxies")
    bx.set_xlabel(r"cut threshold $t$ on $\sigma_{\epsilon,i}$")
    bx.legend(loc="lower right", fontsize=6.6, labelspacing=0.25)
    bx.set_xlim(0.018, 0.0915)
    for t in (0.05, 0.075):
        bx.axvline(t, color=GREY, lw=0.7, ls=":", zorder=0)
    savefig(fig, "fig_width_cut")
    NUM["F6_width_cut"] = dict(rows=rows, no_cut=nocut,
                               selection={str(t): dict(R_sel_over_R_gamma=selv[t]["R_sel_over_R_gamma"],
                                                       R_sel_over_R_gamma_se=selv[t]["R_sel_diag_se"] / selv[t]["R_gamma_diag"],
                                                       R_sel=selv[t]["R_sel_diag"], R_sel_se=selv[t]["R_sel_diag_se"],
                                                       R_gamma=selv[t]["R_gamma_diag"],
                                                       stencil_kept_frac=selv[t]["value"]["frac_by_state"][0])
                                          for t in (0.05, 0.075)})


# ------------------------------------------------------------------ F7
def fig_response():
    J = json.load(open(PT / "B_inference/t2b_xu3like.json"))["tf_equal"]
    cub = np.load(PT / "B_inference/t2b_xu3like.tf_equal.cubes.npz")
    R, sd0 = cub["R"], cub["sd"][:, 0]
    x = np.concatenate([sd0[:, 0] ** 2, sd0[:, 1] ** 2]); y = np.concatenate([R[:, 0, 0], R[:, 1, 1]])
    qs = np.quantile(x, np.linspace(0, 1, 11))
    xb, yb, eb = [], [], []
    for i, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
        m = (x >= lo) & ((x < hi) if i < 9 else (x <= hi))
        xb.append(x[m].mean()); yb.append(y[m].mean()); eb.append(y[m].std(ddof=1) / np.sqrt(m.sum()))
    xb, yb, eb = map(np.array, (xb, yb, eb))
    f = J["fit_R_sigma"]["per_component"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(DCOL, 0.37 * DCOL),
                                 gridspec_kw=dict(wspace=0.24, width_ratios=[1.3, 1.0]))
    xx = np.linspace(0, 1.1 * SIGMA_PI2, 100)
    ax.scatter(1e3 * x, y, s=1.5, color="#b5b5b5", lw=0, alpha=0.55, zorder=1, rasterized=True,
               label="individual galaxies")
    ax.axhline(0, color=GREY, lw=0.6); ax.axhline(1, color=GREY, lw=0.6, ls=":")
    ax.axvline(1e3 * SIGMA_PI2, color=GREY, lw=0.7, ls=":")
    ax.text(1e3 * SIGMA_PI2 - 0.05, -0.2, r"$\sigma_\pi^2$", ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.plot(1e3 * xx, 1 - xx / SIGMA_PI2, color=COLORS["prediction"], lw=1.1, ls="--",
            label=r"Gaussian prediction $1-\sigma_{\rm post}^2/\sigma_\pi^2$", zorder=3)
    ax.plot(1e3 * xx, f["a"] + f["b"] * xx, color=COLORS["calibrated"], lw=1.6, zorder=3,
            label=rf"fit $a+b\sigma_{{\rm post}}^2$: $a={f['a']:.2f}\pm{f['a_se']:.2f}$, $b={f['b']:.0f}\pm{f['b_se']:.0f}$")
    ax.errorbar(1e3 * xb, yb, eb, fmt="o", ms=3.8, color=COLORS["raw"], mfc="white", mew=1.0,
                elinewidth=0.9, capsize=0, zorder=4, label=r"decile means ($R_{11}$, $R_{22}$ pooled)")
    ax.set_xlim(0, 1e3 * xx[-1]); ax.set_ylim(-0.25, 1.75)
    ax.set_xlabel(r"per-component posterior variance $\sigma_{{\rm post},k}^2$ $[10^{-3}]$")
    ax.set_ylabel(r"shear response $R_{kk}$")
    h, l = ax.get_legend_handles_labels()
    order = [3, 2, 1, 0]
    leg = ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper right", fontsize=6.6,
                    labelspacing=0.3, framealpha=0.92, frameon=True, edgecolor="none", fancybox=False,
                    borderpad=0.3)
    leg.legend_handles[3].set_sizes([10])
    ax.set_title("(a)", loc="left", fontsize=8.5)

    b = [v for v in J["by_cosi"] if "R" in v]
    xc = np.array([0.5 * (v["lo"] + v["hi"]) for v in b])
    bx.axhline(0, color=GREY, lw=0.6); bx.axhline(1, color=GREY, lw=0.6, ls=":")
    pred = np.array([1 - v["mean_sigma_comp_sq"] / SIGMA_PI2 for v in b])
    bx.plot(xc, pred, color=COLORS["prediction"], ls="--", lw=1.1,
            label=r"Gaussian prediction $1-\langle\sigma_{\rm post}^2\rangle/\sigma_\pi^2$")
    for k, name, col, mk, off in ((0, "R_{11}", COLORS["g1"], "o", -0.012), (1, "R_{22}", COLORS["g2"], "s", 0.012)):
        bx.errorbar(xc + off, [v["R"][k][k] for v in b], [v["R_se"][k][k] for v in b], fmt=mk, ms=3.8,
                    color=col, mfc="white", mew=1.0, elinewidth=0.9, capsize=0, label=rf"${name}$")
    bx.set_xlim(0, 1); bx.set_ylim(-0.05, 1.1)
    bx.set_xlabel(r"true $\cos i$ (bins of width 0.2)")
    bx.set_ylabel(r"mean shear response")
    bx.legend(loc="center left", bbox_to_anchor=(0.0, 0.66), fontsize=6.6, labelspacing=0.3)
    bx.set_title("(b)", loc="left", fontsize=8.5)
    savefig(fig, "fig_response")
    NUM["F7_response"] = dict(n_bases=int(len(R)), fit=f, decile_x=xb.tolist(), decile_R=yb.tolist(), decile_se=eb.tolist(),
                              overall_R=J["overall"]["R"], overall_R_se=J["overall"]["R_se"],
                              by_cosi=[dict(lo=v["lo"], hi=v["hi"], n=v["n"], R11=v["R"][0][0], R22=v["R"][1][1],
                                            gauss_pred=float(p)) for v, p in zip(b, pred)])


# ------------------------------------------------------------------ F8
def fig_stacking():
    rep = json.load(open(PT / "C_stacking/report.json"))
    M = rep["metrics"]
    fig, ax = plt.subplots(figsize=(COL, 0.8 * COL))
    ax.axhline(0, color=GREY, lw=0.6)
    out = {}
    specs = (("pergal_mean_train", "per-galaxy posterior mean", COLORS["raw"], -1),
             ("stack_kde_mean_train", "stacked likelihood (KDE), posterior mean", COLORS["stacked"], 1))
    for key, lab, col, side in specs:
        blk = M[key]["vs_N_at_R1"]
        Ns = sorted(int(n) for n in blk)
        out[key] = {}
        for comp, mk, sub in (("g1", "o", 0), ("g2", "s", 1)):
            med = np.array([blk[str(n)]["boot_median"][f"{comp}_m"] for n in Ns])
            lo = np.array([blk[str(n)]["boot_p16"][f"{comp}_m"] for n in Ns])
            hi = np.array([blk[str(n)]["boot_p84"][f"{comp}_m"] for n in Ns])
            det = np.array([blk[str(n)]["det"][f"{comp}_m"] for n in Ns])
            xs = np.array(Ns) * 2 ** (0.07 * (2 * side + (sub - 0.5)))
            ax.errorbar(xs, med, [med - lo, hi - med], fmt=mk, ms=3.6, color=col,
                        mfc="white" if sub else col, mew=1.0, elinewidth=0.9, capsize=0,
                        label=rf"{lab}, $m_{comp[1]}$")
            out[key][comp] = dict(N=Ns, boot_median=med.tolist(), boot_p16=lo.tolist(), boot_p84=hi.tolist(),
                                  all_galaxies_point=det.tolist())
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_xlim(0.75, 42); ax.set_ylim(-0.95, 0.12)
    ax.set_xlabel(r"galaxies per shear cell $N$")
    ax.set_ylabel(r"multiplicative bias $m_k$")
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], color=COLORS["raw"], marker="o", ms=3.6, lw=0.9),
          Line2D([], [], color=COLORS["stacked"], marker="o", ms=3.6, lw=0.9),
          Line2D([], [], color="#333333", marker="o", ms=3.6, ls="none"),
          Line2D([], [], color="#333333", marker="s", ms=3.6, mfc="white", mew=1.0, ls="none")]
    ax.legend(hd, ["per-galaxy posterior mean", "stacked likelihood", r"$m_1$", r"$m_2$"],
              loc="lower right", fontsize=6.8, labelspacing=0.3, handletextpad=0.4, ncol=1)
    savefig(fig, "fig_stacking")
    meta = dict(n_cells=25, n_gal_pool=32, n_boot=rep.get("bootstrap"), nuisance_prior="training (unweighted samples)",
                R=1)
    NUM["F8_stacking"] = dict(meta=meta, **out)


# ------------------------------------------------------------------ Tab. 2
def calibration_R(ab, sd):
    return np.clip(ab[0] + ab[1] * sd ** 2, 0.0, None)


def stencil_fit(R, sd0, idx):
    x = np.concatenate([sd0[idx, 0] ** 2, sd0[idx, 1] ** 2]); y = np.concatenate([R[idx, 0, 0], R[idx, 1, 1]])
    return np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]


def calibrated_component(D, ab, k, idx=None):
    """Ensemble response calibration of component k: e_i = ghat_i / R(sigma_k,i)
    with weights R(sigma_k,i)^2 (equivalent to regressing ghat on R(sigma) g over
    the ensemble). Returns (t, e, w) restricted to idx if given."""
    Rk = calibration_R(ab, D["sd"][:, k])
    ok = Rk > 0
    e = np.where(ok, D["e"][:, k] / np.where(ok, Rk, 1.0), 0.0)
    w = Rk ** 2
    if idx is not None:
        return D["g"][idx, k], e[idx], w[idx]
    return D["g"][:, k], e, w


def table(D, nboot=500, seed=7):
    rows = []

    def row_uncal(label, kind, mask, note):
        w = weights(D, kind, mask)
        r = {}
        se2 = 0.0
        for k, comp in enumerate(("g1", "g2")):
            f = fit(D["g"][:, k], D["e"][:, k], w)
            s = sigma_eff_component(D["g"][:, k], D["e"][:, k], w, f)
            r[comp] = dict(m=f["low_m"], m_se=f["low_m_se"], c=f["low_c"], c_se=f["low_c_se"], sigma_eff=s)
            se2 += s ** 2
        wk = w[w > 0]
        post = float(np.sum(w * D["sig"]) / np.sum(w))
        rows.append(dict(label=label, **r, sigma_eff=float(np.sqrt(se2)), sigma_post=post, n_eff=kish(wk),
                         kept=float(mask.mean()), n_kept=int(mask.sum()), note=note))

    allg = np.ones(D["n"], bool)
    cut = D["sig"] < 0.05
    sel = json.load(open(PT / "B_inference/t2b_xu3like.json"))["tf_equal"]["selection"]
    s05 = [s for s in sel if abs(s["threshold"] - 0.05) < 1e-9][0]
    rsel = s05["R_sel_over_R_gamma"]; rsel_se = s05["R_sel_diag_se"] / s05["R_gamma_diag"]
    row_uncal("raw", "raw", allg, "")
    row_uncal("weighted", "weighted", allg, "")
    row_uncal("cut_raw", "raw", cut, "sel")
    row_uncal("cut_weighted", "weighted", cut, "sel")

    # ensemble response calibration with R(sigma) from the Xu3-like stencil
    cub = np.load(PT / "B_inference/t2b_xu3like.tf_equal.cubes.npz")
    Rst, sd0 = cub["R"], cub["sd"][:, 0]
    nb = len(Rst)
    ab = stencil_fit(Rst, sd0, np.arange(nb))
    r = {}; se2 = 0.0; neff = []; kept = []
    rng = np.random.default_rng(seed)
    lows = [np.flatnonzero(np.abs(D["g"][:, k]) < LOW_G) for k in range(2)]
    boot = {c: dict(m=[], c=[]) for c in ("g1", "g2")}
    for _ in range(nboot):
        abb = stencil_fit(Rst, sd0, rng.integers(0, nb, nb))
        for k, comp in enumerate(("g1", "g2")):
            idx = rng.choice(lows[k], len(lows[k]))
            t, e, w = calibrated_component(D, abb, k, idx)
            X = np.column_stack([np.ones(len(t)), t])
            coef = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * (e - t)))
            boot[comp]["c"].append(coef[0]); boot[comp]["m"].append(coef[1])
    for k, comp in enumerate(("g1", "g2")):
        t, e, w = calibrated_component(D, ab, k)
        f = fit(t, e, w)
        s = sigma_eff_component(t, e, w, f)
        se2 += s ** 2
        neff.append(kish(w[w > 0])); kept.append(float(np.mean(w > 0)))
        r[comp] = dict(m=f["low_m"], m_se=float(np.std(boot[comp]["m"], ddof=1)), m_fit_se=f["low_m_se"],
                       c=f["low_c"], c_se=float(np.std(boot[comp]["c"], ddof=1)), c_fit_se=f["low_c_se"],
                       sigma_eff=s, mean_R=float(calibration_R(ab, D["sd"][:, k]).mean()))
    rows.append(dict(label="calibrated", **r, sigma_eff=float(np.sqrt(se2)), sigma_post=None,
                     n_eff=float(np.mean(neff)), n_eff_components=neff, kept=float(np.mean(kept)),
                     kept_components=kept, stencil_fit=dict(a=float(ab[0]), b=float(ab[1])), note="cal"))
    NUM["Tab2"] = dict(rows=rows, R_sel_over_R_gamma_t005=rsel, R_sel_over_R_gamma_t005_se=rsel_se,
                       nboot_calibration=nboot)

    # ---------------------------------------------------------------- LaTeX
    names = {
        "raw": r"Raw posterior mean",
        "weighted": r"Inverse-variance weighted",
        "cut_raw": r"Width cut, equal weights",
        "cut_weighted": r"Width cut, inv.-var.\ weights",
        "calibrated": r"Response calibration",
    }
    notes = {
        "raw": r"no cut",
        "weighted": r"floor $\langle\sigma_{\epsilon,i}\rangle/\sqrt{2}$",
        "cut_raw": r"$\sigma_{\epsilon,i}<0.05$$^{a}$",
        "cut_weighted": r"$\sigma_{\epsilon,i}<0.05$$^{a}$",
        "calibrated": r"stencil $R(\sigma)$$^{b}$",
    }
    L = []
    L.append(r"\begin{table*}")
    L.append(r"\centering")
    L.append(r"\caption{Shear estimators built from the NPE posteriors on the Xu3 test set "
             r"($10^5$ galaxies). $m_k$ and $c_k$ are from a linear fit of $\hat g_k-g_k$ over "
             r"$|g_k|<0.02$. $\langle\sigma_{\epsilon,i}\rangle$ is the weighted mean posterior width; "
             r"$\sigma_{\rm eff}$ is the effective shape noise per unit response (see text); "
             r"$N_{\rm eff}$ is the Kish effective number of galaxies, with the kept fraction in brackets.}")
    L.append(r"\label{tab:estimators}")
    L.append(r"\footnotesize")
    L.append(r"\setlength{\tabcolsep}{3pt}")
    L.append(r"\begin{tabular}{lcccccccl}")
    L.append(r"\toprule")
    L.append(r"Estimator & $m_1$ & $m_2$ & $10^4 c_1$ & $10^4 c_2$ & $\langle\sigma_{\epsilon,i}\rangle$ "
             r"& $\sigma_{\rm eff}$ & $N_{\rm eff}$ (kept) & Notes \\")
    L.append(r"\midrule")
    for row in rows:
        sp = "--" if row["sigma_post"] is None else f"{row['sigma_post']:.4f}"
        ne = f"{row['n_eff']:,.0f}".replace(",", "{,}")
        L.append(
            f"{names[row['label']]} & ${row['g1']['m']:+.3f}\\pm{row['g1']['m_se']:.3f}$ "
            f"& ${row['g2']['m']:+.3f}\\pm{row['g2']['m_se']:.3f}$ "
            f"& ${sci4(row['g1']['c'], row['g1']['c_se'])}$ & ${sci4(row['g2']['c'], row['g2']['c_se'])}$ "
            f"& {sp} & {row['sigma_eff']:.3f} & {ne} ({100 * row['kept']:.0f}\\%) & {notes[row['label']]} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\par\smallskip")
    L.append(r"\begin{minipage}{0.96\linewidth}\footnotesize")
    L.append(r"$^{a}$The cut selects on the observed posterior width, which responds to the shear. "
             rf"On the Xu3-like matched-shear stencil the selection response of this cut is "
             rf"$R_{{\rm sel}}/R_\gamma = {rsel:+.2f}\pm{rsel_se:.2f}$ for the equal-weight estimator, so part of "
             r"the reduction of $|m|$ is a positive selection bias rather than reduced shrinkage. "
             r"The selection response of the weighted cut sample was not measured.\\")
    cr = rows[-1]
    L.append(r"$^{b}$Ensemble calibration: per-galaxy estimates $\hat g_{k,i}/R(\sigma_{k,i})$ weighted by "
             rf"$R(\sigma_{{k,i}})^2$, with $R(\sigma)=a+b\sigma^2$ ($a={cr['stencil_fit']['a']:.3f}$, "
             rf"$b={cr['stencil_fit']['b']:.0f}$) fitted on the Xu3-like matched-shear stencil and clipped at zero. "
             rf"Errors come from a nested bootstrap over stencil and test galaxies ({nboot} resamples); "
             r"$N_{\rm eff}$ and the kept fraction (galaxies with $R>0$) are averaged over the two components. "
             r"The calibration does not change the posterior widths.")
    L.append(r"\end{minipage}")
    L.append(r"\end{table*}")
    (OUT / "tab_estimators.tex").write_text("\n".join(L) + "\n")
    return rows


# ------------------------------------------------------------------ captions
def captions(rows):
    F4 = NUM["F4_raw_bias"]; F6 = NUM["F6_width_cut"]; F7 = NUM["F7_response"]; F8 = NUM["F8_stacking"]
    s05 = F6["selection"]["0.05"]; s075 = F6["selection"]["0.075"]
    txt = f"""# Sec. 6 figure and table captions

All Xu3 numbers use the Xu3 (DESI BGS Any) test set of 100,000 simulated galaxies and the
Tully-Fisher-weighted posterior mean of (g1, g2). sigma_eps,i = sqrt(sigma_g1^2 + sigma_g2^2)
from the posterior standard deviations. m and c come from a linear fit of ghat - g on |g| < 0.02
(image frame); error bars are fit (sandwich) errors unless stated otherwise.

## fig_raw_bias (Fig. F4)
Bias of the raw per-galaxy posterior means. Points show the mean of ghat_k - g_k in bins of the true
shear (width 0.01) with the standard error (smaller than the markers), for g1 (circles) and g2 (squares); all 100,000 Xu3
galaxies with equal weights and no cuts. Solid lines are the linear fits over the shaded window
|g_k| < 0.02 ({F4['g_1']['n_low']:,} galaxies per component); dashed lines extend them to the full prior
range. The fit gives m1 = {F4['g_1']['m']:.3f} +/- {F4['g_1']['m_se']:.3f} and
m2 = {F4['g_2']['m']:.3f} +/- {F4['g_2']['m_se']:.3f}, with c consistent with zero.

## fig_shrinkage (Fig. F5)
Measured multiplicative bias against the mean per-component posterior variance, in 12 equal-count bins
({NUM['F5_shrinkage']['n_per_bin'][0]:,} galaxies each) of a predicted posterior width. The prediction uses
only true galaxy properties that do not depend on the shear or the noise (cross-fitted regression), so
the binning does not select on the noise. Each point is the |g| < 0.02 fit in one bin (equal weights,
TF-weighted posterior means). The dashed line is the Gaussian-shrinkage prediction
m = -<sigma_post^2>/sigma_pi^2 with sigma_pi^2 = 0.2^2/12, the variance of the uniform shear prior; it
is not fitted. chi^2 against the prediction (fit errors only): {NUM['F5_shrinkage']['g1']['chi2']:.1f} (g1) and
{NUM['F5_shrinkage']['g2']['chi2']:.1f} (g2) for 12 bins.

## fig_width_cut (Fig. F6)
Top: m1 (solid) and m2 (dashed) against the threshold t of a posterior-width cut sigma_eps,i < t on Xu3,
for equal weights (blue) and inverse-variance weights w_i = 1/(sigma_eps,i^2/2 + sigma_floor^2) with the
floor sigma_floor = <sigma_eps,i>/sqrt(2) of the kept galaxies (red). Bands are +/-1 sigma fit errors;
open markers at the right are the uncut values. The top axis gives the kept fraction of all galaxies.
Vertical dotted lines mark t = 0.05 and t = 0.075, where the selection response of the cut was measured
on a Xu3-like matched-shear stencil (1000 galaxies, each rendered at five shears with identical noise):
R_sel/R_gamma = {s05['R_sel_over_R_gamma']:+.3f} +/- {s05['R_sel_over_R_gamma_se']:.3f} and
{s075['R_sel_over_R_gamma']:+.3f} +/- {s075['R_sel_over_R_gamma_se']:.3f}, i.e. the cut adds a positive
selection bias that grows as the cut tightens (measured for the equal-weight estimator only). Bottom: number of kept galaxies, in total (of 100,000)
and inside the fit window |g_k| < 0.02 (mean of the two components, of 20,000). The fit-window count is
lower than the overall kept fraction implies because posteriors are wider at small |g|.

## fig_response (Fig. F7)
Shear response on the Xu3-like matched-shear stencil: 1000 galaxies drawn from a Xu3-like catalog, each
rendered at shears (0,0), (+/-0.01, 0), (0, +/-0.01) with the same noise; R_kk is the central finite
difference of the TF-weighted posterior mean, equal weights per galaxy. (a) R_11 and R_22 pooled
against the per-component posterior variance of the zero-shear version: grey points are individual
galaxies, circles are decile means with standard errors, the solid line is the least-squares fit
R = a + b sigma_post^2 (a = {F7['fit']['a']:.3f} +/- {F7['fit']['a_se']:.3f},
b = {F7['fit']['b']:.0f} +/- {F7['fit']['b_se']:.0f}, bootstrap errors), and the dashed line is the Gaussian
prediction a = 1, b = -1/sigma_pi^2 = -300. (b) Mean R_11, R_22 in bins of the true cos i with
bootstrap errors (about 200 galaxies per bin); the dashed line is the Gaussian prediction from the mean
posterior variance of each bin.

## fig_stacking (Fig. F8)
Multiplicative bias against the number of galaxies N per constant-shear cell, with one noise realization
per galaxy. Benchmark: 25 cells with fixed shears g1, g2 in {{-0.08, -0.04, 0, 0.04, 0.08}}, sharing a pool
of 32 galaxies drawn from the training prior; m is fitted over the 25 cell estimates per component.
Blue: average of per-galaxy posterior means; purple: posterior mean of the product of per-galaxy shear
likelihoods, each the NPE shear marginal (a boundary-reflected KDE of the samples) divided by the uniform
training prior. Samples are unweighted, i.e. the nuisance prior is the training prior from which these
galaxies were drawn (not the TF-weighted Xu3 convention). Markers are bootstrap medians, bars the 16-84%
range of 1000 paired bootstrap resamples (N galaxies drawn with replacement from the pool of 32, the same
draw in all 25 cells). Filled markers m1, open markers m2. At N = 32 with all galaxies,
m1, m2 = {F8['stack_kde_mean_train']['g1']['all_galaxies_point'][-1]:+.3f}, {F8['stack_kde_mean_train']['g2']['all_galaxies_point'][-1]:+.3f} (stacked) and
{F8['pergal_mean_train']['g1']['all_galaxies_point'][-1]:+.3f}, {F8['pergal_mean_train']['g2']['all_galaxies_point'][-1]:+.3f} (per-galaxy).

## tab_estimators (Tab. 2)
Estimators on the Xu3 test set (100,000 galaxies, TF-weighted posterior means). Rows: raw (equal
weights, no cut); inverse-variance weights 1/(sigma_eps,i^2/2 + sigma_floor^2) with sigma_floor = <sigma_eps,i>/sqrt(2) (as in Fig. F6); width cut sigma_eps,i < 0.05 with equal and
with inverse-variance weights (on the matched-shear stencil this cut has
R_sel/R_gamma = {s05['R_sel_over_R_gamma']:+.2f} +/- {s05['R_sel_over_R_gamma_se']:.2f}); ensemble response calibration with R(sigma) = a + b sigma^2
fitted on the Xu3-like stencil (Fig. F7), applied as ghat_k,i / R(sigma_k,i) with weights R(sigma_k,i)^2
(equivalent to regressing ghat on R(sigma) g over the ensemble). Errors on m, c are fit errors, except
for the calibrated row, where a nested bootstrap over stencil and test galaxies is used.
<sigma_eps,i> is the estimator-weighted mean posterior width (Xu et al. 2024 definition).
sigma_eff (effective shape noise) is defined identically for every row: per component,
sigma_eff,k = [sum_i w_i^2 r_k,i^2 / sum_i w_i^2]^(1/2) / (1 + m_k), where r_k,i = ehat_k,i - c_k - (1 + m_k) g_k,i
is the residual of the (calibrated, where applicable) estimate about the fitted relation for galaxies with
|g_k| < 0.02, and w_i are the estimator's galaxy weights; sigma_eff = sqrt(sigma_eff,1^2 + sigma_eff,2^2).
With this definition sigma_eff,k / sqrt(N_eff) is approximately the statistical error of the response-corrected ensemble
mean shear, so rows can be compared directly. N_eff = (sum w)^2 / sum w^2 is the Kish effective number of
galaxies over the whole sample; the kept fraction is the fraction with non-zero weight.
Stacking is not included because it is evaluated on a different benchmark.
"""
    (OUT / "captions_sec6.md").write_text(txt)


# ------------------------------------------------------------------ F6b (2026-10-01)
def _cut_stats(D, w):
    out = {}
    for k, comp in enumerate(("g1", "g2")):
        r = fit(D["g"][:, k], D["e"][:, k], w)
        out[comp] = dict(m=r["low_m"], m_se=r["low_m_se"],
                         sigma_eff=sigma_eff_component(D["g"][:, k], D["e"][:, k], w, r))
    out["n_eff"] = kish(w[w > 0])
    out["m_mean"] = 0.5 * (out["g1"]["m"] + out["g2"]["m"])
    out["m_mean_se"] = 0.5 * np.hypot(out["g1"]["m_se"], out["g2"]["m_se"])
    out["noise"] = 0.5 * (out["g1"]["sigma_eff"] + out["g2"]["sigma_eff"]) / np.sqrt(out["n_eff"])
    # Bottom panel (Eason, 2026-10-01, final): bias-corrected 'noise' (trends only).
    # Kept for reference: uncorrected shape noise (weighted mean of the per-galaxy
    # sigma_eps,i, as in KL-I) over sqrt(N_eff), not the bias-corrected scatter.
    ww = w / w.sum()
    out["sig_mean"] = float(np.sum(ww * D["sig"]))
    out["sig_over_sqrtn"] = out["sig_mean"] / np.sqrt(out["n_eff"])
    # error of the weighted mean propagated from the individual shape noises:
    # sqrt(sum w^2 sigma_eps,i^2) / sum w   (equal weights -> sigma_rms / sqrt(N))
    out["sig_prop"] = float(np.sqrt(np.sum(ww ** 2 * D["sig"] ** 2)))
    return out


def fig_cuts(D):
    """m and response-corrected shear noise vs kept (effective) fraction for posterior-width
    cuts, inclination cuts (NPE posterior-mean cos i, and true cos i) and the two weightings."""
    a = np.load(PT / "A_npe/a_npe_stream.npz", allow_pickle=True)
    fn = list(a["feature_names"]); j = fn.index("cosi")
    assert np.allclose(a["truth"][:, :2], D["g"], atol=1e-6)
    cos_post = a["tf_summary"][:, 1, j].astype(float)
    cos_true = a["truth"][:, j].astype(float)
    n = D["n"]
    fr = np.geomspace(0.1, 1.0, 19)
    curves = {}
    for lab, score in (("width", -D["sig"]), ("cosi_post", cos_post), ("cosi_true", cos_true)):
        rows = []
        order = np.argsort(-score)
        for f in fr:
            mask = np.zeros(n, bool); mask[order[:int(round(f * n))]] = True
            st = _cut_stats(D, mask / mask.sum()); st["frac"] = float(mask.mean()); rows.append(st)
        curves[lab] = rows
    sg2 = D["var"].sum(1) / 2
    wts = {"floor": weights(D, "weighted", np.ones(n, bool)), "nofloor": (1 / sg2) / np.sum(1 / sg2)}
    pts = {k: _cut_stats(D, w) for k, w in wts.items()}
    for k in pts:
        pts[k]["frac"] = pts[k]["n_eff"] / n

    C = {"width": COLORS["raw"], "cosi": COLORS["calibrated"], "w": COLORS["weighted"]}
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 1.25 * COL), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.07))
    a1.axhline(0, color=GREY, lw=0.6)
    for lab, col, ls, lw, name in (("width", C["width"], "-", 1.4, r"cut on $\sigma_{\epsilon,i}$"),
                                   ("cosi_post", C["cosi"], "--", 1.4, r"cut on posterior $\cos i$"),
                                   ("cosi_true", C["cosi"], ":", 0.9, r"cut on true $\cos i$")):
        x = np.array([r["frac"] for r in curves[lab]])
        y = np.array([r["m_mean"] for r in curves[lab]]); e = np.array([r["m_mean_se"] for r in curves[lab]])
        nz = np.array([r["noise"] for r in curves[lab]])
        a1.plot(x, y, color=col, ls=ls, lw=lw, label=name)
        if lab != "cosi_true":
            a1.fill_between(x, y - e, y + e, color=col, alpha=0.15, lw=0)
        a2.plot(x, 1e4 * nz, color=col, ls=ls, lw=lw)
    for k, mk, name in (("floor", "D", "weights with floor"), ("nofloor", "*", "weights without floor")):
        r = pts[k]
        ms = 6 if mk == "*" else 4.5
        mfc = "white" if mk == "D" else C["w"]
        a1.errorbar(r["frac"], r["m_mean"], r["m_mean_se"], fmt=mk, ms=ms, color=C["w"],
                    mfc=mfc, mew=1.0, capsize=0, zorder=5, label=name)
        a2.plot(r["frac"], 1e4 * r["noise"], mk, ms=ms, color=C["w"], mfc=mfc, mew=1.0, zorder=5)
    a1.set_ylabel(r"multiplicative bias $(m_1+m_2)/2$")
    a1.set_ylim(-0.7, 0.2)
    a1.legend(loc="lower left", fontsize=6.6, handlelength=2.2)
    a2.set_xscale("log"); a2.set_xlim(0.095, 1.05)
    a2.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0]); a2.set_xticklabels(["10%", "20%", "30%", "50%", "100%"])
    a2.minorticks_off(); a1.minorticks_off()
    a2.set_xlabel(r"kept (or effective) fraction of galaxies, $N_{\rm eff}/N$")
    a2.set_ylabel(r"$\sigma_{\rm eff}/\sqrt{N_{\rm eff}}$ $[10^{-4}]$")
    savefig(fig, "fig_cuts")
    NUM["F6b_cuts"] = dict(curves=curves, weights=pts, frac_grid=fr.tolist())


def main():
    apply_style()
    D = load_xu3()
    fig_raw_bias(D)
    fig_shrinkage()
    fig_width_cut(D)
    fig_cuts(D)
    fig_response()
    fig_stacking()
    rows = table(D)
    captions(rows)
    (OUT / "sec6_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    for r in rows:
        print(f"{r['label']:14s} m1={r['g1']['m']:+.3f}±{r['g1']['m_se']:.3f} m2={r['g2']['m']:+.3f}±{r['g2']['m_se']:.3f} "
              f"c1={1e4*r['g1']['c']:+.2f}±{1e4*r['g1']['c_se']:.2f} c2={1e4*r['g2']['c']:+.2f}±{1e4*r['g2']['c_se']:.2f} "
              f"sig_post={r['sigma_post']} sig_eff={r['sigma_eff']:.4f} (k: {r['g1']['sigma_eff']:.4f},{r['g2']['sigma_eff']:.4f}) "
              f"Neff={r['n_eff']:.0f} kept={r['kept']:.3f}")


if __name__ == "__main__":
    main()
