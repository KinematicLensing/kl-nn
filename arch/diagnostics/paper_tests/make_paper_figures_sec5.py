#!/usr/bin/env python3
"""Paper figures and tables for Sec. 5 (validating the NPE).

Produces, in paper_style.OUT (paper-tests/paper_figs_final/):
  fig_training_loss   F1   train/validation NLL vs epoch
  fig_posterior_snr   F2   g1-g2 posterior of Xu3 galaxy 6447 vs image / Halpha SNR
  fig_pp_shear        F3   P-P curves (PIT ECDF) for g1, g2 on the Xu3 test set
  fig_pp_nuisance     App. 16/84% tail fractions and 68% coverage, all 9 parameters
  fig_corner9d        FA1  9D triangle plot of galaxy 6447
  tab_coverage.tex    Tab. 1  g1, g2 coverage at 50/68/95/99%
  tab_coverage_nuisance.tex   App. 68% coverage of all 9 parameters
  captions_sec5.md    caption drafts (numbers filled from the data files)

Reads only existing results (no inference):
  A_npe/report.json, A_npe/a_npe_stream.npz, the model loss CSV,
  $S/cache/paper_tests_B_inference/r2b/gal6447_*_seed*.npz.

Run with the kl-nn env:
  /jet/home/xwang30/.conda/envs/kl-nn/bin/python make_paper_figures_sec5.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_style import COL, DCOL, COLORS, GETDIST_SETTINGS, LABELS, OUT, apply_style, savefig  # noqa: E402

from getdist import MCSamples, plots  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
PT = Path(os.environ.get("KLNN_PAPER_TESTS_ROOT", S / "reports/paper-tests"))
A_JSON = PT / "A_npe/report.json"
A_NPZ = PT / "A_npe/a_npe_stream.npz"
MODEL_CFG = S / "configs" / f"cfg_{os.environ.get('KLNN_PAPER_MODEL', 'CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair')}.json"
R2B = Path(os.environ.get("KLNN_R2B_DIR", S / "cache/paper_tests_B_inference/r2b"))
GAL = 6447
# NPE samples on the exact noisy data of the MCMC comparison package sent to Lauren (k_mcmc_export.py), i.e. the
# test-set noise realization of this galaxy; her MCMC contours can be overlaid on Fig. posterior_example.
MCMC_NPE = S / "reports/paper-tests-1m/mcmc/npe" / f"gal_{GAL}_npe_samples.npz"
SEEDS = range(5)

NAMES9 = ["g1", "g2", "theta_int", "cosi", "v0", "vcirc", "rscale", "hlr", "halpha_flux_true"]
# Training-prior ranges (model config par_ranges); flux range in log10.
PRIOR = {"g1": (-0.1, 0.1), "g2": (-0.1, 0.1), "theta_int": (-math.pi, math.pi),
         "cosi": (0.0, 1.0), "v0": (-30.0, 30.0), "vcirc": (60.0, 540.0),
         "rscale": (0.1, 5.0), "hlr": (0.1, 5.0), "halpha_flux_true": (-17.0, -14.0)}
# Local label tweaks (units added where LABELS has none).
LAB = dict(LABELS)
LAB["theta_int"] = r"\theta_{\rm int}\,[{\rm rad}]"
LAB["halpha_flux_true"] = r"\log_{10}(F_{{\rm H}\alpha}/{\rm erg\,s^{-1}\,cm^{-2}})"
PLAIN = {"g1": r"$g_1$", "g2": r"$g_2$", "theta_int": r"$\theta_{\rm int}$", "cosi": r"$\cos i$",
         "v0": r"$v_0$", "vcirc": r"$v_{\rm circ}$", "rscale": r"$R_{\rm vscale}$", "hlr": r"$R_h$",
         "halpha_flux_true": r"$\log_{10}F_{{\rm H}\alpha}$"}
CONTOURS = [0.68, 0.95]

CAP: dict[str, str] = {}


# ------------------------------------------------------------------ helpers
def tf_weights(logw):
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def load_r2b(tag, seed):
    d = np.load(R2B / f"gal{GAL}_{tag}_seed{seed}.npz")
    x = d["samples"].astype(np.float64).copy()
    x[:, 8] = np.log10(x[:, 8])
    t = d["truth"].astype(np.float64).copy()
    t[8] = np.log10(t[8])
    w = tf_weights(d["log_weight"])
    return dict(x=x, w=w, truth=t, image_snr=float(d["image_snr"][0]),
                halpha_snr=float(d["central_halpha_snr"][0]),
                ess=float(1.0 / np.sum(w ** 2) / len(w)))


def load_mcmc_npe():
    d = np.load(MCMC_NPE)
    assert list(d["feature_names"]) == NAMES9
    x = d["samples"].astype(np.float64).copy()
    x[:, 8] = np.log10(x[:, 8])
    cat = load_r2b("snrcatalog", 0)  # truth and catalog SNRs (the same for every noise seed)
    w = tf_weights(d["log_tf_weight"].astype(np.float64))
    return dict(x=x, w=w, truth=cat["truth"], image_snr=cat["image_snr"], halpha_snr=cat["halpha_snr"],
                ess=float(1.0 / np.sum(w ** 2) / len(w)))


def wstats(x, w):
    mu = w @ x
    return mu, np.sqrt(w @ (x - mu) ** 2)


def mcsamples(r, names, label):
    return MCSamples(samples=r["x"][:, [NAMES9.index(n) for n in names]], weights=r["w"],
                     names=names, labels=[LAB[n] for n in names], label=label,
                     ranges={n: PRIOR[n] for n in names},
                     settings=dict(GETDIST_SETTINGS, contours=CONTOURS))


def wquantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) - 0.5 * w[o]
    return np.interp(q, c / c[-1], x[o])


# ------------------------------------------------------------------ F1
def fig_training_loss(rep):
    csv = rep["R1"]["loss_csv"]
    df = pd.read_csv(csv)
    ep = np.arange(1, len(df) + 1)
    tr, va = df["train"].to_numpy(), df["valid"].to_numpy()
    best = int(np.argmin(va))
    assert best + 1 == rep["R1"]["best_epoch_1indexed"]

    fig, ax = plt.subplots(figsize=(COL, 0.75 * COL))
    ax.plot(ep, tr, color=COLORS["g1"], lw=1.1, label="training")
    ax.plot(ep, va, color=COLORS["g2"], lw=1.1, label="validation")
    ax.plot(ep[best], va[best], marker="o", ms=4.5, mfc="white", mec=COLORS["g2"], mew=1.1, ls="none",
            zorder=5, label=f"best validation (epoch {ep[best]})")
    lo = min(tr.min(), va.min())
    top = va[2] + 1.0
    ax.set_ylim(lo - 0.6, top)   # the first epochs run off the top
    ax.set_xlim(0, len(df) + 2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL loss")
    ax.legend(loc="upper right")

    # inset: last 60 epochs
    k0 = len(df) - 60
    ins = ax.inset_axes([0.36, 0.33, 0.40, 0.32])
    ins.plot(ep[k0:], tr[k0:], color=COLORS["g1"], lw=1.0)
    ins.plot(ep[k0:], va[k0:], color=COLORS["g2"], lw=1.0)
    ins.plot(ep[best], va[best], marker="o", ms=3.5, mfc="white", mec=COLORS["g2"], mew=1.0, zorder=5)
    ins.tick_params(labelsize=6, length=2)
    ins.set_xlim(ep[k0], ep[-1] + 1)
    savefig(fig, "fig_training_loss")

    r = rep["R1"]
    cfg = json.loads(MODEL_CFG.read_text())
    n_train, n_valid = int(cfg["data"]["size"]), int(cfg["test"]["size"])
    assert int(cfg["train"]["epoch_number"]) == r["n_epochs"]
    CAP["fig_training_loss"] = (
        f"Training (blue) and validation (red) loss of the NPE versus epoch. The loss is the negative "
        f"log-likelihood of the true parameters under the flow, averaged over galaxies, evaluated in the "
        f"normalized (linearly rescaled, prior-box) parameter coordinates the network is trained in; its "
        f"absolute value is therefore defined only up to a constant Jacobian term, and only differences are "
        f"meaningful. The network was trained on {n_train:,} simulated galaxies with fresh noise each epoch "
        f"and validated on a separate set of {n_valid:,} galaxies with fixed noise. The validation loss is lowest "
        f"{'at the final epoch' if r['best_epoch_1indexed'] == r['n_epochs'] else 'at epoch'} ({r['best_epoch_1indexed']} of {r['n_epochs']}; circle), where it has improved by "
        f"only {r['valid_improvement_last_10_epochs']:.4f} nats over the last 10 epochs; the train-validation "
        f"gap there is {r['generalization_gap_valid_minus_train']:.2f} nats. The first "
        f"{int(np.sum((tr > top) | (va > top)))} epochs lie above the plotted range (epoch-1 training and validation losses "
        f"{r['epoch1_train']:.2f} and {r['epoch1_valid']:.2f}). Inset: the last 60 epochs.")
    return dict(best_epoch=int(ep[best]))


# ------------------------------------------------------------------ F2
def choose_seed():
    """Seed whose per-setting posterior widths are closest to the seed average.

    Criterion: minimise sum over the four plotted settings and both shear
    components of |ln(sd_seed / mean_over_seeds(sd))|. Objective, and avoids
    showing a seed whose posterior is atypical (e.g. piled at a prior edge).
    """
    tags = ["snr110", "snr320", "hasnr11", "hasnr50"]
    sd = np.zeros((len(tags), len(SEEDS), 2))
    for i, t in enumerate(tags):
        for j, s in enumerate(SEEDS):
            r = load_r2b(t, s)
            sd[i, j] = wstats(r["x"][:, :2], r["w"])[1]
    score = np.abs(np.log(sd / sd.mean(axis=1, keepdims=True))).sum(axis=(0, 2))
    seed = int(np.argmin(score))
    print("seed scores:", dict(zip(SEEDS, np.round(score, 3))), "-> seed", seed)
    return seed, score


def gd_plotter(width):
    g = plots.get_single_plotter(width_inch=width, ratio=1)
    g.settings.axes_fontsize = 7.5
    g.settings.axes_labelsize = 9
    g.settings.legend_fontsize = 7.5
    g.settings.figure_legend_frame = False
    g.settings.legend_frame = False
    g.settings.alpha_filled_add = 0.75
    g.settings.linewidth_contour = 0.8
    g.settings.num_plot_contours = 2
    g.settings.solid_contour_palefactor = 0.6
    return g


def fig_posterior_snr(seed):
    runs = {t: load_r2b(t, seed) for t in ["snr110", "snr320", "hasnr11", "hasnr50", "snrcatalog"]}
    truth = runs["snrcatalog"]["truth"]
    cat_rhoI, cat_rhoS = runs["snrcatalog"]["image_snr"], runs["snrcatalog"]["halpha_snr"]
    for t in ["snr110", "snr320"]:
        assert abs(runs[t]["halpha_snr"] - cat_rhoS) < 1e-3
    for t in ["hasnr11", "hasnr50"]:
        assert abs(runs[t]["image_snr"] - cat_rhoI) < 1e-3

    panels = [
        ("a", [("snr110", "image_snr", r"\rho_I"), ("snr320", "image_snr", r"\rho_I")],
         rf"image SNR varied, $\rho_S = {cat_rhoS:.1f}$"),
        ("b", [("hasnr11", "halpha_snr", r"\rho_S"), ("hasnr50", "halpha_snr", r"\rho_S")],
         rf"H$\alpha$ SNR varied, $\rho_I = {cat_rhoI:.0f}$"),
    ]
    g = gd_plotter(DCOL / 2)
    fig, axes = plt.subplots(1, 2, figsize=(DCOL * 0.82, DCOL * 0.82 / 2 + 0.6), sharey=True)
    g.fig = fig
    stats = {}
    for ax, (tag, setup, title) in zip(axes, panels):
        roots, cols, handles = [], [], []
        for (run, key, sym), col in zip(setup, [COLORS["snr_low"], COLORS["snr_high"]]):
            r = runs[run]
            mu, sd = wstats(r["x"][:, :2], r["w"])
            stats[run] = dict(mu=mu, sd=sd, ess=r["ess"], snr=r[key])
            roots.append(mcsamples(r, ["g1", "g2"], run))
            cols.append(col)
            handles.append(Patch(facecolor=col, edgecolor=col, alpha=0.8,
                                 label=rf"${sym} = {r[key]:.0f}$  ($\sigma_{{g_1}} = {sd[0]:.3f}$)"))
        handles[1] = Line2D([], [], color=cols[1], lw=1.1, label=handles[1].get_label())
        g.plot_2d(roots, "g1", "g2", filled=[True, False], colors=cols, ax=ax, lims=[-0.1, 0.1, -0.1, 0.1],
                  lws=[0.6, 1.1])
        for run, col in zip([s[0] for s in setup], cols):
            m = stats[run]["mu"]
            ax.plot(*m, marker="o", ms=3.2, mfc=col, mec="white", mew=0.6, zorder=6)
        ax.axvline(truth[0], color=COLORS["truth"], lw=0.7, ls="--", zorder=7)
        ax.axhline(truth[1], color=COLORS["truth"], lw=0.7, ls="--", zorder=7)
        ax.plot(truth[0], truth[1], marker="*", ms=8, color=COLORS["truth"], mec="white", mew=0.5, zorder=8)
        handles.append(Line2D([], [], color=COLORS["truth"], marker="*", ms=7, lw=0.7, ls="--", label="truth"))
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_aspect("equal")
        ax.set_xticks([-0.08, -0.04, 0, 0.04, 0.08])
        ax.set_yticks([-0.08, -0.04, 0, 0.04, 0.08])
        ax.tick_params(labelsize=7.5, direction="in", top=True, right=True)
        ax.set_xlabel(r"$g_1$", fontsize=9)
        ax.set_title(f"({tag}) {title}", fontsize=9)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7,
                  frameon=False, handlelength=1.4, columnspacing=1.0)
    axes[0].set_ylabel(r"$g_2$", fontsize=9)
    axes[1].set_ylabel("")
    fig.subplots_adjust(wspace=0.06)
    savefig(fig, "fig_posterior_snr")

    s = stats
    CAP["fig_posterior_snr"] = (
        f"Shear posterior of one Xu3 test galaxy (index {GAL}; true $\\cos i = {truth[3]:.2f}$, true "
        f"$(g_1, g_2) = ({truth[0]:.4f}, {truth[1]:.4f})$, star and dashed lines) as the data quality is "
        f"changed with everything else fixed. (a) Image SNR $\\rho_I = {s['snr110']['snr']:.0f}$ (light) and "
        f"${s['snr320']['snr']:.0f}$ (dark lines), with the H$\\alpha$ SNR at its catalog value "
        f"$\\rho_S = {cat_rhoS:.1f}$. (b) H$\\alpha$ SNR $\\rho_S = {s['hasnr11']['snr']:.0f}$ (light) and "
        f"${s['hasnr50']['snr']:.0f}$ (dark lines), with the image SNR at its catalog value $\\rho_I = {cat_rhoI:.0f}$. "
        f"Each posterior is $10^4$ NPE draws (half from the 90-degree-rotated copy of the galaxy) reweighted "
        f"to the Tully-Fisher prior; contours enclose 68% and 95% of the posterior mass (kernel "
        f"density estimate with smoothing scale {GETDIST_SETTINGS['smooth_scale_2D']} and prior-boundary "
        f"correction at $|g_k| = 0.1$); dots are posterior means. All four runs use the same noise "
        f"realization (noise seed {seed} of the 5 generated), whose amplitude alone changes with SNR. "
        f"Raising $\\rho_I$ from {s['snr110']['snr']:.0f} to {s['snr320']['snr']:.0f} leaves the posterior essentially unchanged "
        f"($\\sigma_{{g_1}} = {s['snr110']['sd'][0]:.3f}$ vs ${s['snr320']['sd'][0]:.3f}$, "
        f"$\\sigma_{{g_2}} = {s['snr110']['sd'][1]:.3f}$ vs ${s['snr320']['sd'][1]:.3f}$), while raising "
        f"$\\rho_S$ from {s['hasnr11']['snr']:.0f} to {s['hasnr50']['snr']:.0f} shrinks $\\sigma_{{g_1}}$ from "
        f"{s['hasnr11']['sd'][0]:.3f} to {s['hasnr50']['sd'][0]:.3f}. For this galaxy $g_2$ lies almost along "
        f"the disk axes, where shear is degenerate with inclination, and stays close to the prior width "
        f"$0.2/\\sqrt{{12}} = 0.058$ at every SNR.")
    return stats


def fig_posterior_example():
    r = load_mcmc_npe()
    ms = mcsamples(r, ["g1", "g2"], "NPE")
    g = plots.get_subplot_plotter(width_inch=COL)
    g.settings.scaling = False
    g.settings.axes_fontsize = 7.5
    g.settings.axes_labelsize = 9
    g.settings.alpha_filled_add = 0.75
    g.settings.linewidth_contour = 0.8
    g.settings.num_plot_contours = 2
    g.settings.axis_tick_max_labels = 5
    g.triangle_plot([ms], ["g1", "g2"], filled=True, contour_colors=[COLORS["raw"]],
                    param_limits={p: PRIOR[p] for p in ("g1", "g2")},
                    markers={"g1": float(r["truth"][0]), "g2": float(r["truth"][1])},
                    marker_args={"color": COLORS["truth"], "lw": 0.8, "ls": "--"},
                    line_args=[{"color": COLORS["raw"], "lw": 1.0}])
    g.fig.legend(handles=[Patch(color=COLORS["raw"], alpha=0.8, label="NPE (68%, 95%)"),
                          Line2D([], [], color=COLORS["truth"], lw=0.8, ls="--", label="truth")],
                 loc="upper right", bbox_to_anchor=(0.97, 0.97), frameon=False, fontsize=8)
    savefig(g.fig, "fig_posterior_example")
    mu, sd = wstats(r["x"][:, :2], r["w"])
    CAP["fig_posterior_example"] = (
        f"Shear posterior of Xu3 test galaxy {GAL} (true $\\cos i = {r['truth'][3]:.2f}$, catalog SNRs "
        f"$\\rho_I = {r['image_snr']:.0f}$, $\\rho_S = {r['halpha_snr']:.1f}$) on its test-set noise realization, "
        f"from $10^4$ NPE draws reweighted to the TF prior (effective sample fraction {r['ess']:.2f}); "
        f"mean ({mu[0]:.3f}, {mu[1]:.3f}), sd ({sd[0]:.3f}, {sd[1]:.3f}).")
    return dict(mean=mu, sd=sd, ess=r["ess"])


# ------------------------------------------------------------------ F3
def ecdf_on(pit, grid):
    p = np.sort(pit)
    return np.searchsorted(p, grid, side="right") / len(p)


def fig_pp_shear(rep, z):
    rows = {r["parameter"]: r for r in rep["R3"]["rows"]["equal"]}
    n = len(z["pit_g1"])
    assert n == rep["R3"]["n_galaxies"]
    eps = math.sqrt(math.log(2 / 0.05) / (2 * n))
    grid = np.linspace(0, 1, 2001)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL, 1.28 * COL), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.08))
    a1.plot([0, 1], [0, 1], color="0.55", lw=0.7, ls="--", zorder=1)
    a1.fill_between(grid, grid - eps, grid + eps, color="0.85", lw=0, zorder=0)
    a2.axhspan(-100 * eps, 100 * eps, color="0.85", lw=0, zorder=0, label="95% DKW band")
    a2.axhline(0, color="0.55", lw=0.7, ls="--", zorder=1)
    ks = {}
    for p, ls in [("g1", "-"), ("g2", "-")]:
        pit = z[f"pit_{p}"]
        F = ecdf_on(pit, grid)
        # exact two-sided KS distance for the check against report.json
        srt = np.sort(pit)
        e = np.arange(1, n + 1) / n
        ks[p] = float(max(np.max(e - srt), np.max(srt - (e - 1.0 / n))))
        assert abs(ks[p] - rows[p]["ks_distance"]) < 1e-9
        a1.plot(grid, F, color=COLORS[p], lw=1.1, ls=ls, label=PLAIN[p])
        a2.plot(grid, 100 * (F - grid), color=COLORS[p], lw=1.0, ls=ls, label=PLAIN[p])
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1)
    a1.set_ylabel(r"fraction of galaxies with $u_i \leq u$")
    a1.legend(loc="upper left", handlelength=1.5)
    a2.set_ylabel(r"difference $[10^{-2}]$")
    a2.set_xlabel(r"$u$")
    lim = 1.25 * max(eps, *[np.max(np.abs(ecdf_on(z[f'pit_{p}'], grid) - grid)) for p in ("g1", "g2")])
    a2.set_ylim(-100 * lim, 100 * lim)
    h = [Patch(color="0.85", label="95% DKW band")]
    a2.legend(handles=h, loc="lower right", fontsize=7)
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    savefig(fig, "fig_pp_shear")

    CAP["fig_pp_shear"] = (
        f"Calibration of the shear posteriors on the Xu3 test set ({n:,} galaxies, one posterior each, "
        f"galaxies weighted equally). For each galaxy the probability integral transform (PIT) "
        f"$u_i = F_i(g_k^{{\\rm true}})$ is the Tully-Fisher-weighted fraction of its $10^4$ posterior draws "
        f"below the true shear. Top: empirical CDF of the PITs for $g_1$ (blue) and $g_2$ (red); a calibrated "
        f"posterior follows the diagonal. Bottom: the difference from the diagonal, in units of $10^{{-2}}$. The grey band is the 95% "
        f"Dvoretzky-Kiefer-Wolfowitz band, $\\pm\\sqrt{{\\ln(2/0.05)/2N}} = \\pm{eps:.4f}$. The "
        f"Kolmogorov-Smirnov distances are $D = {ks['g1']:.4f}$ ($g_1$) and ${ks['g2']:.4f}$ ($g_2$). The $g_1$ "
        f"curve (slightly positive at small $u$, negative at large $u$) indicates slightly too narrow "
        f"posteriors, by less than one percentage point in coverage (Table 1).")
    return dict(eps=eps, ks=ks, n=n)


# ------------------------------------------------------------------ Tab. 1
def pct(v, se=None, nd=1):
    s = f"{100 * v:.{nd}f}"
    return s if se is None else f"${s} \\pm {100 * se:.{nd}f}$"


def tab_coverage(rep, z):
    rows = {r["parameter"]: r for r in rep["R3"]["rows"]["equal"]}
    n = rep["R3"]["n_galaxies"]
    # cross-check against the per-galaxy PITs
    for p in ("g1", "g2"):
        for L in (0.5, 0.68, 0.95, 0.99):
            a = 0.5 * (1 - L)
            pit = z[f"pit_{p}"]
            c = np.mean((pit >= a) & (pit <= 1 - a))
            assert abs(c - rows[p][f"cov{int(round(100 * L))}_pit"]) < 1e-9
    eps = rows["g1"]["ks_dkw95_band"]
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\caption{Fraction of Xu3 test galaxies whose true shear lies inside the central credible interval of its "
        r"TF-weighted NPE posterior, for several credible levels "
        + rf"($N = {n:,}$".replace(",", "{,}") + r" galaxies). Uncertainties are $\sqrt{L(1-L)/N}$, "
        r"the scatter expected from the finite number of galaxies at credible level $L$.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{ccc}",
        r"\toprule",
        r"Credible level & $g_1$ [\%] & $g_2$ [\%] \\",
        r"\midrule",
    ]
    for L in (50, 68, 95, 99):
        c1, c2 = rows["g1"], rows["g2"]
        lines.append(rf"{L}\% & {pct(c1[f'cov{L}_pit'], c1[f'cov{L}_pit_se'], 2)} & "
                     rf"{pct(c2[f'cov{L}_pit'], c2[f'cov{L}_pit_se'], 2)} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "tab_coverage.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ nuisance (appendix)
def nuisance_rows(rep, z):
    rows = {r["parameter"]: r for r in rep["R3"]["rows"]["equal"]}
    n = rep["R3"]["n_galaxies"]
    truth, summ = z["truth"].astype(np.float64), z["tf_summary"].astype(np.float64)
    out = []
    for j, p in enumerate(NAMES9):
        lo, mid, hi, t = summ[:, 0, j], summ[:, 1, j], summ[:, 2, j], truth[:, j]
        if p == "theta_int":
            wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi  # noqa: E731
            rt = wrap(t - mid)
            below, above = rt < wrap(lo - mid), rt > wrap(hi - mid)
        else:
            below, above = t < lo, t > hi
        b, a = below.mean(), above.mean()
        assert abs(b - rows[p]["frac_below_q16"]) < 1e-9 and abs(a - rows[p]["frac_above_q84"]) < 1e-9
        cov = rows[p]["cov68_summary"]
        assert abs(cov - (1 - b - a)) < 1e-4, (p, cov, 1 - b - a)
        out.append(dict(p=p, below=b, above=a, cov=cov, cov_se=rows[p]["cov68_summary_se"],
                        tail_se=math.sqrt(0.16 * 0.84 / n)))
    return out, n


def fig_pp_nuisance(rep, z):
    rows, n = nuisance_rows(rep, z)
    ypos = np.arange(len(rows))[::-1].astype(float)
    ypos[2:] -= 0.5   # gap between shear and nuisance rows
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 0.95 * COL), sharey=True,
                                 gridspec_kw=dict(wspace=0.08, width_ratios=[1, 1.15]))
    for k, r in enumerate(rows):
        y = ypos[k]
        a1.errorbar(100 * r["cov"], y, xerr=200 * r["cov_se"], fmt="o", ms=3.5, color=COLORS["raw"],
                    mfc=COLORS["raw"], elinewidth=0.9, capsize=0)
        a2.errorbar(100 * r["below"], y + 0.15, xerr=200 * r["tail_se"], fmt="v", ms=3.5,
                    color=COLORS["g1"], elinewidth=0.9, capsize=0)
        a2.errorbar(100 * r["above"], y - 0.15, xerr=200 * r["tail_se"], fmt="^", ms=3.5,
                    color=COLORS["g2"], elinewidth=0.9, capsize=0)
    a1.axvline(68, color="0.5", lw=0.7, ls="--")
    a2.axvline(16, color="0.5", lw=0.7, ls="--")
    for a in (a1, a2):
        a.axhline(ypos[1] - 0.75, color="0.8", lw=0.6)
        a.tick_params(axis="y", length=0)
    a1.set_yticks(ypos)
    a1.set_yticklabels([PLAIN[r["p"]] for r in rows])
    a1.set_ylim(ypos[-1] - 0.7, ypos[0] + 0.7)
    a1.set_xlabel("68% coverage [%]")
    a2.set_xlabel("tail fraction [%]")
    # Limits follow the data (they were fixed for the 100k fibrepair model).
    covs = [100 * r["cov"] for r in rows] + [68.0]
    tails = [100 * r[k] for r in rows for k in ("below", "above")] + [16.0]
    a1.set_xlim(np.floor(min(covs) - 1.0), np.ceil(max(covs) + 1.0))
    a2.set_xlim(np.floor(min(tails) - 0.6), np.ceil(max(tails) + 0.6))
    a2.legend(handles=[Line2D([], [], marker="v", ls="", color=COLORS["g1"], ms=3.5, label=r"truth $<q_{16}$"),
                       Line2D([], [], marker="^", ls="", color=COLORS["g2"], ms=3.5, label=r"truth $>q_{84}$")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5,
              handletextpad=0.2, borderaxespad=0.2, frameon=False)
    savefig(fig, "fig_pp_nuisance")

    by = {r["p"]: r for r in rows}
    worst = max((r for r in rows if r["p"] not in ("g1", "g2")), key=lambda r: abs(r["cov"] - 0.68))
    others = [r for r in rows if r["p"] not in ("g1", "g2", "theta_int")]
    CAP["fig_pp_nuisance"] = (
        f"Calibration of all nine NPE parameters on the Xu3 test set ({n:,} galaxies, equal weights, "
        f"Tully-Fisher-weighted posteriors). Left: fraction of galaxies whose true value lies inside the "
        f"central 68% credible interval (between the 16th and 84th posterior percentiles). Right: fraction "
        f"below the 16th (down triangles) and above the 84th (up triangles) percentile; both should be 16%. "
        f"Error bars are $\\pm 2\\sigma$ binomial. For the shear the full P-P curve is shown in "
        f"Fig.~\\ref{{fig:pp_shear}}; for the seven nuisance parameters the stored posterior summaries are "
        f"only the 16th percentile, the mean and the 84th percentile, so calibration can be tested at $u = 0.16$ and "
        f"$0.84$ only (no full P-P curve and no 95% coverage). The intrinsic position angle is periodic; its "
        f"interval and truth are compared after wrapping relative to the posterior mean. All nuisance "
        f"parameters except $\\theta_{{\\rm int}}$ have 68% coverage between "
        f"{100 * min(r['cov'] for r in others):.1f}% and {100 * max(r['cov'] for r in others):.1f}%; "
        f"$\\theta_{{\\rm int}}$ {'under' if by['theta_int']['cov'] < 0.68 else 'over'}-covers "
        f"({100 * by['theta_int']['cov']:.1f}%), possibly an artefact of "
        f"percentile intervals on a circular coordinate.")
    if worst["p"] != "theta_int":
        print(f"NOTE fig_pp_nuisance: worst-calibrated nuisance is {worst['p']}, not theta_int; "
              "check the caption wording", flush=True)

    # appendix table
    lines = [r"\begin{table}", r"\centering",
             r"\caption{Tail fractions and 68\% coverage of the 16--84th percentile interval for all nine "
             rf"parameters on the Xu3 test set ($N = {n:,}$".replace(",", "{,}") + r" galaxies, equal weights, "
             r"TF-weighted posteriors). Expected values are 16\%, 16\% and 68\%; binomial $1\sigma$ "
             rf"uncertainties are {100 * rows[0]['tail_se']:.2f} and {100 * rows[0]['cov_se']:.2f} percentage "
             r"points.}",
             r"\label{tab:coverage_nuisance}", r"\begin{tabular}{lccc}", r"\toprule",
             r"Parameter & truth $< q_{16}$ [\%] & truth $> q_{84}$ [\%] & 68\% coverage [\%] \\",
             r"\midrule"]
    for k, r in enumerate(rows):
        if k == 2:
            lines.append(r"\midrule")
        lines.append(f"{PLAIN[r['p']]} & {100 * r['below']:.2f} & {100 * r['above']:.2f} & "
                     f"{100 * r['cov']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (OUT / "tab_coverage_nuisance.tex").write_text("\n".join(lines))


# ------------------------------------------------------------------ FA1
def fig_corner9d():
    r = load_mcmc_npe()
    # theta_int is periodic: unwrap around the circular mean so the plot is not split at +-pi.
    j = NAMES9.index("theta_int")
    th = r["x"][:, j]
    cmean = math.atan2(r["w"] @ np.sin(th), r["w"] @ np.cos(th))
    r["x"][:, j] = cmean + (th - cmean + np.pi) % (2 * np.pi) - np.pi
    r["truth"][j] = cmean + (r["truth"][j] - cmean + np.pi) % (2 * np.pi) - np.pi
    # A small number of draws lie far from the posterior core (e.g. at other orientations). They
    # inflate the sample standard deviation that sets the getdist kernel width and grid, which
    # oversmooths the core. Drop draws more than ZCUT robust widths (16-84% half-width) from the
    # median in any nuisance parameter, and report the removed posterior weight.
    ZCUT = 8.0
    bad = np.zeros(len(r["w"]), bool)
    for k in range(2, 9):
        q16, q50, q84 = wquantile(r["x"][:, k], r["w"], [0.16, 0.5, 0.84])
        bad |= np.abs(r["x"][:, k] - q50) > ZCUT * 0.5 * (q84 - q16)
    r["trim"] = dict(zcut=ZCUT, n=int(bad.sum()), weight=float(r["w"][bad].sum()))
    r["x"], r["w"] = r["x"][~bad], r["w"][~bad] / r["w"][~bad].sum()
    print("corner trim:", r["trim"])
    ms = mcsamples(r, NAMES9, "NPE")
    ms.getParamNames().parWithName("halpha_flux_true").label = LABELS["halpha_flux_true"]
    ms.setRanges({"theta_int": (None, None)})   # no boundary correction for the periodic angle
    ms.updateBaseStatistics()
    limits = {}
    for j, p in enumerate(NAMES9):
        if p in ("g1", "g2"):
            limits[p] = PRIOR[p]
            continue
        lo, hi = wquantile(r["x"][:, j], r["w"], [0.005, 0.995])
        lo, hi = min(lo, r["truth"][j]), max(hi, r["truth"][j])
        pad = 0.08 * (hi - lo)
        limits[p] = (max(PRIOR[p][0], lo - pad), min(PRIOR[p][1], hi + pad))
    g = plots.get_subplot_plotter(width_inch=DCOL)
    g.settings.scaling = False
    g.settings.axes_fontsize = 6.5
    g.settings.axes_labelsize = 8.5
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.75
    g.settings.linewidth_contour = 0.7
    g.settings.num_plot_contours = 2
    g.settings.axis_tick_x_rotation = 45
    g.settings.axis_tick_max_labels = 3
    rc = {"xtick.top": False, "ytick.right": False, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
          "xtick.major.width": 0.6, "ytick.major.width": 0.6}
    with plt.rc_context(rc):
        _triangle(g, ms, limits, r)
    savefig(g.fig, "fig_corner9d")
    return _corner_caption(r)


def _triangle(g, ms, limits, r):
    g.triangle_plot([ms], NAMES9, filled=True, contour_colors=[COLORS["raw"]],
                    param_limits=limits,
                    markers={p: float(r["truth"][j]) for j, p in enumerate(NAMES9)},
                    marker_args={"color": COLORS["truth"], "lw": 0.8, "ls": "-"},
                    line_args=[{"color": COLORS["raw"], "lw": 1.0}])
    g.fig.legend(handles=[Patch(color=COLORS["raw"], alpha=0.8, label="NPE posterior (68%, 95%)"),
                          Line2D([], [], color=COLORS["truth"], lw=0.8, label="truth")],
                 loc="upper right", bbox_to_anchor=(0.93, 0.93), frameon=False, fontsize=9)


def _corner_caption(r):
    mu, _ = wstats(r["x"], r["w"])
    CAP["fig_corner9d"] = (
        f"Full nine-parameter NPE posterior of Xu3 test galaxy {GAL} at its catalog SNRs "
        f"($\\rho_I = {r['image_snr']:.0f}$, $\\rho_S = {r['halpha_snr']:.1f}$; true $\\cos i = {r['truth'][3]:.2f}$), "
        f"from $10^4$ NPE draws (half from the 90-degree-rotated copy) reweighted to the Tully-Fisher prior "
        f"(effective sample fraction {r['ess']:.2f}), on the same noisy data as Fig.~\\ref{{fig:posterior-example}}. For display, {r['trim']['n']} isolated draws "
        f"({100 * r['trim']['weight']:.1f}% of the posterior weight) lying more than {r['trim']['zcut']:.0f} times "
        f"the 16-84% half-width from the median in some nuisance parameter are omitted; they would otherwise set the "
        f"kernel width and hide the core. Filled contours enclose 68% "
        f"and 95% of the 2D marginal mass; diagonal panels are 1D marginals (kernel density estimates with "
        f"smoothing scales {GETDIST_SETTINGS['smooth_scale_1D']} (1D) and {GETDIST_SETTINGS['smooth_scale_2D']} "
        f"(2D) and boundary correction at the training-prior edges; $\\theta_{{\\rm int}}$ is unwrapped around its circular mean). Lines mark the true values. The shear axes "
        f"span the full prior $|g_k| \\le 0.1$; other axes span the 0.5-99.5% weighted posterior range. $F_{{{{\\rm H}}\\alpha}}$ is in ${{\\rm erg\\,s^{{-1}}\\,cm^{{-2}}}}$. The "
        f"poorly constrained component $g_2$ is correlated with $\\cos i$, $v_{{\\rm circ}}$ and $R_h$, the "
        f"shape-shear degeneracy that the kinematics and the Tully-Fisher prior break.")
    return dict(ess=r["ess"], mean=mu)


# ------------------------------------------------------------------ main
def main():
    apply_style()
    rep = json.loads(A_JSON.read_text())
    z = np.load(A_NPZ, allow_pickle=True)
    assert list(z["feature_names"]) == NAMES9

    fig_training_loss(rep)
    seed, score = choose_seed()
    fig_posterior_snr(seed)   # no longer in the paper (replaced by fig_posterior_example, 2026-10-28)
    fig_posterior_example()
    fig_pp_shear(rep, z)
    tab_coverage(rep, z)
    fig_pp_nuisance(rep, z)
    fig_corner9d()

    order = ["fig_training_loss", "fig_posterior_snr", "fig_posterior_example", "fig_pp_shear", "fig_pp_nuisance", "fig_corner9d"]
    md = ["# Sec. 5 caption drafts", "",
          "Generated by `kl-nn/arch/diagnostics/paper_tests/make_paper_figures_sec5.py`; every number is "
          "read from the data files at run time. Tables: `tab_coverage.tex` (Tab. 1), "
          "`tab_coverage_nuisance.tex` (appendix).", ""]
    for k in order:
        md += [f"## {k}", "", CAP[k], ""]
    md += ["## Choice of noise seed (Figs. posterior_snr, corner9d)", "",
           "Five noise seeds were generated per SNR setting. The seed shown is the one whose TF-weighted "
           "posterior standard deviations of g1 and g2 in the four plotted settings are closest to the "
           "seed-averaged values (sum of |ln(sd_seed / sd_mean)|). Scores: "
           + ", ".join(f"seed {s}: {v:.3f}" for s, v in zip(SEEDS, score)) + ".", ""]
    (OUT / "captions_sec5.md").write_text("\n".join(md))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
