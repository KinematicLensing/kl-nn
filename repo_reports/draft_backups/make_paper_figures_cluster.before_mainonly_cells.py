#!/usr/bin/env python3
"""Paper figures for the cluster subsection of Sec. 6 (likelihood stacking in a non-constant shear field and
substructure detection).

Outputs (paper_style.OUT; run with KLNN_PAPER_TESTS_ROOT=$S/reports/paper-tests-1m):
  fig_cluster_maps.{pdf,png}       truth + galaxies, likelihood / KS maps (main + sub), difference maps for one
                                   typical realization, and the mean likelihood difference map over the ensemble
  fig_cluster_cells.{pdf,png}      polar-cell stacks against their reference shear, pooled over the ensemble:
                                   residual vs reference g_t and vs galaxies per cell
  fig_cluster_detection.{pdf,png}  single-map z at the subhalo over the ensemble, likelihood map vs KS
  fig_cluster_halofit.{pdf,png}    ln Bayes factor main+sub vs main from the NFW fits (optional paragraph)
  cluster_numbers.json             every number quoted in the text

Inputs (read-only), under $S/reports/paper-tests-1m/H_cluster/:
  s3_sub1_nRR/npe/{report.json, kappa_maps.npz}    h_cluster_analysis.py (tuned on s3_sub1)
  ensemble/summary.json                            h_cluster_ensemble_summary.py
  ensemble/halo_fit_summary.json                   h_cluster_halo_fit_summary.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KLNN_PAPER_TESTS_ROOT", "/ocean/projects/phy250048p/shared/reports/paper-tests-1m")
from paper_style import COL, COLORS, DCOL, OUT, apply_style, savefig  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
H = S / "reports/paper-tests-1m/H_cluster"
TAGS = [f"s3_sub1_n{r:02d}" for r in range(1, 51)]
R_MAX = 3.0


def load(tag):
    cfg = json.loads((S / "samples" / f"paper_tests_H_cluster_{tag}_config.json").read_text())
    man = pd.read_csv(S / "samples" / f"paper_tests_H_cluster_{tag}_manifest.csv")
    return cfg, man, np.load(H / tag / "npe/kappa_maps.npz")


def extent(edges):
    return [edges[0], edges[-1], edges[0], edges[-1]]


def pix_edges(c):
    d = c[1] - c[0]
    return np.append(c - d / 2, c[-1] + d / 2)


def decorate(ax, cfg, label):
    ax.add_patch(Circle((0, 0), R_MAX, fill=False, lw=0.6, ls="--", color="0.4"))
    ax.plot(cfg["sub_x_mpc"], cfg["sub_y_mpc"], marker="x", ms=4, mew=0.9, color="k")
    ax.set_xlim(-R_MAX - 0.2, R_MAX + 0.2); ax.set_ylim(-R_MAX - 0.2, R_MAX + 0.2)
    ax.set_aspect("equal")
    ax.text(0.04, 0.95, label, transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def fig_maps(det, num):
    z_lik = np.array(det["lik"]["z"])
    r_typ = int(np.argmin(np.abs(z_lik - np.median(z_lik))))
    tag = TAGS[r_typ]
    cfg, man, z = load(tag)
    num["maps_example"] = dict(tag=tag, n_gal=len(man), z_lik=float(z_lik[r_typ]), z_ks=float(det["ks"]["z"][r_typ]))
    le, ke = pix_edges(z["centres"]), z["ks_edges"]
    diffs = [np.load(H / t / "npe/kappa_maps.npz") for t in TAGS]
    mean_lik = np.mean([d["lik_sub"] - d["lik_main"] for d in diffs], axis=0)
    mean_ks = np.mean([d["ks_sub"] - d["ks_main"] for d in diffs], axis=0)
    sub_true = z["truth_sub"] - z["truth_main"]
    num["maps_mean_diff_peak"] = dict(lik=float(mean_lik.max()), ks=float(mean_ks.max()),
                                      true_pixel_peak=float(sub_true.max()))

    fig, axes = plt.subplots(2, 3, figsize=(DCOL, 0.70 * DCOL), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.06, hspace=0.10))
    kmax = 0.06
    top = [(z["truth_sub"], le, "(a) true $\\kappa$, main + sub"),
           (z["lik_sub"], le, "(b) likelihood map"),
           (z["ks_sub"], ke, "(c) KS map")]
    for ax, (m, e, lab) in zip(axes[0], top):
        im0 = ax.imshow(m.T, origin="lower", extent=extent(e), cmap="magma", vmin=0, vmax=kmax, interpolation="nearest")
        decorate(ax, cfg, lab)
    axes[0, 0].scatter(man["x_mpc"], man["y_mpc"], s=2.5, c="w", lw=0, alpha=0.9)
    dmax = 0.02
    bot = [(z["lik_sub"] - z["lik_main"], le, "(d) likelihood, difference"),
           (z["ks_sub"] - z["ks_main"], ke, "(e) KS, difference"),
           (mean_lik, le, "(f) likelihood, mean difference")]
    for ax, (m, e, lab) in zip(axes[1], bot):
        im1 = ax.imshow(m.T, origin="lower", extent=extent(e), cmap="RdBu_r", vmin=-dmax, vmax=dmax,
                        interpolation="nearest")
        ax.add_patch(Circle((cfg["sub_x_mpc"], cfg["sub_y_mpc"]), 0.5, fill=False, lw=0.7, color="k"))
        decorate(ax, cfg, lab)
    for ax in axes[1]:
        ax.set_xlabel(r"$x$ [Mpc]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$y$ [Mpc]")
    c0 = fig.colorbar(im0, ax=axes[0].tolist(), fraction=0.025, pad=0.015, extend="max")
    c0.set_label(r"$\kappa$")
    c1 = fig.colorbar(im1, ax=axes[1].tolist(), fraction=0.025, pad=0.015)
    c1.set_label(r"$\Delta\kappa$")
    savefig(fig, "fig_cluster_maps")


def pooled_cells():
    rows = []
    for t in TAGS:
        rep = json.loads((H / t / "npe/report.json").read_text())
        # main+sub scenario only: the main-only data share galaxies and noise, so pooling both would
        # double-count the same noise
        for sc in ("sub",):
            for c in rep["cells_results"][sc]["polar"]:
                rows.append((c["n"], *c["est"], *c["sd"], *c["ref"]))
    return np.array(rows)


def mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def wmean_se(x, s):
    """Precision-weighted mean; the error is the empirical scatter of the weighted terms (the stacked sd are
    ~15% conservative, so 1/sqrt(sum w) would overstate it)."""
    w = 1 / s ** 2
    mu = (w * x).sum() / w.sum()
    return mu, np.sqrt((w ** 2 * (x - mu) ** 2).sum()) / w.sum()


def fig_cells(summ, num):
    a = pooled_cells()
    n, est, sd, ref = a[:, 0], a[:, 1:3], a[:, 3:5], a[:, 5:7]
    res = est - ref
    num["cells"] = dict(n_cells_polar=int(len(a)),
                        polar_main=summ["cells"]["main/polar"]["comp1_linear"],
                        polar_sub=summ["cells"]["sub/polar"]["comp1_linear"],
                        cart_main_g1=summ["cells"]["main/cartesian"]["comp1_linear"],
                        cart_main_g2=summ["cells"]["main/cartesian"].get("comp2_linear"),
                        pull_sd={k: summ["cells"][k]["pull_sd"] for k in summ["cells"]},
                        pull_mean={k: summ["cells"][k]["pull_mean"] for k in summ["cells"]})

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(COL, 1.25 * COL), gridspec_kw=dict(hspace=0.38))
    s3 = n >= 3
    pull = res[s3] / sd[s3]
    bins = np.linspace(-4, 4, 33)
    u = np.linspace(-4, 4, 200)
    ax.plot(u, np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi), color=COLORS["prediction"], lw=0.8, ls=":",
            label=r"$\mathcal{N}(0,1)$")
    for k, (lab, col, ls) in enumerate([(r"$g_{\rm t}$", COLORS["g1"], "-"), (r"$g_\times$", COLORS["g2"], "--")]):
        ax.hist(pull[:, k], bins=bins, density=True, histtype="step", color=col, ls=ls, lw=1.2,
                label=f"{lab}: mean {pull[:, k].mean():+.2f}, sd {pull[:, k].std():.2f}")
    ax.set_xlabel(r"pull $(\hat g - g_{\rm ref})/\sigma$")
    ax.set_ylabel("density")
    ax.set_ylim(0, 0.8)
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.6)
    ax.text(0.97, 0.92, r"cells with $n \geq 3$", transform=ax.transAxes, ha="right", fontsize=7)
    num["cells"]["pulls_n_ge_3"] = dict(n=int(s3.sum()), mean=pull.mean(0).tolist(), sd=pull.std(0).tolist())

    # unweighted: every cell counts once, so cells with broad likelihoods show their shrinkage
    occ = []
    for k in range(1, 11):
        s = n == k if k < 10 else n >= 10
        occ.append((k, s.sum(), *mean_se(res[s, 0]), *mean_se(res[s, 1])))
    occ = np.array(occ)
    bx.errorbar(occ[:, 0] - 0.12, occ[:, 2], occ[:, 3], fmt="o", color=COLORS["g1"], ms=3.5, capsize=1.5, lw=0.9,
                label=r"$g_{\rm t}$")
    bx.errorbar(occ[:, 0] + 0.12, occ[:, 4], occ[:, 5], fmt="s", color=COLORS["g2"], mfc="white", ms=3.5,
                capsize=1.5, lw=0.9, label=r"$g_\times$")
    bx.axhline(0, color="0.5", lw=0.6)
    bx.set_xticks(range(1, 11)); bx.set_xticklabels([str(k) for k in range(1, 10)] + [r"$\geq$10"])
    bx.set_xlabel(r"galaxies per cell $n$")
    bx.set_ylabel(r"mean $\hat g - g_{\rm ref}$")
    bx.legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=1.0)
    num["cells"]["resid_vs_n_unweighted"] = occ.tolist()
    w = 1 / sd[s3, 0] ** 2
    num["cells"]["n_ge_3_gt"] = dict(unweighted=list(mean_se(res[s3, 0])), weighted=list(wmean_se(res[s3, 0], sd[s3, 0])),
                                     mean_ref_gt=float(ref[s3, 0].mean()))
    savefig(fig, "fig_cluster_cells")


def fig_detection(det, num):
    fig, ax = plt.subplots(figsize=(COL, 0.72 * COL))
    bins = np.arange(-1.5, 5.51, 0.5)
    for key, lab, col, ls in (("lik", "likelihood map", COLORS["stacked"], "-"), ("ks", "Kaiser–Squires", "0.35", "--")):
        zz = np.array(det[key]["z"])
        ax.hist(zz, bins=bins, histtype="step", color=col, ls=ls, lw=1.3,
                label=f"{lab} (median {np.median(zz):.1f})")
        num[f"z_{key}"] = dict(median=float(np.median(zz)), q16=float(np.percentile(zz, 16)),
                               q84=float(np.percentile(zz, 84)), frac_gt_2=float(np.mean(zz > 2)),
                               frac_gt_3=float(np.mean(zz > 3)), diff_mean=det[key]["diff_mean"],
                               null_sd=det[key]["null_sd"])
    ax.axvline(2, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel(r"subhalo significance $z$")
    ax.set_ylabel("realizations")
    ax.legend(loc="upper left")
    savefig(fig, "fig_cluster_detection")


def fig_halofit(hf, num):
    fig, ax = plt.subplots(figsize=(COL, 0.72 * COL))
    bins = np.arange(-2, 16.1, 1.0)
    for key, lab, col, ls in (("sub_lnB_fixed", "with subhalo", COLORS["stacked"], "-"),
                              ("main_lnB_fixed", "main halo only", "0.35", "--")):
        b = np.clip(hf[key]["values"], bins[0], bins[-1] - 1e-6)
        ax.hist(b, bins=bins, histtype="step", color=col, ls=ls, lw=1.3, label=lab)
    ax.axvline(1, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel(r"$\ln B$ (main + sub vs main)")
    ax.set_ylabel("realizations")
    ax.legend(loc="upper right")
    num["halo_fit"] = {k: {kk: vv for kk, vv in hf[k].items() if kk != "values"}
                       for k in hf if k not in ("rows",) and isinstance(hf[k], dict)}
    savefig(fig, "fig_cluster_halofit")


def main():
    apply_style()
    summ = json.loads((H / "ensemble/summary.json").read_text())
    hf = json.loads((H / "ensemble/halo_fit_summary.json").read_text())
    det = summ["detection"]
    num = dict(n_realizations=len(TAGS), n_gal=[p["n"] for p in summ["per_realization"]])
    fig_maps(det, num)
    fig_cells(summ, num)
    fig_detection(det, num)
    fig_halofit(hf, num)
    (OUT / "cluster_numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print("wrote figures and", OUT / "cluster_numbers.json")


if __name__ == "__main__":
    main()
