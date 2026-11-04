#!/usr/bin/env python3
"""Paper figures for the cluster subsection of Sec. 6 (likelihood stacking in a non-constant shear field and
substructure detection).

Outputs (paper_style.OUT; run with KLNN_PAPER_TESTS_ROOT=$S/reports/paper-tests-1m):
  fig_cluster_maps.{pdf,png}       truth + galaxies, likelihood / KS maps (main + sub), difference maps for one
                                   typical realization, and the mean likelihood difference map over the ensemble
  fig_cluster_cells.{pdf,png}      main-halo-only polar cells over the ensemble: reference and stacked g_t per
                                   cell (plain means over realizations) and the residual vs galaxies per cell
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
    """Main-halo-only polar cells of every realization: (realization, cell, n, est_t, est_x, sd_t, sd_x, ref_t, ref_x)."""
    rows = []
    for r, t in enumerate(TAGS):
        rep = json.loads((H / t / "npe/report.json").read_text())
        for c in rep["cells_results"]["main"]["polar"]:
            rows.append((r, c["cell"], c["n"], *c["est"], *c["sd"], *c["ref"]))
    return np.array(rows)


def mean_se(x):
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def wmean_se(x, s):
    """Precision-weighted mean; the error is the empirical scatter of the weighted terms (the stacked sd are
    slightly conservative, so 1/sqrt(sum w) would overstate it)."""
    w = 1 / s ** 2
    mu = (w * x).sum() / w.sum()
    return mu, np.sqrt((w ** 2 * (x - mu) ** 2).sum()) / w.sum()


def fmean_se(x, s):
    """Precision-weighted mean with its formal error 1/sqrt(sum w), as for the m and c fits."""
    w = 1 / s ** 2
    return (w * x).sum() / w.sum(), 1 / np.sqrt(w.sum())


def wfit(x, y, s):
    """Weighted least squares y = c + m x with weights 1/s^2; returns (c, m) and their standard errors."""
    X = np.column_stack([np.ones_like(x), x]); w = 1 / s ** 2
    cov = np.linalg.inv(X.T @ (X * w[:, None])); b = cov @ X.T @ (w * y)
    return b, np.sqrt(np.diag(cov))


def draw_polar(ax, cells, values, norm, cmap):
    from matplotlib.patches import Wedge
    for c, v in zip(cells, values):
        col = cmap(norm(v))
        if c["kind"] == "disk":
            ax.add_patch(Circle((0, 0), c["r_hi"], fc=col, ec="0.3", lw=0.4))
        else:
            ax.add_patch(Wedge((0, 0), c["r_hi"], np.degrees(c["phi_lo"]), np.degrees(c["phi_hi"]),
                               width=c["r_hi"] - c["r_lo"], fc=col, ec="0.3", lw=0.4))
    ax.plot(0, 0, marker="+", ms=5, mew=0.9, color="k")
    ax.set_xlim(-R_MAX - 0.15, R_MAX + 0.15); ax.set_ylim(-R_MAX - 0.15, R_MAX + 0.15); ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [Mpc]")


def fig_cells(summ, num):
    from matplotlib.colors import Normalize
    from h_cluster_analysis import polar_cells
    a = pooled_cells()
    cell, n = a[:, 1].astype(int), a[:, 2]
    est, sd, ref = a[:, 3:5], a[:, 5:7], a[:, 7:9]
    res = est - ref
    # Each realization is one cluster and counts once. m, c: a fit over the cells of each realization (cells weighted
    # by their stacked precision, as for a real cluster), then the plain mean and standard error over realizations.
    # Maps and residual vs n: plain means over realizations.
    real = a[:, 0].astype(int)
    out = dict(n_cells=int(len(a)), n_cells_n_ge_3=int((n >= 3).sum()), max_abs_gref_x=float(np.abs(ref[:, 1]).max()),
               rms_gref_x=float(np.sqrt(np.mean(ref[:, 1] ** 2))))
    for lab, sel0 in (("all", n >= 1), ("n_ge_3", n >= 3)):
        per = []
        for r in np.unique(real):
            sel = sel0 & (real == r)
            (c, m), _ = wfit(ref[sel, 0], res[sel, 0], sd[sel, 0])
            (cl, al), _ = wfit(ref[sel, 0], res[sel, 1], sd[sel, 1])   # leakage of g_t into g_x
            w = 1 / sd[sel, 1] ** 2
            per.append((m, c, (w * res[sel, 1]).sum() / w.sum(), al))
        per = np.array(per)
        out[lab] = {k: dict(mean=float(per[:, i].mean()), err=float(per[:, i].std(ddof=1) / np.sqrt(len(per))),
                            median=float(np.median(per[:, i])))
                    for i, k in enumerate(("m_t", "c_t", "c_x", "leak_alpha"))}
    ids = np.unique(cell)
    ref_map = np.array([ref[cell == k, 0].mean() for k in ids])
    est_map = np.array([est[cell == k, 0].mean() for k in ids])
    cfg, man, _ = load(TAGS[0])
    cells, _ = polar_cells(man, cfg)
    cells = [cells[k] for k in ids]
    # both maps are positive everywhere (a single halo), so a sequential colour map from zero
    vmax = float(np.ceil(max(ref_map.max(), est_map.max()) * 200) / 200)
    norm, cmap = Normalize(0, vmax), plt.get_cmap("viridis")
    out["maps"] = dict(ref_range=[float(ref_map.min()), float(ref_map.max())],
                       est_range=[float(est_map.min()), float(est_map.max())], vmax=vmax)

    fig = plt.figure(figsize=(DCOL, 0.34 * DCOL))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.06, 0.32, 1.25], wspace=0.08)
    ax_r, ax_e, cax, bx = (fig.add_subplot(gs[0, k]) for k in (0, 1, 2, 4))
    draw_polar(ax_r, cells, ref_map, norm, cmap); draw_polar(ax_e, cells, est_map, norm, cmap)
    ax_r.set_ylabel(r"$y$ [Mpc]"); ax_e.tick_params(labelleft=False)
    for ax, lab in ((ax_r, r"(a) reference $g_{\rm t}$"), (ax_e, r"(b) stacked $\hat g_{\rm t}$")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$g_{\rm t}$")
    occ = []
    for k in range(1, 11):
        sl = n == k if k < 10 else n >= 10
        occ.append((k, sl.sum(), *mean_se(res[sl, 0]), *mean_se(res[sl, 1])))
    occ = np.array(occ)
    bx.errorbar(occ[:, 0] - 0.12, occ[:, 2], occ[:, 3], fmt="o", color=COLORS["g1"], ms=3.5, capsize=1.5, lw=0.9,
                label=r"$g_{\rm t}$")
    bx.errorbar(occ[:, 0] + 0.12, occ[:, 4], occ[:, 5], fmt="s", color=COLORS["g2"], mfc="white", ms=3.5,
                capsize=1.5, lw=0.9, label=r"$g_\times$")
    bx.axhline(0, color="0.5", lw=0.6)
    bx.set_xticks(range(1, 11)); bx.set_xticklabels([str(k) for k in range(1, 10)] + [r"$\geq$10"])
    bx.set_xlabel(r"galaxies per cell $n$"); bx.set_ylabel(r"$\hat g - g_{\rm ref}$")
    bx.yaxis.set_label_position("right"); bx.yaxis.tick_right()
    bx.text(0.03, 0.97, "(c)", transform=bx.transAxes, va="top", ha="left", fontsize=7.5)
    bx.legend(loc="lower right", ncol=2, handletextpad=0.3, columnspacing=1.0)
    fig.canvas.draw()  # the maps have equal aspect: match the colour bar to their drawn height
    pr, pe_ = ax_r.get_position(), ax_e.get_position()
    cax.set_position([pe_.x1 + 0.012, pe_.y0, 0.012, pe_.height])
    out["resid_vs_n"] = occ.tolist()
    num["cells"] = out
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


def fig_sigma(hs, num):
    """Likelihood-ratio significance of the subhalo at its known position (h_cluster_halo_sigma.py)."""
    fig, ax = plt.subplots(figsize=(COL, 0.72 * COL))
    bins = np.arange(0, 9.01, 0.5)
    for key, lab, col, ls in (("sub_sigma", "with subhalo", COLORS["stacked"], "-"),
                              ("main_sigma", "main halo only", "0.35", "--")):
        s = np.clip([r[key] for r in hs["rows"]], bins[0], bins[-1] - 1e-6)
        ax.hist(s, bins=bins, histtype="step", color=col, ls=ls, lw=1.3, label=lab)
    ax.set_xlabel(r"subhalo significance $\sigma$")
    ax.set_ylabel("realizations")
    ax.legend(loc="upper right")
    num["halo_sigma"] = {k: v for k, v in hs.items() if k != "rows"}
    num["halo_sigma"]["main_sigma"]["max"] = float(max(r["main_sigma"] for r in hs["rows"]))
    savefig(fig, "fig_cluster_sigma")


def fig_localization(hs, num):
    """Posterior-median subhalo position of the free-position fit, one point per realization (data with subhalo),
    colored by the detection significance of the known-position fit."""
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(COL, 0.8 * COL))
    pos = []
    for t in TAGS:
        cfg = json.loads((S / "samples" / f"paper_tests_H_cluster_{t}_config.json").read_text())
        p = json.loads((H / t / "npe/halo_fit.json").read_text())["sub"]["main+sub_free"]["params"]
        kpm = cfg["kpc_per_arcsec"] / 1000.0
        pos.append((p["x_sub_arcsec"]["median"] * kpm, p["y_sub_arcsec"]["median"] * kpm))
    pos = np.array(pos)
    ax.add_patch(Circle((cfg["sub_x_mpc"], cfg["sub_y_mpc"]), 0.5, fill=False, lw=0.8, color="k"))
    ax.plot(0, 0, marker="+", ms=7, mew=1.0, color="k")
    sig = np.array([r["sub_sigma"] for r in hs["rows"]])
    assert [r["tag"] for r in hs["rows"]] == TAGS
    order = np.argsort(-sig)  # low significance drawn on top
    norm = Normalize(0, 7)
    sc = ax.scatter(pos[order, 0], pos[order, 1], c=sig[order], cmap="viridis", norm=norm, s=14,
                    edgecolors="0.2", linewidths=0.3, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, extend="max" if sig.max() > norm.vmax else "neither")
    cb.set_label(r"detection significance $\sigma$")
    ax.plot(cfg["sub_x_mpc"], cfg["sub_y_mpc"], marker="x", ms=6, mew=1.2, color="k", zorder=4)
    assert np.all(np.abs(pos - [0.75, 0.0]) < 1.75), "a fitted position falls outside the zoomed panel"
    ax.set_xlim(-1.0, 2.5); ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [Mpc]"); ax.set_ylabel(r"$y$ [Mpc]")
    off = np.hypot(pos[:, 0] - cfg["sub_x_mpc"], pos[:, 1] - cfg["sub_y_mpc"])
    num["localization"] = dict(offset_median=float(np.median(off)), frac_lt_05=float(np.mean(off < 0.5)),
                               offset_q16=float(np.percentile(off, 16)), offset_q84=float(np.percentile(off, 84)))
    savefig(fig, "fig_cluster_localization")


def main():
    apply_style()
    summ = json.loads((H / "ensemble/summary.json").read_text())
    hf = json.loads((H / "ensemble/halo_fit_summary.json").read_text())
    hs = json.loads((H / "ensemble/halo_sigma.json").read_text())
    num = dict(n_realizations=len(TAGS), n_gal=[p["n"] for p in summ["per_realization"]])
    # The kappa maps (fig_maps, fig_detection) and the ln B histogram (fig_halofit) were dropped from the paper on
    # 2026-10-29; their functions are kept for reference.
    fig_cells(summ, num)
    fig_sigma(hs, num)
    fig_localization(hs, num)
    num["halo_fit"] = {k: {kk: vv for kk, vv in hf[k].items() if kk != "values"}
                       for k in hf if k not in ("rows",) and isinstance(hf[k], dict)}
    # Subhalo mass (known position) averaged over all realizations and over those detected above 3 sigma
    msub = np.array([r["sub_fixed_msub"]["median"] for r in hf["rows"]])
    det3 = np.array([r["sub_sigma"] for r in hs["rows"]]) > 3
    num["msub_mean"] = {lab: dict(n=int(sel.sum()), mean=float(msub[sel].mean()),
                                  se=float(msub[sel].std(ddof=1) / np.sqrt(sel.sum())))
                        for lab, sel in (("all", np.ones_like(det3)), ("sigma_gt_3", det3))}
    (OUT / "cluster_numbers.json").write_text(json.dumps(num, indent=1, default=float))
    print("wrote figures and", OUT / "cluster_numbers.json")


if __name__ == "__main__":
    main()
