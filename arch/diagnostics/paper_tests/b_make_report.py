#!/usr/bin/env python3
"""Group B report: R2b contours, 9D corner plots, T2b response / selection.

Reads only new paper_tests products:
  $S/cache/paper_tests_B_inference/r2b/*.npz              (b_r2b_sample.py)
  $S/cache/paper_tests_B_inference/xu3_sigma_summary.npz  (b_xu3_sigma_summary.py)
  $S/reports/paper-tests/B_inference/t2b_*.json/.cubes.npz (b_t2b_analysis.py)
Writes report.html, report.json and PNG/PDF into B_inference/ and paper_figs/.
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, "/jet/home/xwang30/kl-nn/arch")
sys.path.insert(0, "/jet/home/xwang30/kl-nn/arch/diagnostics")
from shear_bias_report import component_metrics  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
R2B = S / "cache/paper_tests_B_inference/r2b"
OUT = S / "reports/paper-tests/B_inference"
FIGS = S / "reports/paper-tests/paper_figs"
SIGMA_PI = 0.2 / np.sqrt(12)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1b1f24", "#5c6770", "#d9dee3"
NAMES9 = ["g1", "g2", "theta_int", "cosi", "v0", "vcirc", "rscale", "hlr", "halpha_flux_true"]
LABELS9 = [r"$g_1$", r"$g_2$", r"$\theta_{\rm int}$", r"$\cos i$", r"$v_0$", r"$v_{\rm circ}$",
           r"$r_{\rm scale}$", r"$r_{\rm hlr}$", r"$\log_{10}F_{\rm H\alpha}$"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 160})


def wstats(g, w):
    mu = w @ g
    return mu, np.sqrt(w @ (g - mu) ** 2)


def load_r2b(gal, snr, seed):
    d = np.load(R2B / f"gal{gal}_snr{snr}_seed{seed}.npz")
    lw = d["log_weight"]; w = np.exp(lw - lw.max()); w /= w.sum()
    return d, w


def credible_levels(H, fracs):
    flat = np.sort(H.ravel())[::-1]; cum = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(cum, f)] for f in fracs]


def r2b_figure(gal, seed=0):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    edges = np.linspace(-0.1, 0.1, 81); ctr = 0.5 * (edges[1:] + edges[:-1])
    handles = []
    for snr, col, ls in (("50", C1, "-"), ("200", C2, "-")):
        d, w = load_r2b(gal, snr, seed)
        g = d["samples"][:, :2].astype(float)
        H, _, _ = np.histogram2d(g[:, 0], g[:, 1], bins=[edges, edges], weights=w)
        H = gaussian_filter(H, 1.8)
        lv = credible_levels(H, (0.95, 0.68))
        ax.contourf(ctr, ctr, H.T, levels=[lv[0], lv[1], H.max()], colors=[col, col], alpha=0.12)
        ax.contour(ctr, ctr, H.T, levels=lv, colors=col, linewidths=[1.2, 2.0], linestyles=ls)
        mu, sd = wstats(g, w)
        ax.plot(*mu, "o", color=col, ms=6, mec="white", mew=1.2)
        handles.append(plt.Line2D([], [], color=col, lw=2,
                       label=rf"$\rho_I={snr}$: $\sigma_{{g_1}}={sd[0]:.3f}$, $\sigma_{{g_2}}={sd[1]:.3f}$"))
    truth = d["truth"]
    ax.plot(truth[0], truth[1], marker="*", color=INK, ms=13, mec="white", mew=1.0, ls="none")
    handles.append(plt.Line2D([], [], marker="*", color=INK, ls="none", ms=11, label="truth"))
    ax.set_xlim(-0.1, 0.1); ax.set_ylim(-0.1, 0.1); ax.set_aspect("equal")
    ax.set_xlabel(r"$g_1$"); ax.set_ylabel(r"$g_2$")
    rho_s = float(np.asarray(d["central_halpha_snr"]).ravel()[0])
    ax.set_title(rf"Xu3 galaxy {gal}  ($\cos i={truth[3]:.2f}$, $\rho_S={rho_s:.1f}$)", fontsize=10, color=INK)
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


def rhos_figure(gal, seed=0):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    edges = np.linspace(-0.1, 0.1, 81); ctr = 0.5 * (edges[1:] + edges[:-1])
    handles = []
    for tag, lab, col in (("hasnr8", r"$\rho_S=8$", C1), ("hasnr60", r"$\rho_S=60$", C2)):
        d = np.load(R2B / f"gal{gal}_{tag}_seed{seed}.npz")
        lw = d["log_weight"]; w = np.exp(lw - lw.max()); w /= w.sum()
        g = d["samples"][:, :2].astype(float)
        H, _, _ = np.histogram2d(g[:, 0], g[:, 1], bins=[edges, edges], weights=w)
        H = gaussian_filter(H, 1.8); lv = credible_levels(H, (0.95, 0.68))
        ax.contourf(ctr, ctr, H.T, levels=[lv[0], lv[1], H.max()], colors=[col, col], alpha=0.12)
        ax.contour(ctr, ctr, H.T, levels=lv, colors=col, linewidths=[1.2, 2.0])
        mu, sd = wstats(g, w); ax.plot(*mu, "o", color=col, ms=6, mec="white", mew=1.2)
        handles.append(plt.Line2D([], [], color=col, lw=2, label=rf"{lab}: $\sigma_{{g_1}}={sd[0]:.3f}$, $\sigma_{{g_2}}={sd[1]:.3f}$"))
    t = d["truth"]
    ax.plot(t[0], t[1], marker="*", color=INK, ms=13, mec="white", ls="none")
    handles.append(plt.Line2D([], [], marker="*", color=INK, ls="none", ms=11, label="truth"))
    ax.set_xlim(-0.1, 0.1); ax.set_ylim(-0.1, 0.1); ax.set_aspect("equal"); ax.set_xlabel(r"$g_1$"); ax.set_ylabel(r"$g_2$")
    ax.set_title(rf"Xu3 galaxy {gal}, $\rho_I={float(np.asarray(d['image_snr']).ravel()[0]):.0f}$ fixed", fontsize=10, color=INK)
    ax.legend(handles=handles, loc="lower left", fontsize=7.5, framealpha=0.9); fig.tight_layout()
    return fig


def r2b_table(gal, seeds=range(5), tags=("snr50", "snrcatalog", "snr200")):
    rows = {}
    for snr in tags:
        vals = []
        for s in seeds:
            f = R2B / f"gal{gal}_{snr}_seed{s}.npz"
            if not f.is_file():
                continue
            d = np.load(f); lw = d["log_weight"]; w = np.exp(lw - lw.max()); w /= w.sum()
            mu, sd = wstats(d["samples"][:, :2].astype(float), w)
            vals.append([mu[0], mu[1], sd[0], sd[1], np.hypot(*sd), float(np.asarray(d["ess_fraction"]).ravel()[0]),
                         float(np.asarray(d["image_snr"]).ravel()[0])])
        v = np.asarray(vals)
        rows[snr] = dict(n_seeds=len(v), image_snr=float(v[0, 6]), mean_g1=v[:, 0].mean(), mean_g2=v[:, 1].mean(),
                         sd_g1=v[:, 2].mean(), sd_g2=v[:, 3].mean(), sigma_eps=v[:, 4].mean(),
                         sigma_eps_seed_std=v[:, 4].std(ddof=1) if len(v) > 1 else float("nan"),
                         sd_g1_seed_std=v[:, 2].std(ddof=1) if len(v) > 1 else float("nan"),
                         sd_g2_seed_std=v[:, 3].std(ddof=1) if len(v) > 1 else float("nan"),
                         mean_g1_seed_std=v[:, 0].std(ddof=1) if len(v) > 1 else float("nan"),
                         mean_g2_seed_std=v[:, 1].std(ddof=1) if len(v) > 1 else float("nan"),
                         seed0_sigma_eps=float(v[0, 4]), ess_fraction=v[:, 5].mean())
    d, _ = load_r2b(gal, "50", 0)
    return dict(galaxy=gal, truth=d["truth"].tolist(), central_halpha_snr=float(np.asarray(d["central_halpha_snr"]).ravel()[0]),
                rows=rows)


def corner_figure(gal):
    import corner
    d, w = load_r2b(gal, "catalog", 0)
    x = d["samples"].astype(float).copy(); t = d["truth"].astype(float).copy()
    x[:, 8] = np.log10(x[:, 8]); t[8] = np.log10(t[8])
    q = np.quantile(x[:, 2], [0.005, 0.995])
    rng = [(-0.1, 0.1), (-0.1, 0.1), (max(-np.pi, q[0] - 0.15), min(np.pi, q[1] + 0.15)), (0, 1)] + [0.995] * 5
    fig = corner.corner(x, weights=w, labels=LABELS9, truths=t, truth_color=C2, color=C1, range=rng,
                        bins=40, smooth=1.0, levels=(0.393, 0.865), plot_datapoints=False,
                        fill_contours=True, label_kwargs=dict(fontsize=11))
    rho_s = float(np.asarray(d["central_halpha_snr"]).ravel()[0]); rho_i = float(np.asarray(d["image_snr"]).ravel()[0])
    fig.suptitle(rf"Xu3 galaxy {gal}: NPE posterior (TF-weighted, $10^4$ identity+R90 draws), "
                 rf"$\rho_I={rho_i:.0f}$, $\rho_S={rho_s:.1f}$", fontsize=12, y=1.01)
    return fig, dict(galaxy=gal, image_snr=rho_i, central_halpha_snr=rho_s, truth=d["truth"].tolist(),
                     tf_ess_fraction=float(np.asarray(d["ess_fraction"]).ravel()[0]),
                     posterior_mean=(w @ d["samples"].astype(float)).tolist())


def save(fig, name, paper=False):
    p = OUT / f"{name}.png"; fig.savefig(p, bbox_inches="tight")
    if paper:
        fig.savefig(FIGS / f"B_{name}.png", bbox_inches="tight"); fig.savefig(FIGS / f"B_{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return p


def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()


# ---------------- T2b figures ----------------
def t2b_sigma_figure(block, cub, extra=()):
    fit = block["fit_R_sigma"]["per_component"]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sd = cub["sd"][:, 0]; R = cub["R"]
    x_all = np.concatenate([sd[:, 0] ** 2, sd[:, 1] ** 2]); y_all = np.concatenate([R[:, 0, 0], R[:, 1, 1]])
    qs = np.quantile(x_all, np.linspace(0, 1, 11))
    xb, yb, eb = [], [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (x_all >= lo) & (x_all <= hi)
        xb.append(x_all[m].mean()); yb.append(y_all[m].mean()); eb.append(y_all[m].std(ddof=1) / np.sqrt(m.sum()))
    xx = np.linspace(0, 1.15 * SIGMA_PI ** 2, 100)
    ax.plot(xx, 1 - xx / SIGMA_PI ** 2, color=MUTED, ls="--", lw=1.5, label=r"Gaussian shrinkage $1-\sigma^2/\sigma_\pi^2$")
    ax.plot(xx, fit["a"] + fit["b"] * xx, color=C2, lw=2,
            label=rf"fit $a+b\sigma^2$: $a={fit['a']:.3f}\pm{fit['a_se']:.3f}$, $b={fit['b']:.0f}\pm{fit['b_se']:.0f}$")
    ax.errorbar(xb, yb, yerr=eb, fmt="o", color=C1, ms=6, mec="white", capsize=2,
                label=r"deciles of $\sigma_{g_k}^2$ (pooled $R_{11}$, $R_{22}$)")
    for lab, f2, col in extra:
        ax.plot(xx, f2["a"] + f2["b"] * xx, color=col, lw=1.5, ls="-.",
                label=rf"{lab}: $a={f2['a']:.2f}$, $b={f2['b']:.0f}\pm{f2['b_se']:.0f}$")
    ax.set_xlabel(r"per-component posterior variance $\sigma_{g_k}^2$ (zero-shear state)")
    ax.set_ylabel(r"shear response $R_{kk}$")
    ax.axvline(SIGMA_PI ** 2, color=MUTED, lw=0.8, ls=":")
    ax.text(SIGMA_PI ** 2, -0.07, r" $\sigma_\pi^2$", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(-0.1, 1.45); ax.set_xlim(0, xx[-1])
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    return fig


def t2b_cosi_figure(pairs):
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for (lab, block), col, off in zip(pairs, (C1, C2, C3), (-0.012, 0.0, 0.012)):
        b = [x for x in block["by_cosi"] if "R" in x]
        xc = np.array([0.5 * (x["lo"] + x["hi"]) for x in b]) + off
        for k, mk in ((0, "o"), (1, "s")):
            ax.errorbar(xc + (k - 0.5) * 0.01, [x["R"][k][k] for x in b], yerr=[x["R_se"][k][k] for x in b],
                        fmt=mk, color=col, ms=5, mec="white", capsize=2,
                        label=f"{lab}, $R_{{{k+1}{k+1}}}$")
    ax.set_xlabel(r"true $\cos i$ (bin centre)"); ax.set_ylabel("R"); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5, ncol=2); fig.tight_layout()
    return fig


def t2b_selection_figure(pairs):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for (lab, block), col, off in zip(pairs, (C1, C2, C3), (-0.0012, 0.0, 0.0012)):
        sel = block["selection"]
        t = np.array([s["threshold"] for s in sel]) + off
        axes[0].errorbar(t, [s["R_sel_diag"] for s in sel], yerr=[s["R_sel_diag_se"] for s in sel], fmt="o-",
                         color=col, ms=5, mec="white", capsize=2, label=lab)
        axes[1].plot(t, [s["R_gamma_diag"] for s in sel], "o-", color=col, ms=5, mec="white", label=lab)
        rt = [0.5 * (s["value"]["R_tot"][0][0] + s["value"]["R_tot"][1][1]) for s in sel]
        axes[1].plot(t, rt, "s--", color=col, ms=4)
        axes[2].plot(t, [s["value"]["frac_by_state"][0] for s in sel], "o-", color=col, ms=5, mec="white", label=lab)
    axes[0].axhline(0, color=MUTED, lw=1)
    axes[0].set_ylabel(r"$R_{\rm sel}$ (diag mean)"); axes[1].set_ylabel(r"$R_\gamma$ (solid), $R_{\rm tot}$ (dashed)")
    axes[2].set_ylabel("selected fraction (g = 0)")
    for a in axes:
        a.set_xlabel(r"cut $\sigma_{\epsilon,i} < t$")
    axes[0].legend(fontsize=7.5); axes[1].legend(fontsize=6.5); axes[2].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def t2b_hist_figure(cubs, xu3sig):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bins = np.linspace(0, 0.12, 49)
    ax.hist(xu3sig, bins=bins, density=True, histtype="step", lw=2, color=C2, label=r"Xu3 test set (100k)")
    for (lab, cub), col in zip(cubs, (C1, C3)):
        ax.hist(cub["sig"][:, 0], bins=bins, density=True, histtype="step", lw=2, color=col, label=lab)
    for t in (0.03, 0.05, 0.075, 0.1):
        ax.axvline(t, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"$\sigma_{\epsilon,i}=\sqrt{\sigma_{g_1}^2+\sigma_{g_2}^2}$"); ax.set_ylabel("density")
    ax.legend(fontsize=8); fig.tight_layout()
    return fig


def resp_weighted_calibration(cub, nboot=500, seed=7):
    """Nested bootstrap: refit R = a + b sigma_k^2 on resampled stencil bases,
    then regress Xu3 ghat_k on R_fit(sigma_k,i) * g_k (|g_k| < 0.02, equal
    weights, intercept free). Slope - 1 is the residual m after R(sigma)
    calibration of the ensemble (weights proportional to R_i)."""
    d = np.load(S / "cache/paper_tests_B_inference/xu3_sigma_summary.npz")
    truth, mean, sd = d["truth"], d["mean"], d["sd"]
    R = cub["R"]; sd0 = cub["sd"][:, 0]; nb = len(R)
    rng = np.random.default_rng(seed)

    def fit(idx):
        x = np.concatenate([sd0[idx, 0] ** 2, sd0[idx, 1] ** 2]); y = np.concatenate([R[idx, 0, 0], R[idx, 1, 1]])
        X = np.column_stack([np.ones_like(x), x]); return np.linalg.lstsq(X, y, rcond=None)[0]

    def slope(ab, k, idx):
        Rf = np.clip(ab[0] + ab[1] * sd[idx, k] ** 2, 0.0, None)
        X = np.column_stack([np.ones(len(idx)), Rf * truth[idx, k]])
        return np.linalg.lstsq(X, mean[idx, k], rcond=None)[0][1]

    out = {}
    for k, name in ((0, "g1"), (1, "g2")):
        low = np.flatnonzero(np.abs(truth[:, k]) < 0.02)
        val = slope(fit(np.arange(nb)), k, low)
        reps = [slope(fit(rng.integers(0, nb, nb)), k, rng.choice(low, len(low))) for _ in range(nboot)]
        out[name] = dict(m_after_resp_weighted_calibration=float(val - 1), se=float(np.std(reps)), n_low=int(len(low)))
    return out


def xu3_consistency(fit, fit_trace):
    d = np.load(S / "cache/paper_tests_B_inference/xu3_sigma_summary.npz")
    truth, mean, sd = d["truth"], d["mean"], d["sd"]
    w = np.full(len(truth), 1.0)
    out = {}
    sig = np.hypot(sd[:, 0], sd[:, 1])
    out["sigma_eps_mean"] = float(sig.mean()); out["sigma_eps_rms"] = float(np.sqrt(np.mean(sig ** 2)))
    out["sigma_eps_pct"] = np.percentile(sig, [5, 16, 50, 84, 95]).tolist()
    for k, name in ((0, "g1"), (1, "g2")):
        raw = component_metrics(truth[:, k], mean[:, k], 0.02, w)
        low = np.abs(truth[:, k]) < 0.02
        Rfit = np.clip(fit["a"] + fit["b"] * sd[:, k] ** 2, 0.05, None)
        Rg = 1 - sd[:, k] ** 2 / SIGMA_PI ** 2
        cal = component_metrics(truth[:, k], mean[:, k] / Rfit, 0.02, w)
        out[name] = dict(raw_low_m=raw["low_m"], raw_low_m_se=raw["low_m_se"], raw_c=raw["low_c"], raw_c_se=raw["low_c_se"],
                         predicted_1pm_from_stencil_fit=float(Rfit[low].mean()),
                         predicted_1pm_gaussian=float(Rg[low].mean()),
                         calibrated_low_m=cal["low_m"], calibrated_low_m_se=cal["low_m_se"],
                         calibrated_low_c=cal["low_c"], calibrated_low_c_se=cal["low_c_se"],
                         frac_R_clipped=float(np.mean(fit["a"] + fit["b"] * sd[:, k] ** 2 < 0.05)))
    return out, sig


def fmt(v, se=None, p=3):
    if se is None:
        return f"{v:.{p}f}"
    return f"{v:.{p}f} ± {se:.{p}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    report = {}
    # ---- R2b
    r2b = {}
    for gal in (6447, 503):
        if (R2B / f"gal{gal}_snr50_seed0.npz").is_file():
            r2b[gal] = r2b_table(gal)
            save(r2b_figure(gal), f"r2b_g1g2_contours_gal{gal}", paper=(gal == 6447))
    if (R2B / "gal6447_hasnr60_seed4.npz").is_file():
        r2b["6447_rhoS"] = r2b_table(6447, tags=("hasnr8", "snrcatalog", "hasnr60"))
        save(rhos_figure(6447), "r2b_g1g2_contours_rhoS_gal6447", paper=True)
    report["R2b"] = r2b
    # ---- corners
    corners = {}
    for gal in (6447, 54792, 5327):
        if (R2B / f"gal{gal}_snr{'catalog'}_seed0.npz").is_file():
            fig, info = corner_figure(gal)
            save(fig, f"corner9d_gal{gal}", paper=True)
            corners[gal] = info
    report["corner9d"] = corners
    # ---- T2b
    t2b = {}
    for label in ("xu3like", "fibrepair", "legacy45255702"):
        f = OUT / f"t2b_{label}.json"
        if f.is_file():
            t2b[label] = json.loads(f.read_text())
    report["T2b"] = t2b
    cub = lambda label, v: dict(np.load(OUT / f"t2b_{label}.{v}.cubes.npz"))
    if "xu3like" in t2b and "fibrepair" in t2b:
        X, F = t2b["xu3like"], t2b["fibrepair"]
        extra = [("proposal stencil, unweighted mean", F["proposal"]["fit_R_sigma"]["per_component"], C3)]
        save(t2b_sigma_figure(X["tf_equal"], cub("xu3like", "tf_equal"), extra), "t2b_R_vs_sigma", paper=True)
        pairs = [("Xu3-like, TF mean", X["tf_equal"]), ("proposal stencil, unweighted", F["proposal"]),
                 ("proposal stencil, TF target", F["tf_target"])]
        save(t2b_cosi_figure(pairs), "t2b_R_vs_cosi")
        save(t2b_selection_figure(pairs), "t2b_selection_response")
        cons, xu3sig = xu3_consistency(X["tf_equal"]["fit_R_sigma"]["per_component"], None)
        cons_prop, _ = xu3_consistency(F["proposal"]["fit_R_sigma"]["per_component"], None)
        report["T2b_xu3_consistency"] = {"fit_from_xu3like_tf_equal": cons, "fit_from_proposal_stencil_unweighted": cons_prop,
                                         "resp_weighted_xu3like": resp_weighted_calibration(cub("xu3like", "tf_equal")),
                                         "resp_weighted_proposal_stencil": resp_weighted_calibration(cub("fibrepair", "proposal"))}
        save(t2b_hist_figure([("Xu3-like stencil bases (1000)", cub("xu3like", "tf_equal")),
                              ("proposal stencil bases (1000)", cub("fibrepair", "tf_equal"))], xu3sig), "t2b_sigma_hist")
    report["regen_probe"] = json.loads((OUT / "regen_probe.json").read_text())
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print("wrote", OUT / "report.json")


if __name__ == "__main__":
    main()
