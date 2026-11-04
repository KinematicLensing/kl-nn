#!/usr/bin/env python3
"""Paper tests, Group A (shear): R-raw, T1, T3, T4, T5, R6, T2a, T7.

Input: the per-galaxy table written by ``a_shear_extract.py`` (TF-weighted
posterior mean / variance / covariance / quantiles of g1, g2 for all 10^5 Xu3
galaxies, fibrepair model). Output: ``results.json`` and PNG/PDF figures.

Conventions (PAPER_TESTS_PLAN.md):
- estimator: TF-weighted posterior mean of (g1, g2);
- sigma_eps,i = sqrt(sigma_g1,i^2 + sigma_g2,i^2) from TF-weighted 1D std (Xu24);
- bias fit: ghat - g = m g + c on |g| < 0.02, per component, image and galaxy
  frame, via ``shear_bias_report.component_metrics`` / ``fit_design``;
- "raw": equal galaxy weights, full sample, no cuts;
- "weighted": ``shear_bias_report.compose_shape_noise_regularized_weights``
  (1/(sigma_gal^2 + sigma_floor^2) with sigma_gal = sqrt((V1+V2)/2), floor =
  mean sigma_gal of the (retained) sample), unchanged from the existing code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import truncnorm  # noqa: E402,F401

DIAG_DIR = Path(__file__).resolve().parents[1]
if str(DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(DIAG_DIR))

import shear_bias_report as sbr  # noqa: E402
from utils import img_to_gal_axis  # noqa: E402

LOW_G = 0.02
SIGMA_PI = 0.2 / np.sqrt(12.0)
CUTS = [0.03, 0.04, 0.05, 0.075, 0.1]
COMPONENTS = ["g1", "g2", "g+", "gx"]
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
GREY = "#52514e"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 150,
})


# ----------------------------------------------------------------- data
def load(path: Path) -> dict:
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    t = d["truth"]
    fn = list(d["feature_names"])
    g1, g2, th = t[:, 0], t[:, 1], t[:, fn.index("theta_int")]
    v1, v2, c12 = d["tf_var"][:, 0], d["tf_var"][:, 1], d["tf_cov12"]
    c, s = np.cos(2 * th), np.sin(2 * th)
    # galaxy-frame posterior variances: g+ = g1 c - g2 s, gx = g1 s + g2 c
    vp = c * c * v1 + s * s * v2 - 2 * c * s * c12
    vx = s * s * v1 + c * c * v2 + 2 * c * s * c12
    tp, tx = img_to_gal_axis(g1, g2, th)
    ep, ex = img_to_gal_axis(d["tf_mean"][:, 0], d["tf_mean"][:, 1], th)
    qg1, qg2 = d["tf_q_g1"], d["tf_q_g2"]
    lev = list(np.round(d["quantile_levels"], 3))
    i16, i84 = lev.index(0.16), lev.index(0.84)
    h1 = 0.5 * (qg1[:, i84] - qg1[:, i16])
    h2 = 0.5 * (qg2[:, i84] - qg2[:, i16])
    return dict(
        n=len(g1), truth=t, fn=fn, cosi=t[:, fn.index("cosi")], theta=th,
        true={"g1": g1, "g2": g2, "g+": tp, "gx": tx},
        est={"g1": d["tf_mean"][:, 0], "g2": d["tf_mean"][:, 1], "g+": ep, "gx": ex},
        var={"g1": v1, "g2": v2, "g+": vp, "gx": vx},
        v1=v1, v2=v2, h1=h1, h2=h2,
        sig=np.sqrt(v1 + v2),            # Xu24 sigma_eps,i (std convention)
        sig_q=np.sqrt(h1 ** 2 + h2 ** 2),  # 16-84 half-width convention
        image_snr=d["image_snr"], halpha_snr=d["central_halpha_snr"],
        rmag=d["rmag_true"], ess_frac=d["posterior_tf_ess_fraction"],
        raw=d,
    )


# ----------------------------------------------------------------- fits
def fit_component(D, comp, w, low=LOW_G):
    r = sbr.component_metrics(D["true"][comp], D["est"][comp], low, w)
    return r


def m_c_fast(t, e, w, low=LOW_G):
    """Low-|g| linear fit coefficients only (for bootstrap)."""
    sel = np.abs(t) < low
    coef, _, _ = sbr.fit_design(e[sel] - t[sel], np.column_stack((np.ones(sel.sum()), t[sel])), w[sel])
    return coef[1], coef[0]


def weighted_sigma_stats(sig, w):
    w = np.asarray(w, float)
    m = w > 0
    ww = w[m] / w[m].sum()
    return float(np.sum(ww * sig[m])), float(np.sqrt(np.sum(ww * sig[m] ** 2)))


def estimator_weights(D, kind, mask, base=None, var=None):
    """kind: 'raw' or 'weighted'. mask: retained galaxies. base: bootstrap counts."""
    base = np.ones(D["n"]) if base is None else base
    b = base * mask
    if kind == "raw":
        return b / b.sum(), None
    v1, v2 = (D["v1"], D["v2"]) if var is None else var
    w, diag = sbr.compose_shape_noise_regularized_weights(b / b.sum(), v1, v2)
    return w, diag


def full_metrics(D, w):
    out = {}
    for comp in COMPONENTS:
        r = fit_component(D, comp, w)
        out[comp] = {k: r[k] for k in ("low_m", "low_m_se", "low_c", "low_c_se", "c", "c_se",
                                         "cubic_m", "cubic_m_se", "cubic_q", "cubic_q_se",
                                         "n_low", "ess_low", "n", "ess")}
    return out


# ------------------------------------------------------------ uniform toy
def truncnorm_moments(loc, scale, lo, hi):
    """Closed-form mean and variance of N(loc, scale) truncated to [lo, hi]."""
    a, b = (lo - loc) / scale, (hi - loc) / scale
    Z = ndtr(b) - ndtr(a)
    pa, pb = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi), np.exp(-0.5 * b * b) / np.sqrt(2 * np.pi)
    r = (pa - pb) / Z
    mean = loc + scale * r
    var = scale ** 2 * (1 + (a * pa - b * pb) / Z - r * r)
    return mean, var


def uniform_prior_toy(n_grid=160, n_sim=400_000, seed=1):
    """1D toy: g ~ U(-0.02, 0.02) (the fit window), gobs = g + N(0, s_lik),
    posterior under U(-0.1, 0.1) prior x N(gobs, s_lik) is a truncated normal.
    Returns (s_lik grid, mean posterior std, low-|g| slope m)."""
    rng = np.random.default_rng(seed)
    s_grid = np.geomspace(1e-3, 1.0, n_grid)
    g = rng.uniform(-LOW_G, LOW_G, n_sim)
    eps = rng.standard_normal(n_sim)
    mstd, mm = [], []
    for s in s_grid:
        go = g + s * eps
        mu, var = truncnorm_moments(go, s, -0.1, 0.1)
        mstd.append(float(np.mean(np.sqrt(var))))
        x = np.column_stack((np.ones_like(g), g))
        coef = np.linalg.lstsq(x, mu - g, rcond=None)[0]
        mm.append(float(coef[1]))
    return s_grid, np.asarray(mstd), np.asarray(mm)


def toy_m_from_post_std(sig_post, toy):
    _, mstd, mm = toy
    order = np.argsort(mstd)
    x = np.clip(sig_post, mstd.min(), mstd.max())
    return np.interp(x, mstd[order], mm[order])


def shrink_predictions(D, sel, comp, toy):
    v = D["var"][comp][sel]
    s = np.sqrt(v)
    return {
        "plan_formula": float(-np.mean(v / (v + SIGMA_PI ** 2))),
        "gauss_posterior": float(-np.mean(v) / SIGMA_PI ** 2),
        "uniform_toy": float(np.mean(toy_m_from_post_std(s, toy))),
    }


# ----------------------------------------------------------------- tests
def binned(x, y, edges):
    idx = np.digitize(x, edges) - 1
    cen, mu, se, cnt = [], [], [], []
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() < 5:
            continue
        cen.append(np.mean(x[m])); mu.append(np.mean(y[m]))
        se.append(np.std(y[m], ddof=1) / np.sqrt(m.sum())); cnt.append(int(m.sum()))
    return np.array(cen), np.array(mu), np.array(se), np.array(cnt)


def slope_fit(y, x_cols, w=None):
    X = np.column_stack([np.ones(len(y))] + list(x_cols))
    w = np.ones(len(y)) if w is None else w
    coef, se, _ = sbr.fit_design(y, X, w)
    return coef.tolist(), se.tolist()


def gbm_predicted_log_sigma(D, seed=0):
    """Cross-fitted (5-fold) prediction of log sigma_eps,i from shear-independent truths."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    t, fn = D["truth"], D["fn"]
    cols = [D["cosi"], np.log(D["image_snr"]), np.log(D["halpha_snr"]), D["rmag"],
            t[:, fn.index("vcirc")], t[:, fn.index("hlr")], t[:, fn.index("rscale")],
            np.log10(t[:, fn.index("halpha_flux_true")]), t[:, fn.index("v0")],
            np.cos(2 * D["theta"]), np.sin(2 * D["theta"])]
    names = ["cosi", "log image_snr", "log halpha_snr", "rmag", "vcirc", "hlr", "rscale",
             "log10 flux", "v0", "cos2theta", "sin2theta"]
    X = np.column_stack(cols)
    y = np.log(D["sig"])
    pred = np.empty_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=seed)
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict(X[te])
    r2 = 1 - np.var(y - pred) / np.var(y)
    return pred, float(r2), names


def test_rraw(D, out, figdir, paperdir):
    w = np.full(D["n"], 1.0 / D["n"])
    res = full_metrics(D, w)
    sig = D["sig"]
    res["sigma_eps"] = {
        "mean": float(sig.mean()), "rms": float(np.sqrt(np.mean(sig ** 2))),
        "median": float(np.median(sig)),
        "percentiles_5_16_50_84_95": np.percentile(sig, [5, 16, 50, 84, 95]).tolist(),
        "per_component_mean_sigma_g1": float(np.sqrt(D["v1"]).mean()),
        "per_component_mean_sigma_g2": float(np.sqrt(D["v2"]).mean()),
        "prior_std_sigma_pi": SIGMA_PI,
        "prior_sigma_eps_equiv_sqrt2_sigma_pi": float(np.sqrt(2) * SIGMA_PI),
        "frac_sigma_eps_gt_0.075": float(np.mean(sig > 0.075)),
    }
    # R5 figure: binned ghat - g vs g
    edges = np.linspace(-0.1, 0.1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    binned_out = {}
    for ax, comps, title in ((axes[0], ("g1", "g2"), "image frame"), (axes[1], ("g+", "gx"), "galaxy frame")):
        for comp, col, mk in zip(comps, (C1, C2), ("o", "s")):
            t, e = D["true"][comp], D["est"][comp]
            ed = edges if comp in ("g1", "g2") else np.linspace(-0.12, 0.12, 25)
            x, mu, se, cnt = binned(t, e - t, ed)
            binned_out[comp] = dict(x=x.tolist(), mean_resid=mu.tolist(), se=se.tolist(), n=cnt.tolist())
            lab = {"g1": "$g_1$", "g2": "$g_2$", "g+": "$g_+$", "gx": r"$g_\times$"}[comp]
            r = res[comp]
            ax.errorbar(x, mu, se, fmt=mk, ms=4, color=col, lw=1, capsize=0,
                        label=f"{lab}: m = {r['low_m']:+.3f} ± {r['low_m_se']:.3f}")
            xx = np.linspace(ed[0], ed[-1], 50)
            ax.plot(xx, r["low_c"] + r["low_m"] * xx, color=col, lw=1, alpha=0.7)
        ax.axvspan(-LOW_G, LOW_G, color="#f0f0f0", zorder=0)
        ax.axhline(0, color=GREY, lw=0.8)
        ax.set_xlabel("true shear component")
        ax.set_title(f"Raw posterior means, {title}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel(r"$\hat g - g$ (bin mean ± s.e.)")
    fig.tight_layout()
    save(fig, figdir, "Rraw_binned_residual", paperdir, "A_R5_raw_bias")
    res["binned"] = binned_out
    # sigma histogram
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(sig, bins=np.linspace(0, 0.1, 81), color=C1, edgecolor="white", linewidth=0.4)
    for v, lab, ls in ((sig.mean(), "mean", "-"), (np.sqrt(np.mean(sig ** 2)), "RMS", "--"),
                       (np.sqrt(2) * SIGMA_PI, r"$\sqrt{2}\,\sigma_\pi$ (prior)", ":")):
        ax.axvline(v, color=GREY, ls=ls, lw=1.1, label=f"{lab} = {v:.4f}")
    ax.set_xlabel(r"per-galaxy $\sigma_{\epsilon,i}=\sqrt{\sigma_{g_1}^2+\sigma_{g_2}^2}$")
    ax.set_ylabel("galaxies")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, figdir, "Rraw_sigma_hist", paperdir, "A_R5_sigma_hist")
    out["R_raw"] = res
    return w


def test_t1(D, out, figdir, logsig_pred, gbm_r2, gbm_names):
    sig = D["sig"]
    absg = np.hypot(D["true"]["g1"], D["true"]["g2"])
    gp, gx = D["true"]["g+"], D["true"]["gx"]
    res = {"gbm_r2_log_sigma": gbm_r2, "gbm_features": gbm_names}
    resid = np.log(sig) - logsig_pred
    for name, x in (("|g|", absg), ("g+", gp), ("gx", gx)):
        c, s = slope_fit(sig, [x])
        cl, sl = slope_fit(np.log(sig), [x])
        cr, sr = slope_fit(resid, [x])
        c2, s2 = slope_fit(resid, [x, x ** 2])
        res[name] = {
            "dsigma_dg": c[1], "dsigma_dg_se": s[1], "intercept": c[0],
            "dlnsigma_dg": cl[1], "dlnsigma_dg_se": sl[1],
            "dlnsigma_dg_residualized": cr[1], "dlnsigma_dg_residualized_se": sr[1],
            "quad_residualized": {"lin": c2[1], "lin_se": s2[1], "quad": c2[2], "quad_se": s2[2]},
        }
    # galaxy-frame per-component posterior width vs g+
    for comp in ("g+", "gx"):
        sc = np.sqrt(D["var"][comp])
        c, s = slope_fit(np.log(sc), [gp, gx])
        res[f"dln_sigma_{comp}"] = {"vs_g+": c[1], "vs_g+_se": s[1], "vs_gx": c[2], "vs_gx_se": s[2]}
    # by cos i
    cedges = [0, 0.25, 0.5, 0.75, 1.0]
    res["by_cosi"] = []
    for lo, hi in zip(cedges[:-1], cedges[1:]):
        m = (D["cosi"] >= lo) & (D["cosi"] < hi + (1e-9 if hi == 1 else 0))
        row = {"cosi_lo": lo, "cosi_hi": hi, "n": int(m.sum()), "mean_sigma": float(sig[m].mean())}
        for name, x in (("|g|", absg), ("g+", gp), ("gx", gx)):
            c, s = slope_fit(np.log(sig[m]), [x[m]])
            cr, sr = slope_fit(resid[m], [x[m]])
            row[name] = {"dlnsigma_dg": c[1], "se": s[1], "residualized": cr[1], "residualized_se": sr[1]}
        res["by_cosi"].append(row)
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (name, x, ed) in zip(axes, (("|g|", absg, np.linspace(0, 0.14, 15)),
                                         ("g+", gp, np.linspace(-0.12, 0.12, 17)),
                                         ("gx", gx, np.linspace(-0.12, 0.12, 17)))):
        for (lo, hi), col, mk in zip(zip(cedges[:-1], cedges[1:]), (C1, C2, C3, C5), "osD^"):
            m = (D["cosi"] >= lo) & (D["cosi"] <= hi)
            xc, mu, se, _ = binned(x[m], resid[m], ed)
            ax.errorbar(xc, mu, se, fmt=mk + "-", ms=3.5, lw=1, color=col, label=f"cos i ∈ [{lo},{hi}]")
        xc, mu, se, _ = binned(x, resid, ed)
        ax.errorbar(xc, mu, se, fmt="k-", lw=1.8, label="all")
        ax.axhline(0, color=GREY, lw=0.7)
        lab = {"|g|": "true $|g|$", "g+": "true $g_+$", "gx": r"true $g_\times$"}[name]
        r = res[name]
        ax.set_title(f"slope d ln σ/dg = {r['dlnsigma_dg_residualized']:+.3f} ± {r['dlnsigma_dg_residualized_se']:.3f}", fontsize=9)
        ax.set_xlabel(lab)
    axes[0].set_ylabel(r"$\ln\sigma_{\epsilon,i}-\widehat{\ln\sigma}$(shear-free truths)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, figdir, "T1_sigma_vs_shear")
    # also raw (not residualized) sigma vs g+ panel
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for name, x, col in (("g+", gp, C1), ("gx", gx, C2)):
        xc, mu, se, _ = binned(x, sig, np.linspace(-0.12, 0.12, 17))
        ax.errorbar(xc, mu, se, fmt="o-", ms=3.5, lw=1, color=col, label=name)
    ax.set_xlabel("true galaxy-frame shear"); ax.set_ylabel(r"mean $\sigma_{\epsilon,i}$")
    ax.legend(fontsize=8); fig.tight_layout()
    save(fig, figdir, "T1_sigma_raw_vs_gplus")
    out["T1"] = res


def test_t3(D, out, figdir, paperdir, toy):
    sig = D["sig"]
    nb = 15
    edges = np.quantile(sig, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-12
    rows = []
    for i in range(nb):
        m = (sig >= edges[i]) & (sig < edges[i + 1])
        w = m / m.sum()
        row = {"bin": i, "sig_lo": float(edges[i]), "sig_hi": float(edges[i + 1]),
               "n": int(m.sum()), "mean_sigma_eps": float(sig[m].mean())}
        for comp in COMPONENTS:
            r = fit_component(D, comp, w.astype(float))
            row[comp] = {"m": r["low_m"], "m_se": r["low_m_se"], "c": r["low_c"], "n_low": r["n_low"],
                         **shrink_predictions(D, m, comp, toy)}
        rows.append(row)
    # global predicted vs measured
    allm = np.ones(D["n"], bool)
    glob = {comp: shrink_predictions(D, allm, comp, toy) for comp in COMPONENTS}
    out["T3"] = {"sigma_pi": SIGMA_PI, "bins": rows, "full_sample_predictions": glob,
                 "toy_grid": {"s_lik": toy[0].tolist(), "mean_post_std": toy[1].tolist(), "m": toy[2].tolist()}}
    # chi2 of each prediction vs measured (image+galaxy)
    chi = {}
    for key in ("plan_formula", "gauss_posterior", "uniform_toy"):
        for comp in COMPONENTS:
            d = np.array([(r[comp]["m"] - r[comp][key]) / r[comp]["m_se"] for r in rows])
            chi[f"{key}:{comp}"] = {"chi2": float(np.sum(d ** 2)), "ndof": nb,
                                    "mean_diff": float(np.mean([r[comp]["m"] - r[comp][key] for r in rows]))}
    out["T3"]["chi2"] = chi
    # figure
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, comps in ((axes[0], ("g1", "g2")), (axes[1], ("g+", "gx"))):
        x = np.array([r["mean_sigma_eps"] for r in rows])
        for comp, col, mk, dx in zip(comps, (C1, C2), ("o", "s"), (-0.0004, 0.0004)):
            lab = {"g1": "$m_1$", "g2": "$m_2$", "g+": "$m_+$", "gx": r"$m_\times$"}[comp]
            ax.errorbar(x + dx, [r[comp]["m"] for r in rows], [r[comp]["m_se"] for r in rows],
                        fmt=mk, ms=4.5, color=col, lw=1, label=f"measured {lab}")
        comp = comps[0]
        pa = np.array([np.mean([r[c]["plan_formula"] for c in comps]) for r in rows])
        pb = np.array([np.mean([r[c]["gauss_posterior"] for c in comps]) for r in rows])
        pc = np.array([np.mean([r[c]["uniform_toy"] for c in comps]) for r in rows])
        ax.plot(x, pa, color=GREY, ls=":", label=r"$-\langle\sigma^2/(\sigma^2+\sigma_\pi^2)\rangle$")
        ax.plot(x, pb, color=GREY, ls="--", label=r"$-\langle\sigma^2\rangle/\sigma_\pi^2$ (Gaussian, posterior σ)")
        ax.plot(x, pc, color="k", ls="-", lw=1.4, label="uniform-prior toy")
        ax.set_xlabel(r"bin mean $\sigma_{\epsilon,i}$")
        ax.set_title("image frame" if comp == "g1" else "galaxy frame", fontsize=10)
        ax.axhline(0, color=GREY, lw=0.6)
    axes[0].set_ylabel("multiplicative bias m (|g| < 0.02)")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, figdir, "T3_shrinkage", paperdir, "A_D1_shrinkage")


def test_t4(D, out, figdir):
    w = np.full(D["n"], 1.0 / D["n"])
    res = {}
    windows = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15]
    for comp in COMPONENTS:
        t, e = D["true"][comp], D["est"][comp]
        row = {"linear_window": []}
        for lw_ in windows:
            sel = np.abs(t) < lw_
            coef, se, _ = sbr.fit_design(e[sel] - t[sel], np.column_stack((np.ones(sel.sum()), t[sel])), w[sel])
            row["linear_window"].append({"max_abs_g": lw_, "n": int(sel.sum()), "m": coef[1], "m_se": se[1],
                                         "c": coef[0], "c_se": se[0]})
        r = fit_component(D, comp, w)
        row["cubic_full"] = {"m": r["cubic_m"], "m_se": r["cubic_m_se"], "q": r["cubic_q"], "q_se": r["cubic_q_se"]}
        # cubic on full range for ghat vs g with explicit g^2 too (check)
        coef, se, _ = sbr.fit_design(e - t, np.column_stack((np.ones_like(t), t, t ** 2, t ** 3)), w)
        row["cubic_full_with_quadratic"] = {"c": coef[0], "m": coef[1], "m_se": se[1], "q2": coef[2],
                                            "q2_se": se[2], "q3": coef[3], "q3_se": se[3]}
        res[comp] = row
    out["T4"] = res
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax = axes[0]
    for comp, col, mk in (("g1", C1, "o"), ("g2", C2, "s")):
        t, e = D["true"][comp], D["est"][comp]
        x, mu, se, _ = binned(t, e, np.linspace(-0.1, 0.1, 41))
        ax.errorbar(x, mu, se, fmt=mk, ms=3, color=col, lw=1, label=comp)
        r = res[comp]
        xx = np.linspace(-0.1, 0.1, 100)
        lw2 = r["linear_window"][1]
        ax.plot(xx, xx + lw2["c"] + lw2["m"] * xx, color=col, lw=1, alpha=0.8)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], color=GREY, ls=":", lw=1, label="ĝ = g")
    ax.set_xlabel("true g"); ax.set_ylabel("binned mean ĝ"); ax.legend(fontsize=8)
    ax.set_title("ĝ vs g over the prior range; lines: |g|<0.02 fit", fontsize=9)
    ax = axes[1]
    for comp, col, mk in zip(COMPONENTS, (C1, C2, C3, C5), "os^D"):
        lw_ = res[comp]["linear_window"]
        ax.errorbar([r["max_abs_g"] for r in lw_], [r["m"] for r in lw_], [r["m_se"] for r in lw_],
                    fmt=mk + "-", ms=4, lw=1, color=col, label=comp)
    ax.set_xscale("log"); ax.set_xlabel("fit window |g| < g_max"); ax.set_ylabel("m (linear fit)")
    ax.legend(fontsize=8); ax.set_title("linear-fit m vs window", fontsize=9)
    fig.tight_layout()
    save(fig, figdir, "T4_linearity")


def estimator_list(D, sigvec=None, var=None):
    sigvec = D["sig"] if sigvec is None else sigvec
    ests = [("raw", "raw", np.ones(D["n"], bool)), ("weighted", "weighted", np.ones(D["n"], bool))]
    for t in CUTS:
        ests.append((f"raw, σ_ε,i<{t}", "raw", sigvec < t))
        ests.append((f"weighted, σ_ε,i<{t}", "weighted", sigvec < t))
    return ests


def test_t5(D, out, n_boot, seed=123):
    rng = np.random.default_rng(seed)
    ests = estimator_list(D)
    point = {}
    for name, kind, mask in ests:
        w, diag = estimator_weights(D, kind, mask)
        met = {}
        for comp in COMPONENTS:
            r = fit_component(D, comp, w)
            met[comp] = {"m": r["low_m"], "m_fit_se": r["low_m_se"], "c": r["low_c"], "c_fit_se": r["low_c_se"],
                         "n_low": r["n_low"], "ess_low": r["ess_low"]}
        smean, srms = weighted_sigma_stats(D["sig"], w)
        point[name] = {"kind": kind, "n_retained": int(mask.sum()), "frac_retained": float(mask.mean()),
                       "sigma_eps_mean": smean, "sigma_eps_rms": srms,
                       "sigma_eps_mean_unweighted": float(D["sig"][mask].mean()),
                       "floor_sigma_gal": None if diag is None else diag["shape_noise"],
                       "components": met}
    boots = {name: {comp: {"m": [], "c": []} for comp in COMPONENTS} | {"smean": [], "srms": []}
             for name, _, _ in ests}
    for b in range(n_boot):
        counts = rng.multinomial(D["n"], np.full(D["n"], 1.0 / D["n"])).astype(float)
        for name, kind, mask in ests:
            w, _ = estimator_weights(D, kind, mask, base=counts)
            for comp in COMPONENTS:
                m, c = m_c_fast(D["true"][comp], D["est"][comp], w)
                boots[name][comp]["m"].append(m); boots[name][comp]["c"].append(c)
            sm, sr = weighted_sigma_stats(D["sig"], w)
            boots[name]["smean"].append(sm); boots[name]["srms"].append(sr)
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    for name, _, _ in ests:
        p = point[name]
        for comp in COMPONENTS:
            bm, bc = np.array(boots[name][comp]["m"]), np.array(boots[name][comp]["c"])
            p["components"][comp].update(m_boot_se=float(bm.std(ddof=1)), c_boot_se=float(bc.std(ddof=1)),
                                         m_boot_p16_p84=np.percentile(bm, [16, 84]).tolist(),
                                         m_boot_mean=float(bm.mean()))
        p["sigma_eps_mean_boot_se"] = float(np.std(boots[name]["smean"], ddof=1))
        p["sigma_eps_rms_boot_se"] = float(np.std(boots[name]["srms"], ddof=1))
    out["T5"] = {"n_boot": n_boot, "seed": seed, "estimators": point}


def r6_curve(D, sigvec, var, thresholds):
    rows = []
    for t in thresholds:
        mask = sigvec < t
        if mask.sum() < 200:
            continue
        row = {"threshold": float(t), "n_retained": int(mask.sum()), "frac_retained": float(mask.mean())}
        for kind in ("raw", "weighted"):
            w, diag = estimator_weights(D, kind, mask, var=var)
            r = {}
            for comp in ("g1", "g2"):
                f = fit_component(D, comp, w)
                r[comp] = {"m": f["low_m"], "m_se": f["low_m_se"], "c": f["low_c"], "c_se": f["low_c_se"],
                           "n_low": f["n_low"]}
            sm, sr = weighted_sigma_stats(D["sig"], w)
            sqm, sqr = weighted_sigma_stats(D["sig_q"], w)
            r.update(sigma_eps_mean=sm, sigma_eps_rms=sr, sigma_eps_q_mean=sqm, sigma_eps_q_rms=sqr,
                     ess=float(1 / np.sum((w / w.sum()) ** 2)))
            row[kind] = r
        rows.append(row)
    return rows


def plot_r6(rows, fname, figdir, paperdir=None, pname=None, xlabel=r"threshold on $\sigma_{\epsilon,i}$"):
    nocut = rows[-1]
    rows = [r for r in rows if r["threshold"] <= 0.09]
    x = np.array([r["threshold"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    for kind, ls in (("raw", "-"), ("weighted", "--")):
        for comp, col, mk in (("g1", C1, "o"), ("g2", C2, "s")):
            m = np.array([r[kind][comp]["m"] for r in rows]); se = np.array([r[kind][comp]["m_se"] for r in rows])
            ax.plot(x, m, ls=ls, marker=mk, ms=3, color=col, label=f"{comp}, {kind}")
            ax.fill_between(x, m - se, m + se, color=col, alpha=0.12, lw=0)
            ax.plot([x[-1] + 0.002], [nocut[kind][comp]["m"]], marker=mk, ms=5, color=col,
                    mfc="none" if kind == "weighted" else col, ls="none")
    ax.axhline(0, color=GREY, lw=0.7)
    ax.annotate("no cut", (x[-1] + 0.002, ax.get_ylim()[0]), fontsize=7, ha="center", va="bottom")
    ax.set_xlabel(xlabel); ax.set_ylabel("m (|g| < 0.02 fit)"); ax.legend(fontsize=7.5)
    ax = axes[1]
    for kind, ls in (("raw", "-"), ("weighted", "--")):
        ax.plot(x, [r[kind]["sigma_eps_mean"] for r in rows], ls=ls, color=C3, label=f"mean, {kind}")
        ax.plot(x, [r[kind]["sigma_eps_rms"] for r in rows], ls=ls, color=C5, label=f"RMS, {kind}")
    ax.set_xlabel(xlabel); ax.set_ylabel(r"ensemble $\sigma_\epsilon$ of retained"); ax.legend(fontsize=7.5)
    ax = axes[2]
    ax.plot(x, [r["frac_retained"] for r in rows], color="k", label="retained fraction")
    ax.set_xlabel(xlabel); ax.set_ylabel("retained fraction")
    ax2 = ax.twinx()
    ax2.plot(x, [r["raw"]["g1"]["n_low"] for r in rows], color=GREY, ls=":", label="N(|g1|<0.02)")
    ax2.set_ylabel("low-|g1| galaxy count", color=GREY)
    ax2.spines["right"].set_visible(True); ax2.grid(False)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, figdir, fname, paperdir, pname)


def test_r6(D, out, figdir, paperdir):
    thr = np.round(np.concatenate([np.arange(0.015, 0.09001, 0.0025), [0.1, 1.0]]), 4)
    rows = r6_curve(D, D["sig"], None, thr)
    out["R6"] = {"thresholds": thr.tolist(), "rows": rows}
    plot_r6(rows, "R6_width_cut", figdir, paperdir, "A_R6_width_cut")


def matched_fraction_cut(score, frac):
    k = int(round(frac * len(score)))
    order = np.argsort(score, kind="stable")
    m = np.zeros(len(score), bool)
    m[order[:k]] = True
    return m


def binned_sigma_proxy(x, sig, nbins=50):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    means = np.array([sig[idx == i].mean() for i in range(len(edges) - 1)])
    return means[idx]


def test_t2a(D, out, figdir, logsig_pred, toy):
    sig = D["sig"]
    scores = {
        "σ_ε,i (observed)": sig,
        "true cos i (σ-ranked)": binned_sigma_proxy(D["cosi"], sig),
        "ρ_I image SNR (σ-ranked)": binned_sigma_proxy(D["image_snr"], sig),
        "ρ_S Hα SNR (σ-ranked)": binned_sigma_proxy(D["halpha_snr"], sig),
        "GBM σ from shear-free truths": logsig_pred,
    }
    # direction sanity: correlation between each proxy and sigma
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    res = {"fractions": fracs, "curves": {}, "proxy_spearman_with_sigma": {}}
    from scipy.stats import spearmanr
    for name, sc in scores.items():
        res["proxy_spearman_with_sigma"][name] = float(spearmanr(sc, sig).correlation)
        rows = []
        for f in fracs:
            m = matched_fraction_cut(sc, f)
            w = m / m.sum()
            row = {"frac": f, "n": int(m.sum()), "mean_sigma_eps": float(sig[m].mean()),
                   "mean_sigma2_per_comp": float(np.mean(0.5 * (D["v1"][m] + D["v2"][m])))}
            for comp in ("g1", "g2"):
                r = fit_component(D, comp, w.astype(float))
                row[comp] = {"m": r["low_m"], "m_se": r["low_m_se"], "n_low": r["n_low"],
                             **shrink_predictions(D, m, comp, toy)}
            row["m_avg"] = 0.5 * (row["g1"]["m"] + row["g2"]["m"])
            row["m_avg_se"] = 0.5 * np.hypot(row["g1"]["m_se"], row["g2"]["m_se"])
            row["toy_avg"] = 0.5 * (row["g1"]["uniform_toy"] + row["g2"]["uniform_toy"])
            row["excess_vs_toy"] = row["m_avg"] - row["toy_avg"]
            rows.append(row)
        res["curves"][name] = rows
    # compare sigma-cut vs GBM-cut and truth cuts: m at matched mean sigma, via
    # linear interpolation of the truth-cut curve m(mean sigma)
    ref = res["curves"]["σ_ε,i (observed)"]
    comp_rows = []
    for name, rows in res["curves"].items():
        if name.startswith("σ_ε"):
            continue
        xs = np.array([r["mean_sigma_eps"] for r in rows]); ms = np.array([r["m_avg"] for r in rows])
        es = np.array([r["m_avg_se"] for r in rows])
        o = np.argsort(xs)
        for r in ref:
            x = r["mean_sigma_eps"]
            if xs.min() <= x <= xs.max():
                mi = float(np.interp(x, xs[o], ms[o])); ei = float(np.interp(x, xs[o], es[o]))
                comp_rows.append({"proxy": name, "frac_sigma_cut": r["frac"], "mean_sigma_eps": x,
                                  "m_sigma_cut": r["m_avg"], "m_sigma_cut_se": r["m_avg_se"],
                                  "m_proxy_interp": mi, "m_proxy_interp_se": ei,
                                  "diff": r["m_avg"] - mi})
    res["matched_sigma_comparison"] = comp_rows
    out["T2a"] = res
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (name, rows), col, mk in zip(res["curves"].items(), (C1, C2, C3, C4, C5), "osD^v"):
        x = [r["mean_sigma_eps"] for r in rows]
        axes[0].errorbar(x, [r["m_avg"] for r in rows], [r["m_avg_se"] for r in rows], fmt=mk + "-",
                         ms=4, lw=1.1, color=col, label=name)
        axes[1].errorbar([r["frac"] for r in rows], [r["excess_vs_toy"] for r in rows],
                         [r["m_avg_se"] for r in rows], fmt=mk + "-", ms=4, lw=1.1, color=col, label=name)
    axes[0].set_xlabel(r"mean $\sigma_{\epsilon,i}$ of retained sample")
    axes[0].set_ylabel(r"$(m_1+m_2)/2$")
    axes[0].legend(fontsize=7.5)
    axes[1].axhline(0, color=GREY, lw=0.7)
    axes[1].set_xlabel("retained fraction")
    axes[1].set_ylabel("m − uniform-toy prediction")
    fig.tight_layout()
    save(fig, figdir, "T2a_selection")


def test_t7(D, out, figdir):
    ratio1 = D["h1"] / np.sqrt(D["v1"]); ratio2 = D["h2"] / np.sqrt(D["v2"])
    res = {
        "sigma_eps_std": {"mean": float(D["sig"].mean()), "rms": float(np.sqrt(np.mean(D["sig"] ** 2)))},
        "sigma_eps_halfwidth": {"mean": float(D["sig_q"].mean()), "rms": float(np.sqrt(np.mean(D["sig_q"] ** 2)))},
        "halfwidth_over_std_g1_percentiles_5_16_50_84_95": np.percentile(ratio1, [5, 16, 50, 84, 95]).tolist(),
        "halfwidth_over_std_g2_percentiles_5_16_50_84_95": np.percentile(ratio2, [5, 16, 50, 84, 95]).tolist(),
        "spearman_sig_vs_sigq": None,
    }
    from scipy.stats import spearmanr
    res["spearman_sig_vs_sigq"] = float(spearmanr(D["sig"], D["sig_q"]).correlation)
    thr = np.round(np.concatenate([np.arange(0.015, 0.09001, 0.0025), [0.1, 1.0]]), 4)
    rows_q = r6_curve(D, D["sig_q"], (D["h1"] ** 2, D["h2"] ** 2), thr)
    res["R6_halfwidth_rows"] = rows_q
    # matched retained fraction comparison
    cmp = []
    for f in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        row = {"frac": f}
        for lab, sc, var in (("std", D["sig"], None), ("halfwidth", D["sig_q"], (D["h1"] ** 2, D["h2"] ** 2))):
            m = matched_fraction_cut(sc, f)
            for kind in ("raw", "weighted"):
                w, _ = estimator_weights(D, kind, m, var=var)
                r1 = fit_component(D, "g1", w); r2 = fit_component(D, "g2", w)
                row[f"{lab}:{kind}"] = {"m1": r1["low_m"], "m1_se": r1["low_m_se"],
                                        "m2": r2["low_m"], "m2_se": r2["low_m_se"],
                                        "threshold": float(np.max(sc[m]))}
        row["overlap_fraction"] = float(np.mean(matched_fraction_cut(D["sig"], f)[matched_fraction_cut(D["sig_q"], f)])) if f < 1 else 1.0
        cmp.append(row)
    res["matched_fraction"] = cmp
    out["T7"] = res
    plot_r6(rows_q, "T7_R6_halfwidth", figdir, xlabel=r"threshold on half-width $\sigma^{(q)}_{\epsilon,i}$")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    axes[0].hist(ratio1, bins=np.linspace(0.5, 1.5, 81), color=C1, alpha=0.7, label="g1")
    axes[0].hist(ratio2, bins=np.linspace(0.5, 1.5, 81), color=C2, alpha=0.5, label="g2")
    axes[0].axvline(1, color=GREY, lw=0.8); axes[0].set_xlabel("(q84 − q16)/2 ÷ std")
    axes[0].legend(fontsize=8)
    ax = axes[1]
    fr = [r["frac"] for r in cmp]
    for lab, col in (("std", C1), ("halfwidth", C2)):
        for kind, ls in (("raw", "-"), ("weighted", "--")):
            mm = [0.5 * (r[f"{lab}:{kind}"]["m1"] + r[f"{lab}:{kind}"]["m2"]) for r in cmp]
            ax.plot(fr, mm, ls=ls, marker="o", ms=3, color=col, label=f"{lab}, {kind}")
    ax.set_xlabel("retained fraction (matched)"); ax.set_ylabel(r"$(m_1+m_2)/2$"); ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, figdir, "T7_convention")


REF_M = (-0.51247, -0.48568)  # low-|g| m1, m2 of the cache's shear_bias_report (100k fibrepair default)


def reference_check(D, out):
    w0 = np.full(D["n"], 1.0 / D["n"])
    w, diag = sbr.compose_shape_noise_regularized_weights(w0, D["v1"], D["v2"])
    r1 = fit_component(D, "g1", w); r2 = fit_component(D, "g2", w)
    # also with the cache's stored TF mean (must be identical)
    cm = D["raw"]["cache_tf_mean"]
    r1c = sbr.component_metrics(D["true"]["g1"], cm[:, 0], LOW_G, w)
    out["reference_check"] = {
        "expected": {"m1": -0.51247, "m1_se": 0.01551, "m2": -0.48568, "m2_se": 0.01534,
                     "c1_1e4": 0.440, "c2_1e4": 0.807, "sigma_shape": 0.03894, "ess": 90897.2},
        "got": {"m1": r1["low_m"], "m1_se": r1["low_m_se"], "m2": r2["low_m"], "m2_se": r2["low_m_se"],
                "c1_1e4": 1e4 * r1["c"], "c2_1e4": 1e4 * r2["c"],
                "sigma_shape_weighted": diag["weighted_shape_noise"], "floor": diag["shape_noise"],
                "ess": diag["ess"], "m1_from_cache_mean": r1c["low_m"]},
    }
    if REF_M != (-0.51247, -0.48568):
        out["reference_check"]["expected"] = {"m1": REF_M[0], "m2": REF_M[1]}
    ok = abs(r1["low_m"] - REF_M[0]) < 2e-4 and abs(r2["low_m"] - REF_M[1]) < 2e-4
    out["reference_check"]["reproduced"] = bool(ok)
    return ok


def save(fig, figdir, name, paperdir=None, pname=None):
    fig.savefig(figdir / f"{name}.png")
    if paperdir is not None and pname is not None:
        fig.savefig(paperdir / f"{pname}.png")
        fig.savefig(paperdir / f"{pname}.pdf")
    plt.close(fig)


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--paperdir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--only", default="all")
    p.add_argument("--reference-m", default=None,
                   help="m1,m2 (low-|g|) from this cache's shear_bias_report; default: the 100k fibrepair values")
    a = p.parse_args(argv)
    if a.reference_m is not None:
        global REF_M
        REF_M = tuple(float(x) for x in a.reference_m.split(","))
    if (a.outdir / "results.json").exists():
        raise SystemExit("refusing to overwrite results.json")
    a.outdir.mkdir(parents=True, exist_ok=True); a.paperdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = load(a.table)
    out = {"n_galaxies": D["n"], "low_g": LOW_G, "sigma_pi": SIGMA_PI, "table": str(a.table)}
    ok = reference_check(D, out)
    print("reference:", json.dumps(out["reference_check"], indent=1), flush=True)
    if not ok and D["n"] == 100000:
        (a.outdir / "results.json").write_text(json.dumps(jsonable(out), indent=1))
        raise SystemExit("reference NOT reproduced; stopping")
    toy = uniform_prior_toy()
    print(f"toy done {time.time() - t0:.0f}s", flush=True)
    logsig_pred, r2, names = gbm_predicted_log_sigma(D)
    print(f"gbm r2={r2:.3f} {time.time() - t0:.0f}s", flush=True)
    test_rraw(D, out, a.outdir, a.paperdir); print("R-raw", flush=True)
    test_t1(D, out, a.outdir, logsig_pred, r2, names); print("T1", flush=True)
    test_t3(D, out, a.outdir, a.paperdir, toy); print("T3", flush=True)
    test_t4(D, out, a.outdir); print("T4", flush=True)
    test_r6(D, out, a.outdir, a.paperdir); print("R6", flush=True)
    test_t2a(D, out, a.outdir, logsig_pred, toy); print("T2a", flush=True)
    test_t7(D, out, a.outdir); print("T7", flush=True)
    test_t5(D, out, a.n_boot); print("T5", flush=True)
    out["seconds"] = time.time() - t0
    (a.outdir / "results.json").write_text(json.dumps(jsonable(out), indent=1))
    print(f"done {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
