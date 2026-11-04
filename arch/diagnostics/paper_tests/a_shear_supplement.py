#!/usr/bin/env python3
"""Paper tests, Group A (shear): follow-ups to T1 / T2a / T3 prompted by the
first pass (results.json).

S1 (T1 follow-up): sigma_eps,i vs true shear away from the prior edge.
S2 (T2a follow-up): paired-bootstrap selection excess of the observed-sigma cut
    relative to a cut on a shear-free truth-level prediction of sigma (GBM),
    at matched retained fraction, after subtracting the shrinkage expected from
    the difference in retained posterior widths (uniform-prior toy).
S3 (mechanism): at fixed truth, is the posterior width correlated with the
    noise-driven radial shear estimate?
S4 (T3 variant): shrinkage test binned in the shear-free predicted sigma,
    which is immune to noise-driven selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a_shear_analysis as A  # noqa: E402
from a_shear_analysis import sbr  # noqa: E402


def m_avg(D, w):
    m1, _ = A.m_c_fast(D["true"]["g1"], D["est"]["g1"], w)
    m2, _ = A.m_c_fast(D["true"]["g2"], D["est"]["g2"], w)
    return 0.5 * (m1 + m2), m1, m2


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--paperdir", type=Path, required=True)
    p.add_argument("--n-boot", type=int, default=500)
    a = p.parse_args(argv)
    if (a.outdir / "supplement.json").exists():
        raise SystemExit("refusing to overwrite supplement.json")
    D = A.load(a.table)
    toy = A.uniform_prior_toy()
    logsig_pred, r2, _ = A.gbm_predicted_log_sigma(D)
    sig = D["sig"]
    resid = np.log(sig) - logsig_pred
    out = {"gbm_r2": r2}

    # ---- S1: interior slopes
    g1, g2 = D["true"]["g1"], D["true"]["g2"]
    absg = np.hypot(g1, g2)
    gp, gx = D["true"]["g+"], D["true"]["gx"]
    s1 = []
    for box in (0.03, 0.05, 0.07, 0.1):
        m = (np.abs(g1) < box) & (np.abs(g2) < box)
        row = {"box_max_abs_g1_g2": box, "n": int(m.sum())}
        for name, x in (("|g|", absg), ("g+", gp), ("gx", gx)):
            c, s = A.slope_fit(resid[m], [x[m]])
            c2, s2 = A.slope_fit(resid[m], [x[m], x[m] ** 2])
            row[name] = {"dlnsigma_dg": c[1], "se": s[1], "quad_lin": c2[1], "quad_lin_se": s2[1],
                         "quad": c2[2], "quad_se": s2[2]}
        s1.append(row)
    out["S1_interior_slopes"] = s1

    # ---- S3: mechanism: residual ln sigma vs noise-driven radial estimate
    ghat_abs = np.hypot(D["est"]["g1"], D["est"]["g2"])
    # radial component of (ghat - g) along the true shear direction
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = g1 / absg, g2 / absg
    rad = (D["est"]["g1"] - g1) * ux + (D["est"]["g2"] - g2) * uy
    # regress resid on [|g|, rad]
    c, s = A.slope_fit(resid, [absg, rad])
    c3, s3 = A.slope_fit(resid, [absg, ghat_abs])
    from scipy.stats import spearmanr
    out["S3_mechanism"] = {
        "dresid_dradial_at_fixed_absg": c[2], "se": s[2],
        "dresid_dabsghat_at_fixed_absg": c3[2], "se_absghat": s3[2],
        "spearman_resid_vs_radial": float(spearmanr(resid, rad).correlation),
        "spearman_resid_vs_absghat": float(spearmanr(resid, ghat_abs).correlation),
    }
    # by cut: mean radial residual among selected vs not, low-|g| galaxies
    rows = []
    low = absg < 0.02
    for f in (0.1, 0.2, 0.3, 0.5):
        ms = A.matched_fraction_cut(sig, f)
        mg = A.matched_fraction_cut(logsig_pred, f)
        rows.append({"frac": f,
                     "mean_radial_resid_sigma_cut_lowg": float(rad[ms & low].mean()),
                     "mean_radial_resid_gbm_cut_lowg": float(rad[mg & low].mean()),
                     "se_sigma_cut": float(rad[ms & low].std() / np.sqrt((ms & low).sum())),
                     "se_gbm_cut": float(rad[mg & low].std() / np.sqrt((mg & low).sum()))})
    out["S3_mechanism"]["radial_resid_by_cut"] = rows

    # ---- S2: paired bootstrap selection excess
    fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    masks = {f: (A.matched_fraction_cut(sig, f), A.matched_fraction_cut(logsig_pred, f)) for f in fracs}

    def excess(counts, f, kind):
        ms, mg = masks[f]
        res = {}
        for lab, mk in (("sigma", ms), ("gbm", mg)):
            w, _ = A.estimator_weights(D, kind, mk, base=counts)
            mm, _, _ = m_avg(D, w)
            sel = w > 0
            ww = w[sel] / w[sel].sum()
            # toy prediction with the estimator's own galaxy weights
            t1 = A.toy_m_from_post_std(np.sqrt(D["v1"][sel]), toy)
            t2 = A.toy_m_from_post_std(np.sqrt(D["v2"][sel]), toy)
            res[lab] = (mm, float(np.sum(ww * 0.5 * (t1 + t2))))
        dm = res["sigma"][0] - res["gbm"][0]
        dt = res["sigma"][1] - res["gbm"][1]
        return dm, dt, dm - dt, res

    s2 = {}
    rng = np.random.default_rng(7)
    for kind in ("raw", "weighted"):
        pts = {f: excess(None, f, kind) for f in fracs}
        boots = {f: [] for f in fracs}
        for b in range(a.n_boot):
            counts = rng.multinomial(D["n"], np.full(D["n"], 1.0 / D["n"])).astype(float)
            for f in fracs:
                boots[f].append(excess(counts, f, kind)[:3])
        s2[kind] = []
        for f in fracs:
            bb = np.array(boots[f])
            dm, dt, ex, res = pts[f]
            s2[kind].append({"frac": f, "m_sigma_cut": res["sigma"][0], "m_gbm_cut": res["gbm"][0],
                             "toy_sigma_cut": res["sigma"][1], "toy_gbm_cut": res["gbm"][1],
                             "delta_m": dm, "delta_m_se": float(bb[:, 0].std(ddof=1)),
                             "delta_toy": dt, "selection_excess": ex,
                             "selection_excess_se": float(bb[:, 2].std(ddof=1))})
        print("S2", kind, "done", flush=True)
    out["S2_paired_selection_excess"] = {"n_boot": a.n_boot, **s2}

    # ---- S4: T3 binned in predicted sigma
    nb = 12
    edges = np.quantile(logsig_pred, np.linspace(0, 1, nb + 1)); edges[-1] += 1e-12
    s4 = []
    for i in range(nb):
        m = (logsig_pred >= edges[i]) & (logsig_pred < edges[i + 1])
        w = (m / m.sum()).astype(float)
        row = {"bin": i, "n": int(m.sum()), "mean_sigma_eps_obs": float(sig[m].mean()),
               "mean_sigma_eps_pred": float(np.exp(logsig_pred[m]).mean())}
        for comp in A.COMPONENTS:
            r = A.fit_component(D, comp, w)
            row[comp] = {"m": r["low_m"], "m_se": r["low_m_se"], **A.shrink_predictions(D, m, comp, toy)}
        s4.append(row)
    out["S4_T3_predicted_sigma_bins"] = s4
    chi = {}
    for key in ("plan_formula", "gauss_posterior", "uniform_toy"):
        for comp in A.COMPONENTS:
            d = np.array([(r[comp]["m"] - r[comp][key]) / r[comp]["m_se"] for r in s4])
            chi[f"{key}:{comp}"] = {"chi2": float(np.sum(d ** 2)), "ndof": nb,
                                    "mean_diff": float(np.mean([r[comp]["m"] - r[comp][key] for r in s4]))}
    out["S4_chi2"] = chi

    # ---- figures
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for kind, ls, mk in (("raw", "-", "o"), ("weighted", "--", "s")):
        rows_ = s2[kind]
        fr = [r["frac"] for r in rows_]
        axes[0].errorbar(fr, [r["selection_excess"] for r in rows_], [r["selection_excess_se"] for r in rows_],
                         fmt=mk + ls, color=A.C1, ms=4, lw=1.2, label=f"{kind}: Δm − Δm_toy")
        axes[0].plot(fr, [r["delta_m"] for r in rows_], ls=ls, color=A.GREY, lw=1, label=f"{kind}: Δm (no toy correction)")
    axes[0].axhline(0, color=A.GREY, lw=0.7)
    axes[0].set_xlabel("retained fraction")
    axes[0].set_ylabel(r"$m_{\sigma\ \rm cut} - m_{\rm shear\!-\!free\ cut}$")
    axes[0].set_title("selection excess of the observed-σ cut (paired bootstrap)", fontsize=9)
    axes[0].legend(fontsize=7.5)
    ax = axes[1]
    x = np.array([r["mean_sigma_eps_obs"] for r in s4])
    for comp, col, mk, dx in (("g1", A.C1, "o", -0.0003), ("g2", A.C2, "s", 0.0003)):
        ax.errorbar(x + dx, [r[comp]["m"] for r in s4], [r[comp]["m_se"] for r in s4], fmt=mk, ms=4.5,
                    color=col, lw=1, label=f"measured m ({comp})")
    ax.plot(x, [0.5 * (r["g1"]["plan_formula"] + r["g2"]["plan_formula"]) for r in s4], ":", color=A.GREY,
            label=r"$-\langle\sigma^2/(\sigma^2+\sigma_\pi^2)\rangle$")
    ax.plot(x, [0.5 * (r["g1"]["gauss_posterior"] + r["g2"]["gauss_posterior"]) for r in s4], "--", color=A.GREY,
            label=r"$-\langle\sigma^2\rangle/\sigma_\pi^2$")
    ax.plot(x, [0.5 * (r["g1"]["uniform_toy"] + r["g2"]["uniform_toy"]) for r in s4], "-", color="k",
            label="uniform-prior toy")
    ax.axhline(0, color=A.GREY, lw=0.6)
    ax.set_xlabel(r"bin mean $\sigma_{\epsilon,i}$ (bins in shear-free predicted σ)")
    ax.set_ylabel("m (|g| < 0.02)")
    ax.set_title("shrinkage test, selection-free binning", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    A.save(fig, a.outdir, "S2_S4_selection_and_shrinkage")
    # stand-alone paper candidate for the selection-free shrinkage plot
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for comp, col, mk, dx in (("g1", A.C1, "o", -0.0003), ("g2", A.C2, "s", 0.0003)):
        ax.errorbar(x + dx, [r[comp]["m"] for r in s4], [r[comp]["m_se"] for r in s4], fmt=mk, ms=4.5,
                    color=col, lw=1, label=f"measured $m_{comp[-1]}$")
    ax.plot(x, [0.5 * (r["g1"]["gauss_posterior"] + r["g2"]["gauss_posterior"]) for r in s4], "--", color=A.GREY,
            label=r"Gaussian: $-\langle\sigma_g^2\rangle/\sigma_\pi^2$")
    ax.plot(x, [0.5 * (r["g1"]["uniform_toy"] + r["g2"]["uniform_toy"]) for r in s4], "-", color="k",
            label="uniform-prior toy")
    ax.axhline(0, color=A.GREY, lw=0.6)
    ax.set_xlabel(r"bin mean $\sigma_{\epsilon,i}$")
    ax.set_ylabel(r"$m$ ($|g|<0.02$ fit)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    A.save(fig, a.outdir, "S4_shrinkage_selection_free", a.paperdir, "A_D1_shrinkage_selection_free")
    # S1 figure: interior
    (a.outdir / "supplement.json").write_text(json.dumps(A.jsonable(out), indent=1))
    print("done")


if __name__ == "__main__":
    main()
