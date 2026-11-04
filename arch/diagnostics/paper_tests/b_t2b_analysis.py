#!/usr/bin/env python3
"""T2b: shear response, selection response of a posterior-width cut, and the
R(sigma) = a + b sigma^2 fit on the matched five-shear stencil.

Estimator: TF-weighted posterior mean of (g1, g2) per row, from the cached
10^4 identity+R90 candidates and ``posterior_tf_log_weight``.
Width: sigma_eps,i = sqrt(sigma_g1^2 + sigma_g2^2) from TF-weighted 1D marginal
standard deviations (Xu+24 shape-noise definition).

Population weighting: "raw" = equal weight per base galaxy (primary);
"tf" = globally normalized truth-level TF population ratio per base (secondary,
as in the legacy shear-response report).

Uncertainties: bootstrap over base galaxies (all five states of a base move
together), 2000 resamples unless stated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ARCH = Path("/jet/home/xwang30/kl-nn/arch")
sys.path.insert(0, str(ARCH))
from cache_contract import load_cache_partitions  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MANIFEST = S / "samples/shear_response_simv3_cosi_5k_manifest.csv"
STATES = ("zero", "g1_plus", "g1_minus", "g2_plus", "g2_minus")
SIGMA_PI = 0.2 / np.sqrt(12.0)
THRESHOLDS = (0.03, 0.05, 0.075, 0.1)
NBOOT = 2000


def summarize_cache(root: Path, out: Path) -> dict:
    """Per-row TF-weighted mean / sd of g1, g2 (streamed per partition)."""
    if out.is_file():
        return dict(np.load(out))
    cp = load_cache_partitions(root)
    keys = ("truth", "image_snr", "central_halpha_snr", "population_tf_log_ratio",
            "posterior_tf_ess_fraction", "tf_target_mean_estimates")
    cols = {k: [] for k in keys}
    mean, sd, sd_eq, mean_eq = [], [], [], []
    for i, sample_path in enumerate(cp.files["sample"]):
        s = np.load(sample_path, mmap_mode="r")
        lw = np.load(cp.files["posterior_tf_log_weight"][i], mmap_mode="r")
        for a in range(0, s.shape[0], 100):
            g = np.asarray(s[a:a + 100, :, :2], dtype=np.float64)
            l = np.asarray(lw[a:a + 100], dtype=np.float64)
            w = np.exp(l - l.max(axis=1, keepdims=True))
            w /= w.sum(axis=1, keepdims=True)
            mu = np.einsum("nk,nkc->nc", w, g)
            var = np.einsum("nk,nkc->nc", w, (g - mu[:, None]) ** 2)
            mean.append(mu); sd.append(np.sqrt(var))
            mean_eq.append(g.mean(1)); sd_eq.append(g.std(1))
        for k in keys:
            cols[k].append(np.asarray(np.load(cp.files[k][i])))
        print(f"summarized {sample_path}", flush=True)
    res = {k: np.concatenate(v) for k, v in cols.items()}
    res.update(mean=np.concatenate(mean), sd=np.concatenate(sd),
               mean_eq=np.concatenate(mean_eq), sd_eq=np.concatenate(sd_eq))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **res)
    return res


def cubes(summary: dict, manifest: Path = MANIFEST, posterior: str = "tf"):
    man = pd.read_csv(manifest).sort_values("ID").reset_index(drop=True)
    n = len(man)
    assert np.allclose(man[["g1", "g2"]].to_numpy(), summary["truth"][:n, :2], atol=1e-6)
    order = man.state.map({s: k for k, s in enumerate(STATES)}).to_numpy()
    assert np.array_equal(order, np.tile(np.arange(5), n // 5)), "manifest not in stencil order"
    nb = n // 5
    key = "" if posterior == "tf" else "_eq"
    ghat = summary["mean" + key].reshape(nb, 5, 2)
    sd = summary["sd" + key].reshape(nb, 5, 2)
    truth = summary["truth"].reshape(nb, 5, -1)
    delta = float(truth[0, 1, 0])
    # sanity: TF-weighted mean recomputed == cached tf_target_mean_estimates[:, 1, :2]
    cached = summary["tf_target_mean_estimates"][:, 1, :2]
    max_dev = float(np.max(np.abs(cached - summary["mean"])))
    plr = summary["population_tf_log_ratio"].reshape(nb, 5)
    assert np.all(plr == plr[:, :1])
    w_tf = np.exp(plr[:, 0] - plr[:, 0].max()); w_tf /= w_tf.sum()
    return dict(ghat=ghat, sd=sd, sig=np.hypot(sd[..., 0], sd[..., 1]), truth=truth,
                delta=delta, cosi=truth[:, 0, 3], image_snr=summary["image_snr"].reshape(nb, 5)[:, 0],
                halpha_snr=summary["central_halpha_snr"].reshape(nb, 5)[:, 0], w_tf=w_tf,
                max_dev_vs_cached_mean=max_dev,
                ess_fraction=summary["posterior_tf_ess_fraction"].reshape(nb, 5))


def per_base_response(c):
    g, d = c["ghat"], c["delta"]
    R = np.empty((len(g), 2, 2))
    R[:, :, 0] = (g[:, 1] - g[:, 2]) / (2 * d)
    R[:, :, 1] = (g[:, 3] - g[:, 4]) / (2 * d)
    return R


def wmean(x, w):
    w = w / w.sum()
    return np.tensordot(w, x, axes=(0, 0))


def boot_stat(fn, nb, w, rng, nboot=NBOOT):
    """fn(idx, w_idx) -> array; returns (value, se)."""
    val = np.asarray(fn(np.arange(nb), w))
    reps = []
    for _ in range(nboot):
        idx = rng.integers(0, nb, nb)
        reps.append(fn(idx, w[idx]))
    reps = np.asarray(reps)
    return val, np.nanstd(reps, axis=0)


def response_block(R, c, mask, w, rng):
    idx0 = np.flatnonzero(mask)
    if len(idx0) < 5:
        return dict(n=int(len(idx0)))
    Rm = R[idx0]; ww = w[idx0]
    val, se = boot_stat(lambda i, wi: wmean(Rm[i], wi), len(idx0), ww, rng, nboot=1000)
    addv, adds = boot_stat(lambda i, wi: wmean(c["ghat"][idx0][i, 0], wi), len(idx0), ww, rng, nboot=1000)
    return dict(n=int(len(idx0)), weight_ess=float(ww.sum() ** 2 / np.sum(ww ** 2)),
                R=val.tolist(), R_se=se.tolist(), R_diag=float(np.trace(val) / 2),
                R_diag_se=float(np.hypot(se[0, 0], se[1, 1]) / 2),
                c=addv.tolist(), c_se=adds.tolist(),
                mean_sigma_eps=float(wmean(c["sig"][idx0, 0], ww)),
                mean_sigma_comp_sq=float(wmean(0.5 * c["sig"][idx0, 0] ** 2, ww)))


def selection_response(c, t, w, rng):
    """Metacal-style selection response of the cut sigma_eps < t.

    sel_s = sigma_eps measured on state s < t (same noise, sheared image).
    R_gamma  = <R_b> over bases selected at zero shear
    R_sel_ij = [<ghat_i(zero)>_{sel(+j)} - <ghat_i(zero)>_{sel(-j)}] / 2 Delta
    R_tot_ij = [<ghat_i(+j)>_{sel(+j)} - <ghat_i(-j)>_{sel(-j)}] / 2 Delta
    """
    d = c["delta"]; g = c["ghat"]; sig = c["sig"]; R = per_base_response(c)
    sel = sig < t  # (nb, 5)

    def stats(i, wi):
        s = sel[i]; gi = g[i]; out = np.full(3 * 4 + 5 + 2, np.nan)
        def m(x, mask):
            ww = wi * mask
            return np.tensordot(ww, x, axes=(0, 0)) / ww.sum() if ww.sum() > 0 else np.full(x.shape[1:], np.nan)
        Rg = m(R[i], s[:, 0])
        Rsel = np.empty((2, 2)); Rtot = np.empty((2, 2))
        for j, (p, q) in enumerate(((1, 2), (3, 4))):
            Rsel[:, j] = (m(gi[:, 0], s[:, p]) - m(gi[:, 0], s[:, q])) / (2 * d)
            Rtot[:, j] = (m(gi[:, p], s[:, p]) - m(gi[:, q], s[:, q])) / (2 * d)
        frac = np.array([np.sum(wi * s[:, k]) / wi.sum() for k in range(5)])
        dfrac = np.array([(frac[1] - frac[2]) / (2 * d), (frac[3] - frac[4]) / (2 * d)])
        return np.concatenate([Rg.ravel(), Rsel.ravel(), Rtot.ravel(), frac, dfrac])

    val, se = boot_stat(stats, len(g), w, rng)
    flips = {f"{STATES[p]}_vs_{STATES[q]}": int(np.sum(sel[:, p] != sel[:, q]))
             for p, q in ((1, 2), (3, 4), (0, 1), (0, 3))}
    unpack = lambda v: dict(R_gamma=v[0:4].reshape(2, 2).tolist(), R_sel=v[4:8].reshape(2, 2).tolist(),
                            R_tot=v[8:12].reshape(2, 2).tolist(), frac_by_state=v[12:17].tolist(),
                            dfrac_dg=v[17:19].tolist())
    out = dict(threshold=t, value=unpack(val), se=unpack(se), n_flips=flips,
               n_selected_zero=int(sel[:, 0].sum()))
    rsel = val[4:8].reshape(2, 2); rsel_se = se[4:8].reshape(2, 2)
    out["R_sel_diag"] = float((rsel[0, 0] + rsel[1, 1]) / 2)
    out["R_sel_diag_se"] = float(np.hypot(rsel_se[0, 0], rsel_se[1, 1]) / 2)
    rg = val[0:4].reshape(2, 2)
    out["R_gamma_diag"] = float(np.trace(rg) / 2)
    out["R_sel_over_R_gamma"] = out["R_sel_diag"] / out["R_gamma_diag"]
    # paired bootstrap SE of R_sel_diag already accounts for sample noise
    out["R_sel_significance_sigma"] = out["R_sel_diag"] / out["R_sel_diag_se"] if out["R_sel_diag_se"] > 0 else None
    return out


def fit_R_sigma(R, c, w, rng):
    """OLS fits of R = a + b x with bootstrap errors.

    per-component: R11 vs sigma_g1^2 and R22 vs sigma_g2^2 pooled (prediction
      a=1, b=-1/sigma_pi^2 exactly for a Gaussian prior and likelihood, since
      R = 1 - sigma_post^2 / sigma_pi^2);
    trace: (R11+R22)/2 vs sigma_eps^2 (prediction b = -1/(2 sigma_pi^2)).
    sigma is measured on the zero-shear state.
    """
    sd0 = c["sd"][:, 0]; sig0 = c["sig"][:, 0]
    y_comp = np.concatenate([R[:, 0, 0], R[:, 1, 1]])
    x_comp = np.concatenate([sd0[:, 0] ** 2, sd0[:, 1] ** 2])
    w_comp = np.concatenate([w, w])
    y_tr = 0.5 * (R[:, 0, 0] + R[:, 1, 1]); x_tr = sig0 ** 2

    def ols(x, y, ww):
        X = np.column_stack([np.ones_like(x), x])
        A = X.T @ (ww[:, None] * X)
        return np.linalg.solve(A, X.T @ (ww * y))

    nb = len(R)
    res = {}
    for name, fn in (
        ("per_component", lambda i, wi: ols(np.concatenate([sd0[i, 0] ** 2, sd0[i, 1] ** 2]),
                                            np.concatenate([R[i, 0, 0], R[i, 1, 1]]), np.concatenate([wi, wi]))),
        ("trace_vs_sigma_eps_sq", lambda i, wi: ols(sig0[i] ** 2, 0.5 * (R[i, 0, 0] + R[i, 1, 1]), wi)),
        ("R11_vs_sigma_g1_sq", lambda i, wi: ols(sd0[i, 0] ** 2, R[i, 0, 0], wi)),
        ("R22_vs_sigma_g2_sq", lambda i, wi: ols(sd0[i, 1] ** 2, R[i, 1, 1], wi)),
    ):
        v, s = boot_stat(fn, nb, w, rng)
        res[name] = dict(a=float(v[0]), a_se=float(s[0]), b=float(v[1]), b_se=float(s[1]))
    res["per_component"]["b_pred"] = -1 / SIGMA_PI ** 2
    res["trace_vs_sigma_eps_sq"]["b_pred"] = -1 / (2 * SIGMA_PI ** 2)
    res["R11_vs_sigma_g1_sq"]["b_pred"] = res["R22_vs_sigma_g2_sq"]["b_pred"] = -1 / SIGMA_PI ** 2
    # residual scatter of the per-base trace response about the fit
    a, b = res["trace_vs_sigma_eps_sq"]["a"], res["trace_vs_sigma_eps_sq"]["b"]
    res["trace_vs_sigma_eps_sq"]["residual_rms"] = float(np.sqrt(wmean((y_tr - a - b * x_tr) ** 2, w)))
    res["trace_vs_sigma_eps_sq"]["corr"] = float(np.corrcoef(x_tr, y_tr)[0, 1])
    # the 'fixed a=1' one-parameter fit for comparison
    x = x_comp; y = y_comp - 1.0
    bfix = float(np.sum(w_comp * x * y) / np.sum(w_comp * x * x))
    res["per_component"]["b_fixed_a1"] = bfix
    return res


VARIANTS = {
    # name: (posterior weighting, base-galaxy weighting)
    "tf_equal": ("tf", "equal"),      # production estimator, equal weight per base
    "tf_target": ("tf", "tfpop"),     # legacy 'tf_target': TF posterior + TF population weights
    "proposal": ("eq", "equal"),      # unweighted posterior mean, equal weights
}


def analyze(label, cache_root, out_dir, manifest=MANIFEST, variants=("tf_equal", "tf_target", "proposal"), seed=31415):
    rng = np.random.default_rng(seed)
    summ = summarize_cache(cache_root, S / "cache/paper_tests_B_inference" / f"summary_{label}.npz")
    c = cubes(summ, manifest, "tf")
    R = per_base_response(c)
    nb = len(R)
    result = dict(label=label, cache_root=str(cache_root), manifest=str(manifest), nbase=nb, delta_g=c["delta"],
                  max_abs_dev_recomputed_vs_cached_tf_mean=c["max_dev_vs_cached_mean"],
                  tf_population_ess=float(1 / np.sum(c["w_tf"] ** 2)),
                  posterior_tf_ess_fraction_pct=np.percentile(c["ess_fraction"], [1, 16, 50, 84]).tolist(),
                  sigma_eps_zero_pct=np.percentile(c["sig"][:, 0], [5, 16, 50, 84, 95]).tolist(),
                  sigma_eps_mean=float(c["sig"][:, 0].mean()),
                  sigma_eps_rms=float(np.sqrt(np.mean(c["sig"][:, 0] ** 2))))
    cubes_out = {}
    for wname in variants:
        post, basew = VARIANTS[wname]
        c = cubes(summ, manifest, post)
        R = per_base_response(c)
        w = np.full(nb, 1.0 / nb) if basew == "equal" else c["w_tf"]
        cubes_out[wname] = (c, R)
        block = {"posterior_weighting": post, "base_weighting": basew,
                 "sigma_eps_zero_pct": np.percentile(c["sig"][:, 0], [5, 16, 50, 84, 95]).tolist()}
        block["overall"] = response_block(R, c, np.ones(nb, bool), w, rng)
        edges = [0, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 1.0]
        s0 = c["sig"][:, 0]
        block["by_sigma"] = [dict(lo=lo, hi=hi, **response_block(R, c, (s0 >= lo) & (s0 < hi), w, rng))
                             for lo, hi in zip(edges[:-1], edges[1:])]
        cedges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        block["by_cosi"] = [dict(lo=lo, hi=hi, **response_block(R, c, (c["cosi"] >= lo) & (c["cosi"] < hi + 1e-9), w, rng))
                            for lo, hi in zip(cedges[:-1], cedges[1:])]
        block["fit_R_sigma"] = fit_R_sigma(R, c, w, rng)
        block["selection"] = [selection_response(c, t, w, rng) for t in THRESHOLDS]
        result[wname] = block
    return result, cubes_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--variants", default="tf_equal,tf_target,proposal")
    a = p.parse_args()
    variants = tuple(a.variants.split(","))
    res, cubes_out = analyze(a.label, a.cache_root, a.out.parent, a.manifest, variants)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for wname, (c, R) in cubes_out.items():
        np.savez(a.out.with_name(a.out.stem + f".{wname}.cubes.npz"), R=R, ghat=c["ghat"], sd=c["sd"], sig=c["sig"],
                 cosi=c["cosi"], w_tf=c["w_tf"], delta=c["delta"], image_snr=c["image_snr"],
                 halpha_snr=c["halpha_snr"], ess_fraction=c["ess_fraction"])
    print(json.dumps({k: res[k] for k in ("nbase", "sigma_eps_zero_pct", "tf_population_ess")}))
    for wname in variants:
        print(wname, "overall", res[wname]["overall"]["R"], res[wname]["overall"]["R_se"])
        print(wname, "fit", json.dumps(res[wname]["fit_R_sigma"]))
        for s in res[wname]["selection"]:
            print(wname, s["threshold"], "Rsel", s["R_sel_diag"], "+-", s["R_sel_diag_se"], "Rg", s["R_gamma_diag"],
                  "frac0", s["value"]["frac_by_state"][0], "flips", s["n_flips"])


if __name__ == "__main__":
    main()
