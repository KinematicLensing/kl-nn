#!/usr/bin/env python3
"""D: shear coverage conditional on the true shear, vs the shrinkage prediction.

Question: global 16-84% coverage of the TF-weighted g1/g2 posteriors is nominal,
but coverage at fixed true |g| is not (~88% for |g|<0.05, ~62% for |g|>0.05 in
likelihood-stack/02_coverage). Is that what a *correct* posterior under the
bounded U(-0.1, 0.1) shear prior should show, or is the learned posterior wrong?

Inputs (read-only): paper-tests/A_shear/per_galaxy.npz (TF-weighted posterior
mean, variance and 2.5/16/50/84/97.5% quantiles per component, plus truths),
extracted from the Xu3 cache by a_shear_extract.py.

Two predictions of the coverage for each galaxy and component, given its
measured posterior width and its true shear:

1. Gaussian model (analytic). Prior N(0, s_pi^2), s_pi = 0.2/sqrt(12); likelihood
   N(g_obs; g, s_L^2). With k = 1 - s_post^2/s_pi^2 the posterior mean is
   ghat = k g_obs, and ghat - g ~ N(beta, s^2), beta = -g s_post^2/s_pi^2,
   s = s_post sqrt(k). The interval ghat +/- z s_post covers g with probability
   Phi((z s_post - beta)/s) - Phi((-z s_post - beta)/s). s_post >= s_pi is
   impossible in this model; such components are clipped to s_post = s_pi
   (k = 0, posterior = prior: ghat = 0 and the probability becomes the
   indicator |g| < z s_pi).

2. Uniform-prior Monte Carlo toy. Posterior = N(g_obs, s_L^2) truncated to
   [-0.1, 0.1]. s_L (and a location mu_L) per galaxy and component are found
   by matching the truncated Gaussian's mean and variance to the measured
   TF-weighted posterior mean and variance. Components whose (mean, variance)
   is outside the truncated-Gaussian family (too wide; the prior-dominated
   limit) get s_L = inf, i.e. posterior = prior. Then for each galaxy draw
   N_MC values g_obs ~ N(g_true, s_L^2) and compute the PIT
   u = F_post(g_true | g_obs); coverage at level L is P((1-L)/2 <= u <= (1+L)/2).

   If the moment match has no solution (the measured posterior is wider, at its
   mean, than any truncated Gaussian can be), s_L is instead set so that
   N(mean, s_L^2) truncated to the box has the measured variance; if even that
   fails (variance >= the uniform variance, the prior-dominated limit), s_L = inf.
   A variant ("toy_alt") uses the fixed-location width match for every galaxy.

3. NPE self-consistency (model-free). From d_localcal_stream.py: for a correct
   posterior and truths drawn from the prior, P(covered | g in B) =
   sum_i alpha_iB / sum_i beta_iB, with beta_iB galaxy i's posterior mass in truth
   bin B and alpha_iB the part inside its own interval. This is what the NPE's
   own posteriors say the conditional coverage must be if they are correct.

Model-free check: for a correct posterior, coverage conditional on any function
of the data (posterior mean, posterior width) is nominal. So coverage binned in
the posterior mean and in width terciles tests the NPE without any toy model.

Usage: python d_localcal_analysis.py [--n-mc 256] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

S = Path("/ocean/projects/phy250048p/shared")
PER_GALAXY = S / "reports/paper-tests/A_shear/per_galaxy.npz"
EARLIER = S / "reports/likelihood-stack/02_coverage/report.json"
OUT_DIR = S / "reports/paper-tests/D_local_calibration"

H = 0.1                                  # prior half-width: g_k ~ U(-H, H)
SIGMA_PI = 2 * H / np.sqrt(12.0)         # prior standard deviation 0.0577
LEVELS = {"68": 0.68, "95": 0.95}        # central credible levels
QCOL = {"68": (1, 3), "95": (0, 4)}      # columns of tf_q_g* (2.5,16,50,84,97.5)
ABS_SPLIT = 0.05
N_GBINS = 20
N_ABSBINS = 14
COMPONENTS = ("g1", "g2")


# ----------------------------------------------------------------- math helpers
def _log1mexp(x):
    """log(1 - exp(x)) for x <= 0, stable."""
    x = np.minimum(x, -1e-300)
    return np.where(x > -0.693, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def log_phi_diff(lo, hi):
    """log(Phi(hi) - Phi(lo)) for lo <= hi, stable in both tails."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    out = np.empty(np.broadcast(lo, hi).shape)
    lo, hi = np.broadcast_to(lo, out.shape), np.broadcast_to(hi, out.shape)
    right = lo > 0                               # both in the upper tail: use -x
    left = hi < 0                                # both in the lower tail
    mid = ~(right | left)
    lh, ll = log_ndtr(-lo[right]), log_ndtr(-hi[right])
    out[right] = lh + _log1mexp(ll - lh)
    lh, ll = log_ndtr(hi[left]), log_ndtr(lo[left])
    out[left] = lh + _log1mexp(ll - lh)
    out[mid] = np.log(np.maximum(ndtr(hi[mid]) - ndtr(lo[mid]), 1e-300))
    return out


def trunc_moments(mu, sig):
    """Mean and variance of N(mu, sig^2) truncated to [-H, H]."""
    a = (-H - mu) / sig
    b = (H - mu) / sig
    logz = log_phi_diff(a, b)
    lpa = -0.5 * a * a - 0.5 * np.log(2 * np.pi)
    lpb = -0.5 * b * b - 0.5 * np.log(2 * np.pi)
    ra = np.exp(lpa - logz)
    rb = np.exp(lpb - logz)
    m = ra - rb
    v = 1.0 + a * ra - b * rb - m * m
    return mu + sig * m, sig * sig * np.maximum(v, 0.0)


def trunc_cdf(x, mu, sig):
    """CDF at x in [-H, H] of N(mu, sig^2) truncated to [-H, H]."""
    a = (-H - mu) / sig
    b = (H - mu) / sig
    c = np.clip((x - mu) / sig, a, b)
    return np.exp(log_phi_diff(a, c) - log_phi_diff(a, b))


def moment_match(mean, var, n_iter=60):
    """Solve trunc_moments(mu, sig) = (mean, var) by damped Newton in (mu, log sig).

    Returns mu, sig, converged. Components with var at or above what the
    truncated-Gaussian family can reach at that mean do not converge (sig runs
    off to infinity); the caller treats them as prior-dominated (sig = inf).
    """
    mean = np.asarray(mean, float)
    var = np.asarray(var, float)
    sd = np.sqrt(var)
    mu = mean.copy()
    ls = np.log(np.maximum(sd, 1e-6))
    # start from the Gaussian-prior inversion, which is closer for wide posteriors
    frac = np.clip(var / SIGMA_PI**2, 0, 0.95)
    mu = mean / (1 - frac)
    ls = np.log(sd / np.sqrt(1 - frac))
    target = np.stack([mean / H, sd / H])
    eps = 1e-6
    conv = np.zeros(mean.shape, bool)
    for _ in range(n_iter):
        def resid(mu_, ls_):
            m_, v_ = trunc_moments(mu_, np.exp(ls_))
            return np.stack([m_ / H, np.sqrt(v_) / H]) - target
        r = resid(mu, ls)
        conv = np.max(np.abs(r), axis=0) < 1e-7
        if conv.all():
            break
        dm = (resid(mu + eps * H, ls) - r) / (eps * H)
        dl = (resid(mu, ls + eps) - r) / eps
        det = dm[0] * dl[1] - dl[0] * dm[1]
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)
        step_mu = -(dl[1] * r[0] - dl[0] * r[1]) / det
        step_ls = -(-dm[1] * r[0] + dm[0] * r[1]) / det
        step_mu = np.clip(step_mu, -0.5 * H, 0.5 * H)
        step_ls = np.clip(step_ls, -0.5, 0.5)
        upd = ~conv
        mu = np.where(upd, mu + step_mu, mu)
        ls = np.where(upd, np.minimum(ls + step_ls, np.log(100 * H)), ls)
    r = np.stack([x for x in trunc_moments(mu, np.exp(ls))])
    conv = (np.abs(r[0] - mean) / H < 1e-5) & (np.abs(np.sqrt(r[1]) - sd) / H < 1e-5) \
        & (np.exp(ls) < 50 * H)
    return mu, np.exp(ls), conv


def width_match_at_mean(mean, var, n_iter=80):
    """1D fallback: sig such that N(mean, sig^2) truncated to [-H, H] has variance var.

    The location is fixed at the posterior mean. For a location inside the box the
    truncated SD increases monotonically with sig, towards the uniform SD (= SIGMA_PI)
    as sig -> inf; var at or above the value reached at sig = 1000 H returns inf.
    """
    mu = np.clip(np.asarray(mean, float), -H * (1 - 1e-9), H * (1 - 1e-9))
    target = np.sqrt(np.asarray(var, float))
    lo = np.full(mu.shape, np.log(1e-6 * H))
    hi = np.full(mu.shape, np.log(1e3 * H))
    top = np.sqrt(trunc_moments(mu, np.exp(hi))[1])
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        sd = np.sqrt(trunc_moments(mu, np.exp(mid))[1])
        below = sd < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    sig = np.exp(0.5 * (lo + hi))
    return np.where(target >= top, np.inf, sig)


# ------------------------------------------------------------------ predictions
def gaussian_prediction(g, s_post, level):
    """Probability that ghat +/- z s_post covers g in the Gaussian-prior model."""
    z = ndtri(0.5 + level / 2)
    sp = np.minimum(s_post, SIGMA_PI)                      # clip prior-dominated
    frac = (sp / SIGMA_PI) ** 2
    k = 1.0 - frac
    beta = -g * frac
    s = sp * np.sqrt(np.maximum(k, 0.0))
    hw = z * sp
    tiny = s < 1e-12
    s_safe = np.where(tiny, 1.0, s)
    p = ndtr((hw - beta) / s_safe) - ndtr((-hw - beta) / s_safe)
    return np.where(tiny, (np.abs(beta) < hw).astype(float), p)


def toy_pits(g, sig_l, n_mc, rng):
    """(N, n_mc) PITs u = F_post(g | g_obs), g_obs ~ N(g, sig_l^2), truncated posterior.

    sig_l = inf means posterior = prior (uniform), for which u = (g + H)/(2H).
    """
    out = np.empty((g.size, n_mc))
    fin = np.isfinite(sig_l)
    out[~fin] = ((g[~fin] + H) / (2 * H))[:, None]
    gf, sf = g[fin], sig_l[fin]
    chunk = 20000
    idx = np.flatnonzero(fin)
    for i0 in range(0, gf.size, chunk):
        gg = gf[i0:i0 + chunk, None]
        ss = sf[i0:i0 + chunk, None]
        gobs = gg + ss * rng.standard_normal((gg.shape[0], n_mc))
        out[idx[i0:i0 + chunk]] = trunc_cdf(gg, gobs, ss)
    return out


def coverage_from_pits(u, level):
    lo, hi = 0.5 - level / 2, 0.5 + level / 2
    return ((u >= lo) & (u <= hi)).mean(axis=1)


# ---------------------------------------------------------------------- binning
def bin_stats(mask_list, labels, meas, preds, centers=None):
    """Per-bin measured coverage (binomial SE) and each prediction's mean."""
    rows = []
    for j, m in enumerate(mask_list):
        n = int(m.sum())
        if n == 0:
            raise ValueError(f"empty bin {labels[j]}")
        c = float(meas[m].mean())
        se = float(np.sqrt(max(c * (1 - c), 0.25 / n) / n))  # floor for cells at 0% / 100%
        row = {"label": labels[j], "n": n, "measured": c, "measured_se": se}
        if centers is not None:
            row["center"] = float(centers[j])
        for name, p in preds.items():
            pm = float(p[m].mean())
            row[f"pred_{name}"] = pm
            row[f"resid_{name}"] = c - pm
            row[f"pull_{name}"] = (c - pm) / se
            # null SE if the prediction is exact (Poisson-binomial)
            row[f"null_se_{name}"] = float(np.sqrt(np.sum(p[m] * (1 - p[m]))) / n)
        rows.append(row)
    return rows


def chi2(rows, name):
    pulls = np.array([r[f"pull_{name}"] for r in rows])
    return {"chi2": float(np.sum(pulls**2)), "nbins": int(len(pulls)),
            "max_abs_pull": float(np.max(np.abs(pulls)))}


def add_self(rows, masses, set_name, k, lev, sels, bins):
    """Add the NPE self-consistency prediction to rows (one per (galaxy selection, truth bin)).

    pred = sum_i alpha_i / sum_i beta_i over the selected galaxies, where beta_i is
    galaxy i's posterior mass in the truth bin and alpha_i the part of it inside
    the galaxy's own interval. ``bins`` entries are an int bin index or None (all bins).
    null_se_self is the SE of (measured - predicted) if the posteriors are exact,
    from the conditional variance of the truth given each galaxy's data.
    """
    beta = masses[f"{set_name}_beta"][:, k, :].astype(np.float64)
    alpha = masses[f"{set_name}_alpha{lev}"][:, k, :].astype(np.float64)
    n_tot = beta.shape[0]
    for row, sel, b in zip(rows, sels, bins):
        bb = beta[sel].sum(axis=1) if b is None else beta[sel, b]
        aa = alpha[sel].sum(axis=1) if b is None else alpha[sel, b]
        c = float(aa.sum() / bb.sum())
        var_i = aa * (1 - c) ** 2 + (bb - aa) * c * c - (aa - c * bb) ** 2
        row["pred_self"] = c
        row["resid_self"] = row["measured"] - c
        row["pull_self"] = row["resid_self"] / row["measured_se"]
        row["null_se_self"] = float(np.sqrt(max(np.sum(var_i), 0.0)) / bb.sum())
        # floor: one galaxy's worth of binomial noise, so a 100%-covered cell gives a finite pull
        row["pull_self_null"] = row["resid_self"] / max(row["null_se_self"], 0.5 / bb.sum())
        # prior-mass check: expected vs actual number of truths in this cell
        row["expected_n_self"] = float(bb.sum())
        row["expected_n_self_frac_diff"] = float(row["n"] / bb.sum() - 1)
    return rows


# ------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-mc", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--masses", type=Path, default=OUT_DIR / "posterior_bin_masses.npz")
    args = ap.parse_args(argv)
    t0 = time.time()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ("report.json",):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; refusing to overwrite")
    rng = np.random.default_rng(args.seed)

    d = np.load(PER_GALAXY)
    masses = dict(np.load(args.masses))
    if not np.array_equal(masses["truth"], d["truth"][:, :2]):
        raise ValueError("posterior_bin_masses.npz truth order differs from per_galaxy.npz")
    truth = d["truth"][:, :2]
    tf_mean = d["tf_mean"]
    tf_var = d["tf_var"]
    q = {"g1": d["tf_q_g1"], "g2": d["tf_q_g2"]}
    assert np.allclose(d["quantile_levels"], [0.025, 0.16, 0.5, 0.84, 0.975])
    n_gal = truth.shape[0]
    absg = np.hypot(truth[:, 0], truth[:, 1])
    inner, outer = absg < ABS_SPLIT, absg > ABS_SPLIT

    res = {"n_galaxies": int(n_gal), "sigma_pi": float(SIGMA_PI), "prior_half_width": H,
           "abs_g_split": ABS_SPLIT, "n_mc": args.n_mc, "seed": args.seed,
           "abs_g_definition": "|g| = sqrt(g1_true^2 + g2_true^2) (np.hypot), equal galaxy weights",
           "interval_definition": "TF-weighted 16-84% (2.5-97.5%) posterior quantiles from per_galaxy.npz",
           "inputs": {"per_galaxy": str(PER_GALAXY), "earlier_report": str(EARLIER)}}

    # ---- reproduce the earlier split numbers
    earlier = json.loads(EARLIER.read_text())
    ref = {r["slice"]: r for r in earlier["table"] if r["posterior"] == "tf"}
    repro = {"n_inner": int(inner.sum()), "n_outer": int(outer.sum()),
             "reference_n_inner": earlier["n_inner"], "reference_n_outer": earlier["n_outer"]}
    meas = {}
    for k, comp in enumerate(COMPONENTS):
        for lev in LEVELS:
            lo, hi = QCOL[lev]
            meas[(comp, lev)] = ((truth[:, k] >= q[comp][:, lo]) & (truth[:, k] <= q[comp][:, hi])).astype(float)
        # the Gaussian-model interval definition, ghat +/- z sigma_post, measured too
        for lev, L in LEVELS.items():
            z = ndtri(0.5 + L / 2)
            s = np.sqrt(tf_var[:, k])
            meas[(comp, lev + "_meanpm")] = (np.abs(truth[:, k] - tf_mean[:, k]) <= z * s).astype(float)
        for sl, m in (("full", np.ones(n_gal, bool)), ("inner", inner), ("outer", outer)):
            repro[f"{comp}_{sl}"] = float(meas[(comp, "68")][m].mean())
            repro[f"{comp}_{sl}_reference"] = float(ref[sl][f"{comp}_coverage"])
    repro["max_abs_diff"] = max(abs(repro[f"{c}_{s}"] - repro[f"{c}_{s}_reference"])
                                for c in COMPONENTS for s in ("full", "inner", "outer"))
    res["reproduction"] = repro
    print("reproduction", json.dumps(repro, indent=1))

    # ---- per-galaxy predictions
    preds = {}
    toy_info = {}
    sig_l_all = {}
    for k, comp in enumerate(COMPONENTS):
        g = truth[:, k]
        s_post = np.sqrt(tf_var[:, k])
        for lev, L in LEVELS.items():
            preds[(comp, lev, "gauss")] = gaussian_prediction(g, s_post, L)
        t1 = time.time()
        mu_l, sig_mm, conv = moment_match(tf_mean[:, k], tf_var[:, k])
        sig_fix = width_match_at_mean(tf_mean[:, k], tf_var[:, k])
        # main toy: moment match; fallback: width match at the posterior mean; else prior
        sig_l = np.where(conv, sig_mm, sig_fix)
        sig_l_all[comp] = sig_l
        u = toy_pits(g, sig_l, args.n_mc, rng)
        for lev, L in LEVELS.items():
            preds[(comp, lev, "toy")] = coverage_from_pits(u, L)
        # sensitivity variant: width match at the posterior mean for every component
        u_alt = toy_pits(g, sig_fix, args.n_mc, rng)
        for lev, L in LEVELS.items():
            preds[(comp, lev, "toy_alt")] = coverage_from_pits(u_alt, L)
        del u_alt
        # the same toy evaluated on truths drawn from the prior (sanity: nominal)
        g_prior = rng.uniform(-H, H, n_gal)
        u_prior = toy_pits(g_prior, sig_l, 16, rng)
        toy_info[comp] = {
            "frac_moment_match_converged": float(conv.mean()),
            "frac_fallback_width_match": float((~conv & np.isfinite(sig_fix)).mean()),
            "frac_prior_dominated": float((~np.isfinite(sig_l)).mean()),
            "frac_prior_dominated_alt": float((~np.isfinite(sig_fix)).mean()),
            "sigma_L_ratio_mm_over_fixed_pct_5_50_95": np.percentile(
                (sig_mm / sig_fix)[conv & np.isfinite(sig_fix)], [5, 50, 95]).tolist(),
            "frac_sigma_post_ge_sigma_pi": float((s_post >= SIGMA_PI).mean()),
            "median_sigma_L_finite": float(np.median(sig_l[np.isfinite(sig_l)])),
            "median_sigma_post": float(np.median(s_post)),
            "toy_global_68_truths_from_prior": float(coverage_from_pits(u_prior, 0.68).mean()),
            "toy_global_95_truths_from_prior": float(coverage_from_pits(u_prior, 0.95).mean()),
            "gauss_global_68_truths_from_prior": float(gaussian_prediction(g_prior, s_post, 0.68).mean()),
            "seconds": time.time() - t1,
        }
        print(comp, toy_info[comp])
    res["toy_info"] = toy_info

    PRED_NAMES = ("gauss", "toy", "toy_alt")
    ALL_NAMES = ("self",) + PRED_NAMES
    allg = np.ones(n_gal, bool)

    def P(comp, lev):
        return {n: preds[(comp, lev, n)] for n in PRED_NAMES}

    # ---- (a) the |g| split
    split = {}
    for comp in COMPONENTS:
        for lev in LEVELS:
            masks = [np.ones(n_gal, bool), inner, outer]
            rows = bin_stats(masks, ["full", "|g|<0.05", "|g|>0.05"], meas[(comp, lev)], P(comp, lev))
            add_self(rows, masses, "split", COMPONENTS.index(comp), lev, [allg] * 3, [None, 0, 1])
            split[f"{comp}_{lev}"] = rows
            # Gaussian prediction vs its own interval definition (ghat +/- z sigma)
            rows_mp = bin_stats(masks, ["full", "|g|<0.05", "|g|>0.05"],
                                meas[(comp, lev + "_meanpm")], {"gauss": preds[(comp, lev, "gauss")]})
            split[f"{comp}_{lev}_meanpm"] = rows_mp
    res["split"] = split

    # ---- (b) fine bins in true g_k and in |g|
    gedges = np.linspace(-H, H, N_GBINS + 1)
    gcent = 0.5 * (gedges[1:] + gedges[:-1])
    aedges = np.linspace(0, np.sqrt(2) * H, N_ABSBINS + 1)
    acent = 0.5 * (aedges[1:] + aedges[:-1])
    fine = {}
    for k, comp in enumerate(COMPONENTS):
        g = truth[:, k]
        gm = [(g >= gedges[j]) & (g < gedges[j + 1]) for j in range(N_GBINS)]
        am = [(absg >= aedges[j]) & (absg < aedges[j + 1]) for j in range(N_ABSBINS)]
        for lev in LEVELS:
            rows_g = bin_stats(gm, [f"[{gedges[j]:.3f},{gedges[j+1]:.3f})" for j in range(N_GBINS)],
                               meas[(comp, lev)], P(comp, lev), gcent)
            rows_a = bin_stats(am, [f"[{aedges[j]:.3f},{aedges[j+1]:.3f})" for j in range(N_ABSBINS)],
                               meas[(comp, lev)], P(comp, lev), acent)
            add_self(rows_g, masses, "gk", k, lev, [allg] * N_GBINS, list(range(N_GBINS)))
            add_self(rows_a, masses, "absg", k, lev, [allg] * N_ABSBINS, list(range(N_ABSBINS)))
            fine[f"{comp}_{lev}_vs_gk"] = {"rows": rows_g, **{f"chi2_{n}": chi2(rows_g, n) for n in ALL_NAMES},
                                           "chi2_self_null": chi2(rows_g, "self_null")}
            fine[f"{comp}_{lev}_vs_absg"] = {"rows": rows_a, **{f"chi2_{n}": chi2(rows_a, n) for n in ALL_NAMES},
                                             "chi2_self_null": chi2(rows_a, "self_null")}
    res["fine"] = fine

    # ---- (c) width terciles (per component), crossed with the |g| split and |g_k| bins
    terc = {}
    for k, comp in enumerate(COMPONENTS):
        s_post = np.sqrt(tf_var[:, k])
        edges = np.quantile(s_post, [0, 1 / 3, 2 / 3, 1])
        tm = [(s_post >= edges[j]) & (s_post <= edges[j + 1]) if j == 2 else
              (s_post >= edges[j]) & (s_post < edges[j + 1]) for j in range(3)]
        g = truth[:, k]
        entry = {"sigma_edges": [float(e) for e in edges]}
        for lev in LEVELS:
            entry[f"{lev}_overall"] = add_self(
                bin_stats(tm, ["narrow", "middle", "wide"], meas[(comp, lev)], P(comp, lev)),
                masses, "split", k, lev, tm, [None] * 3)
            cross = []
            for j in range(3):
                masks = [tm[j] & inner, tm[j] & outer]
                cross.extend(add_self(bin_stats(masks, [f"T{j+1} |g|<0.05", f"T{j+1} |g|>0.05"],
                                                meas[(comp, lev)], P(comp, lev)),
                                      masses, "split", k, lev, [tm[j], tm[j]], [0, 1]))
            entry[f"{lev}_x_split"] = cross
            per_t = []
            for j in range(3):
                gm = [tm[j] & (g >= gedges[i]) & (g < gedges[i + 1]) for i in range(N_GBINS)]
                rows = bin_stats(gm, [f"{i}" for i in range(N_GBINS)], meas[(comp, lev)], P(comp, lev), gcent)
                add_self(rows, masses, "gk", k, lev, [tm[j]] * N_GBINS, list(range(N_GBINS)))
                per_t.append({"rows": rows, **{f"chi2_{n}": chi2(rows, n) for n in ALL_NAMES},
                              "chi2_self_null": chi2(rows, "self_null")})
            entry[f"{lev}_vs_gk_by_tercile"] = per_t
        entry["median_sigma"] = [float(np.median(s_post[m])) for m in tm]
        entry["frac_ge_sigma_pi"] = [float((s_post[m] >= SIGMA_PI).mean()) for m in tm]
        terc[comp] = entry
    res["terciles"] = terc

    # ---- model-free: coverage conditional on the data (posterior mean ghat_k)
    datacond = {}
    for k, comp in enumerate(COMPONENTS):
        gh = tf_mean[:, k]
        hm = [(gh >= gedges[j]) & (gh < gedges[j + 1]) for j in range(N_GBINS)]
        for lev, L in LEVELS.items():
            rows = bin_stats(hm, [f"{j}" for j in range(N_GBINS)], meas[(comp, lev)],
                             {"nominal": np.full(n_gal, L)}, gcent)
            datacond[f"{comp}_{lev}_vs_ghat"] = {"rows": rows, "chi2_nominal": chi2(rows, "nominal")}
    res["data_conditioned"] = datacond

    # arrays for plotting (small): per-galaxy predictions are not saved
    res["seconds"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(res, indent=1))
    print("wrote", out / "report.json", f"{res['seconds']:.0f}s")
    # sigma_L histogram data for the figure script
    hist = {}
    for comp in COMPONENTS:
        s_post = np.sqrt(tf_var[:, COMPONENTS.index(comp)])
        sl = sig_l_all[comp]
        hist[comp] = {"sigma_post_pct": np.percentile(s_post, [5, 25, 50, 75, 95]).tolist(),
                      "sigma_L_finite_pct": np.percentile(sl[np.isfinite(sl)], [5, 25, 50, 75, 95]).tolist()}
    res["width_percentiles"] = hist
    (out / "report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
