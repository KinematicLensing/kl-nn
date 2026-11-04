#!/usr/bin/env python3
"""H: cluster lensing test -- likelihood stacking in cells, halo fit and kappa maps.

Inputs (from h_cluster_make.py and the NPE caches): the manifest of galaxy
positions and true shears, and for each scenario ("main", "sub") either
  --source npe        the 1M NPE posterior cache of that scenario's dataset, or
  --source synthetic  Gaussian likelihoods with each galaxy's real NPE shear
                      covariance (from its Xu3 source galaxy), centred on a noisy
                      copy of its true shear; the same noise in both scenarios.

Per galaxy, the shear likelihood L_i(g) = q_i(g) / pi_train(g) is a TF-weighted
2D KDE of its posterior samples on a grid over the training box (per-axis Scott
bandwidth, reflection at the box edges, density floor), as in c_stack_analysis.py.

1. Cells. Polar: a central disk plus log-spaced annuli, each split into sectors so
   that every cell holds ~5 galaxies on average; stacked in (g_t, g_x) about the
   main-halo centre. Cartesian: square cells of the same mean occupancy, stacked
   in (g1, g2). All non-empty cells are kept. The cell posterior is the product of
   its galaxies' likelihoods for one common shear (uniform prior on |G| < 0.1).
   Reference truth: (sum F_i)^-1 sum F_i g_i, F_i the inverse covariance of
   galaxy i's likelihood in the cell frame.
2. Bias. Per-cell pulls and linear / cubic fits of the stacked minus reference
   shear against the reference, weighted by the cell variance.
3. Halo fit (Nautilus). log10 M_main [+ log10 M_sub] with Duffy+08
   concentrations, centres fixed at their true positions; wide log-uniform
   priors. L = prod_i L_i(g_model(x_i)). Evidence for main vs main+sub.
4. Kappa maps. (a) Kaiser-Squires on the Cartesian cell shears; (b) a
   likelihood-based map: kappa on a pixel grid, gamma from kappa by the KS
   kernel, g = gamma / (1 - kappa) at each galaxy, maximizing
   sum_i log L_i(g_i) + log smoothness prior (MAP, torch). Residual = (b) minus
   the best-fit main-halo-only model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h_cluster_make import build_halos, duffy08_c200c, lensing  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_1m_frozen_s42_measaxis"
XU3_CACHE = S / "cache" / MODEL / "test_100k_simv3_cosi_xu3_tf_measaxis_testset_tfweighted_v2_10k_s42_righthanded"
LO, HI = -0.1, 0.1
AREA = (HI - LO) ** 2
FLOOR = 1e-3
NGRID = 401
GRID = np.linspace(LO, HI, NGRID)
DG = GRID[1] - GRID[0]
SCENARIOS = ("main", "sub")


# --------------------------------------------------------------------------- inputs
def load_npe(name, tag):
    root = S / "cache" / MODEL / f"{name}_{tag}"
    sample = np.load(next((root / "sample").glob("*.npy")))
    logw = np.load(next((root / "posterior_tf_log_weight").glob("*.npy")))
    truth = np.load(next((root / "truth").glob("*.npy")))
    return sample[..., :2].astype(np.float64), logw.astype(np.float64), truth[:, :2]


def xu3_shear_covariance(source_ids):
    """TF-weighted (g1, g2) posterior covariance of Xu3 galaxies in the 1M cache."""
    out = np.empty((len(source_ids), 2, 2))
    for part in np.unique(source_ids // 1000):
        sel = np.flatnonzero(source_ids // 1000 == part)
        label = f"part{part}of100.npy"
        smp = np.load(XU3_CACHE / "shear_sample" / label, mmap_mode="r")
        lw = np.load(XU3_CACHE / "posterior_tf_log_weight" / label, mmap_mode="r")
        for i in sel:
            j = source_ids[i] % 1000
            x = np.asarray(smp[j], float)
            w = np.exp(lw[j] - lw[j].max()); w /= w.sum()
            mu = w @ x
            out[i] = (x - mu).T @ ((x - mu) * w[:, None])
    return out


def synthetic(truths, covs, seed, nsamp=6000):
    """Posterior samples for a Gaussian likelihood times the training box."""
    rng = np.random.default_rng(seed)
    n = truths[SCENARIOS[0]].shape[0]
    noise = np.stack([rng.multivariate_normal([0, 0], covs[i]) for i in range(n)])
    out = {}
    for sc in SCENARIOS:
        obs = truths[sc] + noise
        smp = np.empty((n, nsamp, 2))
        for i in range(n):
            L = np.linalg.cholesky(covs[i])
            acc = []
            while sum(len(a) for a in acc) < nsamp:
                d = obs[i] + rng.standard_normal((4 * nsamp, 2)) @ L.T
                acc.append(d[np.all(np.abs(d) < HI, axis=1)])
            smp[i] = np.concatenate(acc)[:nsamp]
        out[sc] = (smp, np.zeros((n, nsamp)))
    return out


# --------------------------------------------------------------------------- likelihoods
def kde_loglike_grid(samples, logw):
    """log L_i on GRID x GRID (axis 0 = g1, axis 1 = g2), shape (n, NGRID, NGRID)."""
    n = samples.shape[0]
    out = np.empty((n, NGRID, NGRID), dtype=np.float32)
    for i in range(n):
        w = np.exp(logw[i] - logw[i].max()); w /= w.sum()
        x = samples[i]
        mu = w @ x
        sd = np.sqrt(w @ (x - mu) ** 2)
        neff = 1.0 / np.sum(w * w)
        h = np.maximum(sd * neff ** (-1.0 / 6.0), 1e-3)
        K = []
        for k in range(2):
            d = GRID[:, None]
            xs = x[None, :, k]
            kern = (np.exp(-0.5 * ((d - xs) / h[k]) ** 2)
                    + np.exp(-0.5 * ((d + xs - 2 * LO) / h[k]) ** 2)
                    + np.exp(-0.5 * ((d + xs - 2 * HI) / h[k]) ** 2)) / (h[k] * math.sqrt(2 * math.pi))
            K.append(kern)
        dens = (K[0] * w[None, :]) @ K[1].T
        out[i] = np.log((1.0 - FLOOR) * dens + FLOOR / AREA)
    return out


def interp(logl, idx, g1, g2):
    """Bilinear log L of galaxies idx at shears (g1, g2); arrays broadcast together."""
    u = np.clip((g1 - LO) / DG, 0, NGRID - 1 - 1e-9)
    v = np.clip((g2 - LO) / DG, 0, NGRID - 1 - 1e-9)
    i0, j0 = u.astype(int), v.astype(int)
    tu, tv = u - i0, v - j0
    return ((1 - tu) * (1 - tv) * logl[idx, i0, j0] + tu * (1 - tv) * logl[idx, i0 + 1, j0]
            + (1 - tu) * tv * logl[idx, i0, j0 + 1] + tu * tv * logl[idx, i0 + 1, j0 + 1])


def grid_moments(logl):
    """Mean and covariance of each galaxy's normalized likelihood in (g1, g2)."""
    G1, G2 = np.meshgrid(GRID, GRID, indexing="ij")
    mu = np.empty((logl.shape[0], 2)); cov = np.empty((logl.shape[0], 2, 2))
    for i in range(logl.shape[0]):
        p = np.exp(logl[i] - logl[i].max()); p /= p.sum()
        m1, m2 = (p * G1).sum(), (p * G2).sum()
        d1, d2 = G1 - m1, G2 - m2
        mu[i] = m1, m2
        cov[i] = [[(p * d1 * d1).sum(), (p * d1 * d2).sum()], [(p * d1 * d2).sum(), (p * d2 * d2).sum()]]
    return mu, cov


# --------------------------------------------------------------------------- cells
def tangential_matrix(phi):
    """R with (g_t, g_x) = R (g1, g2); g_t = -Re(g e^{-2i phi}), g_x = -Im(g e^{-2i phi})."""
    c, s = np.cos(2 * phi), np.sin(2 * phi)
    return -np.array([[c, s], [-s, c]])


def polar_cells(man, cfg, per_cell=5.0, n_annuli=4):
    n = cfg["density_per_arcmin2"]
    kpm = cfg["kpc_per_arcsec"] / 1000.0
    mpc2_per_arcmin2 = (60 * kpm) ** 2
    r0 = math.sqrt(per_cell / n * mpc2_per_arcmin2 / math.pi)  # disk holding ~per_cell galaxies
    edges = np.geomspace(r0, cfg["r_max_mpc"], n_annuli + 1)
    phi_sub = math.atan2(cfg["sub_y_mpc"], cfg["sub_x_mpc"])
    cells = [dict(kind="disk", r_lo=0.0, r_hi=r0, phi_lo=-np.pi, phi_hi=np.pi)]
    for a, b in zip(edges[:-1], edges[1:]):
        expected = n * math.pi * (b * b - a * a) / mpc2_per_arcmin2
        ns = max(1, int(round(expected / per_cell)))
        width = 2 * np.pi / ns
        for k in range(ns):
            lo = phi_sub - width / 2 + k * width  # the subhalo direction sits at a sector centre
            cells.append(dict(kind="sector", r_lo=a, r_hi=b, phi_lo=lo, phi_hi=lo + width))
    member = np.full(len(man), -1)
    r, phi = man["r_mpc"].to_numpy(), man["phi"].to_numpy()
    for c, cell in enumerate(cells):
        dphi = (phi - cell["phi_lo"]) % (2 * np.pi)
        inside = (r >= cell["r_lo"]) & (r < cell["r_hi"]) & (dphi < cell["phi_hi"] - cell["phi_lo"])
        member[inside] = c
    return cells, member


def cartesian_cells(man, cfg, per_cell=5.0):
    n = cfg["density_per_arcmin2"]
    kpm = cfg["kpc_per_arcsec"] / 1000.0
    side = math.sqrt(per_cell / n) * 60 * kpm  # Mpc
    R = cfg["r_max_mpc"]
    nside = int(math.ceil(2 * R / side))
    edges = (np.arange(nside + 1) - nside / 2) * side
    ix = np.digitize(man["x_mpc"], edges) - 1
    iy = np.digitize(man["y_mpc"], edges) - 1
    member = ix * nside + iy
    cells = [dict(kind="square", ix=i, iy=j, x_lo=edges[i], x_hi=edges[i + 1],
                  y_lo=edges[j], y_hi=edges[j + 1]) for i in range(nside) for j in range(nside)]
    return cells, member, edges


def stack_cell(logl, idx, frames):
    """Posterior of a common cell shear G (in the cell frame) on a disk grid.

    frames[i] maps (g1, g2) of galaxy idx[i] to the cell frame; G -> (g1, g2) is its
    inverse (orthogonal: the transpose).
    """
    m = 201
    gg = np.linspace(LO, HI, m)
    G1, G2 = np.meshgrid(gg, gg, indexing="ij")
    ok = np.hypot(G1, G2) < HI
    lp = np.zeros_like(G1)
    for i, R in zip(idx, frames):
        g = np.einsum("ji,jab->iab", R, np.stack([G1, G2]))  # R^T G
        lp += interp(logl, i, g[0], g[1])
    lp = np.where(ok, lp, -np.inf)
    p = np.exp(lp - lp[ok].max()); p /= p.sum()
    mean = np.array([(p * G1).sum(), (p * G2).sum()])
    d = np.stack([G1 - mean[0], G2 - mean[1]])
    cov = np.einsum("aij,bij,ij->ab", d, d, p)
    return mean, cov


def analyse_cells(logl, cov_gal, truths, member, ncell, frame_fn):
    rows = []
    for c in range(ncell):
        idx = np.flatnonzero(member == c)
        if len(idx) == 0:
            continue
        frames = [frame_fn(i) for i in idx]
        mean, cov = stack_cell(logl, idx, frames)
        F = [np.linalg.inv(R @ cov_gal[i] @ R.T) for i, R in zip(idx, frames)]
        gt = [R @ truths[i] for i, R in zip(idx, frames)]
        ref = np.linalg.solve(np.sum(F, axis=0), np.sum([f @ g for f, g in zip(F, gt)], axis=0))
        rows.append(dict(cell=c, n=len(idx), est=mean.tolist(), sd=np.sqrt(np.diag(cov)).tolist(),
                         ref=ref.tolist(), plain_mean=np.mean(gt, axis=0).tolist()))
    return rows


def bias_fits(rows):
    out = {}
    for k in (0, 1):
        g = np.array([r["ref"][k] for r in rows]); e = np.array([r["est"][k] for r in rows])
        s = np.array([r["sd"][k] for r in rows]); w = 1 / s ** 2
        res = {}
        for name, X in (("linear", np.column_stack([np.ones_like(g), g])),
                        ("cubic", np.column_stack([np.ones_like(g), g, g ** 3]))):
            W = np.diag(w); cov = np.linalg.inv(X.T @ W @ X); b = cov @ X.T @ W @ (e - g)
            chi2 = float(((e - g - X @ b) ** 2 * w).sum())
            res[name] = dict(c=b[0], c_err=math.sqrt(cov[0, 0]), m=b[1], m_err=math.sqrt(cov[1, 1]),
                             chi2=chi2, dof=len(g) - X.shape[1])
            if name == "cubic":
                res[name].update(b3=b[2], b3_err=math.sqrt(cov[2, 2]))
        pull = (e - g) / s
        res["pull_mean"], res["pull_std"] = float(pull.mean()), float(pull.std())
        out[f"component{k + 1}"] = res
    return out


# --------------------------------------------------------------------------- halo fit
def halo_fit(logl, man, cfg, with_sub):
    from nautilus import Prior, Sampler

    import galsim
    x, y = man["x_arcsec"].to_numpy(), man["y_arcsec"].to_numpy()
    h, om, zl, zs = cfg["h"], cfg["omega_m"], cfg["z_l"], cfg["z_s"]
    idx = np.arange(len(man))
    prior = Prior()
    prior.add_parameter("log10_m_main", dist=(13.5, 16.0))
    if with_sub:
        prior.add_parameter("log10_m_sub", dist=(12.0, 15.5))

    def loglike(par):
        halos = []
        for key, pos in (("log10_m_main", (0.0, 0.0)), ("log10_m_sub", (cfg["sub_x_arcsec"], cfg["sub_y_arcsec"]))):
            if key not in par:
                continue
            m = 10 ** par[key]
            halos.append(galsim.NFWHalo(m * h, duffy08_c200c(m, zl, h), zl,
                                        halo_pos=galsim.PositionD(*pos), omega_m=om, omega_lam=1 - om))
        g1, g2, _ = lensing(halos, x, y, zs)
        inside = (np.abs(g1) < HI) & (np.abs(g2) < HI)
        ll = np.where(inside, interp(logl, idx, np.clip(g1, LO, HI), np.clip(g2, LO, HI)),
                      math.log(FLOOR / AREA))
        return float(ll.sum())

    sampler = Sampler(prior, loglike, n_live=500, seed=11)
    sampler.run(verbose=False)
    pts, logw, _ = sampler.posterior()
    w = np.exp(logw - logw.max()); w /= w.sum()
    names = prior.keys
    summ = {}
    for j, nm in enumerate(names):
        o = np.argsort(pts[:, j]); cw = np.cumsum(w[o])
        q = [float(pts[o, j][np.searchsorted(cw, t)]) for t in (0.16, 0.5, 0.84)]
        summ[nm] = dict(q16=q[0], median=q[1], q84=q[2])
    return dict(log_z=float(sampler.log_z), params=summ, samples=pts.tolist() if len(pts) < 5000 else None)


# --------------------------------------------------------------------------- kappa maps
def ks_kernel(n, pix):
    """Fourier-space KS kernels for a padded n x n grid with pixel size pix."""
    k1 = np.fft.fftfreq(n, d=pix)[:, None] * np.ones((1, n))
    k2 = np.fft.fftfreq(n, d=pix)[None, :] * np.ones((n, 1))
    ksq = k1 ** 2 + k2 ** 2
    ksq[0, 0] = 1.0
    D1 = (k1 ** 2 - k2 ** 2) / ksq
    D2 = 2 * k1 * k2 / ksq
    D1[0, 0] = D2[0, 0] = 0.0
    return D1, D2


def kaiser_squires(g1map, g2map, pix, smooth_pix=1.0):
    """kappa from a shear map on a regular grid (axis 0 = x, axis 1 = y), zero-padded x2."""
    n0 = g1map.shape[0]; n = 2 * n0
    G1 = np.zeros((n, n)); G2 = np.zeros((n, n))
    G1[:n0, :n0], G2[:n0, :n0] = g1map, g2map
    D1, D2 = ks_kernel(n, pix)
    kap = np.real(np.fft.ifft2(D1 * np.fft.fft2(G1) + D2 * np.fft.fft2(G2)))[:n0, :n0]
    if smooth_pix > 0:
        from scipy.ndimage import gaussian_filter
        kap = gaussian_filter(kap, smooth_pix)
    return kap


def likelihood_kappa(logl, man, cfg, pix_mpc=0.5, sigma_kappa=0.1, smooth_len=1.0, steps=400):
    """MAP kappa on a pixel grid from the individual galaxy likelihoods (torch, CPU)."""
    R = cfg["r_max_mpc"]
    n0 = int(math.ceil(2 * R / pix_mpc)); n = 2 * n0
    edges = (np.arange(n0 + 1) - n0 / 2) * pix_mpc
    centres = 0.5 * (edges[1:] + edges[:-1])
    D1, D2 = ks_kernel(n, pix_mpc)
    # gamma = forward KS: gamma_hat = D * kappa_hat with D = (D1 + i D2)
    D1t, D2t = torch.tensor(D1), torch.tensor(D2)
    L = torch.tensor(logl.astype(np.float64))
    gx = torch.tensor((man["x_mpc"].to_numpy() - edges[0]) / pix_mpc - 0.5)
    gy = torch.tensor((man["y_mpc"].to_numpy() - edges[0]) / pix_mpc - 0.5)
    gidx = torch.arange(len(man))
    kap = torch.zeros((n0, n0), dtype=torch.float64, requires_grad=True)

    def bilinear(field, u, v):
        u = u.clamp(0, n0 - 1 - 1e-9); v = v.clamp(0, n0 - 1 - 1e-9)
        i0, j0 = u.long(), v.long(); tu, tv = u - i0, v - j0
        return ((1 - tu) * (1 - tv) * field[i0, j0] + tu * (1 - tv) * field[i0 + 1, j0]
                + (1 - tu) * tv * field[i0, j0 + 1] + tu * tv * field[i0 + 1, j0 + 1])

    def loglike_grid(g1, g2):
        u = ((g1 - LO) / DG).clamp(0, NGRID - 1 - 1e-9); v = ((g2 - LO) / DG).clamp(0, NGRID - 1 - 1e-9)
        i0, j0 = u.long(), v.long(); tu, tv = u - i0, v - j0
        return ((1 - tu) * (1 - tv) * L[gidx, i0, j0] + tu * (1 - tv) * L[gidx, i0 + 1, j0]
                + (1 - tu) * tv * L[gidx, i0, j0 + 1] + tu * tv * L[gidx, i0 + 1, j0 + 1])

    def objective():
        Kf = torch.fft.fft2(torch.nn.functional.pad(kap, (0, n - n0, 0, n - n0)))
        g1 = torch.real(torch.fft.ifft2(D1t * Kf))[:n0, :n0]
        g2 = torch.real(torch.fft.ifft2(D2t * Kf))[:n0, :n0]
        k_at = bilinear(kap, gx, gy)
        r1 = bilinear(g1, gx, gy) / (1 - k_at); r2 = bilinear(g2, gx, gy) / (1 - k_at)
        ll = loglike_grid(r1.clamp(LO, HI), r2.clamp(LO, HI)).sum()
        grad2 = ((kap[1:] - kap[:-1]) ** 2).sum() + ((kap[:, 1:] - kap[:, :-1]) ** 2).sum()
        lp = -0.5 * (kap ** 2).sum() / sigma_kappa ** 2 - 0.5 * grad2 * smooth_len ** 2 / sigma_kappa ** 2
        return -(ll + lp)

    opt = torch.optim.LBFGS([kap], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(); f = objective(); f.backward(); return f

    opt.step(closure)
    return kap.detach().numpy(), centres


def model_kappa(cfg, centres, m_main, m_sub=None):
    import galsim
    h, om, zl, zs = cfg["h"], cfg["omega_m"], cfg["z_l"], cfg["z_s"]
    X, Y = np.meshgrid(centres, centres, indexing="ij")
    xa = X.ravel() * 1000 / cfg["kpc_per_arcsec"]; ya = Y.ravel() * 1000 / cfg["kpc_per_arcsec"]
    kap = np.zeros_like(xa)
    for m, pos in ((m_main, (0.0, 0.0)), (m_sub, (cfg["sub_x_arcsec"], cfg["sub_y_arcsec"]))):
        if m is None:
            continue
        halo = galsim.NFWHalo(m * h, duffy08_c200c(m, zl, h), zl, halo_pos=galsim.PositionD(*pos),
                              omega_m=om, omega_lam=1 - om)
        kap += np.asarray(halo.getConvergence((xa, ya), zs))
    return kap.reshape(X.shape)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_r0")
    p.add_argument("--source", choices=("npe", "synthetic"), required=True)
    p.add_argument("--cache-tag", default="r0")
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    cfg = json.loads((S / "samples" / f"paper_tests_H_cluster_{a.tag}_config.json").read_text())
    man = pd.read_csv(S / "samples" / f"paper_tests_H_cluster_{a.tag}_manifest.csv")
    out = a.out or S / "reports/paper-tests-1m/H_cluster" / a.tag / a.source
    out.mkdir(parents=True, exist_ok=True)
    truths = {sc: man[[f"g1_{sc}", f"g2_{sc}"]].to_numpy() for sc in SCENARIOS}

    if a.source == "synthetic":
        covs = xu3_shear_covariance(man["source_id"].to_numpy())
        draws = synthetic(truths, covs, seed=cfg["seed"] + 1)
    else:
        draws = {}
        for sc in SCENARIOS:
            smp, lw, tr = load_npe(f"paper_tests_H_cluster_{a.tag}_{sc}", a.cache_tag)
            if not np.allclose(tr, truths[sc], atol=1e-6):
                raise SystemExit(f"{sc}: cache truth does not match the manifest")
            draws[sc] = (smp, lw)

    report = dict(tag=a.tag, source=a.source, n_galaxies=len(man), config=cfg)
    phi = man["phi"].to_numpy()
    logls, covs_gal = {}, {}
    for sc in SCENARIOS:
        logls[sc] = kde_loglike_grid(*draws[sc])
        _, covs_gal[sc] = grid_moments(logls[sc])
    np.savez_compressed(out / "loglike_grids.npz", **{sc: logls[sc] for sc in SCENARIOS}, grid=GRID)
    report["median_sigma_g"] = float(np.median(np.sqrt(0.5 * np.trace(covs_gal["main"], axis1=1, axis2=2))))

    pcells, pmem = polar_cells(man, cfg)
    ccells, cmem, cedges = cartesian_cells(man, cfg)
    report["cells"] = dict(polar=dict(n_cells=len(pcells), occupied=int(len(set(pmem[pmem >= 0])))),
                           cartesian=dict(n_cells=len(ccells), occupied=int(len(set(cmem))),
                                          side_mpc=float(cedges[1] - cedges[0])))
    results = {}
    for sc in SCENARIOS:
        pol = analyse_cells(logls[sc], covs_gal[sc], truths[sc], pmem, len(pcells),
                            lambda i: tangential_matrix(phi[i]))
        car = analyse_cells(logls[sc], covs_gal[sc], truths[sc], cmem, len(ccells),
                            lambda i: np.eye(2))
        results[sc] = dict(polar=pol, cartesian=car, polar_bias=bias_fits(pol), cartesian_bias=bias_fits(car))
        print(sc, "polar g_t", json.dumps(results[sc]["polar_bias"]["component1"]["linear"]),
              "| cartesian g1", json.dumps(results[sc]["cartesian_bias"]["component1"]["linear"]), flush=True)
    report["cells_results"] = results

    if not a.no_fit:
        fits = {}
        for sc in SCENARIOS:
            fits[sc] = {mdl: halo_fit(logls[sc], man, cfg, with_sub=(mdl == "main+sub"))
                        for mdl in ("main", "main+sub")}
            fits[sc]["ln_bayes_sub_vs_main"] = fits[sc]["main+sub"]["log_z"] - fits[sc]["main"]["log_z"]
            print(sc, "halo fit", json.dumps({k: (v["params"] if isinstance(v, dict) else v) for k, v in fits[sc].items()}), flush=True)
        report["halo_fit"] = fits

    # kappa maps
    maps = {}
    for sc in SCENARIOS:
        kap, centres = likelihood_kappa(logls[sc], man, cfg)
        maps[sc] = kap
    report["kappa_pixel_mpc"] = 0.5
    truth_k = {sc: model_kappa(cfg, centres, cfg["m_main"], cfg["m_sub"] if sc == "sub" else None) for sc in SCENARIOS}
    if "halo_fit" in report:
        m_fit = 10 ** report["halo_fit"]["sub"]["main"]["params"]["log10_m_main"]["median"]
    else:
        m_fit = cfg["m_main"]
    resid = maps["sub"] - model_kappa(cfg, centres, m_fit)
    # KS on Cartesian cell shears
    nside = len(cedges) - 1
    ks = {}
    for sc in SCENARIOS:
        g1m = np.zeros((nside, nside)); g2m = np.zeros((nside, nside))
        for row in results[sc]["cartesian"]:
            cell = ccells[row["cell"]]
            g1m[cell["ix"], cell["iy"]], g2m[cell["ix"], cell["iy"]] = row["est"]
        ks[sc] = kaiser_squires(g1m, g2m, float(cedges[1] - cedges[0]), smooth_pix=0.7)
    np.savez_compressed(out / "kappa_maps.npz", centres=centres, lik_main=maps["main"], lik_sub=maps["sub"],
                        resid=resid, truth_main=truth_k["main"], truth_sub=truth_k["sub"],
                        ks_main=ks["main"], ks_sub=ks["sub"], ks_edges=cedges)
    plot_maps(out, cfg, man, centres, maps, resid, truth_k, ks, cedges)
    plot_cells(out, results, pcells, ccells, man, cfg)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"wrote {out}")


def plot_maps(out, cfg, man, centres, maps, resid, truth_k, ks, cedges):
    R = cfg["r_max_mpc"]; ext = [centres[0] - 0.25, centres[-1] + 0.25] * 2
    vmax = max(np.abs(truth_k["sub"]).max(), 0.03) * 0.6
    panels = [("truth: main", truth_k["main"]), ("truth: main + sub", truth_k["sub"]),
              ("likelihood map: main", maps["main"]), ("likelihood map: main + sub", maps["sub"]),
              ("residual (sub data − best-fit main)", resid),
              ("KS on cells: main + sub", ks["sub"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (title, img) in zip(axes.ravel(), panels):
        e = [cedges[0], cedges[-1]] * 2 if title.startswith("KS") else ext
        im = ax.imshow(img.T, origin="lower", extent=e, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="0.5"))
        ax.plot([0, cfg["sub_x_mpc"]], [0, cfg["sub_y_mpc"]], "kx", ms=7)
        ax.plot(man["x_mpc"], man["y_mpc"], ".", ms=1.5, color="0.3")
        ax.set_title(title, fontsize=9); ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]")
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\kappa$")
    fig.tight_layout(); fig.savefig(out / "kappa_maps.png", dpi=130); plt.close(fig)


def plot_cells(out, results, pcells, ccells, man, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row_i, sc in enumerate(SCENARIOS):
        for col, kind in enumerate(("polar", "cartesian")):
            ax = axes[row_i, col]
            rows = results[sc][kind]
            e = np.array([r["est"] for r in rows]); g = np.array([r["ref"] for r in rows])
            s = np.array([r["sd"] for r in rows])
            for k, (mk, col_) in enumerate((("o", "C0"), ("s", "C3"))):
                ax.errorbar(g[:, k], e[:, k] - g[:, k], yerr=s[:, k], fmt=mk, color=col_, ms=4,
                            label=("g_t" if kind == "polar" else "g1") if k == 0 else ("g_x" if kind == "polar" else "g2"))
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xlabel("cell reference shear"); ax.set_ylabel("stacked − reference")
            ax.set_title(f"{sc}: {kind} cells ({len(rows)} occupied)", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "cell_bias.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
