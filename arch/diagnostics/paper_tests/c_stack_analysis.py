#!/usr/bin/env python3
"""Paper test T8 (Group C): stack NPE shear likelihoods at constant shear.

For each observation i (cell, galaxy, realization) the NPE marginal shear
posterior q(g|D_i) is estimated from its (g1, g2) samples (nuisances
marginalized by ignoring them). With the uniform training shear prior,
L_i(g) is proportional to q(g|D_i) inside the box, so the cell posterior is

    log p(g | {D_i}) = sum_i log q_i(g) + const,   g in [-0.1, 0.1]^2.

Density models for q_i:
  kde     weighted Gaussian KDE with scipy's full-covariance Scott bandwidth
          (gaussian_kde; kernel covariance f^2 n_eff^(-1/3) Sigma), binned and
          FFT-convolved on the lattice, reflected at the four box edges
          (kde_grid.py), plus a defensive floor q -> (1 - a) q + a / |box|.
  gauss   moment-matched 2D Gaussian (weighted mean and covariance).
  lapkde  quadratic fit to log KDE over its core (KDE >= 5% of its peak):
          a Gaussian that follows the KDE core but is not truncated by the box.

All log q_i are evaluated at the nodes of a lattice (default 400x400, spacing
0.0005). The binned KDE agrees with the exact gaussian_kde to < 0.5% where the
density exceeds 1% of its peak (kde_grid_validate.py). Stacked posterior means
are lattice sums; MAPs use parabolic sub-node refinement. Grid checks: shifted
401x401 lattice and 800x800 lattice (0.00025).

Weightings: ``train`` (unweighted samples, nuisances under the training
prior, which is how the benchmark nuisances were drawn) and ``tf``
(TF-importance-weighted samples, vcirc under the TF prior).

Writes one npz with all estimates to a NEW paper_tests cache directory.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import time

from multiprocessing import Pool
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kde_grid  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
MODEL = "CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
SAMPLES = S / "cache" / "paper_tests_C_stacking" / MODEL / "likelihood_stack_04_benchmark_r90_8192"
LO, HI = -0.1, 0.1
AREA = (HI - LO) ** 2
N_CELLS, N_GAL, N_REAL = 25, 32, 16
N_LIST = (1, 2, 4, 8, 16, 32)
BOOT = 1000
H_MIN = 0.001
SIG_MIN = 0.002
BOOT_SEED = 1420045  # same seed family as the old 04 benchmark nuisance bootstrap
SHEARS = (-0.08, -0.04, 0.0, 0.04, 0.08)
TRUTH = np.array([(a, b) for a in SHEARS for b in SHEARS])

# density configs: name -> (kind, bandwidth factor, floor alpha)
CONFIGS = {
    "kde": ("kde", 1.0, 1e-3),
    "kde_bw0.5": ("kde", 0.5, 1e-3),
    "kde_bw2": ("kde", 2.0, 1e-3),
    "kde_a1e-4": ("kde", 1.0, 1e-4),
    "kde_a1e-2": ("kde", 1.0, 1e-2),
    "kde_a0": ("kde", 1.0, 0.0),
    "gauss": ("gauss", None, None),
    "lapkde": ("lapkde", 1.0, None),
}


def parse_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=Path, default=SAMPLES)
    p.add_argument("--out", type=Path, default=SAMPLES.parent / "analysis_v1.npz")
    p.add_argument("--cells", default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--boot", type=int, default=BOOT)
    p.add_argument("--no-checks", action="store_true")
    p.add_argument("--workers", type=int, default=5, help="processes for the KDEs")
    return p.parse_args(argv)


def lattice(m, dev):
    d = (HI - LO) / m
    return LO + d * (torch.arange(m, device=dev, dtype=torch.float64) + 0.5)


def boot_counts(boot):
    """Paired bootstrap: rows of galaxy counts, shared by all 25 cells."""
    out = {}
    for n in N_LIST:
        rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, n]))
        ids = rng.integers(0, N_GAL, size=(boot, n))
        c = np.zeros((boot + 1, N_GAL))
        c[0, :n] = 1.0  # deterministic prefix row (first n galaxy IDs)
        for b in range(boot):
            c[b + 1] = np.bincount(ids[b], minlength=N_GAL)
        out[n] = c
    return out


def obs_stats(g, w):
    """g (B,S,2), w (B,S) normalized -> mean, cov, std, n_eff."""
    mu = torch.einsum("bs,bsk->bk", w, g)
    d = g - mu[:, None, :]
    cov = torch.einsum("bs,bsi,bsj->bij", w, d, d)
    neff = 1.0 / torch.sum(w * w, dim=1)
    return mu, cov, torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2)), neff


def _kde_one(job):
    x, w, nodes, f = job
    return kde_grid.density(x, w, nodes, edges=False, factor=f)


def kde_density(pool, g, w, nodes, f=1.0):
    """Reflected full-covariance KDE (kde_grid.density) of every observation on nodes x nodes -> (B,P,P) tensor."""
    gn, wn, nn = g.cpu().numpy(), w.cpu().numpy(), nodes.cpu().numpy()
    out = pool.map(_kde_one, [(gn[b], wn[b], nn, f) for b in range(gn.shape[0])], chunksize=8)
    return torch.as_tensor(np.stack(out), device=g.device, dtype=torch.float64)


def kernel_sd(g, w):
    """(B,2) sd along g1, g2 and (B,) smallest sd of each observation's kernel (before the H_MIN floor)."""
    gn, wn = g.cpu().numpy(), w.cpu().numpy()
    cov = np.stack([kde_grid.scipy_kde(gn[b], wn[b]).covariance for b in range(gn.shape[0])])
    return np.sqrt(np.diagonal(cov, axis1=1, axis2=2)), np.sqrt(np.linalg.eigvalsh(cov)[:, 0])


def gauss_logq(mu, cov, nodes1, nodes2):
    prec = torch.linalg.inv(cov)
    x = nodes1[None, :, None] - mu[:, 0, None, None]
    y = nodes2[None, None, :] - mu[:, 1, None, None]
    q = prec[:, 0, 0, None, None] * x * x + 2 * prec[:, 0, 1, None, None] * x * y + prec[:, 1, 1, None, None] * y * y
    return -0.5 * q - 0.5 * torch.logdet(2 * np.pi * cov)[:, None, None]


def lapkde_fit(dens, mu, std, nodes1, nodes2, chunk=16):
    """Weighted quadratic fit of log KDE on its core; returns coefficients."""
    coefs, oks = [], []
    for s0 in range(0, dens.shape[0], chunk):
        sl = slice(s0, s0 + chunk)
        d = dens[sl]
        b = d.shape[0]
        x = (nodes1[None, :, None] - mu[sl, 0, None, None]) / std[sl, 0, None, None]
        y = (nodes2[None, None, :] - mu[sl, 1, None, None]) / std[sl, 1, None, None]
        x = x.expand(b, -1, d.shape[2]).reshape(b, -1)
        y = y.expand(b, d.shape[1], -1).reshape(b, -1)
        d = d.reshape(b, -1)
        wt = torch.where(d >= 0.05 * d.max(dim=1, keepdim=True).values, d, torch.zeros_like(d))
        wt = wt / wt.sum(1, keepdim=True)
        yv = torch.log(torch.clamp(d, min=1e-300))
        X = torch.stack([torch.ones_like(x), x, y, x * x, x * y, y * y], dim=-1)
        Xw = X * wt[..., None]
        A = torch.bmm(Xw.transpose(1, 2), X)
        rhs = torch.bmm(Xw.transpose(1, 2), yv[..., None])[..., 0]
        coef = torch.linalg.solve(A + 1e-12 * torch.eye(6, device=A.device, dtype=A.dtype), rhs)
        q11, q12, q22 = coef[:, 3], coef[:, 4], coef[:, 5]
        coefs.append(coef)
        oks.append((q11 < 0) & (q22 < 0) & (4 * q11 * q22 - q12 ** 2 > 0))
        del X, Xw
    return torch.cat(coefs), torch.cat(oks)


def lapkde_logq(coef, mu, std, nodes1, nodes2):
    x = (nodes1[None, :, None] - mu[:, 0, None, None]) / std[:, 0, None, None]
    y = (nodes2[None, None, :] - mu[:, 1, None, None]) / std[:, 1, None, None]
    c = [coef[:, k, None, None] for k in range(6)]
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


def summarize(surf, nodes1, nodes2):
    """surf (K,P1,P2) log posterior -> (K,6): mean1, mean2, std1, std2, map1, map2."""
    k, p1, p2 = surf.shape
    s = surf - surf.reshape(k, -1).max(dim=1).values[:, None, None]
    p = torch.exp(s)
    z = p.sum(dim=(1, 2))
    m1 = p.sum(2)
    m2 = p.sum(1)
    mean1 = (m1 * nodes1).sum(1) / z
    mean2 = (m2 * nodes2).sum(1) / z
    sd1 = torch.sqrt(torch.clamp((m1 * nodes1 ** 2).sum(1) / z - mean1 ** 2, min=0))
    sd2 = torch.sqrt(torch.clamp((m2 * nodes2 ** 2).sum(1) / z - mean2 ** 2, min=0))
    flat = torch.argmax(s.reshape(k, -1), dim=1)
    i, j = flat // p2, flat % p2
    ar = torch.arange(k, device=surf.device)

    def refine(im, i0, ip, idx, nodes, n):
        den = im - 2 * i0 + ip
        off = torch.where(den < 0, 0.5 * (im - ip) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
        off = torch.where((idx > 0) & (idx < n - 1), off, torch.zeros_like(off))
        step = nodes[1] - nodes[0]
        return nodes[idx] + off * step

    ic = i.clamp(1, p1 - 2)
    jc = j.clamp(1, p2 - 2)
    map1 = refine(s[ar, ic - 1, j], s[ar, i, j], s[ar, ic + 1, j], i, nodes1, p1)
    map2 = refine(s[ar, i, jc - 1], s[ar, i, j], s[ar, i, jc + 1], j, nodes2, p2)
    return torch.stack([mean1, mean2, sd1, sd2, map1, map2], 1)


def run_settings(lcum, counts, nodes1, nodes2, chunk=250, det_only=False):
    """lcum (G,R,P1,P2) cumulative-over-R log q per galaxy.

    Returns dict: 'Rcurve' (R, K, 6) with N=32; 'N16'/'N1' (len(N_LIST), K, 6).
    """
    g, r, p1, p2 = lcum.shape
    flat = lcum.reshape(g, r, -1)
    dev = lcum.device

    def surf_est(cmat, rr):
        res = []
        for s in range(0, cmat.shape[0], chunk):
            c = torch.as_tensor(cmat[s:s + chunk], device=dev, dtype=torch.float64)
            surf = (c @ flat[:, rr]).reshape(-1, p1, p2)
            res.append(summarize(surf, nodes1, nodes2))
        return torch.cat(res, 0).cpu().numpy()

    sel = (lambda c: c[:1]) if det_only else (lambda c: c)
    out = {"Rcurve": np.stack([surf_est(sel(counts[32]), rr) for rr in range(r)])}
    for rr, key in ((r - 1, "N16"), (0, "N1")):
        out[key] = np.stack([surf_est(sel(counts[n]), rr) for n in N_LIST])
    return out


def main(argv=None):
    args = parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; refusing to overwrite")
    pool = Pool(args.workers)  # before any CUDA call: the workers only run numpy
    dev = torch.device(args.device)
    counts = boot_counts(args.boot)
    cells = range(N_CELLS) if args.cells == "all" else [int(c) for c in args.cells.split(",")]
    n400_1, n400_2 = lattice(400, dev), lattice(400, dev)
    n401 = lattice(401, dev)
    n800 = lattice(800, dev)
    results: dict[str, list] = {}
    per_obs: dict[str, list] = {}

    def put(store, key, val):
        store.setdefault(key, []).append(val)

    for cell in cells:
        t0 = time.time()
        z = np.load(args.samples / f"cell{cell:02d}.npz")
        g_np = z["g"]  # (G,R,S,2)
        G, R, Sn, _ = g_np.shape
        g = torch.as_tensor(g_np.reshape(G * R, Sn, 2), device=dev, dtype=torch.float64)
        logw = torch.as_tensor(z["tf_log_weight"].reshape(G * R, Sn), device=dev, dtype=torch.float64)
        put(per_obs, "tf_ess", z["tf_ess"])
        for wname in ("train", "tf"):
            if wname == "train":
                w = torch.full((G * R, Sn), 1.0 / Sn, device=dev, dtype=torch.float64)
            else:
                w = torch.softmax(logw, dim=1)
            mu, cov, std, neff = obs_stats(g, w)
            put(per_obs, f"{wname}_mu", mu.reshape(G, R, 2).cpu().numpy())
            put(per_obs, f"{wname}_std", std.reshape(G, R, 2).cpu().numpy())
            put(per_obs, f"{wname}_corr", (cov[:, 0, 1] / (std[:, 0] * std[:, 1])).reshape(G, R).cpu().numpy())
            put(per_obs, f"{wname}_neff", neff.reshape(G, R).cpu().numpy())
            # guards against degenerate (ESS ~ 1) TF weights: kernel sd >= 0.001 (kde_grid.H_MIN), sigma >= 0.002
            h_axes, h_min = kernel_sd(g, w)
            var_floor = torch.clamp(SIG_MIN ** 2 - torch.diagonal(cov, dim1=1, dim2=2), min=0.0)
            cov = cov + torch.diag_embed(var_floor)
            std_g = torch.sqrt(torch.diagonal(cov, dim1=1, dim2=2))
            put(per_obs, f"{wname}_guarded", ((std < SIG_MIN).any(1).cpu().numpy() | (h_min < H_MIN)).reshape(G, R))
            put(per_obs, f"{wname}_h", h_axes.reshape(G, R, 2))
            dens_cache = {}
            for f in (0.5, 1.0, 2.0):
                dens_cache[f] = kde_density(pool, g, w, n400_1, f)
            put(per_obs, f"{wname}_kde_mass", (dens_cache[1.0].sum((1, 2)) * (0.2 / 400) ** 2).reshape(G, R).cpu().numpy())
            coef, ok = lapkde_fit(dens_cache[1.0], mu, std_g, n400_1, n400_2)
            put(per_obs, f"{wname}_lapkde_ok", ok.reshape(G, R).cpu().numpy())
            put(per_obs, f"{wname}_lapkde_coef", coef.reshape(G, R, 6).cpu().numpy())
            for cname, (kind, f, a) in CONFIGS.items():
                if kind == "kde":
                    lq = torch.log((1 - a) * dens_cache[f] + a / AREA) if a > 0 else torch.log(torch.clamp(dens_cache[f], min=1e-300))
                elif kind == "gauss":
                    lq = gauss_logq(mu, cov, n400_1, n400_2)
                else:
                    lq_l = lapkde_logq(coef, mu, std_g, n400_1, n400_2)
                    lq_g = gauss_logq(mu, cov, n400_1, n400_2)
                    lq = torch.where(ok[:, None, None], lq_l, lq_g)
                lq = lq - lq.reshape(G * R, -1).max(1).values[:, None, None]
                lcum = torch.cumsum(lq.reshape(G, R, 400, 400), dim=1)
                del lq
                res = run_settings(lcum, counts, n400_1, n400_2)
                for key, val in res.items():
                    put(results, f"{wname}/{cname}/{key}", val)
                if cname == "gauss":
                    # analytic untruncated Gaussian product (N=32, R=16) cross-check
                    prec = torch.linalg.inv(cov)
                    ptot = prec.sum(0)
                    mtot = torch.linalg.solve(ptot, torch.einsum("bij,bj->i", prec, mu))
                    put(results, f"{wname}/gauss/analytic_N32R16", mtot.cpu().numpy())
                del lcum
            del dens_cache
            if not args.no_checks:
                # (a) shifted 401 lattice, (b) 800 lattice: KDE f=1, a=1e-3, det rows only
                for tag, nodes in (("m401", n401), ("m800", n800)):
                    dens = kde_density(pool, g, w, nodes)
                    lq = torch.log((1 - 1e-3) * dens + 1e-3 / AREA)
                    del dens
                    lq = lq - lq.reshape(G * R, -1).max(1).values[:, None, None]
                    m = nodes.shape[0]
                    lcum = torch.cumsum(lq.reshape(G, R, m, m), dim=1)
                    del lq
                    res = run_settings(lcum, counts, nodes, nodes, chunk=1, det_only=True)
                    for key, val in res.items():
                        put(results, f"{wname}/kde_{tag}/{key}", val)
                    del lcum
                    torch.cuda.empty_cache()
        print(f"cell {cell} done {time.time() - t0:.1f}s", flush=True)

    pool.close()
    payload = {k: np.stack(v) for k, v in results.items()}
    payload.update({f"obs_{k}": np.stack(v) for k, v in per_obs.items()})
    for n, c in counts.items():
        payload[f"counts_N{n}"] = c
    payload["truth"] = TRUTH[list(cells)]
    payload["meta_json"] = np.array(json.dumps({
        "schema": "paper-tests-C-stacking-analysis-v1", "samples": str(args.samples),
        "configs": CONFIGS, "kde": "kde_grid: scipy gaussian_kde Scott full covariance, binned + FFT, reflected",
        "N_list": N_LIST, "boot": args.boot, "boot_seed": BOOT_SEED,
        "summary_columns": ["mean1", "mean2", "std1", "std2", "map1", "map2"],
        "row0": "deterministic (all galaxies / first N)", "rows1+": "paired bootstrap",
        "lattice": "400x400 node centers, spacing 0.0005, box [-0.1,0.1]^2",
        "cells": list(cells)}))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
