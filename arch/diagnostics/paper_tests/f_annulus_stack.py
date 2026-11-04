#!/usr/bin/env python3
"""Annulus (cluster) likelihood stacking on an Xu3 posterior cache.

Every Xu3 test galaxy has a random true shear g_i. That shear is exactly a
tangential shear of amplitude g_t = |g_i| for a galaxy at the position angle
set by its direction, so the test set can stand in for sources behind a
cluster with no new simulations.

For an NFW lens (Planck18; default M200 = 1e15 Msun, c200 = 4, z_l = 0.1,
sources at z_s = 0.3, the redshift all galaxies are rendered at), the radial
range [r_min, r_max] is split into log-spaced annuli. Each annulus takes the
galaxies whose |g_i| lies within the g_t range the profile spans across it.

Per annulus, the amplitude is estimated from the product of the galaxies'
likelihoods along their own tangential directions e_t,i = g_i / |g_i|:

    p(g_t) ∝ prod_i L_i(g_t e_t,i),   L_i ∝ q(g | data_i) / pi_train(g),

The training shear prior is uniform on the box |g_k| < 0.1, so L_i ∝ q_i
there, and |g_t| < 0.1 keeps every evaluation point inside the box.

Each q_i is a TF-weighted 2D Gaussian KDE of the galaxy's posterior samples:
per-axis Scott bandwidth sigma_k * n_eff^(-1/6), reflection at the box edges,
and a floor, as in c_stack_analysis.py. The cross component, along the
direction rotated 45 degrees in shear angle, is a null test. It cuts through
each galaxy's true shear, g = |g_i| e_t,i + x e_x,i, so a non-zero stacked
x-hat means the shear direction is systematically rotated.

The reference truth per annulus is the mean of the true g_t weighted by
1 / sigma_t,i^2, where sigma_t,i is the width of L_i along e_t,i. m and c come
from a fit across annuli. Errors come from bootstrap resampling of galaxies
within annuli.

--synthetic replaces each galaxy's samples with draws from a Gaussian
likelihood (centred on a noisy copy of its true shear) times the box prior. It
checks the estimator end to end; there m should be about 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lensing_profiles import nfw_reduced_tangential_shear, r200_mpc  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
CANONICAL_CACHE = (
    S / "cache/CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair"
    / "test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded"
)
LO, HI = -0.1, 0.1
AREA = (HI - LO) ** 2
H_MIN, SIG_MIN = 1e-3, 2e-3


def parts(cache: Path, name: str):
    files = sorted((cache / name).glob("part*of*.npy"), key=lambda p: int(p.stem[4:].split("of")[0]))
    if not files:
        raise SystemExit(f"no parts in {cache / name}")
    return files


def annuli(args):
    edges = np.geomspace(args.rmin, args.rmax, args.nbins + 1)
    g_edges, _, _ = nfw_reduced_tangential_shear(
        edges, m200=args.m200, c200=args.c200, z_lens=args.zl, z_source=args.zs)
    rc = np.sqrt(edges[:-1] * edges[1:])
    g_c, _, kappa_c = nfw_reduced_tangential_shear(
        rc, m200=args.m200, c200=args.c200, z_lens=args.zl, z_source=args.zs)
    if g_edges.max() >= HI:
        raise SystemExit("profile exceeds the training prior |g| < 0.1; raise rmin")
    # g_t decreases with R, so annulus b spans [g_edges[b+1], g_edges[b]].
    return edges, g_edges, rc, g_c, kappa_c


def line_logq(x, w, e, tgrid, bw_factor, floor, origin=None):
    """log q_i(origin_i + t e_i) for a batch: x (B,S,2), w (B,S) normalized, e (B,2), tgrid (P,).

    Points outside the prior box get log q = -inf (the training prior is zero there).
    """
    mu = torch.einsum("bs,bsk->bk", w, x)
    var = torch.einsum("bs,bsk->bk", w, (x - mu[:, None]) ** 2)
    sig = torch.clamp(torch.sqrt(var), min=SIG_MIN)
    neff = 1.0 / torch.sum(w * w, dim=1)
    h = torch.clamp(bw_factor * sig * neff[:, None] ** (-1.0 / 6.0), min=H_MIN)  # (B,2)
    nodes = tgrid[None, :, None] * e[:, None, :]  # (B,P,2)
    if origin is not None:
        nodes = nodes + origin[:, None, :]
    dens = torch.ones(x.shape[0], tgrid.numel(), x.shape[1], device=x.device, dtype=x.dtype)
    for k in range(2):
        n = nodes[:, :, k:k + 1]  # (B,P,1)
        xs = x[:, None, :, k]  # (B,1,S)
        hk = h[:, k, None, None]
        ker = (torch.exp(-0.5 * ((n - xs) / hk) ** 2)
               + torch.exp(-0.5 * ((n + xs - 2 * LO) / hk) ** 2)
               + torch.exp(-0.5 * ((n + xs - 2 * HI) / hk) ** 2)) / (hk * math.sqrt(2 * math.pi))
        dens = dens * ker
    q = torch.einsum("bps,bs->bp", dens, w)
    q = (1.0 - floor) * q + floor / AREA
    logq = torch.log(q)
    outside = torch.any(torch.abs(nodes) >= HI, dim=-1)
    return logq.masked_fill(outside, float("-inf")), h


def synthetic_samples(g_true, rng, nsamp, sigma_like):
    """Samples of a Gaussian likelihood around a noisy shear, times the box prior."""
    out = np.empty((len(g_true), nsamp, 2), dtype=np.float32)
    for i, (g, s) in enumerate(zip(g_true, sigma_like)):
        obs = g + rng.normal(0, s, 2)
        draw = []
        while sum(len(d) for d in draw) < nsamp:
            d = obs + rng.normal(0, s, (4 * nsamp, 2))
            draw.append(d[np.all(np.abs(d) < HI, axis=1)])
        out[i] = np.concatenate(draw)[:nsamp]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, default=CANONICAL_CACHE)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--m200", type=float, default=1e15)
    p.add_argument("--c200", type=float, default=4.0)
    p.add_argument("--zl", type=float, default=0.1)
    p.add_argument("--zs", type=float, default=0.3)
    p.add_argument("--rmin", type=float, default=0.1, help="Mpc")
    p.add_argument("--rmax", type=float, default=3.0, help="Mpc")
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--ngrid", type=int, default=1601)
    p.add_argument("--bw-factor", type=float, default=1.0)
    p.add_argument("--floor", type=float, default=1e-3)
    p.add_argument("--weights", choices=("tf", "train"), default="tf")
    p.add_argument("--nboot", type=int, default=1000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-parts", type=int, default=None)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=20261003)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    edges, g_edges, rc, g_c, kappa_c = annuli(args)
    gmin, gmax = g_edges[-1], g_edges[0]
    tgrid = torch.linspace(-0.0999, 0.0999, args.ngrid, device=dev, dtype=torch.float32)
    tg = tgrid.cpu().numpy().astype(np.float64)

    sample_files = parts(args.cache, "shear_sample")
    truth_files = parts(args.cache, "truth")
    logw_files = parts(args.cache, "posterior_tf_log_weight")
    if args.max_parts:
        sample_files, truth_files, logw_files = (f[: args.max_parts] for f in (sample_files, truth_files, logw_files))

    rows = {k: [] for k in ("gabs", "logl_t", "logl_x", "global_index", "h")}
    offset = 0
    for sf, tf_, lf in zip(sample_files, truth_files, logw_files):
        truth = np.load(tf_, mmap_mode="r")
        g = np.asarray(truth[:, :2], dtype=np.float64)
        gabs = np.hypot(g[:, 0], g[:, 1])
        sel = np.flatnonzero((gabs >= gmin) & (gabs <= gmax))
        if sel.size:
            samp = np.load(sf, mmap_mode="r")
            logw = np.load(lf, mmap_mode="r")
            for a in range(0, sel.size, args.batch):
                idx = sel[a:a + args.batch]
                if args.synthetic:
                    x = synthetic_samples(g[idx], rng, samp.shape[1], np.full(idx.size, 0.03))
                    w = np.full(x.shape[:2], 1.0 / x.shape[1])
                else:
                    x = np.asarray(samp[idx], dtype=np.float32)
                    if args.weights == "tf":
                        lw = np.asarray(logw[idx], dtype=np.float64)
                        w = np.exp(lw - lw.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
                    else:
                        w = np.full(x.shape[:2], 1.0 / x.shape[1])
                et = g[idx] / gabs[idx, None]
                ex = np.stack([-et[:, 1], et[:, 0]], 1)  # 45 deg rotation in shear angle
                # (cos2a, sin2a) -> (-sin2a, cos2a)
                X = torch.as_tensor(x, device=dev)
                W = torch.as_tensor(w, device=dev, dtype=torch.float32)
                lt, h = line_logq(X, W, torch.as_tensor(et, device=dev, dtype=torch.float32), tgrid,
                                  args.bw_factor, args.floor)
                # Null test: a cut along the cross direction through the galaxy's true
                # shear, g = |g_i| e_t,i + x e_x,i. The stacked x-hat is non-zero if the
                # shear direction is systematically rotated.
                lx, _ = line_logq(X, W, torch.as_tensor(ex, device=dev, dtype=torch.float32), tgrid,
                                  args.bw_factor, args.floor,
                                  origin=torch.as_tensor(g[idx], device=dev, dtype=torch.float32))
                rows["logl_t"].append(lt.cpu().numpy()); rows["logl_x"].append(lx.cpu().numpy())
                rows["h"].append(h.cpu().numpy()); rows["gabs"].append(gabs[idx])
                rows["global_index"].append(offset + idx)
        offset += len(g)
        print(f"{sf.name}: {sel.size} selected, {time.time() - t0:.0f}s", flush=True)
    data = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez(args.out / "annulus_logl.npz", tgrid=tg, **data)

    # ---- per-annulus stacking
    gabs = data["gabs"]
    bin_of = np.searchsorted(-g_edges, -gabs, side="right") - 1  # g_edges decreasing
    bin_of = np.clip(bin_of, 0, args.nbins - 1)

    def post_stats(logl_sum):
        logl_sum = np.where(np.isfinite(logl_sum), logl_sum, -np.inf)
        lp = logl_sum - logl_sum.max(-1, keepdims=True)
        pw = np.exp(lp); pw /= pw.sum(-1, keepdims=True)
        mean = pw @ tg
        std = np.sqrt(np.maximum(pw @ tg**2 - mean**2, 0))
        return mean, std

    # per-galaxy width along e_t (likelihood alone, uniform prior)
    _, sig_i = post_stats(data["logl_t"])
    res = []
    boot_t = np.empty((args.nboot, args.nbins)); boot_x = np.empty((args.nboot, args.nbins))
    ref = np.empty(args.nbins)
    for b in range(args.nbins):
        m = np.flatnonzero(bin_of == b)
        wi = 1.0 / sig_i[m] ** 2
        ref[b] = np.sum(wi * gabs[m]) / wi.sum()
        mt, st = post_stats(data["logl_t"][m].sum(0))
        mx, sx = post_stats(np.where(np.isfinite(data["logl_x"][m]), data["logl_x"][m], -1e6).sum(0))
        # bootstrap: resampled galaxy counts times per-galaxy log-likelihoods
        counts = rng.multinomial(m.size, np.full(m.size, 1.0 / m.size), size=args.nboot).astype(np.float64)
        boot_t[:, b] = post_stats(counts @ data["logl_t"][m].astype(np.float64))[0]
        lxm = np.where(np.isfinite(data["logl_x"][m]), data["logl_x"][m], -1e6).astype(np.float64)
        boot_x[:, b] = post_stats(counts @ lxm)[0]
        res.append(dict(bin=b, r_lo=float(edges[b]), r_hi=float(edges[b + 1]), r_center=float(rc[b]),
                        g_profile_center=float(g_c[b]), kappa_center=float(kappa_c[b]),
                        g_lo=float(g_edges[b + 1]), g_hi=float(g_edges[b]), n=int(m.size),
                        g_ref=float(ref[b]), g_plain_mean=float(gabs[m].mean()),
                        gt_hat=float(mt), gt_post_std=float(st), gt_boot_std=float(boot_t[:, b].std()),
                        gx_hat=float(mx), gx_post_std=float(sx), gx_boot_std=float(boot_x[:, b].std())))
        print(f"annulus {b}: R=[{edges[b]:.2f},{edges[b+1]:.2f}] n={m.size} ref={ref[b]:.4f} "
              f"gt={mt:.4f}±{boot_t[:, b].std():.4f} (post {st:.4f})  gx={mx:+.4f}±{boot_x[:, b].std():.4f}", flush=True)

    def fit(y, x):
        A = np.column_stack([np.ones_like(x), x])
        c, s = np.linalg.lstsq(A, y, rcond=None)[0]
        return s - 1.0, c
    m_hat, c_hat = fit(np.array([r["gt_hat"] for r in res]), ref)
    mb = np.array([fit(boot_t[k], ref) for k in range(args.nboot)])
    summary = dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        lens=dict(m200=args.m200, c200=args.c200, z_lens=args.zl, z_source=args.zs,
                  r200_mpc=r200_mpc(args.m200, args.c200, args.zl)),
        n_galaxies=int(gabs.size), annuli=res,
        m=float(m_hat), m_boot_std=float(mb[:, 0].std()), c=float(c_hat), c_boot_std=float(mb[:, 1].std()),
        bandwidth_median=np.median(data["h"], axis=0).tolist(), seconds=time.time() - t0,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"m = {m_hat:+.4f} ± {mb[:, 0].std():.4f}, c = {c_hat:+.5f} ± {mb[:, 1].std():.5f}")


if __name__ == "__main__":
    main()
