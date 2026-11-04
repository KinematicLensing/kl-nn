#!/usr/bin/env python3
"""Check kde_grid.density (binned + FFT) against the exact reflected gaussian_kde on real cluster galaxies.

For galaxies spanning the range of likelihood widths in one cluster realization: the relative difference of the
two densities where the density exceeds 1% of its peak, compared on every 4th node of the 401 grid; the ratio of
the likelihood's thin-axis width to the samples'; and the same for cell-centred nodes (Sec. 6.3 convention).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kde_grid as K  # noqa: E402
from h_cluster_analysis import GRID, load_npe  # noqa: E402


def moments(d, nodes):
    X, Y = np.meshgrid(nodes, nodes, indexing="ij")
    p = d / d.sum(); m = np.array([(p * X).sum(), (p * Y).sum()])
    e = np.stack([X - m[0], Y - m[1]], -1)
    return m, np.einsum("ij,ija,ijb->ab", p, e, e)


smp, lw, _ = load_npe("paper_tests_H_cluster_s3_sub1_n01_sub", "r0")
w_all = np.exp(lw - lw.max(1, keepdims=True)); w_all /= w_all.sum(1, keepdims=True)
minor = np.array([np.sqrt(np.linalg.eigvalsh(np.cov(smp[i].T, aweights=w_all[i]))[0]) for i in range(len(smp))])
pick = np.argsort(minor)[np.linspace(0, len(smp) - 1, 12).astype(int)]
sub = GRID[::4]
P = np.stack(np.meshgrid(sub, sub, indexing="ij"), -1).reshape(-1, 2)
cc = K.LO + (np.arange(400) + 0.5) * 0.0005
print(" gal  thin sd   max|rel diff| (>1% peak)   thin-axis width KDE/samples   cell-centred vs edge grid")
worst = 0
for i in pick:
    x, w = smp[i], w_all[i]
    d = K.density(x, w, GRID, edges=True)
    ex = K.exact_density(x, w, P).reshape(len(sub), len(sub))
    ex /= ex.sum() * (sub[1] - sub[0]) ** 2  # same normalization convention on the coarse comparison grid
    db = d[::4, ::4]; db = db / (db.sum() * (sub[1] - sub[0]) ** 2)
    core = ex > 0.01 * ex.max()
    edge = np.zeros_like(core); edge[[0, -1], :] = True; edge[:, [0, -1]] = True
    rel = np.abs(db[core] / ex[core] - 1).max(); worst = max(worst, rel)
    rel_in = np.abs(db[core & ~edge] / ex[core & ~edge] - 1).max()
    print(f"      interior-only max rel diff {rel_in:.4f}; edge nodes in core: {int((core & edge).sum())}")
    _, c_kde = moments(d, GRID)
    cs = np.cov(x.T, aweights=w); ev, V = np.linalg.eigh(cs); v = V[:, 0]
    thin = np.sqrt(v @ c_kde @ v / ev[0])
    dc = K.density(x, w, cc, edges=False)
    mc, _ = moments(dc, cc); me, _ = moments(d, GRID)
    print(f"{i:4d}  {np.sqrt(ev[0]):.4f}   {rel:.4f}                     {thin:.3f}"
          f"                         mean shift {np.abs(mc - me).max():.1e}")
print(f"worst relative difference in the core: {worst:.4f}")
