#!/usr/bin/env python3
"""Shear likelihood of one galaxy on a grid over the training box, from its weighted posterior samples.

The kernel is scipy's: ``scipy.stats.gaussian_kde(samples, weights=w)`` with its default Scott's rule, i.e. a
Gaussian kernel with the full weighted sample covariance times n_eff^(-1/3). Since the likelihoods are long, thin and
tilted with respect to the (g1, g2) axes, a full-covariance kernel matters: a separate width per component widens
the thin direction.

Evaluation is the standard binned KDE: the weighted samples are linearly (cloud-in-cell) binned onto the grid nodes
and convolved with the kernel by FFT (scipy.signal.fftconvolve). The posterior lives inside the box, so the density
that the kernel spreads beyond an edge is folded back across it (reflection method). ``exact_density`` evaluates the
same reflected KDE without binning, with gaussian_kde itself, for validation.

Two grid conventions are supported: nodes that include the box edges (``edges=True``: LO, LO + d, ..., HI) and
cell-centred nodes (``edges=False``: LO + d/2, ..., HI - d/2).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import gaussian_kde

LO, HI = -0.1, 0.1
AREA = (HI - LO) ** 2
H_MIN = 1e-3  # smallest kernel sd along any direction (guards degenerate, ESS ~ 1 weights)


def scipy_kde(samples, w, factor=1.0):
    """gaussian_kde with Scott's rule (n_eff^(-1/6) in 2D) times factor."""
    kde = gaussian_kde(samples.T, weights=w)
    if factor != 1.0:
        kde.set_bandwidth(kde.scotts_factor() * factor)
    return kde


def kernel_cov(samples, w, factor=1.0):
    """Kernel covariance of scipy_kde, floored at H_MIN^2 in every direction."""
    cov = scipy_kde(samples, w, factor).covariance
    ev, V = np.linalg.eigh(cov)
    return (V * np.maximum(ev, H_MIN ** 2)) @ V.T


def _cic(samples, w, x0, d, m):
    """Cloud-in-cell weights of the samples on an m x m grid with first node x0 and spacing d."""
    u = (samples - x0) / d
    i = np.clip(np.floor(u).astype(int), 0, m - 2)
    t = np.clip(u - i, 0.0, 1.0)
    out = np.zeros((m, m))
    for a, wa in ((0, 1 - t[:, 0]), (1, t[:, 0])):
        for b, wb in ((0, 1 - t[:, 1]), (1, t[:, 1])):
            np.add.at(out, (i[:, 0] + a, i[:, 1] + b), w * wa * wb)
    return out


def _fold(a, pad, m, edges):
    """Fold the parts of a padded axis-0 array beyond the box back inside, mirrored about the edges."""
    inner = a[pad:pad + m].copy()
    if edges:  # a node on an edge is its own mirror image
        inner[0] += a[pad]
        inner[m - 1] += a[pad + m - 1]
    for k in range(1, pad + 1):
        lo, hi = (k, m - 1 - k) if edges else (k - 1, m - k)
        if 0 <= lo < m:
            inner[lo] += a[pad - k]
        if 0 <= hi < m:
            inner[hi] += a[pad + m - 1 + k]
    return inner


def density(samples, w, nodes, edges=True, factor=1.0):
    """Reflected KDE of weighted samples (n, 2) on nodes x nodes; normalized to unit mass over the box."""
    m = len(nodes); d = nodes[1] - nodes[0]
    cov = kernel_cov(samples, w, factor)
    r = int(np.ceil(5 * np.sqrt(cov.diagonal().max()) / d)) + 1
    off = np.arange(-r, r + 1) * d
    X, Y = np.meshgrid(off, off, indexing="ij")
    P = np.linalg.inv(cov)
    ker = np.exp(-0.5 * (P[0, 0] * X * X + 2 * P[0, 1] * X * Y + P[1, 1] * Y * Y))
    ker /= ker.sum()
    pad = min(r, m - 1)
    hist = np.pad(_cic(samples, w / w.sum(), nodes[0], d, m), pad)
    dens = np.clip(fftconvolve(hist, ker, mode="same"), 0, None)
    dens = _fold(_fold(dens, pad, m, edges).T, pad, m, edges).T
    return dens / (dens.sum() * d * d)


def log_likelihood(samples, w, nodes, edges=True, factor=1.0, floor=1e-3):
    """log of the KDE density with a defensive floor, (1 - floor) q + floor / |box|."""
    return np.log((1.0 - floor) * density(samples, w, nodes, edges, factor) + floor / AREA)


def exact_density(samples, w, points, factor=1.0):
    """Unbinned reflected KDE at points (k, 2) with gaussian_kde itself: the density summed over the 9 mirror images.

    No H_MIN floor; it matches density() whenever the floor is inactive.
    """
    kde = scipy_kde(samples, w, factor)
    out = np.zeros(len(points))
    for mx in (None, LO, HI):
        for my in (None, LO, HI):
            p = points.copy()
            if mx is not None:
                p[:, 0] = 2 * mx - p[:, 0]
            if my is not None:
                p[:, 1] = 2 * my - p[:, 1]
            out += kde(p.T)
    return out
