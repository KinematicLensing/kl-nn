#!/usr/bin/env python3
"""Individual shear posteriors and their stacked likelihood on one T8 cell.

Answers "how can stacking a few biased likelihoods remove the bias?" on one
constant-shear cell of the T8 benchmark (``likelihood_stack_04_benchmark``).
The galaxies share only the shear, so the figure shows (g1, g2) alone.

* Individual galaxies: the first ``--ngal`` galaxies of the cell, noise
  realization 0, read directly from the T8 sample cache (8192 identity+R90
  draws each, training prior, i.e. unweighted).
* Stack: the product of the same galaxies' shear likelihoods
  L_i(g) ∝ q(g | D_i) / pi_train(g) ∝ q(g | D_i) inside the prior box, with
  q from the T8 KDE (Scott bandwidth, reflection at the box edges, floor 1e-3)
  on the T8 400 x 400 lattice. The individual contours are drawn from the same
  KDE, so the filled stack is exactly the product of the plotted curves.

Writes a new folder; refuses to overwrite.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from c_stack_analysis import AREA, HI, LO, SAMPLES, _axis_kernel, lattice  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
OUT = S / "reports" / "paper-tests" / "C_stacking_shear2d"
GAL_COLORS = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a03b"]
FLOOR = 1e-3
NGRID = 400


def parse_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=int, default=15)
    p.add_argument("--ngal", type=int, default=4)
    p.add_argument("--realization", type=int, default=0)
    p.add_argument("--out", type=Path, default=OUT)
    return p.parse_args(argv)


def kde_logq(g, nodes):
    """T8 KDE log density (train weights, bandwidth factor 1, floor 1e-3), index [g1, g2]."""
    g = torch.as_tensor(g, dtype=torch.float64)  # (B,S,2)
    B, n, _ = g.shape
    w = torch.full((B, n), 1.0 / n, dtype=torch.float64)
    mu = torch.einsum("bs,bsk->bk", w, g)
    std = torch.sqrt(torch.einsum("bs,bsk->bk", w, (g - mu[:, None]) ** 2))
    h = torch.clamp(std * float(n) ** (-1.0 / 6.0), min=1e-3)
    k1 = _axis_kernel(nodes, g[:, :, 0], h[:, 0]) * w[:, None, :]
    k2 = _axis_kernel(nodes, g[:, :, 1], h[:, 1])
    dens = torch.bmm(k1, k2.transpose(1, 2))
    return torch.log((1 - FLOOR) * dens + FLOOR / AREA).numpy()


def hpd_levels(p, fracs=(0.95, 0.68)):
    """Density thresholds enclosing the given probability mass (ascending)."""
    flat = np.sort(p.ravel())[::-1]
    c = np.cumsum(flat) / flat.sum()
    return [flat[np.searchsorted(c, f)] for f in fracs]


def main(argv=None):
    args = parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; refusing to overwrite")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from paper_style import COLORS, apply_style

    z = np.load(SAMPLES / f"cell{args.cell:02d}.npz")
    G = args.ngal
    g = z["g"][:G, args.realization].astype(np.float64)  # (G, S, 2)
    truth = z["truth_full"][:G]
    g_true = truth[0, :2]
    assert np.allclose(truth[:, :2], g_true)

    nodes = lattice(NGRID, "cpu")
    nod = nodes.numpy()
    logq = kde_logq(g, nodes)
    logq -= logq.reshape(G, -1).max(1)[:, None, None]
    ind = np.exp(logq)
    ind /= ind.sum(axis=(1, 2), keepdims=True)
    lstack = logq.sum(0)
    stack = np.exp(lstack - lstack.max())
    stack /= stack.sum()
    G1, G2 = np.meshgrid(nod, nod, indexing="ij")
    stack_mean = np.array([(stack * G1).sum(), (stack * G2).sum()])
    stack_sd = np.sqrt([(stack * (G1 - stack_mean[0]) ** 2).sum(),
                        (stack * (G2 - stack_mean[1]) ** 2).sum()])
    post_mean = g.mean(axis=1)
    post_sd = g.std(axis=1)
    post_corr = np.array([np.corrcoef(g[i, :, 0], g[i, :, 1])[0, 1] for i in range(G)])
    print("true g:", g_true)
    for i in range(G):
        print(f"galaxy {i + 1}: mean {post_mean[i].round(4)}, sd {post_sd[i].round(4)}, "
              f"corr {post_corr[i]:+.2f}, cos i {truth[i, 3]:.2f}, theta_int {truth[i, 2]:+.2f}")
    print("stacked mean", stack_mean.round(4), "sd", stack_sd.round(4))

    apply_style()
    fig = plt.figure(figsize=(4.6, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    lv = hpd_levels(stack)
    ax.contourf(G1, G2, stack, levels=[lv[0], lv[1]], colors=[COLORS["stacked"]], alpha=0.35)
    ax.contourf(G1, G2, stack, levels=[lv[1], stack.max() * 1.01], colors=[COLORS["stacked"]], alpha=0.85)
    for i in range(G):
        ax.contour(G1, G2, ind[i], levels=hpd_levels(ind[i]), colors=GAL_COLORS[i], linewidths=[0.7, 1.1])
        ax.plot(*post_mean[i], marker="o", ms=5, color=GAL_COLORS[i], mec="k", mew=0.5, ls="none")
    ax.axvline(g_true[0], color=COLORS["truth"], lw=0.8)
    ax.axhline(g_true[1], color=COLORS["truth"], lw=0.8)
    ax.plot(*stack_mean, marker="*", ms=10, color="white", mec=COLORS["stacked"], mew=1.0, ls="none")
    ax.set_xlim(LO, HI)
    ax.set_ylim(LO, HI)
    ax.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_yticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax.set_xlabel(r"$g_1$")
    ax.set_ylabel(r"$g_2$")

    for k, a in ((0, axx), (1, axy)):
        for i in range(G):
            m = ind[i].sum(axis=1 - k)
            (a.plot(nod, m / m.max(), color=GAL_COLORS[i], lw=1.0) if k == 0
             else a.plot(m / m.max(), nod, color=GAL_COLORS[i], lw=1.0))
        ms_ = stack.sum(axis=1 - k)
        if k == 0:
            a.fill_between(nod, ms_ / ms_.max(), color=COLORS["stacked"], alpha=0.6, lw=0)
            a.axvline(g_true[0], color=COLORS["truth"], lw=0.8)
            a.set_ylim(0, 1.08)
        else:
            a.fill_betweenx(nod, ms_ / ms_.max(), color=COLORS["stacked"], alpha=0.6, lw=0)
            a.axhline(g_true[1], color=COLORS["truth"], lw=0.8)
            a.set_xlim(0, 1.08)
        a.set_xticks([] if k == 1 else a.get_xticks())
        a.set_yticks([] if k == 0 else a.get_yticks())
        a.tick_params(labelbottom=(k == 1) and False, labelleft=False)
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)
    axx.set_xlim(LO, HI)
    axy.set_ylim(LO, HI)

    handles = [Line2D([], [], color=GAL_COLORS[i], lw=1.1,
                      label=f"galaxy {i + 1} ($\\cos i={truth[i, 3]:.2f}$)") for i in range(G)]
    handles += [Line2D([], [], marker="o", color="0.6", mec="k", mew=0.5, ls="none", ms=5,
                       label="individual posterior means"),
                Patch(color=COLORS["stacked"], alpha=0.75, label=f"stacked likelihood ({G} galaxies)"),
                Line2D([], [], marker="*", color="white", mec=COLORS["stacked"], ls="none", ms=9,
                       label="stacked mean"),
                Line2D([], [], color=COLORS["truth"], lw=0.8, label="true shear")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=7, handlelength=1.4, columnspacing=1.2)

    args.out.mkdir(parents=True)
    stem = args.out / f"stack_shear2d_cell{args.cell:02d}_n{G}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=250, bbox_inches="tight")
    summary = dict(cell=args.cell, realization=args.realization, ngal=G, true_g=g_true.tolist(),
                   samples=str(SAMPLES / f"cell{args.cell:02d}.npz"), weights="training prior (uniform)",
                   kde="T8: Scott bandwidth std*n^(-1/6) (>=1e-3), reflection at +-0.1, floor 1e-3, 400x400 lattice",
                   galaxies=[dict(index=i, post_mean=post_mean[i].tolist(), post_sd=post_sd[i].tolist(),
                                  post_corr=float(post_corr[i]), cosi=float(truth[i, 3]),
                                  theta_int=float(truth[i, 2])) for i in range(G)],
                   stacked=dict(mean=stack_mean.tolist(), sd=stack_sd.tolist()))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {stem}.pdf/.png and summary.json")


if __name__ == "__main__":
    main()
