#!/usr/bin/env python3
"""9D corner plot of individual NPE posteriors and their stacked shear likelihood.

Answers "how can stacking a few biased likelihoods remove the bias?" on one
constant-shear cell of the T8 benchmark (``likelihood_stack_04_benchmark``).

* Individual galaxies: the first ``--ngal`` galaxies of the cell, noise
  realization 0, drawn exactly as T8 drew them (``c_stack_infer.py``: same
  model, seeds, identity+R90 halves, 8192 samples). The script checks that the
  redrawn (g1, g2) samples match the T8 cache. All nine parameters are kept.
  Weights are uniform (training prior), as in the T8 headline numbers.
* Stack: the product of the same galaxies' shear likelihoods
  L_i(g) ∝ q(g | D_i) / pi_train(g) ∝ q(g | D_i) inside the prior box, with
  q from the T8 KDE (Scott bandwidth, reflection at the box edges, floor 1e-3)
  on the T8 400 x 400 lattice. The stack exists only for (g1, g2), since the
  galaxies share nothing else.

The g1-g2 and the g1, g2 1D panels are drawn from the T8 KDE for both the
individual and the stacked curves, so the filled stack is exactly the product
of the plotted line contours. The nuisance panels use getdist.

Writes a new folder; refuses to overwrite.
"""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ARCH = HERE.parents[1]
for p in (HERE, ARCH, ARCH.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import c_stack_infer as ci  # noqa: E402
from c_stack_analysis import AREA, HI, LO, _axis_kernel, lattice  # noqa: E402

S = Path("/ocean/projects/phy250048p/shared")
OUT = S / "reports" / "paper-tests" / "C_stacking_corner"
NAMES9 = ["g1", "g2", "theta_int", "cosi", "v0", "vcirc", "rscale", "hlr", "halpha_flux_true"]
PRIOR = {"g1": (-0.1, 0.1), "g2": (-0.1, 0.1), "theta_int": (-math.pi, math.pi),
         "cosi": (0.0, 1.0), "v0": (-30.0, 30.0), "vcirc": (60.0, 540.0),
         "rscale": (0.1, 5.0), "hlr": (0.1, 5.0), "halpha_flux_true": (-17.0, -14.0)}
GAL_COLORS = ["#1b6ca8", "#d1495b", "#2a9d8f", "#e9a03b"]
FLOOR = 1e-3
NGRID = 400


def parse_args(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=int, default=15)
    p.add_argument("--ngal", type=int, default=4)
    p.add_argument("--realization", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--samples", type=Path, default=None,
                   help="reuse samples.npz written by an earlier run instead of redrawing")
    return p.parse_args(argv)


@torch.inference_mode()
def draw_cell(cell_id, realization, device):
    """Redraw T8's samples for one cell and realization; return 9D physical samples."""
    import pyxis.torch as pxt
    import config
    from data import rotate_90_datavector, rotate_90_parameters
    from model_registry import load_model_config
    from networks import KLNPE
    from train import load_model, seed_everything
    from utils import denormalize
    from diagnostics.likelihood_stack_benchmark import _load_cell

    cfg = load_model_config(ci.MODEL, configs_root=str(S / "configs"))
    config.set_model_config(cfg)
    names = tuple(config.TARGET_NAMES)
    assert list(names) == NAMES9, names
    par_ranges = config.par_ranges
    seed = 20260929  # c_stack_infer default
    seed_everything(seed, deterministic=True)
    model = load_model(KLNPE, path=str(S / "models" / ci.MODEL / f"{ci.MODEL}best"),
                       model_name=ci.MODEL, device=str(device), strict=True,
                       networks_root=str(S / "networks"))
    model.eval()
    channels_last = bool(cfg.train.channels_last)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    dataset = pxt.TorchDataset(str(ci.DATA))
    noisy = np.load(ci.DATA / "benchmark_noisy.npz")
    cell = _load_cell(dataset, cell_id, device, names, par_ranges)
    ctx = {"rmag_true": cell["rmag"], "image_snr": cell["image_snr"],
           "central_halpha_snr": cell["spec_snr"]}
    start = cell_id * ci.N_GAL
    r = realization
    image = torch.as_tensor(noisy["images"][start:start + ci.N_GAL, r], device=device, dtype=torch.float32)
    spec = torch.as_tensor(noisy["spectra"][start:start + ci.N_GAL, r], device=device, dtype=torch.float32)
    half = 8192 // 2
    s_id = ci._sample(model, image, spec, cell["fib_pos"], ctx, half, 2048,
                      ci._seed(seed, cell_id, r, 0), channels_last)
    img_r, spec_r, _, pos_r = rotate_90_datavector(image, spec, fiber_positions=cell["fib_pos"])
    s_r = ci._sample(model, img_r, spec_r, pos_r, ctx, half, 2048,
                     ci._seed(seed, cell_id, r, 1), channels_last)
    s_r = rotate_90_parameters(s_r, inverse=True, feature_names=names)
    bank = torch.cat((s_id, s_r), dim=1).float().cpu().numpy()
    phys = np.asarray(denormalize(bank, par_ranges, feature_names=names,
                                  target_transforms=config.TARGET_TRANSFORMS), np.float64)
    return phys, np.asarray(cell["truth"], np.float64)


def kde_logq(g, nodes):
    """T8 KDE log density (train weights, bandwidth factor 1, floor 1e-3) on nodes x nodes."""
    g = torch.as_tensor(g, dtype=torch.float64)  # (B,S,2)
    B, n, _ = g.shape
    w = torch.full((B, n), 1.0 / n, dtype=torch.float64)
    mu = torch.einsum("bs,bsk->bk", w, g)
    std = torch.sqrt(torch.einsum("bs,bsk->bk", w, (g - mu[:, None]) ** 2))
    h = torch.clamp(std * float(n) ** (-1.0 / 6.0), min=1e-3)
    k1 = _axis_kernel(nodes, g[:, :, 0], h[:, 0]) * w[:, None, :]
    k2 = _axis_kernel(nodes, g[:, :, 1], h[:, 1])
    dens = torch.bmm(k1, k2.transpose(1, 2))
    return torch.log((1 - FLOOR) * dens + FLOOR / AREA).numpy()  # (B, P1, P2), index [g1, g2]


def hpd_levels(p, fracs=(0.95, 0.68)):
    """Density thresholds enclosing the given probability mass (ascending levels)."""
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
    from getdist import MCSamples, plots
    from paper_style import COLORS, DCOL, GETDIST_SETTINGS, apply_style

    if args.samples is not None:
        z = np.load(args.samples)
        phys, truth = z["phys"], z["truth"]
    else:
        phys, truth = draw_cell(args.cell, args.realization, torch.device(args.device))
    # T8 drew on a GPU; CPU and CUDA random streams differ, so a CPU redraw is an
    # independent draw from the same posterior. Check it statistically instead.
    cached = np.load(ci.OUT / f"cell{args.cell:02d}.npz")["g"][:, args.realization].astype(np.float64)
    new_g = phys[..., :2]
    se = np.sqrt(new_g.var(axis=1) / new_g.shape[1] + cached.var(axis=1) / cached.shape[1])
    z_mean = np.abs(new_g.mean(axis=1) - cached.mean(axis=1)) / se
    sd_ratio = new_g.std(axis=1) / cached.std(axis=1)
    max_dev = dict(max_mean_z=float(z_mean.max()), sd_ratio_range=[float(sd_ratio.min()), float(sd_ratio.max())])
    print(f"redraw vs T8 cache over {ci.N_GAL} galaxies: max |mean diff|/MC error = {z_mean.max():.2f}, "
          f"sd ratio {sd_ratio.min():.3f}-{sd_ratio.max():.3f}", flush=True)
    if z_mean.max() > 5 or np.abs(sd_ratio - 1).max() > 0.05:
        raise RuntimeError("redrawn posteriors are inconsistent with the T8 cache")

    G = args.ngal
    x = phys[:G].copy()
    t = truth[:G].copy()
    jf = NAMES9.index("halpha_flux_true")
    x[..., jf] = np.log10(x[..., jf])
    t[:, jf] = np.log10(t[:, jf])
    g_true = truth[0, :2]
    assert np.allclose(truth[:, :2], g_true)

    # Shear likelihoods on the T8 lattice and their product.
    nodes = lattice(NGRID, "cpu")
    nod = nodes.numpy()
    d = nod[1] - nod[0]
    logq = kde_logq(x[..., :2], nodes)
    logq -= logq.reshape(G, -1).max(1)[:, None, None]
    ind = np.exp(logq)
    ind /= ind.sum(axis=(1, 2), keepdims=True)
    lstack = logq.sum(0)
    stack = np.exp(lstack - lstack.max())
    stack /= stack.sum()
    G1, G2 = np.meshgrid(nod, nod, indexing="ij")
    stack_mean = np.array([(stack * G1).sum(), (stack * G2).sum()])
    stack_sd = np.sqrt([(stack * (G1 - stack_mean[0]) ** 2).sum(), (stack * (G2 - stack_mean[1]) ** 2).sum()])
    post_mean = x[..., :2].mean(axis=1)
    post_sd = x[..., :2].std(axis=1)
    print("true g:", g_true)
    for i in range(G):
        print(f"galaxy {i}: posterior mean {post_mean[i]}, sd {post_sd[i]}, cos i {t[i, 3]:.2f}, theta {t[i, 2]:.2f}")
    print("stacked mean", stack_mean, "sd", stack_sd)

    # theta_int: unwrap each galaxy around its circular mean so no posterior is split at +-pi.
    jt = NAMES9.index("theta_int")
    for i in range(G):
        cm = math.atan2(np.sin(x[i, :, jt]).mean(), np.cos(x[i, :, jt]).mean())
        x[i, :, jt] = cm + (x[i, :, jt] - cm + np.pi) % (2 * np.pi) - np.pi
        t[i, jt] = cm + (t[i, jt] - cm + np.pi) % (2 * np.pi) - np.pi

    apply_style()
    labels = {"g1": r"g_1", "g2": r"g_2", "theta_int": r"\theta_{\rm int}\,[{\rm rad}]", "cosi": r"\cos i",
              "v0": r"v_0\,[{\rm km\,s^{-1}}]", "vcirc": r"v_{\rm circ}\,[{\rm km\,s^{-1}}]",
              "rscale": r"R_{\rm vscale}\,[{\rm arcsec}]", "hlr": r"R_h\,[{\rm arcsec}]",
              "halpha_flux_true": r"\log_{10}F_{{\rm H}\alpha}"}
    ranges = dict(PRIOR)
    ranges["theta_int"] = (None, None)
    samples = [MCSamples(samples=x[i], names=NAMES9, labels=[labels[n] for n in NAMES9],
                         label=f"galaxy {i + 1}", ranges=ranges,
                         settings=dict(GETDIST_SETTINGS, contours=[0.68, 0.95]))
               for i in range(G)]
    # Shear panels show the whole prior box; nuisance panels zoom to where the posteriors are.
    limits = {"g1": PRIOR["g1"], "g2": PRIOR["g2"]}
    for j, p in enumerate(NAMES9[2:], start=2):
        lo = min(min(np.quantile(x[i, :, j], 0.002) for i in range(G)), t[:, j].min())
        hi = max(max(np.quantile(x[i, :, j], 0.998) for i in range(G)), t[:, j].max())
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if p != "theta_int":
            lo, hi = max(lo, PRIOR[p][0]), min(hi, PRIOR[p][1])
        limits[p] = (lo, hi)

    gp = plots.get_subplot_plotter(width_inch=DCOL)
    gp.settings.scaling = False
    gp.settings.axes_fontsize = 6.5
    gp.settings.axes_labelsize = 8.5
    gp.settings.linewidth_contour = 0.9
    gp.settings.num_plot_contours = 2
    gp.settings.axis_tick_x_rotation = 45
    gp.settings.axis_tick_max_labels = 3
    gp.settings.legend_frame = False
    gp.triangle_plot(samples, NAMES9, filled=False, contour_colors=GAL_COLORS[:G],
                     contour_lws=[0.9] * G, param_limits=limits,
                     line_args=[{"color": c, "lw": 0.9} for c in GAL_COLORS[:G]], legend_labels=[])

    # Per-galaxy truth markers in every nuisance panel.
    n = len(NAMES9)
    for r_ in range(n):
        for c_ in range(r_ + 1):
            ax = gp.subplots[r_][c_]
            if ax is None:
                continue
            if r_ == c_:
                if r_ >= 2:
                    for i in range(G):
                        ax.axvline(t[i, r_], color=GAL_COLORS[i], lw=0.7, ls=":")
            elif not (r_ < 2 and c_ < 2):
                for i in range(G):
                    ax.plot(t[i, c_], t[i, r_], marker="x", ms=4, mew=1.0, color=GAL_COLORS[i], ls="none")
                if c_ < 2:
                    ax.axvline(g_true[c_], color=COLORS["truth"], lw=0.7)

    # Redraw the shear panels from the T8 KDE: individual lines plus the filled stack.
    ax2 = gp.subplots[1][0]
    ax2.cla()
    for i in range(G):
        ax2.contour(G1, G2, ind[i], levels=hpd_levels(ind[i]), colors=GAL_COLORS[i],
                    linewidths=[0.6, 0.9])
        ax2.plot(*post_mean[i], marker="o", ms=3.5, color=GAL_COLORS[i], mec="k", mew=0.4, ls="none")
    lv = hpd_levels(stack)
    ax2.contourf(G1, G2, stack, levels=[lv[0], lv[1]], colors=[COLORS["stacked"]], alpha=0.35)
    ax2.contourf(G1, G2, stack, levels=[lv[1], stack.max() * 1.01], colors=[COLORS["stacked"]], alpha=0.8)
    ax2.axvline(g_true[0], color=COLORS["truth"], lw=0.7)
    ax2.axhline(g_true[1], color=COLORS["truth"], lw=0.7)
    ax2.plot(*stack_mean, marker="*", ms=7, color="white", mec=COLORS["stacked"], mew=0.8, ls="none")
    ax2.set_xlim(LO, HI); ax2.set_ylim(LO, HI)
    ax2.set_ylabel(r"$g_2$")
    ax2.tick_params(labelbottom=False)
    for k in (0, 1):
        ax1 = gp.subplots[k][k]
        ax1.cla()
        for i in range(G):
            m = ind[i].sum(axis=1 - k)
            ax1.plot(nod, m / m.max(), color=GAL_COLORS[i], lw=0.9)
        ms_ = stack.sum(axis=1 - k)
        ax1.fill_between(nod, ms_ / ms_.max(), color=COLORS["stacked"], alpha=0.6, lw=0)
        ax1.axvline(g_true[k], color=COLORS["truth"], lw=0.7)
        ax1.set_xlim(LO, HI); ax1.set_ylim(0, 1.08)
        ax1.set_yticks([])
        if k == 0:
            ax1.tick_params(labelbottom=False)
    gp.subplots[n - 1][0].set_xlim(LO, HI)
    for r_ in range(n):
        for c_ in range(min(r_ + 1, 2)):
            ax = gp.subplots[r_][c_]
            if ax is not None:
                ax.set_xticks([-0.05, 0.0, 0.05])
    gp.subplots[1][0].set_yticks([-0.05, 0.0, 0.05])

    handles = [Line2D([], [], color=GAL_COLORS[i], lw=1.0,
                      label=f"galaxy {i + 1} ($\\cos i={t[i, 3]:.2f}$)") for i in range(G)]
    handles += [Line2D([], [], marker="o", color="0.5", mec="k", mew=0.4, ls="none", ms=4,
                       label="individual posterior means"),
                Patch(color=COLORS["stacked"], alpha=0.7, label=f"stacked shear likelihood ({G} galaxies)"),
                Line2D([], [], marker="*", color="white", mec=COLORS["stacked"], ls="none", ms=8,
                       label="stacked mean"),
                Line2D([], [], color=COLORS["truth"], lw=0.8, label="true shear"),
                Line2D([], [], color="0.4", marker="x", ls="none", label="true nuisance values")]
    gp.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.97), frameon=False, fontsize=8.5)

    args.out.mkdir(parents=True)
    if args.samples is None:
        np.savez(args.out / "samples.npz", phys=phys, truth=truth)
    stem = args.out / f"stack_corner_cell{args.cell:02d}_n{G}"
    gp.fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    gp.fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    summary = dict(cell=args.cell, realization=args.realization, ngal=G, true_g=g_true.tolist(),
                   redraw_max_abs_diff=max_dev, weights="training prior (uniform)",
                   kde="T8: Scott bandwidth std*n^(-1/6) (>=1e-3), reflection at +-0.1, floor 1e-3, 400x400 lattice",
                   galaxies=[dict(index=i, post_mean=post_mean[i].tolist(), post_sd=post_sd[i].tolist(),
                                  truth={p: float(t[i, j]) for j, p in enumerate(NAMES9)}) for i in range(G)],
                   stacked=dict(mean=stack_mean.tolist(), sd=stack_sd.tolist()))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {stem}.pdf/.png and summary.json")


if __name__ == "__main__":
    main()
