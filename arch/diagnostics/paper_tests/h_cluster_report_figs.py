#!/usr/bin/env python3
"""H: figures for the cluster-test HTML report (setup, cells, synthetic validation).

Reads the manifest/config written by h_cluster_make.py and the outputs of
h_cluster_analysis.py --source synthetic and h_cluster_validate.py; writes PNGs to
$S/reports/paper-tests-1m/H_cluster/<tag>/report_figs/.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h_cluster_analysis as H  # noqa: E402
from h_cluster_make import build_halos, lensing  # noqa: E402

INK, MUTED, GRID = "#1d2630", "#5d6b78", "#d5dce2"
MAIN, SUB, ACC = "#2f6db5", "#c4532d", "#3c8f6a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 150,
                     "figure.facecolor": "white"})


def whiskers(ax, x, y, g1, g2, scale, color, lw=1.4):
    beta = 0.5 * np.arctan2(g2, g1)
    L = np.hypot(g1, g2) * scale
    dx, dy = 0.5 * L * np.cos(beta), 0.5 * L * np.sin(beta)
    ax.plot(np.vstack([x - dx, x + dx]), np.vstack([y - dy, y + dy]), color=color, lw=lw, solid_capstyle="round")


def frame(ax, cfg, sub=True):
    R = cfg["r_max_mpc"]
    ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color=MUTED, lw=0.8))
    ax.plot(0, 0, "+", color=INK, ms=10, mew=1.5)
    if sub:
        ax.plot(cfg["sub_x_mpc"], cfg["sub_y_mpc"], "x", color=SUB, ms=8, mew=1.8)
    ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05); ax.set_aspect("equal")
    ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]")


def fig_setup(cfg, man, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    x, y = man["x_mpc"].to_numpy(), man["y_mpc"].to_numpy()
    ax = axes[0]
    whiskers(ax, x, y, man["g1_main"], man["g2_main"], 6.0, MAIN)
    ax.plot([2.0, 2.0 + 0.02 * 6], [-2.85, -2.85], color=MAIN, lw=2); ax.text(2.0, -2.7, "|g| = 0.02", color=MAIN, fontsize=8)
    frame(ax, cfg, sub=False); ax.set_title(f"true shear, main halo only ({len(man)} galaxies)", fontsize=10)
    ax = axes[1]
    dg1, dg2 = man["g1_sub"] - man["g1_main"], man["g2_sub"] - man["g2_main"]
    whiskers(ax, x, y, dg1, dg2, 30.0, SUB)
    ax.plot([1.8, 1.8 + 0.01 * 30], [-2.85, -2.85], color=SUB, lw=2); ax.text(1.8, -2.7, "|Δg| = 0.01", color=SUB, fontsize=8)
    frame(ax, cfg); ax.set_title("subhalo signal Δg = g(main+sub) − g(main), ×5 longer", fontsize=10)
    ax = axes[2]
    R = cfg["r_max_mpc"]; kpa = cfg["kpc_per_arcsec"]
    gx = np.linspace(-R, R, 301) + 1e-4
    X, Y = np.meshgrid(gx, gx)
    main_h, sub_h = build_halos(cfg)
    xa, ya = X.ravel() * 1000 / kpa, Y.ravel() * 1000 / kpa
    a1, a2, _ = lensing([main_h], xa, ya, cfg["z_s"]); b1, b2, _ = lensing([main_h, sub_h], xa, ya, cfg["z_s"])
    dg = np.hypot(b1 - a1, b2 - a2).reshape(X.shape)
    dg[np.hypot(X, Y) > R] = np.nan
    im = ax.imshow(dg, origin="lower", extent=[-R, R, -R, R], cmap="magma_r", vmin=0, vmax=0.04)
    ax.plot(x, y, ".", color=INK, ms=3)
    frame(ax, cfg); ax.set_title("|Δg| field and galaxy positions", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, label="|Δg|")
    fig.tight_layout(); fig.savefig(out / "setup.png"); plt.close(fig)


def fig_cells(cfg, man, out):
    pcells, pmem = H.polar_cells(man, cfg)
    ccells, cmem, edges = H.cartesian_cells(man, cfg)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    x, y = man["x_mpc"].to_numpy(), man["y_mpc"].to_numpy()
    ax = axes[0]
    radii = sorted({c["r_hi"] for c in pcells})
    for r in radii:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=MAIN, lw=0.9))
    for c in pcells:
        if c["kind"] == "sector":
            ax.plot([c["r_lo"] * math.cos(c["phi_lo"]), c["r_hi"] * math.cos(c["phi_lo"])],
                    [c["r_lo"] * math.sin(c["phi_lo"]), c["r_hi"] * math.sin(c["phi_lo"])], color=MAIN, lw=0.9)
    cnt = np.bincount(pmem[pmem >= 0], minlength=len(pcells))
    for c, cell in enumerate(pcells):
        r = 0.0 if cell["kind"] == "disk" else 0.5 * (cell["r_lo"] + cell["r_hi"])
        ph = 0.5 * (cell["phi_lo"] + cell["phi_hi"])
        ax.text(r * math.cos(ph), r * math.sin(ph) - (0.25 if cell["kind"] == "disk" else 0), str(cnt[c]),
                ha="center", va="center", fontsize=7, color=MAIN)
    ax.plot(x, y, ".", color=INK, ms=2.5)
    frame(ax, cfg); ax.set_title(f"polar cells ({len(pcells)}): numbers = galaxies per cell", fontsize=10)
    ax = axes[1]
    for e in edges:
        ax.axvline(e, color=ACC, lw=0.9); ax.axhline(e, color=ACC, lw=0.9)
    ccnt = np.bincount(cmem, minlength=len(ccells))
    for c, cell in enumerate(ccells):
        if ccnt[c]:
            ax.text(0.5 * (cell["x_lo"] + cell["x_hi"]), 0.5 * (cell["y_lo"] + cell["y_hi"]), str(ccnt[c]),
                    ha="center", va="center", fontsize=7, color=ACC)
    ax.plot(x, y, ".", color=INK, ms=2.5)
    frame(ax, cfg); ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(edges[0], edges[-1])
    ax.set_title(f"Cartesian cells ({int((ccnt > 0).sum())} occupied, {edges[1] - edges[0]:.2f} Mpc)", fontsize=10)
    ax = axes[2]
    bins = np.arange(0.5, max(cnt.max(), ccnt.max()) + 1.5)
    ax.hist(cnt, bins=bins, histtype="step", lw=2, color=MAIN, label=f"polar (mean {cnt.mean():.1f})")
    ax.hist(ccnt[ccnt > 0], bins=bins, histtype="step", lw=2, color=ACC, label=f"Cartesian (mean {ccnt[ccnt > 0].mean():.1f})")
    ax.set_xlabel("galaxies in cell"); ax.set_ylabel("number of cells"); ax.legend(fontsize=8, frameon=False)
    ax.set_title("cell occupancy", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "cells.png"); plt.close(fig)


def fig_validation(vdir, out):
    z = np.load(vdir / "validation_arrays.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    xs = np.linspace(-4, 4, 200)
    for ax, kind, labels in ((axes[0], "polar", ("g_t", "g_×")), (axes[1], "cartesian", ("g₁", "g₂"))):
        for sc, ls in (("main", "-"), ("sub", "--")):
            est, sd, ref = z[f"{sc}_{kind}_est"], z[f"{sc}_{kind}_sd"], z[f"{sc}_{kind}_ref"]
            pull = (est - ref[None]) / sd
            for k, col in ((0, MAIN), (1, SUB)):
                ax.hist(pull[..., k].ravel(), bins=np.linspace(-4, 4, 33), density=True, histtype="step",
                        color=col, ls=ls, lw=1.6,
                        label=f"{labels[k]}, {sc}: mean {pull[..., k].mean():+.2f}, spread {pull[..., k].std():.2f}")
        ax.plot(xs, np.exp(-0.5 * xs ** 2) / math.sqrt(2 * math.pi), color=INK, lw=1, label="ideal N(0, 1)")
        ax.set_xlabel("pull = (stacked − reference) / σ"); ax.set_ylabel("density")
        ax.set_title(f"{kind} cells: pulls over 30 realizations × all cells", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax = axes[2]
    for kind, col, mk in (("polar", MAIN, "o"), ("cartesian", ACC, "s")):
        est, sd, ref, n = z[f"main_{kind}_est"], z[f"main_{kind}_sd"], z[f"main_{kind}_ref"], z[f"main_{kind}_n"]
        res = est - ref[None]  # (R, cells, 2)
        for k, off in ((0, -0.12), (1, 0.12)):
            ks = np.unique(n)
            m = [res[:, n == v, k].mean() for v in ks]
            e = [res[:, n == v, k].std() / math.sqrt(res[:, n == v, k].size) for v in ks]
            ax.errorbar(ks + off + (0.05 if kind == "cartesian" else 0), m, yerr=e, fmt=mk,
                        mfc=col if k == 0 else "white", color=col, ms=5, capsize=2,
                        label=f"{kind}, {'1st' if k == 0 else '2nd'} component")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("galaxies stacked in the cell"); ax.set_ylabel("mean (stacked − reference)")
    ax.set_title("average residual vs. cell occupancy (main scenario)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.tight_layout(); fig.savefig(out / "validation.png"); plt.close(fig)


def fig_tuning(tdir, out):
    tj = json.loads((tdir / "tuning.json").read_text())
    lik = [r for r in tj["rows"] if r["method"] == "lik"]
    ks = sorted([r for r in tj["rows"] if r["method"] == "ks"], key=lambda r: r["setting"][0])
    pixs = sorted({r["setting"][0] for r in lik}); sks = sorted({r["setting"][1] for r in lik})
    sms = sorted({r["setting"][2] for r in lik})
    ch = tj["chosen"]["lik"]["setting"]
    fig, axes = plt.subplots(2, len(pixs) + 1, figsize=(4.1 * (len(pixs) + 1), 7.4),
                             gridspec_kw=dict(width_ratios=[1] * len(pixs) + [1.15]))
    for row, (field, cmap, vmin, vmax, fmt) in enumerate((("contrast", "viridis", None, None, "{:.1f}"),
                                                          ("false_peak", "magma_r", 0, 0.5, "{:.2f}"))):
        allv = [r[field] for r in lik if r[field] is not None]
        lo = vmin if vmin is not None else min(allv); hi = vmax if vmax is not None else max(allv)
        for j, pix in enumerate(pixs):
            ax = axes[row, j]
            M = np.full((len(sks), len(sms)), np.nan)
            for r in lik:
                if r["setting"][0] == pix and r[field] is not None:
                    M[sks.index(r["setting"][1]), sms.index(r["setting"][2])] = r[field]
            im = ax.imshow(M, origin="lower", cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
            for a_ in range(len(sks)):
                for b_ in range(len(sms)):
                    if np.isfinite(M[a_, b_]):
                        bright = (M[a_, b_] - lo) / (hi - lo + 1e-12)
                        dark = bright > 0.55 if cmap == "viridis" else bright < 0.45
                        ax.text(b_, a_, fmt.format(M[a_, b_]), ha="center", va="center", fontsize=9,
                                color=INK if dark else "white")
            if pix == ch[0]:
                ax.add_patch(plt.Rectangle((sms.index(ch[2]) - 0.5, sks.index(ch[1]) - 0.5), 1, 1,
                                           fill=False, ec=SUB, lw=2.5))
            ax.set_xticks(range(len(sms)), [f"{v:g}" for v in sms]); ax.set_yticks(range(len(sks)), [f"{v:g}" for v in sks])
            ax.set_xlabel("smoothness length [pixels]"); ax.set_ylabel("σ_κ")
            ax.set_title(f"likelihood map, {pix:g} Mpc pixels\n{field.replace('_', ' ')}", fontsize=10)
        fig.colorbar(im, ax=axes[row, len(pixs) - 1], fraction=0.046, pad=0.04)
    ax = axes[0, -1]; xs = [r["setting"][0] for r in ks]
    ax.plot(xs, [r["contrast"] for r in ks], "o-", color=MAIN)
    ax.axvline(tj["chosen"]["ks"]["setting"][0], color=SUB, ls="--", lw=1)
    ax.set_xlabel("KS smoothing [cells]"); ax.set_ylabel("contrast"); ax.set_title("Kaiser–Squires\ncontrast", fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax = axes[1, -1]
    ax.plot(xs, [r["false_peak"] for r in ks], "o-", color=MAIN)
    ax.axhline(0.10, color=MUTED, ls=":", lw=1); ax.axvline(tj["chosen"]["ks"]["setting"][0], color=SUB, ls="--", lw=1)
    ax.set_xlabel("KS smoothing [cells]"); ax.set_ylabel("false-peak rate"); ax.set_title("Kaiser–Squires\nfalse peak", fontsize=10)
    ax.set_ylim(0, max(0.5, max(r["false_peak"] for r in ks) * 1.1)); ax.grid(color=GRID, lw=0.6)
    fig.tight_layout(); fig.savefig(out / "tuning.png"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_r0")
    p.add_argument("--only", default="all")
    a = p.parse_args()
    cfg = json.loads((H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_config.json").read_text())
    man = pd.read_csv(H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_manifest.csv")
    base = H.S / "reports/paper-tests-1m/H_cluster" / a.tag
    out = base / "report_figs"; out.mkdir(parents=True, exist_ok=True)
    if a.only in ("all", "setup"):
        fig_setup(cfg, man, out); fig_cells(cfg, man, out)
    if a.only in ("all", "validation"):
        fig_validation(base / "synthetic_validation", out)
    if a.only in ("all", "tuning"):
        fig_tuning(base / "tuning", out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
