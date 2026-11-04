#!/usr/bin/env python3
"""H: summarise the option-B ensemble (independent galaxy sets s3_sub1_nNN, NPE posteriors).

Reads each realization's npe/report.json (cell stacks) and npe/kappa_maps.npz (maps), and writes to
$S/reports/paper-tests-1m/H_cluster/ensemble/:
  summary.json   pooled cell bias (linear fit of stacked - reference on reference), pulls, residual vs
                 occupancy; per-realization subhalo statistics and detection rates
  cells.png      pooled residuals and pull histograms
  detection.png  single-map z per realization (likelihood map and KS) and the aperture signals
  diffmaps.png   mean difference map over realizations, the true subhalo kappa, and example realizations

Single-map z: the main-only maps of all realizations form the null at the subhalo aperture,
z_r = (a_sub,r - mean_r' a_main,r') / sd_r' a_main,r', with a = mean residual (map - true main kappa) in a
0.5 Mpc aperture. The difference map is map(main + sub) - map(main only) of the same realization.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h_cluster_analysis as H  # noqa: E402
import h_cluster_tune as T  # noqa: E402

BASE = H.S / "reports/paper-tests-1m/H_cluster"
OUT = BASE / "ensemble"


def wls(g, e, s, cubic=False):
    X = np.column_stack([np.ones_like(g), g] + ([g ** 3] if cubic else []))
    w = 1 / s ** 2
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    b = cov @ X.T @ (w * (e - g))
    chi2 = float((((e - g) - X @ b) ** 2 * w).sum())
    return dict(c=b[0], c_err=np.sqrt(cov[0, 0]), m=b[1], m_err=np.sqrt(cov[1, 1]), chi2=chi2, dof=len(g) - X.shape[1])


def main():
    tags = [l.split()[0] for l in (OUT / "ensemble_seeds.txt").read_text().split("\n") if l.strip()]
    done = [t for t in tags if (BASE / t / "npe/report.json").exists() and (BASE / t / "npe/kappa_maps.npz").exists()]
    print(f"{len(done)}/{len(tags)} realizations analysed")
    cells = {(sc, kind): [] for sc in H.SCENARIOS for kind in ("polar", "cartesian")}
    ap = {k: [] for k in ("lik_sub", "lik_main", "ks_sub", "ks_main", "lik_diff", "ks_diff")}
    diff_lik, diff_ks, per = [], [], []
    for t in done:
        rep = json.loads((BASE / t / "npe/report.json").read_text())
        cfg = rep["config"]
        for sc in H.SCENARIOS:
            for kind in ("polar", "cartesian"):
                for row in rep["cells_results"][sc][kind]:
                    cells[(sc, kind)].append((row["n"], *row["est"], *row["sd"], *row["ref"]))
        z = np.load(BASE / t / "npe/kappa_maps.npz")
        c = z["centres"]; e = z["ks_edges"]; kc = 0.5 * (e[1:] + e[:-1])
        sub_pos, _ = T.apertures(cfg)
        ap["lik_sub"].append(T.ap_value(z["resid"], c, sub_pos)); ap["lik_main"].append(T.ap_value(z["resid_null"], c, sub_pos))
        ap["ks_sub"].append(T.ap_value(z["ks_resid_sub"], kc, sub_pos)); ap["ks_main"].append(T.ap_value(z["ks_resid_main"], kc, sub_pos))
        dl = z["lik_sub"] - z["lik_main"]; dk = z["ks_sub"] - z["ks_main"]
        ap["lik_diff"].append(T.ap_value(dl, c, sub_pos)); ap["ks_diff"].append(T.ap_value(dk, kc, sub_pos))
        diff_lik.append(dl); diff_ks.append(dk)
        X, Y = np.meshgrid(c, c, indexing="ij")
        i = np.unravel_index(np.argmax(dl), dl.shape)
        far = (np.hypot(X, Y) <= cfg["r_max_mpc"]) & (np.hypot(X - cfg["sub_x_mpc"], Y - cfg["sub_y_mpc"]) > 1.5)
        per.append(dict(tag=t, n=rep["n_galaxies"], lik_peak=float(dl.max()),
                        lik_peak_offset_mpc=float(np.hypot(c[i[0]] - cfg["sub_x_mpc"], c[i[1]] - cfg["sub_y_mpc"])),
                        lik_far_rms=float(dl[far].std())))
    ap = {k: np.array(v) for k, v in ap.items()}
    out = dict(n_realizations=len(done), tags=done)

    # detection
    det = {}
    for m in ("lik", "ks"):
        mu, sd = ap[f"{m}_main"].mean(), ap[f"{m}_main"].std(ddof=1)
        zs = (ap[f"{m}_sub"] - mu) / sd
        det[m] = dict(null_mean=float(mu), null_sd=float(sd), z=zs.tolist(), z_median=float(np.median(zs)),
                      z_16_84=np.percentile(zs, [16, 84]).tolist(), frac_z_gt_2=float((zs > 2).mean()),
                      frac_z_gt_3=float((zs > 3).mean()), diff_mean=float(ap[f"{m}_diff"].mean()),
                      diff_sd=float(ap[f"{m}_diff"].std(ddof=1)))
        for p_, zz in zip(per, zs):
            p_[f"z_{m}"] = float(zz)
    out["detection"] = det
    out["per_realization"] = per

    # pooled cells
    pooled = {}
    for (sc, kind), rows in cells.items():
        a = np.array(rows)
        if not len(a):
            continue
        n, est, sd, ref = a[:, 0], a[:, 1:3], a[:, 3:5], a[:, 5:7]
        pull = (est - ref) / sd
        d = dict(n_cells=len(a), pull_mean=pull.mean(0).tolist(), pull_sd=pull.std(0).tolist())
        for k in (0, 1):
            if sc == "main" and kind == "polar" and k == 1:
                continue  # g_x references are ~0 by symmetry: slope undefined
            d[f"comp{k + 1}_linear"] = wls(ref[:, k], est[:, k], sd[:, k])
            d[f"comp{k + 1}_linear_n_ge_3"] = wls(ref[n >= 3, k], est[n >= 3, k], sd[n >= 3, k])
        d["resid_vs_n"] = [dict(n=int(v), cells=int((n == v).sum()), mean_resid=(est - ref)[n == v].mean(0).tolist(),
                                err=((est - ref)[n == v].std(0) / np.sqrt(max((n == v).sum(), 1))).tolist())
                           for v in np.unique(n)]
        pooled[f"{sc}/{kind}"] = d
    out["cells"] = pooled
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1, default=float))

    for k, d in pooled.items():
        lin = {kk: (round(v["m"], 4), round(v["m_err"], 4), round(v["c"], 5), round(v["c_err"], 5)) for kk, v in d.items() if "linear" in kk}
        print(k, "pull mean", np.round(d["pull_mean"], 3), "sd", np.round(d["pull_sd"], 3), lin)
    for m in ("lik", "ks"):
        d = det[m]
        print(f"{m}: z median {d['z_median']:.2f} [16-84 {d['z_16_84'][0]:.2f}, {d['z_16_84'][1]:.2f}], "
              f"z>2 {d['frac_z_gt_2']:.2f}, z>3 {d['frac_z_gt_3']:.2f}; diff {d['diff_mean']:.4f} +- {d['diff_sd']:.4f}")

    # figures
    cfg = json.loads((BASE / done[0] / "npe/report.json").read_text())["config"]
    z0 = np.load(BASE / done[0] / "npe/kappa_maps.npz"); c = z0["centres"]; e = z0["ks_edges"]
    plot_diffmaps(cfg, c, e, np.array(diff_lik), np.array(diff_ks), z0, done)
    plot_detection(det)
    plot_cells(cells)


def plot_diffmaps(cfg, c, e, dl, dk, z0, tags):
    R = cfg["r_max_mpc"]; pix = c[1] - c[0]
    ext = [c[0] - pix / 2, c[-1] + pix / 2] * 2; kext = [e[0], e[-1]] * 2
    true_l = z0["truth_sub"] - z0["truth_main"]
    picks = np.linspace(0, len(tags) - 1, 3).astype(int)
    panels = [("true subhalo κ", true_l, ext), (f"likelihood: mean over {len(tags)} realizations", dl.mean(0), ext),
              (f"KS: mean over {len(tags)} realizations", dk.mean(0), kext)]
    panels += [(f"likelihood: {tags[i]}", dl[i], ext) for i in picks]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (title, img, ex) in zip(axes.ravel(), panels):
        im = ax.imshow(img.T, origin="lower", extent=ex, cmap="RdBu_r", vmin=-0.02, vmax=0.02)
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ls="--", color="0.45"))
        ax.plot(0, 0, "+", color="k", ms=9, mew=1.4); ax.plot(cfg["sub_x_mpc"], cfg["sub_y_mpc"], "x", color="k", ms=8, mew=1.6)
        ax.set_title(title, fontsize=10); ax.set_xlim(-R * 1.05, R * 1.05); ax.set_ylim(-R * 1.05, R * 1.05)
        ax.set_xlabel("x [Mpc]"); ax.set_ylabel("y [Mpc]")
        plt.colorbar(im, ax=ax, fraction=0.046, label=r"$\Delta\kappa$")
    fig.tight_layout(); fig.savefig(OUT / "diffmaps.png", dpi=130); plt.close(fig)


def plot_detection(det):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bins = np.linspace(-3, 7, 26)
    for m, col, lab in (("lik", "#2f6db5", "likelihood map"), ("ks", "#c4532d", "Kaiser–Squires")):
        d = det[m]
        ax.hist(d["z"], bins=bins, histtype="step", lw=2, color=col,
                label=f"{lab}: median {d['z_median']:.1f}, z>2 in {100 * d['frac_z_gt_2']:.0f}%")
    ax.axvline(2, color="0.5", ls=":"); ax.set_xlabel("single-map z at the subhalo"); ax.set_ylabel("realizations")
    ax.legend(fontsize=9, loc="upper right"); fig.tight_layout(); fig.savefig(OUT / "detection.png", dpi=130); plt.close(fig)


def plot_cells(cells):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, kind in zip(axes[:2], ("polar", "cartesian")):
        for sc, ls in (("main", "-"), ("sub", "--")):
            a = np.array(cells[(sc, kind)]); pull = (a[:, 1:3] - a[:, 5:7]) / a[:, 3:5]
            for k, col in ((0, "#2f6db5"), (1, "#c4532d")):
                ax.hist(pull[:, k], bins=np.linspace(-4, 4, 33), density=True, histtype="step", ls=ls, color=col,
                        label=f"{sc} comp {k + 1}: mean {pull[:, k].mean():+.2f}, sd {pull[:, k].std():.2f}")
        xx = np.linspace(-4, 4, 200); ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), "k", lw=1)
        ax.set_title(f"{kind} cells: pulls (NPE, all realizations)", fontsize=10); ax.set_xlabel("(stacked − reference) / σ")
        ax.legend(fontsize=7)
    ax = axes[2]
    for kind, col in (("polar", "#2f6db5"), ("cartesian", "#3c8f6a")):
        a = np.array(cells[("main", kind)]); n = a[:, 0]; r = a[:, 1] - a[:, 5]
        ns = np.unique(n); ns = ns[[(n == v).sum() >= 10 for v in ns]]
        ax.errorbar(ns + (0.1 if kind == "cartesian" else 0), [r[n == v].mean() for v in ns],
                    [r[n == v].std() / np.sqrt((n == v).sum()) for v in ns], fmt="o", color=col, label=f"{kind}, 1st comp.")
    ax.axhline(0, color="0.5"); ax.set_xlabel("galaxies in the cell"); ax.set_ylabel("mean stacked − reference")
    ax.set_title("bias vs occupancy (main scenario)", fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "cells.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
