#!/usr/bin/env python3
"""H: choose the kappa-map settings on synthetic data only (before looking at the NPE run).

For each synthetic noise realization (Gaussian likelihoods with the galaxies' real NPE shear
covariances; same noise in both scenarios), reconstruct kappa for the main and main+sub
data, and the residual = kappa_rec - true main-halo kappa. Score every setting by:

  contrast   (mean residual in a 0.5 Mpc aperture on the subhalo, sub data
              minus main data) / (scatter of that aperture value over realizations, main data)
  false_peak fraction of null apertures (main data, 16 positions on rings at 1 and 2 Mpc away
              from the subhalo) whose residual exceeds the mean subhalo signal
  rms_err    RMS of (kappa_rec - true kappa) over pixels inside the field, sub data

Likelihood map: pixel {0.5, 0.75, 1.0} Mpc x sigma_kappa {0.03, 0.05, 0.1} x smoothness
length {0.5, 1, 2} pixels. Kaiser-Squires on the Cartesian cells: smoothing {0.5, 0.7, 1.0,
1.5} cells. Choice rule (fixed in advance): the highest contrast among settings with
false_peak <= 0.10; if none qualifies, the lowest false_peak.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h_cluster_analysis as H  # noqa: E402

PIX = (0.5, 0.75, 1.0)
SIGK = (0.03, 0.05, 0.1)
SMOOTH = (0.5, 1.0, 2.0)
KS_SMOOTH = (0.5, 0.7, 1.0, 1.5)
AP_R = 0.5


def disk_points(x0, y0, r=AP_R, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rr = r * np.sqrt(rng.uniform(0, 1, n)); th = rng.uniform(0, 2 * np.pi, n)
    return x0 + rr * np.cos(th), y0 + rr * np.sin(th)


def sample_map(img, centres, x, y):
    """Bilinear value of a pixel map (axis 0 = x) at points (x, y)."""
    pix = centres[1] - centres[0]
    u = np.clip((x - centres[0]) / pix, 0, len(centres) - 1 - 1e-9)
    v = np.clip((y - centres[0]) / pix, 0, len(centres) - 1 - 1e-9)
    i, j = u.astype(int), v.astype(int); tu, tv = u - i, v - j
    return ((1 - tu) * (1 - tv) * img[i, j] + tu * (1 - tv) * img[i + 1, j]
            + (1 - tu) * tv * img[i, j + 1] + tu * tv * img[i + 1, j + 1])


def apertures(cfg):
    sx, sy = cfg["sub_x_mpc"], cfg["sub_y_mpc"]
    phi_sub = np.arctan2(sy, sx)
    null = []
    for r in (1.0, 2.0):
        for k in range(8):
            ph = phi_sub + 2 * np.pi * (k + 0.5) / 8
            px, py = r * np.cos(ph), r * np.sin(ph)
            if np.hypot(px - sx, py - sy) > 1.2:
                null.append((px, py))
    return (sx, sy), null


def ap_value(img, centres, pos):
    x, y = disk_points(*pos)
    return float(sample_map(img, centres, x, y).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_sub1")
    p.add_argument("--realizations", type=int, default=20)
    a = p.parse_args()
    cfg = json.loads((H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_config.json").read_text())
    man = pd.read_csv(H.S / "samples" / f"paper_tests_H_cluster_{a.tag}_manifest.csv")
    out = H.S / "reports/paper-tests-1m/H_cluster" / a.tag / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    truths = {sc: man[[f"g1_{sc}", f"g2_{sc}"]].to_numpy() for sc in H.SCENARIOS}
    covs = H.xu3_shear_covariance(man["source_id"].to_numpy())
    ccells, cmem, cedges = H.cartesian_cells(man, cfg)
    ks_centres = 0.5 * (cedges[1:] + cedges[:-1]); nside = len(ks_centres)
    sub_pos, null_pos = apertures(cfg)
    lik_settings = list(itertools.product(PIX, SIGK, SMOOTH))
    vals = {("lik",) + s: {"sub": [], "main_sub_ap": [], "null": [], "rms": []} for s in lik_settings}
    vals.update({("ks", s): {"sub": [], "main_sub_ap": [], "null": [], "rms": []} for s in KS_SMOOTH})
    truth_cache = {}

    def truth_k(centres, sc):
        key = (len(centres), round(float(centres[1] - centres[0]), 4), sc)
        if key not in truth_cache:
            truth_cache[key] = H.model_kappa(cfg, centres, cfg["m_main"], cfg["m_sub"] if sc == "sub" else None)
        return truth_cache[key]

    for r in range(a.realizations):
        draws = H.synthetic(truths, covs, seed=cfg["seed"] + 5000 + r, nsamp=4000)
        logls = {sc: H.kde_loglike_grid(*draws[sc]) for sc in H.SCENARIOS}
        for (pix, sk, sm) in lik_settings:
            maps = {}
            for sc in H.SCENARIOS:
                maps[sc], centres = H.likelihood_kappa(logls[sc], man, cfg, pix_mpc=pix, sigma_kappa=sk, smooth_len=sm)
            tm = truth_k(centres, "main"); ts = truth_k(centres, "sub")
            res_s, res_m = maps["sub"] - tm, maps["main"] - tm
            X, Y = np.meshgrid(centres, centres, indexing="ij"); inside = np.hypot(X, Y) <= cfg["r_max_mpc"]
            v = vals[("lik", pix, sk, sm)]
            v["sub"].append(ap_value(res_s, centres, sub_pos)); v["main_sub_ap"].append(ap_value(res_m, centres, sub_pos))
            v["null"].append([ap_value(res_m, centres, q) for q in null_pos])
            v["rms"].append(float(np.sqrt(((maps["sub"] - ts)[inside] ** 2).mean())))
        # Kaiser-Squires on the Cartesian cell stacks
        ksin = {}
        for sc in H.SCENARIOS:
            _, cg = H.grid_moments(logls[sc])
            rows = H.analyse_cells(logls[sc], cg, truths[sc], cmem, len(ccells), lambda i: np.eye(2))
            g1m = np.zeros((nside, nside)); g2m = np.zeros((nside, nside))
            for row in rows:
                c = ccells[row["cell"]]; g1m[c["ix"], c["iy"]], g2m[c["ix"], c["iy"]] = row["est"]
            ksin[sc] = (g1m, g2m)
        tm = truth_k(ks_centres, "main"); ts = truth_k(ks_centres, "sub")
        X, Y = np.meshgrid(ks_centres, ks_centres, indexing="ij"); inside = np.hypot(X, Y) <= cfg["r_max_mpc"]
        for sm in KS_SMOOTH:
            m = {sc: H.kaiser_squires(*ksin[sc], float(cedges[1] - cedges[0]), smooth_pix=sm) for sc in H.SCENARIOS}
            res_s, res_m = m["sub"] - tm, m["main"] - tm
            v = vals[("ks", sm)]
            v["sub"].append(ap_value(res_s, ks_centres, sub_pos)); v["main_sub_ap"].append(ap_value(res_m, ks_centres, sub_pos))
            v["null"].append([ap_value(res_m, ks_centres, q) for q in null_pos])
            v["rms"].append(float(np.sqrt(((m["sub"] - ts)[inside] ** 2).mean())))
        print(f"realization {r + 1}/{a.realizations}", flush=True)

    rows = []
    for key, v in vals.items():
        sub, msub, null = np.array(v["sub"]), np.array(v["main_sub_ap"]), np.array(v["null"])
        signal = sub.mean() - msub.mean()
        noise = msub.std()
        rows.append(dict(method=key[0], setting=list(key[1:]), signal=float(signal), noise=float(noise),
                         contrast=float(signal / noise) if noise > 0 else None,
                         false_peak=float(((null - null.mean(axis=0)) > signal).mean()),
                         rms_err=float(np.mean(v["rms"]))))
    def choose(method):
        cand = [r for r in rows if r["method"] == method]
        ok = [r for r in cand if r["false_peak"] <= 0.10]
        c = lambda r: -np.inf if r["contrast"] is None else r["contrast"]
        return max(ok, key=c) if ok else min(cand, key=lambda r: r["false_peak"])
    result = dict(rule="max contrast with false_peak <= 0.10, else min false_peak",
                  aperture_mpc=AP_R, realizations=a.realizations, sub_pos=sub_pos, null_pos=null_pos,
                  rows=rows, chosen=dict(lik=choose("lik"), ks=choose("ks")))
    (out / "tuning.json").write_text(json.dumps(result, indent=1))
    for r in sorted(rows, key=lambda r: -(r["contrast"] if r["contrast"] is not None else -9)):
        print(r["method"], r["setting"], f"contrast {r['contrast']} fp {r['false_peak']:.2f} rms {r['rms_err']:.4f}")
    print("chosen:", json.dumps(result["chosen"]))


if __name__ == "__main__":
    main()
