#!/usr/bin/env python3
"""Likelihood-ratio significance of the subhalo from the NFW halo fit, per realization.

For each realization and data scenario, maximize the per-galaxy likelihood product (h_cluster_halo_fit.py) for
  main       log10 M_main
  main+sub   log10 M_main and M_sub >= 0 at the subhalo's true position
and report sigma = sqrt(2 [ln L_max(main+sub) - ln L_max(main)]): the one-sided significance of a subhalo mass > 0
(Wilks / Chernoff: on data without a subhalo, half the realizations give 0 and the rest |N(0, 1)|).
Writes ensemble/halo_sigma.json with these and the single-map z of the two kappa maps for comparison.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h_cluster_analysis import AREA, FLOOR, HI, LO, S, SCENARIOS, interp  # noqa: E402
from h_cluster_make import duffy08_c200c, lensing  # noqa: E402

H = S / "reports/paper-tests-1m/H_cluster"
TAGS = [f"s3_sub1_n{r:02d}" for r in range(1, 51)]


def make_loglike(logl, man, cfg):
    import galsim
    x, y = man["x_arcsec"].to_numpy(), man["y_arcsec"].to_numpy()
    h, om, zl, zs = cfg["h"], cfg["omega_m"], cfg["z_l"], cfg["z_s"]
    idx = np.arange(len(man))

    def halo(m, pos):
        return galsim.NFWHalo(m * h, duffy08_c200c(m, zl, h), zl, halo_pos=galsim.PositionD(*pos),
                              omega_m=om, omega_lam=1 - om)

    def loglike(log10_m_main, m_sub=0.0):
        halos = [halo(10 ** log10_m_main, (0.0, 0.0))]
        if m_sub > 0:
            halos.append(halo(m_sub, (cfg["sub_x_arcsec"], cfg["sub_y_arcsec"])))
        g1, g2, _ = lensing(halos, x, y, zs)
        inside = (np.abs(g1) < HI) & (np.abs(g2) < HI)
        ll = np.where(inside, interp(logl, idx, np.clip(g1, LO, HI), np.clip(g2, LO, HI)), math.log(FLOOR / AREA))
        return float(ll.sum())
    return loglike


def fit_max(loglike):
    r0 = minimize_scalar(lambda a: -loglike(a), bounds=(13.5, 16.0), method="bounded", options=dict(xatol=1e-4))
    best = (-r0.fun, r0.x, 0.0)
    for s0 in (1.0, 3.0, 6.0):  # M_sub in units of 1e14 Msun, starts around the truth (3)
        r = minimize(lambda p: -loglike(p[0], p[1] * 1e14), x0=[r0.x, s0], method="L-BFGS-B",
                     bounds=[(13.5, 16.0), (0.0, 30.0)])
        if -r.fun > best[0]:
            best = (-r.fun, r.x[0], r.x[1])
    return dict(lnL_main=-r0.fun, m_main_only=r0.x, lnL_sub=best[0], m_main=best[1], m_sub_1e14=best[2],
                sigma=math.sqrt(max(0.0, 2 * (best[0] + r0.fun))))


def main():
    det = json.loads((H / "ensemble/summary.json").read_text())["per_realization"]
    det = {d["tag"]: d for d in det}
    rows = []
    for t in TAGS:
        cfg = json.loads((S / "samples" / f"paper_tests_H_cluster_{t}_config.json").read_text())
        man = pd.read_csv(S / "samples" / f"paper_tests_H_cluster_{t}_manifest.csv")
        z = np.load(H / t / "npe/loglike_grids.npz")
        row = dict(tag=t, z_lik=det[t]["z_lik"], z_ks=det[t]["z_ks"])
        for sc in SCENARIOS:
            f = fit_max(make_loglike(np.asarray(z[sc]), man, cfg))
            row.update({f"{sc}_{k}": float(v) for k, v in f.items()})
        rows.append(row)
        print(t, f"sigma(sub data) {row['sub_sigma']:.2f}  sigma(main data) {row['main_sigma']:.2f}  "
                 f"z_lik {row['z_lik']:.2f}  z_ks {row['z_ks']:.2f}", flush=True)
    out = dict(rows=rows)
    for k in ("sub_sigma", "main_sigma", "z_lik", "z_ks"):
        v = np.array([r[k] for r in rows])
        out[k] = dict(median=float(np.median(v)), q16=float(np.percentile(v, 16)), q84=float(np.percentile(v, 84)),
                      mean=float(v.mean()), frac_gt_2=float((v > 2).mean()), frac_gt_3=float((v > 3).mean()))
        print(k, {kk: round(vv, 2) for kk, vv in out[k].items()})
    (H / "ensemble/halo_sigma.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
