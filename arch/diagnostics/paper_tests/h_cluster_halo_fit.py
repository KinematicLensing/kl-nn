#!/usr/bin/env python3
"""Parametric NFW fit to the per-galaxy NPE likelihoods (no cells, no kappa pixels).

The Weighing the Giants approach (Applegate et al. 2014, Sec. 5, Eqs. 8-11): the posterior for the halo parameters is
the prior times the product over galaxies of each galaxy's likelihood evaluated at the model's reduced shear at that
galaxy's position. The main-halo centre is fixed (WtG anchors on the X-ray centroid); concentrations follow
Duffy et al. (2008). Three models per data scenario:

  main            log10 M_main
  main+sub        log10 M_main, log10 M_sub, subhalo at its true position
  main+sub_free   log10 M_main, log10 M_sub, subhalo position (x, y) free within |x|, |y| < 2.5 Mpc

Reads <tag>/npe/loglike_grids.npz written by h_cluster_analysis.py; writes <tag>/npe/halo_fit.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h_cluster_analysis import AREA, FLOOR, HI, LO, S, SCENARIOS, interp  # noqa: E402
from h_cluster_make import duffy08_c200c, lensing  # noqa: E402

MODELS = ("main", "main+sub", "main+sub_free")
BOX_MPC = 2.5


def fit(logl, man, cfg, model, n_live=500, seed=11):
    from nautilus import Prior, Sampler

    import galsim
    x, y = man["x_arcsec"].to_numpy(), man["y_arcsec"].to_numpy()
    h, om, zl, zs = cfg["h"], cfg["omega_m"], cfg["z_l"], cfg["z_s"]
    idx = np.arange(len(man))
    box = BOX_MPC * 1e3 / cfg["kpc_per_arcsec"]
    prior = Prior()
    prior.add_parameter("log10_m_main", dist=(13.5, 16.0))
    if model != "main":
        prior.add_parameter("log10_m_sub", dist=(12.0, 15.5))
    if model == "main+sub_free":
        prior.add_parameter("x_sub_arcsec", dist=(-box, box))
        prior.add_parameter("y_sub_arcsec", dist=(-box, box))

    def halo(log10_m, pos):
        m = 10 ** log10_m
        return galsim.NFWHalo(m * h, duffy08_c200c(m, zl, h), zl, halo_pos=galsim.PositionD(*pos),
                              omega_m=om, omega_lam=1 - om)

    def loglike(par):
        halos = [halo(par["log10_m_main"], (0.0, 0.0))]
        if "log10_m_sub" in par:
            pos = (par.get("x_sub_arcsec", cfg["sub_x_arcsec"]), par.get("y_sub_arcsec", cfg["sub_y_arcsec"]))
            halos.append(halo(par["log10_m_sub"], pos))
        g1, g2, _ = lensing(halos, x, y, zs)
        inside = (np.abs(g1) < HI) & (np.abs(g2) < HI)
        ll = np.where(inside, interp(logl, idx, np.clip(g1, LO, HI), np.clip(g2, LO, HI)),
                      math.log(FLOOR / AREA))
        return float(ll.sum())

    if len(prior.keys) == 1:
        # Nautilus needs >= 2 parameters: integrate the 1D posterior on a fine grid instead
        lo, hi = prior.dists[0].support()
        pts = np.linspace(lo, hi, 2001)[:, None]
        ll = np.array([loglike({prior.keys[0]: v}) for v in pts[:, 0]])
        log_z = float(np.logaddexp.reduce(ll) - math.log(len(ll)))   # uniform prior: mean of L
        w = np.exp(ll - ll.max()); w /= w.sum()
        n_eff = float(1 / np.sum(w ** 2))
    else:
        sampler = Sampler(prior, loglike, n_live=n_live, seed=seed)
        sampler.run(verbose=False)
        pts, logw, _ = sampler.posterior()
        w = np.exp(logw - logw.max()); w /= w.sum()
        log_z, n_eff = float(sampler.log_z), float(sampler.n_eff)
    summ = {}
    for j, nm in enumerate(prior.keys):
        o = np.argsort(pts[:, j]); cw = np.cumsum(w[o])
        q = [float(pts[o, j][np.searchsorted(cw, t)]) for t in (0.16, 0.5, 0.84)]
        summ[nm] = dict(q16=q[0], median=q[1], q84=q[2])
    # equal-weight resample for plotting
    rng = np.random.default_rng(seed)
    draw = pts[rng.choice(len(pts), size=min(4000, len(pts)), p=w)]
    return dict(log_z=log_z, n_eff=n_eff, params=summ,
                names=list(prior.keys), samples=draw.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_sub1")
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    a = p.parse_args()
    cfg = json.loads((S / "samples" / f"paper_tests_H_cluster_{a.tag}_config.json").read_text())
    man = pd.read_csv(S / "samples" / f"paper_tests_H_cluster_{a.tag}_manifest.csv")
    out = S / "reports/paper-tests-1m/H_cluster" / a.tag / "npe"
    z = np.load(out / "loglike_grids.npz")
    report = dict(tag=a.tag, truth=dict(log10_m_main=math.log10(cfg["m_main"]), log10_m_sub=math.log10(cfg["m_sub"]),
                                        x_sub_arcsec=cfg["sub_x_arcsec"], y_sub_arcsec=cfg["sub_y_arcsec"]))
    for sc in SCENARIOS:
        logl = np.asarray(z[sc])
        report[sc] = {mdl: fit(logl, man, cfg, mdl) for mdl in a.models}
        for mdl in a.models:
            r = report[sc][mdl]
            print(sc, mdl, f"log_z={r['log_z']:.2f}",
                  {k: f"{v['median']:.3f} [{v['q16']:.3f}, {v['q84']:.3f}]" for k, v in r["params"].items()}, flush=True)
        for mdl in a.models[1:]:
            report[sc][f"ln_B_{mdl}_vs_main"] = report[sc][mdl]["log_z"] - report[sc]["main"]["log_z"]
            print(sc, f"ln B({mdl} vs main) = {report[sc][f'ln_B_{mdl}_vs_main']:.2f}", flush=True)
    (out / "halo_fit.json").write_text(json.dumps(report))
    print("wrote", out / "halo_fit.json")


if __name__ == "__main__":
    main()
