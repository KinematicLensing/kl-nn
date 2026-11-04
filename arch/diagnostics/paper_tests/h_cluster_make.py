#!/usr/bin/env python3
"""H: cluster lensing test, step 1 -- background galaxies and the two shear fields.

Places a Poisson number of background galaxies uniformly within R < r_max of a
cluster at z_l, all simulated at z_s = 0.3 (no tomography: every source behind the
lens counts). Each galaxy takes the nuisance parameters of a random Xu3 measured-
axis test galaxy (hlr >= 1", TF vcirc, catalog SNRs) and the shear of two
scenarios at its position:

  main : one NFW halo at the origin
  sub  : the same halo plus an NFW subhalo

Halos come from galsim.NFWHalo (M200c in Msun/h, its own Omega_m / Omega_lam
cosmology; positions in arcsec, x = image column, y = image row, the frame the
simulator applies shear in). Halos are superposed in gamma and kappa, then
g = gamma / (1 - kappa); reduced shears are never added. Concentrations follow
Duffy et al. (2008), 200c full sample.

Both scenarios share positions, nuisance parameters and targeting seeds, so their
data differ only through the shear. Writes, under $S/samples/:
  paper_tests_H_cluster_<tag>_{main,sub}.csv   simulator inputs (ID = row index)
  paper_tests_H_cluster_<tag>_manifest.csv     positions, true g and kappa
  paper_tests_H_cluster_<tag>_config.json      all settings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
import galsim
import numpy as np
import pandas as pd

S = Path("/ocean/projects/phy250048p/shared")
SAMPLES = S / "samples"
SOURCE = SAMPLES / "test_100k_simv3_cosi_xu3_tf_measaxis.csv"
PARAMETERS = ["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"]
AUXILIARY = ["rmag_true", "halpha_flux_true", "image_snr", "central_halpha_snr",
             "fiber_layout", "observation_model_version"]
TARGETING_SEED_OFFSET = 900_000_000  # data_generate/fiber_targeting.py
PRIOR_BOX = 0.1


def duffy08_c200c(m200c_msun, z, h):
    """Duffy et al. (2008) c200c, full sample: A=5.71, B=-0.084, C=-0.47, Mpiv=2e12 Msun/h."""
    return 5.71 * (m200c_msun * h / 2e12) ** -0.084 * (1.0 + z) ** -0.47


def lensing(halos, x, y, z_s):
    """Reduced shear and convergence of superposed halos at (x, y) arcsec."""
    gamma1 = np.zeros_like(x)
    gamma2 = np.zeros_like(x)
    kappa = np.zeros_like(x)
    for halo in halos:
        a, b = halo.getShear((x, y), z_s, reduced=False)
        gamma1 += np.asarray(a)
        gamma2 += np.asarray(b)
        kappa += np.asarray(halo.getConvergence((x, y), z_s))
    return gamma1 / (1.0 - kappa), gamma2 / (1.0 - kappa), kappa


def build_halos(cfg):
    h, om = cfg["h"], cfg["omega_m"]
    main = galsim.NFWHalo(cfg["m_main"] * h, cfg["c_main"], cfg["z_l"],
                          omega_m=om, omega_lam=1.0 - om)
    sub = galsim.NFWHalo(cfg["m_sub"] * h, cfg["c_sub"], cfg["z_l"],
                         halo_pos=galsim.PositionD(cfg["sub_x_arcsec"], cfg["sub_y_arcsec"]),
                         omega_m=om, omega_lam=1.0 - om)
    return main, sub


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="s3_r0")
    p.add_argument("--density", type=float, default=0.108,
                   help="sources per arcmin^2 behind the lens (Xu sample 3, z_l = 0.13: 0.108)")
    p.add_argument("--r-max-mpc", type=float, default=3.0)
    p.add_argument("--z-l", type=float, default=0.13)
    p.add_argument("--z-s", type=float, default=0.3)
    p.add_argument("--m-main", type=float, default=1e15, help="M200c [Msun]")
    p.add_argument("--sub-frac", type=float, default=0.3)
    p.add_argument("--sub-x-mpc", type=float, default=0.5)
    p.add_argument("--sub-y-mpc", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261023)
    a = p.parse_args()

    h, om = Planck18.h, float(Planck18.Om0)
    kpc_per_arcsec = (Planck18.angular_diameter_distance(a.z_l)
                      * (1 * u.arcsec).to(u.rad).value).to(u.kpc).value
    r_max_arcsec = a.r_max_mpc * 1000.0 / kpc_per_arcsec
    area_arcmin2 = np.pi * (r_max_arcsec / 60.0) ** 2
    m_sub = a.sub_frac * a.m_main
    cfg = dict(
        tag=a.tag, density_per_arcmin2=a.density, r_max_mpc=a.r_max_mpc,
        r_max_arcsec=r_max_arcsec, area_arcmin2=area_arcmin2, z_l=a.z_l, z_s=a.z_s,
        h=h, omega_m=om, kpc_per_arcsec=kpc_per_arcsec,
        m_main=a.m_main, c_main=duffy08_c200c(a.m_main, a.z_l, h),
        m_sub=m_sub, c_sub=duffy08_c200c(m_sub, a.z_l, h),
        sub_x_mpc=a.sub_x_mpc, sub_y_mpc=a.sub_y_mpc,
        sub_x_arcsec=a.sub_x_mpc * 1000.0 / kpc_per_arcsec,
        sub_y_arcsec=a.sub_y_mpc * 1000.0 / kpc_per_arcsec,
        seed=a.seed, source_table=str(SOURCE),
        mass_definition="M200c (Msun; passed to GalSim as Msun/h)",
        concentration="Duffy et al. 2008, 200c full sample",
    )

    rng = np.random.default_rng(a.seed)
    n = int(rng.poisson(a.density * area_arcmin2))
    r = r_max_arcsec * np.sqrt(rng.uniform(0.0, 1.0, n))
    phi = rng.uniform(-np.pi, np.pi, n)
    x, y = r * np.cos(phi), r * np.sin(phi)
    main_halo, sub_halo = build_halos(cfg)
    g1m, g2m, km = lensing([main_halo], x, y, a.z_s)
    g1s, g2s, ks = lensing([main_halo, sub_halo], x, y, a.z_s)
    for g in (g1m, g2m, g1s, g2s):
        if not np.all(np.abs(g) < PRIOR_BOX):
            raise SystemExit("a galaxy's shear leaves the training prior |g_k| < 0.1")

    source = pd.read_csv(SOURCE, float_precision="round_trip")
    rows = rng.choice(len(source), size=n, replace=False)
    base = source.iloc[rows].reset_index(drop=True)
    out = {}
    for name, (g1, g2) in (("main", (g1m, g2m)), ("sub", (g1s, g2s))):
        table = base[["ID", *PARAMETERS, *AUXILIARY]].copy()
        table["targeting_seed"] = TARGETING_SEED_OFFSET + table["ID"].astype(int)
        table["fiber_layout"] = "measured_axis"
        table["g1"], table["g2"] = g1, g2
        table.insert(0, "source_id", table.pop("ID"))
        table.insert(0, "ID", np.arange(n))
        table = table.drop(columns=["source_id"])
        out[name] = table
        table.to_csv(SAMPLES / f"paper_tests_H_cluster_{a.tag}_{name}.csv", index=False)
    kpm = kpc_per_arcsec / 1000.0
    manifest = pd.DataFrame(dict(
        ID=np.arange(n), source_id=base["ID"].astype(int), x_arcsec=x, y_arcsec=y,
        x_mpc=x * kpm, y_mpc=y * kpm, r_mpc=r * kpm, phi=phi,
        g1_main=g1m, g2_main=g2m, kappa_main=km, g1_sub=g1s, g2_sub=g2s, kappa_sub=ks))
    manifest.to_csv(SAMPLES / f"paper_tests_H_cluster_{a.tag}_manifest.csv", index=False)
    cfg["n_galaxies"] = n
    (SAMPLES / f"paper_tests_H_cluster_{a.tag}_config.json").write_text(json.dumps(cfg, indent=1))
    dg = np.hypot(g1s - g1m, g2s - g2m)
    print(f"{n} galaxies (expected {a.density * area_arcmin2:.1f}); c_main={cfg['c_main']:.2f}, "
          f"c_sub={cfg['c_sub']:.2f}; max|g| main {np.abs([g1m, g2m]).max():.3f}, sub {np.abs([g1s, g2s]).max():.3f}; "
          f"galaxies with |Δg|>0.01: {(dg > 0.01).sum()}")


if __name__ == "__main__":
    main()
