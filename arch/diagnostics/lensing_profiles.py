"""NFW cluster lensing profile for the annulus stacking test.

The tangential shear and convergence of an NFW halo follow Wright & Brainerd
(2000, ApJ 534, 34). M200 and r200 are defined at 200 times the critical
density at the lens redshift. The cosmology is astropy's Planck18. Distances
are in physical Mpc and surface densities in Msun / Mpc^2.
"""

from __future__ import annotations

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18


def sigma_crit(z_lens: float, z_source: float, cosmology=Planck18) -> float:
    """Critical surface density [Msun / Mpc^2] (physical)."""

    d_l = cosmology.angular_diameter_distance(z_lens)
    d_s = cosmology.angular_diameter_distance(z_source)
    d_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source)
    value = const.c**2 / (4 * np.pi * const.G) * d_s / (d_l * d_ls)
    return float(value.to(u.Msun / u.Mpc**2).value)


def _nfw_scale(m200: float, c200: float, z_lens: float, cosmology=Planck18):
    rho_c = cosmology.critical_density(z_lens).to(u.Msun / u.Mpc**3).value
    r200 = (3.0 * m200 / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0)
    r_s = r200 / c200
    delta_c = (200.0 / 3.0) * c200**3 / (np.log(1.0 + c200) - c200 / (1.0 + c200))
    return r_s, delta_c * rho_c, r200


def _g_aux(x: np.ndarray) -> np.ndarray:
    """2/sqrt(|1-x^2|) * {artanh, arctan}(sqrt(|1-x|/(1+x))), equal to 1 at x = 1."""

    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    lo, hi = x < 1.0, x > 1.0
    out[lo] = 2.0 / np.sqrt(1.0 - x[lo] ** 2) * np.arctanh(np.sqrt((1.0 - x[lo]) / (1.0 + x[lo])))
    out[hi] = 2.0 / np.sqrt(x[hi] ** 2 - 1.0) * np.arctan(np.sqrt((x[hi] - 1.0) / (1.0 + x[hi])))
    return out


def nfw_sigma(radius_mpc, m200, c200, z_lens, cosmology=Planck18):
    """Surface density Sigma(R) and mean enclosed surface density [Msun/Mpc^2]."""

    r_s, rho_s, _ = _nfw_scale(m200, c200, z_lens, cosmology)
    x = np.asarray(radius_mpc, dtype=float) / r_s
    aux = _g_aux(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(np.isclose(x, 1.0), 1.0 / 3.0, (1.0 - aux) / (x**2 - 1.0))
    sigma = 2.0 * r_s * rho_s * f
    mean_sigma = 4.0 * r_s * rho_s * (np.log(x / 2.0) + aux) / x**2
    return sigma, mean_sigma


def nfw_reduced_tangential_shear(
    radius_mpc, *, m200, c200, z_lens, z_source, cosmology=Planck18
):
    """Return (g_t, gamma_t, kappa) at projected radius R [physical Mpc]."""

    sigma, mean_sigma = nfw_sigma(radius_mpc, m200, c200, z_lens, cosmology)
    s_crit = sigma_crit(z_lens, z_source, cosmology)
    kappa = sigma / s_crit
    gamma_t = (mean_sigma - sigma) / s_crit
    return gamma_t / (1.0 - kappa), gamma_t, kappa


def r200_mpc(m200, c200, z_lens, cosmology=Planck18) -> float:
    return float(_nfw_scale(m200, c200, z_lens, cosmology)[2])
