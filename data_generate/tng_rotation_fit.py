from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit

from kl_pipe.parameters import ImagePars
from kl_pipe.tng import TNG50Galaxy, TNGDataVectorGenerator, TNGRenderConfig
from kl_pipe.utils import build_map_grid_from_image_pars


@dataclass(frozen=True)
class RotationFitResult:
    v0: float
    vcirc: float
    rscale: float
    n_profile_bins: int
    rmse: float


def _arctan_velocity(r: np.ndarray, v0: float, vcirc: float, rscale: float) -> np.ndarray:
    return v0 + (2.0 / np.pi) * vcirc * np.arctan(r / rscale)


def _weighted_bin_profile(
    r: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    n_bins: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_min = float(np.min(r))
    r_max = float(np.max(r))
    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
        raise ValueError('Invalid radial range for velocity-profile binning.')

    edges = np.linspace(r_min, r_max, n_bins + 1)
    bin_id = np.digitize(r, edges) - 1

    r_bin = []
    v_bin = []
    w_bin = []
    for k in range(n_bins):
        mask = bin_id == k
        if not np.any(mask):
            continue
        wk = np.sum(w[mask])
        if wk <= 0:
            continue
        rk = np.average(r[mask], weights=w[mask])
        vk = np.average(v[mask], weights=w[mask])
        r_bin.append(rk)
        v_bin.append(vk)
        w_bin.append(wk)

    return np.asarray(r_bin), np.asarray(v_bin), np.asarray(w_bin)


def fit_galaxy_rotation_params(
    gal_idx: int,
    target_redshift: float = 0.3,
    map_shape: Tuple[int, int] = (96, 96),
    pixel_scale: float = 0.2,
    major_axis_half_width_px: float = 2.0,
    n_profile_bins: int = 48,
    min_profile_bins: int = 12,
) -> RotationFitResult:
    """Fit edge-on arctan rotation parameters for one TNG galaxy.

    The fit is always measured at edge-on orientation (i=pi/2, cosi=0) with
    theta_int=0 and no shear, so vcirc is not suppressed by projection.
    """
    tng = TNG50Galaxy(index=gal_idx)
    galaxy = tng.get_galaxy()
    generator = TNGDataVectorGenerator(galaxy)

    image_pars = ImagePars(shape=map_shape, pixel_scale=pixel_scale, indexing='ij')
    edge_on_pars: Dict[str, float] = {
        'theta_int': 0.0,
        'cosi': 0.0,
        'x0': 0.0,
        'y0': 0.0,
        'g1': 0.0,
        'g2': 0.0,
    }
    config = TNGRenderConfig(
        image_pars=image_pars,
        band='r',
        use_dusted=True,
        center_on_peak=True,
        use_native_orientation=False,
        pars=edge_on_pars,
        use_cic_gridding=True,
        target_redshift=target_redshift,
        preserve_gas_stellar_offset=True,
    )

    intensity_map, _ = generator.generate_intensity_map(config, snr=None)
    velocity_map, _ = generator.generate_velocity_map(
        config,
        snr=None,
        intensity_map=intensity_map,
    )

    x_grid, y_grid = build_map_grid_from_image_pars(image_pars, unit='arcsec', centered=True)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)

    strip_half_width = major_axis_half_width_px * pixel_scale
    valid = (
        np.isfinite(velocity_map)
        & np.isfinite(intensity_map)
        & (intensity_map > 0)
        & (np.abs(y_grid) <= strip_half_width)
    )

    if np.count_nonzero(valid) < max(32, min_profile_bins * 2):
        raise RuntimeError(
            f'Insufficient valid points to fit rotation curve for galaxy {gal_idx}.'
        )

    r_vals = x_grid[valid]
    v_vals = np.asarray(velocity_map, dtype=np.float64)[valid]
    w_vals = np.asarray(intensity_map, dtype=np.float64)[valid]
    w_vals = np.clip(w_vals, 0.0, None) + 1e-9

    r_prof, v_prof, w_prof = _weighted_bin_profile(r_vals, v_vals, w_vals, n_profile_bins)
    if r_prof.size < min_profile_bins:
        raise RuntimeError(
            f'Only {r_prof.size} usable profile bins for galaxy {gal_idx}; need at least {min_profile_bins}.'
        )

    center_mask = np.abs(r_prof) <= np.percentile(np.abs(r_prof), 20)
    if np.any(center_mask):
        v0_init = float(np.average(v_prof[center_mask], weights=w_prof[center_mask]))
    else:
        v0_init = float(np.average(v_prof, weights=w_prof))

    vcirc_init = max(20.0, 0.5 * float(np.percentile(v_prof, 95) - np.percentile(v_prof, 5)))
    rscale_init = max(pixel_scale, float(np.percentile(np.abs(r_prof), 35)))

    r_abs_max = max(pixel_scale, float(np.max(np.abs(r_prof))))
    lower = (-400.0, 1.0, max(pixel_scale * 0.25, 1e-3))
    upper = (400.0, 900.0, 2.0 * r_abs_max)

    sigma = 1.0 / np.sqrt(np.clip(w_prof, 1e-9, None))
    sigma /= np.median(sigma)

    popt, _ = curve_fit(
        _arctan_velocity,
        r_prof,
        v_prof,
        p0=(v0_init, vcirc_init, rscale_init),
        bounds=(lower, upper),
        sigma=sigma,
        absolute_sigma=False,
        maxfev=30000,
    )

    residuals = v_prof - _arctan_velocity(r_prof, *popt)
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return RotationFitResult(
        v0=float(popt[0]),
        vcirc=float(popt[1]),
        rscale=float(popt[2]),
        n_profile_bins=int(r_prof.size),
        rmse=rmse,
    )
