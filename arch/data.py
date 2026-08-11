import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
import torch
from torchvision import transforms
from astropy.cosmology import Planck18 as cosmo

import config
from utils import resolve_feature_index

@dataclass
class TFCalculator:
    slope: float
    intercept: float
    scatter: float = 0.1

    def vcirc_to_mag(self, vcirc):
        """
        Calculates apparent magnitude from circular velocity.
        Supports scalar, numpy array, and PyTorch tensor.
        """
        if isinstance(vcirc, torch.Tensor):
            return self.slope * torch.log10(vcirc) + self.intercept
        elif isinstance(vcirc, np.ndarray):
            return self.slope * np.log10(vcirc) + self.intercept
        else:
            import math
            return self.slope * math.log10(vcirc) + self.intercept

    def mag_to_vcirc(self, mag):
        """
        Calculates circular velocity from apparent magnitude.
        Supports scalar, numpy array, and PyTorch tensor.
        """
        if isinstance(mag, torch.Tensor):
            return 10.0 ** ((mag - self.intercept) / self.slope)
        elif isinstance(mag, np.ndarray):
            return 10.0 ** ((mag - self.intercept) / self.slope)
        else:
            return 10.0 ** ((mag - self.intercept) / self.slope)
        
    def sample_mag_from_vcirc(self, vcirc, rng=None):
        """
        Sample a physically consistent apparent magnitude given a vcirc value,
        incorporating intrinsic astrophysical scatter.
        Compatible with scalars, numpy arrays, and PyTorch tensors.
        """
        # 1. Compute the clean mean magnitude
        m_mean = self.vcirc_to_mag(vcirc)

        # 2. Translate the intrinsic dex scatter to magnitude space:
        # sigma_m = |slope| * sigma_intrinsic_dex
        sigma_m_intrinsic = abs(self.slope) * self.scatter

        if isinstance(vcirc, torch.Tensor):
            noise = torch.randn(
                m_mean.shape,
                device=m_mean.device,
                dtype=m_mean.dtype,
                generator=rng,
            )
            return m_mean + noise * sigma_m_intrinsic

        normal = np.random.normal if rng is None else rng.normal
        size = m_mean.shape if isinstance(vcirc, np.ndarray) else None
        return m_mean + normal(0.0, sigma_m_intrinsic, size=size)

def abs_mag_to_snr(abs_mag, z, band='r'):
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    d_L = cosmo.luminosity_distance(z).to('pc').value
    mu = 5 * np.log10(d_L / 10)
    app_mag = abs_mag + mu
    depth = band_depths[band]
    C = depth + 2.5 * np.log10(5)
    log_snr = (C - app_mag) / 2.5
    return 10 ** log_snr

def snr_to_abs_mag(snr, z, band='r'):
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    d_L = cosmo.luminosity_distance(z).to('pc').value
    mu = 5 * np.log10(d_L / 10)
    depth = band_depths[band]
    C = depth + 2.5 * np.log10(5)
    log_snr = np.log10(snr)
    app_mag = C - 2.5 * log_snr
    abs_mag = app_mag - mu
    return abs_mag

def app_mag_to_snr(app_mag, band='r'):
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    depth = band_depths[band]
    if isinstance(app_mag, torch.Tensor):
        C = depth + 2.5 * torch.log10(torch.tensor(5.0, device=app_mag.device))
        log_snr = (C - app_mag) / 2.5
        return 10 ** log_snr
    else:
        C = depth + 2.5 * np.log10(5)
        log_snr = (C - app_mag) / 2.5
        return 10 ** log_snr

def snr_to_app_mag(snr, band='r'):
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    depth = band_depths[band]
    if isinstance(snr, torch.Tensor):
        C = depth + 2.5 * torch.log10(torch.tensor(5.0, device=snr.device))
        log_snr = torch.log10(snr)
        app_mag = C - 2.5 * log_snr
        return app_mag
    else:
        C = depth + 2.5 * np.log10(5)
        log_snr = np.log10(snr)
        app_mag = C - 2.5 * log_snr
        return app_mag

def tf_vcirc_to_mag(vcirc, a, b):
    log_vcirc = np.log10(vcirc)
    return a * log_vcirc + b

def tf_mag_to_vcirc(mag, a, b):
    return 10 ** ((mag - b) / a)


def _rmag_snr_model(log_snr, a, b):
    return a * log_snr + b


def _fit_rmag_snr_relation(source_path=None, fit_path=None):
    if source_path is None:
        source_path = config.rmag_snr_source_path
    if fit_path is None:
        fit_path = config.rmag_snr_fit_path

    with np.load(source_path) as data:
        snr = np.asarray(data['SNR'], dtype=float)
        rmag = np.asarray(data['rmag'], dtype=float)

    valid = np.isfinite(snr) & np.isfinite(rmag) & (snr > 0)
    snr = snr[valid]
    rmag = rmag[valid]
    coeffs, _ = curve_fit(_rmag_snr_model, np.log10(snr), rmag)
    a, b = map(float, coeffs)

    os.makedirs(os.path.dirname(fit_path), exist_ok=True)
    np.savez(
        fit_path,
        a=a,
        b=b,
        source_path=source_path,
        model='rmag = a * log10(SNR) + b',
    )
    return a, b


def _load_rmag_snr_relation(fit_path=None):
    if fit_path is None:
        fit_path = config.rmag_snr_fit_path

    if os.path.exists(fit_path):
        with np.load(fit_path, allow_pickle=False) as data:
            return float(data['a']), float(data['b'])
    return _fit_rmag_snr_relation(fit_path=fit_path)


def _resolve_handedness_flip_feature_indices(feature_names):
    names = list(feature_names)
    g2_idx = resolve_feature_index(names, 'g2')
    try:
        theta_idx = resolve_feature_index(names, 'theta_int')
    except ValueError:
        theta_idx = None
    return g2_idx, theta_idx

D4_ELEMENTS = ("e", "r90", "r180", "r270", "v", "t", "h", "hvt")
"""Canonical D4 element names, retained for compatibility with d4_diffs.py."""

_D4_ACTIONS = {
    # First apply k array rotations, then optionally reflect the row/y axis.
    "e": (0, False),
    "r90": (1, False),
    "r180": (2, False),
    "r270": (3, False),
    "v": (0, True),
    "t": (1, True),
    "h": (2, True),
    "hvt": (3, True),
}
_D4_ALIASES = {
    "identity": "e",
    "flip_y": "v",
    "transpose": "t",
    "flip_x": "h",
    "flip_anti_diagonal": "hvt",
}
D4_INVERSES = {
    "e": "e",
    "r90": "r270",
    "r180": "r180",
    "r270": "r90",
    "v": "v",
    "t": "t",
    "h": "h",
    "hvt": "hvt",
}
D4_REFLECTION_FIBER_PERMUTATION = (0, 1, 2, 4, 3)


def _canonical_d4_element(element):
    key = str(element).strip().lower()
    key = _D4_ALIASES.get(key, key)
    if key not in _D4_ACTIONS:
        valid = ", ".join(D4_ELEMENTS)
        raise ValueError(f"Unknown D4 element '{element}'. Expected one of: {valid}")
    return key


def _wrap_normalized_angle(theta):
    """Wrap theta_int normalized by pi to [-1, 1)."""
    return torch.remainder(theta + 1.0, 2.0) - 1.0


def _rotate_array_coordinates(fp, k):
    """Rotate (..., 2) coordinates consistently with torch.rot90."""
    k = int(k) % 4
    x = fp[..., 0]
    y = fp[..., 1]
    if k == 0:
        return fp.clone()
    if k == 1:
        return torch.stack((y, -x), dim=-1)
    if k == 2:
        return torch.stack((-x, -y), dim=-1)
    return torch.stack((-y, x), dim=-1)


def _swap_reflected_minor_fibers(values, fiber_dim):
    if values.shape[fiber_dim] < 5:
        raise ValueError("D4 reflections require all five canonical fiber entries")
    permutation = list(range(values.shape[fiber_dim]))
    permutation[3], permutation[4] = permutation[4], permutation[3]
    index = torch.tensor(permutation, device=values.device)
    return torch.index_select(values, fiber_dim, index)


def _reflect_row_axis_datavector(
    img,
    spec=None,
    fid=None,
    fp=None,
    g2_idx=None,
    theta_idx=None,
):
    """Apply the authoritative row/y-axis reflection to a datavector."""
    img_out = torch.flip(img, dims=(-2,))
    spec_out = spec.clone() if spec is not None else None
    fid_out = fid.clone() if fid is not None else None
    fp_out = fp.clone() if fp is not None else None

    if fid_out is not None:
        if g2_idx is None:
            raise ValueError("g2_idx is required when reflecting parameters")
        fid_out[..., g2_idx] = -fid_out[..., g2_idx]
        if theta_idx is not None:
            fid_out[..., theta_idx] = _wrap_normalized_angle(
                -fid_out[..., theta_idx]
            )

    if fp_out is not None:
        fp_out = torch.stack((fp_out[..., 0], -fp_out[..., 1]), dim=-1)

    if spec_out is not None:
        spec_out = _swap_reflected_minor_fibers(spec_out, fiber_dim=-2)
    if fp_out is not None:
        fp_out = _swap_reflected_minor_fibers(fp_out, fiber_dim=-2)

    return img_out, spec_out, fid_out, fp_out


def apply_d4_to_datavector(
    img,
    spec=None,
    fid=None,
    fp=None,
    element="e",
    feature_names=None,
):
    """Apply one D4 element to a complete normalized KL datavector.

    Images use their final two axes as (row, column). Fiber positions use the
    matching array-coordinate convention, so r90 maps (x, y) to (y, -x).
    Spectra remain attached to the same directed fibers under rotations.
    Reflections reverse handedness and therefore swap minor+ with minor- in
    both the spectrum and fiber-position arrays.

    Parameters are normalized to [-1, 1], with theta_int normalized by pi.
    Parameters other than g1, g2, and theta_int are D4 scalars.
    """
    element = _canonical_d4_element(element)
    k, reflected = _D4_ACTIONS[element]

    img_out = torch.rot90(img, k=k, dims=(-2, -1)) if k else img.clone()
    spec_out = spec.clone() if spec is not None else None
    fp_out = _rotate_array_coordinates(fp, k) if fp is not None else None

    g2_idx = None
    theta_idx = None
    fid_out = fid.clone() if fid is not None else None
    if fid_out is not None:
        names = list(
            feature_names
            if feature_names is not None
            else config.train.get(
                "feature_names",
                ["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            )
        )
        g1_idx = resolve_feature_index(names, "g1")
        g2_idx = resolve_feature_index(names, "g2")
        theta_idx = resolve_feature_index(names, "theta_int")

        if k % 2:
            fid_out[..., [g1_idx, g2_idx]] = -fid_out[..., [g1_idx, g2_idx]]
        fid_out[..., theta_idx] = _wrap_normalized_angle(
            fid_out[..., theta_idx] - 0.5 * k
        )

    if reflected:
        img_out, spec_out, fid_out, fp_out = _reflect_row_axis_datavector(
            img_out,
            spec_out,
            fid_out,
            fp_out,
            g2_idx=g2_idx,
            theta_idx=theta_idx,
        )

    return img_out, spec_out, fid_out, fp_out

def rotate_90_degrees(img, fid=None, fp=None):
    """Backward-compatible wrapper for the canonical r90 action."""
    img_out, _, fid_out, fp_out = apply_d4_to_datavector(
        img=img,
        fid=fid,
        fp=fp,
        element="r90",
    )
    return img_out, fid_out, fp_out

def rot_90_param_only(fid, reverse=False):
    """Rotate physical parameter vectors by 90 degrees.

    Supports a single ``(P,)`` vector or arrays shaped ``(..., P)``. The first
    two parameters are the spin-2 shear components and the third is
    ``theta_int`` in radians.
    """
    angle = np.pi/2 if reverse else -np.pi/2
    fid_out = np.asarray(fid).copy()
    if fid_out.shape[-1] < 3:
        raise ValueError("parameter vectors must contain g1, g2, and theta_int")
    fid_out[..., :2] = -fid_out[..., :2]
    fid_out[..., 2] = (fid_out[..., 2] + angle + np.pi) % (2*np.pi) - np.pi
    return fid_out

def apply_views(img, spec, img2, spec2):
    # apply random masking to images
    random_masking = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    img = random_masking(img)
    img2 = random_masking(img2)

    # apply random masking to spectra
    spec = random_mask_spec(spec)
    spec2 = random_mask_spec(spec2)

    return img, spec, img2, spec2

def random_mask_spec(spec):
    bs = spec.shape[0]
    mask = torch.ones_like(spec, device=spec.device)
    r_maj = torch.rand(bs, device=spec.device)
    r_min = torch.rand(bs, device=spec.device)

    # --- Major Pair Masking (Indices 0 & 1) ---
    # 25% chance of masking major_1 (index 0)
    mask[(r_maj >= 0.5) & (r_maj < 0.75), :, 0] = 0.0
    # 25% chance of masking major_2 (index 1)
    mask[r_maj >= 0.75, :, 1] = 0.0
    
    # --- Minor Pair Masking (Indices 3 & 4) ---
    # 25% chance of masking minor_1 (index 3)
    mask[(r_min >= 0.5) & (r_min < 0.75), :, 3] = 0.0
    # 25% chance of masking minor_2 (index 4)
    mask[r_min >= 0.75, :, 4] = 0.0

    return spec * mask

def make_exact_half_flip_mask(size, device, generator=None):
    mask = torch.zeros((size,), dtype=torch.bool, device=device)
    nflip = size // 2
    if nflip == 0:
        return mask
    flip_ids = torch.randperm(size, device=device, generator=generator)[:nflip]
    mask[flip_ids] = True
    return mask


def apply_handedness_flip(
    img,
    spec,
    fid,
    fp=None,
    flip_mask=None,
    g2_idx=None,
    theta_idx=None,
):
    """Reflect selected rows using the same action as D4 element v."""
    if flip_mask is None:
        return img, spec, fid, fp
    if flip_mask.dtype != torch.bool:
        raise TypeError("flip_mask must be a torch.bool tensor")
    if flip_mask.ndim != 1:
        raise ValueError("flip_mask must be 1D with one entry per batch row")
    if flip_mask.shape[0] != img.shape[0]:
        raise ValueError("flip_mask length must match batch size")
    if g2_idx is None:
        raise ValueError("g2_idx is required for handedness flipping")
    if not torch.any(flip_mask):
        return img, spec, fid, fp

    reflected = _reflect_row_axis_datavector(
        img[flip_mask],
        spec[flip_mask] if spec is not None else None,
        fid[flip_mask],
        fp[flip_mask] if fp is not None else None,
        g2_idx=g2_idx,
        theta_idx=theta_idx,
    )

    img_out = img.clone()
    spec_out = spec.clone() if spec is not None else None
    fid_out = fid.clone()
    fp_out = fp.clone() if fp is not None else None

    img_out[flip_mask] = reflected[0]
    if spec_out is not None:
        spec_out[flip_mask] = reflected[1]
    fid_out[flip_mask] = reflected[2]
    if fp_out is not None:
        fp_out[flip_mask] = reflected[3]

    return img_out, spec_out, fid_out, fp_out

def _noise_scale_from_seg(data, snr, seg, eps=1e-8):
    snr = torch.clamp(snr, min=eps)
    npix = torch.sum(seg, dim=(-1, -2, -3)).float()
    npix = torch.clamp(npix, min=1.0)
    signal = torch.sum(data * seg.float(), dim=(-1, -2, -3))
    return signal / (torch.sqrt(npix) * snr)


def _estimate_noise_rms(noise, seg, eps=1e-8):
    bkg = (~seg).float()
    bkg_count = torch.sum(bkg, dim=(-1, -2, -3))
    bkg_count_safe = torch.clamp(bkg_count, min=1.0)
    bkg_rms = torch.sqrt(torch.sum((noise * bkg) ** 2, dim=(-1, -2, -3)) / bkg_count_safe)
    full_rms = torch.sqrt(torch.mean(noise ** 2, dim=(-1, -2, -3)))
    return torch.where(bkg_count > eps, bkg_rms, full_rms)


def apply_noise(
    data,
    snr,
    randgen=None,
    device='cpu',
    use_iterative=True,
    base_signal_frac=0.1,
    threshold_sigma=1.5,
    eps=1e-8,
    maxs=None,
):
    if randgen is None:
        noise = torch.randn(data.size(), device=device)
    else:
        noise = torch.randn(data.size(), device=device, generator=randgen)

    if maxs is None:
        maxs = torch.amax(data, dim=(-1, -2, -3))
    else:
        if maxs.ndim != 1 or maxs.shape[0] != data.shape[0]:
            raise ValueError("maxs must be a 1D tensor matching batch size")
    seg_coarse = data > (base_signal_frac * maxs).view(-1, 1, 1, 1)
    factor_coarse = _noise_scale_from_seg(data, snr, seg_coarse, eps=eps)

    if not use_iterative:
        return data + noise * factor_coarse.view(-1, 1, 1, 1)

    coarse_noise = noise * factor_coarse.view(-1, 1, 1, 1)
    coarse_rms = _estimate_noise_rms(coarse_noise, seg_coarse, eps=eps)
    refined_threshold = threshold_sigma * coarse_rms
    seg_refined = data > refined_threshold.view(-1, 1, 1, 1)
    factor_refined = _noise_scale_from_seg(data, snr, seg_refined, eps=eps)
    return data + noise * factor_refined.view(-1, 1, 1, 1)


def sample_magnitudes(n_samples, m_min, m_max, rng=None):
    uniform = np.random.uniform if rng is None else rng.uniform
    u = uniform(0, 1, n_samples)
    val_min = 10 ** (0.6 * m_min)
    val_max = 10 ** (0.6 * m_max)
    m = (5 / 3) * np.log10(u * (val_max - val_min) + val_min)
    return m
