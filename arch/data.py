import os
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
import torch
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
        
    def sample_mag_from_vcirc(self, vcirc):
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
            return m_mean + torch.randn_like(m_mean) * sigma_m_intrinsic
        elif isinstance(vcirc, np.ndarray):
            return m_mean + np.random.normal(0.0, sigma_m_intrinsic, size=m_mean.shape)
        else:
            return m_mean + np.random.normal(0.0, sigma_m_intrinsic)

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


def make_exact_half_flip_mask(size, device):
    mask = torch.zeros((size,), dtype=torch.bool, device=device)
    nflip = size // 2
    if nflip == 0:
        return mask
    flip_ids = torch.randperm(size, device=device)[:nflip]
    mask[flip_ids] = True
    return mask


def apply_handedness_flip(img, spec, fid, fp=None, flip_mask=None, g2_idx=None, theta_idx=None):
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

    img_out = img.clone()
    spec_out = spec.clone() if spec is not None else None
    fid_out = fid.clone()
    fp_out = fp.clone() if fp is not None else None

    img_out[flip_mask] = torch.flip(img_out[flip_mask], dims=(-2,))
    fid_out[flip_mask, g2_idx] = -fid_out[flip_mask, g2_idx]
    if theta_idx is not None:
        fid_out[flip_mask, theta_idx] = -fid_out[flip_mask, theta_idx]

    if spec_out is not None:
        if spec_out.shape[-2] < 5:
            raise ValueError("spec must include 5 fiber spectra to swap minor-axis fibers")
        spec_out[flip_mask] = spec_out[flip_mask][:, :, [0, 1, 2, 4, 3], :]

    if fp_out is not None:
        if fp_out.shape[1] < 5:
            raise ValueError("fp must include 5 fiber positions to swap minor-axis fibers")
        fp_out[flip_mask] = fp_out[flip_mask][:, [0, 1, 2, 4, 3], :]
        fp_out[flip_mask, :, 1] = -fp_out[flip_mask, :, 1]

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


def sample_magnitudes(n_samples, m_min, m_max):
    u = np.random.uniform(0, 1, n_samples)
    val_min = 10 ** (0.6 * m_min)
    val_max = 10 ** (0.6 * m_max)
    m = (5 / 3) * np.log10(u * (val_max - val_min) + val_min)
    return m
