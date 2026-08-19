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

def app_mag_to_snr(app_mag, band='r', depth_5sigma_mag=None):
    """Convert apparent magnitude to the flux SNR at a stated 5-sigma depth."""
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    depth = band_depths[band] if depth_5sigma_mag is None else float(depth_5sigma_mag)
    if isinstance(app_mag, torch.Tensor):
        C = depth + 2.5 * torch.log10(
            torch.tensor(5.0, device=app_mag.device, dtype=app_mag.dtype)
        )
        log_snr = (C - app_mag) / 2.5
        return 10 ** log_snr
    else:
        C = depth + 2.5 * np.log10(5)
        log_snr = (C - app_mag) / 2.5
        return 10 ** log_snr

def snr_to_app_mag(snr, band='r', depth_5sigma_mag=None):
    band_depths = {
        'g': 24.0,
        'r': 23.4,
        'z': 22.5,
        }
    assert band in band_depths, f"Band '{band}' not recognized. Valid bands: {list(band_depths.keys())}"
    depth = band_depths[band] if depth_5sigma_mag is None else float(depth_5sigma_mag)
    if isinstance(snr, torch.Tensor):
        C = depth + 2.5 * torch.log10(
            torch.tensor(5.0, device=snr.device, dtype=snr.dtype)
        )
        log_snr = torch.log10(snr)
        app_mag = C - 2.5 * log_snr
        return app_mag
    else:
        C = depth + 2.5 * np.log10(5)
        log_snr = np.log10(snr)
        app_mag = C - 2.5 * log_snr
        return app_mag


def magnitude_uncertainty_from_snr(snr):
    """First-order AB-magnitude uncertainty for a positive flux SNR."""
    coefficient = 2.5 / np.log(10.0)
    if isinstance(snr, torch.Tensor):
        if bool((~torch.isfinite(snr) | (snr <= 0)).any()):
            raise ValueError("snr must contain finite positive values")
        return coefficient / snr
    values = np.asarray(snr)
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError("snr must contain finite positive values")
    return coefficient / values


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


def transform_d4_parameters(parameters, element="e", feature_names=None):
    """Apply the canonical D4 action to normalized KL parameter vectors.

    The last axis may use any feature order. ``g1``/``g2`` are a spin-2 pair,
    ``theta_int`` is a directed angle normalized by pi, and every other KL
    parameter is a D4 scalar. The angle is always canonicalized to ``[-1, 1)``.
    The action is measure preserving on the canonical coordinate domain.
    """
    if not torch.is_tensor(parameters):
        raise TypeError("parameters must be a torch tensor")
    if parameters.ndim < 1:
        raise ValueError("parameters must have a final feature dimension")
    element = _canonical_d4_element(element)
    if feature_names is None:
        names = list(
            config.train.get(
                "feature_names",
                ["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            )
        )
        if parameters.shape[-1] <= len(names):
            names = names[: parameters.shape[-1]]
    else:
        names = list(feature_names)
    if parameters.shape[-1] != len(names):
        raise ValueError(
            "parameter feature dimension must match feature_names; "
            f"got {parameters.shape[-1]} and {len(names)}"
        )
    g1_idx = resolve_feature_index(names, "g1")
    g2_idx = resolve_feature_index(names, "g2")
    theta_idx = resolve_feature_index(names, "theta_int")
    k, reflected = _D4_ACTIONS[element]

    transformed = parameters.clone()
    if k % 2:
        transformed[..., g1_idx] = -transformed[..., g1_idx]
        transformed[..., g2_idx] = -transformed[..., g2_idx]
    transformed[..., theta_idx] = _wrap_normalized_angle(
        transformed[..., theta_idx] - 0.5 * k
    )
    if reflected:
        transformed[..., g2_idx] = -transformed[..., g2_idx]
        transformed[..., theta_idx] = _wrap_normalized_angle(
            -transformed[..., theta_idx]
        )
    return transformed


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

    fid_out = (
        transform_d4_parameters(fid, element, feature_names=feature_names)
        if fid is not None
        else None
    )

    if reflected:
        img_out, spec_out, _, fp_out = _reflect_row_axis_datavector(
            img_out,
            spec_out,
            None,
            fp_out,
        )

    return img_out, spec_out, fid_out, fp_out


def transform_d4_fiber_mask(fiber_mask, element="e"):
    """Transform a Boolean fiber-observation mask with the canonical D4 action."""
    if fiber_mask.dtype != torch.bool:
        raise TypeError("fiber_mask must be a bool tensor")
    if fiber_mask.ndim < 1:
        raise ValueError("fiber_mask must have a fiber dimension")
    element = _canonical_d4_element(element)
    _, reflected = _D4_ACTIONS[element]
    if not reflected:
        return fiber_mask.clone()
    return _swap_reflected_minor_fibers(fiber_mask, fiber_dim=-1)


def transform_d4_feature_blocks(
    features,
    element="e",
    *,
    scalar_channels,
    spin1_channels,
    spin2_channels,
):
    """Apply D4 actions to scalar, directed spin-1, and spin-2 blocks.

    The final feature axis is laid out as ``[scalars, spin1 pairs, spin2
    pairs]``. Spin-1 follows the directed ``(cos(theta), sin(theta))`` and
    array-coordinate convention. Spin-2 follows ``(g1, g2)``.
    """
    channel_counts = (scalar_channels, spin1_channels, spin2_channels)
    if any(type(value) is not int or value < 0 for value in channel_counts):
        raise ValueError("D4 feature channel counts must be non-negative integers")
    if spin1_channels % 2 or spin2_channels % 2:
        raise ValueError("spin-1 and spin-2 channel counts must be even")
    if sum(channel_counts) != features.shape[-1]:
        raise ValueError(
            "D4 feature channel counts must sum to the final feature dimension"
        )

    element = _canonical_d4_element(element)
    k, reflected = _D4_ACTIONS[element]
    scalar_end = scalar_channels
    spin1_end = scalar_end + spin1_channels
    scalars = features[..., :scalar_end].clone()

    spin1 = features[..., scalar_end:spin1_end].reshape(
        *features.shape[:-1], spin1_channels // 2, 2
    )
    spin1 = _rotate_array_coordinates(spin1, k)
    if reflected:
        spin1 = torch.stack((spin1[..., 0], -spin1[..., 1]), dim=-1)

    spin2 = features[..., spin1_end:].reshape(
        *features.shape[:-1], spin2_channels // 2, 2
    )
    rotation_sign = -1.0 if k % 2 else 1.0
    spin2_x = rotation_sign * spin2[..., 0]
    spin2_y = rotation_sign * spin2[..., 1]
    if reflected:
        spin2_y = -spin2_y
    spin2 = torch.stack((spin2_x, spin2_y), dim=-1)

    return torch.cat(
        (
            scalars,
            spin1.reshape(*features.shape[:-1], spin1_channels),
            spin2.reshape(*features.shape[:-1], spin2_channels),
        ),
        dim=-1,
    )


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


def gaussian_psf_noise_equivalent_pixels(
    psf_fwhm_arcsec=1.0,
    pixel_scale_arcsec=0.2637,
):
    """Return the noise-equivalent area of a Gaussian reference PSF in pixels.

    For a unit-flux circular Gaussian template, ``N_eff = 1 / sum(P**2)``.
    In the well-sampled continuous limit this is ``4 pi sigma_pix**2``. This is
    only the fixed reference used to interpret a quoted five-sigma depth; it
    does not assert that the simulator's Airy-FWHM rendering is Gaussian.
    """
    fwhm = float(psf_fwhm_arcsec)
    pixel_scale = float(pixel_scale_arcsec)
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("psf_fwhm_arcsec must be finite and positive")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError("pixel_scale_arcsec must be finite and positive")
    sigma_pixels = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale)
    return 4.0 * np.pi * sigma_pixels**2


def depth_scaled_total_image_flux(data, rmag_true, depth_5sigma_mag):
    """Scale each clean image's total rendered flux to a common depth magnitude.

    The returned vector is intended only for a dataset-global calibration. It
    must not be used as a per-object noise scale or supplied to the posterior.
    """
    if not isinstance(data, torch.Tensor) or not data.is_floating_point():
        raise TypeError("data must be a floating-point torch.Tensor")
    if data.ndim < 2 or data.shape[0] == 0:
        raise ValueError("data must have a non-empty leading batch dimension")
    magnitude = torch.as_tensor(
        rmag_true, device=data.device, dtype=torch.float64
    )
    if magnitude.ndim != 1 or magnitude.shape[0] != data.shape[0]:
        raise ValueError("rmag_true must be one-dimensional and match batch size")
    depth = float(depth_5sigma_mag)
    if not np.isfinite(depth):
        raise ValueError("depth_5sigma_mag must be finite")
    if bool((~torch.isfinite(magnitude)).any()):
        raise ValueError("rmag_true must contain only finite values")

    # Accumulate in float64 without materializing a full float64 copy of a
    # million-image training tensor on the GPU.
    total_flux = data.sum(
        dim=tuple(range(1, data.ndim)), dtype=torch.float64
    )
    if bool((~torch.isfinite(total_flux) | (total_flux <= 0)).any()):
        raise ValueError("clean image total flux must be finite and positive")
    to_depth = torch.pow(
        total_flux.new_tensor(10.0),
        -0.4 * (total_flux.new_tensor(depth) - magnitude),
    )
    depth_flux = total_flux * to_depth
    if bool((~torch.isfinite(depth_flux) | (depth_flux <= 0)).any()):
        raise ValueError("depth-scaled image flux must be finite and positive")
    return depth_flux


def fixed_image_noise_sigma_from_depth_fluxes(
    depth_scaled_fluxes,
    noise_equivalent_pixels,
):
    """Calibrate one homoscedastic pixel RMS from depth-scaled image fluxes."""
    fluxes = torch.as_tensor(depth_scaled_fluxes)
    if not fluxes.is_floating_point():
        fluxes = fluxes.to(torch.get_default_dtype())
    if fluxes.ndim != 1 or fluxes.numel() == 0:
        raise ValueError("depth_scaled_fluxes must be a non-empty vector")
    if bool((~torch.isfinite(fluxes) | (fluxes <= 0)).any()):
        raise ValueError(
            "depth_scaled_fluxes must contain finite positive values"
        )
    n_eff = float(noise_equivalent_pixels)
    if not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError("noise_equivalent_pixels must be finite and positive")
    reference_flux = deterministic_lower_median(fluxes)
    return reference_flux / (5.0 * np.sqrt(n_eff))


def estimate_fixed_image_noise_sigma(
    data,
    rmag_true,
    *,
    depth_5sigma_mag=23.4,
    psf_fwhm_arcsec=1.0,
    pixel_scale_arcsec=0.2637,
):
    """Estimate the single v2 image-noise RMS for a training population."""
    depth_fluxes = depth_scaled_total_image_flux(
        data, rmag_true, depth_5sigma_mag
    )
    n_eff = gaussian_psf_noise_equivalent_pixels(
        psf_fwhm_arcsec, pixel_scale_arcsec
    )
    return fixed_image_noise_sigma_from_depth_fluxes(depth_fluxes, n_eff)


def apply_fixed_gaussian_image_noise(data, noise_sigma, randgen=None):
    """Add v2 homoscedastic Gaussian image noise with one scalar pixel RMS."""
    if not isinstance(data, torch.Tensor) or not data.is_floating_point():
        raise TypeError("data must be a floating-point torch.Tensor")
    sigma = torch.as_tensor(
        noise_sigma, device=data.device, dtype=data.dtype
    )
    if sigma.numel() != 1:
        raise ValueError("noise_sigma must be a single global scalar")
    sigma = sigma.reshape(())
    if not bool(torch.isfinite(sigma) & (sigma > 0)):
        raise ValueError("noise_sigma must be finite and positive")
    noise = torch.randn(
        data.shape,
        device=data.device,
        dtype=data.dtype,
        generator=randgen,
    )
    return data + noise * sigma


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
    return_scale=False,
):
    """Add the historical homoscedastic Gaussian noise realization."""
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
        output = data + noise * factor_coarse.view(-1, 1, 1, 1)
        return (output, factor_coarse) if return_scale else output

    coarse_noise = noise * factor_coarse.view(-1, 1, 1, 1)
    coarse_rms = _estimate_noise_rms(coarse_noise, seg_coarse, eps=eps)
    refined_threshold = threshold_sigma * coarse_rms
    seg_refined = data > refined_threshold.view(-1, 1, 1, 1)
    factor_refined = _noise_scale_from_seg(data, snr, seg_refined, eps=eps)
    refined_count = seg_refined.sum(dim=(-1, -2, -3))
    valid_refined = torch.isfinite(factor_refined) & (factor_refined > eps)
    valid_refined &= refined_count > 0
    factor_refined = torch.where(
        valid_refined, factor_refined, factor_coarse
    )
    output = data + noise * factor_refined.view(-1, 1, 1, 1)
    return (output, factor_refined) if return_scale else output


def sample_observed_magnitude(rmag_true, image_snr, randgen=None):
    """Draw a catalog measurement from a fixed-depth Gaussian flux model.

    ``image_snr`` is the *expected* flux SNR implied by ``rmag_true`` and the
    configured survey depth.  In units of the fixed flux uncertainty, the
    measurement is therefore ``rho_obs = image_snr + Normal(0, 1)``.  The
    returned magnitude, reported magnitude uncertainty, and ``image_flux_snr``
    are all derived from that noisy observed flux.  Consequently none of the
    returned context scalars algebraically reveals ``rmag_true``.

    The simulator-v2 magnitude range has expected SNR >= 5.  We nevertheless
    redraw the vanishingly rare non-positive Gaussian flux, because an
    ordinary logarithmic magnitude is undefined there.  This is an explicit
    positive-flux catalog selection, not a clamp.
    """
    rmag_true = torch.as_tensor(rmag_true)
    if not rmag_true.is_floating_point():
        rmag_true = rmag_true.to(torch.get_default_dtype())
    image_snr = torch.as_tensor(
        image_snr, device=rmag_true.device, dtype=rmag_true.dtype
    )
    rmag_true, image_snr = torch.broadcast_tensors(rmag_true, image_snr)
    if bool((~torch.isfinite(rmag_true)).any()):
        raise ValueError("rmag_true must contain finite values")
    if bool((~torch.isfinite(image_snr) | (image_snr <= 0)).any()):
        raise ValueError("image_snr must contain finite positive values")
    observed_snr = image_snr + torch.randn(
        rmag_true.shape,
        device=rmag_true.device,
        dtype=rmag_true.dtype,
        generator=randgen,
    )
    for _ in range(16):
        invalid = observed_snr <= 0
        if not bool(invalid.any()):
            break
        observed_snr = observed_snr.clone()
        observed_snr[invalid] = image_snr[invalid] + torch.randn(
            (int(invalid.sum().item()),),
            device=rmag_true.device,
            dtype=rmag_true.dtype,
            generator=randgen,
        )
    if bool((observed_snr <= 0).any()):
        raise RuntimeError(
            "failed to draw a positive observed flux after 16 attempts"
        )

    rmag_obs = rmag_true - 2.5 * torch.log10(observed_snr / image_snr)
    rmag_sigma = magnitude_uncertainty_from_snr(observed_snr)
    return {
        "rmag_obs": rmag_obs,
        "rmag_sigma": rmag_sigma,
        "image_flux_snr": observed_snr,
    }


def deterministic_lower_median(values, dim=None, keepdim=False):
    """Compute ``torch.median``'s lower median without its CUDA kernel.

    PyTorch's CUDA ``median(dim=...)`` implementation always computes indices
    and is rejected by ``torch.use_deterministic_algorithms(True)``.  Sorting
    and selecting the lower middle element is deterministic and preserves
    ``torch.median``'s behavior for even-length inputs.
    """
    values = torch.as_tensor(values)
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    if dim is None:
        flattened = values.reshape(-1)
        return torch.sort(flattened).values[(flattened.numel() - 1) // 2]

    dim = int(dim)
    if dim < 0:
        dim += values.ndim
    if dim < 0 or dim >= values.ndim:
        raise IndexError(
            f"dimension out of range for tensor with {values.ndim} dimensions"
        )
    length = values.shape[dim]
    if length == 0:
        raise ValueError("median dimension must be non-empty")
    result = torch.sort(values, dim=dim).values.select(
        dim, (length - 1) // 2
    )
    return result.unsqueeze(dim) if keepdim else result


def _continuum_subtracted_line_norm(spectra):
    continuum = deterministic_lower_median(
        spectra, dim=-1, keepdim=True
    )
    return torch.linalg.vector_norm(spectra - continuum, dim=-1)


def estimate_spectral_reference_line_norm(spectra, center_fiber_index=2):
    """Estimate a robust fixed H-alpha norm from the offset-fiber population."""
    if spectra.ndim != 4 or spectra.shape[-2] < 2:
        raise ValueError("spectra must have shape (B,C,F,W)")
    if not 0 <= center_fiber_index < spectra.shape[-2]:
        raise ValueError("center_fiber_index is out of range")
    line_norm = _continuum_subtracted_line_norm(spectra)
    offset_mask = torch.ones(
        spectra.shape[-2], dtype=torch.bool, device=spectra.device
    )
    offset_mask[center_fiber_index] = False
    values = line_norm[..., offset_mask].reshape(-1)
    values = values[torch.isfinite(values) & (values > 0)]
    if values.numel() == 0:
        raise ValueError("cannot estimate a positive spectral reference norm")
    return deterministic_lower_median(values)


def apply_spectral_noise(
    data,
    reference_quality,
    reference_line_norm,
    *,
    center_fiber_index=2,
    center_exposure_s=180.0,
    offset_exposure_s=600.0,
    spectral_units="counts",
    randgen=None,
    device=None,
    return_metadata=False,
):
    """Add independent Gaussian spectral noise using a fixed line reference."""
    if data.ndim != 4 or data.shape[-2] < 2:
        raise ValueError("data must have shape (B,C,F,W)")
    if not 0 <= center_fiber_index < data.shape[-2]:
        raise ValueError("center_fiber_index is out of range")
    if center_exposure_s <= 0 or offset_exposure_s <= 0:
        raise ValueError("exposure times must be positive")
    if spectral_units not in ("counts", "count_rate"):
        raise ValueError("spectral_units must be 'counts' or 'count_rate'")
    if device is None:
        device = data.device
    quality = torch.as_tensor(
        reference_quality, device=data.device, dtype=data.dtype
    ).reshape(-1)
    if quality.numel() == 1 and data.shape[0] != 1:
        quality = quality.expand(data.shape[0])
    if quality.numel() != data.shape[0]:
        raise ValueError("reference_quality must be scalar or one value per spectrum")
    if bool((~torch.isfinite(quality) | (quality <= 0)).any()):
        raise ValueError("reference_quality must contain finite positive values")
    reference_line_norm = torch.as_tensor(
        reference_line_norm, device=data.device, dtype=data.dtype
    ).reshape(())
    if not bool(torch.isfinite(reference_line_norm)) or reference_line_norm <= 0:
        raise ValueError("reference_line_norm must be finite and positive")

    sigma_offset = reference_line_norm / quality
    fiber_sigma = sigma_offset[:, None].expand(-1, data.shape[-2]).clone()
    if spectral_units == "counts":
        center_ratio = np.sqrt(center_exposure_s / offset_exposure_s)
    else:
        center_ratio = np.sqrt(offset_exposure_s / center_exposure_s)
    fiber_sigma[:, center_fiber_index] *= center_ratio
    sigma = fiber_sigma[:, None, :, None]
    noise = torch.randn(
        data.shape, device=device, dtype=data.dtype, generator=randgen
    )
    output = data + noise * sigma
    if not return_metadata:
        return output
    achieved_snr = _continuum_subtracted_line_norm(data) / fiber_sigma[:, None, :]
    return output, {
        "noise_sigma": fiber_sigma,
        "achieved_line_snr": achieved_snr,
        "reference_quality": quality,
    }


def sample_magnitudes(n_samples, m_min, m_max, rng=None):
    uniform = np.random.uniform if rng is None else rng.uniform
    u = uniform(0, 1, n_samples)
    val_min = 10 ** (0.6 * m_min)
    val_max = 10 ** (0.6 * m_max)
    m = (5 / 3) * np.log10(u * (val_max - val_min) + val_min)
    return m
