from __future__ import annotations

import numpy as np
import torch

try:
    from . import config
    from .utils import resolve_feature_index
except ImportError:  # Direct execution with arch/ on sys.path.
    import config
    from utils import resolve_feature_index


def app_mag_to_snr(app_mag, *, depth_5sigma_mag=23.4):
    """Ideal flux SNR implied by a stated five-sigma depth."""
    depth = float(depth_5sigma_mag)
    values = torch.as_tensor(app_mag) if isinstance(app_mag, torch.Tensor) else np.asarray(app_mag)
    if isinstance(values, torch.Tensor):
        if bool((~torch.isfinite(values)).any()):
            raise ValueError("app_mag must be finite")
        return 5.0 * torch.pow(10.0, 0.4 * (depth - values))
    if np.any(~np.isfinite(values)):
        raise ValueError("app_mag must be finite")
    return 5.0 * np.power(10.0, 0.4 * (depth - values))


def _wrap_normalized_angle(theta):
    return torch.remainder(theta + 1.0, 2.0) - 1.0


def rotate_90_parameters(parameters, *, inverse=False, feature_names=None):
    """Rotate normalized KL targets with the image-array R90 convention."""
    if not torch.is_tensor(parameters):
        raise TypeError("parameters must be a torch.Tensor")
    names = tuple(feature_names or config.train["feature_names"])
    if parameters.shape[-1] != len(names):
        raise ValueError("parameter feature dimension does not match feature_names")
    g1 = resolve_feature_index(names, "g1")
    g2 = resolve_feature_index(names, "g2")
    theta = resolve_feature_index(names, "theta_int")
    result = parameters.clone()
    result[..., g1] = -result[..., g1]
    result[..., g2] = -result[..., g2]
    result[..., theta] = _wrap_normalized_angle(
        result[..., theta] + (0.5 if inverse else -0.5)
    )
    return result


def rotate_90_datavector(
    image,
    spectra=None,
    parameters=None,
    fiber_positions=None,
    *,
    inverse=False,
    feature_names=None,
):
    """Rotate the complete current datavector by one quarter turn."""
    if not torch.is_tensor(image):
        raise TypeError("image must be a torch.Tensor")
    image_out = torch.rot90(image, k=3 if inverse else 1, dims=(-2, -1))
    spectra_out = spectra.clone() if spectra is not None else None
    parameters_out = (
        rotate_90_parameters(parameters, inverse=inverse, feature_names=feature_names)
        if parameters is not None else None
    )
    if fiber_positions is None:
        positions_out = None
    else:
        x, y = fiber_positions[..., 0], fiber_positions[..., 1]
        positions_out = (
            torch.stack((-y, x), dim=-1)
            if inverse else torch.stack((y, -x), dim=-1)
        )
    return image_out, spectra_out, parameters_out, positions_out


def deterministic_lower_median(values, dim=None, keepdim=False):
    """Deterministic lower median implemented with sorting."""
    values = torch.as_tensor(values)
    if values.numel() == 0:
        raise ValueError("values must be non-empty")
    if dim is None:
        flat = values.reshape(-1)
        return torch.sort(flat).values[(flat.numel() - 1) // 2]
    dim = int(dim) % values.ndim
    length = values.shape[dim]
    if length == 0:
        raise ValueError("median dimension must be non-empty")
    result = torch.sort(values, dim=dim).values.select(dim, (length - 1) // 2)
    return result.unsqueeze(dim) if keepdim else result


def gaussian_psf_noise_equivalent_pixels(
    psf_fwhm_arcsec=1.0, pixel_scale_arcsec=0.2637
):
    fwhm, pixel = float(psf_fwhm_arcsec), float(pixel_scale_arcsec)
    if not np.isfinite(fwhm) or fwhm <= 0 or not np.isfinite(pixel) or pixel <= 0:
        raise ValueError("PSF FWHM and pixel scale must be finite and positive")
    sigma_pixels = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel)
    return 4.0 * np.pi * sigma_pixels**2


def depth_scaled_total_image_flux(data, rmag_true, depth_5sigma_mag):
    if not isinstance(data, torch.Tensor) or not data.is_floating_point():
        raise TypeError("data must be a floating-point torch.Tensor")
    magnitude = torch.as_tensor(rmag_true, device=data.device, dtype=torch.float64)
    if magnitude.ndim != 1 or magnitude.shape[0] != data.shape[0]:
        raise ValueError("rmag_true must match the image batch")
    total = data.sum(dim=tuple(range(1, data.ndim)), dtype=torch.float64)
    if bool((~torch.isfinite(total) | (total <= 0)).any()):
        raise ValueError("clean image flux must be finite and positive")
    scaled = total * torch.pow(
        total.new_tensor(10.0),
        -0.4 * (total.new_tensor(float(depth_5sigma_mag)) - magnitude),
    )
    if bool((~torch.isfinite(scaled) | (scaled <= 0)).any()):
        raise ValueError("depth-scaled flux must be finite and positive")
    return scaled


def fixed_image_noise_sigma_from_depth_fluxes(depth_scaled_fluxes, n_eff):
    fluxes = torch.as_tensor(depth_scaled_fluxes)
    if fluxes.ndim != 1 or fluxes.numel() == 0:
        raise ValueError("depth_scaled_fluxes must be a non-empty vector")
    n_eff = float(n_eff)
    if bool((~torch.isfinite(fluxes) | (fluxes <= 0)).any()) or not np.isfinite(n_eff) or n_eff <= 0:
        raise ValueError("depth fluxes and noise area must be finite and positive")
    return deterministic_lower_median(fluxes) / (5.0 * np.sqrt(n_eff))


def estimate_fixed_image_noise_sigma(
    data, rmag_true, *, depth_5sigma_mag=23.4,
    psf_fwhm_arcsec=1.0, pixel_scale_arcsec=0.2637
):
    fluxes = depth_scaled_total_image_flux(data, rmag_true, depth_5sigma_mag)
    n_eff = gaussian_psf_noise_equivalent_pixels(
        psf_fwhm_arcsec, pixel_scale_arcsec
    )
    return fixed_image_noise_sigma_from_depth_fluxes(fluxes, n_eff)


def apply_fixed_gaussian_image_noise(data, noise_sigma, randgen=None):
    if not isinstance(data, torch.Tensor) or not data.is_floating_point():
        raise TypeError("data must be a floating-point torch.Tensor")
    sigma = torch.as_tensor(noise_sigma, device=data.device, dtype=data.dtype)
    if sigma.numel() != 1 or not bool(torch.isfinite(sigma) & (sigma > 0)):
        raise ValueError("noise_sigma must be one finite positive scalar")
    return data + torch.randn(
        data.shape, device=data.device, dtype=data.dtype, generator=randgen
    ) * sigma.reshape(())


def _continuum_subtracted_line_norm(spectra):
    continuum = deterministic_lower_median(spectra, dim=-1, keepdim=True)
    return torch.linalg.vector_norm(spectra - continuum, dim=-1)


def spectral_reference_line_norm_values(spectra, center_fiber_index=2):
    if spectra.ndim != 4 or not 0 <= center_fiber_index < spectra.shape[-2]:
        raise ValueError("spectra must have shape (B,C,F,W) and a valid center")
    norm = _continuum_subtracted_line_norm(spectra)
    mask = torch.ones(spectra.shape[-2], dtype=torch.bool, device=spectra.device)
    mask[center_fiber_index] = False
    values = norm[..., mask].reshape(-1)
    if values.numel() == 0 or bool((~torch.isfinite(values) | (values <= 0)).any()):
        raise ValueError(
            "all offset-fiber spectral reference norms must be finite and positive"
        )
    return values


def estimate_spectral_reference_line_norm(spectra, center_fiber_index=2):
    values = spectral_reference_line_norm_values(
        spectra, center_fiber_index=center_fiber_index
    )
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
    if data.ndim != 4 or not 0 <= center_fiber_index < data.shape[-2]:
        raise ValueError("data must have shape (B,C,F,W) and a valid center")
    if center_exposure_s <= 0 or offset_exposure_s <= 0:
        raise ValueError("exposure times must be positive")
    if spectral_units not in ("counts", "count_rate"):
        raise ValueError("spectral_units must be counts or count_rate")
    quality = torch.as_tensor(
        reference_quality, device=data.device, dtype=data.dtype
    ).reshape(-1)
    if quality.numel() == 1 and data.shape[0] != 1:
        quality = quality.expand(data.shape[0])
    if quality.numel() != data.shape[0] or bool((~torch.isfinite(quality) | (quality <= 0)).any()):
        raise ValueError("reference_quality must be finite, positive, and match batch")
    reference = torch.as_tensor(
        reference_line_norm, device=data.device, dtype=data.dtype
    ).reshape(())
    if not bool(torch.isfinite(reference) & (reference > 0)):
        raise ValueError("reference_line_norm must be finite and positive")
    offset_sigma = reference / quality
    fiber_sigma = offset_sigma[:, None].expand(-1, data.shape[-2]).clone()
    ratio = (
        np.sqrt(center_exposure_s / offset_exposure_s)
        if spectral_units == "counts"
        else np.sqrt(offset_exposure_s / center_exposure_s)
    )
    fiber_sigma[:, center_fiber_index] *= ratio
    output = data + torch.randn(
        data.shape,
        device=data.device if device is None else device,
        dtype=data.dtype,
        generator=randgen,
    ) * fiber_sigma[:, None, :, None]
    if not return_metadata:
        return output
    return output, {
        "noise_sigma": fiber_sigma,
        "achieved_line_snr": (
            _continuum_subtracted_line_norm(data) / fiber_sigma[:, None, :]
        ),
        "reference_quality": quality,
    }
