import math

import pytest
import torch

import config
from data import (
    app_mag_to_snr,
    apply_fixed_gaussian_image_noise,
    apply_spectral_noise,
    deterministic_lower_median,
    estimate_spectral_reference_line_norm,
    estimate_fixed_image_noise_sigma,
    gaussian_psf_noise_equivalent_pixels,
    spectral_reference_line_norm_values,
)
from train import build_observation_levels, validate_observation_record


def _current_record():
    return {
        "img": torch.ones(1, 8, 8),
        "spec": torch.ones(1, 5, 16),
        "fid_pars": torch.zeros(len(config.TARGET_NAMES)),
        "fib_pos": torch.zeros(5, 2),
        "rmag_true": torch.tensor(20.0),
        "halpha_flux_true": torch.tensor(1.0e-15),
        "observation_model_version": torch.tensor(2),
        "fiber_layout": torch.tensor(1),
        "image_band_code": torch.tensor(0),
        "target_line_code": torch.tensor(0),
        "spectral_units_code": torch.tensor(0),
        "center_fiber_index": torch.tensor(2),
        "center_exposure_s": torch.tensor(180.0),
        "offset_exposure_s": torch.tensor(600.0),
        "image_reference_psf_fwhm_arcsec": torch.tensor(1.0),
        "image_pixel_scale_arcsec": torch.tensor(0.2637),
    }


def test_magnitude_to_depth_snr_formula():
    magnitudes = torch.tensor([23.4, 22.4, 21.4])
    expected = 5.0 * 10.0 ** (0.4 * (23.4 - magnitudes))
    torch.testing.assert_close(app_mag_to_snr(magnitudes), expected)


def test_lower_median_is_deterministic_algorithm_safe():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
        assert deterministic_lower_median(values, dim=1).item() == 2.0
    finally:
        torch.use_deterministic_algorithms(previous)


def test_spectral_reference_uses_every_offset_fiber_value():
    spectra = torch.zeros(2, 1, 5, 5)
    amplitudes = torch.tensor(
        [[1.0, 2.0, 99.0, 3.0, 4.0], [5.0, 6.0, 99.0, 7.0, 8.0]]
    )
    spectra[:, 0, :, 2] = amplitudes
    values = spectral_reference_line_norm_values(
        spectra, center_fiber_index=2
    )
    assert values.shape == (8,)
    assert estimate_spectral_reference_line_norm(
        spectra, center_fiber_index=2
    ) == torch.sort(values).values[3]


def test_fixed_image_noise_uses_one_depth_scale_and_repeats_by_seed():
    images = torch.stack((torch.ones(8, 8), 10.0 * torch.ones(8, 8)))[:, None]
    magnitudes = torch.tensor([20.0, 17.5])
    sigma = estimate_fixed_image_noise_sigma(
        images,
        magnitudes,
        depth_5sigma_mag=23.4,
        psf_fwhm_arcsec=1.0,
        pixel_scale_arcsec=0.2637,
    )
    assert sigma.ndim == 0 and sigma > 0
    noisy_a = apply_fixed_gaussian_image_noise(
        images, sigma, randgen=torch.Generator().manual_seed(17)
    )
    noisy_b = apply_fixed_gaussian_image_noise(
        images, sigma, randgen=torch.Generator().manual_seed(17)
    )
    torch.testing.assert_close(noisy_a, noisy_b)


def test_spectral_noise_has_equal_offsets_and_count_exposure_scaling():
    spectra = torch.zeros(3, 1, 5, 16)
    quality = torch.tensor([3.0, 10.0, 100.0])
    generator = torch.Generator().manual_seed(12)
    noisy, metadata = apply_spectral_noise(
        spectra,
        quality,
        reference_line_norm=torch.tensor(8.0),
        center_fiber_index=2,
        center_exposure_s=180.0,
        offset_exposure_s=600.0,
        spectral_units="counts",
        randgen=generator,
        return_metadata=True,
    )
    assert noisy.shape == spectra.shape
    sigma = metadata["noise_sigma"]
    assert sigma.shape == (3, 5)
    offsets = sigma[:, [0, 1, 3, 4]]
    torch.testing.assert_close(offsets, offsets[:, :1].expand(-1, 4))
    torch.testing.assert_close(
        sigma[:, 2] / sigma[:, 0],
        torch.full((3,), math.sqrt(180.0 / 600.0)),
    )


def test_observation_levels_separate_magnitude_and_log_uniform_spectral_quality():
    magnitudes = torch.linspace(16.0, 23.0, 64)
    image_snr, quality = build_observation_levels(
        magnitudes,
        spectral_generator=torch.Generator().manual_seed(91),
    )
    reversed_snr, reversed_quality = build_observation_levels(
        magnitudes.flip(0),
        spectral_generator=torch.Generator().manual_seed(91),
    )
    assert not torch.equal(image_snr, reversed_snr)
    torch.testing.assert_close(quality, reversed_quality)
    assert bool(((quality >= 3.0) & (quality <= 100.0)).all())


def test_record_validation_requires_current_nine_target_schema():
    record = _current_record()
    rmag, halpha = validate_observation_record(record, location="training record 2")
    assert rmag == pytest.approx(20.0)
    assert halpha == pytest.approx(1.0e-15)

    bad = dict(record)
    bad["fid_pars"] = torch.zeros(8)
    with pytest.raises(ValueError, match="9"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad["observation_model_version"] = torch.tensor(1)
    with pytest.raises(ValueError, match="version"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad.pop("halpha_flux_true")
    with pytest.raises(ValueError, match="halpha_flux_true"):
        validate_observation_record(bad, location="training record 2")
