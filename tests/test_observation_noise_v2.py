import numpy as np
import pytest
import torch

from data import (
    app_mag_to_snr,
    apply_fixed_gaussian_image_noise,
    apply_noise,
    apply_spectral_noise,
    depth_scaled_total_image_flux,
    deterministic_lower_median,
    estimate_spectral_reference_line_norm,
    fixed_image_noise_sigma_from_depth_fluxes,
    gaussian_psf_noise_equivalent_pixels,
    magnitude_uncertainty_from_snr,
    sample_observed_magnitude,
)


def test_magnitude_to_image_snr_uses_forward_depth_relation():
    magnitudes = np.asarray([20.9, 23.4, 25.9])

    snr = app_mag_to_snr(
        magnitudes,
        band="r",
        depth_5sigma_mag=23.4,
    )

    np.testing.assert_allclose(snr, [50.0, 5.0, 0.5], rtol=1e-12)
    torch.testing.assert_close(
        app_mag_to_snr(
            torch.tensor(magnitudes, dtype=torch.float64),
            band="r",
            depth_5sigma_mag=23.4,
        ),
        torch.tensor([50.0, 5.0, 0.5], dtype=torch.float64),
    )


def test_magnitude_uncertainty_is_derived_from_positive_flux_snr():
    snr = np.asarray([5.0, 10.0, 20.0])
    expected = (2.5 / np.log(10.0)) / snr

    np.testing.assert_allclose(magnitude_uncertainty_from_snr(snr), expected)
    torch.testing.assert_close(
        magnitude_uncertainty_from_snr(torch.tensor(snr)),
        torch.tensor(expected),
    )

    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite positive"):
            magnitude_uncertainty_from_snr(invalid)


def test_apply_noise_return_scale_preserves_legacy_default_output():
    clean = torch.ones((6, 1, 8, 8), dtype=torch.float32)
    snr = torch.linspace(10.0, 100.0, len(clean))

    legacy = apply_noise(
        clean,
        snr,
        randgen=torch.Generator().manual_seed(441),
        device="cpu",
    )
    noisy, scale = apply_noise(
        clean,
        snr,
        randgen=torch.Generator().manual_seed(441),
        device="cpu",
        return_scale=True,
    )

    torch.testing.assert_close(noisy, legacy)
    assert scale.shape == (len(clean),)
    assert scale.dtype == clean.dtype
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()


def test_gaussian_psf_equivalent_depth_calibration_matches_closed_form():
    fwhm = 1.0
    pixel_scale = 0.2637
    sigma_pixels = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)) * pixel_scale)
    expected_n_eff = 4.0 * np.pi * sigma_pixels**2

    n_eff = gaussian_psf_noise_equivalent_pixels(fwhm, pixel_scale)

    assert n_eff == pytest.approx(expected_n_eff)
    assert n_eff == pytest.approx(32.57, rel=2e-3)


def test_depth_scaled_flux_and_fixed_sigma_use_one_population_median():
    magnitudes = torch.tensor([20.0, 21.0, 22.0], dtype=torch.float64)
    # All three images represent the same reference source scaled by AB flux.
    total_flux = 100.0 * torch.pow(10.0, -0.4 * (magnitudes - 20.0))
    images = (total_flux / 4.0).reshape(3, 1, 1, 1).expand(-1, 1, 2, 2)
    depth_fluxes = depth_scaled_total_image_flux(images, magnitudes, 23.4)

    expected_f5 = 100.0 * 10.0 ** (-0.4 * (23.4 - 20.0))
    torch.testing.assert_close(
        depth_fluxes, torch.full_like(depth_fluxes, expected_f5)
    )
    n_eff = gaussian_psf_noise_equivalent_pixels(1.0, 0.2637)
    sigma = fixed_image_noise_sigma_from_depth_fluxes(depth_fluxes, n_eff)
    assert sigma.item() == pytest.approx(expected_f5 / (5.0 * np.sqrt(n_eff)))


def test_fixed_image_noise_draw_is_independent_of_clean_morphology_and_flux():
    first_clean = torch.zeros((2, 1, 16, 16), dtype=torch.float64)
    second_clean = torch.linspace(
        0.0, 1000.0, first_clean.numel(), dtype=torch.float64
    ).reshape_as(first_clean)
    sigma = 0.375

    first = apply_fixed_gaussian_image_noise(
        first_clean, sigma, randgen=torch.Generator().manual_seed(909)
    )
    second = apply_fixed_gaussian_image_noise(
        second_clean, sigma, randgen=torch.Generator().manual_seed(909)
    )

    torch.testing.assert_close(first - first_clean, second - second_clean)
    assert float((first - first_clean).std()) == pytest.approx(sigma, rel=0.08)
    with pytest.raises(ValueError, match="single global scalar"):
        apply_fixed_gaussian_image_noise(first_clean, torch.tensor([0.2, 0.3]))


def test_catalog_flux_snr_pulls_are_standard_normal_and_drive_reported_magnitude():
    batch_size = 4096
    rmag_true = torch.linspace(16.0, 23.0, batch_size)
    image_snr = app_mag_to_snr(
        rmag_true,
        band="r",
        depth_5sigma_mag=23.4,
    )
    observed = sample_observed_magnitude(
        rmag_true,
        image_snr,
        randgen=torch.Generator().manual_seed(71),
    )
    observed_snr = observed["image_flux_snr"]
    flux_pull = observed_snr - image_snr

    assert abs(float(flux_pull.mean())) < 0.05
    assert abs(float(flux_pull.std()) - 1.0) < 0.05
    assert not torch.equal(observed_snr, image_snr)
    torch.testing.assert_close(
        observed["rmag_obs"],
        rmag_true - 2.5 * torch.log10(observed_snr / image_snr),
    )
    torch.testing.assert_close(
        observed["rmag_sigma"],
        magnitude_uncertainty_from_snr(observed_snr),
    )


def test_catalog_magnitude_draw_is_reproducible_and_has_no_image_dependency():
    rmag_true = torch.tensor([17.5, 20.0, 22.5])
    image_snr = torch.tensor([100.0, 20.0, 5.0])

    first = sample_observed_magnitude(
        rmag_true,
        image_snr,
        randgen=torch.Generator().manual_seed(8),
    )
    repeated = sample_observed_magnitude(
        rmag_true,
        image_snr,
        randgen=torch.Generator().manual_seed(8),
    )

    assert set(first) == {"rmag_obs", "rmag_sigma", "image_flux_snr"}
    for key in first:
        torch.testing.assert_close(first[key], repeated[key])
    assert not torch.equal(first["image_flux_snr"], image_snr)


def test_catalog_flux_likelihood_matches_seeded_gaussian_draw_exactly():
    rmag_true = torch.tensor([18.0, 20.0, 22.0], dtype=torch.float64)
    expected_snr = torch.tensor([80.0, 30.0, 8.0], dtype=torch.float64)
    noise = torch.randn(
        expected_snr.shape,
        dtype=expected_snr.dtype,
        generator=torch.Generator().manual_seed(112),
    )
    observed_snr = expected_snr + noise

    observed = sample_observed_magnitude(
        rmag_true,
        expected_snr,
        randgen=torch.Generator().manual_seed(112),
    )

    torch.testing.assert_close(observed["image_flux_snr"], observed_snr)
    torch.testing.assert_close(
        observed["rmag_obs"],
        rmag_true - 2.5 * torch.log10(observed_snr / expected_snr),
    )
    torch.testing.assert_close(
        observed["rmag_sigma"],
        (2.5 / np.log(10.0)) / observed_snr,
    )


def _spectra_for_noise_tests():
    spectra = torch.zeros((3, 1, 5, 16), dtype=torch.float32)
    amplitudes = torch.tensor(
        [[[1.0, 2.0, 9.0, 3.0, 4.0]],
         [[1.0, 2.0, 8.0, 3.0, 4.0]],
         [[1.0, 2.0, 7.0, 3.0, 4.0]]]
    )
    spectra[:, :, :, 8] = amplitudes
    return spectra


def test_spectral_reference_norm_ignores_central_fiber():
    spectra = _spectra_for_noise_tests()

    reference = estimate_spectral_reference_line_norm(
        spectra, center_fiber_index=2
    )

    # The deterministic replacement preserves the lower-middle convention.
    assert reference.item() == pytest.approx(2.0)


def test_deterministic_lower_median_preserves_torch_median_semantics():
    values = torch.tensor([9.0, 1.0, 4.0, 2.0])
    assert deterministic_lower_median(values).item() == pytest.approx(2.0)

    matrix = torch.tensor([[9.0, 1.0, 4.0, 2.0], [8.0, 7.0, 6.0, 5.0]])
    expected = torch.tensor([[2.0], [6.0]])
    torch.testing.assert_close(
        deterministic_lower_median(matrix, dim=-1, keepdim=True), expected
    )


def test_v2_reference_calibration_never_calls_nondeterministic_torch_median(
    monkeypatch,
):
    def forbidden_median(*args, **kwargs):
        raise AssertionError("v2 calibration called torch.median")

    monkeypatch.setattr(torch, "median", forbidden_median)
    reference = estimate_spectral_reference_line_norm(
        _spectra_for_noise_tests(), center_fiber_index=2
    )
    sigma = fixed_image_noise_sigma_from_depth_fluxes(
        torch.tensor([3.0, 1.0, 4.0, 2.0]), 16.0
    )

    assert reference.item() == pytest.approx(2.0)
    assert sigma.item() == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("spectral_units", "expected_center_ratio"),
    [
        ("counts", np.sqrt(180.0 / 600.0)),
        ("count_rate", np.sqrt(600.0 / 180.0)),
    ],
)
def test_spectral_noise_offsets_match_and_center_obeys_exposure_units(
    spectral_units,
    expected_center_ratio,
):
    spectra = _spectra_for_noise_tests()
    qualities = torch.tensor([5.0, 10.0, 20.0])
    reference = estimate_spectral_reference_line_norm(spectra)

    _, metadata = apply_spectral_noise(
        spectra,
        qualities,
        reference,
        center_fiber_index=2,
        center_exposure_s=180.0,
        offset_exposure_s=600.0,
        spectral_units=spectral_units,
        randgen=torch.Generator().manual_seed(90),
        return_metadata=True,
    )

    sigma = metadata["noise_sigma"]
    offset_sigma = sigma[:, [0, 1, 3, 4]]
    torch.testing.assert_close(offset_sigma, sigma[:, :1].expand_as(offset_sigma))
    torch.testing.assert_close(
        sigma[:, 2], sigma[:, 0] * expected_center_ratio
    )
    torch.testing.assert_close(metadata["reference_quality"], qualities)
    assert metadata["achieved_line_snr"].shape == (3, 1, 5)


def test_spectral_noise_is_repeatable_and_independent_of_image_noise_stream():
    spectra = _spectra_for_noise_tests()
    reference = estimate_spectral_reference_line_norm(spectra)
    quality = torch.tensor([8.0, 12.0, 30.0])

    first = apply_spectral_noise(
        spectra,
        quality,
        reference,
        randgen=torch.Generator().manual_seed(105),
    )
    repeated = apply_spectral_noise(
        spectra,
        quality,
        reference,
        randgen=torch.Generator().manual_seed(105),
    )
    other_stream = apply_spectral_noise(
        spectra,
        quality,
        reference,
        randgen=torch.Generator().manual_seed(106),
    )

    torch.testing.assert_close(first, repeated)
    assert not torch.equal(first, other_stream)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_quality": 0.0}, "finite positive"),
        ({"reference_quality": np.nan}, "finite positive"),
        ({"reference_line_norm": 0.0}, "finite and positive"),
        ({"center_exposure_s": 0.0}, "exposure times"),
        ({"spectral_units": "flux"}, "counts.*count_rate"),
    ],
)
def test_invalid_spectral_noise_requests_are_rejected(kwargs, message):
    spectra = _spectra_for_noise_tests()
    defaults = {
        "reference_quality": 10.0,
        "reference_line_norm": 2.0,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        apply_spectral_noise(spectra, **defaults)
