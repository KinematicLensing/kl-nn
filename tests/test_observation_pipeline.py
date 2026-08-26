import math

import pytest
import torch

import config
from data import (
    apply_central_halpha_snr_noise,
    apply_image_noise_for_snr,
    central_halpha_line_norm,
    deterministic_lower_median,
    image_matched_filter_norm,
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
        "image_snr": torch.tensor(250.0),
        "central_halpha_snr": torch.tensor(75.0),
        "observation_model_version": torch.tensor(3),
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


def test_lower_median_is_deterministic_algorithm_safe():
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
        assert deterministic_lower_median(values, dim=1).item() == 2.0
    finally:
        torch.use_deterministic_algorithms(previous)


def test_image_noise_uses_per_object_snr_and_repeats_by_seed():
    images = torch.stack((torch.ones(8, 8), 10.0 * torch.ones(8, 8)))[:, None]
    requested_snr = torch.tensor([5.0, 1000.0])
    noisy_a, metadata = apply_image_noise_for_snr(
        images,
        requested_snr,
        randgen=torch.Generator().manual_seed(17),
        return_metadata=True,
    )
    noisy_b = apply_image_noise_for_snr(
        images,
        requested_snr,
        randgen=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(noisy_a, noisy_b)
    expected_norm = image_matched_filter_norm(images)
    torch.testing.assert_close(metadata["clean_matched_filter_norm"], expected_norm)
    torch.testing.assert_close(metadata["noise_sigma"], expected_norm / requested_snr)
    torch.testing.assert_close(metadata["expected_image_snr"], requested_snr)


def test_central_halpha_noise_uses_requested_snr_and_count_exposure_scaling():
    spectra = torch.zeros(3, 1, 5, 16)
    spectra[..., :12] = 1.0
    spectra[:, 0, 2, 7:9] += torch.tensor([1.0, 2.0, 4.0])[:, None]
    requested_snr = torch.tensor([1.0, 10.0, 200.0])
    generator = torch.Generator().manual_seed(12)
    noisy, metadata = apply_central_halpha_snr_noise(
        spectra,
        requested_snr,
        center_fiber_index=2,
        center_exposure_s=180.0,
        offset_exposure_s=600.0,
        spectral_units="counts",
        randgen=generator,
        return_metadata=True,
    )
    assert noisy.shape == spectra.shape
    torch.testing.assert_close(noisy[..., 12:], spectra[..., 12:])
    sigma = metadata["noise_sigma"]
    assert sigma.shape == (3, 5)
    offsets = sigma[:, [0, 1, 3, 4]]
    torch.testing.assert_close(offsets, offsets[:, :1].expand(-1, 4))
    clean_norm = central_halpha_line_norm(spectra, center_fiber_index=2)
    torch.testing.assert_close(sigma[:, 2], clean_norm / requested_snr)
    torch.testing.assert_close(metadata["expected_central_halpha_snr"], requested_snr)
    torch.testing.assert_close(
        sigma[:, 0] / sigma[:, 2],
        torch.full((3,), math.sqrt(600.0 / 180.0)),
    )


def test_observation_levels_validate_explicit_record_backed_snrs():
    requested_image = torch.tensor([5.0, 250.0, 1000.0])
    requested_line = torch.tensor([1.0, 75.0, 200.0])
    image_snr, line_snr = build_observation_levels(
        requested_image, requested_line
    )
    torch.testing.assert_close(image_snr, requested_image)
    torch.testing.assert_close(line_snr, requested_line)

    with pytest.raises(ValueError, match="matching shapes"):
        build_observation_levels(requested_image, requested_line[:2])
    with pytest.raises(ValueError, match="image_snr"):
        build_observation_levels(torch.tensor([4.9]), torch.tensor([1.0]))
    with pytest.raises(ValueError, match="central_halpha_snr"):
        build_observation_levels(torch.tensor([5.0]), torch.tensor([200.1]))


def test_record_validation_requires_current_nine_target_schema():
    record = _current_record()
    rmag, halpha, image_snr, line_snr = validate_observation_record(
        record, location="training record 2"
    )
    assert rmag == pytest.approx(20.0)
    assert halpha == pytest.approx(1.0e-15)
    assert image_snr == pytest.approx(250.0)
    assert line_snr == pytest.approx(75.0)

    bad = dict(record)
    bad["fid_pars"] = torch.zeros(8)
    with pytest.raises(ValueError, match="9"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad["observation_model_version"] = torch.tensor(2)
    with pytest.raises(ValueError, match="version"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad.pop("halpha_flux_true")
    with pytest.raises(ValueError, match="halpha_flux_true"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad.pop("image_snr")
    with pytest.raises(ValueError, match="image_snr"):
        validate_observation_record(bad, location="training record 2")

    bad = dict(record)
    bad["central_halpha_snr"] = torch.tensor(201.0)
    with pytest.raises(ValueError, match="central_halpha_snr"):
        validate_observation_record(bad, location="training record 2")
