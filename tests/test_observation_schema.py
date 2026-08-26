import numpy as np
import pytest

from data_generate import observation_schema as schema


def _header(**overrides):
    header = {
        schema.FITS_OBSERVATION_MODEL_VERSION_KEY: 3,
        schema.FITS_RMAG_TRUE_KEY: 20.25,
        schema.FITS_HALPHA_FLUX_TRUE_KEY: 4.2e-15,
        schema.FITS_IMAGE_SNR_KEY: 240.0,
        schema.FITS_CENTER_HALPHA_SNR_KEY: 37.0,
        schema.FITS_HALPHA_FLUX_UNITS_KEY: schema.HALPHA_FLUX_UNITS,
        schema.FITS_HALPHA_FLUX_SEMANTICS_KEY: schema.HALPHA_FLUX_SEMANTICS,
        schema.FITS_HALPHA_FLUX_TRANSFORM_KEY: schema.HALPHA_FLUX_TRANSFORM,
        schema.FITS_HALPHA_FLUX_API_VERSION_KEY: schema.HALPHA_FLUX_API_VERSION,
        schema.FITS_HALPHA_TOTAL_FLUX_KEY: 2.0e-14,
        schema.FITS_CENTER_HALPHA_APERTURE_KEY: 0.21,
        schema.FITS_FIBER_LAYOUT_KEY: "galaxy_axis",
        schema.FITS_PHOTOMETRY_BAND_KEY: "r",
        schema.FITS_TARGET_LINE_KEY: "Ha",
        schema.FITS_SPECTRAL_UNITS_KEY: "counts",
        schema.FITS_CENTER_FIBER_INDEX_KEY: 2,
        schema.FITS_CENTER_EXPOSURE_KEY: 180.0,
        schema.FITS_OFFSET_EXPOSURE_KEY: 600.0,
        schema.FITS_IMAGE_PSF_FWHM_KEY: 1.0,
        schema.FITS_IMAGE_PIXEL_SCALE_KEY: 0.2637,
    }
    header.update(overrides)
    return header


def test_sed_uses_true_magnitude_and_central_fiber_halpha_only():
    base = {
        "cont_norm_method": "flux",
        "obs_cont_norm_wave": 850.0,
        "obs_cont_norm_flam": 3e-17,
        "em_Ha_flux": 1.2e-16,
        **{f"em_{name}_flux": 1.0 for name in schema.NON_TARGET_EMISSION_LINES},
    }
    configured = schema.configure_sed(
        base,
        rmag_true=20.75,
        halpha_flux_true=4.2e-15,
        r_bandpass="DECam.r.dat",
        center_fiber_obsindex=2,
    )
    assert configured["cont_norm_method"] == "mag"
    assert configured["obs_norm_mag"] == pytest.approx(20.75)
    assert configured["obs_norm_band"] == "DECam.r.dat"
    assert configured["em_Ha_flux"] == pytest.approx(4.2e-15)
    assert configured["em_Ha_flux_semantics"] == "central_fiber"
    assert configured["em_Ha_flux_reference_obsindex"] == 2
    assert "obs_cont_norm_wave" not in configured
    assert "obs_cont_norm_flam" not in configured
    for name in schema.NON_TARGET_EMISSION_LINES:
        assert configured[f"em_{name}_flux"] == 0.0
    assert base["cont_norm_method"] == "flux"


@pytest.mark.parametrize("value", [None, np.nan, 0.0, -1.0])
def test_invalid_halpha_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        schema.validate_halpha_flux(value)


def test_galaxy_axis_offsets_have_one_center_and_fixed_radius():
    offsets = schema.compute_fiber_offsets(
        fiber_offset=1.5,
        g1=0.03,
        g2=-0.02,
        theta_int=0.7,
        sini=0.8,
    )
    assert offsets.shape == (5, 2)
    np.testing.assert_allclose(offsets[2], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(offsets[[0, 1, 3, 4]], axis=1), 1.5, atol=1e-12
    )
    with pytest.raises(ValueError, match="only galaxy_axis"):
        schema.validate_fiber_layout("detector_axis")


def test_strict_header_round_trip_and_lmdb_arrays():
    header = _header()
    assert schema.observation_metadata_from_header(header) == (
        3,
        20.25,
        240.0,
        37.0,
    )
    assert schema.halpha_flux_from_header(header) == pytest.approx(4.2e-15)
    assert schema.fiber_layout_from_header(header) == ("galaxy_axis", 1)
    decoded = schema.observation_instrument_metadata_from_header(header)
    assert decoded[schema.CENTER_FIBER_INDEX_COLUMN] == 2
    arrays = schema.observation_metadata_arrays([header, _header(RMAGTRUE=19.5)])
    assert set(arrays) == {
        schema.RMAG_TRUE_COLUMN,
        schema.HALPHA_FLUX_TRUE_COLUMN,
        schema.IMAGE_SNR_COLUMN,
        schema.CENTRAL_HALPHA_SNR_COLUMN,
        schema.OBSERVATION_MODEL_VERSION_COLUMN,
        schema.FIBER_LAYOUT_COLUMN,
        schema.IMAGE_BAND_CODE_COLUMN,
        schema.TARGET_LINE_CODE_COLUMN,
        schema.SPECTRAL_UNITS_CODE_COLUMN,
        schema.CENTER_FIBER_INDEX_COLUMN,
        schema.CENTER_EXPOSURE_COLUMN,
        schema.OFFSET_EXPOSURE_COLUMN,
        schema.IMAGE_PSF_FWHM_COLUMN,
        schema.IMAGE_PIXEL_SCALE_COLUMN,
        schema.HALPHA_FLUX_UNITS_CODE_COLUMN,
        schema.HALPHA_FLUX_SEMANTICS_CODE_COLUMN,
        schema.HALPHA_FLUX_TRANSFORM_CODE_COLUMN,
        schema.HALPHA_FLUX_API_VERSION_COLUMN,
        schema.HALPHA_TOTAL_FLUX_COLUMN,
        schema.CENTRAL_HALPHA_APERTURE_FRACTION_COLUMN,
    }
    np.testing.assert_allclose(arrays[schema.RMAG_TRUE_COLUMN], [20.25, 19.5])
    assert arrays[schema.OBSERVATION_MODEL_VERSION_COLUMN].dtype == np.int16
    np.testing.assert_allclose(arrays[schema.IMAGE_SNR_COLUMN], [240.0, 240.0])
    np.testing.assert_allclose(
        arrays[schema.CENTRAL_HALPHA_SNR_COLUMN], [37.0, 37.0]
    )
    np.testing.assert_allclose(
        arrays[schema.HALPHA_TOTAL_FLUX_COLUMN], [2.0e-14, 2.0e-14]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"OBSMODV": 99},
        {"FIBLAY": "detector_axis"},
        {"PHOTBAND": "g"},
        {"TARGLINE": "O2"},
        {"SPECUNIT": "flux"},
        {"CENFIB": 1},
        {"CENEXPS": 200.0},
        {"IMGPSF": 0.5},
        {"IMGSNR": 0.0},
        {"CENHASNR": np.nan},
        {"HAFSEM": "total"},
        {"HAFTRAN": "identity"},
        {"HAFAPI": 99},
        {"HACENAP": 0.5},
    ],
)
def test_mixed_or_wrong_schema_fails_closed(change):
    header = _header(**change)
    with pytest.raises(ValueError):
        schema.observation_metadata_arrays([header])


def test_missing_metadata_never_falls_back():
    for key in _header():
        header = _header()
        del header[key]
        with pytest.raises(ValueError, match="missing required"):
            schema.observation_metadata_arrays([header])
