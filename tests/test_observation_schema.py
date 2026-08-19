import numpy as np
import pytest

from data_generate.observation_schema import (
    CENTER_EXPOSURE_COLUMN,
    CENTER_EXPOSURE_S,
    CENTER_FIBER_INDEX_COLUMN,
    CURRENT_OBSERVATION_MODEL_VERSION,
    DEFAULT_HALPHA_FLUX_RANGE,
    FIBER_LAYOUT_CODES,
    FIBER_LAYOUT_COLUMN,
    FITS_CENTER_EXPOSURE_KEY,
    FITS_CENTER_FIBER_INDEX_KEY,
    FITS_FIBER_LAYOUT_KEY,
    FITS_HALPHA_FLUX_TRUE_KEY,
    FITS_IMAGE_PIXEL_SCALE_KEY,
    FITS_IMAGE_PSF_FWHM_KEY,
    FITS_OBSERVATION_MODEL_VERSION_KEY,
    FITS_OFFSET_EXPOSURE_KEY,
    FITS_PHOTOMETRY_BAND_KEY,
    FITS_RMAG_TRUE_KEY,
    FITS_SPECTRAL_UNITS_KEY,
    FITS_TARGET_LINE_KEY,
    GALAXY_AXIS_FIBER_LAYOUT,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_AXIS_FIBER_LAYOUT,
    IMAGE_BAND_CODE_COLUMN,
    IMAGE_PSF_FWHM_COLUMN,
    IMAGE_PIXEL_SCALE_COLUMN,
    LEGACY_OBSERVATION_MODEL_VERSION,
    NON_TARGET_EMISSION_LINES,
    OBSERVATION_MODEL_VERSION_COLUMN,
    OFFSET_EXPOSURE_COLUMN,
    OFFSET_EXPOSURE_S,
    PHOTOMETRY_BAND,
    RMAG_TRUE_COLUMN,
    SPECTRAL_UNITS_CODE_COLUMN,
    TARGET_EMISSION_LINE,
    TARGET_LINE_CODE_COLUMN,
    compute_fiber_offsets,
    configure_sed,
    fiber_layout_from_header,
    halpha_flux_from_header,
    observation_metadata_arrays,
    observation_metadata_from_header,
    observation_instrument_metadata_from_header,
    resolve_halpha_flux,
    resolve_observation_model,
    validate_fiber_layout,
)


def _v2_header(**overrides):
    header = {
        "OBSMODV": 2,
        "RMAGTRUE": 20.25,
        "HAFLUX": 4.2e-15,
        "FIBLAY": "galaxy_axis",
        "PHOTBAND": "r",
        "TARGLINE": "Ha",
        "SPECUNIT": "counts",
        "CENFIB": 2,
        "CENEXPS": 180.0,
        "OFFEXPS": 600.0,
        "IMGPSF": 1.0,
        "IMGPIXS": 0.2637,
    }
    header.update(overrides)
    return header


def _legacy_sed():
    return {
        "cont_norm_method": "flux",
        "obs_cont_norm_wave": 850.0,
        "obs_cont_norm_flam": 3.0e-17,
        "em_Ha_flux": 1.2e-16,
        "em_O2_flux": 8.8e-17,
        "em_Hb_flux": 1.2e-17,
        "em_O3_1_flux": 2.4e-17,
        "em_O3_2_flux": 2.8e-17,
    }


def test_observation_schema_names_are_stable():
    assert RMAG_TRUE_COLUMN == "rmag_true"
    assert HALPHA_FLUX_TRUE_COLUMN == "halpha_flux_true"
    assert OBSERVATION_MODEL_VERSION_COLUMN == "observation_model_version"
    assert FITS_OBSERVATION_MODEL_VERSION_KEY == "OBSMODV"
    assert FITS_RMAG_TRUE_KEY == "RMAGTRUE"
    assert FITS_HALPHA_FLUX_TRUE_KEY == "HAFLUX"
    assert FITS_PHOTOMETRY_BAND_KEY == "PHOTBAND"
    assert FITS_TARGET_LINE_KEY == "TARGLINE"
    assert FITS_FIBER_LAYOUT_KEY == "FIBLAY"
    assert FITS_SPECTRAL_UNITS_KEY == "SPECUNIT"
    assert FITS_CENTER_FIBER_INDEX_KEY == "CENFIB"
    assert FITS_CENTER_EXPOSURE_KEY == "CENEXPS"
    assert FITS_OFFSET_EXPOSURE_KEY == "OFFEXPS"
    assert FITS_IMAGE_PSF_FWHM_KEY == "IMGPSF"
    assert FITS_IMAGE_PIXEL_SCALE_KEY == "IMGPIXS"
    assert PHOTOMETRY_BAND == "r"
    assert TARGET_EMISSION_LINE == "Ha"
    assert FIBER_LAYOUT_CODES == {"image_axis": 0, "galaxy_axis": 1}


def test_omitted_observation_metadata_preserves_legacy_model():
    version, rmag_true = resolve_observation_model(None, None)

    assert version == LEGACY_OBSERVATION_MODEL_VERSION
    assert rmag_true is None
    assert resolve_halpha_flux(version, None) is None


def test_two_nuisances_without_explicit_version_select_v2():
    version, rmag_true = resolve_observation_model(None, 21.25)
    halpha_flux_true = resolve_halpha_flux(version, 4.2e-15)

    assert version == CURRENT_OBSERVATION_MODEL_VERSION
    assert rmag_true == pytest.approx(21.25)
    assert halpha_flux_true == pytest.approx(4.2e-15, rel=1.0e-12, abs=0.0)


@pytest.mark.parametrize(
    ("version", "rmag_true", "message"),
    [
        (2, None, "requires a finite rmag_true"),
        (2, np.nan, "requires a finite rmag_true"),
        (1, 20.0, "only valid for observation model v2"),
        (3, 20.0, "Unsupported observation model version"),
    ],
)
def test_invalid_magnitude_metadata_is_rejected(version, rmag_true, message):
    with pytest.raises(ValueError, match=message):
        resolve_observation_model(version, rmag_true)


@pytest.mark.parametrize(
    ("version", "halpha_flux_true", "message"),
    [
        (2, None, "requires a finite positive halpha_flux_true"),
        (2, np.nan, "requires a finite positive halpha_flux_true"),
        (2, DEFAULT_HALPHA_FLUX_RANGE[0] / 2.0, "outside configured range"),
        (2, DEFAULT_HALPHA_FLUX_RANGE[1] * 2.0, "outside configured range"),
        (1, 4.2e-15, "only valid for observation model v2"),
        (3, 4.2e-15, "Unsupported observation model version"),
    ],
)
def test_invalid_halpha_metadata_is_rejected(
    version,
    halpha_flux_true,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_halpha_flux(
            version,
            halpha_flux_true,
            flux_range=DEFAULT_HALPHA_FLUX_RANGE,
        )


def test_v2_sed_uses_magnitude_and_sampled_halpha_flux():
    base = _legacy_sed()

    configured = configure_sed(
        base,
        version=CURRENT_OBSERVATION_MODEL_VERSION,
        rmag_true=20.75,
        halpha_flux_true=4.2e-15,
        r_bandpass="/calibration/DECam.r.dat",
    )

    assert configured["cont_norm_method"] == "mag"
    assert configured["obs_norm_band"] == "/calibration/DECam.r.dat"
    assert configured["obs_norm_mag"] == pytest.approx(20.75)
    assert "obs_cont_norm_wave" not in configured
    assert "obs_cont_norm_flam" not in configured
    assert configured["em_Ha_flux"] == pytest.approx(4.2e-15, rel=1.0e-12, abs=0.0)
    for line_name in NON_TARGET_EMISSION_LINES:
        assert configured[f"em_{line_name}_flux"] == 0.0

    # Configuration must be functional: callers can safely reuse the template.
    assert base == _legacy_sed()


def test_legacy_sed_configuration_is_unchanged_copy():
    base = _legacy_sed()

    configured = configure_sed(
        base,
        version=LEGACY_OBSERVATION_MODEL_VERSION,
        rmag_true=None,
        halpha_flux_true=None,
        r_bandpass="ignored.dat",
    )

    assert configured == base
    assert configured is not base


def test_header_metadata_defaults_old_fits_to_v1_and_nan_nuisances():
    version, rmag_true = observation_metadata_from_header({})

    assert version == LEGACY_OBSERVATION_MODEL_VERSION
    assert np.isnan(rmag_true)
    assert np.isnan(halpha_flux_from_header({}))


def test_header_metadata_extracts_v2_values():
    header = {
        "OBSMODV": 2,
        "RMAGTRUE": 19.625,
        "HAFLUX": 4.2e-15,
        "PHOTBAND": "r",
        "TARGLINE": "Ha",
    }
    version, rmag_true = observation_metadata_from_header(header)
    halpha_flux_true = halpha_flux_from_header(header, version=version)

    assert version == CURRENT_OBSERVATION_MODEL_VERSION
    assert rmag_true == pytest.approx(19.625)
    assert halpha_flux_true == pytest.approx(4.2e-15, rel=1.0e-12, abs=0.0)


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("RMAGTRUE", "requires a finite rmag_true"),
        ("HAFLUX", "requires a finite positive halpha_flux_true"),
    ],
)
def test_header_v2_requires_both_nuisances(missing_key, message):
    header = _v2_header()
    header.pop(missing_key)
    with pytest.raises(ValueError, match=message):
        observation_metadata_from_header(header)


def test_v2_instrument_schema_is_strict_and_decoded():
    decoded = observation_instrument_metadata_from_header(
        _v2_header(), version=2
    )

    assert decoded == {
        IMAGE_BAND_CODE_COLUMN: 0,
        TARGET_LINE_CODE_COLUMN: 0,
        SPECTRAL_UNITS_CODE_COLUMN: 0,
        CENTER_FIBER_INDEX_COLUMN: 2,
        CENTER_EXPOSURE_COLUMN: pytest.approx(CENTER_EXPOSURE_S),
        OFFSET_EXPOSURE_COLUMN: pytest.approx(OFFSET_EXPOSURE_S),
        IMAGE_PSF_FWHM_COLUMN: pytest.approx(1.0),
        IMAGE_PIXEL_SCALE_COLUMN: pytest.approx(0.2637),
    }


def test_v2_instrument_schema_rejects_missing_or_mismatched_metadata():
    missing = _v2_header()
    missing.pop("SPECUNIT")
    with pytest.raises(ValueError, match="missing required schema metadata"):
        observation_instrument_metadata_from_header(missing, version=2)

    with pytest.raises(ValueError, match="CENEXPS"):
        observation_instrument_metadata_from_header(
            _v2_header(CENEXPS=200.0), version=2
        )
    with pytest.raises(ValueError, match="IMGPSF must be positive"):
        observation_instrument_metadata_from_header(
            _v2_header(IMGPSF=0.0), version=2
        )


def test_legacy_instrument_schema_alone_may_omit_new_metadata():
    decoded = observation_instrument_metadata_from_header({}, version=1)

    assert decoded[IMAGE_BAND_CODE_COLUMN] == -1
    assert decoded[TARGET_LINE_CODE_COLUMN] == -1
    assert decoded[SPECTRAL_UNITS_CODE_COLUMN] == -1
    assert decoded[CENTER_FIBER_INDEX_COLUMN] == -1
    assert np.isnan(decoded[CENTER_EXPOSURE_COLUMN])
    assert np.isnan(decoded[OFFSET_EXPOSURE_COLUMN])
    assert np.isnan(decoded[IMAGE_PSF_FWHM_COLUMN])
    assert np.isnan(decoded[IMAGE_PIXEL_SCALE_COLUMN])


def test_fiber_layout_header_defaults_to_historical_image_axes():
    assert validate_fiber_layout(None) == IMAGE_AXIS_FIBER_LAYOUT
    assert fiber_layout_from_header({}) == (IMAGE_AXIS_FIBER_LAYOUT, 0)
    assert fiber_layout_from_header({"FIBLAY": "galaxy_axis"}) == (
        GALAXY_AXIS_FIBER_LAYOUT,
        1,
    )


def test_v2_fiber_layout_never_uses_the_legacy_fallback():
    header = _v2_header()
    header.pop("FIBLAY")

    with pytest.raises(ValueError, match="missing required FIBLAY"):
        observation_metadata_arrays([header])


def test_unknown_fiber_layout_is_rejected():
    with pytest.raises(ValueError, match="Unsupported fiber layout"):
        validate_fiber_layout("detector_diagonal")


def test_image_axis_offsets_preserve_pairing_under_permutation():
    canonical = np.asarray(
        [[1.5, 0.0], [-1.5, 0.0], [0.0, 0.0], [0.0, 1.5], [0.0, -1.5]]
    )
    permutation = [4, 2, 0, 3, 1]
    offsets = compute_fiber_offsets(
        fiber_layout=IMAGE_AXIS_FIBER_LAYOUT,
        fiber_offset=1.5,
        g1=0.08,
        g2=-0.04,
        theta_int=1.2,
        sini=0.75,
        fiber_permutation=permutation,
    )

    np.testing.assert_allclose(offsets, canonical[permutation], atol=1e-14)


def test_galaxy_axis_offsets_keep_center_and_opposite_pairs():
    offsets = compute_fiber_offsets(
        fiber_layout=GALAXY_AXIS_FIBER_LAYOUT,
        fiber_offset=1.5,
        g1=0.07,
        g2=-0.03,
        theta_int=0.63,
        sini=0.8,
    )

    np.testing.assert_allclose(offsets[0], -offsets[1], atol=1e-14)
    np.testing.assert_allclose(offsets[3], -offsets[4], atol=1e-14)
    np.testing.assert_allclose(offsets[2], [0.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(
        np.linalg.norm(offsets[[0, 1, 3, 4]], axis=1), 1.5
    )
    assert not np.allclose(offsets[0], [1.5, 0.0])


@pytest.mark.parametrize(
    ("sini", "permutation", "message"),
    [
        (-0.01, (0, 1, 2, 3, 4), "sini must be"),
        (1.01, (0, 1, 2, 3, 4), "sini must be"),
        (0.5, (0, 1, 2, 4, 4), "fiber_permutation"),
    ],
)
def test_invalid_fiber_geometry_is_rejected(sini, permutation, message):
    with pytest.raises(ValueError, match=message):
        compute_fiber_offsets(
            fiber_layout=IMAGE_AXIS_FIBER_LAYOUT,
            fiber_offset=1.5,
            g1=0.0,
            g2=0.0,
            theta_int=0.0,
            sini=sini,
            fiber_permutation=permutation,
        )


def test_lmdb_metadata_arrays_have_stable_defaults_dtypes_and_layout_codes():
    metadata = observation_metadata_arrays(
        [
            {},
            _v2_header(),
        ]
    )

    assert set(metadata) == {
        RMAG_TRUE_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        FIBER_LAYOUT_COLUMN,
        IMAGE_BAND_CODE_COLUMN,
        TARGET_LINE_CODE_COLUMN,
        SPECTRAL_UNITS_CODE_COLUMN,
        CENTER_FIBER_INDEX_COLUMN,
        CENTER_EXPOSURE_COLUMN,
        OFFSET_EXPOSURE_COLUMN,
        IMAGE_PSF_FWHM_COLUMN,
        IMAGE_PIXEL_SCALE_COLUMN,
    }
    assert metadata[RMAG_TRUE_COLUMN].dtype == np.float32
    assert metadata[HALPHA_FLUX_TRUE_COLUMN].dtype == np.float32
    assert metadata[OBSERVATION_MODEL_VERSION_COLUMN].dtype == np.int16
    assert metadata[FIBER_LAYOUT_COLUMN].dtype == np.int8
    assert metadata[IMAGE_BAND_CODE_COLUMN].dtype == np.int8
    assert metadata[CENTER_EXPOSURE_COLUMN].dtype == np.float32
    assert np.isnan(metadata[RMAG_TRUE_COLUMN][0])
    assert np.isnan(metadata[HALPHA_FLUX_TRUE_COLUMN][0])
    assert metadata[RMAG_TRUE_COLUMN][1] == pytest.approx(20.25)
    assert metadata[HALPHA_FLUX_TRUE_COLUMN][1] == pytest.approx(4.2e-15, rel=1.0e-6, abs=0.0)
    np.testing.assert_array_equal(
        metadata[OBSERVATION_MODEL_VERSION_COLUMN], [1, 2]
    )
    np.testing.assert_array_equal(metadata[FIBER_LAYOUT_COLUMN], [0, 1])
    np.testing.assert_array_equal(metadata[IMAGE_BAND_CODE_COLUMN], [-1, 0])
    np.testing.assert_array_equal(metadata[TARGET_LINE_CODE_COLUMN], [-1, 0])
    np.testing.assert_array_equal(metadata[SPECTRAL_UNITS_CODE_COLUMN], [-1, 0])
    np.testing.assert_array_equal(metadata[CENTER_FIBER_INDEX_COLUMN], [-1, 2])
    assert np.isnan(metadata[CENTER_EXPOSURE_COLUMN][0])
    assert metadata[CENTER_EXPOSURE_COLUMN][1] == pytest.approx(180.0)
    assert metadata[OFFSET_EXPOSURE_COLUMN][1] == pytest.approx(600.0)
    assert metadata[IMAGE_PSF_FWHM_COLUMN][1] == pytest.approx(1.0)
    assert metadata[IMAGE_PIXEL_SCALE_COLUMN][1] == pytest.approx(0.2637)
