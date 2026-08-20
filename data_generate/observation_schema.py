"""Strict schema for the sole KL-NN simulator-v2 observation model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


OBSERVATION_MODEL_VERSION = 2
RMAG_TRUE_COLUMN = "rmag_true"
HALPHA_FLUX_TRUE_COLUMN = "halpha_flux_true"
OBSERVATION_MODEL_VERSION_COLUMN = "observation_model_version"
FIBER_LAYOUT_COLUMN = "fiber_layout"

FITS_OBSERVATION_MODEL_VERSION_KEY = "OBSMODV"
FITS_RMAG_TRUE_KEY = "RMAGTRUE"
FITS_HALPHA_FLUX_TRUE_KEY = "HAFLUX"
FITS_PHOTOMETRY_BAND_KEY = "PHOTBAND"
FITS_TARGET_LINE_KEY = "TARGLINE"
FITS_FIBER_LAYOUT_KEY = "FIBLAY"
FITS_SPECTRAL_UNITS_KEY = "SPECUNIT"
FITS_CENTER_FIBER_INDEX_KEY = "CENFIB"
FITS_CENTER_EXPOSURE_KEY = "CENEXPS"
FITS_OFFSET_EXPOSURE_KEY = "OFFEXPS"
FITS_IMAGE_PSF_FWHM_KEY = "IMGPSF"
FITS_IMAGE_PIXEL_SCALE_KEY = "IMGPIXS"

PHOTOMETRY_BAND = "r"
TARGET_EMISSION_LINE = "Ha"
SPECTRAL_UNITS = "counts"
CENTER_FIBER_INDEX = 2
CENTER_EXPOSURE_S = 180.0
OFFSET_EXPOSURE_S = 600.0
IMAGE_REFERENCE_PSF_FWHM_ARCSEC = 1.0
IMAGE_PIXEL_SCALE_ARCSEC = 0.2637
NON_TARGET_EMISSION_LINES = ("O2", "Hb", "O3_1", "O3_2")
DEFAULT_HALPHA_FLUX_RANGE = (1.2e-16, 301.43e-16)
GALAXY_AXIS_FIBER_LAYOUT = "galaxy_axis"
FIBER_LAYOUT_CODE = 1

IMAGE_BAND_CODE_COLUMN = "image_band_code"
TARGET_LINE_CODE_COLUMN = "target_line_code"
SPECTRAL_UNITS_CODE_COLUMN = "spectral_units_code"
CENTER_FIBER_INDEX_COLUMN = "center_fiber_index"
CENTER_EXPOSURE_COLUMN = "center_exposure_s"
OFFSET_EXPOSURE_COLUMN = "offset_exposure_s"
IMAGE_PSF_FWHM_COLUMN = "image_reference_psf_fwhm_arcsec"
IMAGE_PIXEL_SCALE_COLUMN = "image_pixel_scale_arcsec"


def validate_rmag_true(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("rmag_true must be finite")
    return value


def validate_halpha_flux(value: float) -> float:
    """Validate integrated observer-frame H-alpha flux [erg s^-1 cm^-2]."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("halpha_flux_true must be finite and positive")
    return value


def configure_sed(
    base_sed: Mapping[str, Any],
    *,
    rmag_true: float,
    halpha_flux_true: float,
    r_bandpass: str,
) -> dict[str, Any]:
    """Configure a magnitude-normalized continuum and H-alpha-only spectrum."""

    magnitude = validate_rmag_true(rmag_true)
    halpha_flux = validate_halpha_flux(halpha_flux_true)
    sed = dict(base_sed)
    sed.update(
        {
            "cont_norm_method": "mag",
            "obs_norm_band": str(r_bandpass),
            "obs_norm_mag": magnitude,
            "em_Ha_flux": halpha_flux,
        }
    )
    sed.pop("obs_cont_norm_wave", None)
    sed.pop("obs_cont_norm_flam", None)
    for line_name in NON_TARGET_EMISSION_LINES:
        sed[f"em_{line_name}_flux"] = 0.0
    return sed


def validate_fiber_layout(value: str) -> str:
    value = str(value)
    if value != GALAXY_AXIS_FIBER_LAYOUT:
        raise ValueError(
            "The current simulator supports only galaxy_axis fiber placement"
        )
    return value


def compute_fiber_offsets(
    *,
    fiber_offset: float,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
) -> np.ndarray:
    """Return the five fibers aligned with the observed galaxy axes."""

    if not np.isfinite(fiber_offset) or fiber_offset <= 0.0:
        raise ValueError("fiber_offset must be finite and positive")
    if not 0.0 <= sini <= 1.0:
        raise ValueError(f"sini must be in [0, 1], got {sini}")
    offsets = np.asarray(
        [
            (fiber_offset, 0.0),
            (-fiber_offset, 0.0),
            (0.0, 0.0),
            (0.0, fiber_offset),
            (0.0, -fiber_offset),
        ],
        dtype=float,
    )
    cosi = np.sqrt(max(0.0, 1.0 - sini**2))
    shear = np.asarray([[1.0 + g1, g2], [g2, 1.0 - g1]])
    rotation = np.asarray(
        [
            [np.cos(theta_int), -np.sin(theta_int)],
            [np.sin(theta_int), np.cos(theta_int)],
        ]
    )
    projection = np.asarray([[1.0, 0.0], [0.0, cosi]])
    transform = shear @ (rotation @ projection)
    observed_axes, _, _ = np.linalg.svd(transform)
    reference_axis = transform @ np.asarray([1.0, 0.0])
    if np.dot(observed_axes[:, 0], reference_axis) < 0:
        observed_axes *= -1.0
    return offsets @ observed_axes


def _required(header: Mapping[str, Any], key: str) -> Any:
    if key not in header:
        raise ValueError(f"Simulator-v2 FITS header is missing required {key}")
    return header[key]


def observation_metadata_from_header(header: Mapping[str, Any]) -> tuple[int, float]:
    version = int(_required(header, FITS_OBSERVATION_MODEL_VERSION_KEY))
    if version != OBSERVATION_MODEL_VERSION:
        raise ValueError(
            f"Expected observation model {OBSERVATION_MODEL_VERSION}, got {version}"
        )
    return version, validate_rmag_true(_required(header, FITS_RMAG_TRUE_KEY))


def halpha_flux_from_header(header: Mapping[str, Any]) -> float:
    return validate_halpha_flux(_required(header, FITS_HALPHA_FLUX_TRUE_KEY))


def fiber_layout_from_header(header: Mapping[str, Any]) -> tuple[str, int]:
    layout = validate_fiber_layout(_required(header, FITS_FIBER_LAYOUT_KEY))
    return layout, FIBER_LAYOUT_CODE


def observation_instrument_metadata_from_header(
    header: Mapping[str, Any],
) -> dict[str, int | float]:
    image_band = str(_required(header, FITS_PHOTOMETRY_BAND_KEY)).strip()
    target_line = str(_required(header, FITS_TARGET_LINE_KEY)).strip()
    spectral_units = str(_required(header, FITS_SPECTRAL_UNITS_KEY)).strip()
    center_index = int(_required(header, FITS_CENTER_FIBER_INDEX_KEY))
    center_exposure = float(_required(header, FITS_CENTER_EXPOSURE_KEY))
    offset_exposure = float(_required(header, FITS_OFFSET_EXPOSURE_KEY))
    image_psf_fwhm = float(_required(header, FITS_IMAGE_PSF_FWHM_KEY))
    image_pixel_scale = float(_required(header, FITS_IMAGE_PIXEL_SCALE_KEY))

    expected = {
        FITS_PHOTOMETRY_BAND_KEY: (image_band, PHOTOMETRY_BAND),
        FITS_TARGET_LINE_KEY: (target_line, TARGET_EMISSION_LINE),
        FITS_SPECTRAL_UNITS_KEY: (spectral_units, SPECTRAL_UNITS),
        FITS_CENTER_FIBER_INDEX_KEY: (center_index, CENTER_FIBER_INDEX),
        FITS_CENTER_EXPOSURE_KEY: (center_exposure, CENTER_EXPOSURE_S),
        FITS_OFFSET_EXPOSURE_KEY: (offset_exposure, OFFSET_EXPOSURE_S),
        FITS_IMAGE_PSF_FWHM_KEY: (
            image_psf_fwhm,
            IMAGE_REFERENCE_PSF_FWHM_ARCSEC,
        ),
        FITS_IMAGE_PIXEL_SCALE_KEY: (
            image_pixel_scale,
            IMAGE_PIXEL_SCALE_ARCSEC,
        ),
    }
    mismatches = []
    for key, (actual, wanted) in expected.items():
        matches = (
            np.isclose(actual, wanted)
            if isinstance(wanted, float)
            else actual == wanted
        )
        if not matches:
            mismatches.append(f"{key}={actual!r} (expected {wanted!r})")
    if mismatches:
        raise ValueError("Simulator-v2 schema mismatch: " + "; ".join(mismatches))
    return {
        IMAGE_BAND_CODE_COLUMN: 0,
        TARGET_LINE_CODE_COLUMN: 0,
        SPECTRAL_UNITS_CODE_COLUMN: 0,
        CENTER_FIBER_INDEX_COLUMN: center_index,
        CENTER_EXPOSURE_COLUMN: center_exposure,
        OFFSET_EXPOSURE_COLUMN: offset_exposure,
        IMAGE_PSF_FWHM_COLUMN: image_psf_fwhm,
        IMAGE_PIXEL_SCALE_COLUMN: image_pixel_scale,
    }


def observation_metadata_arrays(
    headers: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, np.ndarray]:
    """Validate every FITS header and assemble strict LMDB metadata arrays."""

    count = len(headers)
    result = {
        RMAG_TRUE_COLUMN: np.empty(count, dtype=np.float32),
        HALPHA_FLUX_TRUE_COLUMN: np.empty(count, dtype=np.float32),
        OBSERVATION_MODEL_VERSION_COLUMN: np.full(
            count, OBSERVATION_MODEL_VERSION, dtype=np.int16
        ),
        FIBER_LAYOUT_COLUMN: np.full(count, FIBER_LAYOUT_CODE, dtype=np.int8),
        IMAGE_BAND_CODE_COLUMN: np.zeros(count, dtype=np.int8),
        TARGET_LINE_CODE_COLUMN: np.zeros(count, dtype=np.int8),
        SPECTRAL_UNITS_CODE_COLUMN: np.zeros(count, dtype=np.int8),
        CENTER_FIBER_INDEX_COLUMN: np.empty(count, dtype=np.int8),
        CENTER_EXPOSURE_COLUMN: np.empty(count, dtype=np.float32),
        OFFSET_EXPOSURE_COLUMN: np.empty(count, dtype=np.float32),
        IMAGE_PSF_FWHM_COLUMN: np.empty(count, dtype=np.float32),
        IMAGE_PIXEL_SCALE_COLUMN: np.empty(count, dtype=np.float32),
    }
    for index, header in enumerate(headers):
        _, result[RMAG_TRUE_COLUMN][index] = observation_metadata_from_header(header)
        result[HALPHA_FLUX_TRUE_COLUMN][index] = halpha_flux_from_header(header)
        fiber_layout_from_header(header)
        instrument = observation_instrument_metadata_from_header(header)
        for name, value in instrument.items():
            result[name][index] = value
    return result
