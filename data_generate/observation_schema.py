"""Shared schema helpers for versioned simulated observations.

Version 1 is the historical simulator: its continuum has a fixed flux
normalization and the FITS files do not carry magnitude metadata.  Version 2
uses independently sampled true r-band magnitude and H-alpha flux values,
normalizes the continuum by the magnitude, and renders only H-alpha among the
configured emission lines.

The true magnitude and line flux are simulation metadata, not members of the
eight-parameter inference target. Keeping their FITS/LMDB names here prevents
the two schemas from silently drifting apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


LEGACY_OBSERVATION_MODEL_VERSION = 1
CURRENT_OBSERVATION_MODEL_VERSION = 2
SUPPORTED_OBSERVATION_MODEL_VERSIONS = (
    LEGACY_OBSERVATION_MODEL_VERSION,
    CURRENT_OBSERVATION_MODEL_VERSION,
)

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
NON_TARGET_EMISSION_LINES = ("O2", "Hb", "O3_1", "O3_2")
DEFAULT_HALPHA_FLUX_RANGE = (1.2e-16, 301.43e-16)
IMAGE_AXIS_FIBER_LAYOUT = "image_axis"
GALAXY_AXIS_FIBER_LAYOUT = "galaxy_axis"
FIBER_LAYOUT_CODES = {
    IMAGE_AXIS_FIBER_LAYOUT: 0,
    GALAXY_AXIS_FIBER_LAYOUT: 1,
}

# Store categorical schema values as compact numeric arrays in LMDB.  String
# arrays are not portable through every Pyxis/Torch reader used by this repo.
IMAGE_BAND_CODE_COLUMN = "image_band_code"
TARGET_LINE_CODE_COLUMN = "target_line_code"
SPECTRAL_UNITS_CODE_COLUMN = "spectral_units_code"
CENTER_FIBER_INDEX_COLUMN = "center_fiber_index"
CENTER_EXPOSURE_COLUMN = "center_exposure_s"
OFFSET_EXPOSURE_COLUMN = "offset_exposure_s"
IMAGE_PSF_FWHM_COLUMN = "image_reference_psf_fwhm_arcsec"
IMAGE_PIXEL_SCALE_COLUMN = "image_pixel_scale_arcsec"
IMAGE_BAND_CODES = {PHOTOMETRY_BAND: 0}
TARGET_LINE_CODES = {TARGET_EMISSION_LINE: 0}
SPECTRAL_UNITS_CODES = {SPECTRAL_UNITS: 0}


def resolve_observation_model(
    version: int | None,
    rmag_true: float | None,
) -> tuple[int, float | None]:
    """Resolve and validate an observation-model request.

    Omitting both values preserves the historical version-1 CLI.  Supplying a
    magnitude without an explicit version opts into version 2 for convenient
    use by sample-file wrappers.
    """

    if version is None:
        version = (
            CURRENT_OBSERVATION_MODEL_VERSION
            if rmag_true is not None
            else LEGACY_OBSERVATION_MODEL_VERSION
        )
    version = int(version)
    if version not in SUPPORTED_OBSERVATION_MODEL_VERSIONS:
        raise ValueError(
            f"Unsupported observation model version {version}; expected one of "
            f"{SUPPORTED_OBSERVATION_MODEL_VERSIONS}"
        )

    if version == CURRENT_OBSERVATION_MODEL_VERSION:
        if rmag_true is None or not np.isfinite(rmag_true):
            raise ValueError("Observation model v2 requires a finite rmag_true")
        rmag_true = float(rmag_true)
    elif rmag_true is not None:
        raise ValueError("rmag_true is only valid for observation model v2")

    return version, rmag_true


def resolve_halpha_flux(
    version: int,
    halpha_flux_true: float | None,
    *,
    flux_range: tuple[float, float] | None = None,
) -> float | None:
    """Validate the H-alpha line flux used by the observation model.

    KL-tools interprets ``em_Ha_flux`` as the integrated observed-frame line
    flux in ``erg s^-1 cm^-2``. The simulator sampler uses the requested
    DESI-KL fiducial-grid range. Optional bounds let callers validate a
    particular archived proposal; FITS/schema readers otherwise require only
    a finite positive integrated flux.
    """

    version = int(version)
    if version not in SUPPORTED_OBSERVATION_MODEL_VERSIONS:
        raise ValueError(f"Unsupported observation model version {version}")
    if flux_range is not None:
        lower, upper = (float(value) for value in flux_range)
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower <= 0.0
            or lower >= upper
        ):
            raise ValueError(
                "H-alpha flux bounds must be finite, positive, and increasing"
            )
    if version == CURRENT_OBSERVATION_MODEL_VERSION:
        if (
            halpha_flux_true is None
            or not np.isfinite(halpha_flux_true)
            or halpha_flux_true <= 0.0
        ):
            raise ValueError(
                "Observation model v2 requires a finite positive "
                "halpha_flux_true"
            )
        halpha_flux_true = float(halpha_flux_true)
        if flux_range is not None:
            tolerance = 8.0 * np.finfo(float).eps * max(abs(lower), abs(upper))
            if not lower - tolerance <= halpha_flux_true <= upper + tolerance:
                raise ValueError(
                    f"halpha_flux_true={halpha_flux_true!r} is outside "
                    f"configured range [{lower!r}, {upper!r}]"
                )
        return halpha_flux_true
    if halpha_flux_true is not None:
        raise ValueError("halpha_flux_true is only valid for observation model v2")
    return None


def configure_sed(
    base_sed: Mapping[str, Any],
    *,
    version: int,
    rmag_true: float | None,
    halpha_flux_true: float | None = None,
    r_bandpass: str,
) -> dict[str, Any]:
    """Return the SED configuration for a validated observation model."""

    version, rmag_true = resolve_observation_model(version, rmag_true)
    halpha_flux_true = resolve_halpha_flux(version, halpha_flux_true)
    sed = dict(base_sed)
    if version == LEGACY_OBSERVATION_MODEL_VERSION:
        return sed

    sed.update(
        {
            "cont_norm_method": "mag",
            "obs_norm_band": str(r_bandpass),
            "obs_norm_mag": rmag_true,
            "em_Ha_flux": halpha_flux_true,
        }
    )
    # Avoid leaving contradictory, inactive normalization values in archived
    # configurations and ensure H-alpha is the only non-zero emission line.
    sed.pop("obs_cont_norm_wave", None)
    sed.pop("obs_cont_norm_flam", None)
    for line_name in NON_TARGET_EMISSION_LINES:
        sed[f"em_{line_name}_flux"] = 0.0
    return sed


def validate_fiber_layout(fiber_layout: str | None) -> str:
    """Validate a layout, retaining image-axis as the no-flag legacy default."""

    if fiber_layout is None:
        return IMAGE_AXIS_FIBER_LAYOUT
    fiber_layout = str(fiber_layout)
    if fiber_layout not in FIBER_LAYOUT_CODES:
        raise ValueError(
            f"Unsupported fiber layout {fiber_layout!r}; expected one of "
            f"{tuple(FIBER_LAYOUT_CODES)}"
        )
    return fiber_layout


def compute_fiber_offsets(
    *,
    fiber_layout: str,
    fiber_offset: float,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
    fiber_permutation: tuple[int, ...] | list[int] = (0, 1, 2, 3, 4),
) -> np.ndarray:
    """Return ordered five-fiber positions for an image- or galaxy-axis cross.

    The galaxy-axis branch intentionally reproduces the historical simulator's
    observed-axis SVD convention.  The permutation is applied to positions
    before observation configurations are built, keeping every spectrum paired
    with the position stored in its FITS extension.
    """

    fiber_layout = validate_fiber_layout(fiber_layout)
    permutation = np.asarray(fiber_permutation, dtype=int)
    if permutation.shape != (5,) or sorted(permutation.tolist()) != list(range(5)):
        raise ValueError('fiber_permutation must be a permutation of 0,1,2,3,4')
    if not 0.0 <= sini <= 1.0:
        raise ValueError(f'sini must be in [0, 1], got {sini}')

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
    if fiber_layout == GALAXY_AXIS_FIBER_LAYOUT:
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
        offsets = offsets @ observed_axes

    return offsets[permutation]


def fiber_layout_from_header(
    header: Mapping[str, Any],
    *,
    version: int | None = None,
) -> tuple[str, int]:
    """Return the persisted layout name and compact LMDB code."""

    if version is None:
        version = int(
            header.get(
                FITS_OBSERVATION_MODEL_VERSION_KEY,
                LEGACY_OBSERVATION_MODEL_VERSION,
            )
        )
    if (
        int(version) == CURRENT_OBSERVATION_MODEL_VERSION
        and FITS_FIBER_LAYOUT_KEY not in header
    ):
        raise ValueError(
            "Observation model v2 FITS header is missing required "
            f"{FITS_FIBER_LAYOUT_KEY} metadata"
        )
    fiber_layout = validate_fiber_layout(header.get(FITS_FIBER_LAYOUT_KEY))
    return fiber_layout, FIBER_LAYOUT_CODES[fiber_layout]


def observation_metadata_from_header(header: Mapping[str, Any]) -> tuple[int, float]:
    """Read versioned observation metadata from a FITS-like header.

    Historical FITS files contain no v2 population keywords and are interpreted
    as v1. Their unavailable true magnitude and H-alpha flux are represented by
    NaN in the LMDB, which keeps a uniform schema without inventing values.
    """

    version = int(
        header.get(
            FITS_OBSERVATION_MODEL_VERSION_KEY,
            LEGACY_OBSERVATION_MODEL_VERSION,
        )
    )
    rmag_value = header.get(FITS_RMAG_TRUE_KEY)
    rmag_true = None if rmag_value is None else float(rmag_value)
    version, rmag_true = resolve_observation_model(version, rmag_true)
    # Validate all population metadata while retaining this helper's
    # historical two-value return signature.
    halpha_flux_from_header(header, version=version)
    return version, np.nan if rmag_true is None else rmag_true


def halpha_flux_from_header(
    header: Mapping[str, Any],
    *,
    version: int | None = None,
) -> float:
    """Return archived v2 H-alpha flux, or NaN for historical v1 FITS."""

    if version is None:
        version = int(
            header.get(
                FITS_OBSERVATION_MODEL_VERSION_KEY,
                LEGACY_OBSERVATION_MODEL_VERSION,
            )
        )
    value = header.get(FITS_HALPHA_FLUX_TRUE_KEY)
    flux = None if value is None else float(value)
    flux = resolve_halpha_flux(version, flux)
    return np.nan if flux is None else flux


def observation_instrument_metadata_from_header(
    header: Mapping[str, Any],
    *,
    version: int,
) -> dict[str, int | float]:
    """Decode the immutable one-image/one-line schema from a FITS header.

    Old version-1 FITS files predate these keywords and therefore receive
    explicit unavailable sentinels.  Version 2 has no fallback: omitting or
    changing any field is an archive error rather than a silent assumption.
    """

    version = int(version)
    if version not in SUPPORTED_OBSERVATION_MODEL_VERSIONS:
        raise ValueError(f"Unsupported observation model version {version}")

    keys = (
        FITS_PHOTOMETRY_BAND_KEY,
        FITS_TARGET_LINE_KEY,
        FITS_SPECTRAL_UNITS_KEY,
        FITS_CENTER_FIBER_INDEX_KEY,
        FITS_CENTER_EXPOSURE_KEY,
        FITS_OFFSET_EXPOSURE_KEY,
        FITS_IMAGE_PSF_FWHM_KEY,
        FITS_IMAGE_PIXEL_SCALE_KEY,
    )
    missing = [key for key in keys if key not in header]
    if version == CURRENT_OBSERVATION_MODEL_VERSION and missing:
        raise ValueError(
            "Observation model v2 FITS header is missing required schema "
            f"metadata: {missing}"
        )
    if missing:
        return {
            IMAGE_BAND_CODE_COLUMN: -1,
            TARGET_LINE_CODE_COLUMN: -1,
            SPECTRAL_UNITS_CODE_COLUMN: -1,
            CENTER_FIBER_INDEX_COLUMN: -1,
            CENTER_EXPOSURE_COLUMN: np.nan,
            OFFSET_EXPOSURE_COLUMN: np.nan,
            IMAGE_PSF_FWHM_COLUMN: np.nan,
            IMAGE_PIXEL_SCALE_COLUMN: np.nan,
        }

    image_band = str(header[FITS_PHOTOMETRY_BAND_KEY]).strip()
    target_line = str(header[FITS_TARGET_LINE_KEY]).strip()
    spectral_units = str(header[FITS_SPECTRAL_UNITS_KEY]).strip()
    center_index = int(header[FITS_CENTER_FIBER_INDEX_KEY])
    center_exposure = float(header[FITS_CENTER_EXPOSURE_KEY])
    offset_exposure = float(header[FITS_OFFSET_EXPOSURE_KEY])
    image_psf_fwhm = float(header[FITS_IMAGE_PSF_FWHM_KEY])
    image_pixel_scale = float(header[FITS_IMAGE_PIXEL_SCALE_KEY])

    expected = {
        FITS_PHOTOMETRY_BAND_KEY: (image_band, PHOTOMETRY_BAND),
        FITS_TARGET_LINE_KEY: (target_line, TARGET_EMISSION_LINE),
        FITS_SPECTRAL_UNITS_KEY: (spectral_units, SPECTRAL_UNITS),
    }
    mismatches = [
        f"{key}={actual!r} (expected {wanted!r})"
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if not 0 <= center_index < 5:
        mismatches.append(
            f"{FITS_CENTER_FIBER_INDEX_KEY}={center_index!r} "
            "(expected an index in [0, 5))"
        )
    if (
        not np.isfinite(center_exposure)
        or not np.isclose(center_exposure, CENTER_EXPOSURE_S)
    ):
        mismatches.append(
            f"{FITS_CENTER_EXPOSURE_KEY}={center_exposure!r} "
            f"(expected {CENTER_EXPOSURE_S!r})"
        )
    if (
        not np.isfinite(offset_exposure)
        or not np.isclose(offset_exposure, OFFSET_EXPOSURE_S)
    ):
        mismatches.append(
            f"{FITS_OFFSET_EXPOSURE_KEY}={offset_exposure!r} "
            f"(expected {OFFSET_EXPOSURE_S!r})"
        )
    if mismatches:
        raise ValueError(
            "Observation schema metadata does not match the archived "
            "one-r-image/one-Ha-line design: " + "; ".join(mismatches)
        )
    if not np.isfinite(image_psf_fwhm) or image_psf_fwhm <= 0:
        raise ValueError(
            f"{FITS_IMAGE_PSF_FWHM_KEY} must be positive and finite"
        )
    if not np.isfinite(image_pixel_scale) or image_pixel_scale <= 0:
        raise ValueError(
            f"{FITS_IMAGE_PIXEL_SCALE_KEY} must be positive and finite"
        )

    return {
        IMAGE_BAND_CODE_COLUMN: IMAGE_BAND_CODES[image_band],
        TARGET_LINE_CODE_COLUMN: TARGET_LINE_CODES[target_line],
        SPECTRAL_UNITS_CODE_COLUMN: SPECTRAL_UNITS_CODES[spectral_units],
        CENTER_FIBER_INDEX_COLUMN: center_index,
        CENTER_EXPOSURE_COLUMN: center_exposure,
        OFFSET_EXPOSURE_COLUMN: offset_exposure,
        IMAGE_PSF_FWHM_COLUMN: image_psf_fwhm,
        IMAGE_PIXEL_SCALE_COLUMN: image_pixel_scale,
    }


def observation_metadata_arrays(
    headers: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, np.ndarray]:
    """Assemble the optional FITS metadata into uniform LMDB-ready arrays."""

    rmag_true = np.full(len(headers), np.nan, dtype=np.float32)
    halpha_flux_true = np.full(len(headers), np.nan, dtype=np.float32)
    versions = np.empty(len(headers), dtype=np.int16)
    fiber_layouts = np.empty(len(headers), dtype=np.int8)
    image_band_codes = np.empty(len(headers), dtype=np.int8)
    target_line_codes = np.empty(len(headers), dtype=np.int8)
    spectral_units_codes = np.empty(len(headers), dtype=np.int8)
    center_fiber_indices = np.empty(len(headers), dtype=np.int8)
    center_exposures = np.empty(len(headers), dtype=np.float32)
    offset_exposures = np.empty(len(headers), dtype=np.float32)
    image_psf_fwhm = np.empty(len(headers), dtype=np.float32)
    image_pixel_scale = np.empty(len(headers), dtype=np.float32)
    for index, header in enumerate(headers):
        version, magnitude = observation_metadata_from_header(header)
        halpha_flux = halpha_flux_from_header(header, version=version)
        _, layout_code = fiber_layout_from_header(header, version=version)
        instrument = observation_instrument_metadata_from_header(
            header,
            version=version,
        )
        versions[index] = version
        rmag_true[index] = magnitude
        halpha_flux_true[index] = halpha_flux
        fiber_layouts[index] = layout_code
        image_band_codes[index] = instrument[IMAGE_BAND_CODE_COLUMN]
        target_line_codes[index] = instrument[TARGET_LINE_CODE_COLUMN]
        spectral_units_codes[index] = instrument[SPECTRAL_UNITS_CODE_COLUMN]
        center_fiber_indices[index] = instrument[CENTER_FIBER_INDEX_COLUMN]
        center_exposures[index] = instrument[CENTER_EXPOSURE_COLUMN]
        offset_exposures[index] = instrument[OFFSET_EXPOSURE_COLUMN]
        image_psf_fwhm[index] = instrument[IMAGE_PSF_FWHM_COLUMN]
        image_pixel_scale[index] = instrument[IMAGE_PIXEL_SCALE_COLUMN]
    return {
        RMAG_TRUE_COLUMN: rmag_true,
        HALPHA_FLUX_TRUE_COLUMN: halpha_flux_true,
        OBSERVATION_MODEL_VERSION_COLUMN: versions,
        FIBER_LAYOUT_COLUMN: fiber_layouts,
        IMAGE_BAND_CODE_COLUMN: image_band_codes,
        TARGET_LINE_CODE_COLUMN: target_line_codes,
        SPECTRAL_UNITS_CODE_COLUMN: spectral_units_codes,
        CENTER_FIBER_INDEX_COLUMN: center_fiber_indices,
        CENTER_EXPOSURE_COLUMN: center_exposures,
        OFFSET_EXPOSURE_COLUMN: offset_exposures,
        IMAGE_PSF_FWHM_COLUMN: image_psf_fwhm,
        IMAGE_PIXEL_SCALE_COLUMN: image_pixel_scale,
    }
