"""Strict schema for the KL-NN simulator-v3 observation model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


OBSERVATION_MODEL_VERSION = 3
RMAG_TRUE_COLUMN = "rmag_true"
HALPHA_FLUX_TRUE_COLUMN = "halpha_flux_true"
IMAGE_SNR_COLUMN = "image_snr"
CENTRAL_HALPHA_SNR_COLUMN = "central_halpha_snr"
OBSERVATION_MODEL_VERSION_COLUMN = "observation_model_version"
FIBER_LAYOUT_COLUMN = "fiber_layout"

FITS_OBSERVATION_MODEL_VERSION_KEY = "OBSMODV"
FITS_RMAG_TRUE_KEY = "RMAGTRUE"
FITS_HALPHA_FLUX_TRUE_KEY = "HAFLUX"
FITS_IMAGE_SNR_KEY = "IMGSNR"
FITS_CENTER_HALPHA_SNR_KEY = "CENHASNR"
FITS_HALPHA_FLUX_UNITS_KEY = "HAFLXUNT"
FITS_HALPHA_FLUX_SEMANTICS_KEY = "HAFSEM"
FITS_HALPHA_FLUX_TRANSFORM_KEY = "HAFTRAN"
FITS_HALPHA_FLUX_API_VERSION_KEY = "HAFAPI"
FITS_HALPHA_TOTAL_FLUX_KEY = "HATOTFLX"
FITS_CENTER_HALPHA_APERTURE_KEY = "HACENAP"
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
DEFAULT_HALPHA_LOG10_FLUX_RANGE = (-17.0, -14.0)
DEFAULT_HALPHA_FLUX_RANGE = tuple(
    10.0**bound for bound in DEFAULT_HALPHA_LOG10_FLUX_RANGE
)
DEFAULT_IMAGE_SNR_RANGE = (10.0, 1000.0)
DEFAULT_CENTRAL_HALPHA_SNR_RANGE = (1.0, 150.0)
HALPHA_FLUX_UNITS = "erg s^-1 cm^-2"
HALPHA_FLUX_SEMANTICS = "central_fiber_integrated_after_seeing_before_instrument"
HALPHA_FLUX_TRANSFORM = "log10"
HALPHA_FLUX_API_VERSION = 1
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
HALPHA_FLUX_UNITS_CODE_COLUMN = "halpha_flux_units_code"
HALPHA_FLUX_SEMANTICS_CODE_COLUMN = "halpha_flux_semantics_code"
HALPHA_FLUX_TRANSFORM_CODE_COLUMN = "halpha_flux_transform_code"
HALPHA_FLUX_API_VERSION_COLUMN = "halpha_flux_api_version"
HALPHA_TOTAL_FLUX_COLUMN = "halpha_total_flux_derived"
CENTRAL_HALPHA_APERTURE_FRACTION_COLUMN = (
    "central_halpha_aperture_fraction"
)


def validate_rmag_true(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("rmag_true must be finite")
    return value


def validate_halpha_flux(value: float) -> float:
    """Validate an integrated observer-frame H-alpha flux in cgs units."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("halpha_flux_true must be finite and positive")
    return value


def validate_image_snr(value: float) -> float:
    """Validate a positive nominal image matched-filter S/N."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("image_snr must be finite and positive")
    return value


def validate_central_halpha_snr(value: float) -> float:
    """Validate a positive nominal central-fiber H-alpha matched-filter S/N."""

    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("central_halpha_snr must be finite and positive")
    return value


def validate_central_halpha_flux_conversion(
    *,
    central_flux: float,
    total_flux: float,
    aperture_fraction: float,
) -> tuple[float, float, float]:
    """Validate the auditable total-to-central H-alpha flux conversion."""

    central = validate_halpha_flux(central_flux)
    total = validate_halpha_flux(total_flux)
    fraction = float(aperture_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("central H-alpha aperture fraction must lie in (0, 1]")
    if not np.isclose(total * fraction, central, rtol=5e-6, atol=0.0):
        raise ValueError(
            "derived total H-alpha flux and aperture fraction do not reproduce "
            "halpha_flux_true"
        )
    return central, total, fraction


def configure_sed(
    base_sed: Mapping[str, Any],
    *,
    rmag_true: float,
    halpha_flux_true: float,
    r_bandpass: str,
    center_fiber_obsindex: int,
) -> dict[str, Any]:
    """Configure continuum and explicitly request central-fiber H-alpha flux."""

    magnitude = validate_rmag_true(rmag_true)
    halpha_flux = validate_halpha_flux(halpha_flux_true)
    center_fiber_obsindex = int(center_fiber_obsindex)
    if center_fiber_obsindex < 0:
        raise ValueError("center_fiber_obsindex must be non-negative")
    sed = dict(base_sed)
    sed.update(
        {
            "cont_norm_method": "mag",
            "obs_norm_band": str(r_bandpass),
            "obs_norm_mag": magnitude,
            "em_Ha_flux": halpha_flux,
            "em_Ha_flux_semantics": "central_fiber",
            "em_Ha_flux_reference_obsindex": center_fiber_obsindex,
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


MAJOR_FIBER_INDICES = (0, 1)
MINOR_FIBER_INDICES = (3, 4)
DEFAULT_FIBER_OFFSET_ARCSEC = 1.5


def observed_galaxy_axes(
    *,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
) -> np.ndarray:
    """Return a right-handed 2x2 matrix of observed photometric principal axes.

    The rows are unit vectors used as ``offsets @ axes``. Row 0 is the first
    left-singular-vector gauge after the major-axis vector anchor: if that
    singular vector is antiparallel to the image of the intrinsic major axis,
    the whole SVD factor is flipped. Row 1 is the remaining axis, with its
    sign chosen so that the pair is right-handed. The major-axis vector
    anchor does not remove that discrete SVD reflection, so leaving the
    second-row sign free lets the minor-axis fibers swap under an arbitrarily
    small shear.
    """

    if not 0.0 <= sini <= 1.0:
        raise ValueError(f"sini must be in [0, 1], got {sini}")
    cosi = np.sqrt(max(0.0, 1.0 - sini**2))
    shear = np.asarray([[1.0 + g1, g2], [g2, 1.0 - g1]], dtype=float)
    rotation = np.asarray(
        [
            [np.cos(theta_int), -np.sin(theta_int)],
            [np.sin(theta_int), np.cos(theta_int)],
        ],
        dtype=float,
    )
    projection = np.asarray([[1.0, 0.0], [0.0, cosi]], dtype=float)
    transform = shear @ (rotation @ projection)
    observed_axes, _, _ = np.linalg.svd(transform)
    reference_axis = transform @ np.asarray([1.0, 0.0], dtype=float)
    # Align the first left singular vector with the transformed intrinsic
    # major axis. The whole-matrix flip preserves det(U), so it does not
    # remove SVD's discrete reflection.
    if np.dot(observed_axes[:, 0], reference_axis) < 0.0:
        observed_axes *= -1.0
    # Fiber sky positions are ``offsets @ U``, so the second ROW is the
    # minor-axis direction. Flip only that row to force a right-handed
    # frame; the major-axis fibers stay where the vector anchor put them.
    if np.linalg.det(observed_axes) < 0.0:
        observed_axes[1, :] *= -1.0
    return observed_axes


def legacy_major_anchored_fiber_offsets(
    *,
    fiber_offset: float,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
) -> np.ndarray:
    """Reproduce the pre-handedness SVD gauge used by existing simulator-v3 data.

    The first left singular vector is aligned with the transformed intrinsic
    major axis, but both singular vectors are flipped together. That preserves
    a possible reflection, so the stored minor-axis fibers can be exchanged
    relative to :func:`compute_fiber_offsets`.
    """

    if not np.isfinite(fiber_offset) or fiber_offset <= 0.0:
        raise ValueError("fiber_offset must be finite and positive")
    if not 0.0 <= sini <= 1.0:
        raise ValueError(f"sini must be in [0, 1], got {sini}")
    cosi = np.sqrt(max(0.0, 1.0 - sini**2))
    shear = np.asarray([[1.0 + g1, g2], [g2, 1.0 - g1]], dtype=float)
    rotation = np.asarray(
        [
            [np.cos(theta_int), -np.sin(theta_int)],
            [np.sin(theta_int), np.cos(theta_int)],
        ],
        dtype=float,
    )
    projection = np.asarray([[1.0, 0.0], [0.0, cosi]], dtype=float)
    transform = shear @ (rotation @ projection)
    observed_axes, _, _ = np.linalg.svd(transform)
    reference_axis = transform @ np.asarray([1.0, 0.0], dtype=float)
    if np.dot(observed_axes[:, 0], reference_axis) < 0.0:
        observed_axes *= -1.0
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
    return offsets @ observed_axes


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
    return offsets @ observed_galaxy_axes(
        g1=g1,
        g2=g2,
        theta_int=theta_int,
        sini=sini,
    )


def classify_fiber_offset_sign(
    stored_offsets: np.ndarray,
    *,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
    fiber_offset: float = DEFAULT_FIBER_OFFSET_ARCSEC,
    atol: float = 1.0e-6,
) -> str:
    """Classify stored fiber centers against the right-handed axis convention.

    Returns ``match`` if the stored centers already follow
    :func:`compute_fiber_offsets`, ``swap_minor`` if only the two minor-axis
    fibers are exchanged, and ``mismatch`` otherwise.
    """

    stored = np.asarray(stored_offsets, dtype=float)
    expected = compute_fiber_offsets(
        fiber_offset=fiber_offset,
        g1=g1,
        g2=g2,
        theta_int=theta_int,
        sini=sini,
    )
    if stored.shape != expected.shape:
        raise ValueError(
            "stored fiber offsets must have shape "
            f"{expected.shape}; got {stored.shape}"
        )
    if np.allclose(stored, expected, atol=atol, rtol=0.0):
        return "match"
    swapped = expected.copy()
    swapped[MINOR_FIBER_INDICES[0]] = expected[MINOR_FIBER_INDICES[1]]
    swapped[MINOR_FIBER_INDICES[1]] = expected[MINOR_FIBER_INDICES[0]]
    if np.allclose(stored, swapped, atol=atol, rtol=0.0):
        return "swap_minor"
    return "mismatch"


def swap_minor_axis_fibers(spectra: np.ndarray, fiber_positions: np.ndarray):
    """Exchange the two minor-axis fibers in spectra and sky positions."""

    plus, minus = MINOR_FIBER_INDICES
    spectra = np.asarray(spectra)
    fiber_positions = np.asarray(fiber_positions)
    if spectra.shape[-2] <= minus:
        raise ValueError("spectra must include both minor-axis fibers")
    if fiber_positions.shape[-2] <= minus:
        raise ValueError("fiber positions must include both minor-axis fibers")
    swapped_spectra = np.array(spectra, copy=True)
    swapped_positions = np.array(fiber_positions, copy=True)
    swapped_spectra[..., plus, :] = spectra[..., minus, :]
    swapped_spectra[..., minus, :] = spectra[..., plus, :]
    swapped_positions[..., plus, :] = fiber_positions[..., minus, :]
    swapped_positions[..., minus, :] = fiber_positions[..., plus, :]
    return swapped_spectra, swapped_positions


def _required(header: Mapping[str, Any], key: str) -> Any:
    if key not in header:
        raise ValueError(f"Simulator-v3 FITS header is missing required {key}")
    return header[key]


def observation_metadata_from_header(
    header: Mapping[str, Any],
) -> tuple[int, float, float, float]:
    version = int(_required(header, FITS_OBSERVATION_MODEL_VERSION_KEY))
    if version != OBSERVATION_MODEL_VERSION:
        raise ValueError(
            f"Expected observation model {OBSERVATION_MODEL_VERSION}, got {version}"
        )
    return (
        version,
        validate_rmag_true(_required(header, FITS_RMAG_TRUE_KEY)),
        validate_image_snr(_required(header, FITS_IMAGE_SNR_KEY)),
        validate_central_halpha_snr(
            _required(header, FITS_CENTER_HALPHA_SNR_KEY)
        ),
    )


def halpha_flux_from_header(header: Mapping[str, Any]) -> float:
    return validate_halpha_flux(_required(header, FITS_HALPHA_FLUX_TRUE_KEY))


def halpha_flux_conversion_from_header(
    header: Mapping[str, Any],
) -> tuple[float, float, float]:
    return validate_central_halpha_flux_conversion(
        central_flux=_required(header, FITS_HALPHA_FLUX_TRUE_KEY),
        total_flux=_required(header, FITS_HALPHA_TOTAL_FLUX_KEY),
        aperture_fraction=_required(header, FITS_CENTER_HALPHA_APERTURE_KEY),
    )


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
    halpha_units = str(_required(header, FITS_HALPHA_FLUX_UNITS_KEY)).strip()
    halpha_semantics = str(
        _required(header, FITS_HALPHA_FLUX_SEMANTICS_KEY)
    ).strip()
    halpha_transform = str(
        _required(header, FITS_HALPHA_FLUX_TRANSFORM_KEY)
    ).strip()
    halpha_api_version = int(
        _required(header, FITS_HALPHA_FLUX_API_VERSION_KEY)
    )

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
        FITS_HALPHA_FLUX_UNITS_KEY: (halpha_units, HALPHA_FLUX_UNITS),
        FITS_HALPHA_FLUX_SEMANTICS_KEY: (
            halpha_semantics,
            HALPHA_FLUX_SEMANTICS,
        ),
        FITS_HALPHA_FLUX_TRANSFORM_KEY: (
            halpha_transform,
            HALPHA_FLUX_TRANSFORM,
        ),
        FITS_HALPHA_FLUX_API_VERSION_KEY: (
            halpha_api_version,
            HALPHA_FLUX_API_VERSION,
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
        raise ValueError("Simulator-v3 schema mismatch: " + "; ".join(mismatches))
    return {
        IMAGE_BAND_CODE_COLUMN: 0,
        TARGET_LINE_CODE_COLUMN: 0,
        SPECTRAL_UNITS_CODE_COLUMN: 0,
        CENTER_FIBER_INDEX_COLUMN: center_index,
        CENTER_EXPOSURE_COLUMN: center_exposure,
        OFFSET_EXPOSURE_COLUMN: offset_exposure,
        IMAGE_PSF_FWHM_COLUMN: image_psf_fwhm,
        IMAGE_PIXEL_SCALE_COLUMN: image_pixel_scale,
        HALPHA_FLUX_UNITS_CODE_COLUMN: 0,
        HALPHA_FLUX_SEMANTICS_CODE_COLUMN: 1,
        HALPHA_FLUX_TRANSFORM_CODE_COLUMN: 1,
        HALPHA_FLUX_API_VERSION_COLUMN: halpha_api_version,
    }


def observation_metadata_arrays(
    headers: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, np.ndarray]:
    """Validate every FITS header and assemble strict LMDB metadata arrays."""

    count = len(headers)
    result = {
        RMAG_TRUE_COLUMN: np.empty(count, dtype=np.float32),
        HALPHA_FLUX_TRUE_COLUMN: np.empty(count, dtype=np.float32),
        IMAGE_SNR_COLUMN: np.empty(count, dtype=np.float32),
        CENTRAL_HALPHA_SNR_COLUMN: np.empty(count, dtype=np.float32),
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
        HALPHA_FLUX_UNITS_CODE_COLUMN: np.zeros(count, dtype=np.int8),
        HALPHA_FLUX_SEMANTICS_CODE_COLUMN: np.ones(count, dtype=np.int8),
        HALPHA_FLUX_TRANSFORM_CODE_COLUMN: np.ones(count, dtype=np.int8),
        HALPHA_FLUX_API_VERSION_COLUMN: np.full(
            count, HALPHA_FLUX_API_VERSION, dtype=np.int8
        ),
        HALPHA_TOTAL_FLUX_COLUMN: np.empty(count, dtype=np.float32),
        CENTRAL_HALPHA_APERTURE_FRACTION_COLUMN: np.empty(
            count, dtype=np.float32
        ),
    }
    for index, header in enumerate(headers):
        (
            _,
            result[RMAG_TRUE_COLUMN][index],
            result[IMAGE_SNR_COLUMN][index],
            result[CENTRAL_HALPHA_SNR_COLUMN][index],
        ) = observation_metadata_from_header(header)
        central, total, fraction = halpha_flux_conversion_from_header(header)
        result[HALPHA_FLUX_TRUE_COLUMN][index] = central
        result[HALPHA_TOTAL_FLUX_COLUMN][index] = total
        result[CENTRAL_HALPHA_APERTURE_FRACTION_COLUMN][index] = fraction
        fiber_layout_from_header(header)
        instrument = observation_instrument_metadata_from_header(header)
        for name, value in instrument.items():
            result[name][index] = value
    return result
