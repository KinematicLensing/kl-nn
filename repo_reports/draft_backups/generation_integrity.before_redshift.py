"""Atomic writes and completion checks for simulator-v3 FITS files."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import uuid

import numpy as np
from astropy.io import fits

try:
    from .observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        fiber_layout_from_header,
        halpha_flux_conversion_from_header,
        observation_instrument_metadata_from_header,
        observation_metadata_from_header,
    )
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        fiber_layout_from_header,
        halpha_flux_conversion_from_header,
        observation_instrument_metadata_from_header,
        observation_metadata_from_header,
    )


FITS_BLOCK_SIZE = 2880
EXPECTED_FITS_SIZE_BYTES = 46_080
EXPECTED_OBSERVATION_COUNT = 6
EXPECTED_EXTENSION_SHAPES = ((61,),) * 5 + ((48, 48),)
HEADER_METADATA_COLUMNS = (
    RMAG_TRUE_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_SNR_COLUMN,
    CENTRAL_HALPHA_SNR_COLUMN,
)
SIMULATION_PARAMETER_COLUMNS = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
SCIENCE_ROW_FLOAT_COLUMNS = (*SIMULATION_PARAMETER_COLUMNS, *HEADER_METADATA_COLUMNS)
SCIENCE_ROW_FINGERPRINT_VERSION = 1
SCIENCE_ROW_FINGERPRINT_SCHEMA = (
    f"klnn-simulator-v3-science-row-v{SCIENCE_ROW_FINGERPRINT_VERSION}"
)
FITS_SCIENCE_ROW_ID_KEY = "ROWID"
FITS_SCIENCE_ROW_FINGERPRINT_KEY = "ROWFP"
FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY = "ROWFPVER"
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_integer(value: object, *, name: str) -> int:
    """Return an exact integer suitable for a stable science-row identity."""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(numeric)


def simulator_v3_science_row_fingerprint(
    sample_id: int,
    row: Mapping[str, object],
) -> str:
    """Hash the complete simulator-v3 science and observation row."""

    canonical_floats: list[list[str]] = []
    for name in SCIENCE_ROW_FLOAT_COLUMNS:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"Science row requires a finite numeric {name!r} value"
            ) from error
        if not np.isfinite(value):
            raise ValueError(f"Science row {name!r} must be finite, got {value!r}")
        canonical_floats.append([name, value.hex()])

    try:
        fiber_layout = str(row[FIBER_LAYOUT_COLUMN])
        observation_model_version = _canonical_integer(
            row[OBSERVATION_MODEL_VERSION_COLUMN],
            name=OBSERVATION_MODEL_VERSION_COLUMN,
        )
    except KeyError as error:
        raise ValueError(f"Science row is missing {error.args[0]!r}") from error

    payload = {
        "schema": SCIENCE_ROW_FINGERPRINT_SCHEMA,
        "ID": _canonical_integer(sample_id, name="ID"),
        "float64": canonical_floats,
        FIBER_LAYOUT_COLUMN: fiber_layout,
        OBSERVATION_MODEL_VERSION_COLUMN: observation_model_version,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode("ascii")).hexdigest()


def simulator_v3_output_path(
    fits_root: str | os.PathLike[str],
    dataset: str,
    part: int,
    sample_id: int,
) -> Path:
    """Return the canonical output path for one proposal-table row."""

    return Path(fits_root) / str(dataset) / f"part_{int(part)}" / f"gal_{int(sample_id)}.fits"


def quick_fits_completion_error(path: str | os.PathLike[str]) -> str | None:
    """Return why a final-path FITS is clearly incomplete, or ``None``."""

    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "missing"
    except OSError as error:
        return f"cannot stat file: {error}"
    if not path.is_file():
        return "not a regular file"
    if stat.st_size == 0:
        return "zero-byte file"
    if stat.st_size % FITS_BLOCK_SIZE:
        return f"size {stat.st_size} is not a complete FITS block"
    if stat.st_size != EXPECTED_FITS_SIZE_BYTES:
        return (
            f"size {stat.st_size}; expected simulator-v3 size "
            f"{EXPECTED_FITS_SIZE_BYTES}"
        )
    return None


def simulator_v3_fits_completion_error(
    path: str | os.PathLike[str],
    *,
    expected_metadata: Mapping[str, float] | None = None,
    expected_sample_id: int | None = None,
    expected_row_fingerprint: str | None = None,
) -> str | None:
    """Fully validate a simulator-v3 FITS file and optional expected row."""

    path = Path(path)
    error = quick_fits_completion_error(path)
    if error is not None:
        return error
    try:
        with fits.open(path, mode="readonly", memmap=False) as hdus:
            hdus.verify("exception")
            if len(hdus) != 1 + EXPECTED_OBSERVATION_COUNT:
                return (
                    f"found {len(hdus) - 1} observation HDUs; expected "
                    f"{EXPECTED_OBSERVATION_COUNT}"
                )
            header = hdus[0].header
            if int(header.get("OBSNUM", -1)) != EXPECTED_OBSERVATION_COUNT:
                return (
                    f"OBSNUM={header.get('OBSNUM')!r}; expected "
                    f"{EXPECTED_OBSERVATION_COUNT}"
                )
            shapes = tuple(tuple(hdu.shape) for hdu in hdus[1:])
            if shapes != EXPECTED_EXTENSION_SHAPES:
                return (
                    f"observation shapes are {shapes!r}; expected "
                    f"{EXPECTED_EXTENSION_SHAPES!r}"
                )
            _, rmag, image_snr, central_halpha_snr = (
                observation_metadata_from_header(header)
            )
            halpha_flux, _, _ = halpha_flux_conversion_from_header(header)
            fiber_layout_from_header(header)
            observation_instrument_metadata_from_header(header)
            if FITS_SCIENCE_ROW_ID_KEY not in header:
                return f"missing required {FITS_SCIENCE_ROW_ID_KEY} header"
            if FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY not in header:
                return (
                    "missing required "
                    f"{FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY} header"
                )
            if FITS_SCIENCE_ROW_FINGERPRINT_KEY not in header:
                return f"missing required {FITS_SCIENCE_ROW_FINGERPRINT_KEY} header"
            try:
                actual_sample_id = _canonical_integer(
                    header[FITS_SCIENCE_ROW_ID_KEY],
                    name=FITS_SCIENCE_ROW_ID_KEY,
                )
                fingerprint_version = _canonical_integer(
                    header[FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY],
                    name=FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY,
                )
            except ValueError as identity_error:
                return str(identity_error)
            if fingerprint_version != SCIENCE_ROW_FINGERPRINT_VERSION:
                return (
                    f"{FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY}="
                    f"{fingerprint_version}; expected "
                    f"{SCIENCE_ROW_FINGERPRINT_VERSION}"
                )
            actual_fingerprint = str(
                header[FITS_SCIENCE_ROW_FINGERPRINT_KEY]
            ).strip().lower()
            if _SHA256_HEX_PATTERN.fullmatch(actual_fingerprint) is None:
                return (
                    f"{FITS_SCIENCE_ROW_FINGERPRINT_KEY} is not a SHA-256 "
                    "hex digest"
                )
            if expected_sample_id is not None:
                expected_id = _canonical_integer(expected_sample_id, name="ID")
                if actual_sample_id != expected_id:
                    return (
                        f"{FITS_SCIENCE_ROW_ID_KEY}={actual_sample_id}; "
                        f"expected {expected_id}"
                    )
            if expected_row_fingerprint is not None:
                expected_fingerprint = str(expected_row_fingerprint).strip().lower()
                if _SHA256_HEX_PATTERN.fullmatch(expected_fingerprint) is None:
                    raise ValueError(
                        "expected_row_fingerprint must be a SHA-256 hex digest"
                    )
                if not hmac.compare_digest(actual_fingerprint, expected_fingerprint):
                    return (
                        f"{FITS_SCIENCE_ROW_FINGERPRINT_KEY} does not match "
                        "the requested science row"
                    )
            actual_metadata = {
                RMAG_TRUE_COLUMN: rmag,
                HALPHA_FLUX_TRUE_COLUMN: halpha_flux,
                IMAGE_SNR_COLUMN: image_snr,
                CENTRAL_HALPHA_SNR_COLUMN: central_halpha_snr,
            }
            if expected_metadata is not None:
                for name in HEADER_METADATA_COLUMNS:
                    expected = float(expected_metadata[name])
                    actual = actual_metadata[name]
                    if not np.isclose(actual, expected, rtol=1e-12, atol=0.0):
                        return f"{name}={actual!r}; expected {expected!r}"
    except Exception as error:  # Corrupt FITS files can raise several Astropy errors.
        return f"unreadable simulator-v3 FITS: {type(error).__name__}: {error}"
    return None


def atomic_write_simulator_v3_fits(
    output_path: str | os.PathLike[str],
    writer: Callable[[Path], None],
    *,
    expected_sample_id: int | None = None,
    expected_row_fingerprint: str | None = None,
) -> None:
    """Write and validate a temporary FITS, then atomically publish it."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp.fits"
    )
    try:
        writer(temporary_path)
        error = simulator_v3_fits_completion_error(
            temporary_path,
            expected_sample_id=expected_sample_id,
            expected_row_fingerprint=expected_row_fingerprint,
        )
        if error is not None:
            raise RuntimeError(f"Refusing to publish incomplete FITS: {error}")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
