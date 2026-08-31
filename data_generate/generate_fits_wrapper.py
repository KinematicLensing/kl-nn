"""Generate simulator-v3 FITS files from the strict proposal table."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
import subprocess
import sys

import pandas as pd

try:
    from .generation_integrity import (
        HEADER_METADATA_COLUMNS,
        simulator_v3_fits_completion_error,
        simulator_v3_output_path,
        simulator_v3_science_row_fingerprint,
    )
    from .observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        validate_halpha_flux,
        validate_central_halpha_snr,
        validate_image_snr,
        validate_rmag_true,
    )
except ImportError:
    from generation_integrity import (
        HEADER_METADATA_COLUMNS,
        simulator_v3_fits_completion_error,
        simulator_v3_output_path,
        simulator_v3_science_row_fingerprint,
    )
    from observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        validate_halpha_flux,
        validate_central_halpha_snr,
        validate_image_snr,
        validate_rmag_true,
    )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ROOT = "/ocean/projects/phy250048p/shared/samples"
FITS_ROOT = "/ocean/projects/phy250048p/shared/fits"
SIMULATION_PARAMETERS = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
REQUIRED_COLUMNS = (
    *SIMULATION_PARAMETERS,
    RMAG_TRUE_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_SNR_COLUMN,
    CENTRAL_HALPHA_SNR_COLUMN,
    FIBER_LAYOUT_COLUMN,
    OBSERVATION_MODEL_VERSION_COLUMN,
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", type=int, default=0, help="start row")
    parser.add_argument("-j", type=int, default=1, help="exclusive stop row")
    parser.add_argument("-n", type=int, default=1, help="one-based part id")
    parser.add_argument("-s", required=True, help="sample CSV filename")
    parser.add_argument("-d", required=True, help="dataset name")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip structurally valid outputs whose row metadata still match",
    )
    return parser.parse_args(argv)


def _sample_id(row: pd.Series) -> int:
    if "ID" in row.index:
        return int(row["ID"])
    unnamed = [name for name in row.index if str(name).startswith("Unnamed:")]
    if len(unnamed) == 1:
        return int(row[unnamed[0]])
    raise ValueError("Sample table must contain an explicit ID column")


def build_generate_command(row: pd.Series, *, part: int, dataset: str) -> list[str]:
    missing = [name for name in REQUIRED_COLUMNS if name not in row.index]
    if missing:
        raise ValueError(f"Sample row is missing required columns: {missing}")
    version = int(row[OBSERVATION_MODEL_VERSION_COLUMN])
    if version != OBSERVATION_MODEL_VERSION:
        raise ValueError(
            f"Expected observation_model_version={OBSERVATION_MODEL_VERSION}, got {version}"
        )
    layout = str(row[FIBER_LAYOUT_COLUMN])
    if layout != GALAXY_AXIS_FIBER_LAYOUT:
        raise ValueError(f"Expected fiber_layout={GALAXY_AXIS_FIBER_LAYOUT!r}")
    magnitude = validate_rmag_true(row[RMAG_TRUE_COLUMN])
    halpha_flux = validate_halpha_flux(row[HALPHA_FLUX_TRUE_COLUMN])
    image_snr = validate_image_snr(row[IMAGE_SNR_COLUMN])
    central_halpha_snr = validate_central_halpha_snr(
        row[CENTRAL_HALPHA_SNR_COLUMN]
    )
    command = [
        sys.executable,
        join(SCRIPT_DIR, "generate_fits.py"),
        f"-n={int(part)}",
        f"-d={dataset}",
        f"-ID={_sample_id(row)}",
    ]
    command.extend(f"-{name}={float(row[name])}" for name in SIMULATION_PARAMETERS)
    command.extend(
        (
            f"--rmag-true={magnitude}",
            f"--halpha-flux-true={halpha_flux}",
            f"--image-snr={image_snr}",
            f"--central-halpha-snr={central_halpha_snr}",
        )
    )
    return command


def main(argv=None):
    args = parse_args(argv)
    sample_file = join(SAMPLE_ROOT, args.s)
    output_dir = Path(FITS_ROOT) / args.d / f"part_{args.n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the exact binary64 value represented by each CSV token. The
    # generator receives these values through Python's round-trippable repr,
    # and the same values feed the persisted science-row fingerprint.
    samples = pd.read_csv(sample_file, float_precision="round_trip")
    if args.i < 0 or args.j <= args.i or args.i >= len(samples):
        raise ValueError("Requested row interval is empty or outside the sample table")
    generated = 0
    regenerated = 0
    skipped = 0
    for _, row in samples.iloc[args.i : min(args.j, len(samples))].iterrows():
        sample_id = _sample_id(row)
        output_path = simulator_v3_output_path(
            FITS_ROOT,
            args.d,
            args.n,
            sample_id,
        )
        if args.skip_existing:
            expected_metadata = {
                name: float(row[name]) for name in HEADER_METADATA_COLUMNS
            }
            expected_row_fingerprint = simulator_v3_science_row_fingerprint(
                sample_id,
                row,
            )
            completion_error = simulator_v3_fits_completion_error(
                output_path,
                expected_metadata=expected_metadata,
                expected_sample_id=sample_id,
                expected_row_fingerprint=expected_row_fingerprint,
            )
            if completion_error is None:
                skipped += 1
                continue
            if output_path.exists():
                regenerated += 1
                print(
                    f"Regenerating incomplete {output_path}: {completion_error}",
                    flush=True,
                )
        subprocess.run(
            build_generate_command(row, part=args.n, dataset=args.d),
            check=True,
        )
        generated += 1
        if args.skip_existing:
            completion_error = simulator_v3_fits_completion_error(
                output_path,
                expected_metadata=expected_metadata,
                expected_sample_id=sample_id,
                expected_row_fingerprint=expected_row_fingerprint,
            )
            if completion_error is not None:
                raise RuntimeError(
                    f"Generator returned successfully but {output_path} is "
                    f"incomplete: {completion_error}"
                )
    print(
        f"Part {args.n}: generated={generated}, regenerated={regenerated}, "
        f"skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
