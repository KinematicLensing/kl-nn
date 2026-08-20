"""Generate simulator-v2 FITS files from the strict proposal table."""

from __future__ import annotations

import os
from argparse import ArgumentParser
from os.path import join
from pathlib import Path
import subprocess
import sys

import pandas as pd

try:
    from .observation_schema import (
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        validate_halpha_flux,
        validate_rmag_true,
    )
except ImportError:
    from observation_schema import (
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        validate_halpha_flux,
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
        )
    )
    return command


def main(argv=None):
    args = parse_args(argv)
    sample_file = join(SAMPLE_ROOT, args.s)
    output_dir = Path(FITS_ROOT) / args.d / f"part_{args.n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_csv(sample_file)
    if args.i < 0 or args.j <= args.i or args.i >= len(samples):
        raise ValueError("Requested row interval is empty or outside the sample table")
    for _, row in samples.iloc[args.i : min(args.j, len(samples))].iterrows():
        subprocess.run(
            build_generate_command(row, part=args.n, dataset=args.d),
            check=True,
        )


if __name__ == "__main__":
    main()
