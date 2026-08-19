"""Generate a range of FITS simulations from a versioned sample CSV."""

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
        CURRENT_OBSERVATION_MODEL_VERSION,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        resolve_halpha_flux,
        resolve_observation_model,
        validate_fiber_layout,
    )
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import (
        CURRENT_OBSERVATION_MODEL_VERSION,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
        resolve_halpha_flux,
        resolve_observation_model,
        validate_fiber_layout,
    )


SCR_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_ROOT = '/ocean/projects/phy250048p/shared/samples'
FITS_ROOT = '/ocean/projects/phy250048p/shared/fits'
SIMULATION_PARAMETERS = (
    'g1',
    'g2',
    'theta_int',
    'sini',
    'v0',
    'vcirc',
    'rscale',
    'hlr',
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', type=int, default=0, help='start index')
    parser.add_argument('-j', type=int, default=1, help='stop index')
    parser.add_argument('-n', type=int, default=1, help='array id')
    parser.add_argument('-s', type=str, default='samples_small.csv', help='sample file')
    parser.add_argument('-d', type=str, default='small', help='dataset name')
    parser.add_argument('--low_psf', action='store_true', help='whether to use low psf')
    parser.add_argument(
        '--observation-model-version',
        type=int,
        choices=(1, 2),
        help='override the sample-file observation model version',
    )
    parser.add_argument('--fiber-layout', choices=('image_axis', 'galaxy_axis'))
    return parser.parse_args()


def _sample_id(row: pd.Series) -> int:
    if 'ID' in row.index:
        return int(row['ID'])
    unnamed = [name for name in row.index if str(name).startswith('Unnamed:')]
    if unnamed:
        return int(row[unnamed[0]])
    return int(row.iloc[0])


def build_generate_command(
    row: pd.Series,
    *,
    part: int,
    dataset: str,
    low_psf: bool = False,
    observation_model_version: int | None = None,
    fiber_layout: str | None = None,
) -> list[str]:
    """Build one simulator command while preserving the legacy CSV schema."""

    missing = [name for name in SIMULATION_PARAMETERS if name not in row.index]
    if missing:
        raise ValueError(f'Sample row is missing required parameters: {missing}')

    rmag_true = None
    if RMAG_TRUE_COLUMN in row.index and pd.notna(row[RMAG_TRUE_COLUMN]):
        rmag_true = float(row[RMAG_TRUE_COLUMN])

    if observation_model_version is None and OBSERVATION_MODEL_VERSION_COLUMN in row.index:
        value = row[OBSERVATION_MODEL_VERSION_COLUMN]
        if pd.notna(value):
            observation_model_version = int(value)
    version, rmag_true = resolve_observation_model(
        observation_model_version,
        rmag_true,
    )
    halpha_flux_true = None
    if HALPHA_FLUX_TRUE_COLUMN in row.index and pd.notna(
        row[HALPHA_FLUX_TRUE_COLUMN]
    ):
        halpha_flux_true = float(row[HALPHA_FLUX_TRUE_COLUMN])
    halpha_flux_true = resolve_halpha_flux(version, halpha_flux_true)
    if fiber_layout is None and FIBER_LAYOUT_COLUMN in row.index:
        value = row[FIBER_LAYOUT_COLUMN]
        if pd.notna(value):
            fiber_layout = str(value)
    if fiber_layout is None and version == CURRENT_OBSERVATION_MODEL_VERSION:
        fiber_layout = GALAXY_AXIS_FIBER_LAYOUT
    fiber_layout = validate_fiber_layout(fiber_layout)

    command = [
        sys.executable,
        join(SCR_DIR, 'generate_fits.py'),
        f'-n={part}',
        f'-d={dataset}',
        f'-ID={_sample_id(row)}',
    ]
    command.extend(f'-{name}={float(row[name])}' for name in SIMULATION_PARAMETERS)
    command.append(f'--observation-model-version={version}')
    command.append(f'--fiber-layout={fiber_layout}')
    if rmag_true is not None:
        command.append(f'--rmag-true={rmag_true}')
    if halpha_flux_true is not None:
        command.append(f'--halpha-flux-true={halpha_flux_true}')
    if low_psf:
        command.append('--low_psf')
    return command


def main():
    args = parse_args()
    sample_file = join(SAMPLE_ROOT, args.s)
    output_dir = Path(FITS_ROOT) / args.d / f'part_{args.n}'
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(sample_file)
    stop = min(args.j, len(samples))
    for _, row in samples.iloc[args.i:stop].iterrows():
        command = build_generate_command(
            row,
            part=args.n,
            dataset=args.d,
            low_psf=args.low_psf,
            observation_model_version=args.observation_model_version,
            fiber_layout=args.fiber_layout,
        )
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
