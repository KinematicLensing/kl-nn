from __future__ import annotations

import argparse
import glob
import os
import re
from os.path import basename, join

import numpy as np
import pandas as pd

from tng_rotation_fit import fit_galaxy_rotation_params


def parse_args():
    parser = argparse.ArgumentParser(
        description='Backfill TNG sample CSV files with fitted v0/vcirc/rscale columns.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/ocean/projects/phy250048p/shared/samples',
        help='Directory containing samples_tng_10k_*.csv files.',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='samples_tng_10k_*.csv',
        help='Glob pattern to select CSV files.',
    )
    parser.add_argument(
        '--target-redshift',
        type=float,
        default=0.3,
        help='Redshift used when fitting rotation parameters.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite files in place. Without this, run as dry-run.',
    )
    return parser.parse_args()


def infer_galaxy_index(path: str) -> int:
    stem = basename(path)
    m = re.search(r'_(\d+)\.csv$', stem)
    if m is None:
        raise ValueError(f'Could not infer galaxy index from filename: {path}')
    return int(m.group(1))


def get_i_column(df: pd.DataFrame, path: str) -> np.ndarray:
    if 'i' in df.columns:
        i_vals = df['i'].to_numpy(dtype=np.float64)
        # Legacy TNG files stored cos(i) under column name `i`.
        if np.nanmax(i_vals) <= 1.0 and np.nanmin(i_vals) >= -1.0:
            i_vals = np.arccos(np.clip(i_vals, -1.0, 1.0))
        return i_vals

    if 'sini' in df.columns:
        return np.arcsin(np.clip(df['sini'].to_numpy(dtype=np.float64), 0.0, 1.0))

    if 'cosi' in df.columns:
        return np.arccos(np.clip(df['cosi'].to_numpy(dtype=np.float64), -1.0, 1.0))

    raise ValueError(f'No inclination column found in {path}')


def main():
    args = parse_args()
    pattern = join(args.input_dir, args.pattern)
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f'No files matched pattern: {pattern}')

    print(f'Found {len(files)} files to process')
    for path in files:
        gal_idx = infer_galaxy_index(path)
        fit = fit_galaxy_rotation_params(gal_idx, target_redshift=args.target_redshift)

        df = pd.read_csv(path)
        if 'g1' not in df.columns or 'g2' not in df.columns or 'theta_int' not in df.columns:
            raise ValueError(f'Missing required columns in {path}; expected g1,g2,theta_int')

        row_id = (
            df['row_id'].to_numpy(dtype=np.int64)
            if 'row_id' in df.columns
            else np.arange(len(df), dtype=np.int64)
        )

        out_df = pd.DataFrame(
            {
                'row_id': row_id,
                'g1': df['g1'].to_numpy(dtype=np.float64),
                'g2': df['g2'].to_numpy(dtype=np.float64),
                'theta_int': df['theta_int'].to_numpy(dtype=np.float64),
                'i': get_i_column(df, path),
                'v0': np.full(len(df), fit.v0, dtype=np.float64),
                'vcirc': np.full(len(df), fit.vcirc, dtype=np.float64),
                'rscale': np.full(len(df), fit.rscale, dtype=np.float64),
                'rmse': np.full(len(df), fit.rmse, dtype=np.float64),
            }
        )

        if args.overwrite:
            out_df.to_csv(path, index=False)
            status = 'wrote'
        else:
            status = 'dry-run'

        print(
            f'[{status}] {basename(path)}: gal={gal_idx}, v0={fit.v0:.3f}, '
            f'vcirc={fit.vcirc:.3f}, rscale={fit.rscale:.3f}, rmse={fit.rmse:.3f}'
        )


if __name__ == '__main__':
    main()
