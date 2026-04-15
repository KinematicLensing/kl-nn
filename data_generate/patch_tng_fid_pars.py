from __future__ import print_function

import time
from argparse import ArgumentParser
from os.path import isfile, join

import numpy as np
import pandas as pd
import pyxis as px


SAMPLE_ROOT = '/ocean/projects/phy250048p/shared/samples'


def normalize(form, data, pars=None):
    """
    Normalizes data into one of three forms:
    '01': normalize between 0 and 1, pars = (min, max)
    '-11': normalize between -1 and 1, pars = (min, max)
    'std': standardize to center around 0 with std dev of 1, pars = (mean, std)
    """
    if form == 'std':
        mean, std = pars if pars is not None else (data.mean(), data.std())
        return (data - mean) / std

    min_val, max_val = pars if pars is not None else (np.min(data), np.max(data))

    if form == '01':
        return (data - min_val) / (max_val - min_val)

    if form == '-11':
        return (2 * data - (max_val + min_val)) / (max_val - min_val)

    raise ValueError("Invalid form, must be '01', '-11', or 'std'.")


def parse_args():
    parser = ArgumentParser(
        description='Patch an existing TNG pyxis dataset by replacing fid_pars from sample CSVs.'
    )
    parser.add_argument(
        '--input-dataset',
        '-i',
        type=str,
        required=True,
        help='Path to existing pyxis dataset directory to read.',
    )
    parser.add_argument(
        '--output-dataset',
        '-o',
        type=str,
        required=True,
        help='Path to output pyxis dataset directory with patched fid_pars.',
    )
    parser.add_argument(
        '-s',
        type=str,
        default='tng_10k',
        help='sample name suffix used in samples_<name>_<gal>.csv',
    )
    parser.add_argument('--ngal', type=int, default=50, help='number of TNG galaxies')
    parser.add_argument(
        '--rows-per-gal',
        type=int,
        default=10000,
        help='number of rows per galaxy in sample CSVs',
    )
    parser.add_argument(
        '--id-stride',
        type=int,
        default=10000,
        help='global ID stride between galaxies in dataset sample IDs',
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=2048,
        help='number of samples per write chunk',
    )
    parser.add_argument(
        '--map-size-limit',
        type=int,
        default=200000,
        help='px.Writer map_size_limit',
    )
    parser.add_argument(
        '--ram-gb-limit',
        type=int,
        default=2,
        help='px.Writer ram_gb_limit',
    )
    return parser.parse_args()


def get_tng_sample_file(sample_name, gal_idx):
    return join(SAMPLE_ROOT, f'samples_{sample_name}_{gal_idx}.csv')


def load_fiducial_parameters(sample_name, ngal, rows_per_gal):
    """Return normalized fiducials by galaxy index: shape (ngal, rows_per_gal, 7)."""
    fids = np.zeros((ngal, rows_per_gal, 7), dtype=np.float32)

    for gal_idx in range(ngal):
        sample_file = get_tng_sample_file(sample_name, gal_idx)
        if not isfile(sample_file):
            raise FileNotFoundError(f'Missing sample file: {sample_file}')

        df = pd.read_csv(sample_file)

        if len(df) < rows_per_gal:
            raise ValueError(
                f'Sample file {sample_file} has {len(df)} rows, expected at least {rows_per_gal}'
            )

        g1 = df['g1'].to_numpy(dtype=np.float32)[:rows_per_gal]
        g2 = df['g2'].to_numpy(dtype=np.float32)[:rows_per_gal]
        theta_int = df['theta_int'].to_numpy(dtype=np.float32)[:rows_per_gal]

        if 'i' in df.columns:
            incl = df['i'].to_numpy(dtype=np.float32)[:rows_per_gal]
            # Legacy files stored cos(i) under the `i` column name.
            if np.nanmax(incl) <= 1.0 and np.nanmin(incl) >= -1.0:
                incl = np.arccos(np.clip(incl, -1.0, 1.0)).astype(np.float32)
        elif 'sini' in df.columns:
            sini = np.clip(df['sini'].to_numpy(dtype=np.float32)[:rows_per_gal], 0.0, 1.0)
            incl = np.arcsin(sini).astype(np.float32)
        elif 'cosi' in df.columns:
            cosi = np.clip(df['cosi'].to_numpy(dtype=np.float32)[:rows_per_gal], -1.0, 1.0)
            incl = np.arccos(cosi).astype(np.float32)
        else:
            raise ValueError(
                f'Expected an inclination column in {sample_file}, found {list(df.columns)}'
            )

        if 'v0' not in df.columns or 'vcirc' not in df.columns or 'rscale' not in df.columns:
            raise ValueError(
                f'Expected v0/vcirc/rscale columns in {sample_file}, found {list(df.columns)}'
            )

        v0 = df['v0'].to_numpy(dtype=np.float32)[:rows_per_gal]
        vcirc = df['vcirc'].to_numpy(dtype=np.float32)[:rows_per_gal]
        rscale = df['rscale'].to_numpy(dtype=np.float32)[:rows_per_gal]

        g1 = normalize('-11', g1, (-0.1, 0.1)).astype(np.float32)
        g2 = normalize('-11', g2, (-0.1, 0.1)).astype(np.float32)
        theta_int = normalize('-11', theta_int, (-np.pi, np.pi)).astype(np.float32)
        incl = normalize('-11', incl, (0.0, np.pi)).astype(np.float32)
        v0 = normalize('-11', v0, (-30.0, 30.0)).astype(np.float32)
        vcirc = normalize('-11', vcirc, (60.0, 540.0)).astype(np.float32)
        rscale = normalize('-11', rscale, (0.1, 10.0)).astype(np.float32)

        fids[gal_idx, :, 0] = g1
        fids[gal_idx, :, 1] = g2
        fids[gal_idx, :, 2] = theta_int
        fids[gal_idx, :, 3] = incl
        fids[gal_idx, :, 4] = v0
        fids[gal_idx, :, 5] = vcirc
        fids[gal_idx, :, 6] = rscale

    return fids


def _to_scalar_int(value):
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f'Expected scalar id, got shape {value.shape}')
        return int(value.reshape(-1)[0])
    return int(value)


def _stack_field(values):
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.stack(values, axis=0)
    return np.asarray(values)


def main():
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError(f'Invalid chunk size: {args.chunk_size}. Must be > 0.')

    print('Loading normalized fiducials from sample CSV files...')
    fids_by_gal = load_fiducial_parameters(args.s, args.ngal, args.rows_per_gal)
    print(f'Loaded fiducials for {args.ngal} galaxies x {args.rows_per_gal} rows')

    t0 = time.time()
    with px.Reader(args.input_dataset) as db_in, px.Writer(
        dirpath=args.output_dataset,
        map_size_limit=args.map_size_limit,
        ram_gb_limit=args.ram_gb_limit,
    ) as db_out:
        nsamples = len(db_in)
        if nsamples == 0:
            raise ValueError(f'Input dataset is empty: {args.input_dataset}')

        print(f'Patching {nsamples} samples from {args.input_dataset}')
        n_written = 0
        for start in range(0, nsamples, args.chunk_size):
            end = min(start + args.chunk_size, nsamples)
            batch_lists = {}

            for i in range(start, end):
                sample = db_in[i]
                sid = _to_scalar_int(sample['id'])
                gal_idx = sid // args.id_stride
                row_idx = sid % args.id_stride

                if gal_idx < 0 or gal_idx >= args.ngal:
                    raise IndexError(
                        f'Sample id {sid} maps to gal_idx={gal_idx}, outside [0, {args.ngal - 1}]'
                    )
                if row_idx < 0 or row_idx >= args.rows_per_gal:
                    raise IndexError(
                        f'Sample id {sid} maps to row_idx={row_idx}, outside [0, {args.rows_per_gal - 1}]'
                    )

                sample['fid_pars'] = fids_by_gal[gal_idx, row_idx].copy()

                for key, val in sample.items():
                    if key not in batch_lists:
                        batch_lists[key] = []
                    batch_lists[key].append(val)

            batch = {k: _stack_field(v) for k, v in batch_lists.items()}
            db_out.put_samples(batch)
            n_written += end - start

            elapsed = round(time.time() - t0, 2)
            print(f'Patched {n_written}/{nsamples} samples ({elapsed}s)')

    total_t = round(time.time() - t0, 2)
    print(f'Done. Wrote patched dataset to {args.output_dataset} in {total_t}s')


if __name__ == '__main__':
    main()
