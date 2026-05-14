from __future__ import print_function

import time
from argparse import ArgumentParser
from os.path import isfile, join

import numpy as np
import pandas as pd
import pyxis as px
from astropy.io import fits


FITS_ROOT = '/ocean/projects/phy250048p/shared/fits'
SAMPLE_ROOT = '/ocean/projects/phy250048p/shared/samples'
DATASET_ROOT = '/ocean/projects/phy250048p/shared/datasets'


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
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, default='tng_10k', help='sample name suffix used in samples_<name>_<gal>.csv')
    parser.add_argument('-d', type=str, default='test_tng_10k', help='dataset name under shared/fits and shared/datasets')
    parser.add_argument('--nspec', type=int, default=5, help='number of spectra per sample')
    parser.add_argument('--ngal', type=int, default=50, help='number of TNG galaxies to ingest')
    parser.add_argument('--rows-per-gal', type=int, default=10000, help='samples per galaxy and per db entry')
    parser.add_argument(
        '--id-stride',
        type=int,
        default=10000,
        help='global ID stride between galaxies in FITS filenames (e.g. gal_10000 starts galaxy 1)',
    )
    return parser.parse_args()


def get_tng_sample_file(sample_name, gal_idx):
    return join(SAMPLE_ROOT, f'samples_{sample_name}_{gal_idx}.csv')


def load_fiducial_parameters(sample_name, ngal, rows_per_gal):
    # Keep the fiducial parameter order aligned with make_database conventions.
    # For TNG we store: g1, g2, theta_int, i, v0, vcirc, rscale.
    total = ngal * rows_per_gal
    fids = np.zeros((total, 7), dtype=np.float32)

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

        # Match the normalization convention used by make_database.
        g1 = normalize('-11', g1, (-0.1, 0.1)).astype(np.float32)
        g2 = normalize('-11', g2, (-0.1, 0.1)).astype(np.float32)
        theta_int = normalize('-11', theta_int, (-np.pi, np.pi)).astype(np.float32)
        incl = normalize('-11', incl, (0.0, np.pi)).astype(np.float32)
        v0 = normalize('-11', v0, (-30.0, 30.0)).astype(np.float32)
        vcirc = normalize('-11', vcirc, (60.0, 540.0)).astype(np.float32)
        rscale = normalize('-11', rscale, (0.1, 10.0)).astype(np.float32)

        start = gal_idx * rows_per_gal
        end = start + rows_per_gal
        fids[start:end, 0] = g1
        fids[start:end, 1] = g2
        fids[start:end, 2] = theta_int
        fids[start:end, 3] = incl
        fids[start:end, 4] = v0
        fids[start:end, 5] = vcirc
        fids[start:end, 6] = rscale

    return fids


def load_tng_sample(data_dir, gal_idx, sample_id, nspec, spec_len=64):
    fits_file = join(data_dir, f'galaxy_{gal_idx}', f'gal_{sample_id}.fits')
    if not isfile(fits_file):
        raise FileNotFoundError(f'Missing FITS file: {fits_file}')

    with fits.open(fits_file) as hdu:
        image = np.asarray(hdu['IMAGE'].data, dtype=np.float32)
        flux = np.asarray(hdu['FLUX'].data, dtype=np.float32)

    specs = np.zeros((nspec, spec_len), dtype=np.float32)
    n_copy_spec = min(nspec, flux.shape[0])
    n_copy_wave = min(spec_len, flux.shape[1])
    specs[:n_copy_spec, :n_copy_wave] = flux[:n_copy_spec, :n_copy_wave]

    return image, specs


def main():
    args = parse_args()

    data_dir = join(FITS_ROOT, args.d)
    save_dir = join(DATASET_ROOT, args.d)

    fids_all = load_fiducial_parameters(args.s, args.ngal, args.rows_per_gal)

    with px.Writer(dirpath=save_dir, map_size_limit=200000, ram_gb_limit=2) as db:
        for gal_id in range(args.ngal):
            gal_idx = gal_id
            start_time = time.time()
            start_id = gal_id * args.rows_per_gal
            end_id = (gal_id + 1) * args.rows_per_gal
            file_id_start = gal_idx * args.id_stride

            n_this = end_id - start_id
            ids = np.arange(file_id_start, file_id_start + n_this, dtype=np.uint64)

            img_stack = np.zeros((n_this, 1, 48, 48), dtype=np.float32)
            spec_stack = np.zeros((n_this, 1, args.nspec, 64), dtype=np.float32)
            fids = fids_all[start_id:end_id]

            for i, sample_id in enumerate(ids):
                image, specs = load_tng_sample(
                    data_dir,
                    gal_idx,
                    int(sample_id),
                    args.nspec,
                )
                img_stack[i, 0] = image
                spec_stack[i, 0] = specs

            print(img_stack.shape, spec_stack.shape, fids.shape, ids.shape)

            db.put_samples(
                {
                    'img': img_stack,
                    'spec': spec_stack,
                    'fid_pars': fids,
                    'id': ids,
                }
            )

            elapsed = round(time.time() - start_time, 2)
            print(
                f'galaxy entry {gal_idx + 1}/{args.ngal} complete, file IDs {file_id_start}-{file_id_start + n_this - 1}, {elapsed} seconds'
            )


if __name__ == '__main__':
    main()
