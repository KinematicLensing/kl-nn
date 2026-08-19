from __future__ import print_function
import os
from os.path import join
import time
import importlib.util
import sys
from pathlib import Path
import logging
import shutil

import numpy as np
import pandas as pd
import pyxis as px
from astropy.io import fits
from argparse import ArgumentParser

try:
    from .observation_schema import observation_metadata_arrays
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import observation_metadata_arrays


def parse_args(argv=None):
    parser = ArgumentParser()
    parser.add_argument('-s', type=str, default='small', help='sample name')
    parser.add_argument('-d', type=str, default='small', help='dataset name')
    parser.add_argument('-n', type=int, default=4000, help='number of samples per db entry')
    parser.add_argument('-N', type=int, default=250, help='number of db entries')
    parser.add_argument('--nspec', type=int, default=5, help='number of spectra per sample')
    parser.add_argument('--shard_idx', type=int, default=0, help='Index of the current shard (0-indexed)')
    parser.add_argument('--num_shards', type=int, default=1, help='Total number of shards to split workflow')
    parser.add_argument('--merge', action='store_true', help='Combine existing shards into one master dataset')
    return parser.parse_args(argv)


def normalize(form, data, pars=None):
    if form == 'std': 
        mean, std = pars if pars is not None else (data.mean(), data.std())
        return (data - mean) / std
    else:
        min_val, max_val = pars if pars is not None else (np.min(data), np.max(data))
        if form == '01':
            return (data - min_val) / (max_val - min_val)
        elif form == '-11':
            return (2 * data - (max_val + min_val)) / (max_val - min_val)
        else:
            raise ValueError("Invalid form, must be '01', '-11', or 'std'.")

def load_default_par_ranges():
    config_path = Path(__file__).resolve().parents[1] / 'arch' / 'config.py'
    spec = importlib.util.spec_from_file_location('arch_config', config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load config module from {config_path}')
    arch_config = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = arch_config
    spec.loader.exec_module(arch_config)
    return arch_config.MODEL_CONFIG.par_ranges.copy()

par_ranges = load_default_par_ranges()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def normalize_sample_table(samples, parameter_ranges):
    """Normalize only named inference targets, leaving metadata untouched."""

    missing = [name for name in parameter_ranges if name not in samples.columns]
    if missing:
        raise ValueError(f'Sample table is missing inference targets: {missing}')
    normalized = samples.copy()
    for name, values in parameter_ranges.items():
        normalized[name] = normalize('-11', normalized[name], values)
    return normalized


def extract_fiducial_parameters(samples, row_indices, parameter_names):
    """Return exactly the ordered science targets, never auxiliary metadata."""

    names = list(parameter_names)
    missing = [name for name in names if name not in samples.columns]
    if missing:
        raise ValueError(f'Sample table is missing inference targets: {missing}')
    return samples.iloc[np.asarray(row_indices, dtype=int)][names].to_numpy(
        dtype=np.float32,
        copy=True,
    )

def merge_shards(base_save_dir, num_shards, chunk_size):
    """Combines all isolated LMDB shards sequentially into a master database and deletes the shards."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    shard_dirs = [
        Path(f"{base_save_dir}_shard_{s_idx}_of_{num_shards}")
        for s_idx in range(num_shards)
    ]
    missing = [str(shard_dir) for shard_dir in shard_dirs if not shard_dir.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Refusing to create or modify the merged database because expected "
            f"shards are missing: {missing}"
        )

    logger.info(f"Initiating compilation of {num_shards} shards into target destination: {base_save_dir}")
    
    with px.Writer(dirpath=base_save_dir, map_size_limit=200000, ram_gb_limit=12) as final_db:
        for s_idx, shard_dir in enumerate(shard_dirs):
            logger.info(f"Reading and importing data from: {shard_dir}")

            reader = px.Reader(dirpath=str(shard_dir))
            try:
                num_samples = len(reader)
                for start_i in range(0, num_samples, chunk_size):
                    end_i = min(start_i + chunk_size, num_samples)
                    batch_samples = reader[start_i:end_i]
                    
                    if not batch_samples:
                        continue
                    
                    # Convert list of single dictionaries into grouped batch tensor arrays
                    keys = batch_samples[0].keys()
                    batch_dict = {k: np.array([sample[k] for sample in batch_samples]) for k in keys}
                    final_db.put_samples(batch_dict)
            finally:
                reader.close()
                
            logger.info(f"Successfully integrated shard {s_idx + 1}/{num_shards}")
            
    # Automated Shard Post-Cleanup Sequence
    logger.info("Master database finalized. Starting cleanup of shard files...")
    for shard_dir in shard_dirs:
        shutil.rmtree(shard_dir)
        logger.info(f"Removed temporary directory: {shard_dir}")

def main(argv=None):
    args = parse_args(argv)
    n = args.n
    N = args.N
    nspec = args.nspec
    sample_name = args.s
    dataset_name = args.d
    shard_idx = args.shard_idx
    num_shards = args.num_shards
    data_dir = f'/ocean/projects/phy250048p/shared/fits/{dataset_name}/'
    samp_dir = f'/ocean/projects/phy250048p/shared/samples/samples_{sample_name}.csv'
    save_dir = f'/ocean/projects/phy250048p/shared/datasets/{dataset_name}'
    
    # Switch completely to merging mode if flag is activated
    if args.merge:
        merge_shards(save_dir, num_shards, chunk_size=n)
        return

    # Establish dynamic database file path naming architectures
    if num_shards > 1:
        current_save_dir = f"{save_dir}_shard_{shard_idx}_of_{num_shards}"
    else:
        current_save_dir = save_dir

    samples = normalize_sample_table(pd.read_csv(samp_dir), par_ranges)
    
    # Prevent disk write collisions across array tasks on the shared filesystems
    if shard_idx == 0:
        normalized_dir = '/ocean/projects/phy250048p/shared/samples/normalized'
        os.makedirs(normalized_dir, exist_ok=True)
        samples.to_csv(
            join(normalized_dir, f'samples_{dataset_name}_normalized.csv'),
            index=False,
        )
        
    # Split work arrays uniformly into equal parts across nodes
    all_indices = np.arange(N)
    indices_split = np.array_split(all_indices, num_shards)
    shard_indices = indices_split[shard_idx]
    
    logger.info(f"Shard {shard_idx + 1}/{num_shards} live. Handling {len(shard_indices)} entries.")
        
    with px.Writer(dirpath=current_save_dir, map_size_limit=200000, ram_gb_limit=2) as db:
        for index in shard_indices:
            start = time.time()
            folder = index + 1
            img_stack = np.full((n, 1, 48, 48), 0.)
            spec_stack = np.full((n, 1, nspec, 64), 0.)
            fib_pos_stack = np.full((n, nspec, 2), 0.)
            start_id = index * n
            file_id = index * n
            ids = np.arange(start_id, start_id + n, dtype=np.uint64)
            fids = extract_fiducial_parameters(samples, ids, par_ranges)
            primary_headers = []

            for i in range(n):
                ID = file_id + i
                with fits.open(join(data_dir, f'part_{folder}/gal_{ID}.fits')) as hdu:
                    primary_headers.append(hdu[0].header.copy())
                    img_stack[i, 0] = hdu[nspec+1].data
                    
                    for k in range(nspec):
                        fib_pos_stack[i, k] = hdu[k+1].header['FIBERDX'], hdu[k+1].header['FIBERDY']
                        spec = hdu[k+1].data
                        spec_stack[i, 0, k, :spec.shape[0]] = spec
            
            batch = {
                'img': img_stack,
                'spec': spec_stack,
                'fib_pos': fib_pos_stack,
                'fid_pars': fids,
                'id': ids,
            }
            batch.update(observation_metadata_arrays(primary_headers))
            db.put_samples(batch)
            t = round(time.time() - start, 2)
            logger.info(f'Shard {shard_idx + 1}/{num_shards}: Entry {index+1}/{N} completed in {t}s.')

if __name__ == '__main__':
    main()
