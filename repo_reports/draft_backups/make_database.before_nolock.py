from __future__ import print_function
import os
from os.path import join
import time
import importlib.util
import sys
from pathlib import Path
import logging
import shutil
import hashlib
import json

import numpy as np
import pandas as pd
import pyxis as px
from astropy.io import fits
from argparse import ArgumentParser

try:
    from .observation_schema import (
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        observation_metadata_arrays,
    )
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import (
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        observation_metadata_arrays,
    )


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

def load_arch_module(filename, module_name):
    """Load an arch module when this script is launched from data_generate/."""

    module_path = Path(__file__).resolve().parents[1] / 'arch' / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Failed to load module from {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_default_target_schema():
    config_path = Path(__file__).resolve().parents[1] / 'arch' / 'config.py'
    arch_config = load_arch_module(config_path.name, 'arch_config')
    return (
        arch_config.MODEL_CONFIG.par_ranges.copy(),
        dict(arch_config.TARGET_TRANSFORMS),
    )

par_ranges, target_transforms = load_default_target_schema()
arch_utils = load_arch_module('utils.py', 'arch_utils')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_MANIFEST_FILENAME = 'manifest.json'
GENERATION_MANIFEST_SCHEMA = 'klnn-generation-manifest-v1'
TEST_SET_ANALYSIS_MODE = 'test_set'
TEST_SET_POPULATION = 'tf_conformed_catalog'
REQUIRED_TF_KEYS = {
    'slope',
    'intercept',
    'scatter_dex',
    'vcirc_min',
    'vcirc_max',
}


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generation_manifest(sample_path, expected_sample_count):
    """Validate an optional sample sidecar before any LMDB is opened."""

    sample_path = Path(sample_path).expanduser().resolve()
    manifest_path = sample_path.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        return None
    if not sample_path.is_file():
        raise FileNotFoundError(
            f'Generation manifest exists but its sample CSV is missing: {sample_path}'
        )
    with manifest_path.open(encoding='utf-8') as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f'Generation manifest must be a JSON object: {manifest_path}')
    required = {
        'schema',
        'analysis_mode',
        'population',
        'sample_count',
        'redshift',
        'tf',
        'catalog_sampling',
        'parameter_sampling',
        'sample_table',
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f'Generation manifest is missing required keys: {missing}')
    if payload['schema'] != GENERATION_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported generation manifest schema: {payload['schema']!r}")
    if payload['analysis_mode'] != TEST_SET_ANALYSIS_MODE:
        raise ValueError(
            f"Unsupported generation analysis_mode: {payload['analysis_mode']!r}"
        )
    if payload['population'] != TEST_SET_POPULATION:
        raise ValueError(
            f"Unsupported generation population: {payload['population']!r}"
        )
    if int(payload['sample_count']) != int(expected_sample_count):
        raise ValueError(
            'Generation manifest sample_count does not match requested LMDB size: '
            f"{payload['sample_count']} != {expected_sample_count}"
        )
    redshift = float(payload['redshift'])
    if not np.isfinite(redshift) or redshift != 0.3:
        raise ValueError(f'Expected generation redshift 0.3, got {redshift}')
    tf_config = payload['tf']
    if not isinstance(tf_config, dict) or set(tf_config) != REQUIRED_TF_KEYS:
        raise ValueError(
            'Generation manifest tf must contain exactly '
            f'{sorted(REQUIRED_TF_KEYS)}'
        )
    catalog_sampling = payload['catalog_sampling']
    if not isinstance(catalog_sampling, dict):
        raise ValueError(
            'Generation manifest catalog_sampling must be a JSON object'
        )
    eligibility = catalog_sampling.get('eligibility')
    if not isinstance(eligibility, dict):
        raise ValueError(
            'Generation manifest catalog_sampling.eligibility must be a JSON '
            'object'
        )
    hlr_eligibility = eligibility.get('hlr')
    expected_hlr_keys = {'finite', 'minimum', 'maximum', 'bounds'}
    if (
        not isinstance(hlr_eligibility, dict)
        or set(hlr_eligibility) != expected_hlr_keys
    ):
        raise ValueError(
            'Generation manifest catalog_sampling.eligibility.hlr must '
            'contain exactly finite, minimum, maximum, and bounds'
        )
    if hlr_eligibility['finite'] is not True:
        raise ValueError(
            'Generation manifest catalog_sampling.eligibility.hlr.finite '
            'must be true'
        )
    if hlr_eligibility['bounds'] != 'inclusive':
        raise ValueError(
            'Generation manifest catalog_sampling.eligibility.hlr.bounds '
            "must be 'inclusive'"
        )
    for name, expected_bound in zip(
        ('minimum', 'maximum'),
        par_ranges['hlr'],
    ):
        actual = hlr_eligibility[name]
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isfinite(actual)
            or not np.isclose(
                float(actual),
                float(expected_bound),
                rtol=1e-12,
                atol=1e-12,
            )
        ):
            raise ValueError(
                'Generation manifest catalog_sampling.eligibility.hlr.'
                f'{name} must equal {float(expected_bound)!r}'
            )
    for name, expected_range in (
        ('image_snr', DEFAULT_IMAGE_SNR_RANGE),
        ('halpha_snr', DEFAULT_CENTRAL_HALPHA_SNR_RANGE),
    ):
        expected_eligibility = {
            'finite': True,
            'minimum': expected_range[0],
            'maximum': expected_range[1],
        }
        if eligibility.get(name) != expected_eligibility:
            raise ValueError(
                'Generation manifest catalog_sampling.eligibility.'
                f'{name} must equal {expected_eligibility!r}'
            )
    legacy_hlr_cap_fields = {
        'eligible_hlr_capped_count',
        'selected_hlr_capped_count',
    }
    if legacy_hlr_cap_fields & catalog_sampling.keys():
        raise ValueError(
            'Generation manifest uses legacy HLR cap-after-selection '
            'provenance; regenerate with the inclusive HLR eligibility cut'
        )
    parameter_sampling = payload['parameter_sampling']
    expected_inclination = {
        'distribution': 'cosi_uniform_0_1_latin_hypercube',
        'transform': 'sini=sqrt(1-cosi**2)',
    }
    if (
        not isinstance(parameter_sampling, dict)
        or parameter_sampling.get('inclination') != expected_inclination
    ):
        raise ValueError(
            'Generation manifest parameter_sampling.inclination must equal '
            f'{expected_inclination!r}'
        )
    sample_table = payload['sample_table']
    if not isinstance(sample_table, dict):
        raise ValueError('Generation manifest sample_table must be a JSON object')
    recorded_count = int(sample_table.get('row_count', -1))
    if recorded_count != int(expected_sample_count):
        raise ValueError(
            'Generation manifest sample_table row_count does not match requested '
            f'LMDB size: {recorded_count} != {expected_sample_count}'
        )
    recorded_digest = sample_table.get('sha256')
    if not isinstance(recorded_digest, str) or len(recorded_digest) != 64:
        raise ValueError('Generation manifest has no valid sample_table sha256')
    observed_digest = _sha256_file(sample_path)
    if observed_digest != recorded_digest:
        raise ValueError(
            'Sample CSV SHA-256 does not match its generation manifest: '
            f'{observed_digest} != {recorded_digest}'
        )
    return manifest_path


def propagate_generation_manifest(manifest_path, dataset_dir):
    """Copy a validated sidecar to the canonical completed-dataset location."""

    if manifest_path is None:
        return None
    source = Path(manifest_path).resolve(strict=True)
    destination = Path(dataset_dir) / DATASET_MANIFEST_FILENAME
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite dataset manifest: {destination}')
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info(f'Installed generation manifest: {destination}')
    return destination


def normalize_sample_table(
    samples,
    parameter_ranges,
    parameter_transforms=None,
):
    """Normalize named inference targets, leaving observation metadata untouched."""

    normalized = samples.copy()
    if 'cosi' in parameter_ranges:
        if 'sini' not in normalized.columns:
            raise ValueError(
                "Sample table requires simulator parameter 'sini' to derive "
                "inference target 'cosi'"
            )
        sini = normalized['sini'].to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(sini)) or np.any((sini < 0.0) | (sini > 1.0)):
            raise ValueError("Sample table sini values must be finite and in [0, 1]")
        normalized['cosi'] = np.sqrt(np.maximum(0.0, 1.0 - np.square(sini)))
    missing = [name for name in parameter_ranges if name not in normalized.columns]
    if missing:
        raise ValueError(f'Sample table is missing inference targets: {missing}')
    names = tuple(parameter_ranges)
    normalized.loc[:, names] = arch_utils.normalize_targets(
        normalized.loc[:, names].to_numpy(dtype=np.float64),
        parameter_ranges,
        feature_names=names,
        target_transforms=(
            target_transforms
            if parameter_transforms is None
            else parameter_transforms
        ),
    )
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

def merge_shards(
    base_save_dir,
    num_shards,
    chunk_size,
    *,
    generation_manifest_path=None,
    expected_sample_count=None,
):
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
    if Path(base_save_dir).exists():
        raise FileExistsError(
            "Refusing to merge into an existing dataset; choose a new dataset "
            f"name or explicitly remove the stale output: {base_save_dir}"
        )

    logger.info(f"Initiating compilation of {num_shards} shards into target destination: {base_save_dir}")
    
    merged_sample_count = 0
    with px.Writer(dirpath=base_save_dir, map_size_limit=200000, ram_gb_limit=12) as final_db:
        for s_idx, shard_dir in enumerate(shard_dirs):
            logger.info(f"Reading and importing data from: {shard_dir}")

            reader = px.Reader(dirpath=str(shard_dir))
            try:
                num_samples = len(reader)
                merged_sample_count += num_samples
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

    if (
        expected_sample_count is not None
        and merged_sample_count != expected_sample_count
    ):
        raise RuntimeError(
            'Merged LMDB sample count does not match the generation manifest: '
            f'{merged_sample_count} != {expected_sample_count}'
        )
    propagate_generation_manifest(generation_manifest_path, base_save_dir)

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
    samp_dir = f'/ocean/projects/phy250048p/shared/samples/{sample_name}.csv'
    save_dir = f'/ocean/projects/phy250048p/shared/datasets/{dataset_name}'
    expected_sample_count = n * N
    generation_manifest_path = validate_generation_manifest(
        samp_dir,
        expected_sample_count,
    )
    
    # Switch completely to merging mode if flag is activated
    if args.merge:
        merge_shards(
            save_dir,
            num_shards,
            chunk_size=n,
            generation_manifest_path=generation_manifest_path,
            expected_sample_count=expected_sample_count,
        )
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
    if not 0 <= shard_idx < len(indices_split):
        raise ValueError(
            f"shard_idx must lie in [0, {len(indices_split)}), got {shard_idx}"
        )
    shard_indices = indices_split[shard_idx]
    
    logger.info(f"Shard {shard_idx + 1}/{num_shards} live. Handling {len(shard_indices)} entries.")
        
    current_path = Path(current_save_dir)
    if current_path.exists():
        raise FileExistsError(
            "Refusing to append to an existing shard; explicitly remove or "
            f"rename it before regenerating: {current_save_dir}"
        )
    expected_files = [
        Path(data_dir) / f"part_{index + 1}" / f"gal_{sample_id}.fits"
        for index in shard_indices
        for sample_id in range(index * n, (index + 1) * n)
    ]
    missing_files = [str(path) for path in expected_files if not path.is_file()]
    if missing_files:
        preview = missing_files[:20]
        suffix = "" if len(missing_files) <= 20 else f" ... and {len(missing_files) - 20} more"
        raise FileNotFoundError(
            "FITS preflight failed before opening the LMDB writer: "
            f"{preview}{suffix}"
        )

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

    if num_shards == 1:
        propagate_generation_manifest(generation_manifest_path, current_save_dir)

if __name__ == '__main__':
    main()
