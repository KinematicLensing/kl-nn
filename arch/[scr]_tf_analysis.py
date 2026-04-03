import os
import argparse
import json
import logging
from datetime import datetime, timezone
from os.path import join
import numpy as np
import torch
from torch.utils.data import Subset
import pyxis.torch as pxt

from train import (
    load_model,
    sample_density,
)
import config
from utils import (
    denormalize,
)

BASE_SHARED_DIR = '/ocean/projects/phy250048p/shared'
BASE_DATASETS_DIR = join(BASE_SHARED_DIR, 'datasets')
BASE_SAMPLES_DIR = join(BASE_SHARED_DIR, 'samples')
DATA_TYPES = (
    'sample',
    'log_prob',
    'snr',
    'truth',
    'map_estimates',
    'mean_estimates',
    'meta',
)


def parse_args():
    parser = argparse.ArgumentParser(description='Sample posterior density for test galaxies.')
    parser.add_argument(
        '-i',
        type=int,
        default=0,
        help='index of subset of galaxies to sample (for parallelization).',
    )
    parser.add_argument(
        '--ngals',
        type=int,
        default=10000,
        help='Number of galaxies to sample.',
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=5000,
        help='Number of posterior samples to draw per galaxy.',
    )
    parser.add_argument(
        '--stem',
        type=str,
        default='CNN-CNN-flow_1m_tf_noprior',
        help='Model stem used for model input and output file names.',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='test_1m_low_hlr',
        help='Dataset directory name under shared/datasets, or absolute path.',
    )
    parser.add_argument(
        '--sample-set',
        dest='sample_set',
        type=str,
        default='samples_test_1m_low_hlr.csv',
        help='Sample-set CSV name under shared/samples, or absolute path.',
    )
    parser.add_argument(
        '--nparts',
        type=int,
        default=None,
        help='Total number of partitions for partXofN naming. If omitted, read from SLURM env.',
    )
    return parser.parse_args()


def resolve_path(base_dir, value):
    if os.path.isabs(value):
        return value
    return join(base_dir, value)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )


def infer_total_partitions(args):
    if args.nparts is not None:
        if args.nparts <= 0:
            raise ValueError(f'Invalid --nparts value: {args.nparts}. Must be > 0.')
        return args.nparts

    for key in ('SLURM_ARRAY_TASK_COUNT', 'SLURM_ARRAY_TASK_MAX'):
        raw = os.environ.get(key)
        if raw is None:
            continue
        value = int(raw)
        if key == 'SLURM_ARRAY_TASK_MAX':
            task_min = int(os.environ.get('SLURM_ARRAY_TASK_MIN', '1'))
            value = value - task_min + 1
        if value > 0:
            return value

    return args.i + 1


def build_partition_label(partition_idx, total_partitions):
    if partition_idx < 0:
        raise ValueError(f'Invalid partition index: {partition_idx}. Must be >= 0.')
    if total_partitions <= 0:
        raise ValueError(f'Invalid total partitions: {total_partitions}. Must be > 0.')
    if partition_idx >= total_partitions:
        raise ValueError(
            f'Partition index {partition_idx} out of range for total partitions {total_partitions}.'
        )
    return f'part{partition_idx}of{total_partitions}'


def get_cache_path(cache_root, model_name, dataset_name, data_type, file_name):
    if data_type not in DATA_TYPES:
        raise ValueError(f'Unsupported data type: {data_type}')
    return join(cache_root, model_name, dataset_name, data_type, file_name)


def ensure_output_dirs(cache_root, model_name, dataset_name):
    created = {}
    for data_type in DATA_TYPES:
        out_dir = join(cache_root, model_name, dataset_name, data_type)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f'Failed to create output directory: {out_dir}') from exc
        created[data_type] = out_dir
    return created


def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def main():
    setup_logging()
    args = parse_args()

    stem = args.stem
    nsamples = args.nsamples
    data_dir = resolve_path(BASE_DATASETS_DIR, args.dataset)
    samp_dir = resolve_path(BASE_SAMPLES_DIR, args.sample_set)
    model_name = os.path.basename(os.path.normpath(stem))
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    total_partitions = infer_total_partitions(args)
    partition_label = build_partition_label(args.i, total_partitions)

    fig_dir = join(BASE_SHARED_DIR, 'figures')
    model_dir = join(BASE_SHARED_DIR, 'models', stem)
    cache_dir = join(BASE_SHARED_DIR, 'cache')

    os.makedirs(join(fig_dir, stem), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    output_dirs = ensure_output_dirs(cache_dir, model_name, dataset_name)
    logging.info('Using cache output root: %s', join(cache_dir, model_name, dataset_name))
    logging.info('Output data type folders: %s', ','.join(sorted(output_dirs.keys())))
    logging.info('Partition %s started', partition_label)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')
    if not os.path.exists(samp_dir):
        raise FileNotFoundError(f'Sample set not found: {samp_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_file = join(model_dir, f'{stem}149')
    model = load_model(mode=2, path=model_file,strict=True, assign=True, device=device)

    # Get data loader
    test_ds = pxt.TorchDataset(data_dir)
    start = args.i * args.ngals
    end = start + args.ngals
    if end > len(test_ds):
        raise ValueError(
            f'Partition range [{start}, {end}) exceeds dataset size {len(test_ds)} for partition {partition_label}.'
        )
    subset = Subset(test_ds, np.arange(start, end))
    gen = torch.Generator(device=device).manual_seed(42)

    # Collect true values for g and vcirc
    g_true = torch.zeros((args.ngals,2), dtype=torch.float32).to(device)
    vcirc_true = torch.zeros((args.ngals), dtype=torch.float32).to(device)
    for i in range(g_true.shape[0]):
        g_true[i] = subset[i]['fid_pars'][:2]
        vcirc_true[i] = subset[i]['fid_pars'][5]
    vcirc_mu = 0.5*(vcirc_true + 1.)*480. + 60. # center of prior

    modes = [1, 2] # 1: no TF prior, 2: TF prior
    sample_list = np.empty((len(modes), args.ngals, args.nsamples, 3), dtype=np.float32)
    log_prob_list = np.empty((len(modes), args.ngals, args.nsamples), dtype=np.float32)

    # sample for each galaxy and each mode
    for mode in modes:
        model.mode = mode
        samples, log_probs, SNR = sample_density(model, subset, nsamples, 
                                                vcirc_mu=vcirc_mu, 
                                                randgen=gen,
                                                return_log_prob=True, 
                                                device=device)
        for i in range(args.ngals):
            sample_list[mode-1, i] = denormalize(samples[i], par_ranges=config.par_ranges)
            log_prob_list[mode-1, i] = log_probs[i]
    
    trues = np.stack((g_true[:, 0].cpu().numpy(), g_true[:, 1].cpu().numpy(), vcirc_true.cpu().numpy()), axis=-1)
    truth_denorm = denormalize(trues, par_ranges=config.par_ranges)

    # Save samples and SNR to cache directory
    saved_files = {}

    sample_path = get_cache_path(cache_dir, model_name, dataset_name, 'sample', f'{partition_label}.npy')
    np.save(sample_path, np.array(sample_list))
    saved_files['sample'] = sample_path
    logging.info('Saved sample: %s', sample_path)

    log_prob_path = get_cache_path(cache_dir, model_name, dataset_name, 'log_prob', f'{partition_label}.npy')
    np.save(log_prob_path, log_prob_list)
    saved_files['log_prob'] = log_prob_path
    logging.info('Saved log_prob: %s', log_prob_path)

    snr_path = get_cache_path(cache_dir, model_name, dataset_name, 'snr', f'{partition_label}.npy')
    np.save(snr_path, SNR)
    saved_files['snr'] = snr_path
    logging.info('Saved snr: %s', snr_path)

    truth_path = get_cache_path(cache_dir, model_name, dataset_name, 'truth', f'{partition_label}.npy')
    np.save(truth_path, truth_denorm)
    saved_files['truth'] = truth_path
    logging.info('Saved truth: %s', truth_path)

    # Compute MAP and mean estimates for each galaxy and mode
    map_estimates = np.zeros((len(modes), args.ngals, 3)) # g1, g2, vcirc
    mean_estimates = np.zeros((len(modes), args.ngals, 3, 3)) # mean with high and low bounds
    for i in range(args.ngals):
        for mode in modes:
            samples = sample_list[mode-1, i]
            log_probs = log_prob_list[mode-1, i]

            # MAP estimate
            map_idx = np.argmax(log_probs)
            map_estimates[mode-1, i] = samples[map_idx]

            # Mean estimate with bounds
            mean_estimates[mode-1, i, 0] = np.percentile(samples, 16, axis=0) # low bound
            mean_estimates[mode-1, i, 1] = np.mean(samples, axis=0) # mean
            mean_estimates[mode-1, i, 2] = np.percentile(samples, 84, axis=0) # high bound
    
    map_path = get_cache_path(cache_dir, model_name, dataset_name, 'map_estimates', f'{partition_label}.npy')
    np.save(map_path, map_estimates)
    saved_files['map_estimates'] = map_path
    logging.info('Saved map_estimates: %s', map_path)

    mean_path = get_cache_path(cache_dir, model_name, dataset_name, 'mean_estimates', f'{partition_label}.npy')
    np.save(mean_path, mean_estimates)
    saved_files['mean_estimates'] = mean_path
    logging.info('Saved mean_estimates: %s', mean_path)

    manifest = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'partition_index': args.i,
        'total_partitions': total_partitions,
        'partition_label': partition_label,
        'ngals': args.ngals,
        'nsamples': args.nsamples,
        'galaxy_range': {'start': start, 'end': end},
        'paths': {
            key: os.path.relpath(path, join(cache_dir, model_name, dataset_name))
            for key, path in saved_files.items()
        },
        'status': 'success',
        'created_at_utc': now_utc_iso(),
        'args': {
            'stem': args.stem,
            'dataset': args.dataset,
            'sample_set': args.sample_set,
        },
    }
    manifest_path = get_cache_path(cache_dir, model_name, dataset_name, 'meta', f'{partition_label}.json')
    with open(manifest_path, 'w', encoding='ascii') as fp:
        json.dump(manifest, fp, indent=2)
    logging.info('Saved meta: %s', manifest_path)
    logging.info('Partition %s completed successfully', partition_label)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception('tf_analysis failed: %s', exc)
        raise