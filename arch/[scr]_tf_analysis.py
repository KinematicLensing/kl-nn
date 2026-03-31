import os
import argparse
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
    return parser.parse_args()


def resolve_path(base_dir, value):
    if os.path.isabs(value):
        return value
    return join(base_dir, value)

def main():
    args = parse_args()

    stem = args.stem
    nsamples = args.nsamples
    data_dir = resolve_path(BASE_DATASETS_DIR, args.dataset)
    samp_dir = resolve_path(BASE_SAMPLES_DIR, args.sample_set)
    fig_dir = join(BASE_SHARED_DIR, 'figures')
    model_dir = join(BASE_SHARED_DIR, 'models', stem)
    cache_dir = join(BASE_SHARED_DIR, 'cache')

    os.makedirs(join(fig_dir, stem), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

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

    # Save samples and SNR to cache directory
    np.save(join(cache_dir, f'sample_{stem}_{args.i}'), np.array(sample_list))
    np.save(join(cache_dir, f'log_prob_{stem}_{args.i}'), log_probs)
    np.save(join(cache_dir, f'snr_{stem}_{args.i}'), SNR)
    np.save(join(cache_dir, f'truth_{stem}_{args.i}'), denormalize(trues, par_ranges=config.par_ranges))

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
    
    np.save(join(cache_dir, f'map_estimates_{stem}_{args.i}'), map_estimates)
    np.save(join(cache_dir, f'mean_estimates_{stem}_{args.i}'), mean_estimates)

if __name__ == '__main__':
    main()