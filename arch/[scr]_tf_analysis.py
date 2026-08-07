import os
import argparse
import json
import logging
import time
import functools
from datetime import datetime, timezone
from contextlib import nullcontext
from os.path import join
import numpy as np
import torch
from torch.utils.data import Subset
import pyxis.torch as pxt
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

from train import (
    load_model,
    sample_density,
    _resolve_amp_dtype,
)
from networks import KLNPE
import config
from model_registry import load_model_config
from utils import (
    denormalize,
    resolve_feature_index,
)
from data import rot_90_param_only

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
        '--epoch',
        type=int,
        default=199,
        help='Epoch number of the model to load.',
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
    parser.add_argument(
        '--conform-to-tf',
        dest='conform_to_tf',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Conform dataset to TF prior (default: False).',
    )
    parser.add_argument(
        '--cancel-add-noise',
        dest='cancel_add_noise',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Cancel the addition of noise to the images (default: False).',
    )
    parser.add_argument(
        '--compile',
        dest='use_compile',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable torch.compile (default: config.train).',
    )
    parser.add_argument(
        '--compile-mode',
        type=str,
        default=None,
        help='torch.compile mode (default: config.train).',
    )
    parser.add_argument(
        '--compile-backend',
        type=str,
        default=None,
        help='torch.compile backend (default: config.train; use "none" to disable override).',
    )
    parser.add_argument(
        '--amp',
        dest='use_amp',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Enable AMP autocast (default: config.train).',
    )
    parser.add_argument(
        '--amp-dtype',
        type=str,
        default=None,
        help='AMP dtype (float16 or bfloat16; default: config.train).',
    )
    parser.add_argument(
        '--channels-last',
        dest='channels_last',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Use channels_last memory format (default: config.train).',
    )
    parser.add_argument(
        '--inference-mode',
        dest='inference_mode',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Use torch.inference_mode (default: True).',
    )
    parser.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Log timing for key phases.',
    )
    parser.add_argument(
        '--cached-snrs-path',
        type=str,
        default=None,
        help='Path to .npy file containing pre-saved SNR values to use instead of generating new ones.',
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


FID_PARS_INDEX_BY_FEATURE = {
    'g1': 0,
    'g2': 1,
    'theta_int': 2,
    'sini': 3,
    'v0': 4,
    'vcirc': 5,
    'rscale': 6,
    'hlr': 7,
}


def build_truth_array(subset, feature_names, progress=None):
    truth = np.empty((len(subset), len(feature_names)), dtype=np.float32)
    iterator = range(len(subset))
    if progress is not None:
        iterator = progress(iterator, total=len(subset), desc="Collect truth")
    for row_idx in iterator:
        fid_pars = subset[row_idx]['fid_pars']
        if torch.is_tensor(fid_pars):
            fid_pars = fid_pars.detach().cpu().numpy()
        else:
            fid_pars = np.asarray(fid_pars)

        for col_idx, feature_name in enumerate(feature_names):
            try:
                source_idx = FID_PARS_INDEX_BY_FEATURE[feature_name]
            except KeyError as exc:
                raise ValueError(f'Unsupported feature name in config.train["feature_names"]: {feature_name}') from exc
            truth[row_idx, col_idx] = fid_pars[source_idx]
    return truth

def _sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def sampling_progress(iterable, mode, **kwargs):
    cleaned = dict(kwargs)
    cleaned.pop("desc", None)
    return tqdm(iterable, desc=f"Sampling mode {mode}", **cleaned)

def main():
    setup_logging()
    args = parse_args()

    stem = args.stem
    epoch = args.epoch
    nsamples = args.nsamples
    data_dir = resolve_path(BASE_DATASETS_DIR, args.dataset)
    samp_dir = resolve_path(BASE_SAMPLES_DIR, args.sample_set)
    model_name = os.path.basename(os.path.normpath(stem))
    model_cfg = load_model_config(model_name, allow_fallback_current=True)
    config.set_model_config(model_cfg)
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    if args.conform_to_tf:
        dataset_name += '_tf_conformed'
    total_partitions = infer_total_partitions(args)
    partition_label = build_partition_label(args.i, total_partitions)
    nfeatures = config.train['feature_number']
    feature_names = config.train['feature_names']

    if len(feature_names) != nfeatures:
        raise ValueError(
            f"config.train['feature_names'] length {len(feature_names)} does not match feature_number {nfeatures}."
        )

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
    profile = bool(args.profile)
    use_compile = args.use_compile
    effective_use_compile = use_compile if use_compile is not None else bool(config.train.get('use_compile', False))
    compile_mode = args.compile_mode
    compile_backend = args.compile_backend
    if isinstance(compile_backend, str) and compile_backend.lower() == 'none':
        compile_backend = None
    use_amp = args.use_amp if args.use_amp is not None else bool(config.train.get('use_amp', False))
    if device.type != 'cuda':
        use_amp = False
    amp_dtype = args.amp_dtype or config.train.get('amp_dtype', 'float16')
    amp_dtype = _resolve_amp_dtype(amp_dtype)
    channels_last = args.channels_last if args.channels_last is not None else bool(config.train.get('channels_last', False))
    inference_mode = True if args.inference_mode is None else bool(args.inference_mode)

    model_file = join(model_dir, f'{stem}{epoch}')
    models = {}
    use_separate_models = bool(effective_use_compile)

    def get_model_for_mode(mode):
        if mode in models:
            return models[mode]
        if profile:
            _sync_cuda(device)
            start = time.perf_counter()
        model = load_model(
            train_config=config.train,
            Model=KLNPE,
            path=model_file,
            strict=True,
            assign=True,
            device=device,
            model_name=model_name,
            use_compile=use_compile,
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            channels_last=channels_last,
        )
        if profile:
            _sync_cuda(device)
            logging.info('Timing: load model mode %s took %.2fs', mode, time.perf_counter() - start)
        models[mode] = model
        return model

    if not use_separate_models:
        get_model_for_mode(2)

    # Get data loader
    test_ds = pxt.TorchDataset(data_dir)
    start = args.i * args.ngals
    end = start + args.ngals
    if end > len(test_ds):
        raise ValueError(
            f'Partition range [{start}, {end}) exceeds dataset size {len(test_ds)} for partition {partition_label}.'
        )
    subset = Subset(test_ds, np.arange(start, end))

    vcirc_idx = resolve_feature_index(feature_names, 'vcirc', aliases=('v_circ',))
    vcirc_name = feature_names[vcirc_idx]
    vcirc_low, vcirc_high = config.par_ranges[vcirc_name]

    # Collect true vcirc values in normalized space and convert to km/s center of prior.
    if profile:
        _sync_cuda(device)
        start = time.perf_counter()
    vcirc_true = torch.zeros((args.ngals), dtype=torch.float32, device=device)
    vcirc_iter = tqdm(range(args.ngals), desc="Collect vcirc") if args.ngals else range(0)
    for i in vcirc_iter:
        vcirc_true[i] = subset[i]['fid_pars'][vcirc_idx]
    vcirc_mu = 0.5 * (vcirc_true + 1.0) * (vcirc_high - vcirc_low) + vcirc_low

    truth = build_truth_array(subset, feature_names, progress=tqdm)
    if profile:
        _sync_cuda(device)
        logging.info('Timing: prep vcirc/truth took %.2fs', time.perf_counter() - start)
    
    rng_seed = 42
    if args.conform_to_tf:
        logging.info('Conforming dataset to TF prior')
        from data import TFCalculator, app_mag_to_snr
        tf_calc = TFCalculator(slope=config.tf['slope'], intercept=config.tf['intercept'], scatter=config.tf['scatter'])
        app_mag = tf_calc.sample_mag_from_vcirc(vcirc_mu)
        snr_shared = app_mag_to_snr(app_mag)
        print('SNR range: min %.2f, max %.2f, mean %.2f' % (snr_shared.min().item(), snr_shared.max().item(), snr_shared.mean().item()))
    elif args.cached_snrs_path is not None:
        if not os.path.exists(args.cached_snrs_path):
            raise FileNotFoundError(f'Cached SNRs file not found: {args.cached_snrs_path}')
        snrs = []
        for i in range(total_partitions):
            snr_path = os.path.join(args.cached_snrs_path, f'part{i}of{total_partitions}.npy')
            if not os.path.exists(snr_path):
                raise FileNotFoundError(f'Cached SNRs file for partition not found: {snr_path}')
            snrs.append(np.load(snr_path))
        snr_shared = torch.from_numpy(np.concatenate(snrs)).to(device)
        if snr_shared.shape != (args.ngals,):
            raise ValueError(f'Cached SNRs shape {snr_shared.shape} does not match expected ({args.ngals},)')
    else:
        snr_gen = torch.Generator(device=device).manual_seed(rng_seed)
        snr_shared = torch.rand(args.ngals, generator=snr_gen, device=device) * 995 + 5
        app_mag = None

    modes = [2] # 1: no TF prior, 2: TF prior
    sample_list = np.empty((len(modes), args.ngals, args.nsamples, nfeatures), dtype=np.float32)
    log_prob_list = np.empty((len(modes), args.ngals, args.nsamples), dtype=np.float32)
    if args.cancel_add_noise:
        sample_list_rot90 = np.empty((len(modes), args.ngals, args.nsamples, nfeatures), dtype=np.float32)
        log_prob_list_rot90 = np.empty((len(modes), args.ngals, args.nsamples), dtype=np.float32)

    # sample for each galaxy and each mode
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    infer_ctx = torch.inference_mode() if inference_mode else nullcontext()
    with infer_ctx, amp_ctx:
        for j, mode in enumerate(modes):
            model = get_model_for_mode(mode) if use_separate_models else models[2]
            model.mode = mode
            # Reset the noise RNG so each mode sees the same injected img/spec noise.
            noise_gen = torch.Generator(device=device).manual_seed(rng_seed)
            if profile:
                _sync_cuda(device)
                start = time.perf_counter()
            sampling_progress_fn = functools.partial(sampling_progress, mode=mode)
            samples, log_probs, SNR = sample_density(
                model,
                subset,
                nsamples,
                snr=snr_shared,
                mag=app_mag,
                vcirc_mu=vcirc_mu,
                randgen=noise_gen,
                return_log_prob=True,
                apply_add_noise_cancellation=args.cancel_add_noise,
                device=device,
                channels_last=channels_last,
                progress=sampling_progress_fn,
            )
            if profile:
                _sync_cuda(device)
                logging.info('Timing: sample_density mode %s took %.2fs', mode, time.perf_counter() - start)
            post_iter = tqdm(range(args.ngals), desc=f"Postprocess mode {mode}") if args.ngals else range(0)
            post_start = time.perf_counter() if profile else None
            for i in post_iter:
                if args.cancel_add_noise:
                    sample_list[j, i] = denormalize(samples[i, :, 0, :], par_ranges=config.par_ranges, feature_names=feature_names)
                    sample_list_rot90[j, i] = denormalize(samples[i, :, 1, :], par_ranges=config.par_ranges, feature_names=feature_names)
                    log_prob_list[j, i] = log_probs[i, :, 0]
                    log_prob_list_rot90[j, i] = log_probs[i, :, 1]
                else:
                    sample_list[j, i] = denormalize(samples[i], par_ranges=config.par_ranges, feature_names=feature_names)
                    log_prob_list[j, i] = log_probs[i]
            if profile and post_start is not None:
                logging.info('Timing: postprocess mode %s took %.2fs', mode, time.perf_counter() - post_start)

    truth_denorm = denormalize(truth, par_ranges=config.par_ranges, feature_names=feature_names)

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
    map_estimates = np.zeros((len(modes), args.ngals, nfeatures))
    mean_estimates = np.zeros((len(modes), args.ngals, 3, nfeatures))
    for i in range(args.ngals):
        for j, mode in enumerate(modes):
            samples = sample_list[j, i]
            log_probs = log_prob_list[j, i]
            if args.cancel_add_noise:
                samples_rot90 = sample_list_rot90[j, i]
                log_probs_rot90 = log_prob_list_rot90[j, i]

            # MAP estimate
            if args.cancel_add_noise:
                map_idx = np.argmax(log_probs)
                map_idx_rot90 = np.argmax(log_probs_rot90)
                counter_sample = rot_90_param_only(samples_rot90[map_idx_rot90], reverse=True) # rotate best fit back to original orientation
                map_estimates[j, i] = 0.5 * (samples[map_idx] + counter_sample)
            else:
                map_idx = np.argmax(log_probs)
                map_estimates[j, i] = samples[map_idx]

            # Mean estimate with bounds
            mean_estimates[j, i, 0] = np.percentile(samples, 16, axis=0) # low bound
            mean_estimates[j, i, 1] = np.mean(samples, axis=0) # mean
            mean_estimates[j, i, 2] = np.percentile(samples, 84, axis=0) # high bound
    
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
