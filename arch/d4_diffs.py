import os
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from os.path import join
import copy

import numpy as np
import pandas as pd
import torch
import pyxis.torch as pxt
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable, **kwargs):
        return iterable

from train import (
    load_model,
    apply_noise,
    _resolve_amp_dtype,
)
import config
from model_registry import load_model_config

BASE_SHARED_DIR = '/ocean/projects/phy250048p/shared'
BASE_DATASETS_DIR = join(BASE_SHARED_DIR, 'datasets')
BASE_SAMPLES_DIR = join(BASE_SHARED_DIR, 'samples')
DEFAULT_CACHE_ROOT = join(BASE_SHARED_DIR, 'cache')

DATA_TYPES = (
    'd4_diff',
    'd4_truth',
    'd4_pred',
    'meta',
)

D4_TRANS = ['e', 'r90', 'r180', 'r270', 'v', 't', 'h', 'hvt']
D4_SPEC_PERM = {
    'e': [0, 1, 2, 3, 4],
    'r90': [0, 1, 2, 3, 4],
    'r180': [0, 1, 2, 3, 4],
    'r270': [0, 1, 2, 3, 4],
    'v': [0, 1, 2, 4, 3],
    't': [0, 1, 2, 4, 3],
    'h': [0, 1, 2, 4, 3],
    'hvt': [0, 1, 2, 4, 3],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Compute D4 diffs across a dataset partition.')
    parser.add_argument(
        '-i',
        type=int,
        default=0,
        help='Index of subset of galaxies to process (for parallelization).',
    )
    parser.add_argument(
        '--nparts',
        type=int,
        default=None,
        help='Total number of partitions for partXofN naming. If omitted, read from SLURM env.',
    )
    parser.add_argument(
        '--nsamples',
        type=int,
        default=10000,
        help='Number of posterior samples to draw per D4 transform.',
    )
    parser.add_argument(
        '--stem',
        type=str,
        default='CNN-CNN-flow',
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
        default='valid_1m',
        help='Dataset directory name under shared/datasets, or absolute path.',
    )
    parser.add_argument(
        '--sample-set',
        dest='sample_set',
        type=str,
        default='samples_valid_1m.csv',
        help='Sample-set CSV name under shared/samples, or absolute path.',
    )
    parser.add_argument(
        '--mode',
        type=int,
        choices=[0, 1],
        default=0,
        help='0 = no TF prior (model.mode=1, no vcirc); 1 = TF prior (model.mode=2, use vcirc).',
    )
    parser.add_argument(
        '--cache-root',
        type=str,
        default=DEFAULT_CACHE_ROOT,
        help='Root directory for cache outputs.',
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
        '--use-optimization',
        dest='use_optimization',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Toggle optimization bundle (compile/AMP/channels_last).',
    )
    parser.add_argument(
        '--profile',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Log timing for key phases.',
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


def compute_partition_bounds(total_gals, partition_idx, total_partitions):
    if total_gals <= 0:
        raise ValueError(f'Invalid dataset length: {total_gals}. Must be > 0.')
    if partition_idx < 0 or partition_idx >= total_partitions:
        raise ValueError(
            f'Partition index {partition_idx} out of range for total partitions {total_partitions}.'
        )
    base = total_gals // total_partitions
    remainder = total_gals % total_partitions
    start = partition_idx * base + min(partition_idx, remainder)
    size = base + (1 if partition_idx < remainder else 0)
    end = start + size
    if size <= 0:
        raise ValueError(
            f'Partition {partition_idx} receives zero samples (total {total_gals}, parts {total_partitions}).'
        )
    return start, end


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


def apply_d4_transform(image, transform_id):
    """Applies D4 transformation to an image (H, W, C) or (H, W)."""
    if transform_id == 'e':
        return image
    if transform_id == 'r90':
        return torch.rot90(image, k=1)
    if transform_id == 'r180':
        return torch.rot90(image, k=2)
    if transform_id == 'r270':
        return torch.rot90(image, k=3)
    if transform_id == 'v':
        return torch.flip(image, dims=[0])
    if transform_id == 't':
        return torch.transpose(image, 1, 0)
    if transform_id == 'h':
        return torch.flip(image, dims=[1])
    if transform_id == 'hvt':
        return torch.flip(torch.transpose(image, 0, 1), dims=[0, 1])
    return image


def d4_fib_pos(fp, transform_id):
    """Applies D4 transformation to fiber positions."""
    ref = torch.from_numpy(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32))
    if transform_id == 'e':
        return fp
    if transform_id == 'r90':
        return fp[[3, 4, 2, 1, 0]]
    if transform_id == 'r180':
        return fp[[1, 0, 2, 4, 3]]
    if transform_id == 'r270':
        return fp[[4, 3, 2, 0, 1]]
    if transform_id == 'v':
        return fp[[0, 1, 2, 4, 3]] @ ref
    if transform_id == 't':
        return fp[[4, 3, 2, 1, 0]] @ ref
    if transform_id == 'h':
        return fp[[1, 0, 2, 3, 4]] @ ref
    if transform_id == 'hvt':
        return fp[[3, 4, 2, 0, 1]] @ ref
    return fp


def build_d4_datavector_set(gal):
    theta = gal['fid_pars'][2].item() * np.pi
    d4_shear = [(-1) ** i for i in range(8)]
    d4_rot = [
        0,
        -np.pi / 2,
        -np.pi,
        -3 * np.pi / 2,
        -2 * theta,
        -2 * theta - np.pi / 2,
        -2 * theta - np.pi,
        -2 * theta - 3 * np.pi / 2,
    ]
    d4_set = [copy.deepcopy(gal) for _ in range(8)]

    for i, g in enumerate(d4_set):
        t = D4_TRANS[i]
        g2_neg = -1 if i > 3 else 1

        g['img'][0] = apply_d4_transform(torch.clone(g['img'][0]), t)
        g['spec'][0] = g['spec'][0][D4_SPEC_PERM[t]]
        g['fib_pos'] = d4_fib_pos(torch.clone(g['fib_pos'].float()), t)

        g['fid_pars'][0] *= d4_shear[i]
        g['fid_pars'][1] *= d4_shear[i] * g2_neg
        g['fid_pars'][2] = g['fid_pars'][2] * np.pi + d4_rot[i]
        if g['fid_pars'][2] > np.pi:
            g['fid_pars'][2] -= 2 * np.pi
        elif g['fid_pars'][2] < -np.pi:
            g['fid_pars'][2] += 2 * np.pi
        g['fid_pars'][2] /= np.pi

    return d4_set, D4_TRANS.copy()


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def _sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def d4_diff(
    gal_id,
    gal,
    model,
    device,
    fid_pars_phys,
    use_vcirc,
    nsamples,
    channels_last=False,
    use_amp=False,
    amp_dtype=torch.float16,
    inference_mode=True,
):
    d4_set, _ = build_d4_datavector_set(gal)

    model.eval()
    samples = []
    log_probs = []
    snr = torch.rand((1,), device=device) * 995 + 5
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    grad_ctx = torch.inference_mode() if inference_mode else torch.no_grad()
    with grad_ctx, amp_ctx:
        for d4_idx, g in enumerate(d4_set):
            img = apply_noise(g['img'].float().unsqueeze(0).to(device), snr, device=device)
            spec = apply_noise(g['spec'].float().unsqueeze(0).to(device), snr, device=device)
            fp = apply_noise(g['fib_pos'].float().unsqueeze(0).to(device), snr, device=device)
            if channels_last:
                img = img.contiguous(memory_format=torch.channels_last)
                spec = spec.contiguous(memory_format=torch.channels_last)
            vcirc = None
            if use_vcirc:
                try:
                    vcirc_value = float(fid_pars_phys.iloc[gal_id]['vcirc'])
                except KeyError as exc:
                    raise ValueError('vcirc column missing from sample CSV.') from exc
                vcirc = torch.tensor([vcirc_value], device=device)
            samp, lp = model.sample(
                img,
                spec,
                num_samples=nsamples,
                fp=fp,
                vcirc_mu=vcirc,
                return_log_prob=True,
                log_context=f'gal={gal_id} d4={d4_idx}',
            )
            samples.append(samp.detach().cpu().numpy())
            log_probs.append(lp.detach().cpu().numpy())

    samples = np.vstack(samples)
    log_probs = np.vstack(log_probs)
    truth = np.vstack([to_numpy(g['fid_pars']) for g in d4_set])
    map_idx = np.argmax(log_probs, axis=1)
    maps = np.vstack([samples[i, map_idx[i]] for i in range(len(map_idx))])
    diff = maps - truth
    return diff, truth, maps


def main():
    setup_logging()
    args = parse_args()

    data_dir = resolve_path(BASE_DATASETS_DIR, args.dataset)
    samp_dir = resolve_path(BASE_SAMPLES_DIR, args.sample_set)
    cache_root = resolve_path('', args.cache_root)
    stem = args.stem
    epoch = args.epoch
    nsamples = args.nsamples

    if args.mode == 1:
        model_mode = 2
        use_vcirc = True
    else:
        model_mode = 1
        use_vcirc = False

    model_name = os.path.basename(os.path.normpath(stem))
    model_cfg = load_model_config(model_name, allow_fallback_current=True)
    config.set_model_config(model_cfg)
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    total_partitions = infer_total_partitions(args)
    partition_label = build_partition_label(args.i, total_partitions)
    if use_vcirc:
        partition_label += '_tf'

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Dataset directory not found: {data_dir}')

    fid_pars_phys = None
    if use_vcirc:
        if not os.path.exists(samp_dir):
            raise FileNotFoundError(f'Sample set not found: {samp_dir}')
        fid_pars_phys = pd.read_csv(samp_dir)
        if 'vcirc' not in fid_pars_phys.columns:
            raise ValueError('vcirc column missing from sample CSV.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    profile = bool(args.profile)
    use_compile = args.use_compile
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
    use_optimization = args.use_optimization
    if args.mode == 1 and use_optimization is None:
        use_optimization = False
    if use_optimization is False:
        if use_amp:
            logging.info('Disabling AMP via --no-use-optimization.')
        use_amp = False
        if use_compile is None or use_compile:
            logging.info('Disabling torch.compile via --no-use-optimization.')
        use_compile = False
        if channels_last:
            logging.info('Disabling channels_last via --no-use-optimization.')
        channels_last = False

    model_dir = join(BASE_SHARED_DIR, 'models', stem)
    model_file = join(model_dir, f'{stem}{epoch}')
    if profile:
        _sync_cuda(device)
        start = time.perf_counter()
    model = load_model(
        mode=model_mode,
        path=model_file,
        strict=True,
        assign=True,
        device=device,
        use_compile=use_compile,
        compile_mode=compile_mode,
        compile_backend=compile_backend,
        channels_last=channels_last,
    )
    if profile:
        _sync_cuda(device)
        logging.info('Timing: load model took %.2fs', time.perf_counter() - start)
    model.mode = model_mode

    cache_root = os.path.normpath(cache_root)
    os.makedirs(cache_root, exist_ok=True)
    output_dirs = ensure_output_dirs(cache_root, model_name, dataset_name)
    logging.info('Using cache output root: %s', join(cache_root, model_name, dataset_name))
    logging.info('Output data type folders: %s', ','.join(sorted(output_dirs.keys())))
    logging.info('Partition %s started', partition_label)

    test_ds = pxt.TorchDataset(data_dir)
    total_gals = len(test_ds)
    start, end = compute_partition_bounds(total_gals, args.i, total_partitions)
    logging.info('Processing galaxies [%s, %s) out of %s total', start, end, total_gals)

    diffs = []
    truths = []
    preds = []
    iterator = tqdm(range(start, end), desc="D4 diffs") if end > start else range(start, end)
    if profile:
        _sync_cuda(device)
        start = time.perf_counter()
    for gal_id in iterator:
        gal = test_ds[gal_id]
        diff, truth, maps = d4_diff(
            gal_id,
            gal,
            model,
            device,
            fid_pars_phys,
            use_vcirc,
            nsamples,
            channels_last=channels_last,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            inference_mode=inference_mode,
        )
        diffs.append(diff)
        truths.append(truth)
        preds.append(maps)
    if profile:
        _sync_cuda(device)
        logging.info('Timing: sampling loop took %.2fs', time.perf_counter() - start)

    diffs = np.stack(diffs, axis=0)
    truths = np.stack(truths, axis=0)
    preds = np.stack(preds, axis=0)

    saved_files = {}
    diff_path = get_cache_path(cache_root, model_name, dataset_name, 'd4_diff', f'{partition_label}.npy')
    np.save(diff_path, diffs)
    saved_files['d4_diff'] = diff_path
    logging.info('Saved d4_diff: %s', diff_path)

    truth_path = get_cache_path(cache_root, model_name, dataset_name, 'd4_truth', f'{partition_label}.npy')
    np.save(truth_path, truths)
    saved_files['d4_truth'] = truth_path
    logging.info('Saved d4_truth: %s', truth_path)

    pred_path = get_cache_path(cache_root, model_name, dataset_name, 'd4_pred', f'{partition_label}.npy')
    np.save(pred_path, preds)
    saved_files['d4_pred'] = pred_path
    logging.info('Saved d4_pred: %s', pred_path)

    manifest = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'partition_index': args.i,
        'total_partitions': total_partitions,
        'partition_label': partition_label,
        'mode': args.mode,
        'model_mode': model_mode,
        'nsamples': nsamples,
        'galaxy_range': {'start': start, 'end': end},
        'paths': {
            key: os.path.relpath(path, join(cache_root, model_name, dataset_name))
            for key, path in saved_files.items()
        },
        'status': 'success',
        'created_at_utc': now_utc_iso(),
        'args': {
            'stem': args.stem,
            'dataset': args.dataset,
            'sample_set': args.sample_set,
            'cache_root': cache_root,
        },
    }
    manifest_path = get_cache_path(cache_root, model_name, dataset_name, 'meta', f'{partition_label}.json')
    with open(manifest_path, 'w', encoding='ascii') as fp:
        json.dump(manifest, fp, indent=2)
    logging.info('Saved meta: %s', manifest_path)
    logging.info('Partition %s completed successfully', partition_label)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.exception('d4_diffs failed: %s', exc)
        raise
