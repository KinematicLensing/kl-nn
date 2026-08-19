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
    load_v2_observation_metadata,
    checkpoint_image_noise_sigma,
    checkpoint_spectral_reference_line_norm,
    _resolve_amp_dtype,
    seed_everything,
)
from networks import KLNPE
import config
from model_registry import load_model_config
from utils import (
    denormalize,
    resolve_feature_index,
)
from data import gaussian_psf_noise_equivalent_pixels, rot_90_param_only

BASE_SHARED_DIR = '/ocean/projects/phy250048p/shared'
BASE_DATASETS_DIR = join(BASE_SHARED_DIR, 'datasets')
BASE_SAMPLES_DIR = join(BASE_SHARED_DIR, 'samples')
V2_OBSERVATION_ARRAY_TYPES = (
    'image_snr',
    'spectral_quality',
    'spectral_noise_scale',
    'rmag_obs',
    'rmag_sigma',
)
TF_DIAGNOSTIC_ARRAY_TYPES = (
    'tf_effective_sample_size',
    'tf_effective_sample_fraction',
    'tf_max_normalized_weight',
    'tf_candidate_log_normalizer',
)
DATA_TYPES = (
    'sample',
    'log_prob',
    'snr',
    *V2_OBSERVATION_ARRAY_TYPES,
    *TF_DIAGNOSTIC_ARRAY_TYPES,
    'truth',
    'map_estimates',
    'mean_estimates',
    'meta',
)

ANALYSIS_STREAM_OFFSETS = {
    'image_noise': 101,
    'spectral_noise': 211,
    'magnitude_observation': 307,
    'spectral_quality': 401,
}


def analysis_stream_seeds(partition_seed):
    """Derive documented, independent RNG streams for one partition."""
    modulus = 2**63 - 1
    return {
        name: (int(partition_seed) + offset) % modulus
        for name, offset in ANALYSIS_STREAM_OFFSETS.items()
    }

def validate_analysis_observation_args(args, observation, train_mode):
    """Validate version-specific analysis choices and resolve TF inference."""
    observation_model_version = int(observation.get("model_version", 1))
    tf_inference = (
        None if args.tf_inference == "none" else args.tf_inference
    )
    if observation_model_version == 2:
        if args.mode != 1 or int(train_mode) != 1:
            raise ValueError(
                "Observation model v2 analysis requires a mode-1 base posterior"
            )
        if args.conform_to_tf:
            raise ValueError(
                "Observation model v2 forbids --conform-to-tf; use "
                "--tf-inference prior_replacement to replace the base prior"
            )
        if args.cached_snrs_path is not None:
            raise ValueError(
                "Observation model v2 derives image SNR from archived "
                "rmag_true and does not accept --cached-snrs-path"
            )
    elif tf_inference is not None:
        raise ValueError(
            "--tf-inference prior_replacement requires observation model v2"
        )
    return observation_model_version, tf_inference


def summarize_tf_diagnostic_arrays(observation_metadata):
    """Return compact JSON-safe summaries of cached TF diagnostics."""
    if observation_metadata is None:
        return {}
    summary = {}
    for name in TF_DIAGNOSTIC_ARRAY_TYPES:
        if name not in observation_metadata:
            continue
        values = np.asarray(observation_metadata[name], dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(
                f"TF diagnostic {name!r} must be a finite one-dimensional array"
            )
        summary[name] = {
            'min': float(np.min(values)),
            'median': float(np.median(values)),
            'mean': float(np.mean(values)),
            'max': float(np.max(values)),
        }
    return summary


def resolve_analysis_snr_cache(
    returned_snr,
    observation_metadata,
    *,
    observation_model_version,
):
    """Return the report-facing SNR without exposing a v2 latent noise target."""

    if int(observation_model_version) == 2:
        if observation_metadata is None or "image_snr" not in observation_metadata:
            raise RuntimeError(
                "Simulator-v2 sampling must return observed image_snr metadata"
            )
        values = np.asarray(observation_metadata["image_snr"])
    else:
        values = np.asarray(returned_snr)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError("Analysis SNR cache must be a finite one-dimensional array")
    return values


def build_observation_provenance(
    observation,
    *,
    observation_model_version,
    checkpoint_image_noise_sigma,
    checkpoint_reference_line_norm,
    stream_seeds,
    tf_diagnostic_summary=None,
):
    """Build the JSON-safe observation provenance stored with each partition."""
    return {
        'model_version': observation_model_version,
        'fiber_layout': observation.get('fiber_layout'),
        'metadata_source': (
            'validated LMDB v2 observation and instrument schema'
            if observation_model_version == 2
            else 'legacy-v1'
        ),
        'context_fields': (
            list(observation.get('context_fields', ()))
            if observation_model_version == 2
            else []
        ),
        'image_band': observation.get('image_band'),
        'target_line': observation.get('target_line'),
        'halpha_flux_min': (
            observation.get('halpha_flux_min')
            if observation_model_version == 2
            else None
        ),
        'halpha_flux_max': (
            observation.get('halpha_flux_max')
            if observation_model_version == 2
            else None
        ),
        'halpha_flux_distribution': (
            observation.get('halpha_flux_distribution')
            if observation_model_version == 2
            else None
        ),
        'halpha_flux_units': (
            observation.get('halpha_flux_units')
            if observation_model_version == 2
            else None
        ),
        'image_depth_5sigma_mag': observation.get(
            'image_depth_5sigma_mag'
        ),
        'image_depth_calibration': 'Gaussian-PSF-equivalent',
        'image_reference_psf_fwhm_arcsec': observation.get(
            'image_reference_psf_fwhm_arcsec'
        ),
        'image_pixel_scale_arcsec': observation.get(
            'image_pixel_scale_arcsec'
        ),
        'image_reference_noise_equivalent_pixels': (
            gaussian_psf_noise_equivalent_pixels(
                observation['image_reference_psf_fwhm_arcsec'],
                observation['image_pixel_scale_arcsec'],
            )
            if observation_model_version == 2
            else None
        ),
        'checkpoint_image_noise_sigma': (
            checkpoint_image_noise_sigma
            if observation_model_version == 2
            else None
        ),
        'spectral_quality_min': observation.get('spectral_quality_min'),
        'spectral_quality_max': observation.get('spectral_quality_max'),
        'spectral_quality_distribution': observation.get(
            'spectral_quality_distribution'
        ),
        'spectral_units': observation.get('spectral_units'),
        'center_fiber_index': observation.get('center_fiber_index'),
        'center_exposure_s': observation.get('center_exposure_s'),
        'offset_exposure_s': observation.get('offset_exposure_s'),
        'snr_cache_semantics': (
            'observed_catalog_flux_snr'
            if observation_model_version == 2
            else 'legacy_injected_snr'
        ),
        'pixel_noise_target_snr_cached': observation_model_version != 2,
        'checkpoint_spectral_reference_line_norm': (
            checkpoint_reference_line_norm
            if observation_model_version == 2
            else None
        ),
        'rng_stream_seeds': (
            dict(stream_seeds) if observation_model_version == 2 else {}
        ),
        'tf_prior_replacement_diagnostics': (
            dict(tf_diagnostic_summary or {})
            if observation_model_version == 2
            else {}
        ),
    }


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
        '--network-source',
        choices=('archived', 'current'),
        default='archived',
        help='Instantiate the checkpoint with its archived source or the current repository source.',
    )
    parser.add_argument(
        '--cached-snrs-path',
        type=str,
        default=None,
        help='Path to .npy file containing pre-saved SNR values to use instead of generating new ones.',
    )
    parser.add_argument(
        '--mode',
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            'Sampling mode: 1 is the base flow; 2 is the legacy KDE-weighted '
            'TF path (observation model v1 only).'
        ),
    )
    parser.add_argument(
        '--tf-inference',
        choices=('none', 'prior_replacement'),
        default='none',
        help=(
            'Optional explicit TF prior replacement for a simulator-v2 '
            'mode-1 base posterior.'
        ),
    )
    parser.add_argument(
        '--cache-tag',
        type=str,
        default='',
        help='Optional suffix for the output dataset cache name (for experiment isolation).',
    )
    parser.add_argument(
        '--matched-group-size',
        type=int,
        default=1,
        help='Reuse scalar SNR and injected-noise realization within consecutive groups.',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for SNR, noise, TF, and posterior sampling draws.",
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


def resolve_partition_range(partition_idx, galaxies_per_partition, dataset_size):
    '''Return the half-open dataset range for one deterministic partition.'''
    if galaxies_per_partition <= 0:
        raise ValueError('ngals must be positive')
    galaxy_start = partition_idx * galaxies_per_partition
    galaxy_end = galaxy_start + galaxies_per_partition
    if galaxy_end > dataset_size:
        raise ValueError(
            f'Partition range [{galaxy_start}, {galaxy_end}) exceeds dataset '
            f'size {dataset_size}.'
        )
    return galaxy_start, galaxy_end


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

def summarize_posterior_samples(samples, feature_names):
    """Return 16th, mean, and 84th estimates with circular theta handling."""
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("samples must have shape (samples, len(feature_names))")
    summary = np.stack(
        (
            np.percentile(values, 16, axis=0),
            np.mean(values, axis=0),
            np.percentile(values, 84, axis=0),
        ),
        axis=0,
    )
    if "theta_int" in feature_names:
        theta_idx = feature_names.index("theta_int")
        theta = values[:, theta_idx]
        center = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
        residual = np.arctan2(np.sin(theta - center), np.cos(theta - center))
        residual_bounds = np.percentile(residual, (16, 84))
        circular = np.array(
            (center + residual_bounds[0], center, center + residual_bounds[1])
        )
        summary[:, theta_idx] = (circular + np.pi) % (2.0 * np.pi) - np.pi
    return summary


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
    observation_model_version, tf_inference = (
        validate_analysis_observation_args(
            args,
            config.observation,
            config.train.get("mode", 1),
        )
    )

    dataset_name = os.path.basename(os.path.normpath(data_dir))
    if observation_model_version == 2:
        dataset_name += (
            "_tf_prior_replacement"
            if tf_inference == "prior_replacement"
            else "_base_prior"
        )
    elif args.conform_to_tf:
        dataset_name += '_tf_conformed'
    if args.cache_tag:
        dataset_name += f'_{args.cache_tag.strip("_")}'
    total_partitions = infer_total_partitions(args)
    partition_label = build_partition_label(args.i, total_partitions)
    rng_seed = (int(args.seed) + 1_000_003 * int(args.i)) % (2**32)
    stream_seeds = analysis_stream_seeds(rng_seed)
    seed_everything(
        rng_seed,
        deterministic=bool(config.train.get("deterministic", False)),
    )
    nfeatures = config.train['feature_number']
    feature_names = config.train['feature_names']

    if len(feature_names) != nfeatures:
        raise ValueError(
            f"config.train['feature_names'] length {len(feature_names)} does not match feature_number {nfeatures}."
        )
    if args.matched_group_size < 1 or args.ngals % args.matched_group_size:
        raise ValueError('matched-group-size must be positive and divide ngals')

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
            profile_start = time.perf_counter()
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
            use_archived_networks=args.network_source == 'archived',
        )
        if profile:
            _sync_cuda(device)
            logging.info(
                'Timing: load model mode %s took %.2fs',
                mode,
                time.perf_counter() - profile_start,
            )
        logging.info(
            'Loaded checkpoint using %s network source',
            args.network_source,
        )
        models[mode] = model
        return model

    if not use_separate_models:
        get_model_for_mode(args.mode)

    posterior_symmetry = config.train.get("posterior_symmetry", "none")
    if posterior_symmetry == "d4" and args.cancel_add_noise:
        raise ValueError(
            "--cancel-add-noise is redundant and incompatible with the exact D4 posterior"
        )
    if posterior_symmetry == "d4" and (nsamples <= 0 or nsamples % 8):
        raise ValueError("D4 posterior sampling requires --nsamples divisible by 8")

    # Get data loader
    test_ds = pxt.TorchDataset(data_dir)
    try:
        galaxy_start, galaxy_end = resolve_partition_range(
            args.i, args.ngals, len(test_ds)
        )
    except ValueError as exc:
        raise ValueError(f'{exc} Partition: {partition_label}.') from exc
    subset = Subset(test_ds, np.arange(galaxy_start, galaxy_end))

    rmag_true = None
    if observation_model_version == 2:
        rmag_true = load_v2_observation_metadata(
            subset,
            expected_fiber_layout=config.observation["fiber_layout"],
            device=device,
        )
        logging.info(
            "Validated %d simulator-v2 records: fiber_layout=%s, "
            "rmag_true range %.4f--%.4f",
            len(subset),
            config.observation["fiber_layout"],
            float(rmag_true.min().item()),
            float(rmag_true.max().item()),
        )

    vcirc_idx = resolve_feature_index(feature_names, 'vcirc', aliases=('v_circ',))
    vcirc_name = feature_names[vcirc_idx]
    vcirc_low, vcirc_high = config.par_ranges[vcirc_name]

    # Collect true vcirc values in normalized space and convert to km/s center of prior.
    if profile:
        _sync_cuda(device)
        profile_start = time.perf_counter()
    vcirc_true = torch.zeros((args.ngals), dtype=torch.float32, device=device)
    vcirc_iter = tqdm(range(args.ngals), desc="Collect vcirc") if args.ngals else range(0)
    for i in vcirc_iter:
        vcirc_true[i] = subset[i]['fid_pars'][vcirc_idx]
    vcirc_mu = 0.5 * (vcirc_true + 1.0) * (vcirc_high - vcirc_low) + vcirc_low

    truth = build_truth_array(subset, feature_names, progress=tqdm)
    if profile:
        _sync_cuda(device)
        logging.info(
            'Timing: prep vcirc/truth took %.2fs',
            time.perf_counter() - profile_start,
        )
    
    app_mag = None
    if observation_model_version == 2:
        from data import app_mag_to_snr
        snr_shared = app_mag_to_snr(
            rmag_true,
            band=config.observation["image_band"],
            depth_5sigma_mag=config.observation[
                "image_depth_5sigma_mag"
            ],
        )
        logging.info(
            "Derived image SNR from archived magnitude/depth: "
            "min %.4g, max %.4g, mean %.4g",
            float(snr_shared.min().item()),
            float(snr_shared.max().item()),
            float(snr_shared.mean().item()),
        )
    elif args.conform_to_tf:
        logging.info('Conforming legacy-v1 dataset to TF prior')
        from data import TFCalculator, app_mag_to_snr
        tf_calc = TFCalculator(
            slope=config.tf['slope'],
            intercept=config.tf['intercept'],
            scatter=config.tf['scatter'],
        )
        app_mag = tf_calc.sample_mag_from_vcirc(vcirc_mu)
        snr_shared = app_mag_to_snr(app_mag)
        logging.info(
            'SNR range: min %.2f, max %.2f, mean %.2f',
            snr_shared.min().item(),
            snr_shared.max().item(),
            snr_shared.mean().item(),
        )
    elif args.cached_snrs_path is not None:
        if not os.path.exists(args.cached_snrs_path):
            raise FileNotFoundError(f'Cached SNRs file not found: {args.cached_snrs_path}')
        if os.path.isdir(args.cached_snrs_path):
            snr_path = os.path.join(args.cached_snrs_path, f'{partition_label}.npy')
        else:
            snr_path = args.cached_snrs_path
        if not os.path.exists(snr_path):
            raise FileNotFoundError(f'Cached SNR partition not found: {snr_path}')
        snr_shared = torch.from_numpy(np.load(snr_path)).to(device)
        if snr_shared.shape != (args.ngals,):
            raise ValueError(f'Cached SNRs shape {snr_shared.shape} does not match expected ({args.ngals},)')
    else:
        snr_gen = torch.Generator(device=device).manual_seed(rng_seed)
        group_count = args.ngals // args.matched_group_size
        group_snr = torch.rand(group_count, generator=snr_gen, device=device) * 995 + 5
        snr_shared = torch.repeat_interleave(group_snr, args.matched_group_size)
        app_mag = None

    modes = [args.mode]
    sample_list = np.empty((len(modes), args.ngals, args.nsamples, nfeatures), dtype=np.float32)
    log_prob_list = np.empty((len(modes), args.ngals, args.nsamples), dtype=np.float32)
    if args.cancel_add_noise:
        sample_list_rot90 = np.empty((len(modes), args.ngals, args.nsamples, nfeatures), dtype=np.float32)
        log_prob_list_rot90 = np.empty((len(modes), args.ngals, args.nsamples), dtype=np.float32)

    analysis_observation_metadata = None
    checkpoint_fixed_image_noise_sigma = None
    checkpoint_reference_line_norm = None

    # Sample each requested posterior using reproducible observation streams.
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else nullcontext()
    infer_ctx = torch.inference_mode() if inference_mode else nullcontext()
    with infer_ctx, amp_ctx:
        for j, mode in enumerate(modes):
            model = (
                get_model_for_mode(mode)
                if use_separate_models
                else models[args.mode]
            )
            model.mode = mode
            posterior_owner = model
            while hasattr(posterior_owner, "_orig_mod"):
                posterior_owner = posterior_owner._orig_mod
            posterior_owner.mode = mode

            if profile:
                _sync_cuda(device)
                profile_start = time.perf_counter()
            sampling_progress_fn = functools.partial(
                sampling_progress, mode=mode
            )
            if observation_model_version == 2:
                checkpoint_fixed_image_noise_sigma = float(
                    checkpoint_image_noise_sigma(model)
                    .detach()
                    .cpu()
                    .item()
                )
                checkpoint_reference_line_norm = float(
                    checkpoint_spectral_reference_line_norm(model)
                    .detach()
                    .cpu()
                    .item()
                )
                logging.info(
                    "Using checkpoint image_noise_sigma=%.8g and "
                    "spectral_reference_line_norm=%.8g; "
                    "TF inference=%s",
                    checkpoint_fixed_image_noise_sigma,
                    checkpoint_reference_line_norm,
                    tf_inference or "none",
                )
                image_noise_gen = torch.Generator(
                    device=device
                ).manual_seed(stream_seeds["image_noise"])
                spectral_noise_gen = torch.Generator(
                    device=device
                ).manual_seed(stream_seeds["spectral_noise"])
                magnitude_gen = torch.Generator(
                    device=device
                ).manual_seed(stream_seeds["magnitude_observation"])
                spectral_quality_gen = torch.Generator(
                    device=device
                ).manual_seed(stream_seeds["spectral_quality"])
                (
                    samples,
                    log_probs,
                    SNR,
                    analysis_observation_metadata,
                ) = sample_density(
                    model,
                    subset,
                    nsamples,
                    snr=snr_shared,
                    rmag_true=rmag_true,
                    tf_inference=tf_inference,
                    image_randgen=image_noise_gen,
                    spectral_randgen=spectral_noise_gen,
                    magnitude_randgen=magnitude_gen,
                    spectral_quality_randgen=spectral_quality_gen,
                    return_log_prob=True,
                    return_observation_metadata=True,
                    apply_add_noise_cancellation=args.cancel_add_noise,
                    device=device,
                    channels_last=channels_last,
                    matched_group_size=args.matched_group_size,
                    noise_seed=stream_seeds["image_noise"],
                    spectral_noise_seed=stream_seeds["spectral_noise"],
                    magnitude_seed=stream_seeds["magnitude_observation"],
                    spectral_quality_seed=stream_seeds["spectral_quality"],
                    progress=sampling_progress_fn,
                )
            else:
                # Preserve the historical shared-SNR/shared-noise path exactly.
                noise_gen = torch.Generator(device=device).manual_seed(rng_seed)
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
                    matched_group_size=args.matched_group_size,
                    noise_seed=rng_seed + args.i * args.ngals,
                    progress=sampling_progress_fn,
                )
            if profile:
                _sync_cuda(device)
                logging.info(
                    'Timing: sample_density mode %s took %.2fs',
                    mode,
                    time.perf_counter() - profile_start,
                )
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

    report_snr = resolve_analysis_snr_cache(
        SNR,
        analysis_observation_metadata,
        observation_model_version=observation_model_version,
    )
    if report_snr.shape != (args.ngals,):
        raise RuntimeError(
            f"Unexpected report-facing SNR shape {report_snr.shape}; expected "
            f"({args.ngals},)"
        )
    snr_path = get_cache_path(cache_dir, model_name, dataset_name, 'snr', f'{partition_label}.npy')
    np.save(snr_path, report_snr)
    saved_files['snr'] = snr_path
    logging.info(
        'Saved snr (%s): %s',
        (
            'observed catalog flux SNR'
            if observation_model_version == 2
            else 'legacy injected SNR'
        ),
        snr_path,
    )

    if observation_model_version == 2:
        if analysis_observation_metadata is None:
            raise RuntimeError(
                "Simulator-v2 sampling did not return observation metadata"
            )
        cache_types = list(V2_OBSERVATION_ARRAY_TYPES)
        if tf_inference == "prior_replacement":
            cache_types.extend(TF_DIAGNOSTIC_ARRAY_TYPES)
        for data_type in cache_types:
            values = np.asarray(analysis_observation_metadata[data_type])
            if values.shape != (args.ngals,):
                raise RuntimeError(
                    f"Unexpected {data_type} shape {values.shape}; expected "
                    f"({args.ngals},)"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(
                    f"Observation cache {data_type!r} contains non-finite values"
                )
            output_path = get_cache_path(
                cache_dir,
                model_name,
                dataset_name,
                data_type,
                f'{partition_label}.npy',
            )
            np.save(output_path, values)
            saved_files[data_type] = output_path
            logging.info('Saved %s: %s', data_type, output_path)

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
                counter_samples = rot_90_param_only(samples_rot90, reverse=True)

            # MAP estimate
            if args.cancel_add_noise:
                map_idx = np.argmax(log_probs)
                map_idx_rot90 = np.argmax(log_probs_rot90)
                map_estimates[j, i] = samples[map_idx]
                map_estimates[j, i, :2] = 0.5 * (
                    samples[map_idx, :2] + counter_samples[map_idx_rot90, :2]
                )
            else:
                map_idx = np.argmax(log_probs)
                map_estimates[j, i] = samples[map_idx]

            # Mean estimate with bounds
            mean_estimates[j, i] = summarize_posterior_samples(
                samples, feature_names
            )
            if args.cancel_add_noise:
                shear_samples = np.concatenate(
                    [samples[:, :2], counter_samples[:, :2]], axis=0
                )
                mean_estimates[j, i, 0, :2] = np.percentile(shear_samples, 16, axis=0)
                mean_estimates[j, i, 1, :2] = np.mean(shear_samples, axis=0)
                mean_estimates[j, i, 2, :2] = np.percentile(shear_samples, 84, axis=0)
    
    map_path = get_cache_path(cache_dir, model_name, dataset_name, 'map_estimates', f'{partition_label}.npy')
    np.save(map_path, map_estimates)
    saved_files['map_estimates'] = map_path
    logging.info('Saved map_estimates: %s', map_path)

    mean_path = get_cache_path(cache_dir, model_name, dataset_name, 'mean_estimates', f'{partition_label}.npy')
    np.save(mean_path, mean_estimates)
    saved_files['mean_estimates'] = mean_path
    logging.info('Saved mean_estimates: %s', mean_path)

    tf_diagnostic_summary = summarize_tf_diagnostic_arrays(
        analysis_observation_metadata
    )

    manifest = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'partition_index': args.i,
        'total_partitions': total_partitions,
        'partition_label': partition_label,
        'ngals': args.ngals,
        'nsamples': args.nsamples,
        'galaxy_range': {'start': galaxy_start, 'end': galaxy_end},
        'paths': {
            key: os.path.relpath(path, join(cache_dir, model_name, dataset_name))
            for key, path in saved_files.items()
        },
        'status': 'success',
        'created_at_utc': now_utc_iso(),
        'observation': build_observation_provenance(
            config.observation,
            observation_model_version=observation_model_version,
            checkpoint_image_noise_sigma=(
                checkpoint_fixed_image_noise_sigma
            ),
            checkpoint_reference_line_norm=(
                checkpoint_reference_line_norm
            ),
            stream_seeds=stream_seeds,
            tf_diagnostic_summary=tf_diagnostic_summary,
        ),
        'args': {
            'stem': args.stem,
            'epoch': args.epoch,
            'dataset': args.dataset,
            'sample_set': args.sample_set,
            'mode': args.mode,
            'tf_inference': tf_inference or 'none',
            'conform_to_tf': args.conform_to_tf,
            'network_source': args.network_source,
            'cache_tag': args.cache_tag,
            'cached_snrs_path': args.cached_snrs_path,
            'matched_group_size': args.matched_group_size,
            "seed": args.seed,
            "partition_seed": rng_seed,
            "posterior_symmetry": posterior_symmetry,
            'cancel_add_noise': args.cancel_add_noise,
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
