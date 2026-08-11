#!/usr/bin/env python3
"""Apply TF inference treatments to a reusable raw posterior candidate bank.

The input must come from ``[scr]_tf_analysis.py --mode 1``. Processing is one
partition at a time with memory mapping; the large sample bank is never copied
in full. The three treatments are:

* none: raw flow posterior;
* multiply: multiply by a common external TF prior;
* replace: multiply by the TF prior and divide by an approximate sampled
  vcirc marginal (the historical inference-time replacement idea).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank-root', type=Path, required=True)
    parser.add_argument('--partition', type=int, required=True)
    parser.add_argument('--nparts', type=int, default=10)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--slope', type=float, default=-7.22)
    parser.add_argument('--intercept', type=float, default=36.0)
    parser.add_argument('--scatter-dex', type=float, default=0.1)
    parser.add_argument('--kde-bins', type=int, default=256)
    parser.add_argument('--vcirc-index', type=int, default=5)
    return parser.parse_args()


def snr_to_app_mag(snr, depth=23.4):
    return depth + 2.5 * np.log10(5.0) - 2.5 * np.log10(snr)


def tf_log_prob(vcirc, snr, slope, intercept, scatter_dex):
    mag = snr_to_app_mag(snr)
    mu = 10.0 ** ((mag - intercept) / slope)
    sigma_mag = 1.086 / max(float(snr), 1e-8)
    sigma_dex = math.sqrt(scatter_dex**2 + (sigma_mag / slope)**2)
    sigma_ln = sigma_dex * math.log(10.0)
    safe_v = np.maximum(vcirc, 1e-8)
    return (
        -np.log(safe_v)
        - math.log(sigma_ln * math.sqrt(2.0 * math.pi))
        - 0.5 * ((np.log(safe_v) - math.log(mu)) / sigma_ln) ** 2
    )


def approximate_kde_log_prob(values, bins):
    low, high = float(np.min(values)), float(np.max(values))
    if not np.isfinite(low + high) or high <= low:
        return np.zeros_like(values, dtype=np.float64)
    count, edges = np.histogram(values, bins=bins, range=(low, high))
    width = edges[1] - edges[0]
    std = max(float(np.std(values)), 1e-8)
    bandwidth = 1.06 * std * len(values) ** (-0.2)
    smooth = gaussian_filter1d(count.astype(np.float64), max(bandwidth / width, 0.5))
    density = np.maximum(smooth / (len(values) * width), np.finfo(float).tiny)
    index = np.clip(np.searchsorted(edges, values, side='right') - 1, 0, bins - 1)
    return np.log(density[index])


def normalized_weights(log_weight):
    finite = np.isfinite(log_weight)
    if not finite.any():
        return np.full(len(log_weight), 1.0 / len(log_weight))
    shifted = np.where(finite, log_weight - np.max(log_weight[finite]), -np.inf)
    weight = np.exp(shifted)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(len(log_weight), 1.0 / len(log_weight))
    return weight / total


def main():
    args = parse_args()
    label = f'part{args.partition}of{args.nparts}'
    samples = np.load(args.bank_root / 'sample' / f'{label}.npy', mmap_mode='r')
    log_prob = np.load(args.bank_root / 'log_prob' / f'{label}.npy', mmap_mode='r')
    snr = np.load(args.bank_root / 'snr' / f'{label}.npy', mmap_mode='r')
    truth = np.load(args.bank_root / 'truth' / f'{label}.npy', mmap_mode='r')
    if samples.shape[0] != 1 or log_prob.shape[0] != 1:
        raise ValueError('Candidate bank must contain exactly one raw mode')
    n_gal, _, n_feature = samples.shape[1:]
    estimates = np.empty((3, n_gal, 2, n_feature), dtype=np.float32)
    ess = np.empty((3, n_gal), dtype=np.float32)
    treatment_names = ('none', 'multiply', 'replace')
    for i in range(n_gal):
        candidate = np.asarray(samples[0, i], dtype=np.float64)
        raw_lp = np.asarray(log_prob[0, i], dtype=np.float64)
        prior_lp = tf_log_prob(
            candidate[:, args.vcirc_index], snr[i], args.slope,
            args.intercept, args.scatter_dex,
        )
        marginal_lp = approximate_kde_log_prob(candidate[:, args.vcirc_index], args.kde_bins)
        log_weights = (
            np.zeros(len(candidate), dtype=np.float64),
            prior_lp,
            prior_lp - marginal_lp,
        )
        for j, log_weight in enumerate(log_weights):
            weight = normalized_weights(log_weight)
            estimates[j, i, 0] = candidate[np.argmax(raw_lp + log_weight)]
            estimates[j, i, 1] = np.sum(candidate * weight[:, None], axis=0)
            ess[j, i] = 1.0 / np.sum(weight**2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    np.save(args.output_root / f'estimates_{label}.npy', estimates)
    np.save(args.output_root / f'ess_{label}.npy', ess)
    np.save(args.output_root / f'truth_{label}.npy', np.asarray(truth))
    metadata = {
        'bank_root': str(args.bank_root),
        'partition': args.partition,
        'nparts': args.nparts,
        'treatments': treatment_names,
        'estimate_axis': ['MAP', 'weighted_mean'],
        'tf': {
            'slope': args.slope,
            'intercept': args.intercept,
            'scatter_dex': args.scatter_dex,
        },
        'replace_kde': f'{args.kde_bins}-bin Gaussian-smoothed histogram',
        'warning': 'Low ESS means the raw bank needs more candidates for that galaxy.',
    }
    (args.output_root / f'meta_{label}.json').write_text(json.dumps(metadata, indent=2))
    print(f'Wrote {label}; median ESS: ' + ', '.join(
        f'{name}={np.median(ess[j]):.1f}' for j, name in enumerate(treatment_names)
    ))


if __name__ == '__main__':
    main()
