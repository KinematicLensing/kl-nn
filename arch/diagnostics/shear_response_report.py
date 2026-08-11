#!/usr/bin/env python3
"""Measure and hold out a matched finite-shear response calibration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PART_RE = re.compile(r'part(\d+)of(\d+)\.npy$')
STATE_ORDER = ('zero', 'g1_plus', 'g1_minus', 'g2_plus', 'g2_minus')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode-index', type=int, default=0)
    parser.add_argument('--estimator', choices=('map', 'mean'), default='mean')
    parser.add_argument('--calibration-fraction', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=31415)
    return parser.parse_args()


def load_parts(directory, axis):
    found = []
    for path in directory.glob('part*of*.npy'):
        match = PART_RE.match(path.name)
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    found.sort()
    if not found or [x[0] for x in found] != list(range(found[0][1])):
        raise ValueError(f'Incomplete partitions in {directory}')
    return np.concatenate([np.load(x[2], mmap_mode='r') for x in found], axis=axis)


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest).sort_values('ID')
    truth = load_parts(args.cache_root / 'truth', axis=0)
    if args.estimator == 'map':
        estimate = load_parts(args.cache_root / 'map_estimates', axis=1)[args.mode_index, :, :2]
    else:
        estimate = load_parts(args.cache_root / 'mean_estimates', axis=1)[args.mode_index, :, 1, :2]
    if len(manifest) != len(estimate):
        raise ValueError('Manifest and estimate lengths differ')
    state_code = {name: i for i, name in enumerate(STATE_ORDER)}
    state = manifest['state'].map(state_code).to_numpy()
    base_id = manifest['base_id'].to_numpy()
    unique_base = np.unique(base_id)
    cube = np.empty((len(unique_base), len(STATE_ORDER), 2), dtype=np.float64)
    true_cube = np.empty_like(cube)
    for i, base in enumerate(unique_base):
        for name, code in state_code.items():
            where = np.flatnonzero((base_id == base) & (state == code))
            if len(where) != 1:
                raise ValueError(f'Expected one {name} row for base {base}, found {len(where)}')
            cube[i, code] = estimate[where[0]]
            true_cube[i, code] = truth[where[0], :2]
    delta = float(true_cube[0, state_code['g1_plus'], 0])
    response = np.empty((len(unique_base), 2, 2), dtype=np.float64)
    response[:, :, 0] = (
        cube[:, state_code['g1_plus']] - cube[:, state_code['g1_minus']]
    ) / (2 * delta)
    response[:, :, 1] = (
        cube[:, state_code['g2_plus']] - cube[:, state_code['g2_minus']]
    ) / (2 * delta)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(unique_base))
    split = int(round(args.calibration_fraction * len(order)))
    calibration, holdout = order[:split], order[split:]
    calibration_response = np.mean(response[calibration], axis=0)
    calibration_additive = np.mean(cube[calibration, state_code['zero']], axis=0)
    inverse_response = np.linalg.inv(calibration_response)
    corrected = np.einsum('ij,bsj->bsi', inverse_response, cube - calibration_additive)
    holdout_response = np.mean(response[holdout], axis=0)
    corrected_response = inverse_response @ holdout_response
    holdout_additive = np.mean(corrected[holdout, state_code['zero']], axis=0)
    additive_se = np.std(
        corrected[holdout, state_code['zero']], axis=0, ddof=1
    ) / np.sqrt(len(holdout))
    result = {
        'estimator': args.estimator,
        'nbase': len(unique_base),
        'n_calibration': len(calibration),
        'n_holdout': len(holdout),
        'delta_g': delta,
        'calibration_additive': calibration_additive.tolist(),
        'calibration_response': calibration_response.tolist(),
        'raw_holdout_response': holdout_response.tolist(),
        'corrected_holdout_response': corrected_response.tolist(),
        'corrected_holdout_additive': holdout_additive.tolist(),
        'corrected_holdout_additive_se': additive_se.tolist(),
        'targets': {'abs_c': 1e-4, 'abs_m': 1e-2},
        'note': 'Calibration and validation base galaxies are disjoint.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == '.html':
        payload = json.dumps(result, indent=2)
        args.output.write_text(
            '<!doctype html><meta charset="utf-8"><title>Shear response pilot</title>'
            '<style>body{font-family:system-ui;max-width:900px;margin:2rem auto}'
            'pre{background:#f5f5f5;padding:1rem;overflow:auto}</style>'
            '<h1>Matched finite-shear response pilot</h1>'
            '<p>The calibration and holdout galaxy sets are disjoint. Response '
            'is measured with central finite differences from matched simulations.</p>'
            f'<pre>{payload}</pre>'
        )
    else:
        args.output.write_text(json.dumps(result, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
