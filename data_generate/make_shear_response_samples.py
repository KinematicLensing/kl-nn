#!/usr/bin/env python3
"""Create matched five-point finite-shear simulator inputs and a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PARAMETERS = [
    'g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr',
    'dx_disk', 'dy_disk', 'dx_spec', 'dy_spec',
]
STATES = (
    ('zero', 0.0, 0.0),
    ('g1_plus', 1.0, 0.0),
    ('g1_minus', -1.0, 0.0),
    ('g2_plus', 0.0, 1.0),
    ('g2_minus', 0.0, -1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--nbase', type=int, default=1000)
    parser.add_argument('--delta-g', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1729)
    return parser.parse_args()


def main():
    args = parse_args()
    source = pd.read_csv(args.input)
    unnamed = [column for column in source if column.startswith('Unnamed:')]
    if unnamed:
        source = source.drop(columns=unnamed)
    missing = [name for name in PARAMETERS[:8] if name not in source]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    for name in PARAMETERS[8:]:
        if name not in source:
            source[name] = 0.0
    if not (0 < args.nbase <= len(source)):
        raise ValueError(f'nbase must be in [1, {len(source)}]')
    if not (0 < args.delta_g <= 0.1):
        raise ValueError('delta-g must be in (0, 0.1]')

    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(len(source), size=args.nbase, replace=False))
    base = source.iloc[chosen].reset_index(drop=True)
    sample_rows, manifest_rows = [], []
    output_id = 0
    for base_id, (_, row) in enumerate(base.iterrows()):
        nuisance = row[PARAMETERS].astype(float).to_dict()
        source_id = int(chosen[base_id])
        for state, g1_sign, g2_sign in STATES:
            values = dict(nuisance)
            values['g1'] = g1_sign * args.delta_g
            values['g2'] = g2_sign * args.delta_g
            sample_rows.append({'ID': output_id, **values})
            manifest_rows.append({
                'ID': output_id,
                'base_id': base_id,
                'source_row': source_id,
                'state': state,
                'g1': values['g1'],
                'g2': values['g2'],
            })
            output_id += 1

    samples = pd.DataFrame(sample_rows, columns=['ID', *PARAMETERS])
    manifest = pd.DataFrame(manifest_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output, index=False)
    manifest.to_csv(args.manifest, index=False)
    print(f'Wrote {len(samples)} simulations ({args.nbase} matched bases)')
    print(f'Samples: {args.output}')
    print(f'Manifest: {args.manifest}')


if __name__ == '__main__':
    main()
