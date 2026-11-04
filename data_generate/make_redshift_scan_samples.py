#!/usr/bin/env python3
"""Create the redshift-scan simulator inputs (same bases at several redshifts).

Each base row is copied unchanged to every redshift; only the ``redshift``
column differs. Rows are redshift-major (``ID = nbase * k + base_id``), so
part ``k + 1`` of a chunk-``nbase`` generation holds redshift ``k``. The
targeting seed is the source row's production seed, so the measured position
angle and fiber positions are the same at every redshift and match the source
data set.

No ``.manifest.json`` sidecar is written: those enforce z = 0.3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .fiber_targeting import targeting_seed
    from .make_shear_response_samples import AUXILIARY_COLUMNS, PARAMETERS
    from .observation_schema import (
        FIBER_LAYOUT_COLUMN,
        MEASURED_AXIS_FIBER_LAYOUT,
        REDSHIFT_COLUMN,
        TARGETING_SEED_COLUMN,
        halpha_window,
    )
except ImportError:  # Support direct execution from data_generate/.
    from fiber_targeting import targeting_seed
    from make_shear_response_samples import AUXILIARY_COLUMNS, PARAMETERS
    from observation_schema import (
        FIBER_LAYOUT_COLUMN,
        MEASURED_AXIS_FIBER_LAYOUT,
        REDSHIFT_COLUMN,
        TARGETING_SEED_COLUMN,
        halpha_window,
    )

SAMPLES = Path("/ocean/projects/phy250048p/shared/samples")
DEFAULT_REDSHIFTS = "0.14,0.18,0.22,0.26,0.30,0.34,0.38,0.42,0.46,0.50"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=SAMPLES / "test_100k_simv3_cosi_xu3_tf_measaxis.csv")
    parser.add_argument("--output", type=Path,
                        default=SAMPLES / "paper_tests_G_redshift_scan_xu3like_1k_measaxis.csv")
    parser.add_argument("--manifest", type=Path,
                        default=SAMPLES / "paper_tests_G_redshift_scan_xu3like_1k_measaxis_manifest.csv")
    parser.add_argument("--nbase", type=int, default=100)
    parser.add_argument("--redshifts", default=DEFAULT_REDSHIFTS)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def build_scan(source: pd.DataFrame, *, nbase: int, redshifts, seed: int):
    """Return (samples, manifest) data frames for the redshift scan."""

    redshifts = [float(z) for z in redshifts]
    for z in redshifts:
        halpha_window(z)  # validates the redshift
    if len(set(redshifts)) != len(redshifts):
        raise ValueError("redshifts must be distinct")
    missing = [name for name in ("ID", *PARAMETERS, *AUXILIARY_COLUMNS) if name not in source]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not (0 < nbase <= len(source)):
        raise ValueError(f"nbase must be in [1, {len(source)}]")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(source), size=nbase, replace=False))
    base = source.iloc[chosen].reset_index(drop=True)
    sample_rows, manifest_rows = [], []
    for k, z in enumerate(redshifts):
        for base_id, row in base.iterrows():
            output_id = nbase * k + base_id
            source_id = int(row["ID"])
            values = {name: float(row[name]) for name in PARAMETERS}
            metadata = {name: row[name] for name in AUXILIARY_COLUMNS}
            metadata[FIBER_LAYOUT_COLUMN] = MEASURED_AXIS_FIBER_LAYOUT
            sample_rows.append({
                "ID": output_id, **values, **metadata,
                REDSHIFT_COLUMN: z,
                TARGETING_SEED_COLUMN: targeting_seed(source_id),
            })
            manifest_rows.append({
                "ID": output_id, "base_id": base_id, "source_id": source_id,
                "source_row": int(chosen[base_id]), REDSHIFT_COLUMN: z,
            })
    columns = ["ID", *PARAMETERS, *AUXILIARY_COLUMNS, REDSHIFT_COLUMN, TARGETING_SEED_COLUMN]
    return pd.DataFrame(sample_rows, columns=columns), pd.DataFrame(manifest_rows)


def main(argv=None):
    args = parse_args(argv)
    source = pd.read_csv(args.input, float_precision="round_trip")
    samples, manifest = build_scan(
        source,
        nbase=args.nbase,
        redshifts=args.redshifts.split(","),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output, index=False)
    manifest.to_csv(args.manifest, index=False)
    print(f"Wrote {len(samples)} simulations ({args.nbase} bases x "
          f"{len(samples) // args.nbase} redshifts)")
    print(f"Samples: {args.output}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
