#!/usr/bin/env python3
"""Prepare exact fixed-shear simulator-v3 rows for likelihood-stack 04."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE = Path(
    "/ocean/projects/phy250048p/shared/samples/valid_100k_simv3_cosi.csv"
)
DEFAULT_OUTPUT = Path(
    "/ocean/projects/phy250048p/shared/samples/"
    "likelihood_stack_04_benchmark.csv"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".benchmark.json")
SHEAR_VALUES = (-0.08, -0.04, 0.0, 0.04, 0.08)
REQUIRED_COLUMNS = (
    "ID",
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
    "rmag_true",
    "halpha_flux_true",
    "image_snr",
    "central_halpha_snr",
    "fiber_layout",
    "observation_model_version",
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--n-galaxies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_table(
    source: pd.DataFrame,
    *,
    n_galaxies: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    missing = [name for name in REQUIRED_COLUMNS if name not in source.columns]
    if missing:
        raise ValueError(f"source table is missing columns: {missing}")
    if n_galaxies <= 0:
        raise ValueError("n_galaxies must be positive")
    if len(source) < n_galaxies:
        raise ValueError("source table has fewer rows than n_galaxies")

    rng = np.random.default_rng(seed)
    selected = np.sort(
        rng.choice(len(source), size=n_galaxies, replace=False).astype(np.int64)
    )
    rows = []
    mapping = []
    cell = 0
    for g1 in SHEAR_VALUES:
        for g2 in SHEAR_VALUES:
            for base_index in selected:
                row = source.iloc[int(base_index)].copy()
                row["ID"] = len(rows)
                row["g1"] = float(g1)
                row["g2"] = float(g2)
                row["benchmark_cell"] = cell
                row["benchmark_base_index"] = int(base_index)
                rows.append(row)
                mapping.append(
                    {
                        "cell": cell,
                        "g1": float(g1),
                        "g2": float(g2),
                        "benchmark_base_index": int(base_index),
                        "id_start": int(len(rows) - 1),
                    }
                )
            cell += 1
    table = pd.DataFrame(rows)
    table["ID"] = table["ID"].astype(np.int64)
    table["benchmark_cell"] = table["benchmark_cell"].astype(np.int64)
    table["benchmark_base_index"] = table["benchmark_base_index"].astype(np.int64)
    return table, {
        "schema": "likelihood-stack-fixed-shear-v1",
        "seed": int(seed),
        "n_galaxies_per_cell": int(n_galaxies),
        "shear_values": list(SHEAR_VALUES),
        "n_cells": len(SHEAR_VALUES) ** 2,
        "n_rows": len(table),
        "source_rows": selected.tolist(),
        "mapping": mapping,
        "noise_realizations": 16,
        "noise_prefixes": [1, 2, 4, 8, 16],
        "identity_noise_only": True,
        "source": str(DEFAULT_SOURCE),
    }


def main(argv=None):
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if (output.exists() or manifest.exists()) and not args.overwrite:
        raise FileExistsError(
            f"{output} or {manifest} exists; use --overwrite"
        )
    source = pd.read_csv(args.source, float_precision="round_trip")
    table, payload = build_table(
        source,
        n_galaxies=args.n_galaxies,
        seed=args.seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.17g")
    payload["source"] = str(args.source.expanduser().resolve())
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output} ({len(table)} rows)")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
