#!/usr/bin/env python3
"""Dry-run, apply, and re-audit the minimal right-handed fiber-gauge repair."""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

try:
    from .repair_minor_axis_fiber_sign import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        SHARED_ROOT,
        catalog_paths,
        repair_catalog,
    )
except ImportError:
    from repair_minor_axis_fiber_sign import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        SHARED_ROOT,
        catalog_paths,
        repair_catalog,
    )


CAMPAIGN_CATALOGS = (
    {
        "name": "valid_100k_simv3_cosi",
        "required": True,
        "skip_fits": True,
        "skip_lmdb": False,
    },
    {
        "name": "test_100k_simv3_cosi_xu3_tf",
        "required": True,
        "skip_fits": True,
        "skip_lmdb": False,
    },
    {
        "name": "test_xu3_tf_g002_inbox",
        "required": False,
        "skip_fits": True,
        "skip_lmdb": False,
        "id_column": "fits_id",
    },
    {
        "name": "test_100k_simv3_cosi_xu3_tf_g02",
        "required": False,
        "skip_fits": True,
        "skip_lmdb": False,
        "sample_csv": "test_100k_simv3_cosi_xu3_tf.csv",
    },
    {
        "name": "small_10k_simv3_cosi",
        "required": False,
        "skip_fits": True,
        "skip_lmdb": False,
        "sample_csv": "train_1m_simv3_cosi.csv",
        "sample_rows": 10000,
    },
)


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("dry-run", "apply", "verify"),
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON audit destination",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    return parser.parse_args(argv)


def _counter_dict(counts: Counter) -> dict[str, int]:
    return {str(key): int(counts[key]) for key in sorted(counts)}


def _catalog_hard_fail(counts: Counter) -> bool:
    return any(
        counts.get(name, 0)
        for name in (
            "lmdb_mismatch",
            "fits_mismatch",
            "unexpected_id",
            "fits_missing",
        )
    )


def _catalog_still_swapped(counts: Counter) -> bool:
    return any(
        counts.get(name, 0)
        for name in (
            "lmdb_swap_minor",
            "fits_swap_minor",
            "lmdb_mismatch",
            "fits_mismatch",
        )
    )


def _resolve_catalog(spec: dict, shared_root: Path, work_dir: Path) -> dict | None:
    sample_csv, lmdb_dir, fits_root = catalog_paths(
        spec["name"], shared_root=shared_root
    )
    if "sample_csv" in spec:
        sample_csv = shared_root / "samples" / spec["sample_csv"]
    if spec.get("skip_lmdb"):
        lmdb_dir = None
    if spec.get("skip_fits"):
        fits_root = None
    if lmdb_dir is not None and not lmdb_dir.is_dir():
        if spec["required"]:
            raise FileNotFoundError(f"missing required LMDB {lmdb_dir}")
        return None
    if fits_root is not None:
        if not fits_root.exists():
            if spec["required"]:
                raise FileNotFoundError(f"missing required FITS {fits_root}")
            fits_root = None
        elif fits_root.is_symlink():
            fits_root = None
    if not sample_csv.is_file():
        if spec["required"]:
            raise FileNotFoundError(f"missing required CSV {sample_csv}")
        return None
    sample_rows = spec.get("sample_rows")
    id_column = spec.get("id_column")
    if sample_rows or (id_column and id_column != "ID"):
        work_dir.mkdir(parents=True, exist_ok=True)
        table = pd.read_csv(sample_csv)
        if sample_rows:
            table = table.iloc[: int(sample_rows)]
        if id_column and id_column != "ID":
            if id_column not in table.columns:
                raise ValueError(f"{sample_csv} is missing id column {id_column}")
            table = table.copy()
            table["ID"] = table[id_column].astype(int)
        remapped = work_dir / f"{spec['name']}_repair_rows.csv"
        table.to_csv(remapped, index=False)
        sample_csv = remapped
    return {
        "name": spec["name"],
        "required": spec["required"],
        "sample_csv": sample_csv,
        "lmdb_dir": lmdb_dir,
        "fits_root": fits_root,
    }


def run_phase(args) -> dict:
    apply = args.phase == "apply"
    results = []
    failed = []
    for spec in CAMPAIGN_CATALOGS:
        resolved = _resolve_catalog(spec, args.shared_root, args.output.parent)
        if resolved is None:
            results.append(
                {
                    "name": spec["name"],
                    "skipped": True,
                    "reason": "optional catalog missing",
                }
            )
            continue
        print(f"=== {args.phase} {resolved['name']} ===", flush=True)
        counts = repair_catalog(
            resolved["sample_csv"],
            lmdb_dir=resolved["lmdb_dir"],
            fits_root=resolved["fits_root"],
            part_size=2000,
            fiber_offset=DEFAULT_FIBER_OFFSET_ARCSEC,
            apply=apply,
            jobs=args.jobs,
            predict_csv=False,
        )
        record = {
            "name": resolved["name"],
            "skipped": False,
            "required": resolved["required"],
            "sample_csv": str(resolved["sample_csv"]),
            "lmdb": None if resolved["lmdb_dir"] is None else str(resolved["lmdb_dir"]),
            "fits_root": None
            if resolved["fits_root"] is None
            else str(resolved["fits_root"]),
            "counts": _counter_dict(counts),
            "ok": not _catalog_hard_fail(counts),
            "still_swapped": _catalog_still_swapped(counts),
        }
        results.append(record)
        print(json.dumps(record["counts"], indent=2), flush=True)
        if not record["ok"]:
            failed.append(resolved["name"])
        if args.phase == "verify" and record["still_swapped"]:
            failed.append(resolved["name"])
    payload = {
        "phase": args.phase,
        "failed": failed,
        "catalogs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)
    payload = run_phase(args)
    if payload["failed"]:
        print(f"failed catalogs: {payload['failed']}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
