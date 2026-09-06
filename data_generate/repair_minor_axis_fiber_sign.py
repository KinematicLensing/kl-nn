#!/usr/bin/env python3
"""Audit and repair the SVD minor-axis fiber reflection in simulator-v3 data.

Each already-generated row is classified from its stored fiber centers, not
from its ID. The current right-handed offsets are recomputed from the sample
table's ``g1``, ``g2``, ``theta_int``, and ``sini``. If those centers match
except that the two minor-axis fibers are exchanged, the row is ``swap_minor``
and those two spectra/positions are swapped. Numpy's 2x2 SVD is systematically
left-handed for this transform, so existing simulator-v3 catalogs are expected
to be entirely ``swap_minor`` until repaired.

The default is a dry run. Training should use CSV ``sini`` even when the LMDB
``fid_pars`` store ``cosi``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys

import lmdb
import msgpack
import numpy as np
import pandas as pd
from pyxis.pyxis import (
    DATA_DB,
    decode_data,
    encode_data,
)

try:
    from .observation_schema import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        MINOR_FIBER_INDICES,
        classify_fiber_offset_sign,
        legacy_major_anchored_fiber_offsets,
        swap_minor_axis_fibers,
    )
except ImportError:
    from observation_schema import (
        DEFAULT_FIBER_OFFSET_ARCSEC,
        MINOR_FIBER_INDICES,
        classify_fiber_offset_sign,
        legacy_major_anchored_fiber_offsets,
        swap_minor_axis_fibers,
    )


FITS_FIBER_HDUS = tuple(index + 1 for index in MINOR_FIBER_INDICES)
SHARED_ROOT = Path("/ocean/projects/phy250048p/shared")
GENERATION_PART_SIZE = 2000
LMDB_WRITE_BATCH = 256
SIMV3_CATALOGS = (
    "small_10k_simv3",
    "valid_100k_simv3",
    "valid_100k_simv3_cosi",
    "test_100k_simv3_xu1_tf",
    "test_100k_simv3_xu3_tf",
    "test_100k_simv3_xu5_tf",
    "test_100k_simv3_cosi_xu1_tf",
    "test_100k_simv3_cosi_xu3_tf",
    "test_100k_simv3_cosi_xu5_tf",
    "train_1m_simv3",
    "train_1m_simv3_cosi",
)


def predicted_legacy_fiber_sign_label(
    *,
    g1: float,
    g2: float,
    theta_int: float,
    sini: float,
    fiber_offset: float = DEFAULT_FIBER_OFFSET_ARCSEC,
) -> str:
    """Return the classification of data generated with the old SVD gauge."""

    stored = legacy_major_anchored_fiber_offsets(
        fiber_offset=fiber_offset,
        g1=g1,
        g2=g2,
        theta_int=theta_int,
        sini=sini,
    )
    return classify_fiber_offset_sign(
        stored,
        g1=g1,
        g2=g2,
        theta_int=theta_int,
        sini=sini,
        fiber_offset=fiber_offset,
    )


def load_sample_rows(sample_csv: Path) -> pd.DataFrame:
    table = pd.read_csv(sample_csv)
    required = ("ID", "g1", "g2", "theta_int", "sini")
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{sample_csv} is missing columns {missing}")
    return table


def classify_sample_table(
    table: pd.DataFrame,
    *,
    fiber_offset: float = DEFAULT_FIBER_OFFSET_ARCSEC,
) -> np.ndarray:
    """Predict stored-vs-current labels from the generation CSV alone."""

    labels = np.empty(len(table), dtype=object)
    for index, row in enumerate(table.itertuples(index=False)):
        labels[index] = predicted_legacy_fiber_sign_label(
            g1=float(row.g1),
            g2=float(row.g2),
            theta_int=float(row.theta_int),
            sini=float(row.sini),
            fiber_offset=fiber_offset,
        )
    return labels


def _decode_lmdb_sample(packed: bytes) -> dict:
    outer = msgpack.unpackb(packed, raw=False, use_list=True)
    sample = {}
    for key, value in outer.items():
        name = key.decode() if isinstance(key, bytes) else str(key)
        sample[name] = msgpack.unpackb(
            value, raw=False, use_list=False, object_hook=decode_data
        )
    return sample


def _encode_lmdb_sample(sample: dict) -> bytes:
    packed_fields = {}
    for key, value in sample.items():
        array = value if isinstance(value, np.ndarray) else np.asarray(value)
        packed_fields[key] = msgpack.packb(
            array, use_bin_type=True, default=encode_data
        )
    return msgpack.packb(packed_fields, use_bin_type=True)


def _sample_id(sample: dict, fallback: int) -> int:
    stored = sample.get("id")
    if stored is None:
        return fallback
    return int(np.asarray(stored).reshape(-1)[0])


def _row_kwargs(table: pd.DataFrame, fiber_offset: float) -> dict[int, dict]:
    kwargs_by_id = {}
    for row in table.itertuples(index=False):
        kwargs_by_id[int(row.ID)] = dict(
            g1=float(row.g1),
            g2=float(row.g2),
            theta_int=float(row.theta_int),
            sini=float(row.sini),
            fiber_offset=fiber_offset,
        )
    return kwargs_by_id


def _catalog_counts_ok(counts: Counter) -> bool:
    return not any(
        counts.get(name, 0)
        for name in (
            "csv_mismatch",
            "lmdb_mismatch",
            "fits_mismatch",
            "unexpected_id",
            "fits_missing",
            "lmdb_disagrees_with_csv_prediction",
            "fits_disagrees_with_csv_prediction",
        )
    )


def audit_lmdb(
    lmdb_dir: Path,
    table: pd.DataFrame,
    predicted: np.ndarray | None = None,
    *,
    fiber_offset: float = DEFAULT_FIBER_OFFSET_ARCSEC,
    apply: bool = False,
) -> Counter:
    """Classify stored ``fib_pos`` and optionally swap minor-axis arrays."""

    counts = Counter()
    predicted_by_id = (
        {int(sample_id): label for sample_id, label in zip(table["ID"], predicted)}
        if predicted is not None
        else None
    )
    kwargs_by_id = _row_kwargs(table, fiber_offset)
    probe = lmdb.open(str(lmdb_dir), readonly=True, max_dbs=2, lock=False)
    try:
        info = probe.info()
        map_size = int(info["map_size"])
        data_probe = probe.open_db(DATA_DB)
        with probe.begin(db=data_probe) as txn:
            keys = [key for key, _ in txn.cursor()]
    finally:
        probe.close()
    if apply:
        map_size = max(map_size, int(map_size * 2), map_size + (1 << 30))
    env = lmdb.open(
        str(lmdb_dir),
        readonly=not apply,
        max_dbs=2,
        map_size=map_size,
    )
    data_db = env.open_db(DATA_DB)
    batch = LMDB_WRITE_BATCH if apply else max(len(keys), 1)
    try:
        for start in range(0, len(keys), batch):
            chunk = keys[start : start + batch]
            with env.begin(write=apply, db=data_db) as txn:
                for key in chunk:
                    packed = txn.get(key)
                    sample = _decode_lmdb_sample(packed)
                    index = int(key.decode() if isinstance(key, bytes) else key)
                    sample_id = _sample_id(sample, index)
                    kwargs = kwargs_by_id.get(sample_id)
                    if kwargs is None:
                        counts["unexpected_id"] += 1
                        continue
                    stored = np.asarray(sample["fib_pos"], dtype=float)
                    observed = classify_fiber_offset_sign(stored, **kwargs)
                    counts[f"lmdb_{observed}"] += 1
                    if predicted_by_id is not None:
                        expected_label = predicted_by_id.get(sample_id)
                        if expected_label is None:
                            counts["unexpected_id"] += 1
                        elif observed != expected_label:
                            counts["lmdb_disagrees_with_csv_prediction"] += 1
                    if observed == "swap_minor" and apply:
                        sample["spec"], sample["fib_pos"] = swap_minor_axis_fibers(
                            sample["spec"], sample["fib_pos"]
                        )
                        repaired = classify_fiber_offset_sign(
                            sample["fib_pos"], **kwargs
                        )
                        if repaired != "match":
                            raise RuntimeError(
                                f"LMDB id={sample_id} still {repaired} after minor-axis swap"
                            )
                        txn.put(key, _encode_lmdb_sample(sample))
                        counts["lmdb_swapped"] += 1
            done = min(start + len(chunk), len(keys))
            if done == len(keys) or (done // 10000) != (start // 10000):
                print(
                    f"  lmdb {lmdb_dir.name}: {done}/{len(keys)}",
                    flush=True,
                )
    finally:
        env.close()
    return counts


def _fits_path(fits_root: Path, sample_id: int, part_size: int) -> Path:
    part = sample_id // part_size + 1
    return fits_root / f"part_{part}" / f"gal_{sample_id}.fits"


def _fits_worker(payload: tuple) -> tuple[str, bool]:
    path, kwargs, apply = payload
    from astropy.io import fits

    path = Path(path)
    if not path.is_file():
        return "missing", False
    plus_hdu, minus_hdu = FITS_FIBER_HDUS
    with fits.open(path, mode="update" if apply else "readonly", memmap=False) as hdus:
        stored = np.asarray(
            [
                (hdus[fiber].header["FIBERDX"], hdus[fiber].header["FIBERDY"])
                for fiber in range(1, 6)
            ],
            dtype=float,
        )
        observed = classify_fiber_offset_sign(stored, **kwargs)
        if observed != "swap_minor" or not apply:
            return observed, False
        hdus[plus_hdu].data, hdus[minus_hdu].data = (
            hdus[minus_hdu].data.copy(),
            hdus[plus_hdu].data.copy(),
        )
        for card in ("FIBERDX", "FIBERDY"):
            hdus[plus_hdu].header[card], hdus[minus_hdu].header[card] = (
                hdus[minus_hdu].header[card],
                hdus[plus_hdu].header[card],
            )
        hdus.flush()
    return observed, True


def audit_fits(
    fits_root: Path,
    table: pd.DataFrame,
    predicted: np.ndarray | None = None,
    *,
    part_size: int,
    fiber_offset: float = DEFAULT_FIBER_OFFSET_ARCSEC,
    apply: bool = False,
    jobs: int = 1,
) -> Counter:
    counts = Counter()
    predicted_by_id = (
        {int(sample_id): label for sample_id, label in zip(table["ID"], predicted)}
        if predicted is not None
        else None
    )
    payloads = []
    sample_ids = []
    for row in table.itertuples(index=False):
        sample_id = int(row.ID)
        sample_ids.append(sample_id)
        payloads.append(
            (
                str(_fits_path(fits_root, sample_id, part_size)),
                dict(
                    g1=float(row.g1),
                    g2=float(row.g2),
                    theta_int=float(row.theta_int),
                    sini=float(row.sini),
                    fiber_offset=fiber_offset,
                ),
                apply,
            )
        )
    workers = max(1, int(jobs))
    results = []
    if workers == 1:
        iterator = (_fits_worker(payload) for payload in payloads)
        for index, result in enumerate(iterator, 1):
            results.append(result)
            if index % GENERATION_PART_SIZE == 0 or index == len(payloads):
                print(
                    f"  fits {fits_root.name}: {index}/{len(payloads)}",
                    flush=True,
                )
    else:
        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp_context
        ) as pool:
            for index, result in enumerate(
                pool.map(_fits_worker, payloads, chunksize=64), 1
            ):
                results.append(result)
                if index % GENERATION_PART_SIZE == 0 or index == len(payloads):
                    print(
                        f"  fits {fits_root.name}: {index}/{len(payloads)}",
                        flush=True,
                    )
    for sample_id, (observed, swapped) in zip(sample_ids, results):
        if observed == "missing":
            counts["fits_missing"] += 1
            continue
        counts[f"fits_{observed}"] += 1
        if predicted_by_id is not None:
            expected_label = predicted_by_id.get(sample_id)
            if expected_label is None:
                counts["unexpected_id"] += 1
            elif observed != expected_label:
                counts["fits_disagrees_with_csv_prediction"] += 1
        if observed == "mismatch":
            print(f"  fits mismatch id={sample_id}", flush=True)
        if swapped:
            counts["fits_swapped"] += 1
    return counts


def catalog_paths(name: str, *, shared_root: Path = SHARED_ROOT) -> tuple[Path, Path, Path]:
    return (
        shared_root / "samples" / f"{name}.csv",
        shared_root / "datasets" / name,
        shared_root / "fits" / name,
    )


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path)
    parser.add_argument("--lmdb", type=Path)
    parser.add_argument("--fits-root", type=Path)
    parser.add_argument("--part-size", type=int, default=GENERATION_PART_SIZE)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--fiber-offset",
        type=float,
        default=DEFAULT_FIBER_OFFSET_ARCSEC,
    )
    parser.add_argument(
        "--predict-csv",
        action="store_true",
        help="also classify the old SVD gauge from the CSV alone",
    )
    parser.add_argument(
        "--all-simv3",
        action="store_true",
        help="repair every shared simulator-v3 catalog",
    )
    parser.add_argument(
        "--skip-lmdb",
        action="store_true",
        help="do not open shared LMDB catalogs",
    )
    parser.add_argument(
        "--skip-fits",
        action="store_true",
        help="do not open shared FITS catalogs",
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default="",
        help="when using --all-simv3, skip catalogs before this name",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write minor-axis swaps; default is dry-run classification only",
    )
    return parser.parse_args(argv)


def _print_counts(mode: str, label: str, counts: Counter) -> None:
    print(f"{mode} {label}", flush=True)
    for key in sorted(counts):
        print(f"{key}\t{counts[key]}", flush=True)


def repair_catalog(
    sample_csv: Path,
    *,
    lmdb_dir: Path | None,
    fits_root: Path | None,
    part_size: int,
    fiber_offset: float,
    apply: bool,
    jobs: int,
    predict_csv: bool,
) -> Counter:
    table = load_sample_rows(sample_csv)
    predicted = (
        classify_sample_table(table, fiber_offset=fiber_offset)
        if predict_csv
        else None
    )
    counts = Counter()
    if predicted is not None:
        counts.update(Counter(f"csv_{label}" for label in predicted))
    if lmdb_dir is not None:
        counts.update(
            audit_lmdb(
                lmdb_dir,
                table,
                predicted,
                fiber_offset=fiber_offset,
                apply=apply,
            )
        )
    if fits_root is not None:
        counts.update(
            audit_fits(
                fits_root,
                table,
                predicted,
                part_size=part_size,
                fiber_offset=fiber_offset,
                apply=apply,
                jobs=jobs,
            )
        )
    return counts


def main(argv=None) -> int:
    args = parse_args(argv)
    mode = "apply" if args.apply else "dry-run"
    if args.all_simv3:
        failed = []
        catalogs = SIMV3_CATALOGS
        if args.start_from:
            if args.start_from not in catalogs:
                raise SystemExit(
                    f"--start-from must be one of {catalogs}; got {args.start_from!r}"
                )
            catalogs = catalogs[catalogs.index(args.start_from) :]
        for name in catalogs:
            sample_csv, lmdb_dir, fits_root = catalog_paths(name)
            if args.skip_lmdb:
                lmdb_dir = None
            if args.skip_fits:
                fits_root = None
            print(f"=== {mode} {name} ===", flush=True)
            counts = repair_catalog(
                sample_csv,
                lmdb_dir=lmdb_dir,
                fits_root=fits_root,
                part_size=args.part_size,
                fiber_offset=args.fiber_offset,
                apply=args.apply,
                jobs=args.jobs,
                predict_csv=args.predict_csv,
            )
            _print_counts(mode, name, counts)
            if not _catalog_counts_ok(counts):
                failed.append(name)
                print(f"warning: {name} had classification problems", flush=True)
        return 2 if failed else 0

    if args.sample_csv is None:
        raise SystemExit("--sample-csv is required unless --all-simv3 is set")
    counts = repair_catalog(
        args.sample_csv,
        lmdb_dir=args.lmdb,
        fits_root=args.fits_root,
        part_size=args.part_size,
        fiber_offset=args.fiber_offset,
        apply=args.apply,
        jobs=args.jobs,
        predict_csv=args.predict_csv,
    )
    _print_counts(mode, str(args.sample_csv), counts)
    if counts.get("csv_mismatch", 0) or counts.get("lmdb_mismatch", 0) or counts.get("fits_mismatch", 0):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
