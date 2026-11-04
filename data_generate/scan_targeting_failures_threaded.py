#!/usr/bin/env python3
"""Threaded version of scan_targeting_failures.py for whole datasets (I/O-latency bound)."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import csv, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
from scan_targeting_failures import FITS_ROOT, header_values


def main(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=64)
    a = p.parse_args(argv)
    t0 = time.time()
    root = FITS_ROOT / a.dataset
    parts = sorted((d for d in root.iterdir() if d.name.startswith("part_")), key=lambda d: int(d.name[5:]))
    files = [f for d in parts for f in d.iterdir() if f.name.startswith("gal_") and f.name.endswith(".fits")]

    def check(f):
        v = header_values(f)
        return None if int(v["TGTHSMOK"]) == 1 else (a.dataset, int(f.parent.name[5:]), int(f.stem[4:]), v["TGTHSMOK"], v.get("TGTSNR", ""))

    with ThreadPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(check, files, chunksize=256) if r]
    rows.sort(key=lambda r: r[2])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as h:
        w = csv.writer(h); w.writerow(["dataset", "part", "ID", "TGTHSMOK", "TGTSNR"]); w.writerows(rows)
    print(f"{a.dataset}: scanned {len(files)}, failures {len(rows)}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
