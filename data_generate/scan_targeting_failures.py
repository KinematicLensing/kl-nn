#!/usr/bin/env python3
"""List measured-axis galaxies whose targeting position-angle measurement failed.

Reads only the primary FITS header of each ``gal_<ID>.fits`` under
``fits/<dataset>/part_<n>/`` and writes one CSV row per galaxy with
TGTHSMOK != 1 (dataset, part, ID, TGTHSMOK, TGTSNR).
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
from pathlib import Path
import re

FITS_ROOT = Path("/ocean/projects/phy250048p/shared/fits")
CARD = re.compile(rb"(TGTHSMOK|TGTSNR  )=\s*([^/\s]+)")


def header_values(path: Path) -> dict[str, str]:
    with path.open("rb") as handle:
        head = handle.read(2880 * 3)
    values = {key.decode().strip(): value.decode() for key, value in CARD.findall(head)}
    if "TGTHSMOK" not in values:
        raise ValueError(f"{path} has no TGTHSMOK card in its first 3 header blocks")
    return values


def main(argv=None):
    p = ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--parts", required=True, help="first-last part numbers, inclusive")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    first, last = (int(x) for x in args.parts.split("-"))
    rows, n = [], 0
    for part in range(first, last + 1):
        for path in sorted((FITS_ROOT / args.dataset / f"part_{part}").glob("gal_*.fits")):
            n += 1
            v = header_values(path)
            if int(v["TGTHSMOK"]) != 1:
                rows.append((args.dataset, part, int(path.stem[4:]), v["TGTHSMOK"], v.get("TGTSNR", "")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["dataset", "part", "ID", "TGTHSMOK", "TGTSNR"])
        w.writerows(rows)
    print(f"{args.dataset} parts {first}-{last}: scanned {n}, failures {len(rows)}")


if __name__ == "__main__":
    main()
