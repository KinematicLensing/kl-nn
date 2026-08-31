"""Check that a simulator proposal has exactly the expected FITS outputs.

Deep verification checks the versioned science-row fingerprint, but this is
not a checksum of every FITS data byte (there are no CHECKSUM/DATASUM cards).
"""

from __future__ import annotations

from argparse import ArgumentParser
import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys

try:
    from .generation_integrity import (
        HEADER_METADATA_COLUMNS,
        quick_fits_completion_error,
        simulator_v3_fits_completion_error,
        simulator_v3_science_row_fingerprint,
    )
except ImportError:  # Support direct execution from data_generate/.
    from generation_integrity import (
        HEADER_METADATA_COLUMNS,
        quick_fits_completion_error,
        simulator_v3_fits_completion_error,
        simulator_v3_science_row_fingerprint,
    )


FITS_NAME_PATTERN = re.compile(r"gal_(\d+)\.fits")
PART_NAME_PATTERN = re.compile(r"part_(\d+)")


@dataclass
class CheckSummary:
    expected: int = 0
    complete: int = 0
    missing: int = 0
    invalid: int = 0
    unexpected: int = 0
    incomplete_parts: set[int] = field(default_factory=set)

    @property
    def failures(self) -> int:
        return self.missing + self.invalid + self.unexpected


def parse_args(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--sample", required=True, help="proposal CSV path")
    parser.add_argument(
        "--fits-dir",
        required=True,
        help="dataset directory containing part_N directories",
    )
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument(
        "--total",
        type=int,
        help="number of leading proposal rows to check (default: all rows)",
    )
    parser.add_argument(
        "--verify-fits",
        action="store_true",
        help="open every FITS and validate HDUs, schema, and row metadata",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="TSV path for the exact missing, invalid, and unexpected files",
    )
    return parser.parse_args(argv)


def _id_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("Sample CSV has no header")
    if "ID" in fieldnames:
        return "ID"
    unnamed = [name for name in fieldnames if name.startswith("Unnamed:")]
    if len(unnamed) == 1:
        return unnamed[0]
    raise ValueError("Sample CSV must contain an explicit ID column")


def _parse_sample_id(value: str, *, row_index: int) -> int:
    try:
        sample_id = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid ID {value!r} at CSV row {row_index}") from error
    if sample_id != row_index:
        raise ValueError(
            f"CSV row {row_index} has ID={sample_id}; packaging requires IDs "
            "to be exactly 0..TOTAL-1 in row order"
        )
    return sample_id


def _metadata(row: dict[str, str]) -> dict[str, float]:
    try:
        return {name: float(row[name]) for name in HEADER_METADATA_COLUMNS}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Deep FITS verification requires numeric columns: "
            + ", ".join(HEADER_METADATA_COLUMNS)
        ) from error


def _write_failure(
    report,
    *,
    status: str,
    part: int,
    row_index: int | None,
    sample_id: int | None,
    path: Path,
    detail: str,
) -> None:
    report.writerow(
        {
            "status": status,
            "part": part,
            "row": "" if row_index is None else row_index,
            "ID": "" if sample_id is None else sample_id,
            "path": str(path),
            "detail": detail,
        }
    )


def _check_part(
    *,
    part: int,
    expected: dict[
        str,
        tuple[int, int, dict[str, float] | None, str | None],
    ],
    fits_dir: Path,
    verify_fits: bool,
    report,
    summary: CheckSummary,
) -> None:
    part_dir = fits_dir / f"part_{part}"
    actual: dict[str, Path] = {}
    if part_dir.is_dir():
        with os.scandir(part_dir) as entries:
            for entry in entries:
                if entry.name.startswith("gal_") and entry.name.endswith(".fits"):
                    actual[entry.name] = Path(entry.path)

    for filename, (
        row_index,
        sample_id,
        metadata,
        row_fingerprint,
    ) in expected.items():
        summary.expected += 1
        path = actual.pop(filename, None)
        if path is None:
            summary.missing += 1
            summary.incomplete_parts.add(part)
            _write_failure(
                report,
                status="missing",
                part=part,
                row_index=row_index,
                sample_id=sample_id,
                path=part_dir / filename,
                detail="expected path does not exist",
            )
            continue
        if verify_fits:
            error = simulator_v3_fits_completion_error(
                path,
                expected_metadata=metadata,
                expected_sample_id=sample_id,
                expected_row_fingerprint=row_fingerprint,
            )
        else:
            error = quick_fits_completion_error(path)
        if error is None:
            summary.complete += 1
            continue
        summary.invalid += 1
        summary.incomplete_parts.add(part)
        _write_failure(
            report,
            status="invalid",
            part=part,
            row_index=row_index,
            sample_id=sample_id,
            path=path,
            detail=error,
        )

    for filename, path in sorted(actual.items()):
        match = FITS_NAME_PATTERN.fullmatch(filename)
        sample_id = int(match.group(1)) if match else None
        summary.unexpected += 1
        summary.incomplete_parts.add(part)
        _write_failure(
            report,
            status="unexpected",
            part=part,
            row_index=None,
            sample_id=sample_id,
            path=path,
            detail="file is not assigned to this part by the proposal table",
        )


def _check_unexpected_parts(
    *,
    fits_dir: Path,
    expected_part_count: int,
    report,
    summary: CheckSummary,
) -> None:
    if not fits_dir.is_dir():
        return
    with os.scandir(fits_dir) as entries:
        for entry in entries:
            match = PART_NAME_PATTERN.fullmatch(entry.name)
            if not match or not entry.is_dir():
                continue
            part = int(match.group(1))
            if 1 <= part <= expected_part_count:
                continue
            with os.scandir(entry.path) as files:
                for file_entry in files:
                    if not (
                        file_entry.name.startswith("gal_")
                        and file_entry.name.endswith(".fits")
                    ):
                        continue
                    filename_match = FITS_NAME_PATTERN.fullmatch(file_entry.name)
                    sample_id = (
                        int(filename_match.group(1)) if filename_match else None
                    )
                    summary.unexpected += 1
                    summary.incomplete_parts.add(part)
                    _write_failure(
                        report,
                        status="unexpected",
                        part=part,
                        row_index=None,
                        sample_id=sample_id,
                        path=Path(file_entry.path),
                        detail="part directory is outside the expected range",
                    )


def _compact_ranges(values: set[int]) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def check_generation(args) -> CheckSummary:
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.total is not None and args.total <= 0:
        raise ValueError("total must be positive")
    sample_path = Path(args.sample)
    fits_dir = Path(args.fits_dir)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = CheckSummary()

    with sample_path.open(newline="") as sample_file, report_path.open(
        "w", newline=""
    ) as report_file:
        reader = csv.DictReader(sample_file)
        id_column = _id_column(reader.fieldnames)
        report = csv.DictWriter(
            report_file,
            fieldnames=("status", "part", "row", "ID", "path", "detail"),
            delimiter="\t",
        )
        report.writeheader()
        current_part = 1
        expected: dict[
            str,
            tuple[int, int, dict[str, float] | None, str | None],
        ] = {}
        rows_seen = 0
        for row_index, row in enumerate(reader):
            if args.total is not None and row_index >= args.total:
                break
            sample_id = _parse_sample_id(row[id_column], row_index=row_index)
            part = row_index // args.chunk_size + 1
            if part != current_part:
                _check_part(
                    part=current_part,
                    expected=expected,
                    fits_dir=fits_dir,
                    verify_fits=args.verify_fits,
                    report=report,
                    summary=summary,
                )
                current_part = part
                expected = {}
            expected[f"gal_{sample_id}.fits"] = (
                row_index,
                sample_id,
                _metadata(row) if args.verify_fits else None,
                (
                    simulator_v3_science_row_fingerprint(sample_id, row)
                    if args.verify_fits
                    else None
                ),
            )
            rows_seen += 1
        if args.total is not None and rows_seen < args.total:
            raise ValueError(
                f"Sample CSV contains {rows_seen} rows, fewer than TOTAL={args.total}"
            )
        if rows_seen == 0:
            raise ValueError("Sample CSV contains no rows to check")
        _check_part(
            part=current_part,
            expected=expected,
            fits_dir=fits_dir,
            verify_fits=args.verify_fits,
            report=report,
            summary=summary,
        )
        expected_part_count = (rows_seen + args.chunk_size - 1) // args.chunk_size
        _check_unexpected_parts(
            fits_dir=fits_dir,
            expected_part_count=expected_part_count,
            report=report,
            summary=summary,
        )

    print(
        "Mode: "
        + (
            "deep FITS validation"
            if args.verify_fits
            else "quick path/size completeness"
        )
    )
    print(
        f"Expected={summary.expected} complete={summary.complete} "
        f"missing={summary.missing} invalid={summary.invalid} "
        f"unexpected={summary.unexpected}"
    )
    print(f"Incomplete array parts: {_compact_ranges(summary.incomplete_parts)}")
    print(f"Failure manifest: {report_path}")
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        summary = check_generation(args)
    except (OSError, ValueError, csv.Error) as error:
        print(f"Configuration/input error: {error}", file=sys.stderr)
        return 2
    return 1 if summary.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
