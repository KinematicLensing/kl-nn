#!/usr/bin/env python3
"""Build compact posterior summaries after enforcing archived prior support.

The source posterior arrays are opened read-only with NumPy memory mapping and
processed one galaxy at a time.  A draw is retained only when every inferred
feature is finite and lies inside the inclusive physical bounds archived with
the model.  Original cache arrays are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PART_RE = re.compile(r"part(\d+)of(\d+)\.npy$")
SUMMARY_DIR = "in_support_mean_estimates"
RETENTION_DIR = "in_support_retention"
META_DIR = "in_support_meta"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache",
        type=Path,
        help="Existing MODEL/DATASET tf_analysis cache directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Archived model config JSON (default: infer from cache model name).",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path("/ocean/projects/phy250048p/shared/configs"),
        help="Location of cfg_MODEL.json files when --config is omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing compact support-filter outputs.",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def partition_files(directory: Path) -> list[Path]:
    records = []
    for path in directory.glob("part*of*.npy"):
        match = PART_RE.fullmatch(path.name)
        if match:
            records.append((int(match.group(1)), int(match.group(2)), path))
    if not records:
        raise FileNotFoundError(f"No partition arrays in {directory}")
    totals = {record[1] for record in records}
    if len(totals) != 1:
        raise ValueError(f"Mixed partition totals in {directory}: {totals}")
    expected = next(iter(totals))
    records.sort(key=lambda record: record[0])
    indices = [record[0] for record in records]
    if indices != list(range(expected)):
        raise ValueError(f"Incomplete partitions in {directory}: {indices} of {expected}")
    return [record[2] for record in records]


def load_archived_bounds(
    cache: Path, config_path: Path | None, configs_root: Path
) -> tuple[Path, list[str], np.ndarray]:
    model_name = cache.parent.name
    resolved = config_path or configs_root / f"cfg_{model_name}.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"Archived model config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    feature_names = list(payload["train"]["feature_names"])
    ranges = payload["par_ranges"]
    missing = [name for name in feature_names if name not in ranges]
    if missing:
        raise KeyError(f"Missing archived parameter ranges for: {missing}")
    bounds = np.asarray([ranges[name] for name in feature_names], dtype=np.float64)
    if bounds.shape != (len(feature_names), 2):
        raise ValueError("Archived bounds must have shape (features, 2)")
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Archived bounds must be finite ordered intervals")
    return resolved.resolve(), feature_names, bounds


def summarize_samples(
    values: np.ndarray, feature_names: list[str] | tuple[str, ...]
) -> np.ndarray:
    """Return 16th, mean, and 84th summaries, treating theta circularly."""
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("values must have shape (draws, features)")
    if len(values) == 0:
        return np.full((3, len(feature_names)), np.nan, dtype=np.float64)
    summary = np.stack(
        (
            np.percentile(values, 16, axis=0),
            np.mean(values, axis=0),
            np.percentile(values, 84, axis=0),
        ),
        axis=0,
    )
    if "theta_int" in feature_names:
        theta_index = feature_names.index("theta_int")
        theta = values[:, theta_index]
        center = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())
        delta = np.arctan2(np.sin(theta - center), np.cos(theta - center))
        lower, upper = np.percentile(delta, (16, 84))
        circular = np.asarray((center + lower, center, center + upper))
        summary[:, theta_index] = (circular + np.pi) % (2.0 * np.pi) - np.pi
    return summary


def summarize_partition(
    samples: np.ndarray,
    feature_names: list[str],
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply the joint support mask to one memory-mapped sample partition."""
    if samples.ndim != 4:
        raise ValueError("sample partition must have shape (mode, galaxy, draw, feature)")
    modes, galaxies, draws, features = samples.shape
    if features != len(feature_names) or bounds.shape != (features, 2):
        raise ValueError("Feature count is inconsistent with archived bounds")
    summaries = np.full((modes, galaxies, 3, features), np.nan, dtype=np.float64)
    retention = np.zeros((modes, galaxies), dtype=np.float64)
    per_feature_rejected = np.zeros((modes, features), dtype=np.int64)
    jointly_retained = np.zeros(modes, dtype=np.int64)
    finite_draws = np.zeros(modes, dtype=np.int64)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    for mode_index in range(modes):
        for galaxy_index in range(galaxies):
            values = np.asarray(samples[mode_index, galaxy_index])
            finite = np.isfinite(values)
            within = finite & (values >= lower) & (values <= upper)
            keep = np.all(within, axis=1)
            per_feature_rejected[mode_index] += np.count_nonzero(~within, axis=0)
            finite_draws[mode_index] += np.count_nonzero(np.all(finite, axis=1))
            retained = int(np.count_nonzero(keep))
            jointly_retained[mode_index] += retained
            retention[mode_index, galaxy_index] = retained / draws
            summaries[mode_index, galaxy_index] = summarize_samples(
                values[keep], feature_names
            )

    total_per_mode = galaxies * draws
    diagnostics = {
        "total_draws_per_mode": total_per_mode,
        "jointly_retained_count_by_mode": jointly_retained.tolist(),
        "jointly_retained_fraction_by_mode": (
            jointly_retained / total_per_mode
        ).tolist(),
        "fully_finite_count_by_mode": finite_draws.tolist(),
        "per_feature_rejected_count_by_mode": per_feature_rejected.tolist(),
        "per_feature_rejected_fraction_by_mode": (
            per_feature_rejected / total_per_mode
        ).tolist(),
        "zero_retained_galaxies_by_mode": np.count_nonzero(
            retention == 0.0, axis=1
        ).tolist(),
    }
    return summaries, retention, diagnostics


def atomic_save_array(path: Path, array: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_cache(
    cache: Path,
    *,
    config_path: Path | None = None,
    configs_root: Path = Path("/ocean/projects/phy250048p/shared/configs"),
    overwrite: bool = False,
) -> dict:
    cache = cache.resolve()
    sample_paths = partition_files(cache / "sample")
    archived_config, feature_names, bounds = load_archived_bounds(
        cache, config_path, configs_root
    )
    config_sha256 = hashlib.sha256(archived_config.read_bytes()).hexdigest()
    output_dirs = {
        "summary": cache / SUMMARY_DIR,
        "retention": cache / RETENTION_DIR,
        "meta": cache / META_DIR,
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    part_records = []
    retained_sum = None
    total_sum = None
    rejected_sum = None
    for source in sample_paths:
        samples = np.load(source, mmap_mode="r", allow_pickle=False)
        summaries, retention, diagnostics = summarize_partition(
            samples, feature_names, bounds
        )
        summary_path = output_dirs["summary"] / source.name
        retention_path = output_dirs["retention"] / source.name
        meta_path = output_dirs["meta"] / source.with_suffix(".json").name
        atomic_save_array(summary_path, summaries, overwrite)
        atomic_save_array(retention_path, retention, overwrite)
        relative_source = source.relative_to(cache).as_posix()
        part_meta = {
            "schema_version": 1,
            "created_at_utc": now_utc_iso(),
            "source_cache": str(cache),
            "source_sample_part": relative_source,
            "source_shape": list(samples.shape),
            "source_dtype": str(samples.dtype),
            "archived_config": str(archived_config),
            "archived_config_sha256": config_sha256,
            "feature_names": feature_names,
            "inclusive_bounds": {
                name: [float(low), float(high)]
                for name, (low, high) in zip(feature_names, bounds)
            },
            "joint_rule": (
                "retain iff every feature is finite and lower <= value <= upper"
            ),
            "outputs": {
                "summary": summary_path.relative_to(cache).as_posix(),
                "retention": retention_path.relative_to(cache).as_posix(),
            },
            "diagnostics": diagnostics,
        }
        atomic_write_json(meta_path, part_meta, overwrite)
        part_records.append(
            {
                "source": relative_source,
                "summary": summary_path.relative_to(cache).as_posix(),
                "retention": retention_path.relative_to(cache).as_posix(),
                "meta": meta_path.relative_to(cache).as_posix(),
            }
        )
        retained = np.asarray(diagnostics["jointly_retained_count_by_mode"])
        total = np.full_like(retained, diagnostics["total_draws_per_mode"])
        rejected = np.asarray(diagnostics["per_feature_rejected_count_by_mode"])
        retained_sum = retained if retained_sum is None else retained_sum + retained
        total_sum = total if total_sum is None else total_sum + total
        rejected_sum = rejected if rejected_sum is None else rejected_sum + rejected
        print(
            f"{source.name}: retained "
            + ", ".join(f"mode {i} {fraction:.2%}" for i, fraction in enumerate(retention.mean(axis=1)))
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": now_utc_iso(),
        "source_cache": str(cache),
        "archived_config": str(archived_config),
        "archived_config_sha256": config_sha256,
        "feature_names": feature_names,
        "inclusive_bounds": {
            name: [float(low), float(high)]
            for name, (low, high) in zip(feature_names, bounds)
        },
        "joint_rule": "retain iff every feature is finite and lower <= value <= upper",
        "source_parts": part_records,
        "aggregate": {
            "jointly_retained_count_by_mode": retained_sum.tolist(),
            "total_draws_by_mode": total_sum.tolist(),
            "jointly_retained_fraction_by_mode": (retained_sum / total_sum).tolist(),
            "per_feature_rejected_count_by_mode": rejected_sum.tolist(),
            "per_feature_rejected_fraction_by_mode": (rejected_sum / total_sum[:, None]).tolist(),
        },
    }
    atomic_write_json(output_dirs["meta"] / "manifest.json", manifest, overwrite)
    return manifest


def main():
    args = parse_args()
    manifest = process_cache(
        args.cache,
        config_path=args.config,
        configs_root=args.configs_root,
        overwrite=args.overwrite,
    )
    fractions = manifest["aggregate"]["jointly_retained_fraction_by_mode"]
    print(
        "Finished: "
        + ", ".join(f"mode {index} retained {value:.2%}" for index, value in enumerate(fractions))
    )


if __name__ == "__main__":
    main()
