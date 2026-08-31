"""Catalog-backed sampling for TF-conformed simulator test sets.

The DESI cut catalogs are much larger than memory on a single worker.  This
module therefore performs two contiguous, bounded-memory FITS scans: one to
count rows in simulator support and one to materialize uniformly selected
eligible rows.  Magnitude, size, and both S/N values always come from the same
catalog row so their empirical correlations are retained.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

try:
    from .observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_HALPHA_LOG10_FLUX_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )
except ImportError:  # Support direct execution from data_generate/.
    from observation_schema import (
        CENTRAL_HALPHA_SNR_COLUMN,
        DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
        DEFAULT_HALPHA_LOG10_FLUX_RANGE,
        DEFAULT_IMAGE_SNR_RANGE,
        FIBER_LAYOUT_COLUMN,
        GALAXY_AXIS_FIBER_LAYOUT,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        OBSERVATION_MODEL_VERSION,
        OBSERVATION_MODEL_VERSION_COLUMN,
        RMAG_TRUE_COLUMN,
    )

try:
    from arch.tf_prior import TFPrior, sample_truncated_tf_vcirc
except ModuleNotFoundError:  # Support direct execution from data_generate/.
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    from arch.tf_prior import TFPrior, sample_truncated_tf_vcirc


GENERATION_MANIFEST_SCHEMA = "klnn-generation-manifest-v1"
TEST_SET_ANALYSIS_MODE = "test_set"
TEST_SET_POPULATION = "tf_conformed_catalog"
SIMULATION_REDSHIFT = 0.3
DEFAULT_CATALOG_EXTENSION = "SELECTION"
DEFAULT_CATALOG_BLOCK_SIZE = 500_000
DEFAULT_RMAG_RANGE = (15.0, 23.4)

REQUIRED_CATALOG_COLUMNS = (
    "targetid",
    "z",
    "rmag",
    "hlr",
    "img_snr",
    "halpha_snr",
    "xu_effective_weight",
)
UNIFORMLY_SAMPLED_PARAMETERS = ("g1", "g2", "theta_int", "v0", "rscale")
SOURCE_PROVENANCE_COLUMNS = (
    "source_catalog_row",
    "source_targetid",
    "source_catalog_z",
    "source_hlr_raw",
    "source_xu_effective_weight",
)


def _lhs_column(
    nsamples: int,
    seed_sequence: np.random.SeedSequence,
    lower: float,
    upper: float,
) -> np.ndarray:
    sampler = LatinHypercube(
        1,
        scramble=True,
        seed=np.random.default_rng(seed_sequence),
    )
    return lower + sampler.random(nsamples)[:, 0] * (upper - lower)


def _column_names(table: Any) -> dict[str, str]:
    names = tuple(table.columns.names)
    lowered = {str(name).lower(): str(name) for name in names}
    missing = [name for name in REQUIRED_CATALOG_COLUMNS if name not in lowered]
    if missing:
        raise ValueError(f"DESI catalog is missing required columns: {missing}")
    return lowered


def _float_column(block: Any, names: Mapping[str, str], name: str) -> np.ndarray:
    return np.asarray(block[names[name]], dtype=np.float64)


def _eligibility_mask(
    block: Any,
    names: Mapping[str, str],
    *,
    rmag_range: tuple[float, float],
    hlr_range: tuple[float, float],
    image_snr_range: tuple[float, float],
    halpha_snr_range: tuple[float, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = {
        "rmag": _float_column(block, names, "rmag"),
        "hlr": _float_column(block, names, "hlr"),
        "img_snr": _float_column(block, names, "img_snr"),
        "halpha_snr": _float_column(block, names, "halpha_snr"),
    }
    rmag_ok = np.isfinite(values["rmag"]) & (
        values["rmag"] >= rmag_range[0]
    ) & (values["rmag"] <= rmag_range[1])
    hlr_ok = (
        np.isfinite(values["hlr"])
        & (values["hlr"] >= hlr_range[0])
        & (values["hlr"] <= hlr_range[1])
    )
    image_ok = np.isfinite(values["img_snr"]) & (
        values["img_snr"] >= image_snr_range[0]
    ) & (values["img_snr"] <= image_snr_range[1])
    halpha_ok = np.isfinite(values["halpha_snr"]) & (
        values["halpha_snr"] >= halpha_snr_range[0]
    ) & (values["halpha_snr"] <= halpha_snr_range[1])
    masks = {
        "rmag_support": rmag_ok,
        "hlr_support": hlr_ok,
        "image_snr_support": image_ok,
        "halpha_snr_support": halpha_ok,
    }
    return rmag_ok & hlr_ok & image_ok & halpha_ok, values | masks


def _json_header_value(value: Any) -> bool | int | float | str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _source_identity(path: Path, hdu: Any, row_count: int) -> dict[str, Any]:
    stat = path.stat()
    header_keys = (
        "XUSAMPLE",
        "SELMODE",
        "DENSPASS",
        "EXTNAME",
        "EXTVER",
        "DATE",
    )
    header = {
        key: _json_header_value(hdu.header[key])
        for key in header_keys
        if key in hdu.header
    }
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "extension": str(hdu.name),
        "row_count": int(row_count),
        "header": header,
    }


def sample_catalog_rows(
    catalog_path: str | Path,
    nsamples: int,
    *,
    rng: np.random.Generator,
    extension: str = DEFAULT_CATALOG_EXTENSION,
    block_size: int = DEFAULT_CATALOG_BLOCK_SIZE,
    rmag_range: tuple[float, float] = DEFAULT_RMAG_RANGE,
    hlr_range: tuple[float, float] = (0.1, 5.0),
    image_snr_range: tuple[float, float] = DEFAULT_IMAGE_SNR_RANGE,
    halpha_snr_range: tuple[float, float] = DEFAULT_CENTRAL_HALPHA_SNR_RANGE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Uniformly select jointly eligible catalog rows without replacement."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    path = Path(catalog_path).expanduser().resolve(strict=True)

    support_counts = {
        "rmag_support": 0,
        "hlr_support": 0,
        "image_snr_support": 0,
        "halpha_snr_support": 0,
    }
    finite_counts = {"z": 0, "xu_effective_weight": 0}
    eligible_count = 0

    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        try:
            hdu = hdul[extension]
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"DESI catalog {path} has no FITS extension {extension!r}"
            ) from exc
        table = hdu.data
        if table is None:
            raise ValueError(f"DESI catalog extension {extension!r} has no table")
        names = _column_names(table)
        row_count = len(table)
        source_identity = _source_identity(path, hdu, row_count)

        for start in range(0, row_count, block_size):
            stop = min(start + block_size, row_count)
            block = table[start:stop]
            eligible, values = _eligibility_mask(
                block,
                names,
                rmag_range=rmag_range,
                hlr_range=hlr_range,
                image_snr_range=image_snr_range,
                halpha_snr_range=halpha_snr_range,
            )
            for key in support_counts:
                support_counts[key] += int(np.count_nonzero(values[key]))
            finite_counts["z"] += int(
                np.count_nonzero(np.isfinite(_float_column(block, names, "z")))
            )
            finite_counts["xu_effective_weight"] += int(
                np.count_nonzero(
                    np.isfinite(_float_column(block, names, "xu_effective_weight"))
                )
            )
            eligible_count += int(np.count_nonzero(eligible))

        if nsamples > eligible_count:
            raise ValueError(
                f"requested {nsamples} rows but only {eligible_count} catalog "
                "rows satisfy the joint simulator-support criteria"
            )

        selected_eligible_ranks = np.sort(
            rng.choice(eligible_count, size=nsamples, replace=False)
        )
        selected = {
            RMAG_TRUE_COLUMN: np.empty(nsamples, dtype=np.float64),
            "hlr": np.empty(nsamples, dtype=np.float64),
            IMAGE_SNR_COLUMN: np.empty(nsamples, dtype=np.float64),
            CENTRAL_HALPHA_SNR_COLUMN: np.empty(nsamples, dtype=np.float64),
            "source_catalog_row": np.empty(nsamples, dtype=np.int64),
            "source_targetid": np.empty(nsamples, dtype=np.int64),
            "source_catalog_z": np.empty(nsamples, dtype=np.float64),
            "source_hlr_raw": np.empty(nsamples, dtype=np.float64),
            "source_xu_effective_weight": np.empty(nsamples, dtype=np.float64),
        }
        eligible_seen = 0
        rank_cursor = 0
        output_cursor = 0
        for start in range(0, row_count, block_size):
            stop = min(start + block_size, row_count)
            block = table[start:stop]
            eligible, values = _eligibility_mask(
                block,
                names,
                rmag_range=rmag_range,
                hlr_range=hlr_range,
                image_snr_range=image_snr_range,
                halpha_snr_range=halpha_snr_range,
            )
            local_eligible = np.flatnonzero(eligible)
            next_eligible_seen = eligible_seen + local_eligible.size
            next_rank_cursor = int(
                np.searchsorted(
                    selected_eligible_ranks,
                    next_eligible_seen,
                    side="left",
                )
            )
            if next_rank_cursor > rank_cursor:
                block_ranks = (
                    selected_eligible_ranks[rank_cursor:next_rank_cursor]
                    - eligible_seen
                )
                local_rows = local_eligible[block_ranks]
                count = local_rows.size
                target = slice(output_cursor, output_cursor + count)
                raw_hlr = values["hlr"][local_rows]
                selected[RMAG_TRUE_COLUMN][target] = values["rmag"][local_rows]
                selected["hlr"][target] = raw_hlr
                selected[IMAGE_SNR_COLUMN][target] = values["img_snr"][local_rows]
                selected[CENTRAL_HALPHA_SNR_COLUMN][target] = values[
                    "halpha_snr"
                ][local_rows]
                selected["source_catalog_row"][target] = start + local_rows
                selected["source_targetid"][target] = np.asarray(
                    block[names["targetid"]][local_rows], dtype=np.int64
                )
                selected["source_catalog_z"][target] = _float_column(
                    block, names, "z"
                )[local_rows]
                selected["source_hlr_raw"][target] = raw_hlr
                selected["source_xu_effective_weight"][target] = _float_column(
                    block, names, "xu_effective_weight"
                )[local_rows]
                output_cursor += count
                rank_cursor = next_rank_cursor
            eligible_seen = next_eligible_seen

    if output_cursor != nsamples or rank_cursor != nsamples:
        raise RuntimeError("catalog row selection did not materialize every chosen rank")
    permutation = rng.permutation(nsamples)
    selected = {key: value[permutation] for key, value in selected.items()}
    frame = pd.DataFrame(selected)
    audit = {
        "source_catalog": source_identity,
        "eligible_row_count": int(eligible_count),
        "support_counts": {key: int(value) for key, value in support_counts.items()},
        "finite_provenance_counts": finite_counts,
    }
    return frame, audit


def generate_catalog_test_set(
    nsamples: int,
    *,
    catalog_path: str | Path,
    parameter_limits: Mapping[str, tuple[float, float]],
    seed: int | None = None,
    catalog_extension: str = DEFAULT_CATALOG_EXTENSION,
    catalog_block_size: int = DEFAULT_CATALOG_BLOCK_SIZE,
    tf_prior: TFPrior | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate one TF-conformed, catalog-backed simulator test population."""

    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    missing_parameters = [
        name
        for name in (*UNIFORMLY_SAMPLED_PARAMETERS, "sini", "vcirc", "hlr")
        if name not in parameter_limits
    ]
    if missing_parameters:
        raise ValueError(f"parameter_limits is missing: {missing_parameters}")
    prior = TFPrior() if tf_prior is None else tf_prior
    if (prior.vcirc_min, prior.vcirc_max) != tuple(parameter_limits["vcirc"]):
        raise ValueError("TF vcirc bounds must match the simulator parameter range")

    root_seed = np.random.SeedSequence(seed)
    (
        catalog_seed,
        parameter_seed,
        inclination_seed,
        tf_quantile_seed,
        halpha_seed,
    ) = root_seed.spawn(5)
    catalog_rows, audit = sample_catalog_rows(
        catalog_path,
        nsamples,
        rng=np.random.default_rng(catalog_seed),
        extension=catalog_extension,
        block_size=catalog_block_size,
        hlr_range=tuple(parameter_limits["hlr"]),
    )

    ordinary_unit = LatinHypercube(
        len(UNIFORMLY_SAMPLED_PARAMETERS),
        scramble=True,
        seed=np.random.default_rng(parameter_seed),
    ).random(nsamples)
    ordinary = {}
    for column, name in enumerate(UNIFORMLY_SAMPLED_PARAMETERS):
        lower, upper = parameter_limits[name]
        ordinary[name] = lower + ordinary_unit[:, column] * (upper - lower)

    cosi = _lhs_column(nsamples, inclination_seed, 0.0, 1.0)
    sini = np.sqrt(np.maximum(0.0, 1.0 - np.square(cosi)))
    tf_quantiles = _lhs_column(nsamples, tf_quantile_seed, 0.0, 1.0)
    vcirc = sample_truncated_tf_vcirc(
        catalog_rows[RMAG_TRUE_COLUMN].to_numpy(),
        prior,
        quantiles=tf_quantiles,
    )
    log10_halpha_flux = _lhs_column(
        nsamples,
        halpha_seed,
        *DEFAULT_HALPHA_LOG10_FLUX_RANGE,
    )

    samples = pd.DataFrame(
        {
            "g1": ordinary["g1"],
            "g2": ordinary["g2"],
            "theta_int": ordinary["theta_int"],
            "sini": sini,
            "v0": ordinary["v0"],
            "vcirc": vcirc,
            "rscale": ordinary["rscale"],
            "hlr": catalog_rows["hlr"].to_numpy(),
            RMAG_TRUE_COLUMN: catalog_rows[RMAG_TRUE_COLUMN].to_numpy(),
            HALPHA_FLUX_TRUE_COLUMN: np.power(10.0, log10_halpha_flux),
            IMAGE_SNR_COLUMN: catalog_rows[IMAGE_SNR_COLUMN].to_numpy(),
            CENTRAL_HALPHA_SNR_COLUMN: catalog_rows[
                CENTRAL_HALPHA_SNR_COLUMN
            ].to_numpy(),
            FIBER_LAYOUT_COLUMN: GALAXY_AXIS_FIBER_LAYOUT,
            OBSERVATION_MODEL_VERSION_COLUMN: np.full(
                nsamples,
                OBSERVATION_MODEL_VERSION,
                dtype=np.int16,
            ),
        }
    )
    for name in SOURCE_PROVENANCE_COLUMNS:
        samples[name] = catalog_rows[name].to_numpy()

    entropy = root_seed.entropy
    if isinstance(entropy, np.ndarray):
        seed_value: int | list[int] = entropy.astype(int).tolist()
    elif isinstance(entropy, (tuple, list)):
        seed_value = [int(value) for value in entropy]
    else:
        seed_value = int(entropy)
    eligible_count = audit["eligible_row_count"]
    manifest = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "analysis_mode": TEST_SET_ANALYSIS_MODE,
        "population": TEST_SET_POPULATION,
        "sample_count": int(nsamples),
        "redshift": SIMULATION_REDSHIFT,
        "simulation_redshift": SIMULATION_REDSHIFT,
        "seed": seed_value,
        "tf": prior.to_dict(),
        "source_catalog": audit["source_catalog"],
        "catalog_sampling": {
            "method": "uniform_joint_rows_without_replacement",
            "selected_row_count": int(nsamples),
            "eligible_row_count": int(eligible_count),
            "selected_fraction_of_eligible": float(nsamples / eligible_count),
            "block_size": int(catalog_block_size),
            "eligibility": {
                "rmag": {"finite": True, "minimum": 15.0, "maximum": 23.4},
                "hlr": {
                    "finite": True,
                    "minimum": float(parameter_limits["hlr"][0]),
                    "maximum": float(parameter_limits["hlr"][1]),
                    "bounds": "inclusive",
                },
                "image_snr": {
                    "finite": True,
                    "minimum": float(DEFAULT_IMAGE_SNR_RANGE[0]),
                    "maximum": float(DEFAULT_IMAGE_SNR_RANGE[1]),
                },
                "halpha_snr": {
                    "finite": True,
                    "minimum": float(DEFAULT_CENTRAL_HALPHA_SNR_RANGE[0]),
                    "maximum": float(DEFAULT_CENTRAL_HALPHA_SNR_RANGE[1]),
                },
            },
            "support_counts": audit["support_counts"],
            "finite_provenance_counts": audit["finite_provenance_counts"],
            "joint_columns": [
                "rmag",
                "hlr",
                "img_snr",
                "halpha_snr",
            ],
            "xu_effective_weight_policy": "provenance_only_not_sampling_weight",
            "catalog_redshift_policy": "record_source_value_but_do_not_simulate_at_it",
        },
        "parameter_sampling": {
            "uniform_latin_hypercube": {
                name: [float(value) for value in parameter_limits[name]]
                for name in UNIFORMLY_SAMPLED_PARAMETERS
            },
            "inclination": {
                "distribution": "cosi_uniform_0_1_latin_hypercube",
                "transform": "sini=sqrt(1-cosi**2)",
            },
            "vcirc": {
                "distribution": "truncated_tf_conditional_on_catalog_rmag",
                "quantile_design": "uniform_0_1_latin_hypercube",
            },
            "halpha_flux_true": {
                "distribution": "log10_uniform_latin_hypercube",
                "log10_range": [
                    float(DEFAULT_HALPHA_LOG10_FLUX_RANGE[0]),
                    float(DEFAULT_HALPHA_LOG10_FLUX_RANGE[1]),
                ],
            },
        },
    }
    return samples, manifest


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a generated sample table."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_generation_manifest(
    manifest: Mapping[str, Any],
    sample_table_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Write the deterministic sidecar consumed by dataset/cache test mode."""

    sample_path = Path(sample_table_path).expanduser().resolve(strict=True)
    destination = (
        sample_path.with_suffix(".manifest.json")
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    payload = dict(manifest)
    required = {
        "schema",
        "analysis_mode",
        "population",
        "sample_count",
        "redshift",
        "simulation_redshift",
        "tf",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"generation manifest is missing required keys: {missing}")
    payload["sample_table"] = {
        "path": str(sample_path),
        "format": "csv",
        "sha256": sha256_file(sample_path),
        "row_count": int(payload["sample_count"]),
        "id_column": "ID",
        "id_policy": "zero_based_contiguous_row_index",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
