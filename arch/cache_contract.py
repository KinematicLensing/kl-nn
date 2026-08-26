"""Fail-closed reader for current posterior-cache partitions.

The cache writer stores arrays in separate directories, so discovering each
directory independently is unsafe: two complete-looking sets of arrays can
come from different checkpoints or partition layouts.  This module validates
the manifests once and returns the only ordered file list reports should use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


CACHE_SCHEMA = "klnn-posterior-cache-v2"
LEGACY_CACHE_SCHEMA = "klnn-posterior-cache-v1"
SUPPORTED_CACHE_SCHEMAS = (LEGACY_CACHE_SCHEMA, CACHE_SCHEMA)
CURRENT_FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
    "halpha_flux_true",
)
CURRENT_TARGET_TRANSFORMS = {
    name: "log10" if name == "halpha_flux_true" else "identity"
    for name in CURRENT_FEATURE_NAMES
}
EXPECTED_DENSITY_COORDINATES = {
    "stored_base_log_prob": "normalized_target_coordinates",
    "map_selection": "physical_target_coordinates",
    "map_jacobian": "subtract_logabsdet_dphysical_dnormalized",
}
EXPECTED_OBSERVATION_MODEL = {
    "schema_version": 3,
    "context_fields": ["rmag_true", "image_snr", "central_halpha_snr"],
    "halpha_flux_semantics": (
        "central_fiber_integrated_after_seeing_before_instrument"
    ),
    "image_snr_distribution": "uniform",
    "central_halpha_snr_distribution": "uniform",
    "center_exposure_s": 180.0,
    "offset_exposure_s": 600.0,
}
LEGACY_REQUIRED_CACHE_ARRAYS = (
    "sample",
    "base_log_prob",
    "posterior_tf_log_ratio",
    "posterior_tf_log_weight",
    "posterior_tf_weight",
    "posterior_tf_ess",
    "posterior_tf_ess_fraction",
    "posterior_tf_max_weight",
    "posterior_tf_log_mean_ratio",
    "population_tf_log_ratio",
    "truth",
    "rmag_true",
    "spectral_reference_quality",
    "proposal_map_estimates",
    "proposal_mean_estimates",
    "tf_target_map_estimates",
    "tf_target_mean_estimates",
)
REQUIRED_CACHE_ARRAYS = tuple(
    name
    for name in LEGACY_REQUIRED_CACHE_ARRAYS
    if name != "spectral_reference_quality"
) + (
    "image_snr",
    "central_halpha_snr",
    "image_noise_sigma",
    "central_spectral_noise_sigma",
)
EXPECTED_SYMMETRY = {
    "policy": "original_plus_r90_equal_mixture",
    "rotated_joint_rows_inverse_aligned": True,
}
REQUIRED_TF_FIELDS = {
    "slope",
    "intercept",
    "scatter_dex",
    "vcirc_min",
    "vcirc_max",
    "magnitude",
    "magnitude_measurement_error",
    "posterior_log_ratio",
    "posterior_log_weight",
    "posterior_weight_normalization",
    "population_log_ratio_normalization",
    "resampling",
}
LEGACY_REQUIRED_PROVENANCE_FIELDS = {
    "image_noise_sigma",
    "spectral_reference_line_norm",
    "matched_group_size",
    "posterior_sample_seed",
    "image_noise_seed",
    "spectral_noise_seed",
    "spectral_quality_seed",
}
REQUIRED_PROVENANCE_FIELDS = {
    "matched_group_size",
    "posterior_sample_seed",
    "image_noise_seed",
    "spectral_noise_seed",
}
LEGACY_SEED_FIELDS = {
    "posterior_sample_seed",
    "image_noise_seed",
    "spectral_noise_seed",
    "spectral_quality_seed",
}
SEED_FIELDS = {
    "posterior_sample_seed",
    "image_noise_seed",
    "spectral_noise_seed",
}
PART_RE = re.compile(r"^part(\d+)of(\d+)\.(json|npy)$")
PARTITION_SEED_STRIDE = 1_000_003


@dataclass(frozen=True)
class CachePartitions:
    """Validated, ordered metadata for one complete posterior cache."""

    root: Path
    manifests: tuple[dict[str, Any], ...]
    manifest_paths: tuple[Path, ...]
    labels: tuple[str, ...]
    row_ranges: tuple[tuple[int, int], ...]
    feature_names: tuple[str, ...]
    dataset_size: int
    observation_provenance: dict[str, Any]
    files: dict[str, tuple[Path, ...]]

    @property
    def total_rows(self) -> int:
        return self.row_ranges[-1][1]


def _fail(path: Path, message: str) -> ValueError:
    return ValueError(f"Invalid cache manifest {path}: {message}")


def _require_mapping(payload: dict[str, Any], key: str, path: Path) -> dict:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise _fail(path, f"{key!r} must be an object")
    return value


def _require_int(value: Any, *, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(path, f"{name} must be an integer")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _validate_tf(tf: dict[str, Any], path: Path) -> None:
    missing = REQUIRED_TF_FIELDS - set(tf)
    if missing:
        raise _fail(path, f"tf is missing fields {sorted(missing)}")
    for name in ("slope", "intercept", "scatter_dex", "vcirc_min", "vcirc_max"):
        value = tf[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _fail(path, f"tf.{name} must be numeric")
        if not math.isfinite(float(value)):
            raise _fail(path, f"tf.{name} must be finite")
    if float(tf["slope"]) == 0.0:
        raise _fail(path, "tf.slope must be non-zero")
    if float(tf["scatter_dex"]) <= 0.0:
        raise _fail(path, "tf.scatter_dex must be positive")
    if not 0.0 < float(tf["vcirc_min"]) < float(tf["vcirc_max"]):
        raise _fail(path, "tf vcirc bounds must be positive and increasing")
    if tf["magnitude"] != "rmag_true":
        raise _fail(path, "tf.magnitude must be rmag_true")
    if tf["magnitude_measurement_error"] != 0.0:
        raise _fail(path, "TF magnitude measurement error must be zero")
    if tf["posterior_weight_normalization"] != "within_galaxy":
        raise _fail(path, "posterior TF weights must be normalized within galaxy")
    if tf["population_log_ratio_normalization"] != "global_after_partition_concat":
        raise _fail(path, "population TF ratios must be normalized after concatenation")
    if tf["resampling"] is not False:
        raise _fail(path, "TF cache must preserve candidates without resampling")
    for name in ("posterior_log_ratio", "posterior_log_weight"):
        if not isinstance(tf[name], str) or not tf[name]:
            raise _fail(path, f"tf.{name} must document its stored quantity")


def _validate_populations(populations: dict[str, Any], path: Path) -> None:
    if set(populations) != {"proposal", "tf_target"}:
        raise _fail(path, "posterior_populations must contain proposal and tf_target")
    if any(not isinstance(value, str) or not value for value in populations.values()):
        raise _fail(path, "posterior population descriptions must be non-empty strings")


def _validate_provenance(
    provenance: dict[str, Any], path: Path, schema: str
) -> None:
    required = (
        LEGACY_REQUIRED_PROVENANCE_FIELDS
        if schema == LEGACY_CACHE_SCHEMA
        else REQUIRED_PROVENANCE_FIELDS
    )
    seed_fields = (
        LEGACY_SEED_FIELDS if schema == LEGACY_CACHE_SCHEMA else SEED_FIELDS
    )
    missing = required - set(provenance)
    if missing:
        raise _fail(
            path, f"observation_provenance is missing fields {sorted(missing)}"
        )
    if schema == LEGACY_CACHE_SCHEMA:
        for name in ("image_noise_sigma", "spectral_reference_line_norm"):
            value = provenance[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise _fail(path, f"observation_provenance.{name} must be numeric")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise _fail(
                    path,
                    f"observation_provenance.{name} must be positive and finite",
                )
    matched_group_size = _require_int(
        provenance["matched_group_size"],
        name="observation_provenance.matched_group_size",
        path=path,
    )
    if matched_group_size <= 0:
        raise _fail(path, "observation_provenance.matched_group_size must be positive")
    for name in seed_fields:
        _require_int(
            provenance[name],
            name=f"observation_provenance.{name}",
            path=path,
        )


def _expected_array_shape(
    name: str, rows: int, draws: int, features: int
) -> tuple[int, ...]:
    if name == "sample":
        return rows, draws, features
    if name in {
        "base_log_prob",
        "posterior_tf_log_ratio",
        "posterior_tf_log_weight",
        "posterior_tf_weight",
    }:
        return rows, draws
    if name in {
        "posterior_tf_ess",
        "posterior_tf_ess_fraction",
        "posterior_tf_max_weight",
        "posterior_tf_log_mean_ratio",
        "population_tf_log_ratio",
        "rmag_true",
        "spectral_reference_quality",
        "image_snr",
        "central_halpha_snr",
        "image_noise_sigma",
        "central_spectral_noise_sigma",
    }:
        return (rows,)
    if name in {
        "truth",
        "proposal_map_estimates",
        "tf_target_map_estimates",
    }:
        return rows, features
    if name in {"proposal_mean_estimates", "tf_target_mean_estimates"}:
        return rows, 3, features
    raise AssertionError(f"No shape contract for cache array {name!r}")


def load_cache_partitions(root: str | Path) -> CachePartitions:
    """Validate and return every partition of one current cache.

    Validation includes provenance equality, the complete row interval, and
    every array file recorded by the writer.  Seed values may vary by partition,
    but must follow the writer's documented independent-stream construction.
    """

    root = Path(root)
    meta_dir = root / "meta"
    candidates = []
    for path in meta_dir.glob("part*of*.json"):
        match = PART_RE.fullmatch(path.name)
        if match is None or match.group(3) != "json":
            raise ValueError(f"Malformed cache manifest filename: {path}")
        candidates.append((int(match.group(1)), int(match.group(2)), path))
    if not candidates:
        raise FileNotFoundError(f"No cache manifests in {meta_dir}")
    totals = {total for _, total, _ in candidates}
    if len(totals) != 1:
        raise ValueError(f"Mixed partition totals in {meta_dir}: {sorted(totals)}")
    total = totals.pop()
    candidates.sort(key=lambda item: item[0])
    indices = [index for index, _, _ in candidates]
    if indices != list(range(total)):
        raise ValueError(
            f"Incomplete cache manifests in {meta_dir}: found {indices}, expected "
            f"{list(range(total))}"
        )

    payloads: list[dict[str, Any]] = []
    labels: list[str] = []
    row_ranges: list[tuple[int, int]] = []
    reference: dict[str, str] | None = None
    reference_non_seed_provenance: str | None = None
    reference_schema: str | None = None
    reference_base_seed: int | None = None
    reference_sample_tail: tuple[int, int] | None = None
    reference_dataset_size: int | None = None
    previous_end = 0

    for expected_index, (_, _, path) in enumerate(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read cache manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise _fail(path, "top level must be an object")
        schema = payload.get("schema")
        if schema not in SUPPORTED_CACHE_SCHEMAS:
            raise _fail(path, f"schema must be one of {SUPPORTED_CACHE_SCHEMAS!r}")
        if reference_schema is None:
            reference_schema = schema
        elif schema != reference_schema:
            raise _fail(path, "cache schema differs across partitions")

        for name in ("model_name", "checkpoint", "dataset"):
            value = payload.get(name)
            if not isinstance(value, str) or not value:
                raise _fail(path, f"{name} must be a non-empty string")
        dataset_size = _require_int(
            payload.get("dataset_size"), name="dataset_size", path=path
        )
        if dataset_size <= 0:
            raise _fail(path, "dataset_size must be positive")
        if reference_dataset_size is None:
            reference_dataset_size = dataset_size
        elif dataset_size != reference_dataset_size:
            raise _fail(path, "partition invariants differ for ['dataset_size']")

        partition = _require_mapping(payload, "partition", path)
        index = _require_int(partition.get("index"), name="partition.index", path=path)
        part_total = _require_int(
            partition.get("total"), name="partition.total", path=path
        )
        label = partition.get("label")
        expected_label = f"part{expected_index}of{total}"
        if index != expected_index or part_total != total or label != expected_label:
            raise _fail(
                path,
                "filename and partition.{index,total,label} do not identify the same part",
            )
        start = _require_int(
            partition.get("galaxy_start"), name="partition.galaxy_start", path=path
        )
        end = _require_int(
            partition.get("galaxy_end"), name="partition.galaxy_end", path=path
        )
        if start != previous_end or end <= start:
            raise _fail(
                path,
                f"row range [{start}, {end}) is not a positive continuation of {previous_end}",
            )
        previous_end = end

        feature_names = payload.get("feature_names")
        if not isinstance(feature_names, list) or tuple(feature_names) != CURRENT_FEATURE_NAMES:
            raise _fail(path, f"feature_names must equal {CURRENT_FEATURE_NAMES!r}")
        if schema == CACHE_SCHEMA:
            parameter_ranges = _require_mapping(
                payload, "physical_parameter_ranges", path
            )
            if tuple(parameter_ranges) != CURRENT_FEATURE_NAMES:
                raise _fail(
                    path,
                    "physical_parameter_ranges must follow feature_names order",
                )
            for name, bounds in parameter_ranges.items():
                if (
                    not isinstance(bounds, list)
                    or len(bounds) != 2
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in bounds
                    )
                    or float(bounds[0]) >= float(bounds[1])
                ):
                    raise _fail(
                        path,
                        f"physical_parameter_ranges.{name} must increase finitely",
                    )
            if payload.get("target_transforms") != CURRENT_TARGET_TRANSFORMS:
                raise _fail(
                    path,
                    f"target_transforms must equal {CURRENT_TARGET_TRANSFORMS!r}",
                )
            if payload.get("density_coordinates") != EXPECTED_DENSITY_COORDINATES:
                raise _fail(
                    path,
                    f"density_coordinates must equal {EXPECTED_DENSITY_COORDINATES!r}",
                )
            if payload.get("observation_model") != EXPECTED_OBSERVATION_MODEL:
                raise _fail(
                    path,
                    f"observation_model must equal {EXPECTED_OBSERVATION_MODEL!r}",
                )
        sample_shape = payload.get("sample_shape")
        if (
            not isinstance(sample_shape, list)
            or len(sample_shape) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in sample_shape)
        ):
            raise _fail(path, "sample_shape must contain three integers")
        rows, draws, features = sample_shape
        if rows != end - start or draws <= 0 or draws % 2 or features != len(feature_names):
            raise _fail(path, "sample_shape disagrees with rows, R90 pairing, or features")
        sample_tail = (draws, features)
        if reference_sample_tail is None:
            reference_sample_tail = sample_tail
        elif sample_tail != reference_sample_tail:
            raise _fail(path, "sample shape differs across cache partitions")

        symmetry = _require_mapping(payload, "symmetry", path)
        if symmetry != EXPECTED_SYMMETRY:
            raise _fail(path, f"symmetry must equal {EXPECTED_SYMMETRY!r}")
        tf = _require_mapping(payload, "tf", path)
        _validate_tf(tf, path)
        populations = _require_mapping(payload, "posterior_populations", path)
        _validate_populations(populations, path)
        provenance = _require_mapping(payload, "observation_provenance", path)
        _validate_provenance(provenance, path, schema)

        invariant = {
            "model_name": payload["model_name"],
            "checkpoint": payload["checkpoint"],
            "dataset": payload["dataset"],
            "dataset_size": str(dataset_size),
            "feature_names": _canonical(feature_names),
            "symmetry": _canonical(symmetry),
            "tf": _canonical(tf),
            "posterior_populations": _canonical(populations),
            "schema": schema,
        }
        if schema == CACHE_SCHEMA:
            invariant.update(
                {
                    "physical_parameter_ranges": _canonical(
                        payload["physical_parameter_ranges"]
                    ),
                    "target_transforms": _canonical(
                        payload["target_transforms"]
                    ),
                    "density_coordinates": _canonical(
                        payload["density_coordinates"]
                    ),
                    "observation_model": _canonical(
                        payload["observation_model"]
                    ),
                }
            )
        seed_fields = (
            LEGACY_SEED_FIELDS if schema == LEGACY_CACHE_SCHEMA else SEED_FIELDS
        )
        non_seed_provenance = _canonical(
            {key: value for key, value in provenance.items() if key not in seed_fields}
        )
        if reference is None:
            reference = invariant
            reference_non_seed_provenance = non_seed_provenance
        else:
            differing = [name for name, value in invariant.items() if value != reference[name]]
            if differing:
                raise _fail(path, f"partition invariants differ for {differing}")
            if non_seed_provenance != reference_non_seed_provenance:
                raise _fail(path, "non-seed observation provenance differs across partitions")

        posterior_seed = int(provenance["posterior_sample_seed"])
        image_seed = int(provenance["image_noise_seed"])
        if posterior_seed != image_seed:
            raise _fail(path, "posterior and image seed must identify the same partition stream")
        if int(provenance["spectral_noise_seed"]) != image_seed + 101:
            raise _fail(path, "spectral noise seed must equal image seed + 101")
        if (
            schema == LEGACY_CACHE_SCHEMA
            and int(provenance["spectral_quality_seed"]) != image_seed + 307
        ):
            raise _fail(path, "spectral quality seed must equal image seed + 307")
        base_seed = posterior_seed - PARTITION_SEED_STRIDE * expected_index
        if reference_base_seed is None:
            reference_base_seed = base_seed
        elif base_seed != reference_base_seed:
            raise _fail(path, "partition posterior seeds are not independent indexed streams")

        files = _require_mapping(payload, "files", path)
        supplied_arrays = set(files)
        required_arrays = (
            LEGACY_REQUIRED_CACHE_ARRAYS
            if schema == LEGACY_CACHE_SCHEMA
            else REQUIRED_CACHE_ARRAYS
        )
        expected_arrays = set(required_arrays)
        if supplied_arrays != expected_arrays:
            raise _fail(
                path,
                "files must exactly match the current cache arrays; "
                f"missing={sorted(expected_arrays - supplied_arrays)}, "
                f"extra={sorted(supplied_arrays - expected_arrays)}",
            )
        for name in required_arrays:
            expected_relative = f"{name}/{expected_label}.npy"
            if files[name] != expected_relative:
                raise _fail(
                    path,
                    f"files.{name} must be {expected_relative!r}, got {files[name]!r}",
                )

        payloads.append(payload)
        labels.append(expected_label)
        row_ranges.append((start, end))

    assert reference_dataset_size is not None
    if previous_end != reference_dataset_size:
        raise ValueError(
            "Cache partitions do not cover the complete dataset: "
            f"covered rows [0, {previous_end}), dataset_size={reference_dataset_size}"
        )

    ordered_files: dict[str, tuple[Path, ...]] = {}
    assert reference_schema is not None
    required_arrays = (
        LEGACY_REQUIRED_CACHE_ARRAYS
        if reference_schema == LEGACY_CACHE_SCHEMA
        else REQUIRED_CACHE_ARRAYS
    )
    for name in required_arrays:
        paths = tuple(root / name / f"{label}.npy" for label in labels)
        found = []
        for path in (root / name).glob("part*of*.npy"):
            match = PART_RE.fullmatch(path.name)
            if match is None or match.group(3) != "npy":
                raise ValueError(
                    f"Malformed cache array partition filename: {path}"
                )
            found.append((int(match.group(1)), int(match.group(2)), path))
        found.sort(key=lambda item: item[0])
        found_indices = [index for index, _, _ in found]
        found_totals = {total for _, total, _ in found}
        found_names = [path.name for _, _, path in found]
        expected_names = [path.name for path in paths]
        if (
            found_indices != list(range(len(paths)))
            or found_totals != {len(paths)}
            or found_names != expected_names
        ):
            raise ValueError(
                f"Cache array {name!r} partitions do not exactly match manifests: "
                f"found {found_names}, expected {expected_names}"
            )
        for path, (start, end), payload in zip(paths, row_ranges, payloads):
            if not path.is_file():
                raise FileNotFoundError(f"Manifest-recorded cache array is missing: {path}")
            array = np.load(path, mmap_mode="r")
            expected_shape = _expected_array_shape(
                name,
                end - start,
                int(payload["sample_shape"][1]),
                len(CURRENT_FEATURE_NAMES),
            )
            if array.shape != expected_shape:
                raise ValueError(
                    f"Cache array {path} has shape {array.shape}; expected {expected_shape}"
                )
        ordered_files[name] = paths

    assert reference is not None and reference_non_seed_provenance is not None
    seed_fields = (
        LEGACY_SEED_FIELDS
        if reference_schema == LEGACY_CACHE_SCHEMA
        else SEED_FIELDS
    )
    non_seed_provenance = {
        key: value
        for key, value in payloads[0]["observation_provenance"].items()
        if key not in seed_fields
    }
    return CachePartitions(
        root=root,
        manifests=tuple(payloads),
        manifest_paths=tuple(path for _, _, path in candidates),
        labels=tuple(labels),
        row_ranges=tuple(row_ranges),
        feature_names=CURRENT_FEATURE_NAMES,
        dataset_size=reference_dataset_size,
        observation_provenance=non_seed_provenance,
        files=ordered_files,
    )


def load_partitioned_array(
    partitions: CachePartitions, name: str, *, axis: int = 0
) -> np.ndarray:
    """Concatenate one array using the order guaranteed by its manifests."""

    try:
        paths = partitions.files[name]
    except KeyError as exc:
        raise KeyError(f"Unknown cache array {name!r}") from exc
    return np.concatenate(
        [np.load(path, mmap_mode="r") for path in paths], axis=axis
    )
