import json

import numpy as np
import pytest

from cache_contract import (
    CACHE_SCHEMA,
    CURRENT_FEATURE_NAMES,
    CURRENT_TARGET_TRANSFORMS,
    EXPECTED_DENSITY_COORDINATES,
    EXPECTED_OBSERVATION_MODEL,
    LEGACY_CACHE_SCHEMA,
    LEGACY_REQUIRED_CACHE_ARRAYS,
    PARTITION_SEED_STRIDE,
    REQUIRED_CACHE_ARRAYS,
    load_cache_partitions,
    load_partitioned_array,
)


def _array(name, rows, draws, start):
    features = len(CURRENT_FEATURE_NAMES)
    if name == "sample":
        return np.zeros((rows, draws, features), dtype=np.float32)
    if name in {
        "base_log_prob",
        "posterior_tf_log_ratio",
        "posterior_tf_log_weight",
        "posterior_tf_weight",
    }:
        return np.zeros((rows, draws), dtype=np.float32)
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
        return np.ones(rows, dtype=np.float64)
    if name in {
        "truth",
        "proposal_map_estimates",
        "tf_target_map_estimates",
    }:
        result = np.zeros((rows, features), dtype=np.float32)
        if name == "truth":
            result[:, 0] = np.arange(start, start + rows)
        return result
    if name in {"proposal_mean_estimates", "tf_target_mean_estimates"}:
        return np.zeros((rows, 3, features), dtype=np.float32)
    raise AssertionError(name)


def _manifest(index, total, start, end, draws, *, schema=CACHE_SCHEMA):
    label = f"part{index}of{total}"
    seed = 42 + PARTITION_SEED_STRIDE * index
    if schema == LEGACY_CACHE_SCHEMA:
        provenance = {
            "image_noise_sigma": 1.0,
            "spectral_reference_line_norm": 2.0,
            "matched_group_size": 1,
            "posterior_sample_seed": seed,
            "image_noise_seed": seed,
            "spectral_noise_seed": seed + 101,
            "spectral_quality_seed": seed + 307,
        }
        required_arrays = LEGACY_REQUIRED_CACHE_ARRAYS
    else:
        provenance = {
            "matched_group_size": 1,
            "posterior_sample_seed": seed,
            "image_noise_seed": seed,
            "spectral_noise_seed": seed + 101,
        }
        required_arrays = REQUIRED_CACHE_ARRAYS
    payload = {
        "schema": schema,
        "model_name": "current-model",
        "checkpoint": "/models/current-model/current-modelbest",
        "dataset": "/datasets/current",
        "dataset_size": total * (end - start),
        "partition": {
            "index": index,
            "total": total,
            "label": label,
            "galaxy_start": start,
            "galaxy_end": end,
        },
        "feature_names": list(CURRENT_FEATURE_NAMES),
        "sample_shape": [end - start, draws, len(CURRENT_FEATURE_NAMES)],
        "symmetry": {
            "policy": "original_plus_r90_equal_mixture",
            "rotated_joint_rows_inverse_aligned": True,
        },
        "tf": {
            "slope": -7.22,
            "intercept": 36.0,
            "scatter_dex": 0.1,
            "vcirc_min": 60.0,
            "vcirc_max": 540.0,
            "magnitude": "rmag_true",
            "magnitude_measurement_error": 0.0,
            "posterior_log_ratio": "raw log prior ratio",
            "posterior_log_weight": "within-galaxy log-softmax",
            "posterior_weight_normalization": "within_galaxy",
            "population_log_ratio_normalization": (
                "global_after_partition_concat"
            ),
            "resampling": False,
        },
        "posterior_populations": {
            "proposal": "base posterior",
            "tf_target": "TF-weighted posterior",
        },
        "observation_provenance": provenance,
        "files": {
            name: f"{name}/{label}.npy" for name in required_arrays
        },
    }
    if schema == CACHE_SCHEMA:
        payload.update(
            {
                "physical_parameter_ranges": {
                    "g1": [-0.1, 0.1],
                    "g2": [-0.1, 0.1],
                    "theta_int": [0.0, np.pi],
                    "sini": [0.1, 1.0],
                    "v0": [0.0, 20.0],
                    "vcirc": [60.0, 540.0],
                    "rscale": [0.1, 5.0],
                    "hlr": [0.1, 5.0],
                    "halpha_flux_true": [1.0e-17, 1.0e-14],
                },
                "target_transforms": dict(CURRENT_TARGET_TRANSFORMS),
                "density_coordinates": dict(EXPECTED_DENSITY_COORDINATES),
                "observation_model": dict(EXPECTED_OBSERVATION_MODEL),
            }
        )
    return payload


def _write_cache(root, *, total=2, rows=2, draws=4, schema=CACHE_SCHEMA):
    (root / "meta").mkdir(parents=True)
    required_arrays = (
        LEGACY_REQUIRED_CACHE_ARRAYS
        if schema == LEGACY_CACHE_SCHEMA
        else REQUIRED_CACHE_ARRAYS
    )
    for index in range(total):
        start = index * rows
        end = start + rows
        label = f"part{index}of{total}"
        for name in required_arrays:
            directory = root / name
            directory.mkdir(exist_ok=True)
            np.save(
                directory / f"{label}.npy",
                _array(name, rows, draws, start),
            )
        payload = _manifest(index, total, start, end, draws, schema=schema)
        (root / "meta" / f"{label}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


def _mutate_manifest(root, index, total, mutate):
    path = root / "meta" / f"part{index}of{total}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_complete_cache_contract_orders_parts_and_concatenates_rows(tmp_path):
    root = _write_cache(tmp_path / "cache")
    partitions = load_cache_partitions(root)
    assert partitions.labels == ("part0of2", "part1of2")
    assert partitions.row_ranges == ((0, 2), (2, 4))
    assert partitions.feature_names == CURRENT_FEATURE_NAMES
    assert partitions.dataset_size == 4
    assert partitions.total_rows == 4
    assert set(partitions.files) == set(REQUIRED_CACHE_ARRAYS)
    assert "spectral_reference_quality" not in partitions.files
    assert partitions.observation_provenance == {"matched_group_size": 1}
    truth = load_partitioned_array(partitions, "truth")
    np.testing.assert_array_equal(truth[:, 0], np.arange(4))


def test_legacy_v1_cache_remains_readable(tmp_path):
    root = _write_cache(
        tmp_path / "legacy-cache", schema=LEGACY_CACHE_SCHEMA
    )
    partitions = load_cache_partitions(root)
    assert "spectral_reference_quality" in partitions.files
    assert "image_snr" not in partitions.files
    assert partitions.observation_provenance == {
        "image_noise_sigma": 1.0,
        "spectral_reference_line_norm": 2.0,
        "matched_group_size": 1,
    }


def test_numeric_partition_order_supports_more_than_ten_parts(tmp_path):
    root = _write_cache(tmp_path / "cache", total=11, rows=1)
    partitions = load_cache_partitions(root)
    assert partitions.labels[2] == "part2of11"
    assert partitions.labels[10] == "part10of11"
    np.testing.assert_array_equal(
        load_partitioned_array(partitions, "truth")[:, 0], np.arange(11)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__("checkpoint", "/models/wrong"),
            "partition invariants differ.*checkpoint",
        ),
        (
            lambda payload: payload["tf"].__setitem__("scatter_dex", 0.2),
            "partition invariants differ.*tf",
        ),
        (
            lambda payload: payload["posterior_populations"].__setitem__(
                "proposal", "different population"
            ),
            "partition invariants differ.*posterior_populations",
        ),
        (
            lambda payload: payload["observation_provenance"].__setitem__(
                "matched_group_size", 3
            ),
            "non-seed observation provenance differs",
        ),
        (
            lambda payload: payload["partition"].__setitem__(
                "galaxy_start", 3
            ),
            "row range.*not a positive continuation",
        ),
        (
            lambda payload: payload["files"].__setitem__(
                "truth", "truth/wrong.npy"
            ),
            "files.truth must be",
        ),
    ],
)
def test_partition_contract_rejects_mixed_or_misaligned_manifests(
    tmp_path, mutation, message
):
    root = _write_cache(tmp_path / "cache")
    _mutate_manifest(root, 1, 2, mutation)
    with pytest.raises(ValueError, match=message):
        load_cache_partitions(root)


def test_contract_rejects_missing_manifest_and_extra_array_part(tmp_path):
    root = _write_cache(tmp_path / "cache")
    (root / "meta" / "part1of2.json").unlink()
    with pytest.raises(ValueError, match="Incomplete cache manifests"):
        load_cache_partitions(root)

    root = _write_cache(tmp_path / "other")
    np.save(root / "truth" / "part2of2.npy", np.zeros((1, 9)))
    with pytest.raises(ValueError, match="do not exactly match manifests"):
        load_cache_partitions(root)


def test_contract_rejects_contiguous_prefix_that_omits_dataset_tail(tmp_path):
    root = _write_cache(tmp_path / "cache")
    for index in range(2):
        _mutate_manifest(
            root,
            index,
            2,
            lambda payload: payload.__setitem__("dataset_size", 5),
        )
    with pytest.raises(ValueError, match="do not cover the complete dataset"):
        load_cache_partitions(root)
