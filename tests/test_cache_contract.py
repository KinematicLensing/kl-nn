import json

import numpy as np
import pytest

from cache_contract import (
    CURRENT_FEATURE_NAMES,
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
    }:
        return np.zeros(rows, dtype=np.float64)
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


def _manifest(index, total, start, end, draws):
    label = f"part{index}of{total}"
    seed = 42 + PARTITION_SEED_STRIDE * index
    return {
        "schema": "klnn-posterior-cache-v1",
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
        "observation_provenance": {
            "image_noise_sigma": 1.0,
            "spectral_reference_line_norm": 2.0,
            "matched_group_size": 1,
            "posterior_sample_seed": seed,
            "image_noise_seed": seed,
            "spectral_noise_seed": seed + 101,
            "spectral_quality_seed": seed + 307,
        },
        "files": {
            name: f"{name}/{label}.npy" for name in REQUIRED_CACHE_ARRAYS
        },
    }


def _write_cache(root, *, total=2, rows=2, draws=4):
    (root / "meta").mkdir(parents=True)
    for index in range(total):
        start = index * rows
        end = start + rows
        label = f"part{index}of{total}"
        for name in REQUIRED_CACHE_ARRAYS:
            directory = root / name
            directory.mkdir(exist_ok=True)
            np.save(
                directory / f"{label}.npy",
                _array(name, rows, draws, start),
            )
        payload = _manifest(index, total, start, end, draws)
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
    truth = load_partitioned_array(partitions, "truth")
    np.testing.assert_array_equal(truth[:, 0], np.arange(4))


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
                "image_noise_sigma", 3.0
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
