import json

import numpy as np
import pytest

from cache_contract import (
    CACHE_SCHEMA,
    CURRENT_FEATURE_NAMES,
    CURRENT_TARGET_TRANSFORMS,
    EXPECTED_DENSITY_COORDINATES,
    EXPECTED_OBSERVATION_MODEL,
    EXPECTED_TEST_SET_DENSITY_COORDINATES,
    LEGACY_CACHE_SCHEMA,
    LEGACY_REQUIRED_CACHE_ARRAYS,
    PARTITION_SEED_STRIDE,
    REQUIRED_CACHE_ARRAYS,
    STANDARD_ANALYSIS_MODE,
    TEST_SET_ANALYSIS_MODE,
    TEST_SET_CACHE_SCHEMA,
    TEST_SET_REQUIRED_CACHE_ARRAYS,
    load_cache_partitions,
    load_partitioned_array,
)


def _array(name, rows, draws, start):
    features = len(CURRENT_FEATURE_NAMES)
    if name == "sample":
        return np.zeros((rows, draws, features), dtype=np.float32)
    if name == "shear_sample":
        return np.zeros((rows, draws, 2), dtype=np.float32)
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
    if name in {
        "proposal_mean_estimates",
        "tf_target_mean_estimates",
        "target_mean_estimates",
    }:
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
        required_arrays = (
            TEST_SET_REQUIRED_CACHE_ARRAYS
            if schema == TEST_SET_CACHE_SCHEMA
            else REQUIRED_CACHE_ARRAYS
        )
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
    if schema in {CACHE_SCHEMA, TEST_SET_CACHE_SCHEMA}:
        payload.update(
            {
                "physical_parameter_ranges": {
                    "g1": [-0.1, 0.1],
                    "g2": [-0.1, 0.1],
                    "theta_int": [0.0, np.pi],
                    "cosi": [0.0, 1.0],
                    "v0": [0.0, 20.0],
                    "vcirc": [60.0, 540.0],
                    "rscale": [0.1, 5.0],
                    "hlr": [0.1, 5.0],
                    "halpha_flux_true": [1.0e-17, 1.0e-14],
                },
                "target_transforms": dict(CURRENT_TARGET_TRANSFORMS),
                "density_coordinates": dict(
                    EXPECTED_TEST_SET_DENSITY_COORDINATES
                    if schema == TEST_SET_CACHE_SCHEMA
                    else EXPECTED_DENSITY_COORDINATES
                ),
                "observation_model": dict(EXPECTED_OBSERVATION_MODEL),
            }
        )
    if schema == TEST_SET_CACHE_SCHEMA:
        payload["tf"]["population_log_ratio_normalization"] = (
            "not_applicable_already_tf_conformed"
        )
        payload.update(
            {
                "analysis_mode": TEST_SET_ANALYSIS_MODE,
                "posterior_populations": {
                    "test_set": "TF-conformed test population / TF posterior"
                },
                "test_set": {
                    "population": "tf_conformed_catalog",
                    "posterior_candidate_weighting": "tf_importance",
                    "population_weighting": "uniform",
                    "point_estimator": "mean",
                    "map_computed": False,
                    "tf_importance_weighting": True,
                    "shape_noise_regularization": "report_time",
                    "snr_source": "dataset_record",
                    "snr_policy": (
                        "used_as_stored_without_redraw_or_clipping"
                    ),
                    "stored_candidate_parameters": ["g1", "g2"],
                    "tf": {
                        "slope": -7.22,
                        "intercept": 36.0,
                        "scatter_dex": 0.1,
                        "vcirc_min": 60.0,
                        "vcirc_max": 540.0,
                    },
                    "generation_manifest": {
                        "path": "/datasets/current/manifest.json",
                        "sha256": "a" * 64,
                        "schema": "klnn-generation-manifest-v1",
                        "analysis_mode": TEST_SET_ANALYSIS_MODE,
                        "population": "tf_conformed_catalog",
                        "sample_count": total * (end - start),
                        "redshift": 0.3,
                        "simulation_redshift": 0.3,
                        "source_catalog": {"path": "/catalog.fits"},
                        "catalog_sampling": {
                            "eligible_row_count": 100,
                            "eligibility": {
                                "hlr": {
                                    "finite": True,
                                    "minimum": 0.1,
                                    "maximum": 5.0,
                                    "bounds": "inclusive",
                                },
                                "image_snr": {
                                    "finite": True,
                                    "minimum": 10.0,
                                    "maximum": 1000.0,
                                },
                                "halpha_snr": {
                                    "finite": True,
                                    "minimum": 1.0,
                                    "maximum": 150.0,
                                }
                            },
                        },
                        "parameter_sampling": {
                            "inclination": {
                                "distribution": "cosi_uniform_0_1_latin_hypercube",
                                "transform": "sini=sqrt(1-cosi**2)",
                            }
                        },
                        "tf": {
                            "slope": -7.22,
                            "intercept": 36.0,
                            "scatter_dex": 0.1,
                            "vcirc_min": 60.0,
                            "vcirc_max": 540.0,
                        },
                        "sample_table": {
                            "path": "/samples.csv",
                            "sha256": "b" * 64,
                            "row_count": total * (end - start),
                            "id_policy": "zero_based_contiguous_row_index",
                        },
                    },
                },
            }
        )
    return payload


def _write_cache(root, *, total=2, rows=2, draws=4, schema=CACHE_SCHEMA):
    (root / "meta").mkdir(parents=True)
    required_arrays = (
        LEGACY_REQUIRED_CACHE_ARRAYS
        if schema == LEGACY_CACHE_SCHEMA
        else (
            TEST_SET_REQUIRED_CACHE_ARRAYS
            if schema == TEST_SET_CACHE_SCHEMA
            else REQUIRED_CACHE_ARRAYS
        )
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
    assert partitions.analysis_mode == STANDARD_ANALYSIS_MODE
    assert partitions.mode_metadata == {}
    assert partitions.total_rows == 4
    assert set(partitions.files) == set(REQUIRED_CACHE_ARRAYS)
    assert "spectral_reference_quality" not in partitions.files
    assert partitions.observation_provenance == {"matched_group_size": 1}
    truth = load_partitioned_array(partitions, "truth")
    np.testing.assert_array_equal(truth[:, 0], np.arange(4))


def test_compact_test_set_contract_exposes_mode_and_embedded_provenance(
    tmp_path,
):
    root = _write_cache(
        tmp_path / "test-cache", schema=TEST_SET_CACHE_SCHEMA
    )
    partitions = load_cache_partitions(root)
    assert partitions.analysis_mode == TEST_SET_ANALYSIS_MODE
    assert set(partitions.files) == set(TEST_SET_REQUIRED_CACHE_ARRAYS)
    assert "sample" not in partitions.files
    assert "base_log_prob" not in partitions.files
    assert "tf_target_mean_estimates" in partitions.files
    assert "posterior_tf_log_weight" in partitions.files
    assert "population_tf_log_ratio" not in partitions.files
    assert partitions.mode_metadata["population"] == "tf_conformed_catalog"
    generation = partitions.mode_metadata["generation_manifest"]
    assert generation["schema"] == "klnn-generation-manifest-v1"
    assert generation["sample_count"] == partitions.dataset_size
    assert generation["source_catalog"]["path"] == "/catalog.fits"
    shear = load_partitioned_array(partitions, "shear_sample")
    assert shear.shape == (4, 4, 2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__(
                "analysis_mode", STANDARD_ANALYSIS_MODE
            ),
            "analysis_mode must be.*test_set",
        ),
        (
            lambda payload: payload.__setitem__("tf", {}),
            "tf is missing fields",
        ),
        (
            lambda payload: payload["test_set"].__setitem__(
                "tf_importance_weighting", False
            ),
            "tf_importance_weighting",
        ),
        (
            lambda payload: payload["files"].__setitem__(
                "population_tf_log_ratio",
                "population_tf_log_ratio/part0of2.npy",
            ),
            "extra=.*population_tf_log_ratio",
        ),
        (
            lambda payload: payload["test_set"]["generation_manifest"][
                "catalog_sampling"
            ].__setitem__("eligible_hlr_capped_count", 3),
            "HLR cap counters",
        ),
        (
            lambda payload: payload["test_set"]["generation_manifest"][
                "catalog_sampling"
            ]["eligibility"]["hlr"].__setitem__("maximum", 50.0),
            "cap-after-selection caches are invalid",
        ),
        (
            lambda payload: payload["test_set"]["generation_manifest"]
            ["catalog_sampling"]["eligibility"]["image_snr"].__setitem__(
                "minimum", 5.0
            ),
            "eligibility.image_snr",
        ),
        (
            lambda payload: payload["test_set"][
                "generation_manifest"
            ].__setitem__("sample_count", 3),
            "sample_count",
        ),
    ],
)
def test_compact_test_set_contract_fails_closed(tmp_path, mutation, message):
    root = _write_cache(
        tmp_path / "test-cache", schema=TEST_SET_CACHE_SCHEMA
    )
    _mutate_manifest(root, 0, 2, mutation)
    with pytest.raises(ValueError, match=message):
        load_cache_partitions(root)


def test_compact_test_set_contract_requires_direct_cosi_manifest(tmp_path):
    root = _write_cache(
        tmp_path / "test-cache", schema=TEST_SET_CACHE_SCHEMA
    )

    def remove_transform(payload):
        payload["test_set"]["generation_manifest"]["parameter_sampling"][
            "inclination"
        ].pop("transform")

    _mutate_manifest(root, 0, 2, remove_transform)
    with pytest.raises(ValueError, match="parameter_sampling.inclination"):
        load_cache_partitions(root)


def test_compact_test_set_contract_rejects_sini_target_range(tmp_path):
    root = _write_cache(
        tmp_path / "test-cache", schema=TEST_SET_CACHE_SCHEMA
    )

    def replace_cosi(payload):
        ranges = payload["physical_parameter_ranges"]
        ranges["sini"] = ranges.pop("cosi")
        payload["feature_names"][3] = "sini"

    _mutate_manifest(root, 0, 2, replace_cosi)
    with pytest.raises(ValueError, match="feature_names"):
        load_cache_partitions(root)


@pytest.mark.parametrize(
    "schema",
    [CACHE_SCHEMA, TEST_SET_CACHE_SCHEMA],
)
def test_current_contract_rejects_reordered_physical_parameter_ranges(
    tmp_path, schema
):
    root = _write_cache(tmp_path / "cache", schema=schema)

    def reverse_parameter_ranges(payload):
        ranges = payload["physical_parameter_ranges"]
        payload["physical_parameter_ranges"] = {
            name: ranges[name] for name in reversed(tuple(ranges))
        }

    _mutate_manifest(root, 0, 2, reverse_parameter_ranges)
    with pytest.raises(ValueError, match="keys must equal feature_names in order"):
        load_cache_partitions(root)


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
