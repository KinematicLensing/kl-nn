import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "arch" / "cache_posteriors.py"
    spec = importlib.util.spec_from_file_location("cache_posteriors_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_summary_and_target_map_use_same_joint_candidate_rows():
    module = _module()
    names = ("g1", "g2", "theta_int")
    samples = np.asarray(
        [[[0.0, 10.0, 3.10], [1.0, 20.0, -3.12], [2.0, 30.0, 3.13]]]
    )
    base = np.asarray([[3.0, 2.0, 1.0]])
    ratio = np.asarray([[0.0, 2.0, 5.0]])
    weight = np.asarray([[0.05, 0.15, 0.80]])
    summary = module.posterior_summaries(samples, base, ratio, weight, names)
    np.testing.assert_array_equal(summary["proposal_map_estimates"][0], samples[0, 0])
    np.testing.assert_array_equal(summary["tf_target_map_estimates"][0], samples[0, 2])
    np.testing.assert_allclose(summary["tf_target_mean_estimates"][0, 1, :2], [1.75, 27.5])
    assert abs(abs(summary["tf_target_mean_estimates"][0, 1, 2]) - np.pi) < 0.05


def test_cache_cli_rejects_unknown_options_and_has_one_sampling_surface():
    module = _module()
    parser_error = None
    try:
        module.parse_args(
            [
                "-i", "0", "--nparts", "1", "--ngals", "2",
                "--model-name", "m", "--dataset", "d",
                "--unknown-option", "value",
            ]
        )
    except SystemExit as exc:
        parser_error = exc
    assert parser_error is not None
    args = module.parse_args(
        [
            "-i", "0", "--nparts", "1", "--ngals", "2",
            "--model-name", "m", "--dataset", "d", "--nsamples", "20",
        ]
    )
    assert args.nsamples == 20
    test_args = module.parse_args(
        [
            "-i", "0", "--nparts", "1", "--ngals", "2",
            "--model-name", "m", "--dataset", "d", "--test-set",
            "--dataset-manifest", "d/manifest.json",
        ]
    )
    assert test_args.test_set is True
    assert test_args.dataset_manifest == Path("d/manifest.json")
    assert test_args.isotropic_inclination_prior is False
    combined_args = module.parse_args(
        [
            "-i", "0", "--nparts", "1", "--ngals", "2",
            "--model-name", "m", "--dataset", "d", "--test-set",
            "--isotropic-inclination-prior",
        ]
    )
    assert combined_args.test_set is True
    assert combined_args.isotropic_inclination_prior is True


def test_compact_test_set_arrays_store_only_needed_tf_candidate_products():
    module = _module()
    assert set(module.TEST_SET_CACHE_ARRAY_TYPES) == {
        "shear_sample",
        "posterior_tf_log_weight",
        "posterior_tf_ess",
        "posterior_tf_ess_fraction",
        "posterior_tf_max_weight",
        "truth",
        "rmag_true",
        "image_snr",
        "central_halpha_snr",
        "image_noise_sigma",
        "central_spectral_noise_sigma",
        "proposal_mean_estimates",
        "tf_target_mean_estimates",
    }
    assert not (
        set(module.TEST_SET_CACHE_ARRAY_TYPES)
        & {
            "sample",
            "base_log_prob",
            "proposal_map_estimates",
            "posterior_tf_log_ratio",
            "posterior_tf_weight",
            "posterior_tf_log_mean_ratio",
            "population_tf_log_ratio",
        }
    )


def test_combined_test_set_arrays_use_neutral_target_names():
    module = _module()
    assert set(module.COMBINED_TEST_SET_CACHE_ARRAY_TYPES) == {
        "shear_sample",
        "posterior_target_log_weight",
        "posterior_target_ess",
        "posterior_target_ess_fraction",
        "posterior_target_max_weight",
        "truth",
        "rmag_true",
        "image_snr",
        "central_halpha_snr",
        "image_noise_sigma",
        "central_spectral_noise_sigma",
        "proposal_mean_estimates",
        "target_mean_estimates",
    }
    assert not any(
        "tf" in name for name in module.COMBINED_TEST_SET_CACHE_ARRAY_TYPES
    )


def test_proposal_mean_summaries_use_equal_candidate_mass_and_circular_theta():
    module = _module()
    names = ("g1", "g2", "theta_int")
    samples = np.asarray(
        [
            [
                [-0.03, 0.01, 3.10],
                [-0.01, 0.02, -3.12],
                [0.01, 0.03, 3.13],
                [0.03, 0.04, -3.11],
            ]
        ]
    )
    summary = module.proposal_mean_summaries(samples, names)
    assert summary.shape == (1, 3, 3)
    np.testing.assert_allclose(summary[0, 1, :2], [0.0, 0.025])
    assert abs(abs(summary[0, 1, 2]) - np.pi) < 0.05


def test_tf_target_mean_summaries_weight_same_joint_candidate_rows():
    module = _module()
    names = ("g1", "g2", "theta_int")
    samples = np.asarray(
        [
            [
                [-0.03, 1.0, 3.10],
                [-0.01, 2.0, -3.12],
                [0.01, 3.0, 3.13],
                [0.03, 4.0, -3.11],
            ]
        ]
    )
    weight = np.asarray([[0.05, 0.15, 0.30, 0.50]])
    summary = module.tf_target_mean_summaries(samples, weight, names)
    assert summary.shape == (1, 3, 3)
    np.testing.assert_allclose(summary[0, 1, :2], [0.015, 3.25])
    assert abs(abs(summary[0, 1, 2]) - np.pi) < 0.05
    with np.testing.assert_raises_regex(ValueError, "galaxy, draw"):
        module.tf_target_mean_summaries(
            samples, np.ones((1, 3)), names
        )


def _generation_manifest(sample_count=4):
    return {
        "schema": "klnn-generation-manifest-v1",
        "analysis_mode": "test_set",
        "population": "tf_conformed_catalog",
        "sample_count": sample_count,
        "redshift": 0.3,
        "simulation_redshift": 0.3,
        "source_catalog": {"path": "/catalog.fits"},
        "catalog_sampling": {
            "seed": 42,
            "eligibility": {
                "hlr": {
                    "finite": True,
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "bounds": "inclusive",
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
            "sha256": "a" * 64,
            "row_count": sample_count,
            "id_policy": "contiguous_zero_based",
        },
    }


def test_test_set_generation_manifest_is_required_and_embedded(tmp_path):
    module = _module()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    default = module.resolve_dataset_manifest(dataset, tmp_path, None)
    assert default == dataset / "manifest.json"
    default.write_text(json.dumps(_generation_manifest()), encoding="utf-8")
    prior = module.TFPrior()
    embedded = module.load_test_set_generation_manifest(
        default,
        dataset_size=4,
        tf_prior=prior,
        hlr_bounds=(0.1, 5.0),
    )
    assert embedded["path"] == str(default.resolve())
    assert len(embedded["sha256"]) == 64
    assert embedded["source_catalog"]["path"] == "/catalog.fits"

    combined = module.load_test_set_generation_manifest(
        default,
        dataset_size=4,
        tf_prior=prior,
        hlr_bounds=(0.1, 5.0),
        require_isotropic_inclination=True,
    )
    assert (
        combined["parameter_sampling"]["inclination"]["distribution"]
        == "cosi_uniform_0_1_latin_hypercube"
    )

    invalid = _generation_manifest(sample_count=3)
    default.write_text(json.dumps(invalid), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "sample_count"):
        module.load_test_set_generation_manifest(
            default,
            dataset_size=4,
            tf_prior=prior,
            hlr_bounds=(0.1, 5.0),
        )


def test_isotropic_cache_rejects_non_cosi_generation_manifest(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    payload = _generation_manifest()
    payload["parameter_sampling"]["inclination"] = {
        "distribution": "sini_uniform_0_1_latin_hypercube",
        "transform": "identity",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "isotropic inclination"):
        module.load_test_set_generation_manifest(
            path,
            dataset_size=4,
            tf_prior=module.TFPrior(),
            hlr_bounds=(0.1, 5.0),
            require_isotropic_inclination=True,
        )


def test_generation_manifest_tf_must_match_cache_assumption(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    payload = _generation_manifest()
    payload["tf"]["scatter_dex"] = 0.2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "tf.scatter_dex"):
        module.load_test_set_generation_manifest(
            path,
            dataset_size=4,
            tf_prior=module.TFPrior(),
            hlr_bounds=(0.1, 5.0),
        )


def test_generation_manifest_sample_table_count_must_match_dataset(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    payload = _generation_manifest()
    payload["sample_table"]["row_count"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "sample_table.row_count"):
        module.load_test_set_generation_manifest(
            path,
            dataset_size=4,
            tf_prior=module.TFPrior(),
            hlr_bounds=(0.1, 5.0),
        )


def test_generation_manifest_rejects_clamped_hlr_policy(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    payload = _generation_manifest()
    payload["catalog_sampling"]["eligibility"]["hlr"] = {
        "finite": True,
        "minimum": 0.1,
        "maximum_policy": "cap_after_selection",
        "cap": 5.0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(
        ValueError, "catalog_sampling.eligibility.hlr"
    ):
        module.load_test_set_generation_manifest(
            path,
            dataset_size=4,
            tf_prior=module.TFPrior(),
            hlr_bounds=(0.1, 5.0),
        )


def test_generation_manifest_hlr_range_must_match_model_support(tmp_path):
    module = _module()
    path = tmp_path / "manifest.json"
    payload = _generation_manifest()
    payload["catalog_sampling"]["eligibility"]["hlr"]["maximum"] = 4.9
    path.write_text(json.dumps(payload), encoding="utf-8")
    with np.testing.assert_raises_regex(ValueError, "hlr.maximum"):
        module.load_test_set_generation_manifest(
            path,
            dataset_size=4,
            tf_prior=module.TFPrior(),
            hlr_bounds=(0.1, 5.0),
        )


def test_default_checkpoint_requires_saved_best(tmp_path):
    module = _module()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    numbered = model_dir / "model12"
    numbered.touch()
    default = SimpleNamespace(
        checkpoint=None, epoch=None, model_name="model"
    )
    with np.testing.assert_raises_regex(FileNotFoundError, "modelbest"):
        module.resolve_checkpoint(default, tmp_path)

    best = model_dir / "modelbest"
    best.touch()
    assert module.resolve_checkpoint(default, tmp_path) == best

    explicit_epoch = SimpleNamespace(
        checkpoint=None, epoch=12, model_name="model"
    )
    assert module.resolve_checkpoint(explicit_epoch, tmp_path) == numbered


def test_partition_range_is_exact_and_nonoverlapping():
    module = _module()
    assert module.resolve_partition_range(0, 10, 30) == (0, 10)
    assert module.resolve_partition_range(2, 10, 30) == (20, 30)
    module.validate_partition_coverage(3, 10, 30)
    with np.testing.assert_raises_regex(ValueError, "complete dataset"):
        module.validate_partition_coverage(2, 10, 30)


def test_partition_seed_changes_candidate_streams():
    base = 42
    seeds = [base + 1_000_003 * index for index in range(3)]
    assert len(set(seeds)) == 3


def test_physical_map_scores_include_log_flux_jacobian():
    module = _module()
    names = ("halpha_flux_true",)
    normalized = np.asarray([[[-1.0], [1.0]]], dtype=np.float64)
    equal_normalized_density = np.zeros((1, 2), dtype=np.float64)
    scores = module.physical_log_prob_from_normalized(
        normalized,
        equal_normalized_density,
        par_ranges={"halpha_flux_true": [1.0e-17, 1.0e-14]},
        feature_names=names,
        target_transforms={"halpha_flux_true": "log10"},
    )
    assert scores[0, 0] > scores[0, 1]
    np.testing.assert_allclose(scores[0, 0] - scores[0, 1], np.log(1000.0))
