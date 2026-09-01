import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import truncnorm


def _report():
    path = (
        Path(__file__).resolve().parents[1]
        / "arch" / "diagnostics" / "shear_bias_report.py"
    )
    spec = importlib.util.spec_from_file_location("shear_bias_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FEATURES = (
    "g1", "g2", "theta_int", "cosi", "v0", "vcirc", "rscale", "hlr",
    "halpha_flux_true",
)


def _write_part(root, name, value, label="part0of1"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{label}.npy", value)


def _cache_arrays(n, draws=4):
    f = len(FEATURES)
    samples = np.zeros((n, draws, f), dtype=np.float32)
    truth = np.zeros((n, f), dtype=np.float32)
    maps = np.zeros((n, f), dtype=np.float32)
    summary = np.zeros((n, 3, f), dtype=np.float32)
    candidate = np.zeros((n, draws), dtype=np.float32)
    scalar = np.zeros(n, dtype=np.float64)
    return {
        "sample": samples,
        "base_log_prob": candidate,
        "posterior_tf_log_ratio": candidate,
        "posterior_tf_log_weight": np.full((n, draws), -np.log(draws)),
        "posterior_tf_weight": np.full((n, draws), 1.0 / draws),
        "posterior_tf_ess": np.full(n, draws, dtype=np.float64),
        "posterior_tf_ess_fraction": np.ones(n, dtype=np.float64),
        "posterior_tf_max_weight": np.full(n, 1.0 / draws),
        "posterior_tf_log_mean_ratio": scalar,
        "population_tf_log_ratio": scalar,
        "truth": truth,
        "rmag_true": np.full(n, 20.0),
        "spectral_reference_quality": np.full(n, 10.0),
        "proposal_map_estimates": maps,
        "proposal_mean_estimates": summary,
        "tf_target_map_estimates": maps,
        "tf_target_mean_estimates": summary,
    }


def _write_complete_cache(root, *, n, draws=4, overrides=None):
    arrays = _cache_arrays(n, draws)
    arrays.update(overrides or {})
    for name, value in arrays.items():
        _write_part(root, name, value)
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    label = "part0of1"
    manifest = {
        "schema": "klnn-posterior-cache-v1",
        "model_name": "model",
        "checkpoint": "/models/model/modelbest",
        "dataset": "/datasets/dataset",
        "dataset_size": n,
        "partition": {
            "index": 0,
            "total": 1,
            "label": label,
            "galaxy_start": 0,
            "galaxy_end": n,
        },
        "feature_names": list(FEATURES),
        "sample_shape": [n, draws, len(FEATURES)],
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
            "posterior_log_ratio": "raw ratio",
            "posterior_log_weight": "within-galaxy log-softmax",
            "posterior_weight_normalization": "within_galaxy",
            "population_log_ratio_normalization": "global_after_partition_concat",
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
            "posterior_sample_seed": 42,
            "image_noise_seed": 42,
            "spectral_noise_seed": 143,
            "spectral_quality_seed": 349,
        },
        "files": {name: f"{name}/{label}.npy" for name in arrays},
    }
    (meta / f"{label}.json").write_text(json.dumps(manifest))
    return arrays


def _write_compact_test_cache(root, *, n=36, draws=6, combined_prior=False):
    rmag = np.linspace(18.0, 22.0, n, dtype=np.float32)
    truth = np.zeros((n, len(FEATURES)), dtype=np.float32)
    truth[:, 0] = np.linspace(-0.019, 0.019, n)
    truth[:, 1] = np.linspace(0.019, -0.019, n)
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.5, 2.5, n)
    cosi = np.linspace(0.01, 0.99, n)
    truth[:, FEATURES.index("cosi")] = cosi
    truth[:, FEATURES.index("v0")] = np.linspace(-30.0, 30.0, n)
    mean_log10_vcirc = (rmag.astype(np.float64) - 36.0) / -7.22
    lower = (np.log10(60.0) - mean_log10_vcirc) / 0.1
    upper = (np.log10(540.0) - mean_log10_vcirc) / 0.1
    tf_probability = (np.arange(n, dtype=np.float64) + 0.5) / n
    tf_residual = 0.1 * truncnorm.ppf(
        tf_probability, lower, upper
    )
    truth[:, FEATURES.index("vcirc")] = 10 ** (
        mean_log10_vcirc + tf_residual
    )
    truth[:, FEATURES.index("rscale")] = np.linspace(0.2, 4.8, n)
    truth[:, FEATURES.index("hlr")] = np.linspace(0.2, 5.0, n)
    truth[:, FEATURES.index("halpha_flux_true")] = np.geomspace(1e-17, 1e-14, n)

    shear_sample = np.repeat(truth[:, None, :2], draws, axis=1)
    offsets = np.linspace(-0.006, 0.006, draws, dtype=np.float32)
    shear_sample[:, :, 0] += offsets
    shear_sample[:, :, 1] += offsets[::-1]
    mean = truth.copy()
    mean[:, :2] += 1.0e-4 + 0.02 * truth[:, :2]
    width = np.full_like(truth, 0.01)
    summary = np.stack((mean - width, mean, mean + width), axis=1)
    candidate_weight = np.arange(1, draws + 1, dtype=np.float64)
    candidate_weight /= np.sum(candidate_weight)
    candidate_weight = np.broadcast_to(candidate_weight, (n, draws)).copy()
    candidate_log_weight = np.log(candidate_weight)
    candidate_ess = 1.0 / np.sum(candidate_weight**2, axis=1)
    tf_summary = summary.copy()
    tf_summary[:, 1, :2] = np.sum(
        candidate_weight[:, :, None] * shear_sample, axis=1
    )
    if combined_prior:
        tf_summary[:, :, FEATURES.index("vcirc")] += 7.0
        weight_arrays = {
            "posterior_target_log_weight": candidate_log_weight,
            "posterior_target_ess": candidate_ess,
            "posterior_target_ess_fraction": candidate_ess / draws,
            "posterior_target_max_weight": np.max(candidate_weight, axis=1),
            "target_mean_estimates": tf_summary,
        }
    else:
        weight_arrays = {
            "posterior_tf_log_weight": candidate_log_weight,
            "posterior_tf_ess": candidate_ess,
            "posterior_tf_ess_fraction": candidate_ess / draws,
            "posterior_tf_max_weight": np.max(candidate_weight, axis=1),
            "tf_target_mean_estimates": tf_summary,
        }
    arrays = {
        "shear_sample": shear_sample,
        "truth": truth,
        "rmag_true": rmag,
        "image_snr": np.linspace(10.0, 800.0, n, dtype=np.float32),
        "central_halpha_snr": np.linspace(2.0, 150.0, n, dtype=np.float32),
        "image_noise_sigma": np.linspace(0.01, 0.1, n, dtype=np.float32),
        "central_spectral_noise_sigma": np.linspace(0.02, 0.2, n, dtype=np.float32),
        "proposal_mean_estimates": summary,
        **weight_arrays,
    }
    label = "part0of1"
    for name, value in arrays.items():
        _write_part(root, name, value, label)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": (
            "klnn-posterior-cache-test-v3"
            if combined_prior
            else "klnn-posterior-cache-test-v2"
        ),
        "analysis_mode": "test_set",
        "model_name": "model",
        "checkpoint": "/models/model/modelbest",
        "dataset": "/datasets/catalog-test",
        "dataset_size": n,
        "partition": {
            "index": 0,
            "total": 1,
            "label": label,
            "galaxy_start": 0,
            "galaxy_end": n,
        },
        "feature_names": list(FEATURES),
        "physical_parameter_ranges": {
            "g1": [-0.1, 0.1],
            "g2": [-0.1, 0.1],
            "theta_int": [-3.141592653589793, 3.141592653589793],
            "cosi": [0.0, 1.0],
            "v0": [-100.0, 100.0],
            "vcirc": [60.0, 540.0],
            "rscale": [0.1, 5.0],
            "hlr": [0.1, 5.0],
            "halpha_flux_true": [1.0e-17, 1.0e-14],
        },
        "target_transforms": {
            name: "log10" if name == "halpha_flux_true" else "identity"
            for name in FEATURES
        },
        "density_coordinates": {
            "stored_shear_samples": "physical_target_coordinates",
            "posterior_summary": "physical_target_coordinates",
            "map_selection": "not_computed",
        },
        "observation_model": {
            "schema_version": 3,
            "context_fields": ["rmag_true", "image_snr", "central_halpha_snr"],
            "halpha_flux_semantics": (
                "central_fiber_integrated_after_seeing_before_instrument"
            ),
            "image_snr_distribution": "uniform",
            "central_halpha_snr_distribution": "uniform",
            "image_snr_min": 10.0,
            "image_snr_max": 1000.0,
            "central_halpha_snr_min": 1.0,
            "central_halpha_snr_max": 150.0,
            "center_exposure_s": 180.0,
            "offset_exposure_s": 600.0,
        },
        "sample_shape": [n, draws, len(FEATURES)],
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
            "posterior_log_weight": "within-galaxy normalized log weight",
            "posterior_weight_normalization": "within_galaxy",
            "population_log_ratio_normalization": (
                "not_applicable_already_tf_conformed"
            ),
            "resampling": False,
        },
        "test_set": {
            "population": "tf_conformed_catalog",
            "posterior_candidate_weighting": (
                "tf_x_isotropic_inclination_importance"
                if combined_prior
                else "tf_importance"
            ),
            "population_weighting": "uniform",
            "point_estimator": "mean",
            "map_computed": False,
            "tf_importance_weighting": True,
            "shape_noise_regularization": "report_time",
            "snr_source": "dataset_record",
            "snr_policy": "used_as_stored_without_redraw_or_clipping",
            "stored_candidate_parameters": ["g1", "g2"],
            "tf": {
                "slope": -7.22,
                "intercept": 36.0,
                "scatter_dex": 0.1,
                "vcirc_min": 60.0,
                "vcirc_max": 540.0,
            },
            "generation_manifest": {
                "schema": "klnn-generation-manifest-v1",
                "analysis_mode": "test_set",
                "population": "tf_conformed_catalog",
                "redshift": 0.3,
                "simulation_redshift": 0.3,
                "sample_count": n,
                "path": "/samples/xu_sample_1_test_100k.manifest.json",
                "sha256": "a" * 64,
                "source_catalog": {
                    "path": "/catalogs/xu_sample_1_fullfootprint.fits"
                },
                "catalog_sampling": {
                    "eligible_row_count": 36_536_538,
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
                    "path": "/samples/xu_sample_1_test_100k.csv",
                    "sha256": "b" * 64,
                    "row_count": n,
                    "id_policy": "zero_based_contiguous_row_index",
                },
            },
        },
        "posterior_populations": {
            "test_set": (
                "TF-conformed catalog truth / TF + isotropic-inclination posterior"
                if combined_prior
                else "TF-conformed catalog truth / TF posterior"
            )
        },
        "observation_provenance": {
            "matched_group_size": 1,
            "posterior_sample_seed": 42,
            "image_noise_seed": 42,
            "spectral_noise_seed": 143,
        },
        "files": {name: f"{name}/{label}.npy" for name in arrays},
    }
    if combined_prior:
        manifest["test_set"].update(
            inclination_importance_weighting=True,
            inclination_prior={
                "training": "uniform_sini",
                "target": "uniform_cosi_0_1",
                "parameter": "cosi",
                "composition": (
                    "added_to_tf_log_ratio_before_within_galaxy_log_softmax"
                ),
                "resampling": False,
                "bounds": [0.0, 1.0],
            },
        )
        manifest["test_set"]["generation_manifest"]["parameter_sampling"][
            "inclination"
        ]["transform"] = "sini=sqrt(1-cosi**2)"
    (root / "meta" / f"{label}.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return arrays


def _manual_case(
    report, root, truth, samples, log_weight,
    proposal_weight=None, target_weight=None,
):
    n = len(truth)
    _write_complete_cache(
        root,
        n=n,
        draws=samples.shape[1],
        overrides={
            "truth": truth,
            "sample": samples,
            "posterior_tf_log_weight": log_weight,
            "posterior_tf_weight": np.exp(log_weight),
        },
    )
    if proposal_weight is None:
        proposal_weight = np.full(n, 1.0 / n)
    if target_weight is None:
        target_weight = np.full(n, 1.0 / n)
    zeros = np.zeros((n, len(FEATURES)))
    summary = np.stack((zeros - 1, zeros, zeros + 1), axis=1)
    return {
        "case": "model:dataset",
        "root": root,
        "cache_partitions": report.load_cache_partitions(root),
        "truth": truth,
        "feature_names": FEATURES,
        "rmag_true": np.full(n, 20.0),
        "spectral_reference_quality": np.full(n, 10.0),
        "populations": {
            "Proposal population / base posterior": {
                "key": "proposal", "map": zeros, "mean": zeros,
                "summary": summary, "galaxy_weight": proposal_weight,
            },
            "TF target population / TF posterior": {
                "key": "tf_target", "map": zeros, "mean": zeros,
                "summary": summary, "galaxy_weight": target_weight,
            },
        },
    }


def test_component_metrics_recover_weighted_linear_bias():
    report = _report()
    truth = np.linspace(-0.015, 0.015, 40)
    estimate = truth + 2.0e-4 + 0.03 * truth
    weight = np.linspace(1.0, 4.0, len(truth))
    result = report.component_metrics(truth, estimate, 0.02, weight)
    np.testing.assert_allclose(result["low_c"], 2.0e-4, atol=1e-14)
    np.testing.assert_allclose(result["low_m"], 0.03, atol=1e-13)
    assert result["ess"] < len(truth)


def test_load_case_normalizes_population_ratio_after_all_partitions(tmp_path):
    report = _report()
    root = tmp_path / "m" / "d"
    n, f = 3, len(FEATURES)
    truth = np.zeros((n, f), dtype=np.float32)
    summary = np.zeros((n, 3, f), dtype=np.float32)
    _write_complete_cache(
        root,
        n=n,
        overrides={
            "truth": truth,
            "proposal_map_estimates": truth,
            "proposal_mean_estimates": summary,
            "tf_target_map_estimates": truth,
            "tf_target_mean_estimates": summary,
            "population_tf_log_ratio": np.log([1.0, 2.0, 7.0]),
            "posterior_tf_ess": np.full(n, 50.0),
            "posterior_tf_ess_fraction": np.full(n, 0.5),
            "posterior_tf_max_weight": np.full(n, 0.1),
        },
    )
    case = report.load_case(tmp_path, "m:d")
    target = case["populations"]["TF target population / TF posterior"]
    np.testing.assert_allclose(target["galaxy_weight"], [0.1, 0.2, 0.7])
    proposal = case["populations"]["Proposal population / base posterior"]
    np.testing.assert_allclose(proposal["galaxy_weight"], np.full(3, 1 / 3))


def test_load_case_uses_linear_central_halpha_snr_for_v2_cache(
    tmp_path, monkeypatch
):
    report = _report()
    arrays = _cache_arrays(3)
    arrays.pop("spectral_reference_quality")
    arrays.update(
        {
            "image_snr": np.asarray([10.0, 100.0, 1000.0]),
            "central_halpha_snr": np.asarray([1.0, 20.0, 150.0]),
            "image_noise_sigma": np.ones(3),
            "central_spectral_noise_sigma": np.ones(3),
        }
    )

    class _Partitions:
        feature_names = FEATURES
        files = {name: () for name in arrays}
        manifests = ({},)
        analysis_mode = "proposal_and_tf"
        dataset_size = 3

    monkeypatch.setattr(report, "load_cache_partitions", lambda _: _Partitions())
    monkeypatch.setattr(
        report,
        "load_partitioned_array",
        lambda _, name: arrays[name],
    )
    case = report.load_case(tmp_path, "model:dataset")
    assert case["spectral_condition_name"] == "central H-alpha S/N"
    assert case["spectral_condition_log_scale"] is False
    np.testing.assert_array_equal(
        case["spectral_condition"], arrays["central_halpha_snr"]
    )


def test_flat_candidate_pp_ranks_use_tf_candidate_weights(tmp_path):
    report = _report()
    truth = np.zeros((1, len(FEATURES)), dtype=np.float32)
    samples = np.zeros((1, 4, len(FEATURES)), dtype=np.float32)
    samples[0, :, 0] = [-0.02, -0.01, 0.01, 0.02]
    samples[0, :, 1] = samples[0, :, 0]
    log_weight = np.log(np.asarray([[0.1, 0.1, 0.7, 0.1]]))
    case = _manual_case(
        report, tmp_path, truth, samples, log_weight
    )
    diagnostics = report.load_shear_posterior_diagnostics(case)
    proposal = diagnostics["proposal"]
    target = diagnostics["tf_target"]
    np.testing.assert_allclose(proposal["g1"], [0.5])
    np.testing.assert_allclose(target["g1"], [0.2])
    np.testing.assert_allclose(target["g1_retained_mass"], [1.0])
    np.testing.assert_allclose(proposal["shape_noise"], [np.sqrt(2.5e-4)])
    np.testing.assert_allclose(target["shape_noise"], [np.sqrt(1.24e-4)])
    np.testing.assert_allclose(proposal["g1_variance"], [2.5e-4])
    np.testing.assert_allclose(target["g2_variance"], [1.24e-4])


def test_theta_modality_uses_proposal_or_tf_posterior_mass(tmp_path):
    report = _report()
    truth = np.zeros((1, len(FEATURES)), dtype=np.float32)
    samples = np.zeros((1, 8, len(FEATURES)), dtype=np.float32)
    samples[0, :, 2] = [-0.08, -0.04, 0.0, 0.04, 3.04, 3.08, 3.12, -3.12]
    log_weight = np.log(np.asarray([[0.2475] * 4 + [0.0025] * 4]))
    case = _manual_case(report, tmp_path, truth, samples, log_weight)
    proposal = report.load_theta_posterior_diagnostics(case, "proposal")
    target = report.load_theta_posterior_diagnostics(case, "tf_target")
    np.testing.assert_allclose(proposal["opposite_branch_mass"], [0.5])
    np.testing.assert_allclose(target["opposite_branch_mass"], [0.01])
    assert proposal["mode_count"][0] >= 2
    assert target["mode_count"][0] == 1


def test_coverage_uses_global_galaxy_weights():
    report = _report()
    truth = np.zeros((2, 1))
    summary = np.asarray([[[-1.0], [0.0], [1.0]], [[1.0], [2.0], [3.0]]])
    row = report.coverage_metrics(
        truth, summary, np.asarray([0.2, 0.8]), ("g1",), "target"
    )[0]
    np.testing.assert_allclose(row["coverage"], 0.2)
    assert row["ess"] < 2.0


def test_nuisance_bias_metrics_include_mean_map_and_wrapped_theta():
    report = _report()
    n = 6
    truth = np.zeros((n, len(FEATURES)), dtype=np.float64)
    truth[:, 0] = np.linspace(-0.015, 0.015, n)
    truth[:, 1] = np.linspace(0.015, -0.015, n)
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.5, 2.5, n)
    mean = truth.copy()
    map_estimate = truth.copy()
    mean[:, FEATURES.index("theta_int")] += 2.0 * np.pi + 0.1
    map_estimate[:, FEATURES.index("theta_int")] -= 2.0 * np.pi + 0.2
    mean[:, FEATURES.index("cosi")] += 0.05
    map_estimate[:, FEATURES.index("vcirc")] -= 2.0
    case = {
        "truth": truth,
        "feature_names": FEATURES,
        "populations": {
            "population": {
                "key": "proposal",
                "mean": mean,
                "map": map_estimate,
                "galaxy_weight": np.full(n, 1.0 / n),
            }
        },
    }

    rows = report.nuisance_bias_metrics(case)
    by_key = {
        (row["estimator"], row["parameter"]): row for row in rows
    }
    assert len(rows) == 2 * (len(FEATURES) - 2)
    np.testing.assert_allclose(
        by_key[("Mean", "theta_int")]["bias"], 0.1, atol=1e-14
    )
    np.testing.assert_allclose(
        by_key[("MAP", "theta_int")]["bias"], -0.2, atol=1e-14
    )
    np.testing.assert_allclose(by_key[("Mean", "cosi")]["bias"], 0.05)
    np.testing.assert_allclose(by_key[("MAP", "vcirc")]["bias"], -2.0)
    rendered = report.nuisance_bias_table(rows)
    assert "weighted additive bias" in rendered
    assert "halpha_flux_true" in rendered
    assert {
        row["estimator"] for row in report.compute_metrics(case, 0.02)
    } == {"Mean", "MAP"}
    case["report_map"] = False
    assert {row["estimator"] for row in report.nuisance_bias_metrics(case)} == {
        "Mean"
    }
    assert {row["estimator"] for row in report.compute_metrics(case, 0.02)} == {"Mean"}


def test_conditional_calibration_has_cosi_shape_noise_and_map_panels():
    report = _report()
    n = 30
    truth = np.zeros((n, len(FEATURES)), dtype=np.float64)
    truth[:, 0] = np.linspace(-0.08, 0.08, n)
    truth[:, 1] = np.linspace(0.08, -0.08, n)
    truth[:, FEATURES.index("hlr")] = np.linspace(0.2, 2.8, n)
    truth[:, FEATURES.index("cosi")] = np.linspace(0.05, 0.95, n)
    mean = truth.copy()
    map_estimate = truth.copy()
    mean[:, :2] += 1.0e-4 + 0.02 * truth[:, :2]
    map_estimate[:, :2] += -2.0e-4 - 0.01 * truth[:, :2]
    population_label = "population"
    case = {
        "case": "model:dataset",
        "truth": truth,
        "feature_names": FEATURES,
        "rmag_true": np.linspace(16.0, 23.0, n),
        "image_snr": np.linspace(10.0, 900.0, n),
        "spectral_reference_quality": np.geomspace(3.0, 100.0, n),
        "populations": {
            population_label: {
                "mean": mean,
                "map": map_estimate,
                "galaxy_weight": np.full(n, 1.0 / n),
            }
        },
    }
    shape_noise = np.linspace(0.01, 0.03, n)

    mean_curves = report.conditional_shear_calibration(
        case, population_label, "Mean", 3, shape_noise
    )
    map_curves = report.conditional_shear_calibration(
        case, population_label, "MAP", 3, shape_noise
    )

    assert tuple(mean_curves) == (
        "true magnitude",
        "image S/N",
        "spectral reference quality",
        "true hlr",
        "true cosi",
    )
    for condition in mean_curves:
        assert set(mean_curves[condition]) == {"g1", "g2", "shape_noise"}
        np.testing.assert_allclose(mean_curves[condition]["g1"]["m"], 0.02)
        np.testing.assert_allclose(mean_curves[condition]["g1"]["c"], 1.0e-4)
        np.testing.assert_allclose(map_curves[condition]["g2"]["m"], -0.01)
        np.testing.assert_allclose(map_curves[condition]["g2"]["c"], -2.0e-4)
        assert np.all(np.isfinite(mean_curves[condition]["shape_noise"]["value"]))
    figure = report.conditional_shear_calibration_figure(
        case, population_label, "MAP", map_curves
    )
    assert figure.startswith("data:image/png;base64,")


def test_weighted_flag_regularizes_precision_with_ensemble_shape_noise(tmp_path):
    report = _report()
    args = report.parse_args(
        ["--case", "model:dataset", "--output", str(tmp_path / "report.html"),
         "--weighted"]
    )
    assert args.weighted

    weight, diagnostics = report.compose_shape_noise_regularized_weights(
        np.full(3, 1.0 / 3.0),
        np.asarray([1.0, 4.0, np.nan]),
        np.asarray([1.0, 4.0, 1.0]),
    )
    expected = np.asarray([1.0 / (1.0 + 1.5**2), 1.0 / (4.0 + 1.5**2), 0.0])
    expected /= np.sum(expected)
    np.testing.assert_allclose(weight, expected)
    np.testing.assert_allclose(diagnostics["shape_noise"], 1.5)
    np.testing.assert_allclose(diagnostics["shape_noise_variance"], 1.5**2)
    np.testing.assert_allclose(
        diagnostics["weighted_shape_noise"], np.sum(expected[:2] * [1.0, 2.0])
    )
    assert diagnostics["weighted_shape_noise"] < diagnostics["shape_noise"]
    assert diagnostics["invalid_variance_count"] == 1
    np.testing.assert_allclose(
        diagnostics["invalid_variance_population_mass"], 1.0 / 3.0
    )
    np.testing.assert_allclose(diagnostics["population_ess"], 3.0)
    assert diagnostics["ess"] < 2.0


def test_shape_noise_weight_comparison_reports_m_shape_noise_and_ess():
    report = _report()
    n = 80
    truth = np.zeros((n, len(FEATURES)), dtype=np.float64)
    truth[:, 0] = np.linspace(-0.019, 0.019, n)
    truth[:, 1] = np.linspace(0.019, -0.019, n)
    proposal_mean = truth.copy()
    target_mean = truth.copy()
    proposal_mean[:, :2] += 0.03 * truth[:, :2]
    target_mean[:, :2] -= 0.04 * truth[:, :2]
    uniform = np.full(n, 1.0 / n)
    target_population = np.linspace(1.0, 3.0, n)
    target_population /= np.sum(target_population)
    case = {
        "truth": truth,
        "populations": {
            "Original": {
                "key": "proposal", "mean": proposal_mean,
                "galaxy_weight": uniform, "population_weight": uniform.copy(),
            },
            "TF": {
                "key": "tf_target", "mean": target_mean,
                "galaxy_weight": target_population,
                "population_weight": target_population.copy(),
            },
        },
    }
    variance = np.geomspace(1.0e-5, 1.0e-3, n)
    posterior = {
        "proposal": {"g1_variance": variance, "g2_variance": variance},
        "tf_target": {
            "g1_variance": variance[::-1], "g2_variance": variance[::-1]
        },
    }

    rows = report.shape_noise_weight_comparison(case, posterior, 0.02)
    assert len(rows) == 4
    for row in rows:
        expected = 0.03 if row["population_key"] == "proposal" else -0.04
        np.testing.assert_allclose(row["g1_m"], expected, atol=1e-12)
        np.testing.assert_allclose(row["g2_m"], expected, atol=1e-12)
        assert row["g1_ess"] > 0
        assert row["g2_ess"] > 0
    proposal_rows = [
        row for row in rows if row["population_key"] == "proposal"
    ]
    assert [row["weighting"] for row in proposal_rows] == [
        "Population only", "Shape-noise regularized"
    ]
    assert proposal_rows[-1]["reported_ess"] < proposal_rows[0]["reported_ess"]
    assert (
        proposal_rows[-1]["reported_shape_noise"]
        < proposal_rows[0]["reported_shape_noise"]
    )
    rendered = report.shape_noise_weight_comparison_table(rows)
    assert "fixed first-pass" in rendered
    assert "Shape-noise regularized" in rendered
    assert "Invalid variance" in rendered
    assert "g1 N / fit ESS" in rendered


def test_nuisance_curves_share_proposal_bins_and_use_only_means():
    report = _report()
    n = 50
    truth = np.zeros((n, len(FEATURES)), dtype=np.float64)
    for index in range(2, len(FEATURES)):
        truth[:, index] = np.linspace(1.0, 2.0, n) * (index + 1)
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.0, 2.0, n)
    truth[:, FEATURES.index("halpha_flux_true")] = np.geomspace(
        1e-17, 1e-14, n
    )
    proposal_mean = truth.copy()
    target_mean = truth.copy()
    proposal_mean[:, 2:] += 0.1 * truth[:, 2:]
    target_mean[:, 2:] -= 0.2 * truth[:, 2:]
    raw_proposal = np.full(n, 1.0 / n)
    active_proposal = np.geomspace(1.0, 100.0, n)
    active_proposal /= np.sum(active_proposal)
    target_weight = np.linspace(1.0, 2.0, n)
    target_weight /= np.sum(target_weight)
    case = {
        "case": "model:dataset",
        "truth": truth,
        "feature_names": FEATURES,
        "populations": {
            "Original": {
                "key": "proposal", "mean": proposal_mean,
                "map": np.full_like(truth, 999.0),
                "galaxy_weight": active_proposal,
                "population_weight": raw_proposal,
            },
            "TF": {
                "key": "tf_target", "mean": target_mean,
                "map": np.full_like(truth, -999.0),
                "galaxy_weight": target_weight,
                "population_weight": target_weight.copy(),
            },
        },
    }

    curves = report.nuisance_bias_curves(case, 5)
    expected_edges = report.weighted_quantile(
        truth[:, FEATURES.index("vcirc")], np.linspace(0.0, 1.0, 6), raw_proposal
    )
    np.testing.assert_allclose(curves["vcirc"]["edges"], expected_edges)
    assert set(curves["vcirc"]["populations"]) == {"Original", "TF"}
    np.testing.assert_allclose(
        curves["vcirc"]["populations"]["Original"]["m"], 0.1
    )
    np.testing.assert_allclose(
        curves["vcirc"]["populations"]["TF"]["m"], -0.2
    )
    np.testing.assert_allclose(
        curves["halpha_flux_true"]["populations"]["Original"]["m"], 0.1
    )
    np.testing.assert_allclose(
        curves["halpha_flux_true"]["populations"]["TF"]["m"], -0.2
    )
    figure = report.nuisance_bias_figure(case, 5)
    assert figure.startswith("data:image/png;base64,")


def test_importance_table_keeps_raw_tf_population_ess_after_precision_weighting():
    report = _report()
    raw = np.asarray([0.5, 0.5])
    case = {
        "posterior_tf_ess": np.asarray([10.0, 20.0]),
        "posterior_tf_ess_fraction": np.asarray([0.1, 0.2]),
        "posterior_tf_max_weight": np.asarray([0.2, 0.1]),
        "populations": {
            "TF target population / TF posterior": {
                "galaxy_weight": np.asarray([0.99, 0.01]),
                "population_weight": raw,
            }
        },
    }
    rendered = report.importance_table(case)
    assert "effective sample size <b>2.0</b>" in rendered


def test_weighted_report_smoke_includes_regularization_and_nuisance_plot(tmp_path):
    report = _report()
    n, draws = 36, 6
    truth = np.zeros((n, len(FEATURES)), dtype=np.float32)
    truth[:, 0] = np.linspace(-0.019, 0.019, n)
    truth[:, 1] = np.linspace(0.019, -0.019, n)
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.0, 2.0, n)
    truth[:, FEATURES.index("cosi")] = np.linspace(0.1, 0.95, n)
    truth[:, FEATURES.index("v0")] = np.linspace(-30.0, 30.0, n)
    truth[:, FEATURES.index("vcirc")] = np.linspace(80.0, 500.0, n)
    truth[:, FEATURES.index("rscale")] = np.linspace(0.2, 3.0, n)
    truth[:, FEATURES.index("hlr")] = np.linspace(0.2, 3.0, n)
    truth[:, FEATURES.index("halpha_flux_true")] = np.geomspace(1e-17, 1e-14, n)
    samples = np.repeat(truth[:, None, :], draws, axis=1)
    shear_offsets = np.linspace(-0.006, 0.006, draws, dtype=np.float32)
    samples[:, :, 0] += shear_offsets
    samples[:, :, 1] += shear_offsets[::-1]
    proposal_mean = truth.copy()
    target_mean = truth.copy()
    proposal_mean[:, :2] += 0.02 * truth[:, :2]
    target_mean[:, :2] -= 0.03 * truth[:, :2]
    width = np.full_like(truth, 0.01)
    proposal_summary = np.stack(
        (proposal_mean - width, proposal_mean, proposal_mean + width), axis=1
    )
    target_summary = np.stack(
        (target_mean - width, target_mean, target_mean + width), axis=1
    )
    root = tmp_path / "cache" / "model" / "dataset"
    _write_complete_cache(
        root,
        n=n,
        draws=draws,
        overrides={
            "truth": truth,
            "sample": samples,
            "proposal_map_estimates": proposal_mean,
            "proposal_mean_estimates": proposal_summary,
            "tf_target_map_estimates": target_mean,
            "tf_target_mean_estimates": target_summary,
            "population_tf_log_ratio": np.linspace(-1.0, 1.0, n),
            "rmag_true": np.linspace(16.0, 23.0, n),
            "spectral_reference_quality": np.linspace(3.0, 100.0, n),
        },
    )
    output = tmp_path / "weighted.html"

    report.main(
        [
            "--cache-root", str(tmp_path / "cache"),
            "--case", "model:dataset",
            "--output", str(output),
            "--bins", "2",
            "--weighted",
        ]
    )

    document = output.read_text()
    assert "Shape-noise-regularized precision comparison" in document
    assert "fixed first-pass" in document
    assert "Shape-noise regularized" in document
    assert "There are no caps or percentile cuts" in document
    assert "nuisance posterior-mean bias versus truth" in document
    assert "MAP diagnostics are omitted" in document
    assert "Conditional MAP calibration" not in document
    assert ">MAP<" not in document


def test_test_set_cli_can_select_weighted_or_unweighted_ensemble(tmp_path):
    report = _report()
    args = report.parse_args(
        [
            "--case", "model:catalog",
            "--output", str(tmp_path / "report.html"),
            "--test-set",
        ]
    )
    assert args.test_set
    assert not args.weighted
    weighted_args = report.parse_args(
        [
            "--case", "model:catalog",
            "--output", str(tmp_path / "weighted-report.html"),
            "--test-set",
            "--weighted",
        ]
    )
    assert weighted_args.test_set
    assert weighted_args.weighted


def test_compact_test_set_loads_one_tf_posterior_with_uniform_galaxy_mass(
    tmp_path,
):
    report = _report()
    root = tmp_path / "cache" / "model" / "catalog"
    arrays = _write_compact_test_cache(root)

    case = report.load_case(
        tmp_path / "cache", "model:catalog", test_set=True
    )

    assert case["analysis_mode"] == "test_set"
    assert case["candidate_array"] == "shear_sample"
    assert case["report_map"] is False
    assert case["tf_conformance_audit"]["uniformity_status"] == "PASS"
    assert case["tf_conformance_audit"]["row_count"] == len(arrays["truth"])
    assert tuple(case["populations"]) == (
        "TF-conformed test set / TF posterior",
    )
    population = next(iter(case["populations"].values()))
    assert population["key"] == "test_set"
    assert "map" not in population
    np.testing.assert_allclose(
        population["mean"], arrays["tf_target_mean_estimates"][:, 1]
    )
    assert not np.array_equal(
        population["mean"][:, :2],
        arrays["proposal_mean_estimates"][:, 1, :2],
    )
    np.testing.assert_allclose(
        population["population_weight"],
        np.full(len(arrays["truth"]), 1.0 / len(arrays["truth"])),
    )
    assert set(case["cache_partitions"].files) == {
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
    assert "population_tf_log_ratio" not in case["cache_partitions"].files
    diagnostics = report.load_shear_posterior_diagnostics(case)
    assert tuple(diagnostics) == ("test_set",)
    assert np.all(np.isfinite(diagnostics["test_set"]["shape_noise"]))
    row = len(arrays["truth"]) // 2
    np.testing.assert_allclose(
        diagnostics["test_set"]["g1"][row],
        np.sum(np.arange(1.0, 4.0)) / np.sum(np.arange(1.0, 7.0)),
    )
    np.testing.assert_allclose(
        diagnostics["test_set"]["g2"][row],
        np.sum(np.arange(4.0, 7.0)) / np.sum(np.arange(1.0, 7.0)),
    )

    with pytest.raises(ValueError, match="rerun with --test-set"):
        report.load_case(tmp_path / "cache", "model:catalog")



def test_tf_conformance_audit_recovers_truncated_conditional_pit():
    report = _report()
    tf = {
        "slope": -7.22,
        "intercept": 36.0,
        "scatter_dex": 0.1,
        "vcirc_min": 60.0,
        "vcirc_max": 540.0,
    }
    n = 200
    rmag = np.linspace(15.0, 23.4, n)
    probability = (np.arange(n, dtype=np.float64) + 0.5) / n
    mean_log10 = (rmag - tf["intercept"]) / tf["slope"]
    lower = (np.log10(tf["vcirc_min"]) - mean_log10) / tf["scatter_dex"]
    upper = (np.log10(tf["vcirc_max"]) - mean_log10) / tf["scatter_dex"]
    standardized = truncnorm.ppf(probability, lower, upper)
    vcirc = 10 ** (mean_log10 + tf["scatter_dex"] * standardized)

    audit = report.compute_test_set_tf_conformance_audit(
        vcirc, rmag, tf, generation_tf=dict(tf)
    )

    assert audit["uniformity_status"] == "PASS"
    assert audit["quantile_status"] == "PASS"
    assert audit["residual_status"] == "PASS"
    assert audit["row_count"] == n
    np.testing.assert_allclose(audit["ks_distance"], 0.5 / n, atol=2e-13)
    np.testing.assert_allclose(
        audit["pit_quantiles"],
        np.quantile(probability, report.TF_AUDIT_QUANTILES),
        atol=2e-13,
    )
    rendered = report.test_set_tf_conformance_table(
        {"case": "model:xu1", "tf_conformance_audit": audit}
    )
    assert "uniform KS distance D" in rendered
    assert "TF residual [dex]" in rendered
    assert "truncated-CDF PIT" in rendered


def test_tf_conformance_audit_flags_nonconforming_in_support_rows():
    report = _report()
    tf = {
        "slope": -7.22,
        "intercept": 36.0,
        "scatter_dex": 0.1,
        "vcirc_min": 60.0,
        "vcirc_max": 540.0,
    }
    rmag = np.linspace(18.0, 22.0, 100)
    mean_log10 = (rmag - tf["intercept"]) / tf["slope"]
    lower = (np.log10(tf["vcirc_min"]) - mean_log10) / tf["scatter_dex"]
    upper = (np.log10(tf["vcirc_max"]) - mean_log10) / tf["scatter_dex"]
    standardized = truncnorm.ppf(np.full(len(rmag), 0.5), lower, upper)
    vcirc = 10 ** (mean_log10 + tf["scatter_dex"] * standardized)

    audit = report.compute_test_set_tf_conformance_audit(vcirc, rmag, tf)

    assert audit["uniformity_status"] == "FAIL"
    assert audit["quantile_status"] == "FAIL"
    np.testing.assert_allclose(audit["ks_distance"], 0.5, atol=2e-13)


def test_tf_conformance_audit_fails_closed_on_config_and_cached_values():
    report = _report()
    tf = {
        "slope": -7.22,
        "intercept": 36.0,
        "scatter_dex": 0.1,
        "vcirc_min": 60.0,
        "vcirc_max": 540.0,
    }
    with pytest.raises(ValueError, match="contain exactly"):
        report.compute_test_set_tf_conformance_audit(
            np.asarray([200.0]), np.asarray([20.0]), {"slope": -7.22}
        )
    invalid_scatter = {**tf, "scatter_dex": 0.0}
    with pytest.raises(ValueError, match="scatter_dex must be positive"):
        report.compute_test_set_tf_conformance_audit(
            np.asarray([200.0]), np.asarray([20.0]), invalid_scatter
        )
    with pytest.raises(ValueError, match="outside embedded.*support"):
        report.compute_test_set_tf_conformance_audit(
            np.asarray([59.0]), np.asarray([20.0]), tf
        )
    with pytest.raises(ValueError, match="must be finite"):
        report.compute_test_set_tf_conformance_audit(
            np.asarray([200.0]), np.asarray([np.nan]), tf
        )
    with pytest.raises(ValueError, match="must exactly match"):
        report.compute_test_set_tf_conformance_audit(
            np.asarray([200.0]),
            np.asarray([20.0]),
            tf,
            generation_tf={**tf, "scatter_dex": 0.2},
        )


def test_test_set_rejects_standard_cache(tmp_path):
    report = _report()
    root = tmp_path / "cache" / "model" / "standard"
    _write_complete_cache(root, n=4)
    with pytest.raises(ValueError, match="not a compact test-set cache"):
        report.load_case(
            tmp_path / "cache", "model:standard", test_set=True
        )


def test_test_set_report_is_mean_only_and_uses_tf_candidate_weights(tmp_path):
    report = _report()
    root = tmp_path / "cache" / "model" / "xu1"
    _write_compact_test_cache(root)
    output = tmp_path / "test-set.html"

    report.main(
        [
            "--cache-root", str(tmp_path / "cache"),
            "--case", "model:xu1",
            "--output", str(output),
            "--bins", "2",
            "--test-set",
        ]
    )

    document = output.read_text(encoding="utf-8")
    assert "TF-conformed catalog test-set diagnostics" in document
    assert "Cross-cut operative shear summary" in document
    assert "Test-set provenance and generation contract" in document
    assert "Independent TF-conformance audit" in document
    assert "uniform KS distance D" in document
    assert "uniformity status" in document
    assert "xu_sample_1_fullfootprint.fits" in document
    assert "Image and central H-alpha S/N are record-backed" in document
    assert "image S/N" in document
    assert "Mean only" in document
    assert "Galaxy-weighting and shape-noise comparison" in document
    assert "subsequent ensemble statistics is <b>Population only</b>" in document
    assert "equal truth-galaxy mass without posterior-precision weighting" in document
    assert "equal over generated test galaxies (population only)" in document
    assert "TF posterior candidate-weight health" in document
    assert "posterior candidate ESS fraction" in document
    assert "TF importance weights normalized within each galaxy" in document
    assert "tf_importance_weighting" in document
    assert "no TF population ratio is applied across" in document
    assert "TF target population / TF posterior" not in document
    assert "Conditional MAP calibration" not in document
    assert ">MAP<" not in document
    assert "map_computed" not in document



def test_weighted_test_set_report_uses_regularized_galaxy_weights(tmp_path):
    report = _report()
    root = tmp_path / "cache" / "model" / "xu1"
    _write_compact_test_cache(root)
    output = tmp_path / "weighted-test-set.html"

    report.main(
        [
            "--cache-root", str(tmp_path / "cache"),
            "--case", "model:xu1",
            "--output", str(output),
            "--bins", "2",
            "--test-set",
            "--weighted",
        ]
    )

    document = output.read_text(encoding="utf-8")
    assert "subsequent ensemble statistics is <b>Shape-noise regularized</b>" in document
    assert "shape-noise-regularized posterior precision" in document
    assert (
        "1 / (posterior shear variance + fixed ensemble shape-noise variance)"
        in document
    )
    assert "Operative galaxy weighting" in document
    assert "Conditional MAP calibration" not in document
