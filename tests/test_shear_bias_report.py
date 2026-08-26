import importlib.util
import json
from pathlib import Path

import numpy as np


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
    "g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr",
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
            "image_snr": np.asarray([5.0, 100.0, 1000.0]),
            "central_halpha_snr": np.asarray([1.0, 20.0, 200.0]),
            "image_noise_sigma": np.ones(3),
            "central_spectral_noise_sigma": np.ones(3),
        }
    )

    class _Partitions:
        feature_names = FEATURES
        files = {name: () for name in arrays}

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
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.5, 2.5, n)
    mean = truth.copy()
    map_estimate = truth.copy()
    mean[:, FEATURES.index("theta_int")] += 2.0 * np.pi + 0.1
    map_estimate[:, FEATURES.index("theta_int")] -= 2.0 * np.pi + 0.2
    mean[:, FEATURES.index("sini")] += 0.05
    map_estimate[:, FEATURES.index("vcirc")] -= 2.0
    case = {
        "truth": truth,
        "feature_names": FEATURES,
        "populations": {
            "population": {
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
    np.testing.assert_allclose(by_key[("Mean", "sini")]["bias"], 0.05)
    np.testing.assert_allclose(by_key[("MAP", "vcirc")]["bias"], -2.0)
    rendered = report.nuisance_bias_table(rows)
    assert "weighted additive bias" in rendered
    assert "halpha_flux_true" in rendered


def test_conditional_calibration_has_sini_shape_noise_and_map_panels():
    report = _report()
    n = 30
    truth = np.zeros((n, len(FEATURES)), dtype=np.float64)
    truth[:, 0] = np.linspace(-0.08, 0.08, n)
    truth[:, 1] = np.linspace(0.08, -0.08, n)
    truth[:, FEATURES.index("hlr")] = np.linspace(0.2, 2.8, n)
    truth[:, FEATURES.index("sini")] = np.linspace(0.05, 0.95, n)
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
        "spectral reference quality",
        "true hlr",
        "true sini",
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


def test_weighted_flag_and_precision_cap_exclude_invalid_variance(tmp_path):
    report = _report()
    args = report.parse_args(
        ["--case", "model:dataset", "--output", str(tmp_path / "report.html"),
         "--weighted"]
    )
    assert args.weighted

    weight, diagnostics = report.compose_precision_weights(
        np.full(4, 0.25),
        np.asarray([1.0, 0.5, 0.25, np.nan]),
        np.asarray([1.0, 0.5, 0.25, 1.0]),
        50.0,
    )
    np.testing.assert_allclose(weight, [0.2, 0.4, 0.4, 0.0])
    np.testing.assert_allclose(diagnostics["precision_threshold"], 2.0)
    np.testing.assert_allclose(diagnostics["shape_noise_floor"], 1 / np.sqrt(2))
    assert diagnostics["capped_count"] == 1
    assert diagnostics["invalid_variance_count"] == 1
    np.testing.assert_allclose(diagnostics["capped_population_mass"], 1 / 3)
    np.testing.assert_allclose(
        diagnostics["invalid_variance_population_mass"], 0.25
    )


def test_precision_cap_sweep_reports_mean_m_and_ess_for_both_populations():
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

    rows = report.precision_cap_sweep(case, posterior, 0.02)
    assert len(rows) == 8
    for row in rows:
        expected = 0.03 if row["population_key"] == "proposal" else -0.04
        np.testing.assert_allclose(row["g1_m"], expected, atol=1e-12)
        np.testing.assert_allclose(row["g2_m"], expected, atol=1e-12)
        assert row["g1_ess"] > 0
        assert row["g2_ess"] > 0
    proposal_rows = [row for row in rows if row["population_key"] == "proposal"]
    assert proposal_rows[-1]["ess"] < proposal_rows[0]["ess"]
    rendered = report.precision_cap_sweep_table(rows)
    assert "uncapped / all" in rendered
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


def test_weighted_report_smoke_includes_cap_sweep_and_nuisance_plot(tmp_path):
    report = _report()
    n, draws = 36, 6
    truth = np.zeros((n, len(FEATURES)), dtype=np.float32)
    truth[:, 0] = np.linspace(-0.019, 0.019, n)
    truth[:, 1] = np.linspace(0.019, -0.019, n)
    truth[:, FEATURES.index("theta_int")] = np.linspace(-2.0, 2.0, n)
    truth[:, FEATURES.index("sini")] = np.linspace(0.1, 0.95, n)
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
    assert "Posterior-precision cap sweep" in document
    assert "uncapped / all" in document
    assert "population-weighted 95th-percentile cap" in document
    assert "nuisance posterior-mean bias versus truth" in document
