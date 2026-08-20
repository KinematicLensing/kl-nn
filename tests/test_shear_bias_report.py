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
    proposal = report.load_shear_pit_values(case, "proposal")
    target = report.load_shear_pit_values(case, "tf_target")
    np.testing.assert_allclose(proposal["g1"], [0.5])
    np.testing.assert_allclose(target["g1"], [0.2])
    np.testing.assert_allclose(target["g1_retained_mass"], [1.0])


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
