import json

import numpy as np
import pytest
import torch

nautilus = pytest.importorskip("nautilus")

from arch.diagnostics.likelihood_stack_nre_9d_infer import bounded_nre_log_density
from arch.diagnostics.likelihood_stack_nre_9d_nautilus import (
    artifact_path,
    build_uniform_prior,
    load_artifact,
    load_or_create_selection,
    normalized_weights,
    save_artifact,
    summarize_posterior,
)
from arch.diagnostics.likelihood_stack_nre_9d_nautilus_report import main as report_main
from arch.diagnostics.likelihood_stack_nre_9d_nautilus_report import (
    posterior_pit,
    posterior_pp_curve,
)
from arch.tf_prior import TFPrior
from arch import config


NAMES = tuple(config.TARGET_NAMES)


def test_uniform_prior_has_expected_box():
    prior = build_uniform_prior(NAMES)
    assert prior.dimensionality() == 9
    assert np.allclose(prior.unit_to_physical(np.full((2, 9), 0.5)), 0.0)
    points = prior.unit_to_physical(np.asarray([[0.0] * 9, [1.0] * 9]))
    assert np.all(points >= -1.0)
    assert np.all(points <= 1.0)


def test_selection_is_deterministic_and_records_dataset_size(tmp_path):
    path = tmp_path / "selection.npz"
    first = load_or_create_selection(
        path,
        dataset=tmp_path / "small_10k",
        dataset_size=10_000,
        count=100,
        seed=42,
    )
    second = load_or_create_selection(
        path,
        dataset=tmp_path / "small_10k",
        dataset_size=10_000,
        count=100,
        seed=999,
    )
    assert np.array_equal(first, second)
    payload = np.load(path, allow_pickle=False)
    assert int(payload["dataset_size"]) == 10_000
    assert len(np.unique(first)) == 100


def test_bounded_log_density_is_vectorized():
    class ZeroHead:
        def __call__(self, context, parameters):
            return torch.zeros(parameters.shape[0], device=parameters.device)

    values = bounded_nre_log_density(
        np.zeros((4, 9), dtype=np.float64),
        head=ZeroHead(),
        context=torch.zeros(1152),
        rmag_true=18.0,
        parameter_names=NAMES,
        par_ranges=config.par_ranges,
        prior=TFPrior(),
    )
    assert values.shape == (4,)
    assert np.all(np.isfinite(values))


def test_posterior_pp_curve_has_uniform_reference_metrics():
    values, empirical, distance = posterior_pp_curve(
        np.asarray([0.1, 0.3, 0.7, 0.9])
    )
    assert np.allclose(values, [0.1, 0.3, 0.7, 0.9])
    assert np.allclose(empirical, [0.25, 0.5, 0.75, 1.0])
    assert np.isclose(distance, 0.2)


def test_posterior_pit_evaluates_weighted_truth_cdf():
    artifact = {
        "points": np.vstack(
            [np.full(len(NAMES), -0.5), np.full(len(NAMES), 0.5)]
        ),
        "log_weights": np.zeros(2),
        "truth": np.zeros(len(NAMES)),
    }
    pits = posterior_pit(artifact, names=NAMES)
    assert np.allclose(pits[:2], 0.5)


def test_artifact_round_trip_and_posterior_summary(tmp_path):
    rng = np.random.default_rng(42)
    points = rng.uniform(-1.0, 1.0, size=(32, 9))
    log_weights = rng.normal(size=32)
    log_likelihood = rng.normal(size=32)
    truth = np.zeros(9)
    summary = summarize_posterior(
        points,
        log_weights,
        log_likelihood,
        parameter_names=NAMES,
        par_ranges=config.par_ranges,
        target_transforms=config.TARGET_TRANSFORMS,
        sampler_n_eff=25.0,
        sampler_n_like=100,
        sampler_log_z=-3.0,
        sampler_eta=0.2,
        runtime_s=1.5,
    )
    path = artifact_path(tmp_path, 17)
    save_artifact(
        path,
        index=17,
        points=points,
        log_weights=log_weights,
        log_likelihood=log_likelihood,
        truth_physical=truth,
        summary=summary,
        parameter_names=NAMES,
    )
    loaded = load_artifact(path)
    assert loaded["index"] == 17
    assert loaded["points"].shape == (32, 9)
    assert loaded["parameter_names"] == NAMES
    assert np.isclose(loaded["summary"]["nautilus_n_eff"], 25.0)
    assert np.isclose(normalized_weights(loaded["log_weights"]).sum(), 1.0)


def test_report_generation_with_small_campaign(tmp_path):
    run_dir = tmp_path / "cache" / "model" / "run"
    report_dir = tmp_path / "report"
    run_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)
    truth = np.zeros(9)
    indices = np.asarray([3, 7, 11, 19], dtype=np.int64)
    for row, index in enumerate(indices):
        summary = {
            "g1_map_like": 0.0,
            "g2_map_like": 0.0,
            "effective_sample_size": 16.0,
            "nautilus_n_eff": 24.0,
            "n_like": 50 + row,
            "eta": 0.2,
            "log_z_relative": -1.0,
            "runtime_s": 0.1,
            "status": "completed",
        }
        save_artifact(
            artifact_path(run_dir, int(index)),
            index=int(index),
            points=rng.uniform(-1.0, 1.0, size=(24, 9)),
            log_weights=rng.normal(size=24),
            log_likelihood=rng.normal(size=24),
            truth_physical=truth,
            summary=summary,
            parameter_names=NAMES,
        )
    np.savez(run_dir / "selection.npz", index=indices)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "sampler": "nautilus",
                "run_name": "smoke",
                "data_dir": "small_10k",
                "n_expected": 4,
                "selection_file": str(run_dir / "selection.npz"),
            }
        )
    )
    report_main(
        [
            "--run-dir",
            str(run_dir),
            "--report-dir",
            str(report_dir),
            "--selection-file",
            str(run_dir / "selection.npz"),
            "--n-featured",
            "1",
            "--bootstrap-replicates",
            "5",
            "--overwrite",
        ]
    )
    assert (report_dir / "report.html").is_file()
    assert (report_dir / "report.json").is_file()
    assert len(list((report_dir / "corner_plots").glob("*.png"))) == 1
    payload = json.loads((report_dir / "report.json").read_text())
    assert set(payload["posterior_pp"]) == {"g1", "g2"}
    assert "standardized_residual" in payload["per_galaxy"][0]
    assert (report_dir / "normalized_residuals_by_parameter.png").is_file()
    assert (report_dir / "g_estimates_vs_truth.png").is_file()
    assert (report_dir / "posterior_pp.png").is_file()
    assert "m_vs_nuisance_stacks.png" not in payload["metadata"]["figures"].values()
    html = (report_dir / "report.html").read_text()
    assert "<h2>Run metadata</h2>" not in html
    assert "<h2>Per-galaxy posterior summaries</h2>" not in html
