import numpy as np
import pytest
import torch

from arch.diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_N_GALAXIES,
    BENCHMARK_N_REALIZATIONS,
    BENCHMARK_PREFIXES,
    BENCHMARK_SURFACE_SAMPLES,
    benchmark_noise_seeds,
    fit_component_mc,
    fixed_shear_grid,
    fixed_shear_group_ids,
    deterministic_nuisance_bank,
    marginalize_nuisance_log_surface,
    prefix_mean,
    prefix_surface_map,
    prefix_surface_mean,
    stack_log_surfaces,
    validate_exact_truth_groups,
)
from arch.diagnostics.likelihood_stack_nre_core import grid_posterior_moments
from arch.diagnostics.likelihood_stack_benchmark import (
    aggregate_results,
    aggregate_nuisance_results,
    paired_nuisance_bootstrap_indices,
    stacked_nre_means,
    stacked_nre_nuisance_means,
)
from arch.diagnostics.likelihood_stack_nuisance_benchmark import (
    ESTIMATORS as NUISANCE_ESTIMATORS,
    _candidate_bank,
    _log_integrand_9d_surfaces,
    _marginalized_9d_surfaces,
    aggregate_results as aggregate_nuisance_only_results,
    candidate_nre9d_estimators,
)
from arch.tf_prior import TFPrior, tf_log_prior_ratio
from arch import config
from arch.utils import denormalize


def test_fixed_shear_group_ids_have_exact_repeated_truth_layout():
    ids = fixed_shear_group_ids(n_cells=2, n_galaxies=3, n_realizations=4)
    assert ids.shape == (24, 3)
    assert np.array_equal(ids[:4, 0], np.zeros(4, dtype=np.int64))
    assert np.array_equal(ids[:4, 1], np.zeros(4, dtype=np.int64))
    assert np.array_equal(ids[:4, 2], np.arange(4))
    assert ids[4, 0] == 0
    assert ids[4, 1] == 1
    assert ids[-1].tolist() == [1, 2, 3]


def test_exact_truth_validation_rejects_changes_within_noise_group():
    truth = np.zeros((2, 3, 4, 2), dtype=np.float64)
    truth[:, 0, :, :] = 1.0
    truth[:, 1, :, :] = 2.0
    truth[:, 2, :, :] = 3.0
    validate_exact_truth_groups(
        truth,
        n_cells=2,
        n_galaxies=3,
        n_realizations=4,
    )
    truth[1, 2, 3, 1] = 4.0
    with pytest.raises(ValueError, match="fixed across realizations"):
        validate_exact_truth_groups(
            truth,
            n_cells=2,
            n_galaxies=3,
            n_realizations=4,
        )


def test_prefix_mean_aggregates_only_the_realization_axis():
    values = np.arange(2 * 3 * 4 * 2, dtype=np.float64).reshape(2, 3, 4, 2)
    result = prefix_mean(values, prefixes=(1, 2, 4))
    assert result.shape == (2, 3, 3, 2)
    np.testing.assert_array_equal(result[..., 0, :], values[..., 0, :])
    np.testing.assert_allclose(result[..., 2, :], values.mean(axis=-2))


def test_surface_prior_removal_is_explicit_but_constant_for_uniform_prior():
    bank = np.zeros((1, BENCHMARK_SURFACE_SAMPLES, 2), dtype=np.float64)
    bank[0, 0] = (0.1, 0.2)
    log_q = np.zeros((1, BENCHMARK_SURFACE_SAMPLES), dtype=np.float64)
    mapped, score = prefix_surface_map(
        bank,
        log_q,
        prefixes=(1,),
        candidates_per_realization=BENCHMARK_SURFACE_SAMPLES,
        prior_log_density=-9.0 * np.log(2.0),
    )
    np.testing.assert_array_equal(mapped[0], bank[0, 0])
    assert score[0] == pytest.approx(9.0 * np.log(2.0))


def test_candidate_bank_stack_uses_union_and_distinguishes_map_from_mean():
    bank = np.zeros((2, 2, 2), dtype=np.float64)
    bank[0, 0] = (-1.0, 0.0)
    bank[0, 1] = (0.0, 0.0)
    bank[1, 0] = (1.0, 0.0)
    bank[1, 1] = (0.5, 0.0)
    log_q = np.array(
        [
            [5.0, 0.0, -100.0, -100.0],
            [-1000.0, -1000.0, 5.0, 0.0],
        ]
    )
    mapped, _ = prefix_surface_map(
        bank,
        log_q,
        prefixes=(1, 2),
        candidates_per_realization=2,
        prior_log_density=0.0,
    )
    np.testing.assert_array_equal(mapped[0], (-1.0, 0.0))
    np.testing.assert_array_equal(mapped[1], (1.0, 0.0))
    assert mapped[1, 0] != pytest.approx(np.mean([bank[0, 0, 0], bank[1, 0, 0]]))


def test_candidate_bank_stack_mean_uses_surface_weights():
    bank = np.zeros((2, 2, 2), dtype=np.float64)
    bank[0, 0] = (-1.0, 0.0)
    bank[0, 1] = (0.0, 0.0)
    bank[1, 0] = (1.0, 0.0)
    bank[1, 1] = (0.5, 0.0)
    log_q = np.array(
        [
            [5.0, 0.0, -100.0, -100.0],
            [-1000.0, -1000.0, 5.0, 0.0],
        ]
    )

    means = prefix_surface_mean(
        bank,
        log_q,
        prefixes=(1, 2),
        candidates_per_realization=2,
        prior_log_density=0.0,
    )

    assert means[0, 0] == pytest.approx(-1.0, abs=1e-2)
    assert means[1, 0] > 0.8


def test_component_fit_recovers_m_and_c():
    truth = np.array(
        [[-0.08, -0.04], [0.0, 0.0], [0.08, 0.04]], dtype=np.float64
    )
    estimate = truth * np.array([1.1, 0.9]) + np.array([0.003, -0.002])
    fit = fit_component_mc(truth, estimate)
    assert fit["g1_m"] == pytest.approx(0.1)
    assert fit["g2_m"] == pytest.approx(-0.1)
    assert fit["g1_c"] == pytest.approx(0.003)
    assert fit["g2_c"] == pytest.approx(-0.002)


def test_noise_streams_are_reproducible_and_separate():
    first = benchmark_noise_seeds(42, 3, 7, 11)
    assert first == benchmark_noise_seeds(42, 3, 7, 11)
    assert first != benchmark_noise_seeds(42, 3, 7, 12)
    assert first != benchmark_noise_seeds(42, 4, 7, 11)


def test_nuisance_bootstrap_ids_are_paired_and_reproducible():
    first = paired_nuisance_bootstrap_indices(
        8, 3, n_bootstrap=12, seed=1234
    )
    second = paired_nuisance_bootstrap_indices(
        8, 3, n_bootstrap=12, seed=1234
    )
    assert first.shape == (12, 3)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first < 8))

    # One replicate's ID vector produces a fixed offset in every shear cell,
    # demonstrating the paired application of the same nuisance IDs.
    values = np.empty((25, 8, 2), dtype=np.float64)
    for cell in range(25):
        values[cell, :, 0] = cell * 100.0 + np.arange(8)
        values[cell, :, 1] = np.arange(8)
    sampled = values[:, first[0], :].mean(axis=1)
    np.testing.assert_allclose(np.diff(sampled[:, 0]), 100.0)
    np.testing.assert_allclose(sampled[:, 1], sampled[0, 1])


def test_nuisance_results_include_deterministic_bootstrap_m_intervals():
    truth = np.repeat(
        fixed_shear_grid()[:, np.newaxis, :],
        32,
        axis=1,
    )
    slopes = np.linspace(-0.15, 0.2, 32, dtype=np.float64)
    values = truth[:, :, np.newaxis, :] * (1.0 + slopes[None, :, None, None])
    values = np.repeat(values, 16, axis=2)
    surface = np.repeat(
        truth[:, :, np.newaxis, :],
        len(BENCHMARK_PREFIXES),
        axis=2,
    )
    result = {
        "truth_g": truth,
        "npe_mean": values,
        "nre_mean": values,
        "npe_map": values,
        "nre_map": values,
        "surface_map": surface,
        "surface_mean": surface,
        "nre_log_r": np.zeros((25, 32, 16, 441), dtype=np.float64),
        "nre_grid_phys": np.asarray(
            [(g1, g2) for g1 in np.linspace(-0.08, 0.08, 21)
             for g2 in np.linspace(-0.08, 0.08, 21)]
        ),
        "prefixes": np.asarray(BENCHMARK_PREFIXES),
    }

    rows = aggregate_nuisance_results(
        result,
        bootstrap_count=64,
        bootstrap_seed=9876,
    )
    row = next(
        item
        for item in rows
        if item["estimator"] == "npe_mean" and item["n_nuisance"] == 4
    )
    assert row["bootstrap_count"] == 64
    assert row["bootstrap_seed"] == 9876
    assert row["bootstrap_lower_percentile"] == 16.0
    assert row["bootstrap_upper_percentile"] == 84.0
    assert row["bootstrap_center"] == "median"
    for name in ("g1_m", "g2_m", "combined_m"):
        assert row[f"{name}_lower"] <= row[f"{name}_upper"]
    assert row["combined_m_upper"] > row["combined_m_lower"]
    values_r16 = values[:, :, :16, :].mean(axis=2)
    bootstrap_ids = paired_nuisance_bootstrap_indices(
        32, 4, n_bootstrap=64, seed=9876
    )
    bootstrap_fits = [
        fit_component_mc(
            fixed_shear_grid().astype(np.float64),
            values_r16[:, ids, :].mean(axis=1),
        )
        for ids in bootstrap_ids
    ]
    expected_median = np.median(
        [fit["combined_m"] for fit in bootstrap_fits]
    )
    assert row["combined_m"] == pytest.approx(expected_median)
    assert row["prefix_summary"]["combined_m"] != pytest.approx(
        row["combined_m"]
    )

    repeat = aggregate_nuisance_results(
        result,
        bootstrap_count=64,
        bootstrap_seed=9876,
    )
    assert rows == repeat


def test_stacked_nre_mean_sums_all_nuisances_for_each_noise_prefix():
    grid = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    log_r = np.zeros((1, 2, 2, 3), dtype=np.float64)
    log_r[0, 0, 0] = [0.0, 4.0, 0.0]
    log_r[0, 1, 0] = [0.0, 0.0, 4.0]
    log_r[0, 0, 1] = [0.0, 0.0, 8.0]

    means = stacked_nre_means(log_r, grid, prefixes=(1, 2))

    expected_r1 = grid_posterior_moments(
        np.asarray([[0.0, 4.0, 4.0]]), grid
    )["mean"][0]
    expected_r2 = grid_posterior_moments(
        np.asarray([[0.0, 4.0, 12.0]]), grid
    )["mean"][0]
    np.testing.assert_allclose(means[0, 0], expected_r1)
    np.testing.assert_allclose(means[0, 1], expected_r2)


def test_stacked_nre_mean_recovers_a_grid_shear():
    grid = np.asarray(
        [
            [-0.08, -0.08],
            [-0.08, 0.08],
            [0.08, -0.08],
            [0.08, 0.08],
        ]
    )
    log_r = np.zeros((1, 3, 2, 4), dtype=np.float64)
    log_r[..., 3] = 25.0

    means = stacked_nre_means(log_r, grid, prefixes=(1, 2))

    np.testing.assert_allclose(means[0, 0], [0.08, 0.08], atol=1e-10)
    np.testing.assert_allclose(means[0, 1], [0.08, 0.08], atol=1e-10)


def test_stacked_nre_nuisance_prefixes_sum_only_requested_ids():
    grid = np.asarray([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    log_r = np.zeros((1, 3, 2, 3), dtype=np.float64)
    log_r[0, 0, :, 0] = 8.0
    log_r[0, 1, :, 2] = 8.0
    log_r[0, 2, :, 1] = 8.0

    means = stacked_nre_nuisance_means(
        log_r,
        grid,
        nuisance_prefixes=(1, 2, 3),
        realization_prefix=2,
    )

    assert means.shape == (1, 3, 2)
    assert means[0, 0, 0] < -0.9
    assert means[0, 1, 0] == pytest.approx(0.0, abs=1e-3)
    assert means[0, 2, 0] == pytest.approx(0.0, abs=1e-3)


def test_nuisance_only_9d_nre_mean_and_map_use_tf_weighted_candidates():
    samples = np.zeros((2, 4, 9), dtype=np.float64)
    samples[:, :, 0] = np.asarray([-0.5, 0.5, -0.5, 0.5])
    samples[:, :, 1] = np.asarray([-0.5, -0.5, 0.5, 0.5])
    log_ratio = np.zeros((2, 4), dtype=np.float64)
    log_ratio[:, 3] = 8.0
    proposal = np.zeros((2, 4), dtype=np.float64)
    mean, map_estimate, ess = candidate_nre9d_estimators(
        samples,
        log_ratio,
        proposal,
        rmag_true=np.asarray([20.0, 20.0]),
        parameter_names=tuple(config.TARGET_NAMES),
        par_ranges=config.par_ranges,
        prior=TFPrior(),
    )
    assert mean.shape == (2, 2)
    assert map_estimate.shape == (2, 2)
    assert ess.shape == (2,)
    np.testing.assert_allclose(map_estimate, [[0.05, 0.05], [0.05, 0.05]])
    assert np.all(mean > 0.0)
    assert np.all(ess < 4.0)


def test_nuisance_only_aggregation_has_six_estimators_and_six_prefixes():
    truth_cells = fixed_shear_grid()
    grid = truth_cells.copy()
    truth = np.repeat(
        truth_cells[:, np.newaxis, :],
        BENCHMARK_N_GALAXIES,
        axis=1,
    )
    result = {
        "truth_g": truth,
        "grid_phys": grid,
        "npe_log_surface": np.zeros((25, 32, 25), dtype=np.float64),
        "nre2d_log_surface": np.zeros((25, 32, 25), dtype=np.float64),
        "nre9d_log_surface": np.zeros((25, 32, 25), dtype=np.float64),
    }
    for surface_name in (
        "npe_log_surface",
        "nre2d_log_surface",
        "nre9d_log_surface",
    ):
        for cell in range(25):
            result[surface_name][cell, :, cell] = 10.0
    metrics, per_shear = aggregate_nuisance_only_results(result)
    assert len(metrics) == 6 * 6
    assert len(per_shear) == 6 * 6 * 25
    assert {row["n_nuisance"] for row in metrics} == {1, 2, 4, 8, 16, 32}
    assert {row["estimator"] for row in metrics} == {
        name for name, _ in NUISANCE_ESTIMATORS
    }


def test_log_likelihood_surfaces_are_summed_before_estimation():
    surfaces = np.asarray(
        [
            [
                [0.0, 5.0, 0.0],
                [0.0, 4.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ]
    )
    stacked = stack_log_surfaces(surfaces, prefixes=(1, 2, 3))
    np.testing.assert_allclose(
        stacked,
        [[[0.0, 5.0, 0.0], [0.0, 9.0, 0.0], [5.0, 9.0, 0.0]]],
    )


def test_nuisance_bank_is_deterministic_and_bounded():
    first = deterministic_nuisance_bank(8, seed=123)
    second = deterministic_nuisance_bank(8, seed=123)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (8, 7)
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)


def test_nuisance_marginalization_uses_stable_log_mean_and_ess():
    log_integrand = np.asarray(
        [[-1000.0, -1000.0, -1000.0], [0.0, -10.0, -20.0]]
    )
    surface, ess = marginalize_nuisance_log_surface(
        log_integrand,
        return_ess=True,
    )
    assert np.all(np.isfinite(surface))
    assert ess[0] == pytest.approx(3.0)
    assert ess[1] < 1.1


def test_9d_npe_and_nre_surfaces_apply_the_same_tf_weighting():
    class FakeFlow:
        def log_prob(self, parameters, context):
            del context
            return parameters[:, 0] * parameters[:, 5]

    class FakeModel:
        flow = FakeFlow()

    class FakeHead:
        def __call__(self, context, parameters):
            del context
            return parameters[:, 0] * parameters[:, 5]

    grid_norm = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    nuisance_bank = np.zeros((3, 7), dtype=np.float64)
    nuisance_bank[:, 3] = [-1.0, 0.0, 1.0]
    candidates = _candidate_bank(grid_norm, nuisance_bank)
    model = FakeModel()
    nre_head = FakeHead()
    context = torch.zeros((1, 4))
    npe_surface, npe_ess, nre_surface, nre_ess = _marginalized_9d_surfaces(
        model,
        nre_head,
        context,
        grid_norm=grid_norm,
        nuisance_bank_norm=nuisance_bank,
        rmag_true=np.asarray([20.0]),
        names=tuple(config.TARGET_NAMES),
        par_ranges=config.par_ranges,
        tf_prior=TFPrior(),
        chunk_size=16,
    )
    expected_scores = (
        candidates[:, 0] * candidates[:, 5]
    ).reshape(2, 3)
    physical_candidates = denormalize(
        candidates,
        config.par_ranges,
        feature_names=tuple(config.TARGET_NAMES),
        target_transforms=config.TARGET_TRANSFORMS,
    )
    expected_tf = tf_log_prior_ratio(
        physical_candidates[:, 5].reshape(1, -1),
        np.asarray([[20.0]]),
        TFPrior(),
    ).reshape(2, 3)
    expected_surface, expected_ess = marginalize_nuisance_log_surface(
        expected_scores + expected_tf,
        return_ess=True,
    )
    np.testing.assert_allclose(npe_surface[0], expected_surface)
    np.testing.assert_allclose(nre_surface[0], expected_surface)
    np.testing.assert_allclose(npe_ess[0], expected_ess)
    np.testing.assert_allclose(nre_ess[0], expected_ess)


def test_9d_log_integrands_vectorize_over_galaxies_on_one_candidate_bank():
    class FakeFlow:
        def log_prob(self, parameters, context):
            del context
            return parameters[:, 0] * parameters[:, 5]

    class FakeModel:
        flow = FakeFlow()

    class FakeHead:
        def __call__(self, context, parameters):
            del context
            return parameters[:, 0] * parameters[:, 5]

    grid_norm = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
    nuisance_bank = np.zeros((3, 7), dtype=np.float64)
    nuisance_bank[:, 3] = [-1.0, 0.0, 1.0]
    context = torch.zeros((2, 4))
    npe_scores, nre_scores, tf_ratio = _log_integrand_9d_surfaces(
        FakeModel(),
        FakeHead(),
        context,
        grid_norm=grid_norm,
        nuisance_bank_norm=nuisance_bank,
        rmag_true=np.asarray([20.0, 20.0]),
        names=tuple(config.TARGET_NAMES),
        par_ranges=config.par_ranges,
        tf_prior=TFPrior(),
        chunk_size=4,
    )
    assert npe_scores.shape == (2, 2, 3)
    assert nre_scores.shape == (2, 2, 3)
    assert tf_ratio.shape == (2, 2, 3)
    candidates = _candidate_bank(grid_norm, nuisance_bank)
    expected_scores = (candidates[:, 0] * candidates[:, 5]).reshape(2, 3)
    np.testing.assert_allclose(npe_scores[0], expected_scores)
    np.testing.assert_allclose(nre_scores[1], expected_scores)


def test_strict_noise_stack_marginalizes_after_shared_noise_product():
    # The prior on a latent galaxy nuisance parameter is applied once after
    # multiplying repeated-noise likelihoods, rather than once per exposure.
    noise_log_likelihood = np.asarray(
        [
            [[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]],
        ]
    )
    galaxy_prior_log_density = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    strict = marginalize_nuisance_log_surface(
        noise_log_likelihood.sum(axis=0) + galaxy_prior_log_density,
    )
    per_exposure_marginalized = marginalize_nuisance_log_surface(
        (noise_log_likelihood + galaxy_prior_log_density[None]).sum(axis=0),
    )
    assert strict[0] != pytest.approx(per_exposure_marginalized[0])


def test_aggregate_results_includes_stacked_nre_estimator():
    shear_values = (-0.08, -0.04, 0.0, 0.04, 0.08)
    truth_cells = np.asarray(
        [(g1, g2) for g1 in shear_values for g2 in shear_values],
        dtype=np.float64,
    )
    truth_g = np.repeat(
        truth_cells[:, np.newaxis, :],
        BENCHMARK_N_GALAXIES,
        axis=1,
    )
    point = {
        name: np.zeros(
            (
                len(truth_cells),
                BENCHMARK_N_GALAXIES,
                BENCHMARK_N_REALIZATIONS,
                2,
            ),
            dtype=np.float64,
        )
        for name in ("npe_mean", "nre_mean", "npe_map", "nre_map")
    }
    axis = np.linspace(-0.08, 0.08, 21)
    grid = np.asarray([(g1, g2) for g1 in axis for g2 in axis])
    result = {
        **point,
        "surface_map": np.zeros(
            (
                len(truth_cells),
                BENCHMARK_N_GALAXIES,
                len(BENCHMARK_PREFIXES),
                9,
            ),
            dtype=np.float64,
        ),
        "surface_mean": np.zeros(
            (
                len(truth_cells),
                BENCHMARK_N_GALAXIES,
                len(BENCHMARK_PREFIXES),
                9,
            ),
            dtype=np.float64,
        ),
        "truth_g": truth_g,
        "truth_full": np.zeros(
            (len(truth_cells), BENCHMARK_N_GALAXIES, 9),
            dtype=np.float64,
        ),
        "prefixes": np.asarray(BENCHMARK_PREFIXES),
        "nre_log_r": np.zeros(
            (
                len(truth_cells),
                BENCHMARK_N_GALAXIES,
                BENCHMARK_N_REALIZATIONS,
                len(grid),
            ),
            dtype=np.float64,
        ),
        "nre_grid_phys": grid,
    }

    rows, _, _ = aggregate_results(result)

    assert len(rows) == len(BENCHMARK_PREFIXES) * 7
    assert {
        row["estimator"] for row in rows
    } == {
        "npe_mean",
        "nre_mean",
        "npe_map",
        "nre_map",
        "npe_9d_stack_map",
        "npe_9d_stack_mean",
        "nre_stack_mean",
    }
