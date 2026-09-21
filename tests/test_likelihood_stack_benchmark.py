import numpy as np
import pytest

from arch.diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_PREFIXES,
    BENCHMARK_SURFACE_SAMPLES,
    benchmark_noise_seeds,
    fit_component_mc,
    fixed_shear_grid,
    fixed_shear_group_ids,
    prefix_mean,
    prefix_surface_map,
    validate_exact_truth_groups,
)
from arch.diagnostics.likelihood_stack_benchmark import (
    aggregate_nuisance_results,
    paired_nuisance_bootstrap_indices,
)


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
    for name in ("g1_m", "g2_m", "combined_m"):
        assert row[f"{name}_lower"] <= row[f"{name}_upper"]
    assert row["combined_m_upper"] > row["combined_m_lower"]

    repeat = aggregate_nuisance_results(
        result,
        bootstrap_count=64,
        bootstrap_seed=9876,
    )
    assert rows == repeat
