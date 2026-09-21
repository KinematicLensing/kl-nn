import numpy as np
import pytest

from arch.diagnostics.likelihood_stack_benchmark_core import (
    BENCHMARK_PREFIXES,
    BENCHMARK_SURFACE_SAMPLES,
    benchmark_noise_seeds,
    fit_component_mc,
    fixed_shear_group_ids,
    prefix_mean,
    prefix_surface_map,
    validate_exact_truth_groups,
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
