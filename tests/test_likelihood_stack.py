from __future__ import annotations

import numpy as np
import pytest

from diagnostics.likelihood_stack_core import (
    ABS_G_MIN,
    ABS_G_SPLIT,
    COVERAGE_DIR,
    G01_CASE,
    G01_NPE,
    N_REALIZATIONS,
    N_SAMPLES,
    N_SELECTED,
    PREFIXES,
    PROPOSAL_PER_REALIZATION,
    SELECT_DIR,
    STACK_DIR,
    abs_g_split_masks,
    coverage_takeaway,
    draw_selected_indices,
    eligible_mask,
    galaxy_shape_variance,
    interval_coverage,
    inverse_variance_weights,
    map_index,
    noise_seeds,
    prefix_ivw,
    prefix_mean_of_means,
    prefix_stack_map,
    residual_calibration,
    select_large_shrinkage,
    shrinkage_along_truth,
    shrinkage_pool_mask,
    stacked_log_prob,
    takeaway,
    vector_median_abs_residual,
)
from diagnostics.likelihood_stack_coverage import parse_args as parse_coverage_args
from diagnostics.likelihood_stack_select import parse_args as parse_select_args
from diagnostics.likelihood_stack import parse_args as parse_stack_args


def test_cli_defaults_match_the_plan():
    select = parse_select_args([])
    stack = parse_stack_args([])
    assert select.case == G01_CASE
    assert select.n_selected == N_SELECTED == 128
    assert select.abs_g_min == pytest.approx(ABS_G_MIN)
    assert select.report_dir == SELECT_DIR
    assert stack.model_name == G01_NPE
    assert stack.n_realizations == N_REALIZATIONS == 16
    assert stack.n_samples == N_SAMPLES == 2048
    assert stack.proposal_per_realization == PROPOSAL_PER_REALIZATION == 256
    assert stack.report_dir == STACK_DIR
    coverage = parse_coverage_args([])
    assert coverage.case == G01_CASE
    assert coverage.abs_g_split == pytest.approx(ABS_G_SPLIT)
    assert coverage.report_dir == COVERAGE_DIR
    assert PREFIXES == (1, 2, 4, 8, 16)


def test_abs_g_split_masks_exclude_the_cut_from_both_halves():
    truth = np.array(
        [
            [0.03, 0.0],
            [0.05, 0.0],
            [0.08, 0.0],
            [np.nan, 0.0],
        ]
    )
    masks = abs_g_split_masks(truth, split=0.05)
    assert masks["inner"].tolist() == [True, False, False, False]
    assert masks["outer"].tolist() == [False, False, True, False]
    assert masks["full"].tolist() == [True, True, True, False]


def test_interval_coverage_matches_a_known_fraction():
    truth = np.array([0.0, 0.1, 0.2, 0.3])
    lower = np.full(4, -0.05)
    upper = np.array([0.05, 0.05, 0.25, 0.25])
    metrics = interval_coverage(truth, lower, upper)
    assert metrics["n"] == 4
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["coverage_se"] == pytest.approx(np.sqrt(0.5 * 0.5 / 4))
    assert metrics["delta"] == pytest.approx(0.5 - 0.68)


def test_coverage_takeaway_reads_inner_undercoverage():
    rows = [
        {
            "posterior": "proposal",
            "slice": "inner",
            "g1_coverage": 0.46,
            "g2_coverage": 0.46,
        },
        {
            "posterior": "proposal",
            "slice": "outer",
            "g1_coverage": 0.82,
            "g2_coverage": 0.80,
        },
    ]
    text = coverage_takeaway(rows)
    assert "inner half sits well below 68%" in text
    reversed_rows = [
        {
            "posterior": "proposal",
            "slice": "inner",
            "g1_coverage": 0.91,
            "g2_coverage": 0.91,
        },
        {
            "posterior": "proposal",
            "slice": "outer",
            "g1_coverage": 0.62,
            "g2_coverage": 0.63,
        },
    ]
    assert "inner half overcovers" in coverage_takeaway(reversed_rows)
    honest = [
        {
            "posterior": "proposal",
            "slice": "inner",
            "g1_coverage": 0.67,
            "g2_coverage": 0.68,
        },
        {
            "posterior": "proposal",
            "slice": "outer",
            "g1_coverage": 0.69,
            "g2_coverage": 0.70,
        },
    ]
    assert "Both halves sit near 68%" in coverage_takeaway(honest)


def test_shrinkage_is_negative_when_the_mean_moves_toward_zero():
    truth = np.array([[0.06, 0.0], [0.0, -0.08]])
    estimate = np.array([[0.03, 0.0], [0.0, -0.02]])
    shrinkage = shrinkage_along_truth(estimate, truth)
    assert shrinkage[0] == pytest.approx(-0.5)
    assert shrinkage[1] == pytest.approx(-0.75)


def test_eligible_mask_drops_small_shear():
    truth = np.array([[0.01, 0.0], [0.03, 0.0], [0.0, 0.025]])
    mask = eligible_mask(truth, abs_g_min=0.02)
    assert mask.tolist() == [False, True, True]


def test_select_large_shrinkage_draws_from_the_more_shrunk_half():
    rng = np.random.default_rng(0)
    truth = np.column_stack((np.full(40, 0.05), np.zeros(40)))
    estimate = truth.copy()
    estimate[:20, 0] = 0.01
    estimate[20:, 0] = 0.049
    choice = select_large_shrinkage(estimate, truth, n=8, seed=3)
    assert choice["n_eligible"] == 40
    assert choice["n_pool"] == 20
    assert choice["n_selected"] == 8
    assert np.all(choice["selected"] < 20)
    again = select_large_shrinkage(estimate, truth, n=8, seed=3)
    assert again["selected"].tolist() == choice["selected"].tolist()
    other = select_large_shrinkage(estimate, truth, n=8, seed=4)
    assert other["selected"].tolist() != choice["selected"].tolist()
    del rng


def test_draw_selected_indices_requires_a_large_enough_pool():
    with pytest.raises(ValueError, match="need 4"):
        draw_selected_indices(np.array([1, 2, 3]), n=4, seed=0)


def test_shrinkage_pool_mask_keeps_values_at_or_below_the_median():
    shrinkage = np.array([-0.8, -0.1, -0.4, 0.2, np.nan])
    eligible = np.array([True, True, True, True, False])
    pool = shrinkage_pool_mask(shrinkage, eligible)
    # eligible finite values: -0.8, -0.1, -0.4, 0.2; median = -0.25
    assert pool.tolist() == [True, False, True, False, False]


def test_ivw_recovers_the_precision_weighted_mean():
    means = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        ]
    )
    variances = np.array([[1.0, 3.0, 1.0]])
    stacked = prefix_ivw(means, variances, prefixes=(1, 2, 3))
    assert stacked.shape == (1, 3, 2)
    assert stacked[0, 0].tolist() == pytest.approx([0.0, 0.0])
    # weights 1 and 1/3
    assert stacked[0, 1].tolist() == pytest.approx([0.25, 0.25])
    # weights 1, 1/3, 1 -> (0 + 1/3 + 2) / (1 + 1/3 + 1) = 2.333... / 2.333... = 1
    assert stacked[0, 2].tolist() == pytest.approx([1.0, 1.0])


def test_mean_of_means_is_the_equal_weight_prefix_average():
    means = np.array([[[0.0, 2.0], [2.0, 0.0], [4.0, 4.0]]])
    stacked = prefix_mean_of_means(means, prefixes=(1, 2, 3))
    assert stacked[0, 0].tolist() == pytest.approx([0.0, 2.0])
    assert stacked[0, 1].tolist() == pytest.approx([1.0, 1.0])
    assert stacked[0, 2].tolist() == pytest.approx([2.0, 2.0])


def test_zero_variance_draws_get_zero_ivw_weight():
    weights = inverse_variance_weights(np.array([0.0, 2.0, np.nan]))
    assert weights.tolist() == pytest.approx([0.0, 0.5, 0.0])
    assert galaxy_shape_variance(np.array([4.0]), np.array([0.0])) == pytest.approx(2.0)


def test_stack_map_uses_only_the_prefix_observations_and_proposals():
    bank = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 2.0],
        ]
    )
    log_probs = np.array(
        [
            [0.0, 5.0, 0.0, 9.0],
            [0.0, 0.0, 4.0, 1.0],
        ]
    )
    mapped = prefix_stack_map(
        bank, log_probs, prefixes=(1, 2), proposal_per_realization=2
    )
    # Prefix 1 uses first two candidates and the first row: argmax is candidate 1.
    assert mapped[0].tolist() == pytest.approx([1.0, 0.0])
    # Prefix 2 uses all four, sums rows: scores 0, 5, 4, 10 -> candidate 3.
    assert mapped[1].tolist() == pytest.approx([2.0, 2.0])


def test_stacked_log_prob_and_map_index_ignore_nonfinite_tails():
    stacked = stacked_log_prob(np.array([[1.0, np.nan], [2.0, 0.0]]))
    assert stacked[0] == pytest.approx(3.0)
    assert map_index(np.array([np.nan, 1.0, 4.0, -np.inf])) == 2


def test_residual_calibration_reports_multiplicative_m():
    truth = np.array([0.02, 0.04, 0.06, 0.08])
    estimate = 0.5 * truth + 0.001
    metrics = residual_calibration(truth, estimate)
    assert metrics["m"] == pytest.approx(-0.5)
    assert metrics["c"] == pytest.approx(0.001)
    assert vector_median_abs_residual(
        np.column_stack((truth, np.zeros_like(truth))),
        np.column_stack((estimate, np.zeros_like(truth))),
    ) == pytest.approx(np.median(np.abs(estimate - truth)))


def test_takeaway_separates_shrinkage_from_a_wrong_likelihood():
    recovered = [
        {
            "estimator": "ivw",
            "n_realizations": 16,
            "g1_m": -0.45,
            "g2_m": -0.47,
        },
        {
            "estimator": "stack_map",
            "n_realizations": 16,
            "g1_m": -0.04,
            "g2_m": 0.02,
        },
    ]
    text = takeaway(recovered)
    assert "Mean as a point estimator" in text
    still_wrong = [
        {
            "estimator": "ivw",
            "n_realizations": 16,
            "g1_m": -0.4,
            "g2_m": -0.4,
        },
        {
            "estimator": "stack_map",
            "n_realizations": 16,
            "g1_m": -0.4,
            "g2_m": -0.35,
        },
    ]
    assert "miscalibrated" in takeaway(still_wrong)


def test_noise_seeds_are_independent_per_galaxy_and_realization():
    first = noise_seeds(42, 0, 10)
    second = noise_seeds(42, 1, 10)
    other = noise_seeds(42, 0, 11)
    assert first[1] == first[0] + 17
    assert first != second
    assert first != other
    assert second != other
