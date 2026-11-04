import numpy as np

from arch.diagnostics.likelihood_stack_pp_comparison import (
    grid_marginal_pit,
    pp_curve,
    weighted_sample_pit,
)


def test_weighted_sample_pit_uses_midpoint_for_exact_ties():
    samples = np.array(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
        ]
    )
    log_weights = np.zeros((2, 2))
    truth = np.array([[0.0, 0.5], [0.0, 0.0]])

    pits = weighted_sample_pit(samples, log_weights, truth)

    np.testing.assert_allclose(pits, [[0.25, 0.5], [0.5, 0.5]])


def test_grid_marginal_pit_is_half_at_center_for_uniform_grid():
    axis = np.array([-1.0, 0.0, 1.0])
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    log_r = np.zeros((1, len(grid)))

    pits = grid_marginal_pit(log_r, grid, np.array([[0.0, 0.0]]))

    np.testing.assert_allclose(pits, [[0.5, 0.5]])


def test_pp_curve_reports_finite_dkw_band_and_ks_distance():
    diagnostic = pp_curve(np.linspace(0.01, 0.99, 100))

    assert diagnostic["n"] == 100
    assert diagnostic["ks_distance"] < 0.02
    assert diagnostic["dkw_epsilon"] > 0.0
    assert diagnostic["central_coverage"] == 0.68
