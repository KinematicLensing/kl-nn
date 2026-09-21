from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from diagnostics.likelihood_stack_core import (
    ABS_G_SPLIT,
    COVERAGE_DIR,
    G01_NPE,
    NRE_BATCH_SIZE,
    NRE_CONTEXT_DIM,
    NRE_DIR,
    NRE_EPOCHS,
    NRE_GRID_N,
    NRE_HIDDEN_DIMS,
    NRE_INFER_DATASET,
    NRE_LR,
    NRE_NAME,
    NRE_PREFIXES,
    NRE_TRAIN_DATASET,
    NRE_VALID_DATASET,
    STACK_DIR,
)
from diagnostics.likelihood_stack_nre_core import (
    QUANTILES,
    RatioHead,
    discrete_posterior,
    freeze_encoder,
    grid_map,
    grid_marginal_intervals,
    grid_posterior_moments,
    grouped_stack_map,
    nre_bce_loss,
    nre_pair_logits_and_labels,
    nre_slice_masks,
    nre_takeaway,
    normalized_shear_grid,
    prefix_groups,
    shuffle_shear,
    stack_log_r,
    weighted_quantiles,
    zero_logit_bce,
)
from diagnostics.likelihood_stack_nre_infer import (
    STACK_SLICES,
    parse_args as parse_infer_args,
    stack_table,
)
from diagnostics.likelihood_stack_nre_train import parse_args as parse_train_args
from diagnostics.likelihood_stack_write_index import STAGES


def test_cli_defaults_match_the_nre_plan():
    train = parse_train_args([])
    infer = parse_infer_args([])
    assert train.parent_npe == G01_NPE
    assert train.nre_name == NRE_NAME
    assert NRE_NAME != G01_NPE
    assert train.train_data.name == NRE_TRAIN_DATASET
    assert train.valid_data.name == NRE_VALID_DATASET
    assert train.epochs == NRE_EPOCHS == 80
    assert train.batch_size == NRE_BATCH_SIZE == 256
    assert train.learning_rate == pytest.approx(NRE_LR)
    assert tuple(train.hidden_dims) == NRE_HIDDEN_DIMS == (512, 256)
    assert infer.parent_npe == G01_NPE
    assert infer.nre_name == NRE_NAME
    assert infer.data_dir.name == NRE_INFER_DATASET
    assert infer.report_dir == NRE_DIR
    assert infer.grid_n == NRE_GRID_N == 21
    assert infer.prefixes == list(NRE_PREFIXES) == [1, 8, 32, 128]
    assert STAGES[-2][0] == "03_nre"
    assert STAGES[-1][0] == "04_benchmark"
    assert NRE_DIR != STACK_DIR
    assert NRE_DIR != COVERAGE_DIR


def test_shuffle_permutes_g_not_context_and_rejects_batch_one():
    shear = torch.tensor(
        [[0.1, -0.2], [0.3, 0.4], [-0.5, 0.0], [0.0, 0.8]],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(0)
    shuffled = shuffle_shear(shear, generator=generator)
    assert shuffled.shape == shear.shape
    assert not torch.equal(shuffled, shear)
    original = {tuple(row.tolist()) for row in shear}
    assert {tuple(row.tolist()) for row in shuffled} == original
    with pytest.raises(ValueError, match="batch size"):
        shuffle_shear(shear[:1])


def test_joint_rows_are_labeled_one_and_shuffled_rows_zero():
    context = torch.zeros(6, 4)
    shear = torch.linspace(-0.3, 0.3, 12).reshape(6, 2)

    class ReadG1(nn.Module):
        def forward(self, _context, shear_batch):
            return shear_batch[:, 0]

    logits, labels = nre_pair_logits_and_labels(ReadG1(), context, shear)
    assert logits.shape == (12,)
    assert labels[:6].eq(1.0).all()
    assert labels[6:].eq(0.0).all()
    assert torch.equal(logits[:6], shear[:, 0])


def test_zero_head_bce_is_log_two():
    class ZeroHead(nn.Module):
        def forward(self, context, _shear):
            return torch.zeros(context.shape[0], dtype=context.dtype)

    context = torch.randn(8, 3)
    shear = torch.randn(8, 2)
    loss = nre_bce_loss(ZeroHead(), context, shear)
    assert loss.item() == pytest.approx(zero_logit_bce())
    assert loss.item() == pytest.approx(math.log(2.0))


def test_logit_equals_log_r_on_a_toy_ratio():
    """Known r(x, g) = exp(g1 - 0.25 g2); the classifier logit is log r."""

    class ToyRatio(nn.Module):
        def forward(self, _context, shear):
            return shear[:, 0] - 0.25 * shear[:, 1]

    context = torch.zeros(5, 2)
    shear = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, -0.2],
            [-0.8, 0.8],
            [1.0, 1.0],
            [-0.3, 0.6],
        ]
    )
    log_r = torch.log(
        torch.exp(shear[:, 0] - 0.25 * shear[:, 1])
    )
    logits = ToyRatio()(context, shear)
    assert torch.allclose(logits, log_r)


def test_ratio_head_accepts_frozen_context_and_two_shear_components():
    head = RatioHead(context_dim=NRE_CONTEXT_DIM, hidden_dims=NRE_HIDDEN_DIMS)
    context = torch.zeros(4, NRE_CONTEXT_DIM)
    shear = torch.zeros(4, 2)
    logits = head(context, shear)
    assert logits.shape == (4,)
    with pytest.raises(ValueError, match="context"):
        head(torch.zeros(4, 8), shear)


def test_freeze_encoder_drops_gradients():
    encoder = nn.Linear(3, 3)
    assert encoder.weight.requires_grad
    freeze_encoder(encoder)
    assert encoder.training is False
    assert not encoder.weight.requires_grad


def test_stacked_argmax_on_a_grid_recovers_the_shared_peak():
    grid, axis = normalized_shear_grid(21)
    assert grid.shape == (21 * 21, 2)
    assert axis.shape == (21,)
    true = np.array([0.4, -0.2])
    log_r = np.stack(
        [
            -0.5 * np.sum((grid - true) ** 2, axis=1) / 0.08**2,
            -0.5 * np.sum((grid - true) ** 2, axis=1) / 0.12**2,
        ]
    )
    stacked = stack_log_r(log_r)
    peak = grid_map(stacked, grid)
    step = axis[1] - axis[0]
    assert peak == pytest.approx(true, abs=step)
    grouped = grouped_stack_map(log_r, grid, np.array([[0, 1]], dtype=np.int64))
    assert grouped[0] == pytest.approx(peak)


def test_discrete_16_84_covers_a_known_peak():
    grid, axis = normalized_shear_grid(21)
    true = np.array([0.0, 0.0])
    log_r = -0.5 * np.sum((grid - true) ** 2, axis=1) / 0.15**2
    intervals = grid_marginal_intervals(log_r, axis, n_grid=21)
    assert intervals["g1_lower"][0] < 0.0 < intervals["g1_upper"][0]
    mass = discrete_posterior(log_r).reshape(21, 21)
    lo, hi = weighted_quantiles(axis, mass.sum(axis=1), QUANTILES)
    assert lo == pytest.approx(intervals["g1_lower"][0])
    assert hi == pytest.approx(intervals["g1_upper"][0])


def test_grid_posterior_mean_is_the_probability_weighted_shear():
    grid = np.array([[-1.0, -0.5], [0.0, 0.0], [1.0, 0.5]])
    log_r = np.log(np.array([0.2, 0.5, 0.3]))
    moments = grid_posterior_moments(log_r, grid)
    assert moments["mean"][0, 0] == pytest.approx(0.1)
    assert moments["mean"][0, 1] == pytest.approx(0.05)
    assert moments["std"][0, 0] == pytest.approx(np.sqrt(0.49))
    assert moments["std"][0, 1] == pytest.approx(np.sqrt(0.49) / 2.0)


def test_prefix_groups_are_disjoint_and_drop_the_remainder():
    groups = prefix_groups(10, 4, seed=42)
    assert groups.shape == (2, 4)
    assert len(np.unique(groups)) == 8
    empty = prefix_groups(3, 8, seed=42)
    assert empty.shape == (0, 8)


def test_stack_table_keeps_stack_size_separate_from_number_of_groups():
    truth = np.column_stack((np.linspace(-0.1, 0.1, 8), np.zeros(8)))
    npe_mean = 0.5 * truth
    grid = np.array([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0]])
    log_r = -np.square(truth[:, :1] - grid[None, :, 0])
    masks = {
        name: np.ones(8, dtype=bool)
        for name in STACK_SLICES
    }
    rows = stack_table(
        truth,
        log_r,
        npe_mean,
        grid,
        masks,
        prefixes=(1, 2),
        seed=42,
    )
    full_nre = [
        row
        for row in rows
        if row["slice"] == "full" and row["estimator"] == "nre_stack_map"
    ]
    assert [row["stack_size"] for row in full_nre] == [1, 2]
    assert [row["n_groups"] for row in full_nre] == [8, 4]
    assert [row["n"] for row in full_nre] == [8, 4]


def test_nre_slice_masks_split_inner_outer_and_amplitude_bins():
    truth = np.array(
        [
            [0.01, 0.0],
            [0.03, 0.0],
            [0.06, 0.0],
            [0.09, 0.0],
            [0.05, 0.0],
        ]
    )
    masks = nre_slice_masks(truth, split=ABS_G_SPLIT)
    assert masks["inner"].tolist() == [True, True, False, False, False]
    assert masks["outer"].tolist() == [False, False, True, True, False]
    assert int(masks["amp_00_025"].sum()) == 1
    assert int(masks["amp_025_050"].sum()) == 1
    assert int(masks["amp_050_075"].sum()) == 2
    assert int(masks["amp_075_100"].sum()) == 1


def test_nre_takeaway_reads_useable_likelihood_versus_weak_context():
    recovered = [
        {
            "estimator": "nre_stack_map",
            "slice": "outer",
            "stack_size": 128,
            "g1_m": -0.04,
            "g2_m": -0.06,
        },
        {
            "estimator": "npe_mean_of_means",
            "slice": "outer",
            "stack_size": 128,
            "g1_m": -0.82,
            "g2_m": -0.80,
        },
    ]
    text = nre_takeaway([], recovered)
    assert "useable shear likelihood" in text
    both_shrunk = [
        {
            "estimator": "nre_stack_map",
            "slice": "outer",
            "stack_size": 128,
            "g1_m": -0.81,
            "g2_m": -0.79,
        },
        {
            "estimator": "npe_mean_of_means",
            "slice": "outer",
            "stack_size": 128,
            "g1_m": -0.84,
            "g2_m": -0.83,
        },
    ]
    assert "does not invent a likelihood" in nre_takeaway([], both_shrunk)
