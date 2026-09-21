import numpy as np
import pytest
import torch
from torch import nn

from config import TARGET_NAMES, canonical_parameter_ranges
from diagnostics.likelihood_stack_nre_9d_infer import (
    bounded_nre_log_density,
    run_emcee,
    summarize_shear_samples,
)
from diagnostics.likelihood_stack_nre_core import (
    RatioHead,
    nre_pair_logits_and_labels,
    shuffle_parameters,
)
from tf_prior import TFPrior, tf_log_prior_ratio


class LinearNineDHead(nn.Module):
    def forward(self, context, parameters):
        return parameters[:, 5] + 0.25 * context[:, 0]


def test_ratio_head_supports_full_normalized_parameter_vector():
    head = RatioHead(context_dim=7, hidden_dims=(11, 5), parameter_dim=9)
    logits = head(torch.zeros(4, 7), torch.zeros(4, 9))
    assert logits.shape == (4,)
    with pytest.raises(ValueError, match="parameters"):
        head(torch.zeros(4, 7), torch.zeros(4, 2))


def test_nre_negative_examples_shuffle_complete_nine_vector():
    parameters = torch.arange(54, dtype=torch.float32).reshape(6, 9)
    shuffled = shuffle_parameters(
        parameters,
        generator=torch.Generator().manual_seed(4),
    )
    assert shuffled.shape == parameters.shape
    assert not torch.equal(shuffled, parameters)
    assert {
        tuple(row.tolist()) for row in shuffled
    } == {tuple(row.tolist()) for row in parameters}
    context = torch.zeros(6, 3)
    logits, labels = nre_pair_logits_and_labels(
        LinearNineDHead(), context, parameters
    )
    assert logits.shape == (12,)
    assert labels[:6].eq(1.0).all()
    assert labels[6:].eq(0.0).all()
    assert torch.equal(logits[:6], parameters[:, 5])


def test_bounded_density_composes_nre_and_tf_prior_ratio():
    prior = TFPrior(scatter_dex=0.2)
    ranges = canonical_parameter_ranges()
    context = torch.tensor([2.0], dtype=torch.float32)
    theta = np.zeros(9, dtype=np.float64)
    theta[5] = 0.25
    expected_physical = 0.5 * (theta[5] + 1.0) * (540.0 - 60.0) + 60.0
    expected = 0.25 + 0.25 * 2.0 + tf_log_prior_ratio(
        expected_physical, -20.0, prior
    )
    actual = bounded_nre_log_density(
        theta,
        head=LinearNineDHead(),
        context=context,
        rmag_true=-20.0,
        parameter_names=tuple(TARGET_NAMES),
        par_ranges=ranges,
        prior=prior,
    )
    assert actual == pytest.approx(float(expected))


def test_bounded_density_rejects_points_outside_normalized_box():
    theta = np.zeros((2, 9), dtype=np.float64)
    theta[1, 4] = 1.0001
    values = bounded_nre_log_density(
        theta,
        head=LinearNineDHead(),
        context=torch.zeros(1),
        rmag_true=-20.0,
        parameter_names=tuple(TARGET_NAMES),
        par_ranges=canonical_parameter_ranges(),
        prior=TFPrior(),
    )
    assert np.isfinite(values[0])
    assert values[1] == -np.inf


def test_emcee_smoke_and_shear_summary():
    initial = np.random.default_rng(12).uniform(-0.2, 0.2, size=(20, 9))

    def log_density(theta):
        theta = np.asarray(theta)
        if np.any(np.abs(theta) > 1.0):
            return -np.inf
        return float(-0.5 * np.sum(theta**2))

    result = run_emcee(
        log_density,
        initial,
        burnin=2,
        production=6,
        seed=12,
    )
    assert result["chain"].shape == (20 * 6, 9)
    assert result["log_prob"].shape == (20 * 6,)
    assert result["diagnostics"]["burnin"] == 2
    assert 0.0 <= result["diagnostics"]["acceptance_mean"] <= 1.0
    summary = summarize_shear_samples(
        result["chain"],
        result["log_prob"],
        parameter_names=tuple(TARGET_NAMES),
        par_ranges=canonical_parameter_ranges(),
    )
    assert summary["n_samples"] == 120
    assert summary["g1_lower"] <= summary["g1_mean"] <= summary["g1_upper"]
