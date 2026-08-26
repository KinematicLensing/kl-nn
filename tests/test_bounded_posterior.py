"""Numerical and integration contracts for the sole production posterior."""

import math

import pytest
import torch
from torch import nn

from networks import (
    BoundedHybridCircularFlow,
    ConditionalUnitBox,
    IdentityBoundedRationalQuadraticAutoregressiveTransform,
    KLNPE,
    ORACLE_CONTEXT_FIELDS,
)


FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
    "halpha_flux_true",
)
THETA_INDEX = FEATURE_NAMES.index("theta_int")
NFEATURES = len(FEATURE_NAMES)
DIRECT_CONTEXT_DIM = 7
MODEL_CONTEXT_DIM = 1024 + len(ORACLE_CONTEXT_FIELDS)


def _make_flow(
    *,
    dtype=torch.float64,
    theta_index=THETA_INDEX,
    context_features=DIRECT_CONTEXT_DIM,
    num_bounded_layers=1,
    hidden_features=32,
    theta_hidden_features=32,
    seed=81726,
):
    torch.manual_seed(seed)
    return BoundedHybridCircularFlow(
        features=NFEATURES,
        theta_index=theta_index,
        context_features=context_features,
        num_bounded_layers=num_bounded_layers,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=hidden_features,
        theta_hidden_features=theta_hidden_features,
        theta_logit_limit=10.0,
        bounded_logit_limit=10.0,
    ).to(dtype=dtype)


def _repeat_context(context, repeats):
    return context[:, None, :].expand(-1, repeats, -1).reshape(
        -1, context.shape[-1]
    )


def _non_theta_indices(theta_index=THETA_INDEX):
    return [index for index in range(NFEATURES) if index != theta_index]


def _perturb_splines(flow, scale=0.02):
    with torch.no_grad():
        for transform in flow.bounded_transforms:
            final = transform.autoregressive_net.final_layer
            final.weight.normal_(0.0, scale)
            final.bias.add_(
                torch.empty_like(final.bias).uniform_(-0.25, 0.25)
            )
        for conditioner in flow.theta_transform.conditioners:
            conditioner[-1].weight.normal_(0.0, scale)
            conditioner[-1].bias.add_(
                torch.empty_like(conditioner[-1].bias).uniform_(-0.25, 0.25)
            )


class TinyFeatureExtractor(nn.Module):
    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        scalar = image.mean(dim=tuple(range(1, image.ndim)))[:, None]
        return scalar.expand(-1, 1024)


def _observations(batch_size=1, dtype=torch.float64):
    image = torch.randn(batch_size, 1, 3, 3, dtype=dtype)
    spectra = torch.randn(batch_size, 1, 5, 64, dtype=dtype)
    positions = torch.randn(batch_size, 5, 2, dtype=dtype)
    context = {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size, dtype=dtype),
        "image_snr": torch.linspace(
            5.0, 1000.0, batch_size, dtype=dtype
        ),
        "central_halpha_snr": torch.linspace(
            1.0, 200.0, batch_size, dtype=dtype
        ),
    }
    return image, spectra, positions, context


def test_conditional_unit_box_shapes_density_and_support():
    base = ConditionalUnitBox(features=3).double()
    context = torch.randn(2, 5, dtype=torch.float64)
    samples, sample_log_prob = base.sample_and_log_prob(19, context=context)

    assert samples.shape == (2, 19, 3)
    assert sample_log_prob.shape == (2, 19)
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    torch.testing.assert_close(sample_log_prob, torch.zeros_like(sample_log_prob))

    probes = torch.tensor(
        [
            [0.0, 0.5, 1.0],
            [-1e-8, 0.5, 0.5],
            [0.5, 1.0 + 1e-8, 0.5],
            [0.5, float("nan"), 0.5],
        ],
        dtype=torch.float64,
    )
    log_prob = base.log_prob(probes, context=torch.zeros(4, 5).double())
    assert log_prob[0] == 0.0
    assert torch.isneginf(log_prob[1:]).all()


def test_bounded_transform_is_identity_on_closed_unit_box():
    transform = IdentityBoundedRationalQuadraticAutoregressiveTransform(
        features=3,
        hidden_features=16,
        context_features=4,
        num_bins=8,
        num_blocks=1,
        logit_limit=10.0,
    ).double()
    inputs = torch.tensor(
        [[0.0, 0.3, 1.0], [1.0, 0.0, 0.55]], dtype=torch.float64
    )
    context = torch.randn(2, 4, dtype=torch.float64)
    outputs, forward_logdet = transform(inputs, context=context)
    restored, inverse_logdet = transform.inverse(outputs, context=context)

    torch.testing.assert_close(outputs, inputs, atol=2e-10, rtol=0.0)
    torch.testing.assert_close(restored, inputs, atol=2e-10, rtol=0.0)
    torch.testing.assert_close(
        forward_logdet, torch.zeros_like(forward_logdet), atol=1e-8, rtol=0.0
    )
    torch.testing.assert_close(
        inverse_logdet, torch.zeros_like(inverse_logdet), atol=1e-8, rtol=0.0
    )


def test_identity_posterior_is_uniform_on_nine_cube_and_circle():
    flow = _make_flow()
    context = torch.randn(6, DIRECT_CONTEXT_DIM, dtype=torch.float64)
    inputs = torch.empty(6, NFEATURES, dtype=torch.float64).uniform_(-0.9, 0.9)
    inputs[:, THETA_INDEX] = torch.linspace(-0.9, 0.9, 6)

    non_theta_lp, theta_lp = flow.component_log_prob(inputs, context=context)
    joint_lp = flow.log_prob(inputs, context=context)
    torch.testing.assert_close(
        non_theta_lp,
        torch.full_like(non_theta_lp, -(NFEATURES - 1) * math.log(2.0)),
        atol=5e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(
        theta_lp,
        torch.full_like(theta_lp, -math.log(2.0)),
        atol=5e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(joint_lp, non_theta_lp + theta_lp)


def test_posterior_accepts_endpoints_rejects_outside_and_wraps_theta():
    flow = _make_flow()
    context = torch.randn(4, DIRECT_CONTEXT_DIM, dtype=torch.float64)
    endpoints = torch.zeros(4, NFEATURES, dtype=torch.float64)
    endpoints[:, _non_theta_indices()] = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        dtype=torch.float64,
    )
    endpoints[:, THETA_INDEX] = torch.tensor([-1.0, 1.0, 3.0, -3.0])
    endpoint_lp = flow.log_prob(endpoints, context=context)
    assert torch.isfinite(endpoint_lp).all()
    torch.testing.assert_close(endpoint_lp, endpoint_lp[:1].expand_as(endpoint_lp))

    outside = endpoints[:1].expand(NFEATURES - 1, -1).clone()
    for row, index in enumerate(_non_theta_indices()):
        outside[row, index] = 1.0 + 1e-6
    outside_lp = flow.log_prob(
        outside, context=context[:1].expand(NFEATURES - 1, -1)
    )
    assert torch.isneginf(outside_lp).all()


@pytest.mark.parametrize("theta_index", (0, 2, 8))
def test_samples_support_arbitrary_theta_index(theta_index):
    flow = _make_flow(theta_index=theta_index)
    context = torch.randn(2, DIRECT_CONTEXT_DIM, dtype=torch.float64)
    samples, log_prob = flow.sample_and_log_prob(37, context=context)
    rescored = flow.log_prob(
        samples.reshape(-1, NFEATURES),
        context=_repeat_context(context, samples.shape[1]),
    ).reshape_as(log_prob)

    assert samples.shape == (2, 37, NFEATURES)
    assert torch.isfinite(samples).all()
    assert torch.all(samples[..., _non_theta_indices(theta_index)] >= -1.0)
    assert torch.all(samples[..., _non_theta_indices(theta_index)] <= 1.0)
    assert torch.all(samples[..., theta_index] >= -1.0)
    assert torch.all(samples[..., theta_index] < 1.0)
    torch.testing.assert_close(log_prob, rescored, atol=2e-8, rtol=2e-8)


def test_float32_learned_posterior_roundtrips_rescores_and_backpropagates():
    flow = _make_flow(dtype=torch.float32, num_bounded_layers=2)
    _perturb_splines(flow)
    context = torch.randn(5, DIRECT_CONTEXT_DIM)
    inputs = torch.empty(5, NFEATURES).uniform_(-0.8, 0.8).requires_grad_()

    latent = flow.transform_to_noise(inputs, context=context)
    samples, sample_lp = flow.sample_and_log_prob(11, context=context)
    rescored = flow.log_prob(
        samples.reshape(-1, NFEATURES),
        context=_repeat_context(context, samples.shape[1]),
    ).reshape_as(sample_lp)
    loss = -flow.log_prob(inputs, context=context).mean()
    loss.backward()

    assert torch.isfinite(latent).all()
    assert torch.isfinite(samples).all()
    assert torch.isfinite(sample_lp).all()
    torch.testing.assert_close(sample_lp, rescored, atol=3e-4, rtol=3e-4)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    gradients = [
        parameter.grad
        for parameter in flow.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(item).all() for item in gradients)


def test_four_layer_extreme_conditioner_smoke_stays_in_support():
    flow = _make_flow(
        dtype=torch.float32,
        num_bounded_layers=4,
        hidden_features=16,
        theta_hidden_features=16,
    )
    with torch.no_grad():
        for transform in flow.bounded_transforms:
            transform.autoregressive_net.final_layer.weight.fill_(50.0)
            transform.autoregressive_net.final_layer.bias.fill_(-50.0)
        for conditioner in flow.theta_transform.conditioners:
            conditioner[-1].weight.fill_(50.0)
            conditioner[-1].bias.fill_(-50.0)
    context = torch.randn(2, DIRECT_CONTEXT_DIM)
    samples = flow.sample(16, context=context)

    assert torch.isfinite(samples).all()
    assert torch.all(samples[..., _non_theta_indices()] >= -1.0)
    assert torch.all(samples[..., _non_theta_indices()] <= 1.0)
    assert torch.all(samples[..., THETA_INDEX] >= -1.0)
    assert torch.all(samples[..., THETA_INDEX] < 1.0)


def test_theta_factor_is_conditional_on_non_theta_and_continuous_at_seam():
    flow = _make_flow()
    context = torch.randn(1, DIRECT_CONTEXT_DIM, dtype=torch.float64)
    optimizer = torch.optim.Adam(flow.theta_transform.parameters(), lr=3e-3)
    first = torch.tensor(
        [[-0.7, 0.2, -0.55, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6]],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [[0.65, -0.3, 0.55, -0.4, 0.5, -0.6, 0.7, -0.2, 0.4]],
        dtype=torch.float64,
    )
    cross_first = first.clone()
    cross_second = second.clone()
    cross_first[:, THETA_INDEX] = second[:, THETA_INDEX]
    cross_second[:, THETA_INDEX] = first[:, THETA_INDEX]

    def interaction():
        _, aa = flow.component_log_prob(first, context=context)
        _, bb = flow.component_log_prob(second, context=context)
        _, ab = flow.component_log_prob(cross_first, context=context)
        _, ba = flow.component_log_prob(cross_second, context=context)
        return aa + bb - ab - ba

    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        (-interaction().sum()).backward()
        optimizer.step()
    assert interaction().detach().abs().item() > 1e-4

    seam = first.expand(2, -1).clone()
    seam[:, THETA_INDEX] = torch.tensor([-1.0, 1.0])
    torch.testing.assert_close(
        flow.log_prob(seam, context=context.expand(2, -1))[0],
        flow.log_prob(seam, context=context.expand(2, -1))[1],
        atol=2e-8,
        rtol=2e-8,
    )


def test_klnpe_samples_and_scores_single_observation_candidate_bank():
    flow = _make_flow(context_features=MODEL_CONTEXT_DIM)
    model = KLNPE(
        feature_extractor=TinyFeatureExtractor(),
        flow=flow,
        feature_names=FEATURE_NAMES,
    ).double().eval()
    image, spectra, positions, context = _observations(dtype=torch.float64)
    samples, sample_lp = model.sample(
        image,
        spectra,
        23,
        fp=positions,
        observation_context=context,
        return_log_prob=True,
    )
    candidate_lp = model.posterior_log_prob(
        image,
        spectra,
        samples[0],
        positions,
        context,
    )

    assert samples.shape == (1, 23, NFEATURES)
    assert sample_lp.shape == (1, 23)
    assert candidate_lp.shape == (23,)
    torch.testing.assert_close(candidate_lp, sample_lp[0], atol=2e-8, rtol=2e-8)


def test_klnpe_rejects_halpha_truth_as_context_and_supports_batched_banks():
    flow = _make_flow(context_features=MODEL_CONTEXT_DIM)
    model = KLNPE(
        feature_extractor=TinyFeatureExtractor(),
        flow=flow,
        feature_names=FEATURE_NAMES,
    ).double().eval()
    image, spectra, positions, context = _observations(
        batch_size=2, dtype=torch.float64
    )
    candidates = torch.empty(2, 5, NFEATURES, dtype=torch.float64).uniform_(
        -0.8, 0.8
    )
    scores = model.posterior_log_prob(
        image, spectra, candidates, positions, context
    )
    assert scores.shape == (2, 5)

    leaked = dict(context)
    leaked["halpha_flux_true"] = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="posterior targets.*halpha_flux_true"):
        model.posterior_log_prob(
            image,
            spectra,
            candidates,
            positions,
            leaked,
        )
