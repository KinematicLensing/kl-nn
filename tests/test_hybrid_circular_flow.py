"""Contract and numerical tests for the affine/circular hybrid posterior.

The hybrid represents the joint density with the chain-rule factorization

    q(x, theta | context) = q(x | context) q(theta | x, context),

where ``x`` contains every parameter except ``theta_int``.  These regressions
therefore test both the circular topology and the dependency which prevents the
factorization from becoming an independence assumption.
"""

import copy
import pytest
import torch
from torch import nn

import config
import networks
from data import D4_ELEMENTS, transform_d4_parameters
from networks import HybridAffineCircularFlow, KLNPE


FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
THETA_INDEX = FEATURE_NAMES.index("theta_int")
FEATURE_DIM = 1024
CONTEXT_DIM = 7


class TinyFeatureExtractor(nn.Module):
    """Cheap feature fixture satisfying both ordinary and D4 KLNPE APIs."""

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        scalar = image.mean(dim=tuple(range(1, image.ndim)))[:, None]
        return scalar.expand(-1, FEATURE_DIM)

    def transform_features(self, features, element):
        del element
        return features


def _make_flow(*, theta_index=THETA_INDEX, dtype=torch.float64):
    torch.manual_seed(81426)
    return HybridAffineCircularFlow(
        features=len(FEATURE_NAMES),
        theta_index=theta_index,
        context_features=CONTEXT_DIM,
        num_affine_layers=1,
        num_theta_layers=1,
        num_bins=8,
        logit_limit=10.0,
    ).to(dtype=dtype)


def _repeat_context(context, repeats):
    return context[:, None, :].expand(-1, repeats, -1).reshape(-1, context.shape[-1])


def test_hybrid_shapes_support_and_sample_log_prob_consistency():
    flow = _make_flow()
    context = torch.randn(3, CONTEXT_DIM, dtype=torch.float64)

    samples = flow.sample(11, context=context)
    paired_samples, paired_log_prob = flow.sample_and_log_prob(13, context=context)

    assert samples.shape == (3, 11, len(FEATURE_NAMES))
    assert paired_samples.shape == (3, 13, len(FEATURE_NAMES))
    assert paired_log_prob.shape == (3, 13)
    assert torch.isfinite(samples).all()
    assert torch.isfinite(paired_samples).all()
    assert torch.isfinite(paired_log_prob).all()
    assert torch.all(samples[..., THETA_INDEX] >= -1.0)
    assert torch.all(samples[..., THETA_INDEX] < 1.0)
    assert torch.all(paired_samples[..., THETA_INDEX] >= -1.0)
    assert torch.all(paired_samples[..., THETA_INDEX] < 1.0)

    rescored = flow.log_prob(
        paired_samples.reshape(-1, len(FEATURE_NAMES)),
        context=_repeat_context(context, paired_samples.shape[1]),
    ).reshape_as(paired_log_prob)
    torch.testing.assert_close(paired_log_prob, rescored, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("theta_index", (0, 2, 7))
def test_hybrid_reassembles_arbitrary_physical_theta_index(theta_index):
    flow = _make_flow(theta_index=theta_index)
    context = torch.randn(2, CONTEXT_DIM, dtype=torch.float64)
    inputs = torch.randn(2, len(FEATURE_NAMES), dtype=torch.float64) * 0.2
    inputs[:, theta_index] = torch.tensor((-0.75, 0.65), dtype=torch.float64)

    noise = flow.transform_to_noise(inputs, context=context)
    samples = flow.sample(31, context=context)

    assert flow.theta_index == theta_index
    assert noise.shape == inputs.shape
    assert samples.shape == (2, 31, len(FEATURE_NAMES))
    assert torch.isfinite(noise).all()
    assert torch.all(noise[:, theta_index] >= -1.0)
    assert torch.all(noise[:, theta_index] < 1.0)
    assert torch.all(samples[..., theta_index] >= -1.0)
    assert torch.all(samples[..., theta_index] < 1.0)

    at_left = inputs.clone()
    at_right = inputs.clone()
    at_left[:, theta_index] = -1.0
    at_right[:, theta_index] = 1.0
    torch.testing.assert_close(
        flow.log_prob(at_left, context=context),
        flow.log_prob(at_right, context=context),
        atol=2e-8,
        rtol=2e-8,
    )


def test_component_log_prob_factorizes_joint_and_theta_is_normalized():
    flow = _make_flow()
    context = torch.randn(1, CONTEXT_DIM, dtype=torch.float64)
    fixed = torch.tensor(
        [[0.13, -0.21, 0.0, 0.31, -0.17, 0.27, -0.09, 0.18]],
        dtype=torch.float64,
    )

    affine_lp, theta_lp = flow.component_log_prob(fixed, context=context)
    joint_lp = flow.log_prob(fixed, context=context)
    torch.testing.assert_close(joint_lp, affine_lp + theta_lp)

    # Midpoint quadrature avoids evaluating the duplicated +1 seam.  Identity
    # initialization makes this exact up to floating-point summation, while
    # still checking the public conditional-density decomposition.
    count = 2048
    spacing = 2.0 / count
    theta = -1.0 + (torch.arange(count, dtype=torch.float64) + 0.5) * spacing
    grid = fixed.expand(count, -1).clone()
    grid[:, THETA_INDEX] = theta
    grid_context = context.expand(count, -1)
    affine_grid, theta_grid = flow.component_log_prob(grid, context=grid_context)

    torch.testing.assert_close(
        affine_grid,
        affine_lp.expand_as(affine_grid),
        atol=2e-8,
        rtol=2e-8,
    )
    integral = torch.exp(theta_grid).sum() * spacing
    torch.testing.assert_close(
        integral,
        torch.ones((), dtype=integral.dtype),
        atol=2e-4,
        rtol=2e-4,
    )


def test_hybrid_seam_is_continuous_and_canonicalized():
    flow = _make_flow()
    context = torch.randn(4, CONTEXT_DIM, dtype=torch.float64)
    inputs = torch.randn(4, len(FEATURE_NAMES), dtype=torch.float64) * 0.15
    eps = 1e-7

    left = inputs.clone()
    right = inputs.clone()
    duplicate = inputs.clone()
    left[:, THETA_INDEX] = -1.0 + eps
    right[:, THETA_INDEX] = 1.0 - eps
    duplicate[:, THETA_INDEX] = 1.0
    canonical = inputs.clone()
    canonical[:, THETA_INDEX] = -1.0

    torch.testing.assert_close(
        flow.log_prob(canonical, context=context),
        flow.log_prob(duplicate, context=context),
        atol=2e-8,
        rtol=2e-8,
    )
    torch.testing.assert_close(
        flow.log_prob(left, context=context),
        flow.log_prob(right, context=context),
        atol=2e-5,
        rtol=2e-5,
    )


def test_float32_learned_theta_spline_is_finite_and_roundtrips():
    """Exercise non-identity spline numerics in the training dtype."""
    torch.manual_seed(19)
    flow = _make_flow(dtype=torch.float32)
    final_layer = flow.theta_transform.conditioners[0][-1]
    with torch.no_grad():
        final_layer.weight.normal_(0.0, 0.02)
        final_layer.bias.uniform_(-3.0, 3.0)

    count = 512
    inputs = torch.randn(count, 8, dtype=torch.float32) * 0.2
    inputs[:, THETA_INDEX] = torch.linspace(
        -1.0 + 1e-5,
        1.0 - 1e-5,
        count,
        dtype=torch.float32,
    )
    context = torch.randn(count, CONTEXT_DIM, dtype=torch.float32)
    non_theta = inputs[:, list(flow.non_theta_indices)]
    condition = torch.cat((context, non_theta), dim=-1)

    latent, forward_logdet = flow.theta_transform(
        inputs[:, THETA_INDEX], condition
    )
    restored, inverse_logdet = flow.theta_transform.inverse(latent, condition)
    circular_residual = torch.remainder(
        restored - inputs[:, THETA_INDEX] + 1.0, 2.0
    ) - 1.0

    assert torch.isfinite(latent).all()
    assert torch.isfinite(restored).all()
    assert torch.isfinite(forward_logdet).all()
    assert torch.isfinite(inverse_logdet).all()
    assert circular_residual.abs().max() < 1e-4
    assert (forward_logdet + inverse_logdet).abs().max() < 2e-3

    loss = -flow.log_prob(inputs, context=context).mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in flow.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_hybrid_sample_score_roundtrip_and_physical_order_latent():
    flow = _make_flow()
    context = torch.randn(2, CONTEXT_DIM, dtype=torch.float64)
    samples, sample_lp = flow.sample_and_log_prob(9, context=context)
    flat = samples.reshape(-1, len(FEATURE_NAMES))
    flat_context = _repeat_context(context, samples.shape[1])

    noise = flow.transform_to_noise(flat, context=flat_context)
    rescored = flow.log_prob(flat, context=flat_context).reshape_as(sample_lp)

    assert noise.shape == flat.shape
    assert torch.isfinite(noise).all()
    assert torch.all(noise[:, THETA_INDEX] >= -1.0)
    assert torch.all(noise[:, THETA_INDEX] < 1.0)
    torch.testing.assert_close(sample_lp, rescored, atol=2e-8, rtol=2e-8)


def test_hybrid_log_prob_has_finite_input_context_and_parameter_gradients():
    flow = _make_flow()
    inputs = (torch.randn(6, len(FEATURE_NAMES), dtype=torch.float64) * 0.2).requires_grad_()
    inputs.data[:, THETA_INDEX] = torch.linspace(-0.8, 0.8, 6, dtype=torch.float64)
    context = torch.randn(6, CONTEXT_DIM, dtype=torch.float64, requires_grad=True)

    loss = -flow.log_prob(inputs, context=context).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    parameter_grads = [
        parameter.grad
        for parameter in flow.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_grads
    assert all(torch.isfinite(gradient).all() for gradient in parameter_grads)
    assert sum(float(gradient.square().sum()) for gradient in parameter_grads) > 0.0
    diagnostics = {
        **flow.last_component_diagnostics,
        **flow.theta_transform.last_diagnostics,
    }
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert diagnostics["theta_bounded_logit_abs_max"] <= flow.theta_transform.logit_limit
    assert diagnostics["theta_derivative_min"] >= flow.theta_transform.min_derivative


def test_theta_conditional_can_depend_on_non_theta_draw():
    """Prove the hybrid is conditional, rather than a product of marginals."""
    flow = _make_flow()
    context = torch.randn(1, CONTEXT_DIM, dtype=torch.float64)
    optimizer = torch.optim.Adam(flow.parameters(), lr=3e-3)

    x_a = torch.tensor(
        [[-0.7, 0.2, -0.55, 0.1, -0.2, 0.3, -0.4, 0.5]],
        dtype=torch.float64,
    )
    x_b = torch.tensor(
        [[0.65, -0.3, 0.55, -0.4, 0.5, -0.6, 0.7, -0.2]],
        dtype=torch.float64,
    )
    a_other_theta = x_a.clone()
    b_other_theta = x_b.clone()
    a_other_theta[:, THETA_INDEX] = x_b[:, THETA_INDEX]
    b_other_theta[:, THETA_INDEX] = x_a[:, THETA_INDEX]

    def interaction():
        _, aa = flow.component_log_prob(x_a, context=context)
        _, bb = flow.component_log_prob(x_b, context=context)
        _, ab = flow.component_log_prob(a_other_theta, context=context)
        _, ba = flow.component_log_prob(b_other_theta, context=context)
        return aa + bb - ab - ba

    # Identity initialization may initially make q(theta | x, c) uniform.  A
    # few direct updates verify that the computational graph can learn an
    # x-dependent conditional rather than demanding accidental random coupling.
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        objective = interaction().sum()
        (-objective).backward()
        optimizer.step()

    learned_interaction = interaction().detach().abs().item()
    assert learned_interaction > 1e-4


def test_hybrid_klnpe_selection_and_d4_theta_seam_compatibility():
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        configured = copy.deepcopy(original)
        configured.flow.flow_type = "hybrid_circular"
        configured.flow.num_layers = 1
        configured.flow.num_bins = 8
        configured.train.feature_number = len(FEATURE_NAMES)
        configured.train.feature_names = list(FEATURE_NAMES)
        configured.train.posterior_symmetry = "d4"
        config.set_model_config(configured)

        model = KLNPE(
            feature_extractor=TinyFeatureExtractor(),
            mode=1,
            batch_size=2,
            nfeatures=len(FEATURE_NAMES),
            nspec=5,
            posterior_symmetry="d4",
        ).double().eval()
        assert isinstance(model.flow, HybridAffineCircularFlow)
        assert model.flow.theta_index == THETA_INDEX

        raw_features = torch.randn(2, FEATURE_DIM, dtype=torch.float64)
        parameters = torch.tensor(
            [
                [0.1, -0.2, -1.0, 0.3, -0.4, 0.5, -0.6, 0.7],
                [0.1, -0.2, 1.0, 0.3, -0.4, 0.5, -0.6, 0.7],
            ],
            dtype=torch.float64,
        )
        density = model._d4_mixture_log_prob_from_features(
            raw_features[:1].expand(2, -1), parameters
        )
        torch.testing.assert_close(density[0], density[1], atol=2e-8, rtol=2e-8)

        # Each D4-transformed parameter vector remains evaluable by the same
        # hybrid density, including directed theta transformations at the seam.
        orbit = torch.cat(
            [
                transform_d4_parameters(
                    parameters[:1], element, feature_names=FEATURE_NAMES
                )
                for element in D4_ELEMENTS
            ],
            dim=0,
        )
        orbit_context = torch.randn(len(D4_ELEMENTS), FEATURE_DIM, dtype=torch.float64)
        orbit_lp = model.flow.log_prob(orbit, context=orbit_context)
        assert orbit_lp.shape == (len(D4_ELEMENTS),)
        assert torch.isfinite(orbit_lp).all()
    finally:
        config.set_model_config(original)


def test_tf_weights_are_normalized_over_global_ddp_batch(monkeypatch):
    model = object.__new__(KLNPE)
    nn.Module.__init__(model)
    model.vcirc_idx = 5
    model.vcirc_min = 60.0
    model.vcirc_max = 540.0
    model.vcirc_jac = 240.0
    model._get_tf_prior_params = lambda mag, snr: (
        torch.full_like(mag, 200.0),
        torch.full_like(mag, 0.2),
    )
    local_log_weights = torch.tensor([0.0, -1.0], dtype=torch.float64)
    remote_log_weights = torch.tensor([-0.5, -2.0], dtype=torch.float64)

    class FakePrior:
        def __init__(self, loc, scale):
            del loc, scale

        def log_prob(self, values):
            return local_log_weights.to(values)

    monkeypatch.setattr(torch.distributions, "LogNormal", FakePrior)
    monkeypatch.setattr(networks.dist, "is_available", lambda: True)
    monkeypatch.setattr(networks.dist, "is_initialized", lambda: True)

    remote_scaled = torch.exp(remote_log_weights)

    def fake_all_reduce(tensor, op):
        if op == networks.dist.ReduceOp.MAX:
            # The local maximum is already the global maximum.
            return
        assert op == networks.dist.ReduceOp.SUM
        tensor.add_(
            tensor.new_tensor(
                [
                    remote_scaled.sum(),
                    remote_scaled.square().sum(),
                    float(remote_scaled.numel()),
                ]
            )
        )

    monkeypatch.setattr(networks.dist, "all_reduce", fake_all_reduce)
    true = torch.zeros(2, len(FEATURE_NAMES), dtype=torch.float64)
    mag = torch.tensor([20.0, 21.0], dtype=torch.float64)
    snr = torch.tensor([50.0, 25.0], dtype=torch.float64)

    weights = model._compute_tf_weights(true, mag, snr)

    local_scaled = torch.exp(local_log_weights)
    global_sum = local_scaled.sum() + remote_scaled.sum()
    global_sum_sq = local_scaled.square().sum() + remote_scaled.square().sum()
    global_mean = global_sum / 4.0
    torch.testing.assert_close(weights, local_scaled / global_mean)
    diagnostics = model.last_tf_diagnostics
    torch.testing.assert_close(
        diagnostics["effective_sample_size"],
        global_sum.square() / global_sum_sq,
    )
    torch.testing.assert_close(
        diagnostics["effective_sample_fraction"],
        global_sum.square() / global_sum_sq / 4.0,
    )
    torch.testing.assert_close(
        diagnostics["max_normalized_weight"],
        torch.tensor(1.0, dtype=torch.float64) / global_mean,
    )
