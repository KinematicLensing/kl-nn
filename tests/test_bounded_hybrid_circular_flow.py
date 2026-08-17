"""Contracts for the compact seven-parameter plus circular-theta posterior.

The bounded hybrid represents

    q(x, theta | context) = q_box(x | context) q_circle(theta | x, context),

with every non-angular normalized parameter on the closed interval ``[-1, 1]``
and directed ``theta_int`` on the half-open circle ``[-1, 1)``.  These tests
deliberately exercise the endpoints: silently clipping an unbounded flow would
make sampling and density evaluation describe different distributions.
"""

import copy
import math

import pytest
import torch
from torch import nn

import config
from data import D4_ELEMENTS, transform_d4_parameters
from networks import (
    BoundedHybridCircularFlow,
    ConditionalUnitBox,
    HybridAffineCircularFlow,
    IdentityBoundedRationalQuadraticAutoregressiveTransform,
    KLNPE,
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
)
THETA_INDEX = FEATURE_NAMES.index("theta_int")
NFEATURES = len(FEATURE_NAMES)
CONTEXT_DIM = 7
FEATURE_DIM = 1024


class TinyFeatureExtractor(nn.Module):
    """Cheap feature fixture satisfying ordinary and D4 KLNPE APIs."""

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        scalar = image.mean(dim=tuple(range(1, image.ndim)))[:, None]
        return scalar.expand(-1, FEATURE_DIM)

    def transform_features(self, features, element):
        del element
        return features


class PresetFlow(nn.Module):
    """Return a prescribed sample tensor from KLNPE's sampling guard."""

    def __init__(self, template):
        super().__init__()
        self.register_buffer("template", torch.as_tensor(template))

    def sample(self, num_samples, context):
        return self.template.to(context).expand(
            context.shape[0], num_samples, -1
        ).clone()


def _make_flow(
    *,
    dtype=torch.float64,
    theta_index=THETA_INDEX,
    num_bounded_layers=1,
    hidden_features=32,
    theta_hidden_features=32,
    seed=81726,
):
    torch.manual_seed(seed)
    return BoundedHybridCircularFlow(
        features=NFEATURES,
        theta_index=theta_index,
        context_features=CONTEXT_DIM,
        num_bounded_layers=num_bounded_layers,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=hidden_features,
        theta_hidden_features=theta_hidden_features,
        logit_limit=10.0,
        bounded_logit_limit=10.0,
    ).to(dtype=dtype)


def _repeat_context(context, repeats):
    return context[:, None, :].expand(-1, repeats, -1).reshape(
        -1, context.shape[-1]
    )


def _non_theta_indices(theta_index=THETA_INDEX):
    return [index for index in range(NFEATURES) if index != theta_index]


def _perturb_spline_outputs(flow, scale=0.02):
    """Move both factors away from identity without making sharp splines."""
    with torch.no_grad():
        for transform in flow.affine_flow._transform._transforms:
            autoregressive_net = getattr(transform, "autoregressive_net", None)
            if autoregressive_net is not None:
                autoregressive_net.final_layer.weight.normal_(0.0, scale)
                autoregressive_net.final_layer.bias.add_(
                    torch.empty_like(autoregressive_net.final_layer.bias).uniform_(
                        -0.25, 0.25
                    )
                )
        for conditioner in flow.theta_transform.conditioners:
            conditioner[-1].weight.normal_(0.0, scale)
            conditioner[-1].bias.add_(
                torch.empty_like(conditioner[-1].bias).uniform_(-0.25, 0.25)
            )


def _sampling_guard(flow_type, sample):
    model = KLNPE.__new__(KLNPE)
    nn.Module.__init__(model)
    model.nfeatures = NFEATURES
    model.feature_names = FEATURE_NAMES
    model.theta_idx = THETA_INDEX
    model.flow_type = flow_type
    model.flow = PresetFlow(sample)
    return model


def test_conditional_unit_box_shapes_log_prob_and_support():
    base = ConditionalUnitBox(features=3).double()
    context = torch.randn(2, 5, dtype=torch.float64)

    samples, sample_log_prob = base.sample_and_log_prob(19, context=context)

    assert samples.shape == (2, 19, 3)
    assert sample_log_prob.shape == (2, 19)
    assert samples.dtype == context.dtype
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
    probe_context = torch.zeros(4, 5, dtype=torch.float64)
    log_prob = base.log_prob(probes, context=probe_context)

    assert log_prob[0] == 0.0
    assert torch.isneginf(log_prob[1:]).all()


def test_identity_bounded_spline_is_identity_on_closed_unit_box():
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


def test_bounded_hybrid_identity_density_is_uniform_on_physical_cube():
    flow = _make_flow()
    context = torch.randn(6, CONTEXT_DIM, dtype=torch.float64)
    inputs = torch.empty(6, NFEATURES, dtype=torch.float64).uniform_(-0.9, 0.9)
    inputs[:, THETA_INDEX] = torch.linspace(-0.9, 0.9, 6)

    bounded_lp, theta_lp = flow.component_log_prob(inputs, context=context)
    joint_lp = flow.log_prob(inputs, context=context)

    torch.testing.assert_close(
        bounded_lp,
        torch.full_like(bounded_lp, -(NFEATURES - 1) * math.log(2.0)),
        atol=5e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(
        theta_lp,
        torch.full_like(theta_lp, -math.log(2.0)),
        atol=5e-9,
        rtol=0.0,
    )
    torch.testing.assert_close(joint_lp, bounded_lp + theta_lp)
    torch.testing.assert_close(
        joint_lp,
        torch.full_like(joint_lp, -NFEATURES * math.log(2.0)),
        atol=5e-9,
        rtol=0.0,
    )


def test_bounded_hybrid_accepts_exact_endpoints_and_rejects_outside():
    flow = _make_flow()
    non_theta = _non_theta_indices()
    context = torch.randn(4, CONTEXT_DIM, dtype=torch.float64)
    endpoints = torch.zeros(4, NFEATURES, dtype=torch.float64)
    endpoints[:, non_theta] = torch.tensor(
        [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        dtype=torch.float64,
    )
    endpoints[:, THETA_INDEX] = torch.tensor([-1.0, 1.0, -0.2, 0.7])

    endpoint_lp = flow.log_prob(endpoints, context=context)
    assert torch.isfinite(endpoint_lp).all()
    torch.testing.assert_close(endpoint_lp[0], endpoint_lp[1])

    outside = endpoints[:1].expand(len(non_theta), -1).clone()
    for row, feature_index in enumerate(non_theta):
        outside[row, feature_index] = 1.0 + 1e-6
    outside_lp = flow.log_prob(
        outside, context=context[:1].expand(len(non_theta), -1)
    )
    assert torch.isneginf(outside_lp).all()


@pytest.mark.parametrize("theta_index", (0, 2, 7))
def test_bounded_hybrid_samples_every_non_theta_dimension_in_support(theta_index):
    flow = _make_flow(theta_index=theta_index)
    context = torch.randn(3, CONTEXT_DIM, dtype=torch.float64)

    samples = flow.sample(257, context=context)

    assert samples.shape == (3, 257, NFEATURES)
    assert torch.isfinite(samples).all()
    non_theta = _non_theta_indices(theta_index)
    assert torch.all(samples[..., non_theta] >= -1.0)
    assert torch.all(samples[..., non_theta] <= 1.0)
    assert torch.all(samples[..., theta_index] >= -1.0)
    assert torch.all(samples[..., theta_index] < 1.0)


def test_float32_learned_bounded_flow_roundtrips_rescores_and_backpropagates():
    torch.manual_seed(1708)
    flow = _make_flow(dtype=torch.float32)
    _perturb_spline_outputs(flow)
    context = torch.randn(5, CONTEXT_DIM, dtype=torch.float32)

    samples, paired_log_prob = flow.sample_and_log_prob(23, context=context)
    flat = samples.reshape(-1, NFEATURES)
    flat_context = _repeat_context(context, samples.shape[1])
    rescored = flow.log_prob(flat, context=flat_context).reshape_as(
        paired_log_prob
    )

    assert torch.isfinite(samples).all()
    assert torch.isfinite(paired_log_prob).all()
    torch.testing.assert_close(paired_log_prob, rescored, atol=5e-4, rtol=5e-4)

    physical = torch.empty(64, NFEATURES, dtype=torch.float32).uniform_(-0.9, 0.9)
    physical[:, THETA_INDEX] = torch.linspace(-0.95, 0.95, 64)
    physical_context = torch.randn(64, CONTEXT_DIM, dtype=torch.float32)
    bounded = physical[:, _non_theta_indices()]
    latent, forward_logdet = flow.affine_flow._transform(
        bounded, context=physical_context
    )
    restored, inverse_logdet = flow.affine_flow._transform.inverse(
        latent, context=physical_context
    )

    assert torch.isfinite(latent).all()
    assert torch.isfinite(forward_logdet).all()
    assert torch.isfinite(inverse_logdet).all()
    torch.testing.assert_close(restored, bounded, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        atol=3e-3,
        rtol=0.0,
    )

    train_inputs = physical.detach().requires_grad_()
    train_context = physical_context.detach().requires_grad_()
    loss = -flow.log_prob(train_inputs, context=train_context).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert train_inputs.grad is not None and torch.isfinite(train_inputs.grad).all()
    assert train_context.grad is not None and torch.isfinite(train_context.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in flow.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert sum(float(gradient.square().sum()) for gradient in parameter_gradients) > 0.0


def _assert_finite_in_support_samples(flow, context, num_samples):
    samples, sample_log_prob = flow.sample_and_log_prob(
        num_samples, context=context
    )
    flat_samples = samples.reshape(-1, NFEATURES)
    flat_context = _repeat_context(context, samples.shape[1])
    rescored = flow.log_prob(
        flat_samples, context=flat_context
    ).reshape_as(sample_log_prob)

    non_theta = _non_theta_indices()
    assert samples.shape == (context.shape[0], num_samples, NFEATURES)
    assert torch.isfinite(samples).all()
    assert torch.all(samples[..., non_theta] >= -1.0)
    assert torch.all(samples[..., non_theta] <= 1.0)
    assert torch.all(samples[..., THETA_INDEX] >= -1.0)
    assert torch.all(samples[..., THETA_INDEX] < 1.0)
    assert torch.isfinite(sample_log_prob).all()
    assert torch.isfinite(rescored).all()
    torch.testing.assert_close(sample_log_prob, rescored)


def test_float32_four_layer_saturated_splines_sample_inside_support():
    """Regress the auditor's float32 compact-RQS inverse failure exactly."""
    flow = _make_flow(
        dtype=torch.float32,
        num_bounded_layers=4,
        hidden_features=16,
        theta_hidden_features=16,
        seed=3,
    ).eval()
    assert len(flow.bounded_transforms) == 4
    with torch.no_grad():
        for transform in flow.bounded_transforms:
            final = transform.autoregressive_net.final_layer
            final.weight.zero_()
            final.bias.normal_(mean=0.0, std=5.0)

    context = torch.randn(1, CONTEXT_DIM, dtype=torch.float32)
    _assert_finite_in_support_samples(flow, context, num_samples=1000)


def test_float32_four_layer_extreme_conditioner_smoke_stays_in_support():
    """Exercise saturated weights, logits, and large contexts together."""
    flow = _make_flow(
        dtype=torch.float32,
        num_bounded_layers=4,
        hidden_features=32,
        seed=1,
    ).eval()
    with torch.no_grad():
        for transform in flow.bounded_transforms:
            final = transform.autoregressive_net.final_layer
            final.weight.normal_(mean=0.0, std=50.0)
            final.bias.normal_(mean=0.0, std=50.0)

    context = 10.0 * torch.randn(2, CONTEXT_DIM, dtype=torch.float32)
    _assert_finite_in_support_samples(flow, context, num_samples=500)


def test_bounded_theta_is_conditional_on_non_theta_parameters_and_has_a_seam():
    flow = _make_flow()
    context = torch.randn(1, CONTEXT_DIM, dtype=torch.float64)
    optimizer = torch.optim.Adam(flow.theta_transform.parameters(), lr=3e-3)
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

    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        (-interaction().sum()).backward()
        optimizer.step()
    assert interaction().detach().abs().item() > 1e-4

    left = x_a.expand(2, -1).clone()
    right = left.clone()
    left[:, THETA_INDEX] = -1.0
    right[:, THETA_INDEX] = 1.0
    seam_context = context.expand(2, -1)
    torch.testing.assert_close(
        flow.log_prob(left, context=seam_context),
        flow.log_prob(right, context=seam_context),
        atol=2e-8,
        rtol=2e-8,
    )


def test_bounded_hybrid_klnpe_integrates_with_d4_density_and_sampling():
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        configured = copy.deepcopy(original)
        configured.flow.flow_type = "bounded_hybrid_circular"
        configured.flow.num_layers = 1
        configured.flow.num_bins = 8
        configured.flow.bounded_logit_limit = 10.0
        configured.train.feature_number = NFEATURES
        configured.train.feature_names = list(FEATURE_NAMES)
        configured.train.posterior_symmetry = "d4"
        config.set_model_config(configured)

        model = KLNPE(
            feature_extractor=TinyFeatureExtractor(),
            mode=1,
            batch_size=1,
            nfeatures=NFEATURES,
            nspec=5,
            posterior_symmetry="d4",
        ).double().eval()
        assert isinstance(model.flow, BoundedHybridCircularFlow)
        # The historical name remains an intentional optimizer-group alias.
        assert model.flow.affine_flow is not None

        raw_features = torch.randn(1, FEATURE_DIM, dtype=torch.float64)
        parameters = torch.tensor(
            [[0.1, -0.2, 1.0, 0.3, -0.4, 0.5, -0.6, 0.7]],
            dtype=torch.float64,
        )
        orbit = torch.cat(
            [
                transform_d4_parameters(
                    parameters, element, feature_names=FEATURE_NAMES
                )
                for element in D4_ELEMENTS
            ]
        )
        orbit_context = torch.randn(
            len(D4_ELEMENTS), FEATURE_DIM, dtype=torch.float64
        )
        orbit_lp = model.flow.log_prob(orbit, context=orbit_context)
        assert orbit_lp.shape == (len(D4_ELEMENTS),)
        assert torch.isfinite(orbit_lp).all()

        samples = model._d4_sample_from_features(raw_features, 24)
        assert samples.shape == (1, 24, NFEATURES)
        assert torch.isfinite(samples).all()
        assert torch.all(samples[..., _non_theta_indices()] >= -1.0)
        assert torch.all(samples[..., _non_theta_indices()] <= 1.0)
        assert torch.all(samples[..., THETA_INDEX] >= -1.0)
        assert torch.all(samples[..., THETA_INDEX] < 1.0)
    finally:
        config.set_model_config(original)


def test_bounded_sampling_guard_rejects_outside_instead_of_clamping():
    valid = torch.tensor(
        [-1.0, 1.0, 0.4, -0.5, 0.25, -0.75, 0.9, -0.1]
    )
    valid_model = _sampling_guard("bounded_hybrid_circular", valid)
    context = torch.zeros(1, CONTEXT_DIM)
    returned = valid_model._draw_flow_samples(3, context)
    torch.testing.assert_close(returned, valid.expand(1, 3, -1))

    invalid = valid.clone()
    invalid[0] = 1.25
    invalid_model = _sampling_guard("bounded_hybrid_circular", invalid)
    with pytest.raises(RuntimeError, match=r"bounded|support|\[-1, 1\]"):
        invalid_model._draw_flow_samples(3, context)


def test_legacy_affine_and_hybrid_flows_remain_unbounded_and_selectable():
    bounded = _make_flow()
    legacy = HybridAffineCircularFlow(
        features=NFEATURES,
        theta_index=THETA_INDEX,
        context_features=CONTEXT_DIM,
        num_affine_layers=1,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=32,
        theta_hidden_features=32,
    ).double()
    context = torch.randn(2, CONTEXT_DIM, dtype=torch.float64)
    outside = torch.zeros(2, NFEATURES, dtype=torch.float64)
    outside[:, 0] = torch.tensor([1.25, -1.25])

    assert torch.isneginf(bounded.log_prob(outside, context=context)).all()
    assert torch.isfinite(legacy.log_prob(outside, context=context)).all()

    # The pre-existing affine sampling path retains its historical clamp.  The
    # strict support assertion above is specific to the new bounded family.
    affine_model = _sampling_guard("affine", outside[0])
    affine_samples = affine_model._draw_flow_samples(2, context[:1])
    assert affine_samples[..., 0].eq(1.25).all()
