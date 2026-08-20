"""Low-level tests for the periodic spline kernel used by theta_int."""

import math

import torch

from circular_spline import (
    DEFAULT_MIN_DERIVATIVE,
    unconstrained_rational_quadratic_spline,
)


def _parameters(batch, bins=8, *, dtype=torch.float64, seed=17):
    generator = torch.Generator().manual_seed(seed)
    widths = torch.randn(batch, 1, bins, generator=generator, dtype=dtype) * 0.2
    heights = torch.randn(batch, 1, bins, generator=generator, dtype=dtype) * 0.2
    derivatives = (
        torch.randn(batch, 1, bins, generator=generator, dtype=dtype) * 0.2
    )
    return widths, heights, derivatives


def _circular_residual(actual, expected):
    return torch.remainder(actual - expected + 1.0, 2.0) - 1.0


def test_periodic_spline_roundtrip_and_logdet_cancel():
    inputs = torch.linspace(-0.99, 0.99, 41, dtype=torch.float64).unsqueeze(1)
    widths, heights, derivatives = _parameters(inputs.shape[0])
    outputs, forward_logdet = unconstrained_rational_quadratic_spline(
        inputs,
        widths,
        heights,
        derivatives,
        tails="circular",
    )
    restored, inverse_logdet = unconstrained_rational_quadratic_spline(
        outputs,
        widths,
        heights,
        derivatives,
        inverse=True,
        tails="circular",
    )

    torch.testing.assert_close(restored, inputs, atol=1e-9, rtol=1e-9)
    torch.testing.assert_close(
        forward_logdet + inverse_logdet,
        torch.zeros_like(forward_logdet),
        atol=1e-9,
        rtol=1e-9,
    )


def test_periodic_spline_has_one_value_and_jacobian_at_seam():
    inputs = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    widths, heights, derivatives = _parameters(1, seed=23)
    widths = widths.expand(2, -1, -1)
    heights = heights.expand(2, -1, -1)
    derivatives = derivatives.expand(2, -1, -1)
    outputs, logdet = unconstrained_rational_quadratic_spline(
        inputs,
        widths,
        heights,
        derivatives,
        tails="circular",
    )

    assert _circular_residual(outputs[0], outputs[1]).abs().max() < 1e-12
    torch.testing.assert_close(logdet[0], logdet[1], atol=1e-12, rtol=0.0)


def test_identity_parameters_produce_identity_circle_map():
    bins = 8
    inputs = torch.linspace(-1.0, 1.0, 33, dtype=torch.float64).unsqueeze(1)
    widths = torch.zeros(inputs.shape[0], 1, bins, dtype=torch.float64)
    heights = torch.zeros_like(widths)
    derivative_value = math.log(math.expm1(1.0 - DEFAULT_MIN_DERIVATIVE))
    derivatives = torch.full_like(widths, derivative_value)
    outputs, logdet = unconstrained_rational_quadratic_spline(
        inputs,
        widths,
        heights,
        derivatives,
        tails="circular",
    )

    torch.testing.assert_close(outputs, inputs, atol=2e-10, rtol=0.0)
    torch.testing.assert_close(
        logdet, torch.zeros_like(logdet), atol=2e-10, rtol=0.0
    )


def test_periodic_spline_reports_autograd_jacobian():
    point = torch.tensor([[0.17]], dtype=torch.float64, requires_grad=True)
    widths, heights, derivatives = _parameters(1, seed=29)
    output, logdet = unconstrained_rational_quadratic_spline(
        point,
        widths,
        heights,
        derivatives,
        tails="circular",
    )
    derivative = torch.autograd.grad(output.sum(), point)[0]
    torch.testing.assert_close(
        logdet, torch.log(derivative.abs()), atol=1e-10, rtol=1e-10
    )
