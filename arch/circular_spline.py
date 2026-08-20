"""Low-level rational-quadratic spline helpers for periodic ``theta_int``.

The production posterior uses nflows' compact autoregressive spline for every
bounded non-angular parameter. Only the directed angle needs custom spline
arithmetic: its left and right endpoints are the same point and therefore must
share a derivative. Keeping that small numerical kernel here avoids carrying
the retired full-vector circular-flow implementation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs, eps=1e-6):
    """Return the enclosing spline-bin index, including the right endpoint."""
    bin_locations = bin_locations.clone()
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """Evaluate a monotone rational-quadratic spline and its log Jacobian."""
    num_bins = unnormalized_widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0), value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0), value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    locations = cumheights if inverse else cumwidths
    bin_idx = _searchsorted(locations, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.square() - 4 * a * c
        scale = torch.maximum(b.square().abs(), (4 * a * c).abs()).clamp_min(1.0)
        tolerance = 100 * torch.finfo(discriminant.dtype).eps * scale
        if torch.any(discriminant < -tolerance):
            raise ValueError("Spline inversion failed: negative discriminant")
        discriminant = discriminant.clamp_min(0.0)
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) * theta_one_minus_theta
        derivative_numerator = input_delta.square() * (
            input_derivatives_plus_one * root.square()
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).square()
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (
        input_delta * theta.square()
        + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        input_derivatives + input_derivatives_plus_one - 2 * input_delta
    ) * theta_one_minus_theta
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.square() * (
        input_derivatives_plus_one * theta.square()
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).square()
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="circular",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """Evaluate a circular spline on ``[-tail_bound, tail_bound]``.

    Circular derivatives contain one value per bin. Appending the first value
    at the right endpoint makes the value and Jacobian continuous at the seam.
    Callers canonicalize equivalent angle representatives before this kernel.
    """
    if tails != "circular":
        raise ValueError("Only circular spline tails are supported")
    if not isinstance(tail_bound, (int, float)) or float(tail_bound) <= 0:
        raise ValueError("tail_bound must be a positive scalar")

    derivatives = torch.cat(
        (unnormalized_derivatives, unnormalized_derivatives[..., :1]),
        dim=-1,
    )
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    clamped = inputs.clamp(min=-tail_bound, max=tail_bound)
    transformed, logabsdet = rational_quadratic_spline(
        inputs=clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=derivatives,
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    return (
        torch.where(inside, transformed, inputs),
        torch.where(inside, logabsdet, torch.zeros_like(logabsdet)),
    )
