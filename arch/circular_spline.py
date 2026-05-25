import numpy as np
import torch
from torch.nn import functional as F

from nflows.transforms.autoregressive import AutoregressiveTransform
from nflows.transforms import made as made_module
from nflows.utils import torchutils

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def _searchsorted(bin_locations, inputs, eps=1e-6):
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
    num_bins = unnormalized_widths.shape[-1]

    if torch.is_tensor(left):
        limits_are_tensor = True
    else:
        limits_are_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if limits_are_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    if limits_are_tensor:
        cumheights = (top[..., None] - bottom[..., None]) * cumheights + bottom[..., None]
    else:
        cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        if torch.any(discriminant < 0):
            raise ValueError("Spline inversion failed: negative discriminant")

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (
        input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
    )
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    return outputs, logabsdet


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., 0] = constant
        unnormalized_derivatives_[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif tails == "circular":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(0, 1))
        unnormalized_derivatives_[..., -1] = unnormalized_derivatives_[..., 0]

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif isinstance(tails, (list, tuple)):
        if len(tails) != inputs.shape[-1]:
            raise ValueError("tails must have one entry per input feature")
        unnormalized_derivatives_ = unnormalized_derivatives.clone()
        linear_mask = [t == "linear" for t in tails]
        circular_mask = [t == "circular" for t in tails]
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., linear_mask, 0] = constant
        unnormalized_derivatives_[..., linear_mask, -1] = constant
        unnormalized_derivatives_[..., circular_mask, -1] = unnormalized_derivatives_[
            ..., circular_mask, 0
        ]
        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_[inside_interval_mask]
        right = tail_bound_[inside_interval_mask]
        bottom = -tail_bound_[inside_interval_mask]
        top = tail_bound_[inside_interval_mask]
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound

    outputs_masked, logabsdet_masked = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives_[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    if outputs.dtype == outputs_masked.dtype and logabsdet.dtype == logabsdet_masked.dtype:
        outputs[inside_interval_mask] = outputs_masked
        logabsdet[inside_interval_mask] = logabsdet_masked
    else:
        outputs[inside_interval_mask] = outputs_masked.to(outputs.dtype)
        logabsdet[inside_interval_mask] = logabsdet_masked.to(logabsdet.dtype)

    return outputs, logabsdet


class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        self.num_bins = num_bins
        self.min_bin_width = DEFAULT_MIN_BIN_WIDTH
        self.min_bin_height = DEFAULT_MIN_BIN_HEIGHT
        self.min_derivative = DEFAULT_MIN_DERIVATIVE
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        if self.tails == "circular":
            return self.num_bins * 3
        if self.tails is None or isinstance(self.tails, (list, tuple)):
            return self.num_bins * 3 + 1
        raise ValueError(f"Unsupported tails mode: {self.tails}")

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs,
        )

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)


class CircularAutoregressiveRationalQuadraticSpline(
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
):
    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        ind_circ,
        num_context_channels=None,
        num_bins=8,
        tail_bound=1.0,
        activation=F.relu,
        dropout_probability=0.0,
        random_mask=False,
        use_batch_norm=False,
    ):
        if num_input_channels <= 0:
            raise ValueError("num_input_channels must be positive")
        if not isinstance(ind_circ, (list, tuple)):
            raise ValueError("ind_circ must be a list or tuple of indices")
        ind_circ = list(ind_circ)
        if len(ind_circ) == 0:
            tails = "linear"
        else:
            tails = [
                "circular" if i in ind_circ else "linear"
                for i in range(num_input_channels)
            ]
        super().__init__(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
