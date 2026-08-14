import numpy as np
import torch
from torch import nn
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
        # Roundoff can make an analytically non-negative discriminant slightly
        # negative. Reject a materially invalid inverse, but clamp roundoff.
        scale = torch.maximum(b.pow(2).abs(), (4 * a * c).abs()).clamp_min(1.0)
        tolerance = 100 * torch.finfo(discriminant.dtype).eps * scale
        if torch.any(discriminant < -tolerance):
            raise ValueError("Spline inversion failed: negative discriminant")
        discriminant = discriminant.clamp_min(0.0)

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
    if tails == "linear":
        constant = np.log(np.exp(1 - min_derivative) - 1)
        constant_tensor = torch.full_like(unnormalized_derivatives[..., :1], constant)
        unnormalized_derivatives_ = torch.cat(
            [constant_tensor, unnormalized_derivatives, constant_tensor],
            dim=-1,
        )
    elif tails == "circular":
        unnormalized_derivatives_ = torch.cat(
            [unnormalized_derivatives, unnormalized_derivatives[..., :1]],
            dim=-1,
        )
    elif isinstance(tails, (list, tuple)):
        if len(tails) != inputs.shape[-1]:
            raise ValueError("tails must have one entry per input feature")
        linear_mask = torch.tensor(
            [t == "linear" for t in tails],
            device=inputs.device,
            dtype=torch.bool,
        )
        circular_mask = torch.tensor(
            [t == "circular" for t in tails],
            device=inputs.device,
            dtype=torch.bool,
        )
        constant = np.log(np.exp(1 - min_derivative) - 1)
        constant_tensor = torch.full_like(unnormalized_derivatives[..., :1], constant)
        linear_mask = linear_mask.view(1, -1, 1)
        circular_mask = circular_mask.view(1, -1, 1)
        first = torch.where(linear_mask, constant_tensor, unnormalized_derivatives[..., :1])
        last = torch.where(linear_mask, constant_tensor, unnormalized_derivatives[..., -1:])
        last = torch.where(circular_mask, unnormalized_derivatives[..., :1], last)
        middle = unnormalized_derivatives[..., 1:-1]
        unnormalized_derivatives_ = torch.cat([first, middle, last], dim=-1)
    else:
        raise RuntimeError(f"{tails} tails are not implemented.")

    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_
        right = tail_bound_
        bottom = -tail_bound_
        top = tail_bound_
        inputs_clamped = torch.maximum(torch.minimum(inputs, tail_bound_), -tail_bound_)
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound
        inputs_clamped = inputs.clamp(min=-tail_bound, max=tail_bound)

    outputs_masked, logabsdet_masked = rational_quadratic_spline(
        inputs=inputs_clamped,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives_,
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    outputs = torch.where(inside_interval_mask, outputs_masked, inputs)
    zero_logabsdet = torch.zeros_like(logabsdet_masked)
    logabsdet = torch.where(inside_interval_mask, logabsdet_masked, zero_logabsdet)

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
        identity_init=True,
        use_batch_norm=False,
    ):
        if num_input_channels <= 0:
            raise ValueError("num_input_channels must be positive")
        if not isinstance(ind_circ, (list, tuple)):
            raise ValueError("ind_circ must be a list or tuple of indices")
        if any(type(index) is not int for index in ind_circ):
            raise TypeError("circular indices must be integers")
        if len(set(ind_circ)) != len(ind_circ):
            raise ValueError("circular indices must be unique")
        if any(index < 0 or index >= num_input_channels for index in ind_circ):
            raise ValueError("circular index is out of bounds")
        if not np.isfinite(tail_bound) or float(tail_bound) != 1.0:
            raise ValueError(
                "theta_int is normalized by pi, so circular tail_bound must equal 1.0"
            )
        ind_circ = list(ind_circ)
        self.ind_circ = tuple(ind_circ)
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
        if identity_init:
            self._initialize_identity()

    def _canonicalize_circular(self, values):
        if not self.ind_circ:
            return values
        canonical = values.clone()
        circular_indices = list(self.ind_circ)
        bound = float(self.tail_bound)
        period = 2.0 * bound
        circular = canonical[..., circular_indices]
        outside = torch.isfinite(circular) & (
            (circular < -bound) | (circular >= bound)
        )
        wrapped = torch.remainder(circular + bound, period) - bound
        canonical[..., circular_indices] = torch.where(
            outside, wrapped, circular
        )
        return canonical

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        """Apply the spline in the canonical circular coordinate chart.

        ``+tail_bound`` and ``-tail_bound`` are the same point. Floating-point
        spline arithmetic can land a few ulps across that seam. Canonicalizing
        both inputs and outputs prevents such a representative from entering
        the identity-tail branch of a later layer. This is modulo arithmetic,
        not clipping, and has unit Jacobian almost everywhere.
        """
        inputs = self._canonicalize_circular(inputs)
        outputs, logabsdet = super()._elementwise(
            inputs, autoregressive_params, inverse=inverse
        )
        return self._canonicalize_circular(outputs), logabsdet

    def _initialize_identity(self):
        """Start at the identity map, including unit boundary derivatives."""
        final_layer = self.autoregressive_net.final_layer
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        derivative_value = np.log(np.expm1(1.0 - self.min_derivative))
        with torch.no_grad():
            bias = final_layer.bias.view(-1, self._output_dim_multiplier())
            bias[:, 2 * self.num_bins :] = derivative_value
