import logging
from collections.abc import Mapping
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
import math
import normflows as nf
from nflows.flows.base import Flow
from nflows.distributions.base import Distribution
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.permutations import Permutation, ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform
from nflows.utils import torchutils

from circular_spline import (
    DEFAULT_MIN_DERIVATIVE,
    CircularAutoregressiveRationalQuadraticSpline,
    unconstrained_rational_quadratic_spline,
)

import config
from utils import resolve_feature_index
from data import (
    D4_ELEMENTS,
    D4_INVERSES,
    TFCalculator,
    apply_d4_to_datavector,
    transform_d4_feature_blocks,
    transform_d4_fiber_mask,
    transform_d4_parameters,
)

FLOW_TYPES = (
    "affine",
    "circular_rqs",
    "hybrid_circular",
    "bounded_hybrid_circular",
)
DEFAULT_OBSERVATION_CONTEXT_FIELDS = (
    "rmag_obs",
    "rmag_sigma",
    "image_snr",
    "spectral_reference_quality",
    "spectral_noise_scale",
)



class ConditionalNormalWithCircularTheta(Distribution):
    """Conditional Gaussian base on R^(D-1) times Uniform(S1).

    The circular coordinate is the final feature and is represented on the
    canonical normalized interval [-1, 1). Its density is exactly 1/2 with
    respect to that coordinate. This is a proper compact latent base, unlike
    drawing theta from a Gaussian and wrapping the result afterward.
    """

    def __init__(self, features, context_encoder):
        super().__init__()
        if type(features) is not int or features < 2:
            raise ValueError("features must be an integer of at least two")
        self.features = features
        self.linear_features = features - 1
        self.context_encoder = context_encoder
        self.register_buffer(
            "_normal_log_z",
            torch.tensor(
                0.5 * self.linear_features * math.log(2.0 * math.pi),
                dtype=torch.float64,
            ),
            persistent=False,
        )

    def _compute_params(self, context):
        if context is None:
            raise ValueError("Context cannot be None for the conditional base")
        params = self.context_encoder(context)
        expected = 2 * self.linear_features
        if params.shape != (context.shape[0], expected):
            raise RuntimeError(
                "Circular base context encoder must return shape "
                f"({context.shape[0]}, {expected}); got {tuple(params.shape)}"
            )
        return params.chunk(2, dim=-1)

    def _log_prob(self, inputs, context):
        if inputs.ndim != 2 or inputs.shape[1] != self.features:
            raise ValueError(
                f"Expected inputs with shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )
        means, log_stds = self._compute_params(context)
        linear = inputs[:, :-1]
        normalized = (linear - means) * torch.exp(-log_stds)
        log_prob = -0.5 * normalized.square().sum(dim=-1)
        log_prob -= log_stds.sum(dim=-1)
        log_prob -= self._normal_log_z.to(dtype=inputs.dtype)
        log_prob -= math.log(2.0)
        theta = inputs[:, -1]
        on_circle = torch.isfinite(theta) & (theta >= -1.0) & (theta < 1.0)
        return torch.where(
            on_circle,
            log_prob,
            torch.full_like(log_prob, -torch.inf),
        )

    def _sample(self, num_samples, context):
        means, log_stds = self._compute_params(context)
        context_size = context.shape[0]
        means = torchutils.repeat_rows(means, num_samples)
        stds = torchutils.repeat_rows(torch.exp(log_stds), num_samples)
        linear = means + stds * torch.randn_like(means)
        theta = 2.0 * torch.rand(
            context_size * num_samples,
            1,
            device=context.device,
            dtype=linear.dtype,
        ) - 1.0
        samples = torch.cat((linear, theta), dim=-1)
        return torchutils.split_leading_dim(
            samples, [context_size, num_samples]
        )

    def _mean(self, context):
        means, _ = self._compute_params(context)
        return torch.cat((means, means.new_zeros(means.shape[0], 1)), dim=-1)


class PeriodicThetaFlow(Flow):
    """Flow whose public theta coordinate is canonicalized on the circle."""

    def __init__(self, transform, distribution, theta_index):
        super().__init__(transform, distribution)
        self.theta_index = int(theta_index)

    def _canonicalize_theta(self, inputs):
        canonical = inputs.clone()
        theta = canonical[..., self.theta_index]
        outside = torch.isfinite(theta) & ((theta < -1.0) | (theta >= 1.0))
        wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
        canonical[..., self.theta_index] = torch.where(
            outside, wrapped, theta
        )
        return canonical

    def _sample(self, num_samples, context):
        return self._canonicalize_theta(super()._sample(num_samples, context))

    def _log_prob(self, inputs, context):
        return super()._log_prob(self._canonicalize_theta(inputs), context)

    def transform_to_noise(self, inputs, context=None):
        return super().transform_to_noise(
            self._canonicalize_theta(inputs), context=context
        )


class ConditionalCircularThetaTransform(nn.Module):
    """One-dimensional circular RQS conditioned on Euclidean parameters.

    The conditioner sees the image/spectrum context and all seven non-angular
    parameters. Consequently this models p(theta | x_non_theta, context),
    rather than an independent theta density. Only theta enters an RQS.
    """

    def __init__(
        self,
        condition_features,
        *,
        num_layers=1,
        num_bins=8,
        hidden_features=128,
        logit_limit=10.0,
        tail_bound=1.0,
    ):
        super().__init__()
        if type(condition_features) is not int or condition_features <= 0:
            raise ValueError("condition_features must be a positive integer")
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if type(num_bins) is not int or num_bins < 2:
            raise ValueError("num_bins must be an integer of at least two")
        if not math.isfinite(float(logit_limit)) or float(logit_limit) <= 0:
            raise ValueError("logit_limit must be finite and positive")
        if float(tail_bound) != 1.0:
            raise ValueError(
                "theta_int is normalized by pi, so tail_bound must equal 1.0"
            )

        self.condition_features = condition_features
        self.num_layers = num_layers
        self.num_bins = num_bins
        self.logit_limit = float(logit_limit)
        self.tail_bound = float(tail_bound)
        self.min_derivative = DEFAULT_MIN_DERIVATIVE
        self.conditioners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(condition_features, hidden_features),
                    nn.SiLU(),
                    nn.Linear(hidden_features, hidden_features),
                    nn.SiLU(),
                    nn.Linear(hidden_features, 3 * num_bins),
                )
                for _ in range(num_layers)
            ]
        )
        self._initialize_identity()

    @staticmethod
    def _canonicalize_public(theta):
        """Select a half-open representative for every finite circle value."""
        if not torch.isfinite(theta).all():
            raise FloatingPointError("theta_int contains a non-finite value")
        outside = (theta < -1.0) | (theta >= 1.0)
        wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
        return torch.where(outside, wrapped, theta)

    @staticmethod
    def _canonicalize_internal(theta, *, operation):
        """Canonicalize every finite circular representative modulo two."""
        if not torch.isfinite(theta).all():
            raise FloatingPointError(
                f"non-finite theta_int produced during circular {operation}"
            )
        outside = (theta < -1.0) | (theta >= 1.0)
        wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
        return torch.where(outside, wrapped, theta)

    def _bounded_parameters(self, conditioner, condition):
        raw = conditioner(condition)
        bounded = self.logit_limit * torch.tanh(raw / self.logit_limit)
        widths = bounded[..., : self.num_bins].unsqueeze(-2)
        heights = bounded[..., self.num_bins : 2 * self.num_bins].unsqueeze(-2)
        derivatives = bounded[..., 2 * self.num_bins :].unsqueeze(-2)
        return raw, widths, heights, derivatives

    def _apply_layer(self, theta, condition, conditioner, *, inverse):
        raw, widths, heights, derivatives = self._bounded_parameters(
            conditioner, condition
        )
        transformed, logabsdet = unconstrained_rational_quadratic_spline(
            inputs=theta.unsqueeze(-1),
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=inverse,
            tails="circular",
            tail_bound=self.tail_bound,
        )
        transformed = transformed.squeeze(-1)
        wrap_mask = torch.isfinite(transformed) & (
            (transformed < -1.0) | (transformed >= 1.0)
        )
        wrap_excursion = torch.maximum(
            (-1.0 - transformed).clamp_min(0.0),
            (transformed - 1.0).clamp_min(0.0),
        )
        transformed = self._canonicalize_internal(
            transformed,
            operation="inverse" if inverse else "forward",
        )
        return (
            transformed,
            logabsdet.squeeze(-1),
            raw,
            wrap_mask.sum(),
            wrap_excursion.max(),
        )

    def forward(self, theta, condition):
        if theta.ndim != 1:
            raise ValueError("theta must have shape (batch,)")
        if condition.shape != (theta.shape[0], self.condition_features):
            raise ValueError(
                "condition must have shape "
                f"({theta.shape[0]}, {self.condition_features})"
            )
        value = self._canonicalize_public(theta)
        total_logabsdet = torch.zeros_like(value)
        raw_max = value.new_zeros(())
        bounded_max = value.new_zeros(())
        derivative_min = value.new_tensor(torch.inf)
        derivative_max = value.new_tensor(-torch.inf)
        wrap_count = value.new_zeros(())
        max_wrap_excursion = value.new_zeros(())
        for conditioner in self.conditioners:
            value, logabsdet, raw, layer_wrap_count, layer_wrap_excursion = self._apply_layer(
                value, condition, conditioner, inverse=False
            )
            total_logabsdet = total_logabsdet + logabsdet
            raw_max = torch.maximum(raw_max, raw.detach().abs().max())
            bounded = self.logit_limit * torch.tanh(
                raw.detach() / self.logit_limit
            )
            bounded_max = torch.maximum(bounded_max, bounded.abs().max())
            derivatives = self.min_derivative + F.softplus(
                bounded[..., 2 * self.num_bins :]
            )
            derivative_min = torch.minimum(
                derivative_min, derivatives.min()
            )
            derivative_max = torch.maximum(
                derivative_max, derivatives.max()
            )
            wrap_count = wrap_count + layer_wrap_count
            max_wrap_excursion = torch.maximum(
                max_wrap_excursion, layer_wrap_excursion
            )
        self.last_diagnostics = {
            "theta_raw_logit_abs_max": raw_max.detach(),
            "theta_bounded_logit_abs_max": bounded_max.detach(),
            "theta_derivative_min": derivative_min.detach(),
            "theta_derivative_max": derivative_max.detach(),
            "theta_wrap_count": wrap_count.detach(),
            "theta_max_wrap_excursion": max_wrap_excursion.detach(),
            "theta_logdet_min": total_logabsdet.detach().min(),
            "theta_logdet_max": total_logabsdet.detach().max(),
        }
        return value, total_logabsdet

    def inverse(self, latent_theta, condition):
        if latent_theta.ndim != 1:
            raise ValueError("latent_theta must have shape (batch,)")
        if condition.shape != (
            latent_theta.shape[0],
            self.condition_features,
        ):
            raise ValueError(
                "condition must have shape "
                f"({latent_theta.shape[0]}, {self.condition_features})"
            )
        value = self._canonicalize_public(latent_theta)
        total_logabsdet = torch.zeros_like(value)
        for conditioner in reversed(self.conditioners):
            value, logabsdet, _, _, _ = self._apply_layer(
                value, condition, conditioner, inverse=True
            )
            total_logabsdet = total_logabsdet + logabsdet
        return value, total_logabsdet

    def diagnostics(self, theta, condition):
        """Return detached spline-health diagnostics for a training batch."""
        value = self._canonicalize_public(theta)
        raw_max = value.new_zeros(())
        bounded_max = value.new_zeros(())
        logdet_min = value.new_tensor(torch.inf)
        logdet_max = value.new_tensor(-torch.inf)
        for conditioner in self.conditioners:
            raw, widths, heights, derivatives = self._bounded_parameters(
                conditioner, condition
            )
            bounded_max = torch.maximum(
                bounded_max,
                torch.stack(
                    (
                        widths.abs().max(),
                        heights.abs().max(),
                        derivatives.abs().max(),
                    )
                ).max(),
            )
            raw_max = torch.maximum(raw_max, raw.abs().max())
            value, logdet, _, _, _ = self._apply_layer(
                value, condition, conditioner, inverse=False
            )
            logdet_min = torch.minimum(logdet_min, logdet.min())
            logdet_max = torch.maximum(logdet_max, logdet.max())
        return {
            "theta_raw_logit_abs_max": raw_max.detach(),
            "theta_bounded_logit_abs_max": bounded_max.detach(),
            "theta_logdet_min": logdet_min.detach(),
            "theta_logdet_max": logdet_max.detach(),
        }

    def _initialize_identity(self):
        target = math.log(math.expm1(1.0 - self.min_derivative))
        if abs(target) >= self.logit_limit:
            raise ValueError("logit_limit is too small for identity initialization")
        raw_derivative_bias = self.logit_limit * math.atanh(
            target / self.logit_limit
        )
        for conditioner in self.conditioners:
            final = conditioner[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            with torch.no_grad():
                final.bias[2 * self.num_bins :] = raw_derivative_bias


class ConditionalUnitBox(Distribution):
    """Uniform distribution on ``[0, 1]^D`` with nflows context semantics.

    The density itself does not depend on context, but accepting context is
    essential: :class:`nflows.flows.base.Flow` uses it to determine the
    leading batch dimension of conditional samples.
    """

    def __init__(self, features):
        super().__init__()
        if type(features) is not int or features <= 0:
            raise ValueError("features must be a positive integer")
        self.features = features
        self.register_buffer("_reference", torch.zeros(features))

    def _validate_inputs(self, inputs):
        if inputs.ndim != 2 or inputs.shape[-1] != self.features:
            raise ValueError(
                f"Expected inputs with shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )

    def _log_prob(self, inputs, context):
        self._validate_inputs(inputs)
        on_support = (
            torch.isfinite(inputs)
            & (inputs >= 0.0)
            & (inputs <= 1.0)
        ).all(dim=-1)
        zeros = inputs.new_zeros(inputs.shape[0])
        return torch.where(
            on_support,
            zeros,
            torch.full_like(zeros, -torch.inf),
        )

    def _sample(self, num_samples, context):
        if context is None:
            return torch.rand(
                num_samples,
                self.features,
                device=self._reference.device,
                dtype=self._reference.dtype,
            )
        return torch.rand(
            context.shape[0],
            num_samples,
            self.features,
            device=context.device,
            dtype=context.dtype,
        )

    def _mean(self, context):
        if context is None:
            return self._reference.new_full((self.features,), 0.5)
        return context.new_full((context.shape[0], self.features), 0.5)


class IdentityBoundedRationalQuadraticAutoregressiveTransform(
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
):
    """Compact autoregressive RQS with bounded logits and identity start.

    ``tails=None`` in nflows defines the spline exactly on the unit interval.
    The MADE output is initialized so widths and heights are uniform and every
    knot derivative is one. Tanh-bounding the logits prevents a conditioner
    excursion from collapsing bins or driving derivatives arbitrarily large.
    """

    def __init__(
        self,
        *,
        features,
        hidden_features,
        context_features=None,
        num_bins=8,
        num_blocks=2,
        logit_limit=10.0,
        min_bin_width=1e-3,
        min_bin_height=1e-3,
        min_derivative=1e-3,
    ):
        if not math.isfinite(float(logit_limit)) or float(logit_limit) <= 0:
            raise ValueError("logit_limit must be finite and positive")
        self.logit_limit = float(logit_limit)
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_bins=num_bins,
            tails=None,
            num_blocks=num_blocks,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )
        self._initialize_identity()
        self.last_diagnostics = {}

    def _initialize_identity(self):
        target = math.log(math.expm1(1.0 - self.min_derivative))
        if abs(target) >= self.logit_limit:
            raise ValueError(
                "logit_limit is too small for identity initialization"
            )
        raw_derivative_bias = self.logit_limit * math.atanh(
            target / self.logit_limit
        )
        final = self.autoregressive_net.final_layer
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        multiplier = self._output_dim_multiplier()
        with torch.no_grad():
            bias = final.bias.view(-1, multiplier)
            bias[:, 2 * self.num_bins :] = raw_derivative_bias

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        raw = autoregressive_params
        bounded = self.logit_limit * torch.tanh(
            raw / self.logit_limit
        )
        # Stock nflows 0.14 can lose a few 1e-5 near a compact boundary while
        # solving the inverse quadratic in float32.  In a stack, that tiny
        # excursion becomes an out-of-domain input to the next spline.  The
        # mathematical inverse is inside [0, 1], so evaluate only the small
        # elementwise inverse solve in float64 and cast its in-support result
        # back to the public dtype.  Conditioner networks and forward training
        # remain in their configured precision.
        use_precise_inverse = inverse and inputs.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
        spline_inputs = inputs.double() if use_precise_inverse else inputs
        spline_parameters = (
            bounded.double() if use_precise_inverse else bounded
        )
        outputs, logabsdet = super()._elementwise(
            spline_inputs, spline_parameters, inverse=inverse
        )
        if use_precise_inverse:
            outputs = outputs.to(dtype=inputs.dtype)
            logabsdet = logabsdet.to(dtype=inputs.dtype)
        shaped = bounded.view(
            bounded.shape[0],
            inputs.shape[1],
            self._output_dim_multiplier(),
        )
        derivatives = self.min_derivative + F.softplus(
            shaped[..., 2 * self.num_bins :]
        )
        self.last_diagnostics = {
            "raw_logit_abs_max": raw.detach().abs().max(),
            "bounded_logit_abs_max": bounded.detach().abs().max(),
            "derivative_min": derivatives.detach().min(),
            "derivative_max": derivatives.detach().max(),
            "logdet_min": logabsdet.detach().min(),
            "logdet_max": logabsdet.detach().max(),
        }
        return outputs, logabsdet


class HybridAffineCircularFlow(nn.Module):
    """Affine seven-parameter marginal and correlated circular theta factor.

    The factorization q(x, theta | context) =
    q_affine(x | context) q_circular(theta | x, context) retains the complete
    dependence between theta and every other inferred parameter.
    """

    def __init__(
        self,
        *,
        features,
        theta_index,
        context_features,
        num_affine_layers=12,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=256,
        num_blocks=2,
        theta_hidden_features=128,
        logit_limit=10.0,
    ):
        super().__init__()
        if type(features) is not int or features < 2:
            raise ValueError("features must be an integer of at least two")
        if type(theta_index) is not int or not 0 <= theta_index < features:
            raise ValueError("theta_index is out of bounds")
        if type(context_features) is not int or context_features <= 0:
            raise ValueError("context_features must be a positive integer")
        if type(num_affine_layers) is not int or num_affine_layers <= 0:
            raise ValueError("num_affine_layers must be a positive integer")

        self.features = features
        self.theta_index = theta_index
        self.context_features = context_features
        self.non_theta_indices = tuple(
            index for index in range(features) if index != theta_index
        )
        self.linear_features = features - 1

        affine_base = ConditionalDiagonalNormal(
            shape=[self.linear_features],
            context_encoder=MLP(
                [context_features, 128, 64, 2 * self.linear_features]
            ),
        )
        affine_transforms = []
        for _ in range(num_affine_layers):
            affine_transforms.append(
                ReversePermutation(features=self.linear_features)
            )
            affine_transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.linear_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks,
                    context_features=context_features,
                )
            )
        self.affine_flow = Flow(
            CompositeTransform(affine_transforms), affine_base
        )
        self.theta_transform = ConditionalCircularThetaTransform(
            context_features + self.linear_features,
            num_layers=num_theta_layers,
            num_bins=num_bins,
            hidden_features=theta_hidden_features,
            logit_limit=logit_limit,
        )

    def _validate_context(self, context, batch_size):
        if context is None:
            raise ValueError("context cannot be None for a conditional flow")
        if context.shape != (batch_size, self.context_features):
            raise ValueError(
                "context must have shape "
                f"({batch_size}, {self.context_features}); got {tuple(context.shape)}"
            )

    def _split_inputs(self, inputs):
        if inputs.ndim != 2 or inputs.shape[-1] != self.features:
            raise ValueError(
                f"inputs must have shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )
        return (
            inputs[:, list(self.non_theta_indices)],
            inputs[:, self.theta_index],
        )

    def _assemble(self, non_theta, theta):
        output = non_theta.new_empty((*non_theta.shape[:-1], self.features))
        output[..., list(self.non_theta_indices)] = non_theta
        output[..., self.theta_index] = theta
        return output

    def component_log_prob(self, inputs, context):
        non_theta, theta = self._split_inputs(inputs)
        self._validate_context(context, inputs.shape[0])
        affine_log_prob = self.affine_flow.log_prob(
            non_theta, context=context
        )
        condition = torch.cat((context, non_theta), dim=-1)
        _, theta_logabsdet = self.theta_transform(theta, condition)
        theta_log_prob = theta_logabsdet - math.log(2.0)
        self.last_component_diagnostics = {
            "affine_log_prob_mean": affine_log_prob.detach().mean(),
            "theta_log_prob_mean": theta_log_prob.detach().mean(),
        }
        return affine_log_prob, theta_log_prob

    def log_prob(self, inputs, context=None):
        affine_log_prob, theta_log_prob = self.component_log_prob(
            inputs, context
        )
        return affine_log_prob + theta_log_prob

    def sample(self, num_samples, context=None):
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if context is None or context.ndim != 2:
            raise ValueError("context must have shape (batch, context_features)")
        self._validate_context(context, context.shape[0])
        non_theta = self.affine_flow.sample(num_samples, context=context)
        batch_size = context.shape[0]
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_features
        )
        condition = torch.cat((expanded_context, non_theta), dim=-1).reshape(
            batch_size * num_samples, -1
        )
        latent_theta = 2.0 * torch.rand(
            batch_size * num_samples,
            device=context.device,
            dtype=non_theta.dtype,
        ) - 1.0
        theta, _ = self.theta_transform.inverse(latent_theta, condition)
        return self._assemble(
            non_theta, theta.reshape(batch_size, num_samples)
        )

    def sample_and_log_prob(self, num_samples, context=None):
        samples = self.sample(num_samples, context=context)
        batch_size = samples.shape[0]
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_features
        )
        log_prob = self.log_prob(
            samples.reshape(batch_size * num_samples, self.features),
            context=expanded_context.reshape(batch_size * num_samples, -1),
        ).reshape(batch_size, num_samples)
        return samples, log_prob

    def transform_to_noise(self, inputs, context=None):
        non_theta, theta = self._split_inputs(inputs)
        self._validate_context(context, inputs.shape[0])
        affine_noise = self.affine_flow.transform_to_noise(
            non_theta, context=context
        )
        condition = torch.cat((context, non_theta), dim=-1)
        theta_noise, _ = self.theta_transform(theta, condition)
        return self._assemble(affine_noise, theta_noise)

    def theta_diagnostics(self, inputs, context=None):
        non_theta, theta = self._split_inputs(inputs)
        self._validate_context(context, inputs.shape[0])
        condition = torch.cat((context, non_theta), dim=-1)
        return self.theta_transform.diagnostics(theta, condition)


class BoundedHybridCircularFlow(HybridAffineCircularFlow):
    """Compact seven-parameter marginal and correlated circular theta.

    Public non-angular coordinates live on ``[-1, 1]``. A fixed affine map
    moves them to ``[0, 1]`` before compact autoregressive splines and a unit
    box base. The angular factor remains
    ``q(theta_int | x_non_theta, context)``, so bounding the scalar marginal
    does not remove any theta correlations.
    """

    def __init__(
        self,
        *,
        features,
        theta_index,
        context_features,
        num_bounded_layers=12,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=256,
        num_blocks=2,
        theta_hidden_features=128,
        logit_limit=10.0,
        bounded_logit_limit=10.0,
    ):
        nn.Module.__init__(self)
        if type(features) is not int or features < 2:
            raise ValueError("features must be an integer of at least two")
        if type(theta_index) is not int or not 0 <= theta_index < features:
            raise ValueError("theta_index is out of bounds")
        if type(context_features) is not int or context_features <= 0:
            raise ValueError("context_features must be a positive integer")
        if type(num_bounded_layers) is not int or num_bounded_layers <= 0:
            raise ValueError("num_bounded_layers must be a positive integer")

        self.features = features
        self.theta_index = theta_index
        self.context_features = context_features
        self.non_theta_indices = tuple(
            index for index in range(features) if index != theta_index
        )
        self.linear_features = features - 1

        bounded_transforms = [
            PointwiseAffineTransform(scale=0.5, shift=0.5)
        ]
        for _ in range(num_bounded_layers):
            bounded_transforms.append(
                ReversePermutation(features=self.linear_features)
            )
            bounded_transforms.append(
                IdentityBoundedRationalQuadraticAutoregressiveTransform(
                    features=self.linear_features,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    num_blocks=num_blocks,
                    logit_limit=bounded_logit_limit,
                )
            )
        # Keep this historical attribute name so optimizer grouping continues
        # to isolate the non-theta marginal without special-case traversal.
        self.affine_flow = Flow(
            CompositeTransform(bounded_transforms),
            ConditionalUnitBox(self.linear_features),
        )
        self.theta_transform = ConditionalCircularThetaTransform(
            context_features + self.linear_features,
            num_layers=num_theta_layers,
            num_bins=num_bins,
            hidden_features=theta_hidden_features,
            logit_limit=logit_limit,
        )

    @property
    def bounded_transforms(self):
        return tuple(
            transform
            for transform in self.affine_flow._transform._transforms
            if isinstance(
                transform,
                IdentityBoundedRationalQuadraticAutoregressiveTransform,
            )
        )

    @staticmethod
    def _support_mask(non_theta):
        return (
            torch.isfinite(non_theta)
            & (non_theta >= -1.0)
            & (non_theta <= 1.0)
        ).all(dim=-1)

    @staticmethod
    def _safe_for_density(non_theta):
        finite = torch.isfinite(non_theta)
        clipped = non_theta.clamp(min=-1.0, max=1.0)
        return torch.where(finite, clipped, torch.zeros_like(non_theta))

    def _assert_support(self, non_theta, *, operation):
        support = self._support_mask(non_theta)
        if not bool(support.all()):
            count = int((~support).sum().item())
            raise RuntimeError(
                f"bounded_hybrid_circular produced {count} non-theta rows "
                f"outside [-1, 1] during {operation}; refusing to clamp"
            )

    def _bounded_diagnostics(self, support):
        diagnostics = [
            transform.last_diagnostics
            for transform in self.bounded_transforms
            if transform.last_diagnostics
        ]
        output = {
            "bounded_support_violation_count": (~support)
            .sum()
            .detach()
            .to(dtype=self.affine_flow._distribution._reference.dtype)
        }
        if not diagnostics:
            return output
        output.update(
            {
                "bounded_raw_logit_abs_max": torch.stack(
                    [item["raw_logit_abs_max"] for item in diagnostics]
                ).max(),
                "bounded_logit_abs_max": torch.stack(
                    [item["bounded_logit_abs_max"] for item in diagnostics]
                ).max(),
                "bounded_derivative_min": torch.stack(
                    [item["derivative_min"] for item in diagnostics]
                ).min(),
                "bounded_derivative_max": torch.stack(
                    [item["derivative_max"] for item in diagnostics]
                ).max(),
                "bounded_logdet_min": torch.stack(
                    [item["logdet_min"] for item in diagnostics]
                ).min(),
                "bounded_logdet_max": torch.stack(
                    [item["logdet_max"] for item in diagnostics]
                ).max(),
            }
        )
        return output

    def component_log_prob(self, inputs, context):
        non_theta, theta = self._split_inputs(inputs)
        self._validate_context(context, inputs.shape[0])
        support = self._support_mask(non_theta)
        safe_non_theta = self._safe_for_density(non_theta)
        bounded_log_prob = self.affine_flow.log_prob(
            safe_non_theta, context=context
        )
        bounded_log_prob = torch.where(
            support,
            bounded_log_prob,
            torch.full_like(bounded_log_prob, -torch.inf),
        )

        finite_theta = torch.isfinite(theta)
        safe_theta = torch.where(finite_theta, theta, torch.zeros_like(theta))
        condition = torch.cat((context, safe_non_theta), dim=-1)
        _, theta_logabsdet = self.theta_transform(safe_theta, condition)
        theta_log_prob = theta_logabsdet - math.log(2.0)
        theta_log_prob = torch.where(
            finite_theta,
            theta_log_prob,
            torch.full_like(theta_log_prob, -torch.inf),
        )
        self.last_component_diagnostics = {
            "affine_log_prob_mean": bounded_log_prob.detach().mean(),
            "bounded_log_prob_mean": bounded_log_prob.detach().mean(),
            "theta_log_prob_mean": theta_log_prob.detach().mean(),
            **self._bounded_diagnostics(support),
        }
        return bounded_log_prob, theta_log_prob

    def sample(self, num_samples, context=None):
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if context is None or context.ndim != 2:
            raise ValueError("context must have shape (batch, context_features)")
        self._validate_context(context, context.shape[0])
        non_theta = self.affine_flow.sample(num_samples, context=context)
        self._assert_support(non_theta, operation="sampling")
        batch_size = context.shape[0]
        expanded_context = context[:, None, :].expand(
            batch_size, num_samples, self.context_features
        )
        condition = torch.cat((expanded_context, non_theta), dim=-1).reshape(
            batch_size * num_samples, -1
        )
        latent_theta = 2.0 * torch.rand(
            batch_size * num_samples,
            device=context.device,
            dtype=non_theta.dtype,
        ) - 1.0
        theta, _ = self.theta_transform.inverse(latent_theta, condition)
        samples = self._assemble(
            non_theta, theta.reshape(batch_size, num_samples)
        )
        if not torch.isfinite(samples).all():
            raise RuntimeError(
                "bounded_hybrid_circular produced non-finite samples"
            )
        return samples

    def transform_to_noise(self, inputs, context=None):
        non_theta, theta = self._split_inputs(inputs)
        self._validate_context(context, inputs.shape[0])
        self._assert_support(non_theta, operation="transform_to_noise")
        if not torch.isfinite(theta).all():
            raise RuntimeError(
                "theta_int must be finite during transform_to_noise"
            )
        bounded_noise = self.affine_flow.transform_to_noise(
            non_theta, context=context
        )
        condition = torch.cat((context, non_theta), dim=-1)
        theta_noise, _ = self.theta_transform(theta, condition)
        return self._assemble(bounded_noise, theta_noise)


### Main Network ###
class KLNPE(nn.Module):
    '''
    Main network consisting of feature extraction branches for images and spectra,
    followed by either point estimate or density estimate layers.
    '''
    def __init__(self, 
                 feature_extractor=None,
                 mode=None,    # 0 = point estimate, 1 = density estimate, 2 = density estimate with TF prior
                 batch_size=None,
                 nfeatures=None,
                 nspec=None,
                 # Lognormal TF prior parameters (only used when mode == 2)
                 vcirc_dex=None,   # scatter in dex; fixed, represents TF relation scatter
                 vcirc_min=None,
                 vcirc_max=None,
                 vcirc_idx=None,
                 backbone_type=None,
                 posterior_symmetry=None):

        # Resolve configuration-backed defaults at construction time.  Python
        # evaluates function defaults when this module is imported, before a
        # launcher-supplied config is loaded and propagated to spawned workers.
        if mode is None:
            mode = config.train['mode']
        if batch_size is None:
            batch_size = config.train['batch_size']
        if nfeatures is None:
            nfeatures = config.train['feature_number']
        if nspec is None:
            nspec = config.data['nspec']
        if vcirc_dex is None:
            vcirc_dex = config.tf['scatter']
        vcirc_bounds = config.par_ranges.get('vcirc', [60.0, 540.0])
        if vcirc_min is None:
            vcirc_min = vcirc_bounds[0]
        if vcirc_max is None:
            vcirc_max = vcirc_bounds[1]

        self.bs = batch_size
        self.nfeatures = nfeatures
        self.nspecs = nspec
        observation_config = getattr(config, "observation", {})
        self.observation_model_version = int(
            observation_config.get("model_version", 1)
        )
        configured_context_fields = tuple(
            observation_config.get(
                "context_fields", DEFAULT_OBSERVATION_CONTEXT_FIELDS
            )
        )
        if self.observation_model_version == 1:
            # Keep the historical flow input size and state-dict layout exact.
            self.observation_context_fields = ()
        elif self.observation_model_version == 2:
            if configured_context_fields != DEFAULT_OBSERVATION_CONTEXT_FIELDS:
                raise ValueError(
                    "observation.context_fields must be exactly "
                    f"{list(DEFAULT_OBSERVATION_CONTEXT_FIELDS)!r}; got "
                    f"{list(configured_context_fields)!r}"
                )
            self.observation_context_fields = configured_context_fields
        else:
            raise ValueError("observation model_version must be 1 or 2")
        self.observation_context_features = len(self.observation_context_fields)
        self.flow_context_features = 1024 + self.observation_context_features

        rmag_min = float(observation_config.get("rmag_min", 15.0))
        rmag_max = float(observation_config.get("rmag_max", 23.4))
        if not rmag_min < rmag_max:
            raise ValueError("observation rmag_min must be below rmag_max")
        self.observation_rmag_midpoint = 0.5 * (rmag_min + rmag_max)
        self.observation_rmag_half_range = 0.5 * (rmag_max - rmag_min)
        quality_min = float(observation_config.get("spectral_quality_min", 3.0))
        quality_max = float(observation_config.get("spectral_quality_max", 100.0))
        if not 0.0 < quality_min < quality_max:
            raise ValueError(
                "spectral quality bounds must satisfy 0 < min < max"
            )
        self.observation_quality_log_midpoint = 0.5 * (
            math.log(quality_min) + math.log(quality_max)
        )
        self.observation_quality_log_half_range = 0.5 * math.log(
            quality_max / quality_min
        )
        self.feature_names = tuple(
            config.train.get(
                'feature_names',
                ["g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr"],
            )
        )
        if len(self.feature_names) != self.nfeatures:
            raise ValueError(
                "feature_names length must equal nfeatures; "
                f"got {len(self.feature_names)} and {self.nfeatures}"
            )
        if posterior_symmetry is None:
            posterior_symmetry = config.train.get("posterior_symmetry", "none")
        self.posterior_symmetry = str(posterior_symmetry).lower()
        if self.posterior_symmetry not in ("none", "d4"):
            raise ValueError("posterior_symmetry must be 'none' or 'd4'")
        if mode in (0, 1, 2):
            self.mode = mode
        else:
            raise ValueError('Mode must be 0 (point estimate), 1 (density estimate), or 2 (density estimate with TF prior)!')

        # Initialize apparent magnitude TF calculator with standard values
        self.tf_calc = TFCalculator(slope=config.tf['slope'], intercept=config.tf['intercept'])

        # Lognormal TF prior settings (only used when mode == 2)
        # dex is fixed (TF scatter); mu is supplied per-galaxy at runtime from magnitude measurements
        self.vcirc_dex = float(vcirc_dex)
        self.vcirc_log_scale = vcirc_dex * torch.log(torch.tensor(10.)).item()  # convert dex -> natural-log std
        self.vcirc_min = float(vcirc_min)
        self.vcirc_max = float(vcirc_max)
        self.vcirc_jac = 0.5 * (self.vcirc_max - self.vcirc_min)  # |dv/dx| for x in [-1, 1]
        if vcirc_idx is None:
            vcirc_idx = resolve_feature_index(
                self.feature_names, 'vcirc', aliases=('v_circ',)
            )
        self.vcirc_idx = int(vcirc_idx)
        if self.vcirc_idx < 0 or self.vcirc_idx >= self.nfeatures:
            raise ValueError(
                f'vcirc_idx={self.vcirc_idx} is out of bounds for nfeatures={self.nfeatures}'
            )

        super(KLNPE, self).__init__()
        if self.observation_model_version == 2:
            self.register_buffer(
                "image_noise_sigma", torch.tensor(float("nan"))
            )
            self.register_buffer(
                "spectral_reference_line_norm", torch.tensor(float("nan"))
            )


        if backbone_type is None:
            backbone_type = config.train.get("backbone_type", "legacy")
        self.feature_extractor = (
            build_feature_extractor(backbone_type, nspec=self.nspecs)
            if feature_extractor is None
            else feature_extractor
        )
        if self.posterior_symmetry == "d4":
            if self.mode == 0:
                raise ValueError("D4 posterior symmetry requires density mode 1 or 2")
            if not callable(getattr(self.feature_extractor, "transform_features", None)):
                raise ValueError(
                    "D4 posterior symmetry requires an equivariant feature extractor "
                    "with transform_features(features, element)"
                )

        # Define point estimate or density estimate layers
        if self.mode == 0:
            ### Fully-connected layers
            self.fully_connected_layer = MLP([1024, 512, 256, self.nfeatures])
            self.loss = nn.MSELoss()
        elif self.mode >= 1:
            # Normalizing flow for density estimation
            self.layer_norm = nn.LayerNorm(1024)
            if not bool(config.train.get("context_norm_trainable", True)):
                with torch.no_grad():
                    self.layer_norm.weight.fill_(1.0)
                    self.layer_norm.bias.zero_()
                self.layer_norm.requires_grad_(False)
            self.setup_flows()
            if self.flow_type == "circular_rqs":
                self.flow = PeriodicThetaFlow(
                    self.transform, self.base, theta_index=self.theta_idx
                )
            elif self.flow_type == "affine":
                self.flow = Flow(self.transform, self.base)
    def _prepare_observation_context(
        self, observation_context, batch_size, reference
    ):
        """Validate and standardize the observed scalar flow context.

        Tensor column order is recorded by observation.context_fields. Mapping
        inputs are safer at API boundaries because latent simulator quantities
        such as rmag_true and halpha_flux_true can be rejected by name.
        """
        if getattr(self, "observation_model_version", 1) == 1:
            if observation_context is not None:
                raise ValueError(
                    "observation_context is unavailable for legacy "
                    "observation model_version=1"
                )
            return None
        if observation_context is None:
            raise ValueError(
                "observation_context is required for observation model_version=2"
            )

        if isinstance(observation_context, Mapping):
            supplied = set(observation_context)
            if "rmag_true" in supplied:
                raise ValueError(
                    "observation_context must not contain latent rmag_true; "
                    "pass the noisy catalog measurement rmag_obs"
                )
            if "halpha_flux_true" in supplied:
                raise ValueError(
                    "observation_context must not contain latent "
                    "halpha_flux_true; spectral line strength is represented "
                    "only by the observed spectrum and noise metadata"
                )
            expected = set(self.observation_context_fields)
            if supplied != expected:
                missing = sorted(expected - supplied)
                extra = sorted(supplied - expected)
                raise ValueError(
                    "observation_context mapping keys do not match the "
                    f"archived context_fields; missing={missing}, extra={extra}"
                )
            columns = []
            for name in self.observation_context_fields:
                column = torch.as_tensor(
                    observation_context[name],
                    device=reference.device,
                    dtype=reference.dtype,
                )
                if column.ndim == 0:
                    column = column.expand(batch_size)
                elif column.ndim == 2 and column.shape[-1] == 1:
                    column = column[:, 0]
                if column.ndim != 1 or column.shape[0] != batch_size:
                    raise ValueError(
                        f"observation_context[{name!r}] must be scalar or "
                        f"have shape ({batch_size},); got {tuple(column.shape)}"
                    )
                columns.append(column)
            observed = torch.stack(columns, dim=-1)
        else:
            observed = torch.as_tensor(
                observation_context,
                device=reference.device,
                dtype=reference.dtype,
            )
            expected_shape = (
                batch_size,
                self.observation_context_features,
            )
            if observed.shape != expected_shape:
                raise ValueError(
                    "observation_context tensor must follow archived field "
                    f"order {list(self.observation_context_fields)!r} and have "
                    f"shape {expected_shape}; got {tuple(observed.shape)}"
                )

        if not bool(torch.isfinite(observed).all()):
            raise ValueError("observation_context must contain only finite values")
        positive = observed[:, 1:]
        if not bool((positive > 0).all()):
            raise ValueError(
                "rmag_sigma, image_snr, spectral_reference_quality, and "
                "spectral_noise_scale must all be positive"
            )
        reference_line_norm = self.spectral_reference_line_norm.to(
            device=observed.device, dtype=observed.dtype
        )
        if not bool(
            torch.isfinite(reference_line_norm)
            & (reference_line_norm > 0)
        ):
            raise RuntimeError(
                "spectral_reference_line_norm must be set to the positive "
                "training-set reference before using simulator-v2 context"
            )

        standardized = observed.clone()
        standardized[:, 0] = (
            observed[:, 0] - self.observation_rmag_midpoint
        ) / self.observation_rmag_half_range
        five_sigma_mag_error = (2.5 / math.log(10.0)) / 5.0
        standardized[:, 1] = torch.log10(
            observed[:, 1] / five_sigma_mag_error
        )
        standardized[:, 2] = torch.log10(observed[:, 2] / 5.0)
        standardized[:, 3] = (
            torch.log(observed[:, 3])
            - self.observation_quality_log_midpoint
        ) / self.observation_quality_log_half_range
        standardized[:, 4] = torch.log10(
            observed[:, 4] / reference_line_norm
        )
        return standardized

    def _flow_context_from_features(
        self, raw_features, prepared_observation_context
    ):
        visual_context = self.layer_norm(raw_features)
        if getattr(self, "observation_model_version", 1) == 1:
            return visual_context
        if (
            prepared_observation_context is None
            or prepared_observation_context.shape
            != (raw_features.shape[0], self.observation_context_features)
        ):
            raise ValueError(
                "prepared observation context does not match feature batch"
            )
        return torch.cat(
            (visual_context, prepared_observation_context), dim=-1
        )


    
    def forward(
        self,
        x,
        y,
        true,
        fp,
        mag=None,
        snr=None,
        observation_context=None,
    ):
        '''
        x: image tensor
        y: spectrum tensor
        true: target tensor of shape (batch, nfeatures)
        fp: fiber position tensor of shape (batch, nspecs, 2)
        '''
        raw_features = self.feature_extractor(x, y, fp)
        training_diagnostics = {
            "raw_feature_rms": raw_features.detach().square().mean().sqrt(),
        }

        if self.mode == 0:
            prediction = self.fully_connected_layer(raw_features)
            return self.loss(prediction, true)
        prepared_observation_context = self._prepare_observation_context(
            observation_context, raw_features.shape[0], raw_features
        )

        if self.posterior_symmetry == "d4":
            branch_log_prob = self._d4_branch_log_prob_from_features(
                raw_features, true, prepared_observation_context
            )
            per_galaxy_log_prob = branch_log_prob.mean(dim=1)
        else:
            context = self._flow_context_from_features(
                raw_features, prepared_observation_context
            )
            per_galaxy_log_prob = self.flow.log_prob(true, context=context)

        if self.mode == 2:
            # TF quantities are D4 scalars. Compute one weight per galaxy and
            # apply it only after averaging that galaxy's eight branch scores.
            weights = self._compute_tf_weights(true, mag, snr)
            training_diagnostics.update(
                getattr(self, "last_tf_diagnostics", {})
            )
            loss = -(weights * per_galaxy_log_prob).mean()
        else:
            loss = -per_galaxy_log_prob.mean()
        if getattr(self, "flow_type", "affine") in (
            "hybrid_circular",
            "bounded_hybrid_circular",
        ):
            training_diagnostics.update(
                getattr(self.flow.theta_transform, "last_diagnostics", {})
            )
            training_diagnostics.update(
                getattr(self.flow, "last_component_diagnostics", {})
            )
        self.last_training_diagnostics = training_diagnostics
        return loss

    def _d4_contexts_from_features(
        self, raw_features, prepared_observation_context=None
    ):
        """Build group-major contexts with invariant observed scalars."""
        if self.posterior_symmetry != "d4":
            raise RuntimeError("D4 contexts require posterior_symmetry='d4'")
        if raw_features.ndim != 2 or raw_features.shape[-1] != 1024:
            raise ValueError("raw feature tensor must have shape (batch, 1024)")
        visual_contexts = torch.stack(
            tuple(
                self.layer_norm(
                    self.feature_extractor.transform_features(raw_features, element)
                )
                for element in D4_ELEMENTS
            ),
            dim=0,
        )
        if getattr(self, "observation_model_version", 1) == 1:
            return visual_contexts
        expected_shape = (
            raw_features.shape[0],
            self.observation_context_features,
        )
        if (
            prepared_observation_context is None
            or prepared_observation_context.shape != expected_shape
        ):
            raise ValueError(
                "prepared observation context does not match D4 feature batch"
            )
        invariant_context = prepared_observation_context.unsqueeze(0).expand(
            len(D4_ELEMENTS), -1, -1
        )
        return torch.cat((visual_contexts, invariant_context), dim=-1)

    def _d4_branch_log_prob_from_features(
        self,
        raw_features,
        parameters,
        prepared_observation_context=None,
    ):
        """Return the eight branch log densities with shape (batch, group)."""
        if parameters.ndim != 2 or parameters.shape[0] != raw_features.shape[0]:
            raise ValueError("parameters must have shape (batch, nfeatures)")
        contexts = self._d4_contexts_from_features(
            raw_features, prepared_observation_context
        )
        transformed_parameters = torch.stack(
            tuple(
                transform_d4_parameters(
                    parameters,
                    element,
                    feature_names=self.feature_names,
                )
                for element in D4_ELEMENTS
            ),
            dim=0,
        )
        group_count, batch_size = transformed_parameters.shape[:2]
        log_prob = self.flow.log_prob(
            transformed_parameters.reshape(group_count * batch_size, self.nfeatures),
            context=contexts.reshape(group_count * batch_size, -1),
        )
        return log_prob.reshape(group_count, batch_size).transpose(0, 1)

    def _d4_mixture_log_prob_from_features(
        self,
        raw_features,
        parameters,
        prepared_observation_context=None,
    ):
        branch_log_prob = self._d4_branch_log_prob_from_features(
            raw_features, parameters, prepared_observation_context
        )
        return torch.logsumexp(branch_log_prob, dim=1) - math.log(len(D4_ELEMENTS))

    def posterior_log_prob(
        self,
        x,
        y,
        parameters,
        fp,
        observation_context=None,
    ):
        """Evaluate the configured conditional posterior at normalized parameters."""
        if self.mode == 0:
            raise RuntimeError("posterior_log_prob requires density mode 1 or 2")
        raw_features = self.feature_extractor(x, y, fp)
        prepared_observation_context = self._prepare_observation_context(
            observation_context, raw_features.shape[0], raw_features
        )
        if self.posterior_symmetry == "d4":
            return self._d4_mixture_log_prob_from_features(
                raw_features, parameters, prepared_observation_context
            )
        context = self._flow_context_from_features(
            raw_features, prepared_observation_context
        )
        return self.flow.log_prob(parameters, context=context)
    
    def posterior_mean(self, samples):
        """Return an equivariant posterior mean from normalized sample clouds.

        Scalars and shear use arithmetic means. ``theta_int`` uses its directed
        circular mean on the normalized ``[-1, 1)`` coordinate.
        """
        if samples.ndim < 2 or samples.shape[-1] != self.nfeatures:
            raise ValueError("samples must have shape (..., samples, nfeatures)")
        mean = samples.mean(dim=-2)
        theta_idx = resolve_feature_index(self.feature_names, "theta_int")
        theta = samples[..., theta_idx]
        theta_mean = torch.atan2(
            torch.sin(math.pi * theta).mean(dim=-1),
            torch.cos(math.pi * theta).mean(dim=-1),
        ) / math.pi
        mean[..., theta_idx] = torch.remainder(theta_mean + 1.0, 2.0) - 1.0
        return mean

    def point_estimate(self, x, y, fp):
        '''
        Run through feature extraction and return point estimate of parameters
        '''
        z = self.feature_extractor(x, y, fp)
        z = self.fully_connected_layer(z)
        return z
    
    def extract_latent(
        self, x, y, true, fp, observation_context=None
    ):
        '''
        Run through feature extraction but map from true parameters to latent space in flow
        '''
        raw_features = self.feature_extractor(x, y, fp)
        prepared_observation_context = self._prepare_observation_context(
            observation_context, raw_features.shape[0], raw_features
        )
        context = self._flow_context_from_features(
            raw_features, prepared_observation_context
        )
        latent = self.flow.transform_to_noise(true, context=context)

        return latent

    def _norm_to_vcirc(self, v_norm):
        v_norm = v_norm.clamp(min=-1.0, max=1.0)
        v_circ = self.vcirc_min + 0.5 * (v_norm + 1.0) * (self.vcirc_max - self.vcirc_min)
        return v_circ.clamp(min=1e-8)

    def _tf_log_prob_from_vnorm(self, v_norm, vcirc_mu):
        """
        v_norm: normalized vcirc in [-1, 1], shape (...)
        vcirc_mu: TF prior center in km/s, broadcastable to v_norm
        """
        min_val = 1
        v_circ = self._norm_to_vcirc(v_norm)
        v_circ = torch.where(
            torch.isfinite(v_circ) & (v_circ > 0),
            v_circ,
            torch.full_like(v_circ, min_val),
        )
        mu = vcirc_mu.to(device=v_circ.device, dtype=v_circ.dtype)
        mu = torch.where(
            torch.isfinite(mu) & (mu > 0),
            mu,
            torch.full_like(mu, min_val),
        )
        prior = torch.distributions.LogNormal(
            loc=torch.log(mu),
            scale=torch.full_like(v_circ, self.vcirc_log_scale)
        )
        # return prior.log_prob(v_circ)
        return prior.log_prob(v_circ) + torch.log(torch.full_like(v_circ, self.vcirc_jac))
    
    def _get_tf_prior_params(self, mag, snr):
        """
        Computes the LogNormal prior parameters for vcirc based on magnitude.
        Uses 0.1 dex base width, modified with a magnitude-dependent observational error.
        """
        # Calculate the expected vcirc center (mu)
        vcirc_mu = self.tf_calc.mag_to_vcirc(mag)
        
        if snr is None:
            # Calculate magnitude-dependent observational uncertainty
            # SNR = 5 * 10**(-0.4 * (mag - 23.4))
            snr = 5.0 * torch.pow(10.0, -0.4 * (mag - 23.4))

        sigma_m = 1.086 / snr
        
        # Propagate error into the TF space (sigma_total_dex = sqrt(0.1^2 + (sigma_m / slope)^2))
        slope = self.tf_calc.slope
        sigma_total_dex = torch.sqrt(self.vcirc_dex**2 + (sigma_m / slope)**2)
        
        # Convert dex back to natural log space for LogNormal prior evaluation
        sigma_total_ln = sigma_total_dex * math.log(10.0)
        
        return vcirc_mu, sigma_total_ln

    @staticmethod
    def _log_standard_normal_interval(lower, upper):
        """Return log(Phi(upper) - Phi(lower)) without tail cancellation."""
        lower, upper = torch.broadcast_tensors(lower, upper)
        if bool((lower >= upper).any()):
            raise ValueError("normal interval lower bound must be below upper bound")
        original_dtype = lower.dtype
        lower = lower.to(torch.float64)
        upper = upper.to(torch.float64)

        def log_difference(log_larger, log_smaller):
            return log_larger + torch.log(-torch.expm1(log_smaller - log_larger))

        result = torch.empty_like(lower)
        negative = upper <= 0
        positive = lower >= 0
        crossing = ~(negative | positive)
        if bool(negative.any()):
            result[negative] = log_difference(
                torch.special.log_ndtr(upper[negative]),
                torch.special.log_ndtr(lower[negative]),
            )
        if bool(positive.any()):
            result[positive] = log_difference(
                torch.special.log_ndtr(-lower[positive]),
                torch.special.log_ndtr(-upper[positive]),
            )
        if bool(crossing.any()):
            result[crossing] = log_difference(
                torch.special.log_ndtr(upper[crossing]),
                torch.special.log_ndtr(lower[crossing]),
            )
        return result.to(original_dtype)

    def tf_prior_log_prob(self, v_circ, mag, mag_sigma):
        """Log truncated TF density with respect to physical velocity."""
        v_circ = torch.as_tensor(v_circ)
        if not v_circ.is_floating_point():
            v_circ = v_circ.to(dtype=torch.get_default_dtype())
        mag = torch.as_tensor(mag, device=v_circ.device, dtype=v_circ.dtype)
        mag_sigma = torch.as_tensor(
            mag_sigma, device=v_circ.device, dtype=v_circ.dtype
        )
        v_circ, mag, mag_sigma = torch.broadcast_tensors(
            v_circ, mag, mag_sigma
        )
        if not bool(torch.isfinite(mag).all()):
            raise ValueError("mag must contain only finite values")
        if not bool(torch.isfinite(mag_sigma).all()) or bool((mag_sigma < 0).any()):
            raise ValueError("mag_sigma must contain finite, non-negative values")
        slope = float(self.tf_calc.slope)
        if not math.isfinite(slope) or slope == 0.0:
            raise ValueError("TF slope must be finite and non-zero")
        if not math.isfinite(self.vcirc_dex) or self.vcirc_dex <= 0.0:
            raise ValueError("TF intrinsic scatter must be finite and positive")
        if not (0.0 < self.vcirc_min < self.vcirc_max):
            raise ValueError("vcirc bounds must satisfy 0 < min < max")

        mean_log10 = (mag - float(self.tf_calc.intercept)) / slope
        sigma_log10 = torch.sqrt(
            torch.full_like(mag_sigma, self.vcirc_dex ** 2)
            + (mag_sigma / slope).square()
        )
        lower = (math.log10(self.vcirc_min) - mean_log10) / sigma_log10
        upper = (math.log10(self.vcirc_max) - mean_log10) / sigma_log10
        log_truncation_mass = self._log_standard_normal_interval(lower, upper)
        on_support = (
            torch.isfinite(v_circ)
            & (v_circ >= self.vcirc_min)
            & (v_circ <= self.vcirc_max)
        )
        safe_v = torch.where(on_support, v_circ, torch.ones_like(v_circ))
        standardized = (torch.log10(safe_v) - mean_log10) / sigma_log10
        log_density = (
            -0.5 * standardized.square()
            - torch.log(sigma_log10)
            - 0.5 * math.log(2.0 * math.pi)
            - log_truncation_mass
            - torch.log(safe_v)
            - math.log(math.log(10.0))
        )
        return torch.where(
            on_support, log_density, torch.full_like(log_density, -torch.inf)
        )

    @staticmethod
    def _per_galaxy_observation(value, batch_size, samples, name):
        value = torch.as_tensor(
            value, device=samples.device, dtype=samples.dtype
        ).reshape(-1)
        if value.numel() == 1 and batch_size != 1:
            value = value.expand(batch_size)
        if value.numel() != batch_size:
            raise ValueError(
                f"{name} must be scalar or contain one value per galaxy; "
                f"got {value.numel()} values for batch size {batch_size}"
            )
        return value

    def tf_prior_replacement_weights(self, samples, mag, mag_sigma):
        """Return per-galaxy pi_TF(v|m)/pi_0(v) weights for joint draws."""
        if samples.ndim == 2:
            samples = samples.unsqueeze(0)
        if samples.ndim != 3 or samples.shape[-1] != self.nfeatures:
            raise ValueError("samples must have shape (B, N, nfeatures)")
        if samples.shape[1] <= 0:
            raise ValueError("each galaxy must have at least one candidate sample")
        batch_size = samples.shape[0]
        mag = self._per_galaxy_observation(mag, batch_size, samples, "mag")
        mag_sigma = self._per_galaxy_observation(
            mag_sigma, batch_size, samples, "mag_sigma"
        )
        v_norm = samples[..., self.vcirc_idx]
        v_circ = self.vcirc_min + 0.5 * (v_norm + 1.0) * (
            self.vcirc_max - self.vcirc_min
        )
        log_tf = self.tf_prior_log_prob(
            v_circ, mag[:, None], mag_sigma[:, None]
        )
        log_prior_ratio = log_tf + math.log(self.vcirc_max - self.vcirc_min)
        finite = torch.isfinite(log_prior_ratio)
        valid_rows = finite.any(dim=1)
        if not bool(valid_rows.all()):
            invalid = torch.nonzero(~valid_rows, as_tuple=False).flatten().tolist()
            raise RuntimeError(
                "TF prior replacement has no finite-weight candidates for "
                f"galaxy indices {invalid}; increase the candidate bank"
            )
        safe_log_ratio = torch.where(
            finite,
            log_prior_ratio,
            torch.full_like(log_prior_ratio, -torch.inf),
        )
        weight_dtype = (
            torch.float32
            if safe_log_ratio.dtype in (torch.float16, torch.bfloat16)
            else safe_log_ratio.dtype
        )
        weights = torch.softmax(safe_log_ratio.to(weight_dtype), dim=1)
        if not bool(torch.isfinite(weights).all()):
            raise RuntimeError("TF prior replacement produced non-finite weights")
        ess = weights.sum(dim=1).square() / weights.square().sum(dim=1)
        diagnostics = {
            "effective_sample_size": ess.detach(),
            "effective_sample_fraction": (ess / samples.shape[1]).detach(),
            "max_normalized_weight": weights.max(dim=1).values.detach(),
            "candidate_log_normalizer": torch.logsumexp(
                safe_log_ratio, dim=1
            ).detach(),
        }
        return weights, safe_log_ratio, diagnostics

    def _compute_tf_weights(self, true, mag, snr):
        """
        Compute TF importance weights normalized over the global DDP batch.

        Rank-local normalization gives different effective objectives on each
        replica and makes the result depend on the rank partition. The detached
        log weights are therefore stabilized by a global maximum, then reduced
        as global sum/sum-of-squares/count statistics. The stored diagnostics
        make the very peaky effective weighting visible during training.
        """
        if mag is None:
            weights = torch.ones(
                true.size(0), device=true.device, dtype=true.dtype
            )
            global_count = weights.new_tensor(float(weights.numel()))
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            self.last_tf_diagnostics = {
                "effective_sample_size": global_count.detach(),
                "effective_sample_fraction": weights.new_tensor(1.0),
                "max_normalized_weight": weights.new_tensor(1.0),
            }
            return weights
            
        v_norm = true[:, self.vcirc_idx]
        v_circ = self._norm_to_vcirc(v_norm)
        
        # Regularize non-finite elements
        min_val = 1.0
        v_circ = torch.where(
            torch.isfinite(v_circ) & (v_circ > 0),
            v_circ,
            torch.full_like(v_circ, min_val),
        )
        
        # Compute prior parameters
        vcirc_mu, sigma_total_ln = self._get_tf_prior_params(mag, snr)
        
        # Match devices and dtypes
        vcirc_mu = vcirc_mu.to(device=v_circ.device, dtype=v_circ.dtype)
        sigma_total_ln = sigma_total_ln.to(device=v_circ.device, dtype=v_circ.dtype)
        
        # Setup the physical LogNormal distribution
        prior = torch.distributions.LogNormal(
            loc=torch.log(vcirc_mu),
            scale=sigma_total_ln
        )
        
        # Calculate the physical log prob and apply change of variables back to normalized [-1, 1] space
        log_prob_physical = prior.log_prob(v_circ)
        log_jacobian = math.log(self.vcirc_jac)
        log_prob_tf = log_prob_physical + log_jacobian
        
        # Convert to importance weights without overflow and normalize across
        # all DDP ranks, not independently within each rank.
        log_weights = log_prob_tf.detach()
        log_weights = torch.where(
            torch.isfinite(log_weights),
            log_weights,
            torch.full_like(log_weights, -torch.inf),
        )
        global_max = log_weights.max()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        if not torch.isfinite(global_max):
            weights = torch.ones_like(log_weights)
            global_count = weights.new_tensor(float(weights.numel()))
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            self.last_tf_diagnostics = {
                "effective_sample_size": global_count.detach(),
                "effective_sample_fraction": weights.new_tensor(1.0),
                "max_normalized_weight": weights.new_tensor(1.0),
            }
            return weights

        scaled_weights = torch.exp(log_weights - global_max)
        statistics = torch.stack(
            (
                scaled_weights.sum(),
                scaled_weights.square().sum(),
                scaled_weights.new_tensor(float(scaled_weights.numel())),
            )
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        global_sum, global_sum_sq, global_count = statistics
        global_mean = global_sum / global_count.clamp_min(1.0)
        weights = scaled_weights / global_mean.clamp_min(
            torch.finfo(scaled_weights.dtype).tiny
        )
        effective_sample_size = global_sum.square() / global_sum_sq.clamp_min(
            torch.finfo(scaled_weights.dtype).tiny
        )
        self.last_tf_diagnostics = {
            "effective_sample_size": effective_sample_size.detach(),
            "effective_sample_fraction": (
                effective_sample_size / global_count.clamp_min(1.0)
            ).detach(),
            "max_normalized_weight": (1.0 / global_mean).detach(),
        }
        return weights

    def _kde_log_density_1d(self, values):
        """
        Gaussian KDE log-density estimate at sample locations.
        values: (N,)
        returns: (N,)
        """
        n = values.shape[0]
        if n < 2:
            return torch.zeros_like(values)

        std = values.std(unbiased=False).clamp(min=1e-6)
        bandwidth = (1.06 * std * (float(n) ** (-1.0 / 5.0))).clamp(min=1e-6)

        diffs = (values[:, None] - values[None, :]) / bandwidth
        log_norm = torch.log(torch.tensor(math.sqrt(2.0 * math.pi), device=values.device, dtype=values.dtype) * bandwidth)
        log_kernel = -0.5 * diffs.pow(2) - log_norm
        return torch.logsumexp(log_kernel, dim=1) - torch.log(torch.tensor(float(n), device=values.device, dtype=values.dtype))

    def setup_flows(self):
        """Set up the selected conditional posterior flow.

        The circular construction moves ``theta_int`` to the final
        autoregressive coordinate once. Every later permutation leaves it
        there, so Euclidean coordinates never condition on an unwrapped angle,
        while theta can still condition on all seven Euclidean parameters.
        """
        num_layers = int(config.flow['num_layers'])
        if num_layers <= 0:
            raise ValueError("flow num_layers must be positive")
        hidden_units = 256
        num_blocks = 2
        context_size = self.flow_context_features

        self.flow_type = str(config.flow.get('flow_type', 'affine')).lower()
        if self.flow_type not in FLOW_TYPES:
            raise ValueError(
                f"flow_type must be one of {FLOW_TYPES}; got {self.flow_type!r}"
            )
        self.theta_idx = resolve_feature_index(self.feature_names, "theta_int")

        if self.flow_type == "bounded_hybrid_circular":
            if self.nfeatures < 2:
                raise ValueError(
                    "bounded_hybrid_circular requires at least two parameters"
                )
            self.flow = BoundedHybridCircularFlow(
                features=self.nfeatures,
                theta_index=self.theta_idx,
                context_features=context_size,
                num_bounded_layers=num_layers,
                num_theta_layers=config.flow.get("theta_num_layers", 1),
                num_bins=config.flow.get("num_bins", 8),
                hidden_features=hidden_units,
                num_blocks=num_blocks,
                logit_limit=config.flow.get("theta_logit_limit", 10.0),
                bounded_logit_limit=config.flow.get(
                    "bounded_logit_limit", 10.0
                ),
            )
            return

        if self.flow_type == "hybrid_circular":
            if self.nfeatures < 2:
                raise ValueError(
                    "hybrid_circular requires at least two parameters"
                )
            self.flow = HybridAffineCircularFlow(
                features=self.nfeatures,
                theta_index=self.theta_idx,
                context_features=context_size,
                num_affine_layers=num_layers,
                num_theta_layers=config.flow.get("theta_num_layers", 1),
                num_bins=config.flow.get("num_bins", 8),
                hidden_features=hidden_units,
                num_blocks=num_blocks,
                logit_limit=config.flow.get("theta_logit_limit", 10.0),
            )
            return

        if self.flow_type == "affine":
            # Preserve the historical architecture and state-dict layout.
            self.base = ConditionalDiagonalNormal(
                shape=[self.nfeatures],
                context_encoder=MLP(
                    [context_size, 128, 64, self.nfeatures * 2]
                ),
            )
            transforms = []
            for _ in range(num_layers):
                transforms.append(ReversePermutation(features=self.nfeatures))
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=self.nfeatures,
                        hidden_features=hidden_units,
                        num_blocks=num_blocks,
                        context_features=context_size,
                    )
                )
            self.transform = CompositeTransform(transforms)
            return

        if self.nfeatures < 2:
            raise ValueError("circular_rqs requires at least two parameters")
        num_bins = config.flow.get('num_bins', 8)
        if type(num_bins) is not int or num_bins < 2:
            raise ValueError("flow num_bins must be an integer of at least two")

        non_theta = [
            index for index in range(self.nfeatures) if index != self.theta_idx
        ]
        canonical_order = non_theta + [self.theta_idx]
        internal_theta_idx = self.nfeatures - 1
        scalar_reverse = list(reversed(range(internal_theta_idx))) + [
            internal_theta_idx
        ]
        self.circular_internal_theta_idx = internal_theta_idx
        self.circular_boundary_permutation = tuple(canonical_order)
        self.circular_layer_permutation = tuple(scalar_reverse)

        transforms = [
            Permutation(torch.tensor(canonical_order, dtype=torch.long))
        ]
        for _ in range(num_layers):
            # Unlike the old implementation, this permutation cannot move
            # theta away from the index declared circular in the spline.
            transforms.append(
                Permutation(torch.tensor(scalar_reverse, dtype=torch.long))
            )
            transforms.append(
                CircularAutoregressiveRationalQuadraticSpline(
                    num_input_channels=self.nfeatures,
                    num_blocks=num_blocks,
                    num_hidden_channels=hidden_units,
                    ind_circ=[internal_theta_idx],
                    num_context_channels=context_size,
                    num_bins=num_bins,
                    tail_bound=1.0,
                    identity_init=True,
                )
            )

        self.base = ConditionalNormalWithCircularTheta(
            features=self.nfeatures,
            context_encoder=MLP(
                [context_size, 128, 64, 2 * (self.nfeatures - 1)]
            ),
        )
        self.transform = CompositeTransform(transforms)

    def _draw_flow_samples(
        self,
        num_samples,
        context,
        *,
        sample_id=None,
        canonical_theta=False,
    ):
        """Draw finite samples for every context row, with bounded-coordinate safety."""
        max_tries = 5
        bad_samples_tolerance = 0.75
        theta_idx = getattr(
            self,
            "theta_idx",
            resolve_feature_index(self.feature_names, "theta_int"),
        )
        last_bad = None
        for attempt in range(max_tries):
            samples = self.flow.sample(num_samples, context=context)
            is_circular = getattr(self, "flow_type", "affine") in (
                "circular_rqs",
                "hybrid_circular",
                "bounded_hybrid_circular",
            )
            is_bounded = (
                getattr(self, "flow_type", "affine")
                == "bounded_hybrid_circular"
            )
            if is_circular:
                # A circular sample is an equivalence class modulo two in the
                # theta/pi coordinate. Always return its canonical [-1, 1)
                # representative; unlike clipping, this preserves the angle.
                samples = samples.clone()
                theta = samples[..., theta_idx]
                outside = torch.isfinite(theta) & (
                    (theta < -1.0) | (theta >= 1.0)
                )
                wrapped = torch.remainder(theta + 1.0, 2.0) - 1.0
                samples[..., theta_idx] = torch.where(outside, wrapped, theta)
            nonperiodic = [
                index
                for index in range(self.nfeatures)
                if index != theta_idx
            ]
            if is_bounded:
                bounded = samples[..., nonperiodic]
                on_support = (
                    torch.isfinite(bounded)
                    & (bounded >= -1.0)
                    & (bounded <= 1.0)
                ).all(dim=-1)
                if not bool(on_support.all()):
                    count = int((~on_support).sum().item())
                    raise RuntimeError(
                        "bounded_hybrid_circular produced "
                        f"{count} non-theta rows outside [-1, 1] during "
                        "KLNPE sampling; refusing to clamp"
                    )
            if canonical_theta:
                samples = samples.clone()
                if not is_bounded:
                    samples[..., nonperiodic] = samples[
                        ..., nonperiodic
                    ].clamp(-1.5, 1.5)
                if not is_circular:
                    samples[..., theta_idx] = torch.remainder(
                        samples[..., theta_idx] + 1.0, 2.0
                    ) - 1.0
            else:
                samples = samples.clone()
                if not is_bounded:
                    samples[..., nonperiodic] = samples[
                        ..., nonperiodic
                    ].clamp(-1.5, 1.5)
                if not is_circular:
                    samples[..., theta_idx] = samples[..., theta_idx].clamp(-1.5, 1.5)

            finite = torch.isfinite(samples).all(dim=-1)
            bad_count = int((~finite).sum().item())
            total_count = finite.numel()
            last_bad = (bad_count, total_count)
            if bad_count / total_count > bad_samples_tolerance:
                logging.warning(
                    "Sampling for %s produced %d/%d non-finite samples; "
                    "retrying (remaining=%d)",
                    sample_id,
                    bad_count,
                    total_count,
                    max_tries - attempt - 1,
                )
                continue
            if bad_count:
                repaired = samples.clone()
                repair_failed = False
                for context_index in range(samples.shape[0]):
                    valid_indices = torch.nonzero(
                        finite[context_index], as_tuple=False
                    ).flatten()
                    invalid_indices = torch.nonzero(
                        ~finite[context_index], as_tuple=False
                    ).flatten()
                    if invalid_indices.numel() and not valid_indices.numel():
                        repair_failed = True
                        break
                    if invalid_indices.numel():
                        choices = valid_indices[
                            torch.randint(
                                valid_indices.numel(),
                                (invalid_indices.numel(),),
                                device=samples.device,
                            )
                        ]
                        repaired[context_index, invalid_indices] = samples[
                            context_index, choices
                        ]
                if repair_failed:
                    continue
                samples = repaired
            return samples
        raise RuntimeError(
            f"Sampling for {sample_id} failed after {max_tries} attempts; "
            f"last non-finite count was {last_bad}"
        )

    def _apply_tf_prior_replacement(self, samples, mag, mag_sigma):
        """Resample complete joint rows using the explicit TF/base-prior ratio."""
        weights, log_prior_ratio, diagnostics = (
            self.tf_prior_replacement_weights(samples, mag, mag_sigma)
        )
        indices = torch.multinomial(
            weights, num_samples=samples.shape[1], replacement=True
        )
        joint_indices = indices.unsqueeze(-1).expand(-1, -1, samples.shape[-1])
        resampled = torch.gather(samples, dim=1, index=joint_indices)
        selected_log_ratio = torch.gather(log_prior_ratio, dim=1, index=indices)
        self.last_tf_inference_diagnostics = diagnostics
        return resampled, selected_log_ratio

    def _apply_tf_resampling(self, samples, mag, snr):
        """Apply the existing per-galaxy TF replacement to a candidate bank."""
        if samples.shape[0] != 1:
            raise ValueError("TF inference resampling currently requires batch size 1")
        if mag is None or snr is None:
            raise ValueError("mode 2 sampling requires both mag and snr")
        candidates = samples[0]
        v_circ = self._norm_to_vcirc(candidates[:, self.vcirc_idx])
        mag_tensor = torch.as_tensor(
            mag, device=candidates.device, dtype=candidates.dtype
        ).reshape(-1)
        snr_tensor = torch.as_tensor(
            snr, device=candidates.device, dtype=candidates.dtype
        ).reshape(-1)
        vcirc_mu, sigma_total_ln = self._get_tf_prior_params(mag_tensor, snr_tensor)
        prior = torch.distributions.LogNormal(
            loc=torch.log(vcirc_mu[0]),
            scale=torch.full_like(v_circ, sigma_total_ln[0]),
        )
        tf_log_p_v = prior.log_prob(v_circ)
        flow_log_p_v = self._kde_log_density_1d(v_circ)
        log_w = tf_log_p_v - flow_log_p_v
        safe_log_w = log_w.float()
        finite = torch.isfinite(safe_log_w)
        fallback = not bool(finite.any())
        if not fallback:
            safe_log_w = torch.where(
                finite,
                safe_log_w,
                torch.full_like(safe_log_w, -torch.inf),
            )
            maximum = safe_log_w.max()
            fallback = not bool(torch.isfinite(maximum))
        if not fallback:
            weights = torch.softmax(safe_log_w - maximum, dim=0)
            fallback = not bool(torch.isfinite(weights).all()) or not bool(weights.sum() > 0)
        if fallback:
            logging.warning(
                "Mode 2 sampling: invalid log-weights; falling back to uniform resampling."
            )
            weights = torch.full(
                (candidates.shape[0],),
                1.0 / candidates.shape[0],
                device=candidates.device,
                dtype=torch.float32,
            )
            log_w = torch.zeros_like(log_w)
        else:
            log_w = torch.where(
                torch.isfinite(log_w),
                log_w,
                torch.full_like(log_w, -torch.inf),
            )
        indices = torch.multinomial(
            weights, num_samples=candidates.shape[0], replacement=True
        )
        return candidates[indices].unsqueeze(0), log_w[indices]

    def _d4_sample_from_features(
        self,
        raw_features,
        num_samples,
        sample_id=None,
        prepared_observation_context=None,
    ):
        if raw_features.shape[0] != 1:
            raise ValueError("D4 posterior sampling currently requires one galaxy")
        if num_samples <= 0 or num_samples % len(D4_ELEMENTS):
            raise ValueError("num_samples must be positive and divisible by 8")
        contexts = self._d4_contexts_from_features(
            raw_features, prepared_observation_context
        )[:, 0]
        per_component = num_samples // len(D4_ELEMENTS)
        branch_samples = self._draw_flow_samples(
            per_component,
            contexts,
            sample_id=sample_id,
            canonical_theta=True,
        )
        aligned = tuple(
            transform_d4_parameters(
                branch_samples[index],
                D4_INVERSES[element],
                feature_names=self.feature_names,
            )
            for index, element in enumerate(D4_ELEMENTS)
        )
        return torch.cat(aligned, dim=0).unsqueeze(0)

    def _d4_sample_log_prob(
        self,
        raw_features,
        samples,
        chunk_size=256,
        prepared_observation_context=None,
    ):
        if raw_features.shape[0] != 1 or samples.shape[0] != 1:
            raise ValueError("D4 sample scoring currently requires one galaxy")
        scores = []
        for start in range(0, samples.shape[1], chunk_size):
            candidates = samples[0, start : start + chunk_size]
            candidate_features = raw_features.expand(candidates.shape[0], -1)
            candidate_observation_context = (
                None
                if prepared_observation_context is None
                else prepared_observation_context.expand(
                    candidates.shape[0], -1
                )
            )
            scores.append(
                self._d4_mixture_log_prob_from_features(
                    candidate_features,
                    candidates,
                    candidate_observation_context,
                )
            )
        return torch.cat(scores, dim=0)

    def sample(
        self,
        x,
        y,
        num_samples,
        fp,
        mag=None,
        snr=None,
        return_log_prob=False,
        log_context=None,
        sample_id=None,
        tf_inference=None,
        mag_sigma=None,
        observation_context=None,
    ):
        """Sample one galaxy's configured conditional posterior.

        The D4 posterior uses a balanced eight-component mixture. Samples drawn
        in each transformed frame are mapped back to the input frame before TF
        replacement and before returning. ``tf_inference='prior_replacement'``
        applies the explicit pi_TF/pi_0 ratio to a mode-1 candidate bank and
        requires an observed magnitude plus ``mag_sigma``. ``log_context``
        is retained only for backward call compatibility.
        """
        del log_context
        if self.mode == 0:
            raise RuntimeError("sample requires density mode 1 or 2")
        if x.shape[0] != 1:
            raise ValueError("sample currently requires a single-galaxy batch")
        raw_features = self.feature_extractor(x, y, fp)
        prepared_observation_context = self._prepare_observation_context(
            observation_context, raw_features.shape[0], raw_features
        )

        if self.posterior_symmetry == "d4":
            samples = self._d4_sample_from_features(
                raw_features,
                num_samples,
                sample_id=sample_id,
                prepared_observation_context=prepared_observation_context,
            )
        else:
            context = self._flow_context_from_features(
                raw_features, prepared_observation_context
            )
            samples = self._draw_flow_samples(
                num_samples,
                context,
                sample_id=sample_id,
                canonical_theta=False,
            )

        if tf_inference not in (None, "prior_replacement"):
            raise ValueError(
                "tf_inference must be None or 'prior_replacement'; "
                f"got {tf_inference!r}"
            )
        tf_log_correction = None
        if tf_inference == "prior_replacement":
            if self.mode != 1:
                raise ValueError(
                    "TF prior replacement requires a mode-1 base posterior; "
                    "using it with mode 2 would apply TF information twice"
                )
            if mag is None or mag_sigma is None:
                raise ValueError(
                    "TF prior replacement requires both mag and mag_sigma"
                )
            samples, tf_log_correction = self._apply_tf_prior_replacement(
                samples, mag, mag_sigma
            )
        elif self.mode == 2:
            # Backward-compatible legacy mode-2 behavior/checkpoints.
            samples, tf_log_correction = self._apply_tf_resampling(
                samples, mag, snr
            )

        if not return_log_prob:
            return samples.reshape(1, num_samples, self.nfeatures)

        if self.posterior_symmetry == "d4":
            flow_log_prob = self._d4_sample_log_prob(
                raw_features,
                samples,
                prepared_observation_context=prepared_observation_context,
            )
        else:
            context = self._flow_context_from_features(
                raw_features, prepared_observation_context
            ).expand(num_samples, -1)
            flow_log_prob = self.flow.log_prob(
                samples.reshape(num_samples, self.nfeatures),
                context=context,
            )
        if tf_log_correction is not None:
            flow_log_prob = flow_log_prob + tf_log_correction.reshape(-1)
        return samples.reshape(1, num_samples, self.nfeatures), flow_log_prob

    def evaluate_conditional_2d(
        self,
        x,
        y,
        true_params,
        idx1,
        idx2,
        fp=None,
        grid_bins=200,
        bounds=(-1, 1),
        observation_context=None,
    ):
        '''
        Diagnostic: Sample 2 parameters conditional on all other parameters being fixed.
        Evaluates the flow log_prob over a 2D grid and samples from the resulting PDF.
        
        true_params: Tensor of shape (1, nfeatures) containing the fixed parameter values.
        idx1, idx2: Integers representing the parameter indices for g1 and g2.
        '''
        # 1. Extract context 'z' identically to your sample() function
        raw_features = self.feature_extractor(x, y, fp)
        prepared_observation_context = self._prepare_observation_context(
            observation_context, raw_features.shape[0], raw_features
        )
        z = self._flow_context_from_features(
            raw_features, prepared_observation_context
        )
        
        # 2. Create a 2D grid for the two parameters
        g1_vals = torch.linspace(bounds[0], bounds[1], grid_bins, device=z.device)
        g2_vals = torch.linspace(bounds[0], bounds[1], grid_bins, device=z.device)
        G1, G2 = torch.meshgrid(g1_vals, g2_vals, indexing='ij')
        
        flat_g1 = G1.flatten()
        flat_g2 = G2.flatten()
        num_grid_points = flat_g1.size(0)
        
        # 3. Prepare the massive batch of parameter vectors
        # Clone the true_params vector for every point on the grid
        theta_grid = true_params.repeat(num_grid_points, 1)
        
        # Overwrite the g1 and g2 columns with the grid points
        theta_grid[:, idx1] = flat_g1
        theta_grid[:, idx2] = flat_g2
        
        # Repeat the context vector to match the grid size
        z_rep = z.repeat(num_grid_points, 1)
        
        # 4. Evaluate log probabilities for the entire grid in one forward pass
        log_probs = self.flow.log_prob(theta_grid, context=z_rep)
        
        # 5. Convert to normalized probabilities 
        # (Subtract max for numerical stability before exp to avoid overflow)
        probs = torch.exp(log_probs - log_probs.max())
        probs = probs / probs.sum()
        
        return probs.view(grid_bins, grid_bins), g1_vals, g2_vals

class FeatureExtractor(nn.Module):
    def __init__(self, nspec=config.data['nspec']):
        super().__init__()
        self.nspecs = nspec
        
        # Vision Transformer for images
        # self.img_net = VisionTransformer(
        #     in_channels=1, embed_dim=512, img_size=48, patch_size=6, 
        #     num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1
        # )
        self.img_net = ImgCNN()

        # CNN for spectra
        self.spec_net = SpecCNN(self.nspecs)

    def forward(self, x, y, fp):
        '''Extracts raw features independently for both modalities'''
        x = nn.functional.normalize(x, dim=[2, 3])
        y = nn.functional.normalize(y, dim=[2, 3])
        # fp = nn.functional.normalize(fp, dim=[1, 2])
        fp = fp / 1.5
        fp = fp.view(fp.size(0), -1)
        
        img_feats = self.img_net(x).view(x.size(0), -1)     # Shape: (B, 512)
        spec_feats = self.spec_net(y).view(y.size(0), -1)   # Shape: (B, 512 - 2*nspecs)
        spec_feats = torch.cat((spec_feats, fp), dim=-1)     # Shape: (B, 512)
        
        z = torch.cat((img_feats, spec_feats), -1)

        return z
    

BACKBONE_TYPES = ("legacy", "stage3", "stage4_d4")


class DivisibleMeanPool1d(nn.Module):
    """Deterministic adaptive-mean equivalent for evenly divisible bins."""

    def __init__(self, output_size):
        super().__init__()
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        self.output_size = int(output_size)

    def forward(self, inputs):
        input_size = inputs.shape[-1]
        if input_size < self.output_size or input_size % self.output_size:
            raise ValueError(
                "spectral feature length must be a positive multiple of "
                f"{self.output_size}; got {input_size}"
            )
        bin_size = input_size // self.output_size
        return inputs.reshape(
            *inputs.shape[:-1], self.output_size, bin_size
        ).mean(dim=-1)


class SharedSpecCNN(nn.Module):
    """Encode every fiber spectrum with the same wavelength-aware Conv1d network."""

    def __init__(self, embedding_dim=128, pooled_length=8):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if pooled_length <= 0:
            raise ValueError("pooled_length must be positive")
        self.embedding_dim = int(embedding_dim)
        self.pooled_length = int(pooled_length)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            # For the 64-bin spectra this maps 16 -> 8 exactly like adaptive
            # average pooling, but its CUDA backward is deterministic.
            DivisibleMeanPool1d(self.pooled_length),
        )
        # Flattening the reduced wavelength axis, rather than globally averaging it,
        # preserves absolute line-position information needed for radial velocity.
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * self.pooled_length, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
        )

    def forward(self, spectra):
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, fibers, wavelength)")
        batch_size, _, fiber_count, wavelength_count = spectra.shape
        if wavelength_count < 4:
            raise ValueError("spectra must contain at least four wavelength samples")
        shared_input = spectra[:, 0].reshape(
            batch_size * fiber_count, 1, wavelength_count
        )
        encoded = self.projection(self.encoder(shared_input))
        return encoded.reshape(batch_size, fiber_count, self.embedding_dim)


class FiberSetAttention(nn.Module):
    """Position-free self-attention over a set of physical fiber tokens."""

    def __init__(self, token_dim=128, num_heads=4, feedforward_dim=256):
        super().__init__()
        if token_dim <= 0 or token_dim % num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads")
        if feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        self.self_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(token_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(token_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, token_dim),
        )
        self.feedforward_norm = nn.LayerNorm(token_dim)

    def forward(self, tokens, observed_mask, key_padding_mask=None):
        if tokens.ndim != 3:
            raise ValueError("tokens must have shape (batch, fibers, token_dim)")
        if observed_mask.shape != tokens.shape[:2]:
            raise ValueError("observed_mask shape must match the token batch and fibers")
        if observed_mask.dtype != torch.bool:
            raise TypeError("observed_mask must be a bool tensor")
        if key_padding_mask is not None:
            if key_padding_mask.shape != observed_mask.shape:
                raise ValueError("key_padding_mask shape must match observed_mask")
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be a bool tensor")

        attended, _ = self.self_attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        observed = observed_mask.unsqueeze(-1).to(dtype=tokens.dtype)
        tokens = self.attention_norm(tokens + attended) * observed
        tokens = self.feedforward_norm(tokens + self.feedforward(tokens)) * observed
        return tokens


class Stage3FeatureExtractor(nn.Module):
    """Permutation-aware image/spectra fusion for the Stage 3 CCL experiment."""

    output_dim = 1024

    def __init__(
        self,
        nspec=config.data['nspec'],
        spectral_embedding_dim=128,
        token_dim=128,
        num_heads=4,
        fiber_position_scale=1.5,
        img_net=None,
    ):
        super().__init__()
        if nspec <= 0:
            raise ValueError("nspec must be positive")
        if fiber_position_scale <= 0 or not math.isfinite(fiber_position_scale):
            raise ValueError("fiber_position_scale must be positive and finite")
        self.nspecs = int(nspec)
        self.fiber_position_scale = float(fiber_position_scale)
        self.img_net = ImgCNN() if img_net is None else img_net
        self.spec_net = SharedSpecCNN(embedding_dim=spectral_embedding_dim)

        # Coordinates contain spin-1 (x,y), scalar (r^2), and spin-2
        # (x^2-y^2,2xy) components. There is deliberately no storage-index
        # embedding: physical positions, spectra, and masks define each token.
        coordinate_dim = 5
        spectral_strength_dim = 1
        observation_dim = 1
        self.token_projection = nn.Sequential(
            nn.Linear(
                spectral_embedding_dim
                + coordinate_dim
                + spectral_strength_dim
                + observation_dim,
                token_dim,
            ),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.fiber_set_encoder = FiberSetAttention(
            token_dim=token_dim,
            num_heads=num_heads,
            feedforward_dim=2 * token_dim,
        )
        self.image_query = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, token_dim),
        )
        self.image_fiber_attention = nn.MultiheadAttention(
            token_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.attended_norm = nn.LayerNorm(token_dim)
        self.fiber_projection = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )
        self.fusion_norm = nn.LayerNorm(self.output_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.GELU(),
            nn.Linear(512, self.output_dim),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    @staticmethod
    def _validate_inputs(image, spectra, fiber_positions, fiber_mask):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must have shape (batch, 1, height, width)")
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError("spectra must have shape (batch, 1, fibers, wavelength)")
        if fiber_positions.ndim != 3 or fiber_positions.shape[-1] != 2:
            raise ValueError("fiber_positions must have shape (batch, fibers, 2)")
        if image.shape[0] != spectra.shape[0] or image.shape[0] != fiber_positions.shape[0]:
            raise ValueError("image, spectra, and fiber_positions batch sizes must match")
        if spectra.shape[2] != fiber_positions.shape[1]:
            raise ValueError("spectra and fiber_positions must have the same fiber count")
        if fiber_mask is not None:
            if fiber_mask.dtype != torch.bool:
                raise TypeError("fiber_mask must be a bool tensor")
            if fiber_mask.shape != spectra.shape[:1] + spectra.shape[2:3]:
                raise ValueError("fiber_mask must have shape (batch, fibers)")

    @staticmethod
    def _coordinate_features(fiber_positions):
        x_coord, y_coord = fiber_positions.unbind(dim=-1)
        return torch.stack(
            (
                x_coord,
                y_coord,
                x_coord.square() + y_coord.square(),
                x_coord.square() - y_coord.square(),
                2.0 * x_coord * y_coord,
            ),
            dim=-1,
        )

    @staticmethod
    def _relative_spectral_strength(spectra):
        """Return per-fiber L2 norms relative to the full spectral datavector."""
        fiber_norms = torch.linalg.vector_norm(spectra, dim=-1, keepdim=True)
        total_norm = torch.linalg.vector_norm(fiber_norms, dim=2, keepdim=True)
        return fiber_norms / total_norm.clamp_min(torch.finfo(spectra.dtype).tiny)

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        self._validate_inputs(image, spectra, fiber_positions, fiber_mask)
        batch_size, _, fiber_count, _ = spectra.shape
        explicit_fiber_mask = fiber_mask is not None
        if fiber_mask is None:
            fiber_mask = torch.ones(
                (batch_size, fiber_count),
                dtype=torch.bool,
                device=spectra.device,
            )
        else:
            fiber_mask = fiber_mask.to(device=spectra.device)
            # Explicit variable-fiber batches require at least one real token.
            # The all-observed training path skips this host-visible check.
            if torch.any(~fiber_mask.any(dim=1)):
                raise ValueError("every sample must contain at least one observed fiber")
        key_padding_mask = ~fiber_mask if explicit_fiber_mask else None

        spectral_mask = fiber_mask[:, None, :, None]
        position_mask = fiber_mask.unsqueeze(-1)
        spectra = torch.where(spectral_mask, spectra, torch.zeros_like(spectra))
        fiber_positions = torch.where(
            position_mask,
            fiber_positions,
            torch.zeros_like(fiber_positions),
        )

        image = F.normalize(image, dim=(-2, -1))
        # Per-fiber normalization isolates line shape, while the relative norm
        # retains all amplitude information present under the legacy global
        # spectral normalization.
        relative_spectral_strength = self._relative_spectral_strength(spectra)
        spectra = F.normalize(spectra, dim=-1)
        normalized_positions = fiber_positions / self.fiber_position_scale

        image_features = self.img_net(image).reshape(batch_size, -1)
        if image_features.shape[1] != 512:
            raise ValueError(
                f"image encoder must return 512 features; got {image_features.shape[1]}"
            )
        spectral_features = self.spec_net(spectra)
        coordinate_features = self._coordinate_features(normalized_positions)
        spectral_strength_feature = relative_spectral_strength[:, 0].to(
            spectral_features.dtype
        )
        observation_feature = fiber_mask.unsqueeze(-1).to(spectral_features.dtype)
        fiber_tokens = self.token_projection(
            torch.cat(
                (
                    spectral_features,
                    coordinate_features,
                    spectral_strength_feature,
                    observation_feature,
                ),
                dim=-1,
            )
        )
        fiber_tokens = fiber_tokens * observation_feature
        fiber_tokens = self.fiber_set_encoder(
            fiber_tokens,
            fiber_mask,
            key_padding_mask=key_padding_mask,
        )

        image_query = self.image_query(image_features).unsqueeze(1)
        attended_fibers, _ = self.image_fiber_attention(
            image_query,
            fiber_tokens,
            fiber_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_fibers = self.attended_norm(attended_fibers.squeeze(1))
        fiber_features = self.fiber_projection(attended_fibers)

        joint_features = torch.cat((image_features, fiber_features), dim=-1)
        fused_features = joint_features + self.fusion_mlp(
            self.fusion_norm(joint_features)
        )
        return self.output_norm(fused_features)


class D4OrbitFeatureExtractor(nn.Module):
    """Exactly D4-equivariant multimodal features from a shared orbit backbone.

    The raw 1024 channels are interpreted as scalar, directed spin-1, and
    spin-2 blocks. Every forward pass evaluates all eight complete datavector
    views in one shared-backbone batch, maps each output back to the input
    frame, and averages the aligned orbit.
    """

    output_dim = 1024
    default_scalar_channels = 512
    default_spin1_channels = 256
    default_spin2_channels = 256

    def __init__(
        self,
        nspec=config.data['nspec'],
        base_backbone=None,
        scalar_channels=default_scalar_channels,
        spin1_channels=default_spin1_channels,
        spin2_channels=default_spin2_channels,
    ):
        super().__init__()
        channel_counts = (scalar_channels, spin1_channels, spin2_channels)
        if any(type(value) is not int or value < 0 for value in channel_counts):
            raise ValueError("D4 feature channel counts must be non-negative integers")
        if spin1_channels % 2 or spin2_channels % 2:
            raise ValueError("spin-1 and spin-2 channel counts must be even")
        if sum(channel_counts) != self.output_dim:
            raise ValueError(
                f"D4 feature channel counts must sum to {self.output_dim}"
            )
        self.nspecs = int(nspec)
        self.scalar_channels = scalar_channels
        self.spin1_channels = spin1_channels
        self.spin2_channels = spin2_channels
        self.base_backbone = (
            Stage3FeatureExtractor(nspec=self.nspecs)
            if base_backbone is None
            else base_backbone
        )

    def transform_features(self, features, element):
        """Express equivariant features in the frame transformed by ``element``."""
        return transform_d4_feature_blocks(
            features,
            element,
            scalar_channels=self.scalar_channels,
            spin1_channels=self.spin1_channels,
            spin2_channels=self.spin2_channels,
        )

    def _build_orbit_batch(self, image, spectra, fiber_positions, fiber_mask=None):
        image_views = []
        spectrum_views = []
        position_views = []
        mask_views = []
        for element in D4_ELEMENTS:
            view_image, view_spectra, _, view_positions = apply_d4_to_datavector(
                image,
                spectra,
                fp=fiber_positions,
                element=element,
            )
            image_views.append(view_image)
            spectrum_views.append(view_spectra)
            position_views.append(view_positions)
            if fiber_mask is not None:
                mask_views.append(transform_d4_fiber_mask(fiber_mask, element))

        orbit_image = torch.cat(image_views, dim=0)
        orbit_spectra = torch.cat(spectrum_views, dim=0)
        orbit_positions = torch.cat(position_views, dim=0)
        orbit_mask = torch.cat(mask_views, dim=0) if mask_views else None
        if image.is_contiguous(memory_format=torch.channels_last):
            orbit_image = orbit_image.contiguous(memory_format=torch.channels_last)
        if spectra.is_contiguous(memory_format=torch.channels_last):
            orbit_spectra = orbit_spectra.contiguous(memory_format=torch.channels_last)
        return orbit_image, orbit_spectra, orbit_positions, orbit_mask

    def aligned_orbit_features(
        self,
        image,
        spectra,
        fiber_positions,
        fiber_mask=None,
    ):
        """Return all eight raw orbit features aligned to the input frame."""
        if fiber_positions is None:
            raise ValueError("fiber_positions are required for the D4 orbit backbone")
        batch_size = image.shape[0]
        orbit = self._build_orbit_batch(
            image,
            spectra,
            fiber_positions,
            fiber_mask=fiber_mask,
        )
        if orbit[3] is None:
            raw_features = self.base_backbone(orbit[0], orbit[1], orbit[2])
        else:
            raw_features = self.base_backbone(*orbit)
        expected_shape = (len(D4_ELEMENTS) * batch_size, self.output_dim)
        if raw_features.shape != expected_shape:
            raise ValueError(
                "D4 base backbone must return shape "
                f"{expected_shape}; got {tuple(raw_features.shape)}"
            )
        raw_features = raw_features.reshape(
            len(D4_ELEMENTS), batch_size, self.output_dim
        )
        return torch.stack(
            tuple(
                self.transform_features(raw_features[index], D4_INVERSES[element])
                for index, element in enumerate(D4_ELEMENTS)
            ),
            dim=0,
        )

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        return self.aligned_orbit_features(
            image,
            spectra,
            fiber_positions,
            fiber_mask=fiber_mask,
        ).mean(dim=0)



class _D4PairLinear(nn.Module):
    """Bias-free multiplicity mixing that preserves a two-component irrep."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        if input_channels % 2 or output_channels % 2:
            raise ValueError("D4 pair-linear channel counts must be even")
        self.input_pairs = input_channels // 2
        self.output_pairs = output_channels // 2
        self.weight = nn.Parameter(torch.empty(self.output_pairs, self.input_pairs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, features):
        pairs = features.reshape(*features.shape[:-1], self.input_pairs, 2)
        projected = torch.einsum("...ic,oi->...oc", pairs, self.weight)
        return projected.reshape(*features.shape[:-1], 2 * self.output_pairs)


class D4EquivariantCCLProjector(nn.Module):
    """Projection head whose scalar/spin blocks obey the D4 feature action."""

    def __init__(
        self,
        scalar_channels=512,
        spin1_channels=256,
        spin2_channels=256,
        output_dim=128,
    ):
        super().__init__()
        if output_dim <= 0 or output_dim % 8:
            raise ValueError("D4 CCL projector output_dim must be divisible by 8")
        self.scalar_channels = int(scalar_channels)
        self.spin1_channels = int(spin1_channels)
        self.spin2_channels = int(spin2_channels)
        self.output_scalar_channels = output_dim // 2
        self.output_spin1_channels = output_dim // 4
        self.output_spin2_channels = output_dim // 4
        hidden_channels = max(128, 4 * self.output_scalar_channels)
        self.scalar_projector = nn.Sequential(
            nn.Linear(self.scalar_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, self.output_scalar_channels),
        )
        self.spin1_projector = _D4PairLinear(
            self.spin1_channels, self.output_spin1_channels
        )
        self.spin2_projector = _D4PairLinear(
            self.spin2_channels, self.output_spin2_channels
        )

    def forward(self, features):
        expected = self.scalar_channels + self.spin1_channels + self.spin2_channels
        if features.shape[-1] != expected:
            raise ValueError(
                f"D4 CCL projector expected {expected} features; "
                f"got {features.shape[-1]}"
            )
        scalar_end = self.scalar_channels
        spin1_end = scalar_end + self.spin1_channels
        return torch.cat(
            (
                self.scalar_projector(features[..., :scalar_end]),
                self.spin1_projector(features[..., scalar_end:spin1_end]),
                self.spin2_projector(features[..., spin1_end:]),
            ),
            dim=-1,
        )


def build_feature_extractor(backbone_type, nspec=config.data['nspec']):
    if backbone_type == "legacy":
        return FeatureExtractor(nspec=nspec)
    if backbone_type == "stage3":
        return Stage3FeatureExtractor(nspec=nspec)
    if backbone_type == "stage4_d4":
        return D4OrbitFeatureExtractor(nspec=nspec)
    raise ValueError(
        f"backbone_type must be one of {BACKBONE_TYPES}; got {backbone_type!r}"
    )

class VICRegPretrain(nn.Module):
    def __init__(self, backbone=None, projector_dim=128):
        super().__init__()

        self.backbone = (
            build_feature_extractor(
                config.pretrain.get("backbone_type", "legacy"),
                nspec=config.data["nspec"],
            )
            if backbone is None
            else backbone
        )
        
        # Get feature dimensions dynamically
        dim_in = 1024
        
        # Projector networks mapping onto a common high-dimensional space
        self.projector = MLP([dim_in, 2048, 512, projector_dim], use_batchnorm=True, use_dropout=True)
        
        self.vicreg_loss = VICRegLoss(lam=20.0, mu=20.0, nu=5.0, gamma=1.0)

    def forward(self, x1, y1, fp1, x2, y2, fp2, return_components=False):
        # 1. Extract features from different views
        z1 = self.backbone.forward(x1, y1, fp1)
        z2 = self.backbone.forward(x2, y2, fp2)
        
        # 2. Project to high-dimensional space
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if not dist.is_initialized():
            # 3. Compute loss
            loss = self.vicreg_loss(z1, z2, return_components=return_components)
            return loss
        else:
            # 3. Gather features across distributed processes
            z1_gathered = torch.cat(all_gather(z1), dim=0)
            z2_gathered = torch.cat(all_gather(z2), dim=0)

            # 4. Compute loss on gathered features
            loss = self.vicreg_loss(z1_gathered, z2_gathered, return_components=return_components)
            return loss
        
    def extract_features(self, x, y, fp):
        z = self.backbone.forward(x, y, fp)
        # z = self.projector(z)
        return z

class VICRegLoss(nn.Module):
    def __init__(self, lam=25.0, mu=25.0, nu=1.0, gamma=1.0, eps=1e-4):
        super().__init__()
        self.lam = lam
        self.mu = mu
        self.nu = nu
        self.gamma = gamma
        self.eps = eps

    def forward(self, z1, z2, return_components=False):
        # 1. Invariance Loss
        sim_loss = F.mse_loss(z1, z2)

        # Center the representations
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        # 2. Variance Loss
        std_z1 = torch.sqrt(z1.var(dim=0) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std_z1)) + torch.mean(F.relu(self.gamma - std_z2))

        # 3. Covariance Loss
        batch_size = z1.size(0)
        num_features = z1.size(1)
        
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        # Mask out diagonal elements
        diag_mask = ~torch.eye(num_features, device=z1.device).bool()
        cov_loss = (cov_z1[diag_mask].pow(2).sum() / num_features) + (cov_z2[diag_mask].pow(2).sum() / num_features)

        # Compute effective dimensions
        eig_z1 = torch.linalg.eigvalsh(cov_z1)
        eig_z2 = torch.linalg.eigvalsh(cov_z2)
        sum_ev_z1 = eig_z1.sum()
        sum_sq_ev_z1 = (eig_z1 ** 2).sum()
        sum_ev_z2 = eig_z2.sum()
        sum_sq_ev_z2 = (eig_z2 ** 2).sum()
        eff_dim_z1 = (sum_ev_z1 ** 2) / (sum_sq_ev_z1)
        eff_dim_z2 = (sum_ev_z2 ** 2) / (sum_sq_ev_z2)

        sim_component = self.lam * sim_loss
        var_component = self.mu * var_loss
        cov_component = self.nu * cov_loss
        total_loss = sim_component + var_component + cov_component

        # Total Loss
        if return_components:
            return total_loss, sim_component, var_component, cov_component, eff_dim_z1, eff_dim_z2
        else:
            return total_loss

class CCLPretrain(nn.Module):
    def __init__(self, backbone=None, projector_dim=128):
        super().__init__()

        self.backbone = (
            build_feature_extractor(
                config.pretrain.get("backbone_type", "legacy"),
                nspec=config.data["nspec"],
            )
            if backbone is None
            else backbone
        )

        # Get feature dimensions dynamically
        dim_in = 1024

        # Stage 4 keeps the contrastive head equivariant too. Its spin blocks
        # use bias-free multiplicity mixing, while scalar channels may use an
        # unrestricted scalar MLP. Legacy/Stage 3 checkpoints retain the
        # original projection head exactly.
        if isinstance(self.backbone, D4OrbitFeatureExtractor):
            self.projector = D4EquivariantCCLProjector(
                scalar_channels=self.backbone.scalar_channels,
                spin1_channels=self.backbone.spin1_channels,
                spin2_channels=self.backbone.spin2_channels,
                output_dim=projector_dim,
            )
        else:
            self.projector = MLP(
                [dim_in, 2048, 512, projector_dim],
                use_batchnorm=True,
                use_dropout=False,
            )

        feature_names = list(config.train["feature_names"])
        configured_scales = config.pretrain.get("ccl_label_scales", {})
        label_scales = [
            float(configured_scales.get(name, 1.0)) for name in feature_names
        ]
        theta_idx = resolve_feature_index(feature_names, "theta_int")
        self.ccl_loss = ContinuousContrastiveLoss(
            temperature=float(config.pretrain.get("ccl_temperature", 0.1)),
            sigma_label=float(config.pretrain.get("ccl_sigma_label", 0.15)),
            d_cutoff=float(config.pretrain.get("ccl_d_cutoff", 0.40)),
            label_scales=label_scales,
            theta_idx=theta_idx,
            distance_reduction=config.pretrain.get(
                "ccl_distance_reduction", "mean"
            ),
        )

    def _compute_ccl_loss(self, projected_features, labels, return_diagnostics):
        if dist.is_initialized():
            projected_features = torch.cat(all_gather(projected_features), dim=0)
            labels = torch.cat(all_gather(labels), dim=0)
        return self.ccl_loss(
            projected_features,
            labels,
            return_diagnostics=return_diagnostics,
        )

    def forward(self, x, y, fp, labels, return_diagnostics=False):
        backbone_features = self.backbone.forward(x, y, fp)
        projected_features = self.projector(backbone_features)
        return self._compute_ccl_loss(
            projected_features,
            labels,
            return_diagnostics,
        )

    def extract_features(self, x, y, fp):
        return self.backbone.forward(x, y, fp)


class ContinuousContrastiveLoss(nn.Module):
    """Continuous contrastive loss with fixed normalized parameter geometry."""

    def __init__(
        self,
        temperature=0.1,
        sigma_label=0.15,
        d_cutoff=0.40,
        label_scales=None,
        theta_idx=None,
        distance_reduction="mean",
    ):
        super().__init__()
        for name, value in (
            ("temperature", temperature),
            ("sigma_label", sigma_label),
            ("d_cutoff", d_cutoff),
        ):
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if theta_idx is not None and type(theta_idx) is not int:
            raise TypeError("theta_idx must be an integer or None")
        if distance_reduction not in ("mean", "sum"):
            raise ValueError("distance_reduction must be 'mean' or 'sum'")

        scales = torch.as_tensor(
            [1.0] if label_scales is None else label_scales,
            dtype=torch.float32,
        )
        if (
            scales.ndim != 1
            or torch.any(~torch.isfinite(scales))
            or torch.any(scales <= 0)
        ):
            raise ValueError(
                "label_scales must be a 1D sequence of positive finite values"
            )

        self.temperature = float(temperature)
        self.sigma_label = float(sigma_label)
        self.theta_idx = theta_idx
        self.distance_reduction = distance_reduction
        self.register_buffer("label_scales", scales)
        self.register_buffer(
            "delta_bg",
            torch.exp(
                torch.tensor(
                    -(d_cutoff ** 2) / (2 * (sigma_label ** 2)),
                    dtype=torch.float32,
                )
            ),
        )

    def pairwise_label_distance_sq(self, labels):
        """Return fixed-scale pairwise distances for normalized KL labels."""
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, nfeatures)")
        if self.label_scales.numel() not in (1, labels.shape[1]):
            raise ValueError(
                "label_scales must contain one value or one value per label feature"
            )
        if self.theta_idx is not None and not 0 <= self.theta_idx < labels.shape[1]:
            raise ValueError("theta_idx is outside the label feature dimension")

        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)
        if self.theta_idx is not None:
            theta_delta = label_diff[..., self.theta_idx]
            theta_delta = torch.atan2(
                torch.sin(math.pi * theta_delta),
                torch.cos(math.pi * theta_delta),
            ) / math.pi
            label_diff = label_diff.clone()
            label_diff[..., self.theta_idx] = theta_delta

        scaled_diff_sq = (label_diff / self.label_scales) ** 2
        if self.distance_reduction == "mean":
            return torch.mean(scaled_diff_sq, dim=-1)
        return torch.sum(scaled_diff_sq, dim=-1)

    def _target_distribution(self, labels):
        batch_size = labels.shape[0]
        if batch_size < 2:
            raise ValueError("continuous contrastive loss requires at least two rows")
        label_dist_sq = self.pairwise_label_distance_sq(labels).float()
        weights = torch.exp(-label_dist_sq / (2 * (self.sigma_label ** 2)))

        mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
        weights_masked = weights.masked_fill(mask, 0.0)
        row_sum = torch.sum(weights_masked, dim=1, keepdim=True)
        delta_bg = self.delta_bg.to(device=labels.device, dtype=weights.dtype)
        target_mass = row_sum / (row_sum + delta_bg)
        positive_probs = weights_masked / row_sum.clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        weights_norm = positive_probs * target_mass
        return mask, weights_norm, positive_probs, target_mass.squeeze(1)

    @staticmethod
    def _target_statistics(positive_probs, target_mass):
        row_entropy = -torch.sum(
            torch.special.xlogy(positive_probs, positive_probs), dim=1
        )
        concentration = torch.sum(positive_probs.square(), dim=1)
        effective_positives = concentration.clamp_min(
            torch.finfo(concentration.dtype).tiny
        ).reciprocal()
        effective_positives = torch.where(
            target_mass > 0, effective_positives, torch.zeros_like(concentration)
        )
        uniform_baseline = target_mass * math.log(positive_probs.shape[1] - 1)
        return {
            "target_entropy": torch.mean(target_mass * row_entropy).detach(),
            "uniform_baseline": torch.mean(uniform_baseline).detach(),
            "effective_positives": torch.mean(effective_positives).detach(),
            "target_mass": torch.mean(target_mass).detach(),
        }

    def target_statistics(self, labels):
        """Return batch-level diagnostics for the soft-positive target only."""
        _, _, positive_probs, target_mass = self._target_distribution(labels)
        return self._target_statistics(positive_probs, target_mass)

    def forward(self, z, labels, return_diagnostics=False):
        z = F.normalize(z, dim=1)
        batch_size = z.shape[0]
        if batch_size < 2:
            raise ValueError("continuous contrastive loss requires at least two rows")
        if labels.shape[0] != batch_size:
            raise ValueError("z and labels must have the same batch dimension")

        mask, weights_norm, positive_probs, target_mass = (
            self._target_distribution(labels)
        )

        sim_matrix = torch.matmul(z, z.T) / self.temperature
        log_prob_sim = F.log_softmax(
            sim_matrix.masked_fill(mask, -torch.inf),
            dim=1,
        )
        # Avoid the undefined 0 * -inf diagonal contribution.
        log_prob_sim = log_prob_sim.masked_fill(mask, 0.0)

        loss = -torch.sum(weights_norm * log_prob_sim, dim=1).mean()
        if not return_diagnostics:
            return loss

        diagnostics = self._target_statistics(positive_probs, target_mass)
        diagnostics["excess_loss"] = (
            loss.detach() - diagnostics["target_entropy"]
        )
        return loss, diagnostics



class MLP(nn.Module):
    '''
    A simple MLP with Linear and ReLU
    '''
    
    def __init__(self, layers, use_batchnorm=False, use_dropout=False):
        
        super(MLP,self).__init__()

        modules = nn.ModuleList([])
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i],layers[i+1]))
            if i != len(layers)-2:
                modules.append(nn.ReLU(True))
                if use_batchnorm:
                    modules.append(nn.BatchNorm1d(layers[i+1], affine=False))
                if use_dropout:
                    modules.append(nn.Dropout(0.1))

        self.mlp = nn.Sequential(*modules)

    def forward(self,x):

        x = self.mlp(x)
        return x

class ResidualBlock(nn.Module):
    '''
    A residual block object that skips layers until stride > 1, i.e. the size of data shrinks
    '''
    
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1,bias=False):
        
        super(ResidualBlock,self).__init__()
        
        self.cnn1 =nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size,1,padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()
            
            
    def forward(self,x):
        
        residual = x
        
        x = self.cnn1(x)
        x = self.cnn2(x)
        
        x += self.shortcut(residual)
        
        x = nn.ReLU(True)(x)
        return x

### ViT classes ###
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=6, img_size=48, embed_dim=512, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)  
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        return x
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2)
        x = x + attn_output
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=1, embed_dim=512, img_size=48, patch_size=6, num_layers=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        self.patch_embed = PatchEmbedding(in_channels, patch_size, img_size, embed_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_token = x[:, 0]
        return cls_token

### Spectra RNN ###
class SpecRNN(nn.Module):
    def __init__(self, nspec, hidden_size=256, num_layers=2, bidirectional=True):
        super().__init__()

        # Local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
        )

        # RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=64,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

    def forward(self, x):
        feat = self.cnn_spec(x) 
        
        feat = feat.mean(dim=-1)
        feat = feat.permute(0, 2, 1)

        # RNN along the spectral dimension
        rnn_out, _ = self.rnn_spec(feat)
        rnn_feat = rnn_out[:, -1, :]

        # Project to 512-dim feature
        out = self.proj(rnn_feat)
        return out
    
class LargeSpecRNN(nn.Module):
    def __init__(self, nspecs, hidden_size=1024, num_layers=4, bidirectional=True):
        super().__init__()

        self.nspecs = nspecs

        # Deeper local feature extractor across time
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  # Reduce temporal dim
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        # Larger RNN across spectral dimension
        self.rnn_spec = nn.GRU(
            input_size=256,          # CNN feature dim per spectral bin
            hidden_size=hidden_size,  # Increased to 1024
            num_layers=num_layers,    # Increased to 4 layers
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if num_layers > 1 else 0  # Add dropout between layers
        )

        # Larger output projection
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Sequential(
            nn.Linear(rnn_out_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512-2*self.nspecs),  # Final feature dim reduced to accommodate fiber position if needed
            nn.BatchNorm1d(512-2*self.nspecs),
            nn.ReLU(True)
        )

        # Add a linear layer to fuse the 1024 concatenated features back to 512
        self.pool_fusion = nn.Linear(1024-4*self.nspecs, 512-2*self.nspecs)  # Adjust input dim to account for removed fiber position features

    def forward(self, x):
        # x: (batch, 1, nspec, time)
        batch_size = x.size(0)
        
        # CNN processing
        x = self.cnn_spec(x)  # (batch, 256, nspec, time/2)
        
        # Reshape for RNN: merge batch and time, treat spectral bins as sequence
        b, c, nspec, t = x.size()
        x = x.permute(0, 3, 2, 1)  # (batch, time, nspec, channels)
        x = x.reshape(b * t, nspec, c)  # (batch*time, nspec, channels)
        
        # RNN processing across spectral dimension
        x, _ = self.rnn_spec(x)  # (batch*time, nspec, hidden*2)
        
        # Global pooling across spectral dimension
        x = x.mean(dim=1)  # (batch*time, hidden*2)
        
        # Projection
        x = self.proj(x)  # (batch*time, 512)
        
        # Reshape back
        x = x.view(b, t, -1)  # (batch, time, 512)

        # 1. Extract the sharp features
        x_max = x.max(dim=1)[0]   # (batch, 512)
        
        # 2. Extract the global continuum
        x_avg = x.mean(dim=1)     # (batch, 512)
        
        # 3. Combine them
        x_cat = torch.cat([x_max, x_avg], dim=-1) # (batch, 1024)
        
        # 4. Fuse back to the required 512 dimension
        x = self.pool_fusion(x_cat) # (batch, 512)
        
        # Optional: Add a non-linearity depending on your architecture design
        x = nn.functional.relu(x)
        
        return x

### Spec CNN ###
class SpecCNN(nn.Module):

    def __init__(self, nspecs):
        super(SpecCNN, self).__init__()

        self.nspecs = nspecs

        self.cnn_spec = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512-2*nspecs, kernel_size=(self.nspecs, 4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512-2*nspecs),
            nn.ReLU(True),
            
        )

    def forward(self, x):
        
        x = self.cnn_spec(x)
        
        return x

### Image CNN ###
class ImgCNN(nn.Module):
    def __init__(self):
        super(ImgCNN, self).__init__()

        self.cnn_img = nn.Sequential(

            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128, 2),
            
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, 2),
            
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512, 2),
            
            nn.AvgPool2d(3),
            
        )

    def forward(self, x):
        
        x = self.cnn_img(x)
        
        return x

class ForkCNN(nn.Module):
    def __init__(self, nspecs):
        pass
