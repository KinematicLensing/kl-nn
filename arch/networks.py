"""Current KL-NN feature, pretraining, and posterior architecture.

There is deliberately one production path:

* independent image-CNN, joint spectral-CNN, and metadata-MLP branches;
* direct concatenation of the three branches for CCL pretraining; and
* a nine-dimensional bounded hybrid posterior, with eight compact scalar
  coordinates and a directed circular ``theta_int`` coordinate.

The spectral branch intentionally assumes the fixed five-fiber, 64-bin
observation used by the current simulator. Alternate backbones, point-estimate
heads, training-time population weighting, and alternate flow families are
intentionally absent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F

from nflows.distributions.base import Distribution
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import (
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.transforms.standard import PointwiseAffineTransform

try:
    from . import config
    from .circular_spline import (
        DEFAULT_MIN_DERIVATIVE,
        unconstrained_rational_quadratic_spline,
    )
    from .utils import resolve_feature_index
except ImportError:  # Direct execution with arch/ on sys.path or a snapshot.
    import config
    from circular_spline import (
        DEFAULT_MIN_DERIVATIVE,
        unconstrained_rational_quadratic_spline,
    )
    from utils import resolve_feature_index


IMAGE_FEATURE_DIM = 512
SPECTRAL_FEATURE_DIM = 512
METADATA_FEATURE_DIM = 128
FEATURE_DIM = IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM + METADATA_FEATURE_DIM
TARGET_COUNT = 9
ORACLE_CONTEXT_FIELDS = tuple(config.ORACLE_CONTEXT_FIELDS)


def _configured_nspec() -> int:
    return int(config.data["nspec"])


def _configured_feature_names() -> tuple[str, ...]:
    return tuple(config.train["feature_names"])


def _validate_feature_schema(feature_names) -> tuple[str, ...]:
    names = tuple(feature_names)
    if len(names) != TARGET_COUNT:
        raise ValueError(
            f"the posterior requires exactly {TARGET_COUNT} targets; got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    required = {
        "g1",
        "g2",
        "theta_int",
        "cosi",
        "v0",
        "vcirc",
        "rscale",
        "hlr",
        "halpha_flux_true",
    }
    if set(names) != required:
        raise ValueError(
            "feature_names must contain the current nine-target schema; "
            f"missing={sorted(required - set(names))}, "
            f"extra={sorted(set(names) - required)}"
        )
    return names


class MLP(nn.Module):
    """Small configurable MLP used by the CCL projection head."""

    def __init__(self, layers, use_batchnorm=False, use_dropout=False):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must include input and output dimensions")
        modules = []
        for index, (in_features, out_features) in enumerate(
            zip(layers[:-1], layers[1:])
        ):
            modules.append(nn.Linear(in_features, out_features))
            if index != len(layers) - 2:
                modules.append(nn.ReLU(inplace=True))
                if use_batchnorm:
                    modules.append(nn.BatchNorm1d(out_features, affine=False))
                if use_dropout:
                    modules.append(nn.Dropout(0.1))
        self.mlp = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.mlp(inputs)


class OracleContextNormalizer(nn.Module):
    """Validate and normalize the three recorded observation scalars."""

    output_dim = 3

    def __init__(
        self,
        context_fields=None,
        *,
        rmag_min=None,
        rmag_max=None,
        image_snr_min=None,
        image_snr_max=None,
        central_halpha_snr_min=None,
        central_halpha_snr_max=None,
    ):
        super().__init__()
        fields = tuple(
            config.observation["context_fields"]
            if context_fields is None
            else context_fields
        )
        if fields != ORACLE_CONTEXT_FIELDS:
            raise ValueError(
                "context_fields must contain exactly the independent oracle "
                f"fields {ORACLE_CONTEXT_FIELDS!r}; got {fields!r}"
            )
        self.context_fields = fields

        rmag_min = float(
            config.observation["rmag_min"] if rmag_min is None else rmag_min
        )
        rmag_max = float(
            config.observation["rmag_max"] if rmag_max is None else rmag_max
        )
        image_snr_min = float(
            config.observation["image_snr_min"]
            if image_snr_min is None
            else image_snr_min
        )
        image_snr_max = float(
            config.observation["image_snr_max"]
            if image_snr_max is None
            else image_snr_max
        )
        central_halpha_snr_min = float(
            config.observation["central_halpha_snr_min"]
            if central_halpha_snr_min is None
            else central_halpha_snr_min
        )
        central_halpha_snr_max = float(
            config.observation["central_halpha_snr_max"]
            if central_halpha_snr_max is None
            else central_halpha_snr_max
        )
        if not math.isfinite(rmag_min) or not math.isfinite(rmag_max) or rmag_min >= rmag_max:
            raise ValueError("rmag bounds must be finite and increasing")
        for name, lower, upper in (
            ("image S/N", image_snr_min, image_snr_max),
            (
                "central H-alpha S/N",
                central_halpha_snr_min,
                central_halpha_snr_max,
            ),
        ):
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower <= 0
                or lower >= upper
            ):
                raise ValueError(
                    f"{name} bounds must be finite, positive, and increasing"
                )

        self.register_buffer(
            "rmag_midpoint", torch.tensor(0.5 * (rmag_min + rmag_max))
        )
        self.register_buffer(
            "rmag_half_range", torch.tensor(0.5 * (rmag_max - rmag_min))
        )
        self.register_buffer(
            "image_snr_midpoint",
            torch.tensor(0.5 * (image_snr_min + image_snr_max)),
        )
        self.register_buffer(
            "image_snr_half_range",
            torch.tensor(0.5 * (image_snr_max - image_snr_min)),
        )
        self.register_buffer(
            "central_halpha_snr_midpoint",
            torch.tensor(
                0.5 * (central_halpha_snr_min + central_halpha_snr_max)
            ),
        )
        self.register_buffer(
            "central_halpha_snr_half_range",
            torch.tensor(
                0.5 * (central_halpha_snr_max - central_halpha_snr_min)
            ),
        )

    def _mapping_to_tensor(self, context, batch_size, reference):
        supplied = set(context)
        latent_targets = sorted(supplied.intersection(config.TARGET_NAMES))
        if latent_targets:
            raise ValueError(
                "posterior targets must not appear in observation_context: "
                f"{latent_targets!r}"
            )
        expected = set(self.context_fields)
        if supplied != expected:
            raise ValueError(
                "observation_context keys must exactly match the oracle fields; "
                f"missing={sorted(expected - supplied)}, "
                f"extra={sorted(supplied - expected)}"
            )

        columns = []
        for name in self.context_fields:
            column = torch.as_tensor(
                context[name], device=reference.device, dtype=reference.dtype
            )
            if column.ndim == 0:
                column = column.expand(batch_size)
            elif column.ndim == 2 and column.shape[-1] == 1:
                column = column[:, 0]
            if column.shape != (batch_size,):
                raise ValueError(
                    f"observation_context[{name!r}] must be scalar or have "
                    f"shape ({batch_size},); got {tuple(column.shape)}"
                )
            columns.append(column)
        return torch.stack(columns, dim=-1)

    def forward(self, context, batch_size, reference):
        if context is None:
            raise ValueError("observation_context is required")
        if isinstance(context, Mapping):
            values = self._mapping_to_tensor(context, batch_size, reference)
        else:
            values = torch.as_tensor(
                context, device=reference.device, dtype=reference.dtype
            )
            if values.shape != (batch_size, self.output_dim):
                raise ValueError(
                    "tensor observation_context must follow field order "
                    f"{self.context_fields!r} and have shape "
                    f"({batch_size}, {self.output_dim}); got {tuple(values.shape)}"
                )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("observation_context must be finite")
        if not bool((values[:, 1:] > 0).all()):
            raise ValueError("image_snr and central_halpha_snr must be positive")

        normalized = values.clone()
        normalized[:, 0] = (
            values[:, 0] - self.rmag_midpoint.to(values)
        ) / self.rmag_half_range.to(values)
        normalized[:, 1] = (
            values[:, 1] - self.image_snr_midpoint.to(values)
        ) / self.image_snr_half_range.to(values)
        normalized[:, 2] = (
            values[:, 2] - self.central_halpha_snr_midpoint.to(values)
        ) / self.central_halpha_snr_half_range.to(values)
        return normalized


class ConditionalCircularThetaTransform(nn.Module):
    """Circular RQS for ``q(theta_int | non_theta, context)``."""

    def __init__(
        self,
        condition_features,
        *,
        num_layers=1,
        num_bins=8,
        hidden_features=128,
        logit_limit=10.0,
        tail_bound=1.0,
        spline_function=None,
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
        self.spline_function = (
            unconstrained_rational_quadratic_spline
            if spline_function is None
            else spline_function
        )
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
        self.last_diagnostics = {}

    @staticmethod
    def canonicalize(theta):
        if not bool(torch.isfinite(theta).all()):
            raise FloatingPointError("theta_int contains a non-finite value")
        return torch.remainder(theta + 1.0, 2.0) - 1.0

    def _bounded_parameters(self, conditioner, condition):
        raw = conditioner(condition)
        bounded = self.logit_limit * torch.tanh(raw / self.logit_limit)
        return (
            raw,
            bounded[..., : self.num_bins].unsqueeze(-2),
            bounded[..., self.num_bins : 2 * self.num_bins].unsqueeze(-2),
            bounded[..., 2 * self.num_bins :].unsqueeze(-2),
        )

    def _apply_layer(self, theta, condition, conditioner, *, inverse):
        raw, widths, heights, derivatives = self._bounded_parameters(
            conditioner, condition
        )
        transformed, logabsdet = self.spline_function(
            inputs=theta.unsqueeze(-1),
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            inverse=inverse,
            tails="circular",
            tail_bound=self.tail_bound,
        )
        transformed = transformed.squeeze(-1)
        excursion = torch.maximum(
            (-1.0 - transformed).clamp_min(0.0),
            (transformed - 1.0).clamp_min(0.0),
        )
        wrap_mask = (transformed < -1.0) | (transformed >= 1.0)
        return (
            self.canonicalize(transformed),
            logabsdet.squeeze(-1),
            raw,
            wrap_mask.sum(),
            excursion.max(),
        )

    def forward(self, theta, condition):
        if theta.ndim != 1:
            raise ValueError("theta must have shape (batch,)")
        if condition.shape != (theta.shape[0], self.condition_features):
            raise ValueError(
                "condition must have shape "
                f"({theta.shape[0]}, {self.condition_features})"
            )
        value = self.canonicalize(theta)
        total_logabsdet = torch.zeros_like(value)
        raw_max = value.new_zeros(())
        bounded_max = value.new_zeros(())
        derivative_min = value.new_tensor(torch.inf)
        derivative_max = value.new_tensor(-torch.inf)
        wrap_count = value.new_zeros(())
        max_wrap_excursion = value.new_zeros(())
        for conditioner in self.conditioners:
            value, logabsdet, raw, wraps, excursion = self._apply_layer(
                value, condition, conditioner, inverse=False
            )
            total_logabsdet = total_logabsdet + logabsdet
            bounded = self.logit_limit * torch.tanh(raw / self.logit_limit)
            derivatives = self.min_derivative + F.softplus(
                bounded[..., 2 * self.num_bins :]
            )
            raw_max = torch.maximum(raw_max, raw.detach().abs().max())
            bounded_max = torch.maximum(
                bounded_max, bounded.detach().abs().max()
            )
            derivative_min = torch.minimum(
                derivative_min, derivatives.detach().min()
            )
            derivative_max = torch.maximum(
                derivative_max, derivatives.detach().max()
            )
            wrap_count = wrap_count + wraps.detach()
            max_wrap_excursion = torch.maximum(
                max_wrap_excursion, excursion.detach()
            )
        self.last_diagnostics = {
            "theta_raw_logit_abs_max": raw_max,
            "theta_bounded_logit_abs_max": bounded_max,
            "theta_derivative_min": derivative_min,
            "theta_derivative_max": derivative_max,
            "theta_wrap_count": wrap_count,
            "theta_max_wrap_excursion": max_wrap_excursion,
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
        value = self.canonicalize(latent_theta)
        total_logabsdet = torch.zeros_like(value)
        for conditioner in reversed(self.conditioners):
            value, logabsdet, _, _, _ = self._apply_layer(
                value, condition, conditioner, inverse=True
            )
            total_logabsdet = total_logabsdet + logabsdet
        return value, total_logabsdet

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
    """Uniform base distribution on ``[0, 1]^D`` with nflows semantics."""

    def __init__(self, features):
        super().__init__()
        if type(features) is not int or features <= 0:
            raise ValueError("features must be a positive integer")
        self.features = features
        self.register_buffer("_reference", torch.zeros(features))

    def _log_prob(self, inputs, context):
        del context
        if inputs.ndim != 2 or inputs.shape[-1] != self.features:
            raise ValueError(
                f"inputs must have shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )
        support = (
            torch.isfinite(inputs) & (inputs >= 0.0) & (inputs <= 1.0)
        ).all(dim=-1)
        zeros = inputs.new_zeros(inputs.shape[0])
        return torch.where(support, zeros, torch.full_like(zeros, -torch.inf))

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
    """Compact nflows RQS with bounded conditioner logits and identity start."""

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
            raise ValueError("logit_limit is too small for identity initialization")
        raw_derivative_bias = self.logit_limit * math.atanh(
            target / self.logit_limit
        )
        final = self.autoregressive_net.final_layer
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        with torch.no_grad():
            bias = final.bias.view(-1, self._output_dim_multiplier())
            bias[:, 2 * self.num_bins :] = raw_derivative_bias

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        raw = autoregressive_params
        bounded = self.logit_limit * torch.tanh(raw / self.logit_limit)
        precise_inverse = inverse and inputs.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        )
        spline_inputs = inputs.double() if precise_inverse else inputs
        spline_parameters = bounded.double() if precise_inverse else bounded
        outputs, logabsdet = super()._elementwise(
            spline_inputs, spline_parameters, inverse=inverse
        )
        if precise_inverse:
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


class BoundedHybridCircularFlow(nn.Module):
    """Nine-target compact posterior with one correlated circular coordinate.

    The joint factorization is

    ``q(x, theta | context) = q_box(x | context) q_circle(theta | x, context)``,

    where all eight entries of ``x`` live on the closed normalized interval
    ``[-1, 1]`` and ``theta_int`` lives on the half-open circle ``[-1, 1)``.
    """

    def __init__(
        self,
        *,
        features=TARGET_COUNT,
        theta_index=2,
        context_features=FEATURE_DIM,
        num_bounded_layers=4,
        num_theta_layers=1,
        num_bins=8,
        hidden_features=256,
        num_blocks=2,
        theta_hidden_features=128,
        theta_logit_limit=10.0,
        bounded_logit_limit=10.0,
        non_theta_flow=None,
        theta_transform=None,
    ):
        super().__init__()
        if features != TARGET_COUNT:
            raise ValueError(
                f"BoundedHybridCircularFlow requires {TARGET_COUNT} targets"
            )
        if type(theta_index) is not int or not 0 <= theta_index < features:
            raise ValueError("theta_index is out of bounds")
        if type(context_features) is not int or context_features <= 0:
            raise ValueError("context_features must be a positive integer")
        if type(num_bounded_layers) is not int or num_bounded_layers <= 0:
            raise ValueError("num_bounded_layers must be a positive integer")
        if type(num_theta_layers) is not int or num_theta_layers <= 0:
            raise ValueError("num_theta_layers must be a positive integer")
        if type(num_bins) is not int or num_bins < 2:
            raise ValueError("num_bins must be an integer of at least two")

        self.features = features
        self.theta_index = theta_index
        self.context_features = context_features
        self.non_theta_indices = tuple(
            index for index in range(features) if index != theta_index
        )
        self.non_theta_features = features - 1

        if non_theta_flow is None:
            transforms = [PointwiseAffineTransform(scale=0.5, shift=0.5)]
            for _ in range(num_bounded_layers):
                transforms.extend(
                    (
                        ReversePermutation(features=self.non_theta_features),
                        IdentityBoundedRationalQuadraticAutoregressiveTransform(
                            features=self.non_theta_features,
                            hidden_features=hidden_features,
                            context_features=context_features,
                            num_bins=num_bins,
                            num_blocks=num_blocks,
                            logit_limit=bounded_logit_limit,
                        ),
                    )
                )
            non_theta_flow = Flow(
                CompositeTransform(transforms),
                ConditionalUnitBox(self.non_theta_features),
            )
        if not isinstance(non_theta_flow, nn.Module):
            raise TypeError("non_theta_flow must be an nn.Module")
        self.non_theta_flow = non_theta_flow

        if theta_transform is None:
            theta_transform = ConditionalCircularThetaTransform(
                context_features + self.non_theta_features,
                num_layers=num_theta_layers,
                num_bins=num_bins,
                hidden_features=theta_hidden_features,
                logit_limit=theta_logit_limit,
            )
        if not isinstance(theta_transform, nn.Module):
            raise TypeError("theta_transform must be an nn.Module")
        self.theta_transform = theta_transform
        self.last_component_diagnostics = {}

    @property
    def bounded_transforms(self):
        transform = getattr(self.non_theta_flow, "_transform", None)
        children = getattr(transform, "_transforms", ())
        return tuple(
            item
            for item in children
            if isinstance(
                item, IdentityBoundedRationalQuadraticAutoregressiveTransform
            )
        )

    def _validate_context(self, context, batch_size):
        if context is None or context.shape != (
            batch_size,
            self.context_features,
        ):
            actual = None if context is None else tuple(context.shape)
            raise ValueError(
                "context must have shape "
                f"({batch_size}, {self.context_features}); got {actual}"
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
                f"bounded posterior produced {count} rows outside [-1, 1] "
                f"during {operation}; refusing to clamp"
            )

    def _bounded_diagnostics(self, support, reference):
        diagnostics = [
            transform.last_diagnostics
            for transform in self.bounded_transforms
            if transform.last_diagnostics
        ]
        output = {
            "bounded_support_violation_count": (~support)
            .sum()
            .detach()
            .to(dtype=reference.dtype),
        }
        if diagnostics:
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
        non_theta_log_prob = self.non_theta_flow.log_prob(
            safe_non_theta, context=context
        )
        non_theta_log_prob = torch.where(
            support,
            non_theta_log_prob,
            torch.full_like(non_theta_log_prob, -torch.inf),
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
            "non_theta_log_prob_mean": non_theta_log_prob.detach().mean(),
            "theta_log_prob_mean": theta_log_prob.detach().mean(),
            **self._bounded_diagnostics(support, inputs),
        }
        return non_theta_log_prob, theta_log_prob

    def log_prob(self, inputs, context=None):
        non_theta_log_prob, theta_log_prob = self.component_log_prob(
            inputs, context
        )
        return non_theta_log_prob + theta_log_prob

    def sample(self, num_samples, context=None):
        if type(num_samples) is not int or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")
        if context is None or context.ndim != 2:
            raise ValueError("context must have shape (batch, context_features)")
        self._validate_context(context, context.shape[0])
        non_theta = self.non_theta_flow.sample(num_samples, context=context)
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
        if not bool(torch.isfinite(samples).all()):
            raise RuntimeError("bounded posterior produced non-finite samples")
        return samples

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
        self._assert_support(non_theta, operation="transform_to_noise")
        if not bool(torch.isfinite(theta).all()):
            raise RuntimeError("theta_int must be finite during transform_to_noise")
        non_theta_noise = self.non_theta_flow.transform_to_noise(
            non_theta, context=context
        )
        condition = torch.cat((context, non_theta), dim=-1)
        theta_noise, _ = self.theta_transform(theta, condition)
        return self._assemble(non_theta_noise, theta_noise)


class ResidualBlock(nn.Module):
    """Residual image-CNN block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        kernel_size=3,
        padding=1,
    ):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                1,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        return self.activation(self.cnn2(self.cnn1(inputs)) + self.shortcut(inputs))


class ImgCNN(nn.Module):
    """Production 48x48 image encoder returning 512 features."""

    output_dim = 512

    def __init__(self):
        super().__init__()
        self.cnn_img = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
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

    def forward(self, inputs):
        return self.cnn_img(inputs)


class JointSpecCNN(nn.Module):
    """Joint CNN for the fixed ordered five-fiber by 64-bin spectrum."""

    output_dim = SPECTRAL_FEATURE_DIM
    wavelength_count = 64

    def __init__(self, nspec=None):
        super().__init__()
        nspec = _configured_nspec() if nspec is None else int(nspec)
        if nspec <= 0:
            raise ValueError("nspec must be positive")
        self.nspecs = nspec
        self.cnn_spec = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256,
                self.output_dim,
                kernel_size=(self.nspecs, 4),
                bias=False,
            ),
            nn.BatchNorm2d(self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, spectra):
        expected = (
            spectra.shape[0],
            1,
            self.nspecs,
            self.wavelength_count,
        )
        if tuple(spectra.shape) != expected:
            raise ValueError(
                "spectra must have shape "
                f"(batch, 1, {self.nspecs}, {self.wavelength_count}); "
                f"got {tuple(spectra.shape)}"
            )
        features = self.cnn_spec(spectra).flatten(start_dim=1)
        if features.shape != (spectra.shape[0], self.output_dim):
            raise RuntimeError(
                "joint spectral CNN produced an unexpected feature shape "
                f"{tuple(features.shape)}"
            )
        return features


class MetadataMLP(nn.Module):
    """Encode ordered fiber coordinates and the three catalog scalars."""

    output_dim = METADATA_FEATURE_DIM

    def __init__(self, nspec=None, hidden_dim=64):
        super().__init__()
        nspec = _configured_nspec() if nspec is None else int(nspec)
        if nspec <= 0 or hidden_dim <= 0:
            raise ValueError("nspec and hidden_dim must be positive")
        self.nspecs = nspec
        self.input_dim = 2 * nspec + len(ORACLE_CONTEXT_FIELDS)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, metadata):
        if metadata.ndim != 2 or metadata.shape[1] != self.input_dim:
            raise ValueError(
                "metadata must have shape "
                f"(batch, {self.input_dim}); got {tuple(metadata.shape)}"
            )
        return self.mlp(metadata)


class SimpleFusionFeatureExtractor(nn.Module):
    """Concatenate independent image, joint-spectrum, and metadata features."""

    output_dim = FEATURE_DIM

    def __init__(
        self,
        nspec=None,
        *,
        fiber_position_scale=1.5,
        img_net=None,
        spec_net=None,
        metadata_net=None,
        context_normalizer=None,
        context_fields=None,
    ):
        super().__init__()
        nspec = _configured_nspec() if nspec is None else int(nspec)
        if nspec <= 0:
            raise ValueError("nspec must be positive")
        if fiber_position_scale <= 0 or not math.isfinite(fiber_position_scale):
            raise ValueError("fiber_position_scale must be positive and finite")
        if context_normalizer is not None and context_fields is not None:
            raise ValueError(
                "pass context_normalizer or context_fields, not both"
            )
        self.nspecs = nspec
        self.fiber_position_scale = float(fiber_position_scale)
        self.img_net = ImgCNN() if img_net is None else img_net
        self.spec_net = JointSpecCNN(nspec=nspec) if spec_net is None else spec_net
        self.metadata_net = (
            MetadataMLP(nspec=nspec) if metadata_net is None else metadata_net
        )
        self.context_normalizer = (
            OracleContextNormalizer(context_fields=context_fields)
            if context_normalizer is None
            else context_normalizer
        )

    def _validate_inputs(self, image, spectra, fiber_positions, fiber_mask):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must have shape (batch, 1, height, width)")
        expected_spectra = (
            image.shape[0],
            1,
            self.nspecs,
            JointSpecCNN.wavelength_count,
        )
        if tuple(spectra.shape) != expected_spectra:
            raise ValueError(
                "spectra must have shape "
                f"(batch, 1, {self.nspecs}, "
                f"{JointSpecCNN.wavelength_count}); got {tuple(spectra.shape)}"
            )
        expected_positions = (image.shape[0], self.nspecs, 2)
        if tuple(fiber_positions.shape) != expected_positions:
            raise ValueError(
                "fiber_positions must have shape "
                f"(batch, {self.nspecs}, 2); got {tuple(fiber_positions.shape)}"
            )
        if fiber_mask is not None:
            if fiber_mask.dtype != torch.bool:
                raise TypeError("fiber_mask must be a bool tensor")
            if tuple(fiber_mask.shape) != expected_positions[:2]:
                raise ValueError(
                    "fiber_mask must have shape "
                    f"(batch, {self.nspecs})"
                )
            if not bool(fiber_mask.all()):
                raise ValueError(
                    "the fixed-order extractor requires all configured fibers"
                )

    def forward(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        self._validate_inputs(image, spectra, fiber_positions, fiber_mask)
        batch_size = image.shape[0]

        image_features = self.img_net(
            F.normalize(image, dim=(-2, -1))
        ).reshape(batch_size, -1)
        spectral_features = self.spec_net(
            F.normalize(spectra, dim=(-2, -1))
        ).reshape(batch_size, -1)
        normalized_context = self.context_normalizer(
            observation_context, batch_size, image_features
        )
        normalized_positions = (
            fiber_positions / self.fiber_position_scale
        ).reshape(batch_size, -1)
        metadata_features = self.metadata_net(
            torch.cat((normalized_positions, normalized_context), dim=-1)
        )

        expected_shapes = (
            ("image", image_features, IMAGE_FEATURE_DIM),
            ("spectral", spectral_features, SPECTRAL_FEATURE_DIM),
            ("metadata", metadata_features, METADATA_FEATURE_DIM),
        )
        for name, features, width in expected_shapes:
            if features.shape != (batch_size, width):
                raise ValueError(
                    f"{name} branch must return shape "
                    f"({batch_size}, {width}); got {tuple(features.shape)}"
                )
        return torch.cat(
            (image_features, spectral_features, metadata_features), dim=-1
        )


def build_feature_extractor(nspec=None, **kwargs):
    """Construct the sole supported feature extractor."""
    return SimpleFusionFeatureExtractor(nspec=nspec, **kwargs)


class ContinuousContrastiveLoss(nn.Module):
    """Continuous contrastive loss on the normalized nine-target geometry."""

    def __init__(
        self,
        temperature=0.1,
        sigma_label=0.15,
        d_cutoff=0.40,
        label_scales=None,
        theta_idx=2,
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
        if type(theta_idx) is not int:
            raise TypeError("theta_idx must be an integer")
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
                    -(d_cutoff**2) / (2 * sigma_label**2),
                    dtype=torch.float32,
                )
            ),
        )

    def pairwise_label_distance_sq(self, labels, candidates=None):
        candidates = labels if candidates is None else candidates
        if labels.ndim != 2 or labels.shape[1] != TARGET_COUNT:
            raise ValueError(
                f"labels must have shape (batch, {TARGET_COUNT})"
            )
        if candidates.ndim != 2 or candidates.shape[1] != TARGET_COUNT:
            raise ValueError(
                f"candidates must have shape (batch, {TARGET_COUNT})"
            )
        if self.label_scales.numel() not in (1, labels.shape[1]):
            raise ValueError(
                "label_scales must contain one value or one value per target"
            )
        if not 0 <= self.theta_idx < labels.shape[1]:
            raise ValueError("theta_idx is outside the target dimension")

        difference = labels.unsqueeze(1) - candidates.unsqueeze(0)
        theta_delta = difference[..., self.theta_idx]
        theta_delta = torch.atan2(
            torch.sin(math.pi * theta_delta),
            torch.cos(math.pi * theta_delta),
        ) / math.pi
        difference = difference.clone()
        difference[..., self.theta_idx] = theta_delta
        squared = (difference / self.label_scales) ** 2
        return squared.mean(dim=-1) if self.distance_reduction == "mean" else squared.sum(dim=-1)

    def _target_distribution(
        self, anchor_labels, candidate_labels, anchor_indices
    ):
        candidate_count = candidate_labels.shape[0]
        if candidate_count < 2:
            raise ValueError("continuous contrastive loss requires at least two rows")
        distances = self.pairwise_label_distance_sq(
            anchor_labels, candidate_labels
        ).float()
        weights = torch.exp(-distances / (2 * self.sigma_label**2))
        candidates = torch.arange(
            candidate_count, device=candidate_labels.device
        )
        diagonal = anchor_indices[:, None] == candidates[None, :]
        weights = weights.masked_fill(diagonal, 0.0)
        row_sum = weights.sum(dim=1, keepdim=True)
        delta_bg = self.delta_bg.to(
            device=candidate_labels.device, dtype=weights.dtype
        )
        target_mass = row_sum / (row_sum + delta_bg)
        positive_probs = weights / row_sum.clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        return (
            diagonal,
            positive_probs * target_mass,
            positive_probs,
            target_mass.squeeze(1),
        )

    @staticmethod
    def _target_statistics(positive_probs, target_mass):
        row_entropy = -torch.sum(
            torch.special.xlogy(positive_probs, positive_probs), dim=1
        )
        concentration = positive_probs.square().sum(dim=1)
        effective = concentration.clamp_min(
            torch.finfo(concentration.dtype).tiny
        ).reciprocal()
        effective = torch.where(
            target_mass > 0, effective, torch.zeros_like(effective)
        )
        return {
            "target_entropy": (target_mass * row_entropy).mean().detach(),
            "uniform_baseline": (
                target_mass * math.log(positive_probs.shape[1] - 1)
            ).mean().detach(),
            "effective_positives": effective.mean().detach(),
            "target_mass": target_mass.mean().detach(),
        }

    def target_statistics(self, labels):
        indices = torch.arange(labels.shape[0], device=labels.device)
        _, _, positive_probs, target_mass = self._target_distribution(
            labels, labels, indices
        )
        return self._target_statistics(positive_probs, target_mass)

    def forward(
        self,
        embeddings,
        labels,
        return_diagnostics=False,
        *,
        anchor_start=0,
        anchor_count=None,
    ):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (batch, features)")
        if labels.shape != (embeddings.shape[0], TARGET_COUNT):
            raise ValueError(
                f"labels must have shape ({embeddings.shape[0]}, {TARGET_COUNT})"
            )
        anchor_start = int(anchor_start)
        anchor_count = (
            embeddings.shape[0] if anchor_count is None else int(anchor_count)
        )
        anchor_stop = anchor_start + anchor_count
        if not 0 <= anchor_start < anchor_stop <= embeddings.shape[0]:
            raise ValueError("anchor rows must be a non-empty in-bounds slice")
        # Keep the quadratic similarity and softmax arithmetic in float32 even
        # when the encoders/projector run under CUDA autocast.
        embeddings = F.normalize(embeddings.float(), dim=1)
        anchor_embeddings = embeddings[anchor_start:anchor_stop]
        anchor_labels = labels[anchor_start:anchor_stop]
        anchor_indices = torch.arange(
            anchor_start, anchor_stop, device=labels.device
        )
        diagonal, target, positive_probs, target_mass = (
            self._target_distribution(anchor_labels, labels, anchor_indices)
        )
        similarities = anchor_embeddings @ embeddings.T / self.temperature
        log_prob = F.log_softmax(
            similarities.masked_fill(diagonal, -torch.inf), dim=1
        ).masked_fill(diagonal, 0.0)
        loss = -(target * log_prob).sum(dim=1).mean()
        if not return_diagnostics:
            return loss
        diagnostics = self._target_statistics(positive_probs, target_mass)
        diagnostics["excess_loss"] = loss.detach() - diagnostics["target_entropy"]
        return loss, diagnostics


class CCLPretrain(nn.Module):
    """CCL pretraining over the fully fused observation representation."""

    def __init__(
        self,
        backbone=None,
        projector=None,
        projector_dim=128,
        context_normalizer=None,
        context_fields=None,
    ):
        super().__init__()
        if backbone is None:
            self.backbone = build_feature_extractor(
                context_normalizer=context_normalizer,
                context_fields=context_fields,
            )
        else:
            if context_normalizer is not None or context_fields is not None:
                raise ValueError(
                    "context options belong to the supplied backbone"
                )
            self.backbone = backbone
        declared_feature_dim = getattr(self.backbone, "output_dim", None)
        if declared_feature_dim != FEATURE_DIM:
            raise ValueError(
                "CCL backbone must declare output_dim="
                f"{FEATURE_DIM}; got {declared_feature_dim!r}. "
                "Archived feature extractors are incompatible."
            )
        self.projector = (
            MLP(
                [FEATURE_DIM, 2048, 512, projector_dim],
                use_batchnorm=True,
                use_dropout=False,
            )
            if projector is None
            else projector
        )

        feature_names = _validate_feature_schema(_configured_feature_names())
        configured_scales = config.pretrain.get("ccl_label_scales", {})
        label_scales = [
            float(configured_scales.get(name, 1.0)) for name in feature_names
        ]
        self.ccl_loss = ContinuousContrastiveLoss(
            temperature=float(config.pretrain.get("ccl_temperature", 0.1)),
            sigma_label=float(config.pretrain.get("ccl_sigma_label", 0.15)),
            d_cutoff=float(config.pretrain.get("ccl_d_cutoff", 0.40)),
            label_scales=label_scales,
            theta_idx=resolve_feature_index(feature_names, "theta_int"),
            distance_reduction=config.pretrain.get(
                "ccl_distance_reduction", "mean"
            ),
        )

    def _joint_features(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        features = self.backbone(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        if features.shape != (image.shape[0], FEATURE_DIM):
            raise ValueError(
                "feature extractor must return shape "
                f"({image.shape[0]}, {FEATURE_DIM}); got {tuple(features.shape)}"
            )
        return features

    def forward(
        self,
        image,
        spectra,
        fiber_positions,
        labels,
        observation_context,
        fiber_mask=None,
        return_diagnostics=False,
    ):
        if labels.shape != (image.shape[0], TARGET_COUNT):
            raise ValueError(
                f"labels must have shape ({image.shape[0]}, {TARGET_COUNT})"
            )
        projected = self.projector(
            self._joint_features(
                image,
                spectra,
                fiber_positions,
                observation_context,
                fiber_mask=fiber_mask,
            )
        )
        anchor_start = 0
        anchor_count = projected.shape[0]
        if dist.is_initialized():
            projected_parts = all_gather(projected)
            label_parts = all_gather(labels)
            rank = dist.get_rank()
            anchor_start = sum(
                part.shape[0] for part in projected_parts[:rank]
            )
            projected = torch.cat(projected_parts, dim=0)
            labels = torch.cat(label_parts, dim=0)
        return self.ccl_loss(
            projected,
            labels,
            return_diagnostics=return_diagnostics,
            anchor_start=anchor_start,
            anchor_count=anchor_count,
        )

    def extract_features(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        return self.backbone(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )


class KLNPE(nn.Module):
    """Density-only nine-target neural posterior estimator."""

    def __init__(
        self,
        feature_extractor=None,
        flow=None,
        *,
        nfeatures=None,
        feature_names=None,
        nspec=None,
        context_normalizer=None,
        context_fields=None,
    ):
        super().__init__()
        feature_names = _validate_feature_schema(
            _configured_feature_names()
            if feature_names is None
            else feature_names
        )
        nfeatures = len(feature_names) if nfeatures is None else int(nfeatures)
        if nfeatures != TARGET_COUNT or nfeatures != len(feature_names):
            raise ValueError(
                f"KLNPE requires exactly {TARGET_COUNT} named targets"
            )
        self.nfeatures = nfeatures
        self.feature_names = feature_names
        self.theta_idx = resolve_feature_index(feature_names, "theta_int")
        if feature_extractor is None:
            self.feature_extractor = build_feature_extractor(
                nspec=nspec,
                context_normalizer=context_normalizer,
                context_fields=context_fields,
            )
        else:
            if context_normalizer is not None or context_fields is not None:
                raise ValueError(
                    "context options belong to the supplied feature extractor"
                )
            self.feature_extractor = feature_extractor
        declared_feature_dim = getattr(
            self.feature_extractor, "output_dim", None
        )
        if declared_feature_dim != FEATURE_DIM:
            raise ValueError(
                "pretrained feature extractor is incompatible with the current "
                f"{FEATURE_DIM}-feature CNN-CNN-metadata architecture; "
                f"got output_dim={declared_feature_dim!r}"
            )
        self.layer_norm = nn.LayerNorm(FEATURE_DIM)
        if not bool(config.train.get("feature_norm_trainable", True)):
            with torch.no_grad():
                self.layer_norm.weight.fill_(1.0)
                self.layer_norm.bias.zero_()
            self.layer_norm.requires_grad_(False)
        self.flow_context_features = FEATURE_DIM
        self.flow = (
            BoundedHybridCircularFlow(
                features=self.nfeatures,
                theta_index=self.theta_idx,
                context_features=self.flow_context_features,
                num_bounded_layers=int(config.flow["num_layers"]),
                num_theta_layers=int(config.flow["theta_num_layers"]),
                num_bins=int(config.flow["num_bins"]),
                theta_logit_limit=float(config.flow["theta_logit_limit"]),
                bounded_logit_limit=float(config.flow["bounded_logit_limit"]),
            )
            if flow is None
            else flow
        )
        if not isinstance(self.flow, nn.Module):
            raise TypeError("flow must be an nn.Module")
        self.last_training_diagnostics = {}

    def _raw_features(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        features = self.feature_extractor(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        if features.shape != (image.shape[0], FEATURE_DIM):
            raise ValueError(
                "feature extractor must return shape "
                f"({image.shape[0]}, {FEATURE_DIM}); got {tuple(features.shape)}"
            )
        return features

    def _flow_context(self, raw_features):
        return self.layer_norm(raw_features)

    def forward(
        self,
        image,
        spectra,
        true,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        if true.shape != (image.shape[0], self.nfeatures):
            raise ValueError(
                "true must have shape "
                f"({image.shape[0]}, {self.nfeatures}); got {tuple(true.shape)}"
            )
        raw_features = self._raw_features(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        flow_reference = next(
            (
                parameter
                for parameter in self.flow.parameters()
                if parameter.is_floating_point()
            ),
            None,
        )
        flow_dtype = (
            torch.float32
            if flow_reference is None
            else flow_reference.dtype
        )
        with torch.autocast(
            device_type=raw_features.device.type, enabled=False
        ):
            flow_features = raw_features.to(dtype=flow_dtype)
            context = self._flow_context(flow_features)
            flow_targets = true.to(dtype=flow_dtype)
            log_prob = self.flow.log_prob(flow_targets, context=context)
        diagnostics = {
            "raw_feature_rms": (
                raw_features.detach().float().square().mean().sqrt()
            ),
            **getattr(self.flow, "last_component_diagnostics", {}),
            **getattr(
                getattr(self.flow, "theta_transform", None),
                "last_diagnostics",
                {},
            ),
        }
        self.last_training_diagnostics = diagnostics
        return -log_prob.mean()

    def posterior_log_prob(
        self,
        image,
        spectra,
        parameters,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        """Score paired rows or candidate banks without repeating encoders.

        Supported parameter shapes are ``(B, 9)``, ``(B, S, 9)``, and—when
        the observation batch has size one—``(S, 9)``. The latter returns an
        ``(S,)`` score vector and is the inference path used for a two-view
        posterior mixture.
        """
        raw_features = self._raw_features(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        context = self._flow_context(raw_features)
        batch_size = context.shape[0]

        if parameters.ndim == 2:
            if parameters.shape[-1] != self.nfeatures:
                raise ValueError(
                    f"parameters must end in {self.nfeatures} targets"
                )
            if parameters.shape[0] == batch_size:
                score_context = context
            elif batch_size == 1:
                score_context = context.expand(parameters.shape[0], -1)
            else:
                raise ValueError(
                    "a two-dimensional candidate bank requires one observation"
                )
            return self.flow.log_prob(parameters, context=score_context)

        if parameters.ndim == 3 and parameters.shape[0] == batch_size:
            if parameters.shape[-1] != self.nfeatures:
                raise ValueError(
                    f"parameters must end in {self.nfeatures} targets"
                )
            sample_count = parameters.shape[1]
            expanded = context[:, None, :].expand(
                batch_size, sample_count, self.flow_context_features
            )
            return self.flow.log_prob(
                parameters.reshape(-1, self.nfeatures),
                context=expanded.reshape(-1, self.flow_context_features),
            ).reshape(batch_size, sample_count)
        raise ValueError(
            "parameters must have shape (B, 9), (S, 9) for B=1, or (B, S, 9)"
        )

    def sample(
        self,
        image,
        spectra,
        num_samples,
        *,
        fiber_positions=None,
        fp=None,
        observation_context,
        fiber_mask=None,
        return_log_prob=False,
    ):
        """Return posterior samples with shape ``(B, S, 9)``."""
        if fiber_positions is not None and fp is not None:
            raise ValueError("pass fiber_positions or fp, not both")
        fiber_positions = fp if fiber_positions is None else fiber_positions
        if fiber_positions is None:
            raise ValueError("fiber_positions is required")
        raw_features = self._raw_features(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        context = self._flow_context(raw_features)
        if return_log_prob:
            samples, log_prob = self.flow.sample_and_log_prob(
                num_samples, context=context
            )
            return samples, log_prob
        return self.flow.sample(num_samples, context=context)

    def extract_latent(
        self,
        image,
        spectra,
        parameters,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        if parameters.shape != (image.shape[0], self.nfeatures):
            raise ValueError(
                "parameters must have shape "
                f"({image.shape[0]}, {self.nfeatures})"
            )
        raw_features = self._raw_features(
            image,
            spectra,
            fiber_positions,
            observation_context,
            fiber_mask=fiber_mask,
        )
        context = self._flow_context(raw_features)
        return self.flow.transform_to_noise(parameters, context=context)

    def posterior_mean(self, samples):
        if samples.ndim < 2 or samples.shape[-1] != self.nfeatures:
            raise ValueError("samples must have shape (..., samples, 9)")
        mean = samples.mean(dim=-2)
        theta = samples[..., self.theta_idx]
        theta_mean = torch.atan2(
            torch.sin(math.pi * theta).mean(dim=-1),
            torch.cos(math.pi * theta).mean(dim=-1),
        ) / math.pi
        mean[..., self.theta_idx] = torch.remainder(theta_mean + 1.0, 2.0) - 1.0
        return mean
