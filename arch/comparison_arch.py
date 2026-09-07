"""Additive comparison NPE: position-queried spectra with two posterior heads.

The concat CNN-CNN-Meta path in ``networks.py`` remains the default. This
module is selected with ``--arch comparison`` (factorized shear Gaussian) or
``--arch comparison_joint`` (same encoder, concat's nine-target hybrid flow).
Fiber dropout is intentionally absent so 100k comparisons use the same
five-fiber observations as concat.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import config
    from .networks import (
        BoundedHybridCircularFlow,
        OracleContextNormalizer,
        TARGET_COUNT,
        _configured_feature_names,
        _configured_nspec,
        _validate_feature_schema,
        resolve_feature_index,
    )
except ImportError:  # Direct execution with arch/ on sys.path.
    import config
    from networks import (
        BoundedHybridCircularFlow,
        OracleContextNormalizer,
        TARGET_COUNT,
        _configured_feature_names,
        _configured_nspec,
        _validate_feature_schema,
        resolve_feature_index,
    )


PHOTO_FEATURE_DIM = 128
KINEMATIC_FEATURE_DIM = 256
CATALOG_FEATURE_DIM = 64
COMPARISON_FEATURE_DIM = (
    PHOTO_FEATURE_DIM + KINEMATIC_FEATURE_DIM + CATALOG_FEATURE_DIM
)
WAVELENGTH_COUNT = 64
POOLED_WAVELENGTH_COUNT = 16
SHEAR_INDICES = (0, 1)
NUISANCE_SLICE = slice(2, 9)


def _conv1d_bn_relu(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    )


class PhotometricShapeHead(nn.Module):
    """Small image encoder for observed shape, not the 512-d ImgCNN."""

    output_dim = PHOTO_FEATURE_DIM

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, self.output_dim)

    def forward(self, image):
        features = self.net(image).flatten(1)
        return self.proj(features)


class SharedFiberSpectrumEncoder(nn.Module):
    """Wide shared 1-D CNN with the JointSpecCNN channel schedule."""

    output_dim = KINEMATIC_FEATURE_DIM
    wavelength_count = WAVELENGTH_COUNT
    pooled_wavelength_count = POOLED_WAVELENGTH_COUNT

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            _conv1d_bn_relu(1, 16),
            _conv1d_bn_relu(16, 16),
            nn.MaxPool1d(2),
            _conv1d_bn_relu(16, 32),
            _conv1d_bn_relu(32, 32),
            nn.MaxPool1d(2),
            _conv1d_bn_relu(32, 64),
            _conv1d_bn_relu(64, 64),
            _conv1d_bn_relu(64, 128),
            _conv1d_bn_relu(128, 128),
            _conv1d_bn_relu(128, 256),
            _conv1d_bn_relu(256, 256),
            nn.Conv1d(
                256,
                self.output_dim,
                kernel_size=self.pooled_wavelength_count,
                bias=False,
            ),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, spectra):
        # spectra: (batch * n_fibers, 1, wavelength)
        if spectra.ndim != 3 or spectra.shape[1] != 1:
            raise ValueError(
                "spectra must have shape (batch, 1, wavelength); "
                f"got {tuple(spectra.shape)}"
            )
        if spectra.shape[-1] != self.wavelength_count:
            raise ValueError(
                f"expected {self.wavelength_count} wavelength bins; "
                f"got {spectra.shape[-1]}"
            )
        encoded = self.net(spectra)
        if encoded.shape[-1] != 1:
            raise RuntimeError(
                "shared fiber encoder did not collapse wavelength; "
                f"got {tuple(encoded.shape)}"
            )
        return encoded.squeeze(-1)


class PositionQueryFiberPool(nn.Module):
    """Cross-attend position queries to shared spectral tokens, then mask-pool."""

    def __init__(self, token_dim=KINEMATIC_FEATURE_DIM, num_heads=8):
        super().__init__()
        if token_dim % num_heads:
            raise ValueError("token_dim must divide num_heads")
        self.token_dim = token_dim
        self.query = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.attention = nn.MultiheadAttention(
            token_dim, num_heads, batch_first=True
        )

    def forward(self, tokens, positions, fiber_mask):
        queries = self.query(positions)
        padding_mask = ~fiber_mask
        if bool(padding_mask.all(dim=-1).any()):
            raise ValueError("fiber_mask must keep at least one fiber in every row")
        attended, _ = self.attention(
            queries,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        mixed = attended + tokens
        weights = fiber_mask.to(dtype=mixed.dtype).unsqueeze(-1)
        return (mixed * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class ComparisonFeatureExtractor(nn.Module):
    """Photometric shape plus position-queried kinematics plus catalog context."""

    output_dim = COMPARISON_FEATURE_DIM

    def __init__(
        self,
        nspec=None,
        *,
        fiber_position_scale=1.5,
        photo_net=None,
        spec_net=None,
        pool=None,
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
        self.configured_nspec = nspec
        self.fiber_position_scale = float(fiber_position_scale)
        self.photo_net = PhotometricShapeHead() if photo_net is None else photo_net
        self.spec_net = (
            SharedFiberSpectrumEncoder() if spec_net is None else spec_net
        )
        self.pool = (
            PositionQueryFiberPool() if pool is None else pool
        )
        self.catalog_net = nn.Sequential(
            nn.Linear(len(config.ORACLE_CONTEXT_FIELDS), 64),
            nn.GELU(),
            nn.Linear(64, CATALOG_FEATURE_DIM),
            nn.GELU(),
            nn.LayerNorm(CATALOG_FEATURE_DIM),
        )
        self.context_normalizer = (
            OracleContextNormalizer(context_fields=context_fields)
            if context_normalizer is None
            else context_normalizer
        )

    def _validate_inputs(self, image, spectra, fiber_positions, fiber_mask):
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("image must have shape (batch, 1, height, width)")
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError(
                "spectra must have shape (batch, 1, n_fibers, wavelength)"
            )
        n_fibers = spectra.shape[2]
        if n_fibers < 1:
            raise ValueError("at least one fiber is required")
        if spectra.shape[-1] != WAVELENGTH_COUNT:
            raise ValueError(
                f"spectra must have {WAVELENGTH_COUNT} wavelength bins; "
                f"got {spectra.shape[-1]}"
            )
        if tuple(fiber_positions.shape) != (image.shape[0], n_fibers, 2):
            raise ValueError(
                "fiber_positions must have shape "
                f"(batch, {n_fibers}, 2); got {tuple(fiber_positions.shape)}"
            )
        if fiber_mask is None:
            return torch.ones(
                image.shape[0],
                n_fibers,
                dtype=torch.bool,
                device=spectra.device,
            )
        if fiber_mask.dtype != torch.bool:
            raise TypeError("fiber_mask must be a bool tensor")
        if tuple(fiber_mask.shape) != (image.shape[0], n_fibers):
            raise ValueError(
                f"fiber_mask must have shape (batch, {n_fibers}); "
                f"got {tuple(fiber_mask.shape)}"
            )
        return fiber_mask

    def forward(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        fiber_mask = self._validate_inputs(
            image, spectra, fiber_positions, fiber_mask
        )
        batch_size, _, n_fibers, wavelength = spectra.shape
        photo = self.photo_net(F.normalize(image, dim=(-2, -1)))
        normalized_spectra = F.normalize(spectra, dim=-1)
        flat = normalized_spectra.reshape(batch_size * n_fibers, 1, wavelength)
        tokens = self.spec_net(flat).reshape(
            batch_size, n_fibers, self.spec_net.output_dim
        )
        normalized_positions = fiber_positions / self.fiber_position_scale
        kinematic = self.pool(tokens, normalized_positions, fiber_mask)
        catalog = self.catalog_net(
            self.context_normalizer(observation_context, batch_size, photo)
        )
        if photo.shape != (batch_size, PHOTO_FEATURE_DIM):
            raise ValueError(
                "photometric head must return "
                f"({batch_size}, {PHOTO_FEATURE_DIM}); got {tuple(photo.shape)}"
            )
        if kinematic.shape != (batch_size, KINEMATIC_FEATURE_DIM):
            raise ValueError(
                "kinematic pool must return "
                f"({batch_size}, {KINEMATIC_FEATURE_DIM}); "
                f"got {tuple(kinematic.shape)}"
            )
        if catalog.shape != (batch_size, CATALOG_FEATURE_DIM):
            raise ValueError(
                "catalog branch must return "
                f"({batch_size}, {CATALOG_FEATURE_DIM}); "
                f"got {tuple(catalog.shape)}"
            )
        return torch.cat((photo, kinematic, catalog), dim=-1)


class BoundedDiagonalGaussian(nn.Module):
    """Diagonal Gaussian on artanh(g) for normalized shear in [-1, 1]."""

    def __init__(self, context_features, features=2):
        super().__init__()
        if features < 1:
            raise ValueError("features must be positive")
        self.features = features
        self.context_features = context_features
        self.net = nn.Sequential(
            nn.Linear(context_features, context_features),
            nn.GELU(),
            nn.Linear(context_features, 2 * features),
        )

    def _params(self, context):
        if context.ndim != 2 or context.shape[-1] != self.context_features:
            raise ValueError(
                "context must have shape "
                f"(batch, {self.context_features}); got {tuple(context.shape)}"
            )
        loc, log_scale = self.net(context).chunk(2, dim=-1)
        log_scale = log_scale.clamp(-5.0, 2.0)
        return loc, log_scale

    def log_prob(self, values, context):
        if values.ndim != 2 or values.shape[-1] != self.features:
            raise ValueError(
                f"values must have shape (batch, {self.features}); "
                f"got {tuple(values.shape)}"
            )
        loc, log_scale = self._params(context)
        support = (
            torch.isfinite(values)
            & (values > -1.0)
            & (values < 1.0)
        ).all(dim=-1)
        safe = values.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        latent = torch.atanh(safe)
        scale = torch.exp(log_scale)
        log_pz = (
            -0.5
            * (
                ((latent - loc) / scale).square()
                + 2.0 * log_scale
                + math.log(2.0 * math.pi)
            )
        ).sum(dim=-1)
        logdet = -torch.log1p(-safe.square()).sum(dim=-1)
        return torch.where(
            support, log_pz + logdet, torch.full_like(log_pz, -torch.inf)
        )

    def sample(self, num_samples, context):
        loc, log_scale = self._params(context)
        batch_size = context.shape[0]
        noise = torch.randn(
            batch_size,
            num_samples,
            self.features,
            device=context.device,
            dtype=context.dtype,
        )
        latent = loc[:, None, :] + noise * torch.exp(log_scale)[:, None, :]
        return (1.0 - 1.0e-6) * torch.tanh(latent)


class ComparisonFactorizedFlow(nn.Module):
    """q(nuisances | x) q(g | x) with a 7-d hybrid flow and a shear Gaussian."""

    def __init__(self, context_features):
        super().__init__()
        self.features = TARGET_COUNT
        self.context_features = context_features
        self.nuisance_flow = BoundedHybridCircularFlow(
            features=7,
            theta_index=0,
            context_features=context_features,
            num_bounded_layers=int(config.flow["num_layers"]),
            num_theta_layers=int(config.flow["theta_num_layers"]),
            num_bins=int(config.flow["num_bins"]),
            theta_logit_limit=float(config.flow["theta_logit_limit"]),
            bounded_logit_limit=float(config.flow["bounded_logit_limit"]),
        )
        self.shear = BoundedDiagonalGaussian(
            context_features, features=len(SHEAR_INDICES)
        )
        self.last_component_diagnostics = {}

    @property
    def non_theta_flow(self):
        return self.nuisance_flow.non_theta_flow

    @property
    def theta_transform(self):
        return self.nuisance_flow.theta_transform

    def _split(self, inputs):
        if inputs.ndim != 2 or inputs.shape[-1] != self.features:
            raise ValueError(
                f"inputs must have shape (batch, {self.features}); "
                f"got {tuple(inputs.shape)}"
            )
        return inputs[:, list(SHEAR_INDICES)], inputs[:, NUISANCE_SLICE]

    def log_prob(self, inputs, context=None):
        shear, nuisance = self._split(inputs)
        shear_log_prob = self.shear.log_prob(shear, context)
        nuisance_log_prob = self.nuisance_flow.log_prob(nuisance, context)
        nested = dict(self.nuisance_flow.last_component_diagnostics)
        self.last_component_diagnostics = {
            "g_log_prob_mean": shear_log_prob.detach().mean(),
            **nested,
        }
        return shear_log_prob + nuisance_log_prob

    def sample(self, num_samples, context=None):
        nuisance = self.nuisance_flow.sample(num_samples, context=context)
        shear = self.shear.sample(num_samples, context)
        samples = nuisance.new_empty((*nuisance.shape[:-1], self.features))
        samples[..., list(SHEAR_INDICES)] = shear
        samples[..., NUISANCE_SLICE] = nuisance
        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        samples = self.sample(num_samples, context=context)
        batch_size = samples.shape[0]
        expanded = context[:, None, :].expand(
            batch_size, num_samples, self.context_features
        )
        log_prob = self.log_prob(
            samples.reshape(batch_size * num_samples, self.features),
            context=expanded.reshape(batch_size * num_samples, -1),
        ).reshape(batch_size, num_samples)
        return samples, log_prob

    def transform_to_noise(self, inputs, context=None):
        shear, nuisance = self._split(inputs)
        nuisance_noise = self.nuisance_flow.transform_to_noise(
            nuisance, context=context
        )
        loc, log_scale = self.shear._params(context)
        safe = shear.clamp(min=-1.0 + 1.0e-6, max=1.0 - 1.0e-6)
        latent = torch.atanh(safe)
        shear_noise = (latent - loc) * torch.exp(-log_scale)
        return torch.cat((shear_noise, nuisance_noise), dim=-1)


def _default_comparison_flow(context_features, theta_index):
    architecture = str(config.train.get("architecture", "concat"))
    if architecture == "comparison_joint":
        return BoundedHybridCircularFlow(
            features=TARGET_COUNT,
            theta_index=theta_index,
            context_features=context_features,
            num_bounded_layers=int(config.flow["num_layers"]),
            num_theta_layers=int(config.flow["theta_num_layers"]),
            num_bins=int(config.flow["num_bins"]),
            theta_logit_limit=float(config.flow["theta_logit_limit"]),
            bounded_logit_limit=float(config.flow["bounded_logit_limit"]),
        )
    return ComparisonFactorizedFlow(context_features)


class ComparisonKLNPE(nn.Module):
    """Density-only nine-target NPE with the comparison encoder.

    ``--arch comparison`` uses a factorized shear Gaussian. ``--arch
    comparison_joint`` keeps the same encoder and restores concat's nine-target
    hybrid flow so the two posteriors can be isolated.
    """

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
                f"ComparisonKLNPE requires exactly {TARGET_COUNT} named targets"
            )
        self.nfeatures = nfeatures
        self.feature_names = feature_names
        self.theta_idx = resolve_feature_index(feature_names, "theta_int")
        if feature_extractor is None:
            self.feature_extractor = ComparisonFeatureExtractor(
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
        declared = getattr(self.feature_extractor, "output_dim", None)
        if declared != COMPARISON_FEATURE_DIM:
            raise ValueError(
                "comparison extractor must declare output_dim="
                f"{COMPARISON_FEATURE_DIM}; got {declared!r}"
            )
        self.layer_norm = nn.LayerNorm(COMPARISON_FEATURE_DIM)
        if not bool(config.train.get("feature_norm_trainable", True)):
            with torch.no_grad():
                self.layer_norm.weight.fill_(1.0)
                self.layer_norm.bias.zero_()
            self.layer_norm.requires_grad_(False)
        self.flow_context_features = COMPARISON_FEATURE_DIM
        self.flow = (
            _default_comparison_flow(
                self.flow_context_features, self.theta_idx
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
        if features.shape != (image.shape[0], COMPARISON_FEATURE_DIM):
            raise ValueError(
                "feature extractor must return shape "
                f"({image.shape[0]}, {COMPARISON_FEATURE_DIM}); "
                f"got {tuple(features.shape)}"
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
            context = self._flow_context(raw_features.to(dtype=flow_dtype))
            log_prob = self.flow.log_prob(
                true.to(dtype=flow_dtype), context=context
            )
        diagnostics = {
            "raw_feature_rms": (
                raw_features.detach().float().square().mean().sqrt()
            ),
            **getattr(self.flow, "last_component_diagnostics", {}),
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
        return samples.mean(dim=1)
