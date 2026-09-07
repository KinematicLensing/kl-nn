"""Additive theory-inspired KL geometric NPE.

Concat CNN-CNN-Meta remains the default. This module is selected with
``--arch kl_geom``. It measures photometric quadrupole, fiber position angle,
and H-alpha centroids, applies Xu et al. 2024's first-order shear map, and
lets the nine-target hybrid flow model a residual around that estimate.
"""

from __future__ import annotations

import math

import torch
from torch import nn

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
    from .utils import gal_to_img_axis
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
    from utils import gal_to_img_axis


WAVELENGTH_COUNT = 64
WAVELENGTH_MIN_NM = 851.0
WAVELENGTH_MAX_NM = 855.81
HALPHA_REST_NM = 656.4589
SIMULATOR_REDSHIFT = 0.3
HALPHA_OBS_NM = HALPHA_REST_NM * (1.0 + SIMULATOR_REDSHIFT)
SPEED_OF_LIGHT_KMS = 299792.458
MAJOR_PLUS_INDEX = 0
MAJOR_MINUS_INDEX = 1
CENTER_FIBER_INDEX = 2
MINOR_PLUS_INDEX = 3
MINOR_MINUS_INDEX = 4
STAT_DIM = 9
GEOM_CONTEXT_DIM = 64
SHEAR_SCALE = 0.1
RESIDUAL_CLAMP = 1.0 - 1.0e-4
EPS_E = 1.0e-3
EPS_V_KMS = 1.0
EPS_SINI = 1.0e-3
VCIRC_FLOOR_KMS = 1.0


def _disable_dynamo(fn):
    """Keep classical stats/Xu-map out of torch.compile graphs.

    ``denormalize`` and moment construction use Python bound lookups that
    Dynamo treats as new graphs. Compiling them desynchronizes DDP ranks.
    """

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(fn) if disable is not None else fn


def wavelength_grid_nm(wavelength_count=WAVELENGTH_COUNT, *, device=None, dtype=None):
    return torch.linspace(
        WAVELENGTH_MIN_NM,
        WAVELENGTH_MAX_NM,
        int(wavelength_count),
        device=device,
        dtype=dtype,
    )


def wavelength_to_velocity_kms(wavelength_nm):
    return SPEED_OF_LIGHT_KMS * (wavelength_nm - HALPHA_OBS_NM) / HALPHA_OBS_NM


def thin_disk_e_int(cosi):
    safe = cosi.clamp(min=0.0, max=1.0)
    return (1.0 - safe) / (1.0 + safe).clamp_min(EPS_E)


@_disable_dynamo
def photometric_quadrupole(image, *, pixel_scale_arcsec=None, psf_fwhm_arcsec=None):
    """Return PSF-corrected reduced ellipticity ``(e1, e2, e_obs)``.

    Weighted second moments yield the polarization
    chi = (Qxx-Qyy, 2 Qxy) / (Qxx+Qyy). Xu et al. 2024 and the thin-disk
    e_int = (1-q)/(1+q) use reduced ellipticity
    eps = chi / (1 + sqrt(1 - |chi|^2)).
    """

    if image.ndim == 3:
        intensity = image
    elif image.ndim == 4 and image.shape[1] == 1:
        intensity = image[:, 0]
    else:
        raise ValueError(
            "image must have shape (batch, height, width) or (batch, 1, height, width)"
        )
    batch, height, width = intensity.shape
    intensity = intensity.float()
    weights = intensity.clamp_min(0.0)
    weight_sum = weights.reshape(batch, -1).sum(dim=-1).clamp_min(1.0e-12)
    yy = torch.arange(height, device=image.device, dtype=intensity.dtype)
    xx = torch.arange(width, device=image.device, dtype=intensity.dtype)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)
    xbar = (weights * grid_x).reshape(batch, -1).sum(dim=-1) / weight_sum
    ybar = (weights * grid_y).reshape(batch, -1).sum(dim=-1) / weight_sum
    dx = grid_x - xbar[:, None, None]
    dy = grid_y - ybar[:, None, None]
    qxx = (weights * dx * dx).reshape(batch, -1).sum(dim=-1) / weight_sum
    qyy = (weights * dy * dy).reshape(batch, -1).sum(dim=-1) / weight_sum
    qxy = (weights * dx * dy).reshape(batch, -1).sum(dim=-1) / weight_sum
    if pixel_scale_arcsec is None:
        pixel_scale_arcsec = float(config.observation["image_pixel_scale_arcsec"])
    if psf_fwhm_arcsec is None:
        psf_fwhm_arcsec = float(
            config.observation["image_reference_psf_fwhm_arcsec"]
        )
    sigma_psf = (float(psf_fwhm_arcsec) / float(pixel_scale_arcsec)) / (
        2.0 * math.sqrt(2.0 * math.log(2.0))
    )
    psf_q = intensity.new_tensor(sigma_psf * sigma_psf)
    qxx_gal = qxx - psf_q
    qyy_gal = qyy - psf_q
    trace = qxx_gal + qyy_gal
    use_corrected = trace > EPS_E
    qxx_use = torch.where(use_corrected, qxx_gal, qxx)
    qyy_use = torch.where(use_corrected, qyy_gal, qyy)
    qxy_use = torch.where(use_corrected, qxy, qxy)
    denom = (qxx_use + qyy_use).clamp_min(EPS_E)
    chi1 = (qxx_use - qyy_use) / denom
    chi2 = (2.0 * qxy_use) / denom
    chi = torch.sqrt((chi1 * chi1 + chi2 * chi2).clamp_min(0.0)).clamp(max=0.99)
    scale = 1.0 / (1.0 + torch.sqrt((1.0 - chi * chi).clamp_min(0.0)))
    e1 = chi1 * scale
    e2 = chi2 * scale
    e_obs = (chi * scale).clamp(max=0.99)
    return e1, e2, e_obs


def flux_weighted_centroid_nm(spectra, wavelength_nm=None):
    """Return a differentiable H-alpha centroid in nm for each fiber."""

    if spectra.ndim != 4 or spectra.shape[1] != 1:
        raise ValueError(
            "spectra must have shape (batch, 1, n_fibers, wavelength)"
        )
    batch, _, n_fibers, nwave = spectra.shape
    flux = spectra[:, 0].float().clamp_min(0.0)
    grid = (
        wavelength_grid_nm(nwave, device=spectra.device, dtype=flux.dtype)
        if wavelength_nm is None
        else wavelength_nm.to(device=spectra.device, dtype=flux.dtype)
    )
    if grid.shape != (nwave,):
        raise ValueError("wavelength grid must match the spectral axis")
    weights = flux.reshape(batch * n_fibers, nwave)
    total = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    centroid = (weights * grid).sum(dim=-1, keepdim=True) / total
    return centroid.reshape(batch, n_fibers)


def fiber_position_angle(fiber_positions):
    """Observed photometric PA from the stored +major fiber offset."""

    if fiber_positions.ndim != 3 or fiber_positions.shape[-1] != 2:
        raise ValueError("fiber_positions must have shape (batch, n_fibers, 2)")
    major = fiber_positions[:, MAJOR_PLUS_INDEX]
    return torch.atan2(major[..., 1], major[..., 0])


def odd_axis_velocities(velocity_kms):
    """Return (v0, v_major, v_minor) from five galaxy-axis fibers."""

    if velocity_kms.ndim != 2 or velocity_kms.shape[-1] < 5:
        raise ValueError("velocity_kms must have shape (batch, n_fibers>=5)")
    v0 = velocity_kms[:, CENTER_FIBER_INDEX]
    v_major = 0.5 * (
        velocity_kms[:, MAJOR_PLUS_INDEX] - velocity_kms[:, MAJOR_MINUS_INDEX]
    )
    v_minor = 0.5 * (
        velocity_kms[:, MINOR_PLUS_INDEX] - velocity_kms[:, MINOR_MINUS_INDEX]
    )
    return v0, v_major, v_minor


@_disable_dynamo
def xu_reduced_shear(e_obs, theta_phot, v_major, v_minor, vcirc):
    """Xu et al. 2024 first-order (g1, g2) in the sky frame, in physical units.

    ``g_x`` keeps the sign of ``v_minor / v_major``. Face-on and tiny
    ``v_major`` rows return zero rather than an unstable ratio.
    """

    e_obs_safe = e_obs.clamp(min=EPS_E, max=0.99)
    vcirc_safe = vcirc.clamp_min(VCIRC_FLOOR_KMS)
    kinematic = v_major.abs() >= EPS_V_KMS
    sini = (v_major.abs() / vcirc_safe).clamp(min=EPS_SINI, max=1.0 - 1.0e-4)
    cosi = torch.sqrt((1.0 - sini * sini).clamp_min(0.0))
    e_int = thin_disk_e_int(cosi)
    gplus = (e_obs_safe * e_obs_safe - e_int * e_int) / (
        2.0 * e_obs_safe * e_obs_safe * (1.0 - e_int * e_int).clamp_min(EPS_E)
    )
    v_major_safe = torch.where(
        kinematic, v_major, torch.ones_like(v_major) * EPS_V_KMS
    )
    gcross = (
        -(v_minor / v_major_safe)
        * (2.0 * e_int)
        / (
            cosi.clamp_min(EPS_SINI)
            * (2.0 * e_int + 1.0 + e_obs_safe * e_obs_safe)
        )
    )
    gplus = torch.where(kinematic, gplus, torch.zeros_like(gplus))
    gcross = torch.where(kinematic, gcross, torch.zeros_like(gcross))
    g1, g2 = gal_to_img_axis(gplus, gcross, theta_phot)
    return g1.clamp(-0.15, 0.15), g2.clamp(-0.15, 0.15)


@_disable_dynamo
def _physical_vcirc(normalized_parameters):
    """Affine identity denormalization of ``vcirc`` only.

    The full nine-target ``denormalize`` helper re-reads every feature bound
    with Python floats. Dynamo compiles that as a new graph per bound and
    hits ``recompile_limit``, which desynchronizes DDP allreduces.
    """

    index = resolve_feature_index(config.TARGET_NAMES, "vcirc")
    transform = config.TARGET_TRANSFORMS.get("vcirc", "identity")
    if transform != "identity":
        raise ValueError("kl_geom requires identity vcirc")
    lower, upper = config.par_ranges["vcirc"]
    lower_f = float(lower)
    upper_f = float(upper)
    if not math.isfinite(lower_f) or not math.isfinite(upper_f) or lower_f >= upper_f:
        raise ValueError("vcirc bounds must be finite and increasing")
    return (
        0.5 * (normalized_parameters[..., index] + 1.0) * (upper_f - lower_f)
        + lower_f
    )


def _normalized_shear(g_physical):
    return g_physical / SHEAR_SCALE


class FiberCentroidResidual(nn.Module):
    """Tiny shared 1-D residual on classical per-fiber centroids, in nm."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(8, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, spectra):
        if spectra.ndim != 4 or spectra.shape[1] != 1:
            raise ValueError(
                "spectra must have shape (batch, 1, n_fibers, wavelength)"
            )
        batch, _, n_fibers, nwave = spectra.shape
        tokens = spectra.reshape(batch * n_fibers, 1, nwave)
        delta_nm = self.net(tokens).reshape(batch, n_fibers)
        return 0.1 * delta_nm


class GeometricStatEncoder(nn.Module):
    """Map classical KL stats plus oracle context to a small flow condition."""

    output_dim = GEOM_CONTEXT_DIM

    def __init__(self, context_normalizer=None, context_fields=None):
        super().__init__()
        if context_normalizer is None:
            self.context_normalizer = OracleContextNormalizer(
                context_fields=context_fields
            )
        else:
            if context_fields is not None:
                raise ValueError(
                    "context_fields belong to the supplied context normalizer"
                )
            self.context_normalizer = context_normalizer
        self.centroid_residual = FiberCentroidResidual()
        self.mlp = nn.Sequential(
            nn.Linear(STAT_DIM, GEOM_CONTEXT_DIM),
            nn.GELU(),
            nn.Linear(GEOM_CONTEXT_DIM, GEOM_CONTEXT_DIM),
        )

    @_disable_dynamo
    def sufficient_stats(self, image, spectra, fiber_positions):
        _e1, _e2, e_obs = photometric_quadrupole(image)
        theta_phot = fiber_position_angle(fiber_positions)
        centroid_nm = flux_weighted_centroid_nm(spectra) + self.centroid_residual(
            spectra
        )
        velocity = wavelength_to_velocity_kms(centroid_nm)
        v0, v_major, v_minor = odd_axis_velocities(velocity)
        return {
            "e_obs": e_obs,
            "theta_phot": theta_phot,
            "v0": v0,
            "v_major": v_major,
            "v_minor": v_minor,
        }

    def _stat_vector(self, stats, observation_context, image):
        oracle = self.context_normalizer(observation_context, image.shape[0], image)
        two_theta = 2.0 * stats["theta_phot"]
        parts = (
            stats["e_obs"].unsqueeze(-1),
            torch.cos(two_theta).unsqueeze(-1),
            torch.sin(two_theta).unsqueeze(-1),
            (stats["v0"] / 30.0).unsqueeze(-1),
            (stats["v_major"] / 200.0).unsqueeze(-1),
            (stats["v_minor"] / 50.0).unsqueeze(-1),
            oracle,
        )
        return torch.cat(parts, dim=-1)

    def forward(self, image, spectra, fiber_positions, observation_context):
        stats = self.sufficient_stats(image, spectra, fiber_positions)
        features = self.mlp(self._stat_vector(stats, observation_context, image))
        return features, stats


def _expand_stat(stat, batch_size, sample_count):
    if stat.shape[0] == batch_size and sample_count is None:
        return stat
    if stat.shape[0] == 1 and sample_count is None:
        return stat.expand(batch_size)
    if sample_count is None:
        raise ValueError("stat batch does not match parameters")
    if stat.shape[0] == batch_size:
        return stat[:, None].expand(batch_size, sample_count)
    if stat.shape[0] == 1:
        return stat[:, None].expand(batch_size, sample_count)
    raise ValueError("stat batch does not match parameters")


def _broadcast_stats(stats, batch_size, sample_count=None):
    return {
        name: _expand_stat(value, batch_size, sample_count)
        for name, value in stats.items()
    }


@_disable_dynamo
def geometric_ghat_normalized(stats, normalized_parameters):
    vcirc = _physical_vcirc(normalized_parameters)
    g1, g2 = xu_reduced_shear(
        stats["e_obs"],
        stats["theta_phot"],
        stats["v_major"],
        stats["v_minor"],
        vcirc,
    )
    ghat = torch.stack((_normalized_shear(g1), _normalized_shear(g2)), dim=-1)
    return torch.nan_to_num(ghat, nan=0.0, posinf=1.5, neginf=-1.5)


@_disable_dynamo
def parameters_to_flow_targets(parameters, stats):
    residual = parameters[..., :2] - geometric_ghat_normalized(stats, parameters)
    residual = torch.nan_to_num(residual, nan=0.0)
    clipped = residual.clamp(-RESIDUAL_CLAMP, RESIDUAL_CLAMP)
    targets = parameters.clone()
    targets[..., :2] = clipped
    clip_fraction = (residual.abs() > RESIDUAL_CLAMP).float().mean()
    return targets, clip_fraction


@_disable_dynamo
def flow_targets_to_parameters(flow_samples, stats):
    ghat = geometric_ghat_normalized(stats, flow_samples)
    composed = flow_samples.clone()
    composed[..., :2] = (ghat + flow_samples[..., :2]).clamp(
        -RESIDUAL_CLAMP, RESIDUAL_CLAMP
    )
    return composed


class GeometricKLNPE(nn.Module):
    """Nine-target NPE with Xu-map residual shear."""

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
                f"GeometricKLNPE requires exactly {TARGET_COUNT} named targets"
            )
        self.nfeatures = nfeatures
        self.feature_names = feature_names
        self.theta_idx = resolve_feature_index(feature_names, "theta_int")
        if nspec is not None and int(nspec) != _configured_nspec():
            raise ValueError("kl_geom uses the current five-fiber spectral schema")
        if feature_extractor is None:
            self.feature_extractor = GeometricStatEncoder(
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
        if declared != GEOM_CONTEXT_DIM:
            raise ValueError(
                "kl_geom encoder must declare output_dim="
                f"{GEOM_CONTEXT_DIM}; got {declared!r}"
            )
        self.layer_norm = nn.LayerNorm(GEOM_CONTEXT_DIM)
        if not bool(config.train.get("feature_norm_trainable", True)):
            with torch.no_grad():
                self.layer_norm.weight.fill_(1.0)
                self.layer_norm.bias.zero_()
            self.layer_norm.requires_grad_(False)
        self.flow_context_features = GEOM_CONTEXT_DIM
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

    def _encode(self, image, spectra, fiber_positions, observation_context):
        with torch.autocast(device_type=image.device.type, enabled=False):
            features, stats = self.feature_extractor(
                image, spectra, fiber_positions, observation_context
            )
            features = features.float()
            stats = {name: value.float() for name, value in stats.items()}
        if features.shape != (image.shape[0], GEOM_CONTEXT_DIM):
            raise ValueError(
                "feature extractor must return shape "
                f"({image.shape[0]}, {GEOM_CONTEXT_DIM}); "
                f"got {tuple(features.shape)}"
            )
        return features, stats

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
        del fiber_mask
        if true.shape != (image.shape[0], self.nfeatures):
            raise ValueError(
                "true must have shape "
                f"({image.shape[0]}, {self.nfeatures}); got {tuple(true.shape)}"
            )
        raw_features, stats = self._encode(
            image, spectra, fiber_positions, observation_context
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
            torch.float32 if flow_reference is None else flow_reference.dtype
        )
        with torch.autocast(
            device_type=raw_features.device.type, enabled=False
        ):
            context = self._flow_context(raw_features.to(dtype=flow_dtype))
            stats = {
                name: value.to(dtype=flow_dtype) for name, value in stats.items()
            }
            flow_targets, clip_fraction = parameters_to_flow_targets(
                true.to(dtype=flow_dtype), stats
            )
            log_prob = self.flow.log_prob(flow_targets, context=context)
        ghat = geometric_ghat_normalized(stats, true.to(dtype=flow_dtype))
        residual = flow_targets[..., :2]
        diagnostics = {
            "raw_feature_rms": (
                raw_features.detach().float().square().mean().sqrt()
            ),
            "ghat_rms": ghat.detach().float().square().mean().sqrt(),
            "residual_rms": residual.detach().float().square().mean().sqrt(),
            "residual_clip_fraction": clip_fraction.detach().float(),
            **getattr(self.flow, "last_component_diagnostics", {}),
        }
        self.last_training_diagnostics = diagnostics
        return -log_prob.mean()

    def _score_parameters(self, parameters, context, stats):
        if parameters.ndim == 2:
            if parameters.shape[-1] != self.nfeatures:
                raise ValueError(
                    f"parameters must end in {self.nfeatures} targets"
                )
            batch_size = context.shape[0]
            if parameters.shape[0] == batch_size:
                score_context = context
                score_stats = stats
            elif batch_size == 1:
                score_context = context.expand(parameters.shape[0], -1)
                score_stats = _broadcast_stats(stats, parameters.shape[0])
            else:
                raise ValueError(
                    "a two-dimensional candidate bank requires one observation"
                )
            flow_targets, _ = parameters_to_flow_targets(parameters, score_stats)
            return self.flow.log_prob(flow_targets, context=score_context)
        if parameters.ndim == 3 and parameters.shape[0] == context.shape[0]:
            if parameters.shape[-1] != self.nfeatures:
                raise ValueError(
                    f"parameters must end in {self.nfeatures} targets"
                )
            batch_size, sample_count, _ = parameters.shape
            expanded = context[:, None, :].expand(
                batch_size, sample_count, self.flow_context_features
            )
            score_stats = _broadcast_stats(stats, batch_size, sample_count)
            flow_targets, _ = parameters_to_flow_targets(parameters, score_stats)
            return self.flow.log_prob(
                flow_targets.reshape(-1, self.nfeatures),
                context=expanded.reshape(-1, self.flow_context_features),
            ).reshape(batch_size, sample_count)
        raise ValueError(
            "parameters must have shape (B, 9), (S, 9) for B=1, or (B, S, 9)"
        )

    def posterior_log_prob(
        self,
        image,
        spectra,
        parameters,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        del fiber_mask
        raw_features, stats = self._encode(
            image, spectra, fiber_positions, observation_context
        )
        context = self._flow_context(raw_features)
        return self._score_parameters(parameters, context, stats)

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
        del fiber_mask
        if fiber_positions is not None and fp is not None:
            raise ValueError("pass fiber_positions or fp, not both")
        fiber_positions = fp if fiber_positions is None else fiber_positions
        if fiber_positions is None:
            raise ValueError("fiber_positions is required")
        raw_features, stats = self._encode(
            image, spectra, fiber_positions, observation_context
        )
        context = self._flow_context(raw_features)
        if return_log_prob:
            flow_samples, log_prob = self.flow.sample_and_log_prob(
                num_samples, context=context
            )
            batch_size = flow_samples.shape[0]
            score_stats = _broadcast_stats(stats, batch_size, num_samples)
            return flow_targets_to_parameters(flow_samples, score_stats), log_prob
        flow_samples = self.flow.sample(num_samples, context=context)
        batch_size = flow_samples.shape[0]
        score_stats = _broadcast_stats(stats, batch_size, num_samples)
        return flow_targets_to_parameters(flow_samples, score_stats)

    def extract_latent(
        self,
        image,
        spectra,
        parameters,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        del fiber_mask
        if parameters.shape != (image.shape[0], self.nfeatures):
            raise ValueError(
                "parameters must have shape "
                f"({image.shape[0]}, {self.nfeatures})"
            )
        raw_features, stats = self._encode(
            image, spectra, fiber_positions, observation_context
        )
        context = self._flow_context(raw_features)
        flow_targets, _ = parameters_to_flow_targets(parameters, stats)
        return self.flow.transform_to_noise(flow_targets, context=context)

    def posterior_mean(self, samples):
        return samples.mean(dim=1)
