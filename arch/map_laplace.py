"""Laplace covariance of a physical TF-target sample MAP.

The cached MAP is the highest-scoring identity/R90 mixture sample after the
physical Jacobian and TF prior replacement. The Laplace pair is the 2x2 shear
block of the inverse Hessian of that same log-density, evaluated at the sample
MAP. Encoder context is treated as constant: only the flow, Jacobian, and TF
ratio are differentiated.
"""

from __future__ import annotations

import math

import numpy as np
import torch

try:
    from . import config
    from .data import rotate_90_parameters
    from .tf_prior import TFPrior
    from .utils import (
        denormalization_logabsdet,
        normalize_targets,
        resolve_feature_index,
    )
except ImportError:  # Direct execution from arch/.
    import config
    from data import rotate_90_parameters
    from tf_prior import TFPrior
    from utils import (
        denormalization_logabsdet,
        normalize_targets,
        resolve_feature_index,
    )


BOUND_EPS = 1e-3
HESSIAN_EIG_MIN = 1e-12


def _log_standard_normal_interval_torch(
    lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
    """Return ``log(Phi(upper) - Phi(lower))`` without tail cancellation."""

    if torch.any(lower >= upper):
        raise ValueError("normal interval lower bound must be below upper bound")

    def log_difference(log_larger, log_smaller):
        return log_larger + torch.log(-torch.expm1(log_smaller - log_larger))

    log_ndtr = torch.special.log_ndtr
    result = torch.empty_like(lower)
    negative = upper <= 0.0
    positive = lower >= 0.0
    crossing = ~(negative | positive)
    if torch.any(negative):
        result = torch.where(
            negative,
            log_difference(log_ndtr(upper), log_ndtr(lower)),
            result,
        )
    if torch.any(positive):
        result = torch.where(
            positive,
            log_difference(log_ndtr(-lower), log_ndtr(-upper)),
            result,
        )
    if torch.any(crossing):
        result = torch.where(
            crossing,
            log_difference(log_ndtr(upper), log_ndtr(lower)),
            result,
        )
    return result


def truncated_tf_log_prob_torch(
    vcirc: torch.Tensor,
    rmag_true: torch.Tensor,
    prior: TFPrior,
) -> torch.Tensor:
    """Torch twin of :func:`tf_prior.truncated_tf_log_prob`."""

    velocity = torch.as_tensor(vcirc)
    magnitude = torch.as_tensor(rmag_true, device=velocity.device, dtype=velocity.dtype)
    if not torch.isfinite(magnitude).all():
        raise ValueError("rmag_true must contain only finite values")
    velocity, magnitude = torch.broadcast_tensors(velocity, magnitude)

    mean_log10 = (magnitude - prior.intercept) / prior.slope
    sigma = prior.scatter_dex
    lower = (math.log10(prior.vcirc_min) - mean_log10) / sigma
    upper = (math.log10(prior.vcirc_max) - mean_log10) / sigma
    log_truncation = _log_standard_normal_interval_torch(lower, upper)

    on_support = (
        torch.isfinite(velocity)
        & (velocity >= prior.vcirc_min)
        & (velocity <= prior.vcirc_max)
    )
    safe_velocity = torch.where(on_support, velocity, torch.ones_like(velocity))
    standardized = (torch.log10(safe_velocity) - mean_log10) / sigma
    log_density = (
        -0.5 * torch.square(standardized)
        - math.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - log_truncation
        - torch.log(safe_velocity)
        - math.log(math.log(10.0))
    )
    return torch.where(on_support, log_density, torch.full_like(log_density, -math.inf))


def tf_log_prior_ratio_torch(
    vcirc: torch.Tensor,
    rmag_true: torch.Tensor,
    prior: TFPrior,
) -> torch.Tensor:
    """Return ``log p_TF(v|m_true) - log p_uniform(v)``."""

    return truncated_tf_log_prob_torch(vcirc, rmag_true, prior) - prior.base_log_density


def mixture_normalized_log_prob(
    flow,
    parameters_normalized: torch.Tensor,
    context_original: torch.Tensor,
    context_rotated: torch.Tensor,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> torch.Tensor:
    """Identity/R90 mixture log-density in normalized coordinates."""

    if parameters_normalized.ndim == 1:
        parameters_normalized = parameters_normalized.unsqueeze(0)
    names = tuple(feature_names or config.train["feature_names"])
    log_original = flow.log_prob(parameters_normalized, context=context_original)
    rotated = rotate_90_parameters(
        parameters_normalized, feature_names=names
    )
    log_rotated = flow.log_prob(rotated, context=context_rotated)
    return torch.logsumexp(torch.stack((log_original, log_rotated)), dim=0) - math.log(
        2.0
    )


def physical_tf_target_log_prob(
    flow,
    parameters_physical: torch.Tensor,
    context_original: torch.Tensor,
    context_rotated: torch.Tensor,
    rmag_true: torch.Tensor,
    prior: TFPrior,
    *,
    par_ranges,
    feature_names: tuple[str, ...] | list[str],
    target_transforms,
) -> torch.Tensor:
    """Physical TF-target log-density whose argmax is the cached MAP."""

    names = tuple(feature_names)
    if parameters_physical.ndim == 1:
        physical = parameters_physical.unsqueeze(0)
        squeeze = True
    else:
        physical = parameters_physical
        squeeze = False
    normalized = normalize_targets(
        physical,
        par_ranges=par_ranges,
        feature_names=names,
        target_transforms=target_transforms,
    )
    context_original = context_original.expand(normalized.shape[0], -1)
    context_rotated = context_rotated.expand(normalized.shape[0], -1)
    log_mix = mixture_normalized_log_prob(
        flow,
        normalized,
        context_original,
        context_rotated,
        feature_names=names,
    )
    log_phys = log_mix - denormalization_logabsdet(
        normalized,
        par_ranges=par_ranges,
        feature_names=names,
        target_transforms=target_transforms,
    )
    vcirc_index = resolve_feature_index(names, "vcirc")
    log_tf = tf_log_prior_ratio_torch(
        physical[..., vcirc_index], rmag_true, prior
    )
    log_target = log_phys + log_tf
    return log_target.squeeze(0) if squeeze else log_target


def near_normalized_bound(
    parameters_normalized: torch.Tensor,
    *,
    feature_names: tuple[str, ...] | list[str],
    eps: float = BOUND_EPS,
) -> bool:
    """True when a non-circular coordinate sits on the training cube wall."""

    names = tuple(feature_names)
    theta_index = resolve_feature_index(names, "theta_int")
    values = parameters_normalized.reshape(-1)
    for index, value in enumerate(values):
        if index == theta_index:
            continue
        if not torch.isfinite(value) or abs(float(value)) >= 1.0 - eps:
            return True
    return False


def laplace_covariance_from_log_prob(log_prob_fn, parameters: torch.Tensor):
    """Return ``(H^{-1})_{gg}`` and an ok flag for a scalar log-density."""

    parameters = torch.as_tensor(parameters).reshape(-1)
    n_features = int(parameters.numel())
    invalid = (
        np.full((2, 2), np.nan, dtype=np.float64),
        False,
    )
    if n_features < 2 or not torch.isfinite(parameters).all():
        return invalid

    def scalar_log_prob(theta):
        value = log_prob_fn(theta)
        return value.reshape(())

    try:
        with torch.enable_grad():
            location = parameters.detach().requires_grad_(True)
            value = scalar_log_prob(location)
            if not torch.isfinite(value):
                return invalid
            hessian = torch.autograd.functional.hessian(
                scalar_log_prob, location.detach()
            )
    except RuntimeError:
        return invalid

    hessian = hessian.detach()
    information = -hessian
    if information.shape != (n_features, n_features):
        return invalid
    if not torch.isfinite(information).all():
        return invalid
    eigenvalues = torch.linalg.eigvalsh(information)
    if not torch.isfinite(eigenvalues).all() or bool(
        torch.any(eigenvalues <= HESSIAN_EIG_MIN)
    ):
        return invalid
    try:
        covariance = torch.linalg.inv(information)
    except RuntimeError:
        return invalid
    shear = covariance[:2, :2]
    if not torch.isfinite(shear).all():
        return invalid
    return shear.cpu().numpy().astype(np.float64), True


def tf_target_laplace_covariances(
    model,
    map_physical: np.ndarray,
    context_original: np.ndarray,
    context_rotated: np.ndarray,
    rmag_true: np.ndarray,
    prior: TFPrior,
    *,
    par_ranges=None,
    feature_names=None,
    target_transforms=None,
    device=None,
):
    """Evaluate the Laplace shear covariance at every TF-target sample MAP."""

    maps = np.asarray(map_physical, dtype=np.float64)
    original = np.asarray(context_original)
    rotated = np.asarray(context_rotated)
    magnitudes = np.asarray(rmag_true, dtype=np.float64)
    n_galaxies = maps.shape[0]
    if original.shape[0] != n_galaxies or rotated.shape[0] != n_galaxies:
        raise ValueError("flow contexts must contain one row per MAP")
    if magnitudes.shape != (n_galaxies,):
        raise ValueError("rmag_true must contain one value per galaxy")
    names = tuple(feature_names or config.train["feature_names"])
    ranges = par_ranges if par_ranges is not None else config.par_ranges
    transforms = (
        target_transforms
        if target_transforms is not None
        else config.TARGET_TRANSFORMS
    )
    if device is None:
        flow_device = next(model.flow.parameters()).device
    else:
        flow_device = torch.device(device)
    flow_dtype = next(model.flow.parameters()).dtype
    covariances = np.full((n_galaxies, 2, 2), np.nan, dtype=np.float64)
    ok = np.zeros(n_galaxies, dtype=bool)
    flow = model.flow

    for index in range(n_galaxies):
        physical = torch.as_tensor(
            maps[index], device=flow_device, dtype=torch.float64
        )
        if not torch.isfinite(physical).all():
            continue
        normalized = normalize_targets(
            physical.unsqueeze(0),
            par_ranges=ranges,
            feature_names=names,
            target_transforms=transforms,
        )[0]
        if near_normalized_bound(normalized, feature_names=names):
            continue
        context_0 = torch.as_tensor(
            original[index], device=flow_device, dtype=flow_dtype
        ).unsqueeze(0)
        context_r = torch.as_tensor(
            rotated[index], device=flow_device, dtype=flow_dtype
        ).unsqueeze(0)
        magnitude = torch.as_tensor(
            magnitudes[index], device=flow_device, dtype=torch.float64
        )

        def log_prob(theta, ctx0=context_0, ctxr=context_r, mag=magnitude):
            theta = theta.to(dtype=torch.float64)
            return physical_tf_target_log_prob(
                flow,
                theta,
                ctx0.to(dtype=theta.dtype),
                ctxr.to(dtype=theta.dtype),
                mag.to(dtype=theta.dtype),
                prior,
                par_ranges=ranges,
                feature_names=names,
                target_transforms=transforms,
            )

        shear_cov, valid = laplace_covariance_from_log_prob(log_prob, physical)
        covariances[index] = shear_cov
        ok[index] = valid
    return covariances, ok
