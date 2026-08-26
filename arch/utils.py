"""Small shared utilities for the current simulator pipeline."""

from __future__ import annotations

import numpy as np
import torch


SUPPORTED_TARGET_TRANSFORMS = frozenset(("identity", "log10"))


def resolve_feature_index(feature_names, target_name, aliases=None):
    """Return the unique index of a named feature, with optional aliases."""
    names = tuple(feature_names)
    candidates = (target_name, *(aliases or ()))
    matches = [index for index, name in enumerate(names) if name in candidates]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one of {candidates!r} in feature_names; "
            f"found {len(matches)} in {names!r}"
        )
    return matches[0]


def _feature_transforms(feature_names, target_transforms):
    transforms = target_transforms or {}
    unknown = set(transforms) - set(feature_names)
    if unknown:
        raise ValueError(
            f"target transforms contain unknown features: {sorted(unknown)}"
        )
    result = tuple(transforms.get(name, "identity") for name in feature_names)
    unsupported = sorted(set(result) - SUPPORTED_TARGET_TRANSFORMS)
    if unsupported:
        raise ValueError(f"unsupported target transforms: {unsupported}")
    return result


def _transform(values, transform, *, inverse=False):
    if transform == "identity":
        return values
    if transform != "log10":
        raise ValueError(f"unsupported target transform {transform!r}")
    if inverse:
        if torch.is_tensor(values):
            return torch.pow(values.new_tensor(10.0), values)
        return np.power(10.0, values)
    if torch.is_tensor(values):
        if torch.any(~torch.isfinite(values)) or torch.any(values <= 0.0):
            raise ValueError("log10 target values must be finite and positive")
        return torch.log10(values)
    values = np.asarray(values)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("log10 target values must be finite and positive")
    return np.log10(values)


def _transformed_bounds(bounds, transform):
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"target bounds must be finite and increasing: {bounds!r}")
    transformed = _transform(np.asarray([lower, upper]), transform)
    return float(transformed[0]), float(transformed[1])


def normalize_targets(
    samples,
    par_ranges,
    feature_names=None,
    target_transforms=None,
):
    """Map physical targets to ``[-1, 1]`` after named target transforms."""

    names = tuple(feature_names or par_ranges)
    values = samples.clone() if torch.is_tensor(samples) else np.array(samples, copy=True)
    if values.shape[-1] != len(names):
        raise ValueError(
            f"sample width {values.shape[-1]} does not match {len(names)} features"
        )
    transforms = _feature_transforms(names, target_transforms)
    for index, (name, transform) in enumerate(zip(names, transforms)):
        if name not in par_ranges:
            raise KeyError(f"missing physical range for feature {name!r}")
        lower, upper = _transformed_bounds(par_ranges[name], transform)
        transformed = _transform(values[..., index], transform)
        values[..., index] = (
            2.0 * transformed - (upper + lower)
        ) / (upper - lower)
    return values


def denormalize(
    samples,
    par_ranges,
    feature_names=None,
    target_transforms=None,
):
    """Map ``[-1, 1]`` targets through inverse transforms to physical units."""

    names = tuple(feature_names or par_ranges)
    values = samples.clone() if torch.is_tensor(samples) else np.array(samples, copy=True)
    if values.shape[-1] != len(names):
        raise ValueError(
            f"sample width {values.shape[-1]} does not match {len(names)} features"
        )
    transforms = _feature_transforms(names, target_transforms)
    for index, (name, transform) in enumerate(zip(names, transforms)):
        if name not in par_ranges:
            raise KeyError(f"missing physical range for feature {name!r}")
        lower, upper = _transformed_bounds(par_ranges[name], transform)
        transformed = (
            0.5 * (values[..., index] + 1.0) * (upper - lower) + lower
        )
        values[..., index] = _transform(transformed, transform, inverse=True)
    return values


def denormalization_logabsdet(
    samples,
    par_ranges,
    feature_names=None,
    target_transforms=None,
):
    """Return ``log|d physical / d normalized|`` for target-density conversion."""

    names = tuple(feature_names or par_ranges)
    physical = denormalize(
        samples,
        par_ranges,
        feature_names=names,
        target_transforms=target_transforms,
    )
    transforms = _feature_transforms(names, target_transforms)
    if torch.is_tensor(physical):
        result = torch.zeros(
            physical.shape[:-1], dtype=physical.dtype, device=physical.device
        )
    else:
        result = np.zeros(physical.shape[:-1], dtype=physical.dtype)
    for index, (name, transform) in enumerate(zip(names, transforms)):
        lower, upper = _transformed_bounds(par_ranges[name], transform)
        constant = np.log(0.5 * (upper - lower))
        if transform == "identity":
            result = result + constant
        elif torch.is_tensor(physical):
            result = (
                result
                + constant
                + np.log(np.log(10.0))
                + torch.log(physical[..., index])
            )
        else:
            result = (
                result
                + constant
                + np.log(np.log(10.0))
                + np.log(physical[..., index])
            )
    return result


def img_to_gal_axis(g1, g2, theta):
    """Convert image-frame shear to the simulator's clockwise galaxy frame.

    The simulator defines positive ``theta`` clockwise, while positive ``g2``
    points toward the conventional ``+pi/4`` image direction. Consequently,
    at ``theta=+pi/4``, positive ``g2`` maps to negative ``g_plus``.
    """
    if torch.is_tensor(g1):
        cosine, sine = torch.cos(2 * theta), torch.sin(2 * theta)
    else:
        cosine, sine = np.cos(2 * theta), np.sin(2 * theta)
    return g1 * cosine - g2 * sine, g1 * sine + g2 * cosine


def gal_to_img_axis(g_plus, g_cross, theta):
    """Inverse of :func:`img_to_gal_axis`."""
    if torch.is_tensor(g_plus):
        cosine, sine = torch.cos(2 * theta), torch.sin(2 * theta)
    else:
        cosine, sine = np.cos(2 * theta), np.sin(2 * theta)
    return (
        g_plus * cosine + g_cross * sine,
        -g_plus * sine + g_cross * cosine,
    )
