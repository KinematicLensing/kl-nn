"""Small shared utilities for the current simulator-v2 pipeline."""

from __future__ import annotations

import numpy as np
import torch


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


def denormalize(samples, par_ranges, feature_names=None):
    """Map normalized ``[-1, 1]`` parameters to their physical ranges."""
    names = tuple(feature_names or par_ranges)
    values = samples.clone() if torch.is_tensor(samples) else np.array(samples, copy=True)
    if values.shape[-1] != len(names):
        raise ValueError(
            f"sample width {values.shape[-1]} does not match {len(names)} features"
        )
    for index, name in enumerate(names):
        if name not in par_ranges:
            raise KeyError(f"missing physical range for feature {name!r}")
        lower, upper = par_ranges[name]
        values[..., index] = (
            0.5 * (values[..., index] + 1.0) * (upper - lower) + lower
        )
    return values


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
