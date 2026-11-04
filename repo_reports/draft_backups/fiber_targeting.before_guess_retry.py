"""Fiber targeting from a noisy image (``measured_axis`` fiber layout).

Real fiber surveys place offset fibers along a galaxy's *photometrically
measured* axes. DESI-PV, for example, uses the Siena Galaxy Atlas position
angle, which comes from second moments of the Legacy Surveys light
distribution. The ``galaxy_axis`` layout instead uses the noiseless true
sheared axes, which hands the network a noise-free shear direction for nearly
round galaxies.

This module emulates targeting:

1. Add one fixed noise realization to the clean r-band image, using the
   Legacy Surveys DR9 r-band CCD noise model from ``generate_fits.py``
   (``galsim.CCDNoise``, as kl-tools would use if noise were on).
2. Measure the position angle from adaptive second moments
   (``galsim.hsm.FindAdaptiveMom``).
3. Place the five fibers along the measured axes.

The targeting noise is independent of the image noise the network sees during
training, which is redrawn every epoch. Real targeting likewise comes from a
separate, deeper catalog measurement.
"""

from __future__ import annotations

import numpy as np

# Legacy Surveys DR9 r band, as configured in generate_fits.py.
LS_R_SKY_LEVEL = 44.54  # per second, before dividing by gain
LS_R_EXPTIME = 60.0  # seconds
LS_R_GAIN = 4.0
LS_R_READ_NOISE = 2.6

# Seeds for the targeting noise are offset so they never collide with the
# training or inference noise streams.
TARGETING_SEED_OFFSET = 900_000_000


def targeting_seed(sample_id: int) -> int:
    return TARGETING_SEED_OFFSET + int(sample_id)


def targeting_noisy_image(
    clean_image: np.ndarray,
    *,
    seed: int,
    sky_level: float = LS_R_SKY_LEVEL,
    exptime: float = LS_R_EXPTIME,
    gain: float = LS_R_GAIN,
    read_noise: float = LS_R_READ_NOISE,
) -> np.ndarray:
    """Return ``clean_image`` plus one CCD-noise realization (sky subtracted)."""

    import galsim

    image = galsim.Image(np.array(clean_image, dtype=np.float64, copy=True))
    noise = galsim.CCDNoise(
        rng=galsim.BaseDeviate(int(seed)),
        gain=gain,
        read_noise=read_noise,
        sky_level=sky_level * exptime / gain,
    )
    image.addNoise(noise)
    return image.array


def targeting_sky_sigma(
    *,
    sky_level: float = LS_R_SKY_LEVEL,
    exptime: float = LS_R_EXPTIME,
    gain: float = LS_R_GAIN,
    read_noise: float = LS_R_READ_NOISE,
) -> float:
    """Per-pixel noise (ADU) from sky and read noise alone."""

    sky_adu = sky_level * exptime / gain
    return float(np.sqrt(sky_adu / gain + (read_noise / gain) ** 2))


def targeting_snr(clean_image: np.ndarray) -> float:
    """Matched-filter SNR of the clean image against sky and read noise."""

    norm = float(np.linalg.norm(np.asarray(clean_image, dtype=np.float64)))
    return norm / targeting_sky_sigma()


def measure_position_angle(
    image: np.ndarray, *, fallback_seed: int
) -> tuple[float, bool]:
    """Position angle of the major axis in (-pi/2, pi/2], from adaptive moments.

    The angle is counter-clockwise from the +x axis (array columns), the same
    frame as the FIBERDX/FIBERDY offsets. If HSM fails, which mostly happens
    for faint, small or nearly round galaxies, return a uniformly random angle
    and ``False``, since a survey would then have no usable position angle.
    """

    import galsim

    result = galsim.hsm.FindAdaptiveMom(
        galsim.Image(np.asarray(image, dtype=np.float64)), strict=False
    )
    ok = result.moments_status == 0 and np.isfinite(
        result.observed_shape.e1
    ) and np.isfinite(result.observed_shape.e2)
    if ok:
        beta = 0.5 * np.arctan2(
            result.observed_shape.e2, result.observed_shape.e1
        )
    else:
        beta = np.random.default_rng(int(fallback_seed)).uniform(
            -0.5 * np.pi, 0.5 * np.pi
        )
    beta = float(beta)
    if beta <= -0.5 * np.pi:
        beta += np.pi
    return beta, bool(ok)


def measured_axis_fiber_offsets(
    beta: float, *, fiber_offset: float
) -> np.ndarray:
    """Five fiber offsets (arcsec) along the axes at position angle ``beta``.

    The order matches ``compute_fiber_offsets``: major +, major -, centre,
    minor +, minor -. The major direction is (cos beta, sin beta) with beta in
    (-pi/2, pi/2], since an image only fixes the axis up to 180 degrees. The
    minor direction is the major one rotated 90 degrees counter-clockwise, so
    the frame is right-handed.
    """

    if not np.isfinite(fiber_offset) or fiber_offset <= 0.0:
        raise ValueError("fiber_offset must be finite and positive")
    axes = np.asarray(
        [
            [np.cos(beta), np.sin(beta)],
            [-np.sin(beta), np.cos(beta)],
        ],
        dtype=float,
    )
    offsets = np.asarray(
        [
            (fiber_offset, 0.0),
            (-fiber_offset, 0.0),
            (0.0, 0.0),
            (0.0, fiber_offset),
            (0.0, -fiber_offset),
        ],
        dtype=float,
    )
    return offsets @ axes
