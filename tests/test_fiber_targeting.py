import math

import numpy as np
import pytest

galsim = pytest.importorskip("galsim")

from data_generate.fiber_targeting import measure_position_angle


def _elliptical_gaussian(*, hlr_pix, q, beta_deg, n=256):
    profile = galsim.Gaussian(half_light_radius=hlr_pix).shear(
        q=q, beta=beta_deg * galsim.degrees
    )
    return profile.drawImage(nx=n, ny=n, scale=1.0, method="no_pixel").array


def test_large_elongated_oblique_disk_converges_after_retry():
    # The default 5-pixel HSM start does not converge for this shape.
    image = _elliptical_gaussian(hlr_pix=15.0, q=0.15, beta_deg=30.0)
    default = galsim.hsm.FindAdaptiveMom(galsim.Image(image), strict=False)
    assert default.moments_status != 0

    beta, ok = measure_position_angle(image, fallback_seed=1)
    assert ok
    assert math.degrees(beta) == pytest.approx(30.0, abs=0.1)


def test_default_start_is_used_when_it_converges():
    image = _elliptical_gaussian(hlr_pix=4.0, q=0.6, beta_deg=-20.0, n=64)
    default = galsim.hsm.FindAdaptiveMom(galsim.Image(image), strict=False)
    assert default.moments_status == 0
    expected = 0.5 * np.arctan2(default.observed_shape.e2, default.observed_shape.e1)

    beta, ok = measure_position_angle(image, fallback_seed=1)
    assert ok
    assert beta == expected


def test_unweighted_angle_matches_hsm_frame():
    from data_generate.fiber_targeting import unweighted_position_angle

    for beta_deg in (-60.0, -10.0, 25.0, 80.0):
        image = _elliptical_gaussian(hlr_pix=4.0, q=0.5, beta_deg=beta_deg, n=64)
        hsm = galsim.hsm.FindAdaptiveMom(galsim.Image(image), strict=False)
        expected = 0.5 * np.arctan2(hsm.observed_shape.e2, hsm.observed_shape.e1)
        assert unweighted_position_angle(image) == pytest.approx(expected, abs=1e-6)


def test_unmeasurable_image_falls_back_to_unweighted_moments():
    from data_generate.fiber_targeting import unweighted_position_angle

    image = np.random.default_rng(3).normal(0.0, 1.0, size=(48, 48))  # pure noise
    beta, ok = measure_position_angle(image, fallback_seed=7)
    assert not ok
    assert beta == unweighted_position_angle(image)
    assert -np.pi / 2 < beta <= np.pi / 2
