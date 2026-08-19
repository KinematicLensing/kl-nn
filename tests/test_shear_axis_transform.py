import numpy as np
import pytest
import torch

from utils import gal_to_img_axis, img_to_gal_axis


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("theta_value", "expected_plus", "expected_cross"),
    [
        (0.0, (1.0, 0.0), (0.0, 1.0)),
        (np.pi / 4, (0.0, -1.0), (1.0, 0.0)),
        (-np.pi / 4, (0.0, 1.0), (-1.0, 0.0)),
    ],
)
def test_image_basis_maps_to_original_galaxy_axis_convention(
    backend, theta_value, expected_plus, expected_cross
):
    """Lock the simulator's original, clockwise ``theta_int`` handedness."""
    if backend == "torch":
        g1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
        g2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        theta = torch.tensor(theta_value, dtype=torch.float64)
        expected_plus = torch.tensor(expected_plus, dtype=torch.float64)
        expected_cross = torch.tensor(expected_cross, dtype=torch.float64)
        assert_close = torch.testing.assert_close
    else:
        g1 = np.asarray([1.0, 0.0])
        g2 = np.asarray([0.0, 1.0])
        theta = np.asarray(theta_value)
        expected_plus = np.asarray(expected_plus)
        expected_cross = np.asarray(expected_cross)
        assert_close = np.testing.assert_allclose

    g_plus, g_cross = img_to_gal_axis(g1, g2, theta)

    assert_close(g_plus, expected_plus, atol=1e-12, rtol=0.0)
    assert_close(g_cross, expected_cross, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_clockwise_45_degree_galaxy_basis_has_expected_image_axes(backend):
    if backend == "torch":
        g_plus = torch.tensor([1.0, 0.0], dtype=torch.float64)
        g_cross = torch.tensor([0.0, 1.0], dtype=torch.float64)
        theta = torch.full((2,), np.pi / 4, dtype=torch.float64)
        expected_g1 = torch.tensor([0.0, 1.0], dtype=torch.float64)
        expected_g2 = torch.tensor([-1.0, 0.0], dtype=torch.float64)
        assert_close = torch.testing.assert_close
    else:
        g_plus = np.asarray([1.0, 0.0])
        g_cross = np.asarray([0.0, 1.0])
        theta = np.full(2, np.pi / 4)
        expected_g1 = np.asarray([0.0, 1.0])
        expected_g2 = np.asarray([-1.0, 0.0])
        assert_close = np.testing.assert_allclose

    g1, g2 = gal_to_img_axis(g_plus, g_cross, theta)

    assert_close(g1, expected_g1, atol=1e-12, rtol=0.0)
    assert_close(g2, expected_g2, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_image_galaxy_axis_transforms_round_trip_batched_values(backend):
    rng = np.random.default_rng(20260813)
    g1_np = rng.normal(size=(4, 3))
    g2_np = rng.normal(size=(4, 3))
    theta_np = rng.uniform(-np.pi, np.pi, size=(4, 1))

    if backend == "torch":
        g1 = torch.as_tensor(g1_np, dtype=torch.float64)
        g2 = torch.as_tensor(g2_np, dtype=torch.float64)
        theta = torch.as_tensor(theta_np, dtype=torch.float64)
        assert_close = torch.testing.assert_close
    else:
        g1 = g1_np
        g2 = g2_np
        theta = theta_np
        assert_close = np.testing.assert_allclose

    g_plus, g_cross = img_to_gal_axis(g1, g2, theta)
    restored_g1, restored_g2 = gal_to_img_axis(g_plus, g_cross, theta)

    assert_close(restored_g1, g1, atol=1e-12, rtol=1e-12)
    assert_close(restored_g2, g2, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_galaxy_image_axis_transforms_round_trip_batched_values(backend):
    rng = np.random.default_rng(20260814)
    g_plus_np = rng.normal(size=(4, 3))
    g_cross_np = rng.normal(size=(4, 3))
    theta_np = rng.uniform(-np.pi, np.pi, size=(4, 1))

    if backend == "torch":
        g_plus = torch.as_tensor(g_plus_np, dtype=torch.float64)
        g_cross = torch.as_tensor(g_cross_np, dtype=torch.float64)
        theta = torch.as_tensor(theta_np, dtype=torch.float64)
        assert_close = torch.testing.assert_close
    else:
        g_plus = g_plus_np
        g_cross = g_cross_np
        theta = theta_np
        assert_close = np.testing.assert_allclose

    g1, g2 = gal_to_img_axis(g_plus, g_cross, theta)
    restored_plus, restored_cross = img_to_gal_axis(g1, g2, theta)

    assert_close(restored_plus, g_plus, atol=1e-12, rtol=1e-12)
    assert_close(restored_cross, g_cross, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_image_to_galaxy_axis_transform_preserves_shear_norm(backend):
    g1_np = np.asarray([0.10, -0.03, 0.04])
    g2_np = np.asarray([-0.02, 0.08, -0.05])
    theta_np = np.asarray([0.0, np.pi / 4, -np.pi / 4])

    if backend == "torch":
        g1 = torch.as_tensor(g1_np, dtype=torch.float64)
        g2 = torch.as_tensor(g2_np, dtype=torch.float64)
        theta = torch.as_tensor(theta_np, dtype=torch.float64)
        assert_close = torch.testing.assert_close
    else:
        g1 = g1_np
        g2 = g2_np
        theta = theta_np
        assert_close = np.testing.assert_allclose

    g_plus, g_cross = img_to_gal_axis(g1, g2, theta)

    assert_close(
        g_plus**2 + g_cross**2,
        g1**2 + g2**2,
        atol=1e-12,
        rtol=1e-12,
    )
