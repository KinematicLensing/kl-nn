import math

import torch

from data import rotate_90_datavector, rotate_90_parameters
from train import make_ccl_training_batch, make_npe_training_batch


FEATURES = (
    "g1", "g2", "theta_int", "sini", "v0", "vcirc", "rscale", "hlr",
    "halpha_flux_true",
)


def test_r90_parameters_round_trip_and_leave_six_scalars_invariant():
    values = torch.tensor(
        [
            [0.1, -0.2, -0.9, 0.4, -0.3, 0.2, 0.1, -0.5, 0.8],
            [-0.3, 0.5, 0.8, -0.1, 0.6, -0.4, 0.7, 0.2, -0.9],
        ]
    )
    rotated = rotate_90_parameters(values, feature_names=FEATURES)
    torch.testing.assert_close(rotated[:, :2], -values[:, :2])
    torch.testing.assert_close(rotated[:, 3:], values[:, 3:])
    expected_theta = torch.remainder(values[:, 2] - 0.5 + 1.0, 2.0) - 1.0
    torch.testing.assert_close(rotated[:, 2], expected_theta)
    restored = rotate_90_parameters(rotated, inverse=True, feature_names=FEATURES)
    torch.testing.assert_close(restored, values)


def test_r90_datavector_geometry_and_inverse_are_exact():
    image = torch.arange(16.0).reshape(1, 1, 4, 4)
    spectra = torch.randn(1, 1, 5, 8, generator=torch.Generator().manual_seed(2))
    parameters = torch.zeros(1, 9)
    positions = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, -1.0]]])
    transformed = rotate_90_datavector(image, spectra, parameters, positions)
    image_r, spectra_r, parameters_r, positions_r = transformed
    torch.testing.assert_close(image_r, torch.rot90(image, 1, (-2, -1)))
    torch.testing.assert_close(spectra_r, spectra)
    torch.testing.assert_close(positions_r[..., 0], positions[..., 1])
    torch.testing.assert_close(positions_r[..., 1], -positions[..., 0])
    restored = rotate_90_datavector(
        image_r, spectra_r, parameters_r, positions_r, inverse=True
    )
    for actual, expected in zip(restored, (image, spectra, parameters, positions)):
        torch.testing.assert_close(actual, expected)


def test_ccl_batch_contains_identity_and_r90_views():
    batch = 3
    image = torch.randn(batch, 1, 4, 4)
    spectra = torch.randn(batch, 1, 5, 8)
    targets = torch.rand(batch, 9) * 2 - 1
    positions = torch.randn(batch, 5, 2)
    context = {
        "rmag_true": torch.tensor([18.0, 20.0, 22.0]),
        "spectral_reference_quality": torch.tensor([5.0, 20.0, 80.0]),
    }
    ccl = make_ccl_training_batch(image, spectra, targets, positions, context)

    assert ccl[0].shape[0] == 2 * batch
    torch.testing.assert_close(ccl[0][:batch], image)
    torch.testing.assert_close(ccl[0][batch:], torch.rot90(image, 1, (-2, -1)))
    torch.testing.assert_close(ccl[4]["rmag_true"], context["rmag_true"].repeat(2))
    torch.testing.assert_close(
        ccl[4]["spectral_reference_quality"],
        context["spectral_reference_quality"].repeat(2),
    )
    assert math.isfinite(float(ccl[2].mean()))


def test_npe_batch_selects_exactly_one_reproducible_view_per_row():
    batch = 3
    image = torch.randn(batch, 1, 4, 4)
    spectra = torch.randn(batch, 1, 5, 8)
    targets = torch.rand(batch, 9) * 2 - 1
    positions = torch.randn(batch, 5, 2)
    context = {
        "rmag_true": torch.tensor([18.0, 20.0, 22.0]),
        "spectral_reference_quality": torch.tensor([5.0, 20.0, 80.0]),
    }
    rotate_mask = torch.tensor([False, True, False])
    rotated = rotate_90_datavector(image, spectra, targets, positions)
    npe = make_npe_training_batch(
        image,
        spectra,
        targets,
        positions,
        context,
        rotate_mask=rotate_mask,
    )

    assert npe[0].shape[0] == batch
    for actual, identity, rotated_value in zip(
        npe[:4], (image, spectra, targets, positions), rotated
    ):
        shape = (batch,) + (1,) * (identity.ndim - 1)
        expected = torch.where(
            rotate_mask.reshape(shape), rotated_value, identity
        )
        torch.testing.assert_close(actual, expected)
    assert npe[4] is context
