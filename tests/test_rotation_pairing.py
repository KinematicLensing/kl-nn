import numpy as np
import pytest
import torch

from data import rot_90_param_only, rotate_90_degrees
from train import pair_rotation_branches


def test_pair_rotation_branches_preserves_galaxy_and_branch_order():
    values = np.array([
        [[10], [11]],
        [[20], [21]],
        [[30], [31]],
        [[40], [41]],
    ])

    paired = pair_rotation_branches(values)

    assert paired.shape == (2, 2, 2, 1)
    np.testing.assert_array_equal(paired[0, :, 0, 0], [10, 11])
    np.testing.assert_array_equal(paired[0, :, 1, 0], [20, 21])
    np.testing.assert_array_equal(paired[1, :, 0, 0], [30, 31])
    np.testing.assert_array_equal(paired[1, :, 1, 0], [40, 41])


def test_pair_rotation_log_probabilities_use_same_order():
    log_prob = np.array([[10, 11], [20, 21], [30, 31], [40, 41]])
    paired = pair_rotation_branches(log_prob)
    np.testing.assert_array_equal(paired[0, :, 0], [10, 11])
    np.testing.assert_array_equal(paired[0, :, 1], [20, 21])


def test_pair_rotation_branches_rejects_odd_leading_dimension():
    with pytest.raises(ValueError, match="even leading dimension"):
        pair_rotation_branches(np.zeros((3, 2, 8)))


def test_rot_90_parameter_rotation_is_vectorized_and_round_trips():
    values = np.array([
        [0.01, -0.02, 3.0, 0.5],
        [-0.03, 0.04, -3.0, 0.7],
    ])
    rotated = rot_90_param_only(values)
    restored = rot_90_param_only(rotated, reverse=True)

    np.testing.assert_allclose(restored, values)
    assert np.all((-np.pi <= rotated[:, 2]) & (rotated[:, 2] < np.pi))


def test_rotate_90_degrees_uses_image_array_convention_for_fiber_positions():
    img = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    fid = torch.tensor([[0.1, -0.2, 0.75, 0.4]], dtype=torch.float32)
    # major+, major-, center, minor+, minor-
    fp = torch.tensor(
        [[[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, -2.0]]],
        dtype=torch.float32,
    )

    img_rot, fid_rot, fp_rot = rotate_90_degrees(img, fid, fp)

    assert torch.equal(img_rot, torch.rot90(img, k=1, dims=(-2, -1)))
    torch.testing.assert_close(
        fid_rot, torch.tensor([[-0.1, 0.2, 0.25, 0.4]], dtype=torch.float32)
    )
    # In array coordinates, k=1 sends right -> up: (x, y) -> (y, -x).
    expected_fp = torch.tensor(
        [[[0.0, -2.0], [0.0, 2.0], [0.0, 0.0], [2.0, 0.0], [-2.0, 0.0]]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(fp_rot, expected_fp)


def test_four_90_degree_datavector_rotations_are_identity():
    img = torch.arange(18, dtype=torch.float32).reshape(2, 1, 3, 3)
    fid = torch.tensor(
        [[0.1, -0.2, 0.75, 0.4], [-0.3, 0.25, -0.8, -0.1]],
        dtype=torch.float32,
    )
    fp = torch.tensor(
        [
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
            [[1.0, 2.0], [-1.0, -2.0], [0.0, 0.0], [2.0, -1.0], [-2.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    img_rot, fid_rot, fp_rot = img.clone(), fid.clone(), fp.clone()
    for _ in range(4):
        img_rot, fid_rot, fp_rot = rotate_90_degrees(img_rot, fid_rot, fp_rot)

    assert torch.equal(img_rot, img)
    torch.testing.assert_close(fid_rot, fid)
    torch.testing.assert_close(fp_rot, fp)


def test_correct_pairing_does_not_cancel_two_different_galaxies():
    # Each rotated branch counter-rotates back to its own original shear.
    originals = np.array([[[0.02, -0.01, 0.1]], [[-0.03, 0.04, -0.2]]])
    rotated = rot_90_param_only(originals)
    interleaved = np.stack(
        [originals[0], rotated[0], originals[1], rotated[1]], axis=0
    )
    paired = pair_rotation_branches(interleaved)
    counter = rot_90_param_only(paired[:, :, 1], reverse=True)
    symmetrized = 0.5 * (paired[:, :, 0, :2] + counter[:, :, :2])

    np.testing.assert_allclose(symmetrized, originals[:, :, :2])
