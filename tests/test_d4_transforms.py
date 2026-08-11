import itertools

import pytest
import torch

from data import (
    D4_ELEMENTS,
    D4_INVERSES,
    apply_d4_to_datavector,
)


def _datavector():
    img = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    spec = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    fid = torch.tensor(
        [[0.10, -0.20, 0.75, -0.4, 0.3, -0.2, 0.1, 0.9]],
        dtype=torch.float32,
    )
    fp = torch.tensor(
        [[[2.0, 0.5], [-2.0, -0.5], [0.25, -0.1], [-0.5, 2.0], [0.5, -2.0]]],
        dtype=torch.float32,
    )
    return img, spec, fid, fp


def _apply(values, element):
    return apply_d4_to_datavector(*values, element=element)


def _assert_datavectors_equal(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def _datavectors_equal(left, right):
    return all(torch.equal(a, b) for a, b in zip(left, right))


def test_r90_transforms_complete_directed_datavector():
    img, spec, fid, fp = _datavector()

    img_out, spec_out, fid_out, fp_out = apply_d4_to_datavector(
        img, spec, fid, fp, element="r90"
    )

    assert torch.equal(img_out, torch.rot90(img, k=1, dims=(-2, -1)))
    assert torch.equal(spec_out, spec)
    torch.testing.assert_close(
        fid_out,
        torch.tensor(
            [[-0.10, 0.20, 0.25, -0.4, 0.3, -0.2, 0.1, 0.9]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        fp_out,
        torch.tensor(
            [[[0.5, -2.0], [-0.5, 2.0], [-0.1, -0.25], [2.0, 0.5], [-2.0, -0.5]]],
            dtype=torch.float32,
        ),
    )


def test_reflection_swaps_minor_fiber_spectrum_position_pairs():
    img, spec, fid, fp = _datavector()

    img_out, spec_out, fid_out, fp_out = apply_d4_to_datavector(
        img, spec, fid, fp, element="v"
    )

    assert torch.equal(img_out, torch.flip(img, dims=(-2,)))
    assert torch.equal(spec_out[..., 0:3, :], spec[..., 0:3, :])
    assert torch.equal(spec_out[..., 3, :], spec[..., 4, :])
    assert torch.equal(spec_out[..., 4, :], spec[..., 3, :])
    torch.testing.assert_close(
        fid_out,
        torch.tensor(
            [[0.10, 0.20, -0.75, -0.4, 0.3, -0.2, 0.1, 0.9]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        fp_out,
        torch.tensor(
            [[[2.0, -0.5], [-2.0, 0.5], [0.25, 0.1], [0.5, 2.0], [-0.5, -2.0]]],
            dtype=torch.float32,
        ),
    )


def test_d4_generators_satisfy_group_relations():
    original = _datavector()

    rotated = original
    for _ in range(4):
        rotated = _apply(rotated, "r90")
    _assert_datavectors_equal(rotated, original)

    reflected = _apply(_apply(original, "v"), "v")
    _assert_datavectors_equal(reflected, original)

    srs = _apply(_apply(_apply(original, "v"), "r90"), "v")
    inverse_rotation = _apply(original, "r270")
    _assert_datavectors_equal(srs, inverse_rotation)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_every_d4_element_round_trips_with_its_inverse(element):
    original = _datavector()
    transformed = _apply(original, element)
    restored = _apply(transformed, D4_INVERSES[element])

    _assert_datavectors_equal(restored, original)
    assert torch.all((-1.0 <= transformed[2][..., 2]) & (transformed[2][..., 2] < 1.0))


def test_all_d4_element_pairs_are_closed():
    original = _datavector()
    orbit = [_apply(original, element) for element in D4_ELEMENTS]

    for first, second in itertools.product(D4_ELEMENTS, repeat=2):
        composed = _apply(_apply(original, first), second)
        assert any(_datavectors_equal(composed, member) for member in orbit), (
            first,
            second,
        )


def test_unknown_d4_element_is_rejected():
    with pytest.raises(ValueError, match="Unknown D4 element"):
        _apply(_datavector(), "r45")

