import torch

from data import (
    apply_d4_to_datavector,
    apply_handedness_flip,
    make_exact_half_flip_mask,
)


def test_make_exact_half_flip_mask_count_and_seed():
    generator_1 = torch.Generator().manual_seed(20260810)
    generator_2 = torch.Generator().manual_seed(20260810)

    mask_even = make_exact_half_flip_mask(
        10,
        device="cpu",
        generator=generator_1,
    )
    repeated = make_exact_half_flip_mask(
        10,
        device="cpu",
        generator=generator_2,
    )
    mask_odd = make_exact_half_flip_mask(11, device="cpu")

    assert mask_even.dtype == torch.bool
    assert int(mask_even.sum().item()) == 5
    assert int(mask_odd.sum().item()) == 5
    assert torch.equal(mask_even, repeated)


def _handedness_datavector():
    img = torch.arange(4 * 1 * 3 * 2, dtype=torch.float32).view(4, 1, 3, 2)
    spec = torch.arange(4 * 1 * 5 * 4, dtype=torch.float32).view(4, 1, 5, 4)
    fid = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, -0.7, -1.0],
        ],
        dtype=torch.float32,
    )
    fp = torch.arange(1, 4 * 5 * 2 + 1, dtype=torch.float32).view(4, 5, 2)
    return img, spec, fid, fp


def test_apply_handedness_flip_transforms_only_masked_rows():
    img, spec, fid, fp = _handedness_datavector()
    mask = torch.tensor([False, True, False, True], dtype=torch.bool)

    out_img, out_spec, out_fid, out_fp = apply_handedness_flip(
        img,
        spec,
        fid,
        fp=fp,
        flip_mask=mask,
        g2_idx=1,
        theta_idx=2,
    )

    assert torch.equal(out_img[~mask], img[~mask])
    assert torch.equal(out_img[mask], torch.flip(img[mask], dims=(-2,)))

    permutation = [0, 1, 2, 4, 3]
    assert torch.equal(out_spec[~mask], spec[~mask])
    assert torch.equal(out_spec[mask], spec[mask][:, :, permutation, :])

    assert torch.equal(out_fid[~mask], fid[~mask])
    torch.testing.assert_close(out_fid[mask, 0], fid[mask, 0])
    torch.testing.assert_close(out_fid[mask, 1], -fid[mask, 1])
    expected_theta = torch.remainder(-fid[mask, 2] + 1.0, 2.0) - 1.0
    torch.testing.assert_close(out_fid[mask, 2], expected_theta)

    expected_fp = fp[mask][:, permutation, :].clone()
    expected_fp[..., 1] = -expected_fp[..., 1]
    assert torch.equal(out_fp[~mask], fp[~mask])
    torch.testing.assert_close(out_fp[mask], expected_fp)


def test_handedness_flip_matches_canonical_d4_reflection():
    img, spec, fid, fp = _handedness_datavector()
    mask = torch.ones(img.shape[0], dtype=torch.bool)

    augmented = apply_handedness_flip(
        img,
        spec,
        fid,
        fp=fp,
        flip_mask=mask,
        g2_idx=1,
        theta_idx=2,
    )
    canonical = apply_d4_to_datavector(
        img,
        spec,
        fid,
        fp,
        element="v",
        feature_names=["g1", "g2", "theta_int"],
    )

    for actual, expected in zip(augmented, canonical):
        torch.testing.assert_close(actual, expected)


def test_apply_handedness_flip_noop_with_all_false_mask():
    img, spec, fid, fp = _handedness_datavector()
    mask = torch.zeros(img.shape[0], dtype=torch.bool)

    outputs = apply_handedness_flip(
        img,
        spec,
        fid,
        fp=fp,
        flip_mask=mask,
        g2_idx=1,
        theta_idx=2,
    )

    for actual, expected in zip(outputs, (img, spec, fid, fp)):
        assert torch.equal(actual, expected)
