import pytest
import torch

train = pytest.importorskip("train")


def test_make_exact_half_flip_mask_count():
    mask_even = train.make_exact_half_flip_mask(10, device="cpu")
    mask_odd = train.make_exact_half_flip_mask(11, device="cpu")

    assert mask_even.dtype == torch.bool
    assert int(mask_even.sum().item()) == 5
    assert int(mask_odd.sum().item()) == 5


def test_apply_handedness_flip_transforms_only_masked_rows():
    img = torch.arange(4 * 1 * 3 * 2, dtype=torch.float32).view(4, 1, 3, 2)
    spec = torch.randn(4, 1, 5, 4, dtype=torch.float32)
    fid = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    fp = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor([False, True, False, True], dtype=torch.bool)

    out_img, out_spec, out_fid, out_fp = train.apply_handedness_flip(
        img,
        spec,
        fid,
        fp=fp,
        flip_mask=mask,
        g2_idx=1,
        theta_idx=2,
    )

    assert torch.allclose(out_spec, spec)
    assert torch.equal(out_img[0], img[0])
    assert torch.equal(out_img[2], img[2])
    assert torch.equal(out_img[1], torch.flip(img[1], dims=(-2,)))
    assert torch.equal(out_img[3], torch.flip(img[3], dims=(-2,)))

    assert torch.equal(out_fid[[0, 2], :], fid[[0, 2], :])
    assert torch.allclose(out_fid[mask, 0], fid[mask, 0])
    assert torch.allclose(out_fid[mask, 1], -fid[mask, 1])
    assert torch.allclose(out_fid[mask, 2], -fid[mask, 2])

    assert torch.allclose(out_fp[mask, :, 0], fp[mask, :, 0])
    assert torch.allclose(out_fp[mask, :, 1], -fp[mask, :, 1])
    assert torch.equal(out_fp[~mask], fp[~mask])


def test_apply_handedness_flip_noop_with_all_false_mask():
    img = torch.randn(3, 1, 4, 4)
    spec = torch.randn(3, 1, 5, 8)
    fid = torch.randn(3, 3)
    fp = torch.randn(3, 5, 2)
    mask = torch.zeros(3, dtype=torch.bool)

    out_img, out_spec, out_fid, out_fp = train.apply_handedness_flip(
        img,
        spec,
        fid,
        fp=fp,
        flip_mask=mask,
        g2_idx=1,
        theta_idx=2,
    )

    assert torch.equal(out_img, img)
    assert torch.equal(out_spec, spec)
    assert torch.equal(out_fid, fid)
    assert torch.equal(out_fp, fp)
