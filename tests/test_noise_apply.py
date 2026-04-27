import pytest
import torch

train = pytest.importorskip("train")


def test_apply_noise_preserves_shape_dtype():
    data = torch.ones((8, 1, 16, 16), dtype=torch.float32)
    snr = torch.full((8,), 50.0)

    out = train.apply_noise(data, snr, device="cpu", use_iterative=True)

    assert out.shape == data.shape
    assert out.dtype == data.dtype


def test_apply_noise_deterministic_with_generator():
    data = torch.ones((4, 1, 8, 8), dtype=torch.float32)
    snr = torch.full((4,), 100.0)

    gen1 = torch.Generator(device="cpu")
    gen1.manual_seed(1234)
    out1 = train.apply_noise(data, snr, randgen=gen1, device="cpu", use_iterative=True)

    gen2 = torch.Generator(device="cpu")
    gen2.manual_seed(1234)
    out2 = train.apply_noise(data, snr, randgen=gen2, device="cpu", use_iterative=True)

    assert torch.allclose(out1, out2)


def test_apply_noise_no_nan_inf_on_edge_inputs():
    data = torch.zeros((6, 1, 12, 12), dtype=torch.float32)
    data[0] = 1e-12
    snr = torch.tensor([0.0, 1e-8, 1e-4, 1.0, 10.0, 1000.0], dtype=torch.float32)

    out_iter = train.apply_noise(data, snr, device="cpu", use_iterative=True)
    out_single = train.apply_noise(data, snr, device="cpu", use_iterative=False)

    assert torch.isfinite(out_iter).all()
    assert torch.isfinite(out_single).all()


def test_apply_noise_snr_scaling_sanity():
    data = torch.ones((512, 1, 8, 8), dtype=torch.float32)
    snr_low = torch.full((512,), 20.0)
    snr_high = torch.full((512,), 200.0)

    gen_low = torch.Generator(device="cpu")
    gen_low.manual_seed(100)
    noisy_low = train.apply_noise(data, snr_low, randgen=gen_low, device="cpu", use_iterative=True)

    gen_high = torch.Generator(device="cpu")
    gen_high.manual_seed(200)
    noisy_high = train.apply_noise(data, snr_high, randgen=gen_high, device="cpu", use_iterative=True)

    low_rms = (noisy_low - data).std()
    high_rms = (noisy_high - data).std()

    assert low_rms > high_rms * 2.0
