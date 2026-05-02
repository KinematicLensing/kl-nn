import numpy as np
import pytest
import torch

train = pytest.importorskip("train")


def test_generate_snr_rmag_uses_fitted_relation(tmp_path, monkeypatch):
    fit_path = tmp_path / "rmag_snr_fit.npz"
    np.savez(
        fit_path,
        a=-2.0,
        b=8.0,
        source_path="/tmp/source.npz",
        model="rmag = a * log10(SNR) + b",
    )
    monkeypatch.setattr(train.config, "rmag_snr_fit_path", str(fit_path), raising=False)
    monkeypatch.setattr(np.random, "power", lambda power, size=None: np.array([0.0, 0.5, 1.0], dtype=float))

    trainer = train.CNNTrainer.__new__(train.CNNTrainer)
    trainer.device = torch.device("cpu")

    out = trainer.generate_snr(3, mode="rmag", min=16, max=20)

    expected_rmag = np.array([16.0, 18.0, 20.0], dtype=float)
    expected_snr = 10 ** ((expected_rmag - 8.0) / -2.0)

    assert out.shape == (3,)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.tensor(expected_snr, dtype=torch.float32))