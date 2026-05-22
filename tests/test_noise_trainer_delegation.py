import pytest
import torch
from torch import nn, optim

train = pytest.importorskip("train")


class _DummyDataset:
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        # CNNTrainer probes dataset[0] during init to infer fib_pos availability.
        # Return a minimal sample shape compatible with that check.
        return {
            "img": torch.zeros((1, 48, 48), dtype=torch.float32),
            "spec": torch.zeros((1, 5, 64), dtype=torch.float32),
            "fid_pars": torch.zeros((2,), dtype=torch.float32),
        }


def test_trainer_apply_noise_delegates_to_shared_function(monkeypatch):
    train_ds = _DummyDataset()
    valid_ds = _DummyDataset()

    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    trainer = train.CNNTrainer(
        world_size=1,
        model=model,
        nfeatures=2,
        train_ds=train_ds,
        valid_ds=valid_ds,
        optimizer=optimizer,
        gpu_id=0,
        save_every=1,
        batch_size=2,
    )

    expected = torch.full((2, 1, 4, 4), 7.0)
    calls = {}

    def _fake_apply_noise(data, snr, randgen=None, device="cpu", use_iterative=True, **kwargs):
        calls["data"] = data
        calls["snr"] = snr
        calls["device"] = device
        calls["use_iterative"] = use_iterative
        return expected

    monkeypatch.setattr(train, "apply_noise", _fake_apply_noise)

    data = torch.ones((2, 1, 4, 4))
    snr = torch.full((2,), 80.0)
    out = trainer._apply_noise(data, snr)

    assert torch.equal(out, expected)
    assert calls["device"] == trainer.device
    assert calls["use_iterative"] is True
    assert torch.equal(calls["data"], data)
    assert torch.equal(calls["snr"], snr)
