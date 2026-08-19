import inspect

import pytest
import torch
from torch import nn

import config
import train
from data import apply_noise


class DummyDataset:
    def __init__(self, size: int, nfeatures: int) -> None:
        self.size = size
        self.nfeatures = nfeatures

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return {
            "img": torch.rand((1, 48, 48), dtype=torch.float32),
            "spec": torch.rand((1, 5, 64), dtype=torch.float32),
            "fid_pars": torch.rand((self.nfeatures,), dtype=torch.float32),
        }


def _set_basic_train_config(monkeypatch, nfeatures: int, batch_size: int) -> None:
    monkeypatch.setitem(config.train, "mode", 0)
    monkeypatch.setitem(config.train, "batch_size", batch_size)
    monkeypatch.setitem(config.train, "feature_number", nfeatures)
    base_names = ["g1", "g2", "vcirc"]
    monkeypatch.setitem(config.train, "feature_names", base_names[:nfeatures])
    monkeypatch.setitem(config.train, "enable_handedness_flip", False)
    monkeypatch.setitem(config.train, "epoch_number", 1)
    monkeypatch.setitem(config.train, "initial_learning_rate", 1e-3)
    monkeypatch.setitem(config.train, "weight_decay", 0.0)


def test_train_config_has_optimization_defaults():
    for key in (
        "use_amp",
        "amp_dtype",
        "use_compile",
        "compile_mode",
        "compile_backend",
        "use_fused_adamw",
        "cudnn_benchmark",
        "channels_last",
        "ddp_static_graph",
        "ddp_find_unused_parameters",
        "ddp_gradient_as_bucket_view",
        "ddp_broadcast_buffers",
        "noise_cache_maxs",
    ):
        assert key in config.train
    assert config.train["use_amp"] is False
    assert config.train["use_compile"] is False
    assert config.train["use_fused_adamw"] is True
    assert config.train["cudnn_benchmark"] is True
    assert config.train["channels_last"] is True


def test_apply_noise_cached_maxs_matches_uncached():
    data = torch.arange(2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)
    snr = torch.tensor([10.0, 50.0], dtype=torch.float32)
    gen_a = torch.Generator().manual_seed(1234)
    gen_b = torch.Generator().manual_seed(1234)
    maxs = torch.amax(data, dim=(-1, -2, -3))

    out_uncached = apply_noise(data, snr, randgen=gen_a, device="cpu")
    out_cached = apply_noise(data, snr, randgen=gen_b, device="cpu", maxs=maxs)

    assert torch.allclose(out_uncached, out_cached)


def test_load_train_objs_fused_adamw_flag(monkeypatch):
    _set_basic_train_config(monkeypatch, nfeatures=3, batch_size=2)
    monkeypatch.setitem(config.train, "use_fused_adamw", True)
    monkeypatch.setattr(train.pxt, "TorchDataset", lambda path: DummyDataset(4, 3))

    train_ds, valid_ds, model, optimizer = train.load_train_objs(
        torch.nn.Linear,
        rank=0,
        train_config=config.train,
        train_mode="train",
        device=torch.device("cpu"),
        in_features=1,
        out_features=1,
    )

    fused_supported = (
        "fused" in inspect.signature(torch.optim.AdamW).parameters
        and torch.cuda.is_available()
        and next(model.parameters()).is_cuda
    )
    fused_value = bool(optimizer.defaults.get("fused", False))
    assert fused_value == fused_supported
    assert len(train_ds) == 4
    assert len(valid_ds) == 4
    assert model is not None


def test_load_train_objs_passes_current_mode_to_klnpe_explicitly(monkeypatch):
    _set_basic_train_config(monkeypatch, nfeatures=3, batch_size=2)
    monkeypatch.setitem(config.train, "mode", 1)
    monkeypatch.setitem(config.train, "backbone_type", "stage4_d4")
    monkeypatch.setitem(config.train, "posterior_symmetry", "d4")
    monkeypatch.setattr(train.pxt, "TorchDataset", lambda path: DummyDataset(4, 3))

    class CapturingKLNPE(train.KLNPE):
        def __init__(self, feature_extractor=None, **kwargs):
            nn.Module.__init__(self)
            self.feature_extractor = feature_extractor or nn.Linear(1, 1)
            self.head = nn.Parameter(torch.zeros(()))
            self.received = kwargs
            self.mode = kwargs["mode"]

    model_config = dict(config.train)
    model_config["pretrained_name"] = "dummy"
    monkeypatch.setattr(
        train,
        "load_model",
        lambda *args, **kwargs: type(
            "Pretrained", (), {"backbone": nn.Linear(1, 1)}
        )(),
    )

    _, _, model, _ = train.load_train_objs(
        CapturingKLNPE,
        rank=0,
        train_config=model_config,
        train_mode="train",
        epoch=19,
        device=torch.device("cpu"),
    )

    assert model.mode == 1
    assert model.received["mode"] == 1
    assert model.received["batch_size"] == 2
    assert model.received["nfeatures"] == 3
    assert model.received["backbone_type"] == "stage4_d4"
    assert model.received["posterior_symmetry"] == "d4"


def test_load_model_can_explicitly_use_current_source(tmp_path, monkeypatch):
    class CurrentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))

    expected = CurrentModel()
    with torch.no_grad():
        expected.weight.copy_(torch.tensor([1.25, -0.5]))
    checkpoint = tmp_path / "model" / "model0"
    checkpoint.parent.mkdir()
    torch.save(expected.state_dict(), checkpoint)

    def archived_loader_should_not_run(*args, **kwargs):
        raise AssertionError("archived source was unexpectedly loaded")

    monkeypatch.setattr(train, "load_networks_module_for_model", archived_loader_should_not_run)
    restored = train.load_model(
        config.train,
        Model=CurrentModel,
        path=str(checkpoint),
        use_archived_networks=False,
    )
    torch.testing.assert_close(restored.weight, expected.weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for training smoke test")
def test_trainer_amp_compile_channels_last_smoke(monkeypatch):
    _set_basic_train_config(monkeypatch, nfeatures=3, batch_size=2)
    monkeypatch.setitem(config.train, "use_amp", True)
    monkeypatch.setitem(config.train, "amp_dtype", "float16")
    monkeypatch.setitem(config.train, "use_compile", True)
    monkeypatch.setitem(config.train, "compile_mode", "default")
    monkeypatch.setitem(config.train, "compile_backend", None)
    monkeypatch.setitem(config.train, "channels_last", True)
    monkeypatch.setitem(config.train, "noise_cache_maxs", True)

    device = torch.device("cuda:0")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
    ).to(device=device, memory_format=torch.channels_last)
    model = train._maybe_compile_model(model, log=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randn((2, 1, 16, 16), device=device)
    inputs = inputs.contiguous(memory_format=torch.channels_last)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs)
        loss = outputs.square().mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert inputs.is_contiguous(memory_format=torch.channels_last)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(param).all() for param in model.parameters())
