"""Compile/AMP contracts, including the frozen NPE feature boundary."""

import logging

import pytest
import torch
from torch import nn

import config
import train
from networks import (
    BoundedHybridCircularFlow,
    CCLPretrain,
    KLNPE,
    Stage3FeatureExtractor,
)


class _RecordingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.compile_calls = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)

    def forward(self, values):
        return self.head(self.feature_extractor(values))


def test_compile_is_in_place_and_preserves_optimizer_parameter_identity(caplog):
    model = _RecordingModule()
    feature_extractor = model.feature_extractor
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    state_keys = tuple(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())

    with caplog.at_level(logging.INFO):
        compiled = train._maybe_compile_model(
            model,
            {
                "use_compile": True,
                "compile_mode": "reduce-overhead",
                "compile_backend": "eager",
            },
            logging.getLogger("compile-test"),
        )

    assert compiled is model
    assert compiled.feature_extractor is feature_extractor
    assert {id(parameter) for parameter in compiled.parameters()} == parameter_ids
    assert tuple(compiled.state_dict()) == state_keys
    assert {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    } == parameter_ids
    assert model.compile_calls == [
        {"mode": "reduce-overhead", "backend": "eager"}
    ]
    assert "Enabled torch.compile" in caplog.text


def test_requested_compile_failure_is_not_silently_downgraded_to_eager():
    class _BrokenCompile(_RecordingModule):
        def compile(self, **kwargs):
            del kwargs
            raise SyntaxError("broken compiler environment")

    with pytest.raises(RuntimeError, match="refusing to silently run eager") as error:
        train._maybe_compile_model(
            _BrokenCompile(),
            {"use_compile": True, "compile_mode": "default"},
        )
    assert isinstance(error.value.__cause__, SyntaxError)


def test_requested_compile_is_not_silently_ignored_when_unavailable(monkeypatch):
    model = _RecordingModule()
    monkeypatch.delattr(torch, "compile")

    with pytest.raises(RuntimeError, match="torch.compile is unavailable"):
        train._maybe_compile_model(
            model,
            {"use_compile": True, "compile_mode": "default"},
        )


def test_disabled_compile_does_not_touch_the_module():
    model = _RecordingModule()
    assert train._maybe_compile_model(model, {"use_compile": False}) is model
    assert model.compile_calls == []


class _AutocastFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 1024)
        self.last_output_dtype = None

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        result = self.projection(image.mean(dim=(-2, -1)))
        self.last_output_dtype = result.dtype
        return result


class _DtypeRecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.last_true_dtype = None
        self.last_context_dtype = None

    def log_prob(self, true, context):
        self.last_true_dtype = true.dtype
        self.last_context_dtype = context.dtype
        residual = self.scale * context[:, : true.shape[1]] - true
        return -residual.square().mean(dim=1)


def test_npe_amp_keeps_encoder_autocast_but_runs_flow_and_loss_in_float32():
    extractor = _AutocastFeatureExtractor()
    extractor.requires_grad_(False)
    flow = _DtypeRecordingFlow()
    model = KLNPE(feature_extractor=extractor, flow=flow)
    batch_size = 4
    context = {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size),
        "image_snr": torch.linspace(5.0, 1000.0, batch_size),
        "central_halpha_snr": torch.linspace(1.0, 200.0, batch_size),
    }

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(
            torch.randn(batch_size, 1, 4, 4),
            None,
            torch.zeros(batch_size, len(config.TARGET_NAMES)),
            None,
            observation_context=context,
        )
    loss.backward()

    assert extractor.last_output_dtype == torch.bfloat16
    assert flow.last_true_dtype == torch.float32
    assert flow.last_context_dtype == torch.float32
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(flow.scale.grad)

class _TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)

    def forward(self, image):
        return self.projection(image.mean(dim=(-2, -1)))


def _feature_extractor():
    return Stage3FeatureExtractor(
        nspec=5,
        spectral_embedding_dim=16,
        token_dim=16,
        num_heads=4,
        img_net=_TinyImageEncoder(),
        spec_net=_TinySpectralEncoder(),
    )


class _TinySpectralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = 16
        self.projection = nn.Linear(1, self.embedding_dim)

    def forward(self, spectra):
        summary = spectra.mean(dim=-1).squeeze(1).unsqueeze(-1)
        return self.projection(summary)


def _cuda_observations(batch_size=8):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(9182)
    image = torch.randn(
        batch_size, 1, 12, 12, device=device, generator=generator
    )
    spectra = torch.randn(
        batch_size, 1, 5, 64, device=device, generator=generator
    )
    positions = torch.randn(
        batch_size, 5, 2, device=device, generator=generator
    )
    targets = (
        torch.rand(
            batch_size,
            len(config.TARGET_NAMES),
            device=device,
            generator=generator,
        )
        * 1.6
        - 0.8
    )
    context = {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size, device=device),
        "image_snr": torch.linspace(
            5.0, 1000.0, batch_size, device=device
        ),
        "central_halpha_snr": torch.linspace(
            1.0, 200.0, batch_size, device=device
        ),
    }
    return image, spectra, positions, targets, context


def _compiled_amp_step(model, optimizer, forward):
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    state_keys = tuple(model.state_dict())
    feature_extractor = getattr(model, "feature_extractor", None)
    model = train._maybe_compile_model(
        model,
        {
            "use_compile": True,
            "compile_mode": "default",
            "compile_backend": "inductor",
        },
    )
    assert {id(parameter) for parameter in model.parameters()} == parameter_ids
    assert tuple(model.state_dict()) == state_keys
    if feature_extractor is not None:
        assert model.feature_extractor is feature_extractor

    scaler = torch.amp.GradScaler("cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = forward(model)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    scaler.step(optimizer)
    scaler.update()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA node to exercise TorchInductor and FP16 autocast",
)
def test_cuda_pretraining_compiles_and_runs_one_amp_optimizer_step():
    torch.manual_seed(int(config.pretrain["seed"]))
    image, spectra, positions, targets, context = _cuda_observations()
    model = CCLPretrain(
        backbone=_feature_extractor(),
        projector=nn.Sequential(
            nn.Linear(1024 + len(config.ORACLE_CONTEXT_FIELDS), 64),
            nn.GELU(),
            nn.Linear(64, 16),
        ),
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    _compiled_amp_step(
        model,
        optimizer,
        lambda owner: owner(
            image,
            spectra,
            positions,
            labels=targets,
            observation_context=context,
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA node to exercise TorchInductor and FP16 autocast",
)
def test_cuda_npe_compiles_without_replacing_frozen_feature_variables_and_uses_amp():
    torch.manual_seed(int(config.train["seed"]))
    image, spectra, positions, targets, context = _cuda_observations()
    extractor = _feature_extractor().cuda()
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    flow = BoundedHybridCircularFlow(
        features=len(config.TARGET_NAMES),
        theta_index=config.TARGET_NAMES.index("theta_int"),
        context_features=1024 + len(config.ORACLE_CONTEXT_FIELDS),
        num_bounded_layers=1,
        num_theta_layers=1,
        num_bins=4,
        hidden_features=16,
        num_blocks=1,
        theta_hidden_features=16,
    ).cuda()
    model = KLNPE(
        feature_extractor=extractor,
        flow=flow,
        feature_names=config.TARGET_NAMES,
    ).cuda()
    optimizer = torch.optim.AdamW(
        train._npe_optimizer_parameters(model, config.train),
        lr=3e-4,
    )

    _compiled_amp_step(
        model,
        optimizer,
        lambda owner: owner(
            image,
            spectra,
            targets,
            fiber_positions=positions,
            observation_context=context,
        ),
    )
