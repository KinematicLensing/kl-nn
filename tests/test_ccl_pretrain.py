"""CCL oracle-context and checkpoint-state contracts."""

import copy

import torch
from torch import nn

import networks
from networks import CCLPretrain, KLNPE


FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
    "halpha_flux_true",
)


class IdentityBackbone(nn.Module):
    def forward(
        self, image, spectra, fiber_positions, fiber_mask=None
    ):
        del spectra, fiber_positions, fiber_mask
        return image


class EmptyFlow(nn.Module):
    pass


def _inputs():
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn((8, 1024), generator=generator)
    labels = torch.rand((8, len(FEATURE_NAMES)), generator=generator) * 2.0 - 1.0
    context = {
        "rmag_true": torch.linspace(18.0, 21.0, features.shape[0]),
        "image_snr": torch.linspace(5.0, 1000.0, features.shape[0]),
        "central_halpha_snr": torch.linspace(
            1.0, 200.0, features.shape[0]
        ),
    }
    return features, labels, context


def _ccl_model():
    return CCLPretrain(
        backbone=IdentityBackbone(),
        projector=nn.Linear(
            networks.FEATURE_DIM + len(networks.ORACLE_CONTEXT_FIELDS), 16
        ),
    )


def _npe_model():
    return KLNPE(
        feature_extractor=IdentityBackbone(),
        flow=EmptyFlow(),
        feature_names=FEATURE_NAMES,
    )


def test_ccl_pretrain_matches_direct_oracle_context_loss():
    model = _ccl_model().eval()
    features, labels, context = _inputs()
    normalized_context = model.context_normalizer(
        context, features.shape[0], features
    )
    projected = model.projector(
        torch.cat((features, normalized_context), dim=-1)
    )
    expected, expected_diagnostics = model.ccl_loss(
        projected,
        labels,
        return_diagnostics=True,
    )
    actual, diagnostics = model(
        features,
        None,
        None,
        labels,
        context,
        return_diagnostics=True,
    )

    torch.testing.assert_close(actual, expected)
    assert diagnostics.keys() == expected_diagnostics.keys()


def test_distributed_ccl_uses_only_rank_local_anchor_rows(monkeypatch):
    model = _ccl_model().eval()
    features, labels, context = _inputs()
    normalized_context = model.context_normalizer(
        context, features.shape[0], features
    )
    local_projected = model.projector(
        torch.cat((features, normalized_context), dim=-1)
    )
    remote_projected = local_projected.detach() + 0.25
    remote_labels = torch.roll(labels, shifts=1, dims=0)
    expected = model.ccl_loss(
        torch.cat((remote_projected, local_projected), dim=0),
        torch.cat((remote_labels, labels), dim=0),
        anchor_start=labels.shape[0],
        anchor_count=labels.shape[0],
    )

    def fake_all_gather(value):
        if value.shape[-1] == local_projected.shape[-1]:
            return remote_projected, value
        return remote_labels, value

    monkeypatch.setattr(networks.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(networks.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(networks, "all_gather", fake_all_gather)
    actual = model(features, None, None, labels, context)

    torch.testing.assert_close(actual, expected)


def test_ccl_pretrain_backbone_extraction_and_strict_reload():
    model = _ccl_model().eval()
    reloaded = _ccl_model().eval()
    reloaded.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    features, _, _ = _inputs()
    extracted = reloaded.extract_features(features, None, None)
    torch.testing.assert_close(extracted, features)


def test_retired_global_calibration_buffers_are_not_persisted():
    for make_model in (_ccl_model, _npe_model):
        model = make_model()
        assert not hasattr(model, "image_noise_sigma")
        assert not hasattr(model, "spectral_reference_line_norm")
        state = copy.deepcopy(model.state_dict())
        assert "image_noise_sigma" not in state
        assert "spectral_reference_line_norm" not in state

        restored = make_model()
        restored.load_state_dict(state, strict=True)
        assert not hasattr(restored, "image_noise_sigma")
        assert not hasattr(restored, "spectral_reference_line_norm")
