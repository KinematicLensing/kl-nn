"""CCL oracle-context and persistent calibration contracts."""

import copy

import torch
from torch import nn

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
        "spectral_reference_quality": torch.linspace(
            5.0, 30.0, features.shape[0]
        ),
    }
    return features, labels, context


def _ccl_model():
    return CCLPretrain(
        backbone=IdentityBackbone(),
        projector=nn.Linear(1026, 16),
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


def test_ccl_pretrain_backbone_extraction_and_strict_reload():
    model = _ccl_model().eval()
    reloaded = _ccl_model().eval()
    reloaded.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    features, _, _ = _inputs()
    extracted = reloaded.extract_features(features, None, None)
    torch.testing.assert_close(extracted, features)


def test_calibration_buffers_are_persistent_for_pretraining_and_npe():
    for make_model in (_ccl_model, _npe_model):
        model = make_model()
        assert torch.isnan(model.image_noise_sigma)
        assert torch.isnan(model.spectral_reference_line_norm)
        with torch.no_grad():
            model.image_noise_sigma.copy_(torch.tensor(0.125))
            model.spectral_reference_line_norm.copy_(torch.tensor(3.5))

        restored = make_model()
        restored.load_state_dict(
            copy.deepcopy(model.state_dict()), strict=True
        )
        assert restored.image_noise_sigma.item() == 0.125
        assert restored.spectral_reference_line_norm.item() == 3.5
        assert restored.image_noise_sigma.item() > 0.0
        assert restored.spectral_reference_line_norm.item() > 0.0
