import torch
from torch import nn

import config
from networks import CCLPretrain


FEATURE_NAMES = [
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
]


class IdentityBackbone(nn.Module):
    def forward(self, x, y, fp):
        del y, fp
        return x


def _inputs():
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn((8, 1024), generator=generator)
    labels = torch.rand((8, len(FEATURE_NAMES)), generator=generator) * 2.0 - 1.0
    return features, labels


def test_ccl_pretrain_matches_direct_contrastive_loss(monkeypatch):
    monkeypatch.setitem(config.train, "feature_names", FEATURE_NAMES)
    model = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    features, labels = _inputs()

    projected = model.projector(features)
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
        return_diagnostics=True,
    )

    torch.testing.assert_close(actual, expected)
    assert diagnostics.keys() == expected_diagnostics.keys()


def test_ccl_pretrain_backbone_extraction_and_strict_reload(monkeypatch):
    monkeypatch.setitem(config.train, "feature_names", FEATURE_NAMES)
    model = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    reloaded = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    features, _ = _inputs()

    extracted = reloaded.extract_features(features, None, None)

    torch.testing.assert_close(extracted, features)
