import torch
from torch import nn
import torch.nn.functional as F

import config
from networks import CCLPretrain, NormalizedShearMSELoss


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


def _configure_objective(monkeypatch, objective, weight=1.0):
    monkeypatch.setitem(config.train, "feature_names", FEATURE_NAMES)
    monkeypatch.setitem(config.pretrain, "ccl_objective", objective)
    monkeypatch.setitem(config.pretrain, "ccl_shear_loss_weight", weight)


def _inputs():
    generator = torch.Generator().manual_seed(2026)
    features = torch.randn((8, 1024), generator=generator)
    labels = torch.rand((8, len(FEATURE_NAMES)), generator=generator) * 2.0 - 1.0
    return features, labels


def test_normalized_shear_mse_has_expected_scale():
    prediction = torch.zeros((2, 2))
    target = torch.tensor(((-1.0, 1.0), (0.0, 0.0)))
    loss = NormalizedShearMSELoss(normalization=3.0)(prediction, target)

    torch.testing.assert_close(loss, 3.0 * F.mse_loss(prediction, target))


def test_original_ccl_objective_is_unchanged_and_has_no_shear_state(monkeypatch):
    _configure_objective(monkeypatch, "ccl")
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
    assert model.shear_head is None
    assert not any(key.startswith("shear_head.") for key in model.state_dict())


def test_ccl_shear_objective_adds_separate_backbone_loss(monkeypatch):
    weight = 1.25
    _configure_objective(monkeypatch, "ccl_shear", weight=weight)
    model = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    features, labels = _inputs()
    nn.init.zeros_(model.shear_head.weight)
    nn.init.zeros_(model.shear_head.bias)

    total, diagnostics = model(
        features,
        None,
        None,
        labels,
        return_diagnostics=True,
    )
    expected_shear = 3.0 * labels[:, :2].square().mean()

    torch.testing.assert_close(diagnostics["shear_loss"], expected_shear)
    torch.testing.assert_close(
        diagnostics["weighted_shear_loss"],
        weight * expected_shear,
    )
    torch.testing.assert_close(
        total,
        diagnostics["ccl_loss"] + diagnostics["weighted_shear_loss"],
    )

    total.backward()
    assert model.shear_head.weight.grad is not None
    assert torch.isfinite(model.shear_head.weight.grad).all()
    assert any(key.startswith("shear_head.") for key in model.state_dict())


def test_probe_backbone_extraction_and_strict_reload_still_work(monkeypatch):
    _configure_objective(monkeypatch, "ccl_shear")
    model = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    reloaded = CCLPretrain(backbone=IdentityBackbone(), projector_dim=16).eval()
    reloaded.load_state_dict(model.state_dict(), strict=True)
    features, _ = _inputs()

    extracted = reloaded.extract_features(features, None, None)

    torch.testing.assert_close(extracted, features)
