import copy

import pytest
import torch
from torch import nn

import config
import networks
from networks import (
    CCLPretrain,
    DivisibleMeanPool1d,
    FiberSetAttention,
    SharedSpecCNN,
    Stage3FeatureExtractor,
    build_feature_extractor,
)


class TinyImageEncoder(nn.Module):
    """Cheap 512-feature image branch for fusion-contract unit tests."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)

    def forward(self, image):
        pooled = image.mean(dim=(-2, -1))
        return self.projection(pooled)


def _inputs(batch_size=2, fiber_count=5, *, requires_grad=False):
    generator = torch.Generator().manual_seed(20260811 + fiber_count)
    image = torch.randn(
        (batch_size, 1, 12, 12), generator=generator, requires_grad=requires_grad
    )
    spectra = torch.randn(
        (batch_size, 1, fiber_count, 64),
        generator=generator,
        requires_grad=requires_grad,
    )
    fiber_positions = torch.randn(
        (batch_size, fiber_count, 2),
        generator=generator,
        requires_grad=requires_grad,
    )
    return image, spectra, fiber_positions


def _extractor():
    torch.manual_seed(17)
    return Stage3FeatureExtractor(nspec=5, img_net=TinyImageEncoder())


def test_shared_spectral_encoder_preserves_absolute_wavelength_location():
    torch.manual_seed(5)
    encoder = SharedSpecCNN(embedding_dim=32).eval()
    spectra = torch.zeros((1, 1, 2, 64))
    spectra[0, 0, 0, 20] = 1.0
    spectra[0, 0, 1, 28] = 1.0

    encoded = encoder(spectra)

    assert not torch.allclose(encoded[:, 0], encoded[:, 1])


def test_divisible_mean_pool_matches_adaptive_pool_without_adaptive_kernel():
    inputs = torch.randn((3, 7, 16), requires_grad=True)
    pool = DivisibleMeanPool1d(8)

    output = pool(inputs)

    torch.testing.assert_close(
        output,
        torch.nn.functional.adaptive_avg_pool1d(inputs, 8),
    )
    output.square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()

    with pytest.raises(ValueError, match="positive multiple"):
        pool(torch.randn((1, 2, 15)))


def test_shared_spectral_encoder_is_fiber_permutation_equivariant_and_differentiable():
    encoder = SharedSpecCNN(embedding_dim=32).eval()
    _, spectra, _ = _inputs(requires_grad=True)
    spectra = spectra.detach().requires_grad_(True)
    spectra.data[:, :, 3] = spectra.data[:, :, 0]
    permutation = torch.tensor([3, 1, 4, 0, 2])

    encoded = encoder(spectra)
    permuted = encoder(spectra[:, :, permutation, :])

    assert encoded.shape == (2, 5, 32)
    torch.testing.assert_close(encoded[:, 0], encoded[:, 3])
    torch.testing.assert_close(permuted, encoded[:, permutation], atol=1e-6, rtol=1e-5)
    assert any(isinstance(layer, nn.Conv1d) for layer in encoder.modules())
    assert not any(isinstance(layer, nn.Conv2d) for layer in encoder.modules())

    encoded.square().mean().backward()
    assert spectra.grad is not None
    assert torch.isfinite(spectra.grad).all()
    first_conv = next(layer for layer in encoder.modules() if isinstance(layer, nn.Conv1d))
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()


@pytest.mark.parametrize("fiber_count", (3, 5))
def test_stage3_extractor_output_shape_finite_and_variable_fiber_count(fiber_count):
    extractor = _extractor().eval()
    image, spectra, fiber_positions = _inputs(fiber_count=fiber_count)

    output = extractor(image, spectra, fiber_positions)

    assert output.shape == (2, 1024)
    assert torch.isfinite(output).all()



def test_stage3_real_image_encoder_production_shape_channels_last():
    torch.manual_seed(23)
    extractor = Stage3FeatureExtractor(nspec=5).eval()
    image = torch.randn((1, 1, 48, 48)).contiguous(
        memory_format=torch.channels_last
    )
    spectra = torch.randn((1, 1, 5, 64)).contiguous(
        memory_format=torch.channels_last
    )
    fiber_positions = torch.randn((1, 5, 2))

    with torch.inference_mode():
        output = extractor(image, spectra, fiber_positions)

    assert output.shape == (1, 1024)
    assert torch.isfinite(output).all()

def test_stage3_joint_fiber_permutation_is_invariant_but_association_matters():
    extractor = _extractor().eval()
    image, spectra, fiber_positions = _inputs()
    mask = torch.tensor(
        [[True, True, True, False, True], [True, False, True, True, True]]
    )
    permutation = torch.tensor([4, 2, 0, 3, 1])

    reference = extractor(image, spectra, fiber_positions, mask)
    joint_permutation = extractor(
        image,
        spectra[:, :, permutation, :],
        fiber_positions[:, permutation],
        mask[:, permutation],
    )
    torch.testing.assert_close(reference, joint_permutation, atol=2e-5, rtol=2e-5)

    # Use all observed fibers for the negative control so it can only detect
    # broken spectrum-position association, not a changed mask selection.
    association_reference = extractor(image, spectra, fiber_positions)
    spectra_only = extractor(
        image,
        spectra[:, :, permutation, :],
        fiber_positions,
    )
    assert not torch.allclose(
        association_reference, spectra_only, atol=1e-5, rtol=1e-5
    )



def test_relative_spectral_strength_preserves_legacy_global_normalization():
    _, spectra, _ = _inputs()
    relative_strength = Stage3FeatureExtractor._relative_spectral_strength(spectra)
    reconstructed = torch.nn.functional.normalize(spectra, dim=-1) * relative_strength
    legacy_normalized = torch.nn.functional.normalize(spectra, dim=(-2, -1))

    torch.testing.assert_close(reconstructed, legacy_normalized)

    scaled = spectra.clone()
    scaled[:, :, 0] *= 2.0
    scaled_strength = Stage3FeatureExtractor._relative_spectral_strength(scaled)
    assert not torch.allclose(relative_strength, scaled_strength)

def test_masked_fiber_values_are_ignored_and_receive_zero_gradient():
    extractor = _extractor().eval()
    image, spectra, fiber_positions = _inputs(requires_grad=True)
    mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, False]]
    )

    reference = extractor(image, spectra, fiber_positions, mask)
    altered_spectra = spectra.detach().clone()
    altered_positions = fiber_positions.detach().clone()
    altered_spectra[~mask[:, None, :, None].expand_as(altered_spectra)] = 1e5
    altered_positions[~mask.unsqueeze(-1).expand_as(altered_positions)] = -1e5
    altered = extractor(image.detach(), altered_spectra, altered_positions, mask)
    torch.testing.assert_close(reference.detach(), altered, atol=1e-6, rtol=1e-6)

    weights = torch.linspace(-1.0, 1.0, reference.shape[1])
    (reference * weights).sum().backward()
    spectral_mask = mask[:, None, :, None].expand_as(spectra)
    position_mask = mask.unsqueeze(-1).expand_as(fiber_positions)
    assert torch.count_nonzero(spectra.grad[~spectral_mask]) == 0
    assert torch.count_nonzero(fiber_positions.grad[~position_mask]) == 0
    assert torch.count_nonzero(spectra.grad[spectral_mask]) > 0
    assert torch.count_nonzero(fiber_positions.grad[position_mask]) > 0
    gradients = (
        image.grad,
        next(layer for layer in extractor.spec_net.modules() if isinstance(layer, nn.Conv1d)).weight.grad,
        extractor.token_projection[0].weight.grad,
        extractor.fiber_set_encoder.self_attention.in_proj_weight.grad,
        extractor.image_fiber_attention.in_proj_weight.grad,
        extractor.fusion_mlp[0].weight.grad,
    )
    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_omitted_mask_matches_all_observed_mask_for_channels_last_inputs():
    extractor = _extractor().eval()
    image, spectra, fiber_positions = _inputs(batch_size=1)
    image = image.contiguous(memory_format=torch.channels_last)
    spectra = spectra.contiguous(memory_format=torch.channels_last)
    all_observed = torch.ones((1, 5), dtype=torch.bool)

    implicit = extractor(image, spectra, fiber_positions)
    explicit = extractor(image, spectra, fiber_positions, all_observed)

    torch.testing.assert_close(implicit, explicit)


def test_stage3_mask_and_shape_errors_are_explicit():
    extractor = _extractor().eval()
    image, spectra, fiber_positions = _inputs()

    with pytest.raises(TypeError, match="bool"):
        extractor(image, spectra, fiber_positions, torch.ones((2, 5)))
    with pytest.raises(ValueError, match="shape"):
        extractor(image, spectra, fiber_positions, torch.ones((2, 4), dtype=torch.bool))
    with pytest.raises(ValueError, match="same fiber count"):
        extractor(image, spectra, fiber_positions[:, :4])
    with pytest.raises(ValueError, match="at least one"):
        extractor(
            image,
            spectra,
            fiber_positions,
            torch.zeros((2, 5), dtype=torch.bool),
        )


def test_backbone_factory_and_ccl_select_stage3(monkeypatch):
    monkeypatch.setattr(networks, "ImgCNN", TinyImageEncoder)
    monkeypatch.setitem(config.pretrain, "backbone_type", "stage3")
    backbone = build_feature_extractor("stage3", nspec=5)
    assert isinstance(backbone, Stage3FeatureExtractor)
    with pytest.raises(ValueError, match="backbone_type"):
        build_feature_extractor("unknown", nspec=5)

    model = CCLPretrain(projector_dim=16)
    assert isinstance(model.backbone, Stage3FeatureExtractor)
    image, spectra, fiber_positions = _inputs(batch_size=4)
    labels = torch.rand((4, 8)) * 2.0 - 1.0
    loss = model(image, spectra, fiber_positions, labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.backbone.image_fiber_attention.in_proj_weight.grad is not None



def test_klnpe_uses_recorded_stage3_backbone_for_fresh_construction(monkeypatch):
    monkeypatch.setattr(networks, "ImgCNN", TinyImageEncoder)
    monkeypatch.setitem(config.train, "backbone_type", "stage3")

    model = networks.KLNPE(mode=0, batch_size=2, nfeatures=8, nspec=5)

    assert isinstance(model.feature_extractor, Stage3FeatureExtractor)

def test_stage3_state_dict_round_trip():
    original = _extractor().eval()
    torch.manual_seed(999)
    restored = Stage3FeatureExtractor(nspec=5, img_net=TinyImageEncoder()).eval()
    inputs = _inputs(batch_size=1)
    assert not torch.allclose(original(*inputs), restored(*inputs))

    restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)

    torch.testing.assert_close(original(*inputs), restored(*inputs))
