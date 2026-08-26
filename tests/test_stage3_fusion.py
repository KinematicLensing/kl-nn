"""Contracts for the sole Stage-3 multimodal feature extractor."""

import copy

import pytest
import torch
from torch import nn

import networks
from networks import (
    CCLPretrain,
    DivisibleMeanPool1d,
    SharedSpecCNN,
    Stage3FeatureExtractor,
    build_feature_extractor,
)


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, 512)

    def forward(self, image):
        return self.projection(image.mean(dim=(-2, -1)))


def _inputs(batch_size=2, fiber_count=5, *, requires_grad=False):
    generator = torch.Generator().manual_seed(20260811 + fiber_count)
    image = torch.randn(
        batch_size,
        1,
        12,
        12,
        generator=generator,
        requires_grad=requires_grad,
    )
    spectra = torch.randn(
        batch_size,
        1,
        fiber_count,
        64,
        generator=generator,
        requires_grad=requires_grad,
    )
    positions = torch.randn(
        batch_size,
        fiber_count,
        2,
        generator=generator,
        requires_grad=requires_grad,
    )
    return image, spectra, positions


def _extractor():
    torch.manual_seed(17)
    return Stage3FeatureExtractor(nspec=5, img_net=TinyImageEncoder())


def test_shared_spectral_encoder_preserves_wavelength_location():
    torch.manual_seed(5)
    encoder = SharedSpecCNN(embedding_dim=32).eval()
    spectra = torch.zeros(1, 1, 2, 64)
    spectra[0, 0, 0, 20] = 1.0
    spectra[0, 0, 1, 28] = 1.0
    encoded = encoder(spectra)
    assert not torch.allclose(encoded[:, 0], encoded[:, 1])


def test_divisible_pool_matches_adaptive_pool_and_is_differentiable():
    inputs = torch.randn(3, 7, 16, requires_grad=True)
    output = DivisibleMeanPool1d(8)(inputs)
    torch.testing.assert_close(
        output, torch.nn.functional.adaptive_avg_pool1d(inputs, 8)
    )
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    with pytest.raises(ValueError, match="positive multiple"):
        DivisibleMeanPool1d(8)(torch.randn(1, 2, 15))


def test_shared_spectral_encoder_is_fiber_permutation_equivariant():
    encoder = SharedSpecCNN(embedding_dim=32).eval()
    _, spectra, _ = _inputs(requires_grad=True)
    permutation = torch.tensor([3, 1, 4, 0, 2])
    encoded = encoder(spectra)
    permuted = encoder(spectra[:, :, permutation])
    torch.testing.assert_close(
        permuted, encoded[:, permutation], atol=1e-6, rtol=1e-5
    )
    encoded.square().mean().backward()
    assert spectra.grad is not None and torch.isfinite(spectra.grad).all()


@pytest.mark.parametrize("fiber_count", (3, 5))
def test_stage3_output_shape_is_finite_for_variable_fiber_count(fiber_count):
    output = _extractor().eval()(*_inputs(fiber_count=fiber_count))
    assert output.shape == (2, 1024)
    assert torch.isfinite(output).all()


def test_production_image_encoder_shape_with_channels_last():
    extractor = Stage3FeatureExtractor(nspec=5).eval()
    image = torch.randn(1, 1, 48, 48).contiguous(
        memory_format=torch.channels_last
    )
    spectra = torch.randn(1, 1, 5, 64).contiguous(
        memory_format=torch.channels_last
    )
    with torch.inference_mode():
        output = extractor(image, spectra, torch.randn(1, 5, 2))
    assert output.shape == (1, 1024)
    assert torch.isfinite(output).all()


def test_joint_fiber_permutation_is_invariant_but_association_matters():
    extractor = _extractor().eval()
    image, spectra, positions = _inputs()
    mask = torch.tensor(
        [[True, True, True, False, True], [True, False, True, True, True]]
    )
    permutation = torch.tensor([4, 2, 0, 3, 1])
    reference = extractor(image, spectra, positions, mask)
    permuted = extractor(
        image,
        spectra[:, :, permutation],
        positions[:, permutation],
        mask[:, permutation],
    )
    torch.testing.assert_close(reference, permuted, atol=2e-5, rtol=2e-5)

    association_reference = extractor(image, spectra, positions)
    broken = extractor(image, spectra[:, :, permutation], positions)
    assert not torch.allclose(association_reference, broken, atol=1e-5, rtol=1e-5)


def test_relative_spectral_strength_matches_global_normalization():
    _, spectra, _ = _inputs()
    strength = Stage3FeatureExtractor._relative_spectral_strength(spectra)
    reconstructed = torch.nn.functional.normalize(spectra, dim=-1) * strength
    expected = torch.nn.functional.normalize(spectra, dim=(-2, -1))
    torch.testing.assert_close(reconstructed, expected)


def test_absolute_line_strength_preserves_global_amplitude_and_fiber_order():
    _, spectra, _ = _inputs()
    permutation = torch.tensor([3, 1, 4, 0, 2])
    strength = Stage3FeatureExtractor._absolute_spectral_strength(spectra)
    scaled = Stage3FeatureExtractor._absolute_spectral_strength(4.0 * spectra)
    permuted = Stage3FeatureExtractor._absolute_spectral_strength(
        spectra[:, :, permutation]
    )

    assert strength.shape == (2, 1, 5, 1)
    assert torch.all(scaled > strength)
    torch.testing.assert_close(permuted, strength[:, :, permutation])


def test_global_spectral_amplitude_changes_stage3_features():
    extractor = _extractor().eval()
    image, spectra, positions = _inputs()
    with torch.inference_mode():
        reference = extractor(image, spectra, positions)
        scaled = extractor(image, 4.0 * spectra, positions)
    assert not torch.allclose(reference, scaled, atol=1e-7, rtol=1e-7)


def test_masked_fiber_values_are_ignored_and_receive_zero_gradient():
    extractor = _extractor().eval()
    image, spectra, positions = _inputs(requires_grad=True)
    mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, False]]
    )
    reference = extractor(image, spectra, positions, mask)
    altered_spectra = spectra.detach().clone()
    altered_positions = positions.detach().clone()
    altered_spectra[~mask[:, None, :, None].expand_as(altered_spectra)] = 1e5
    altered_positions[~mask.unsqueeze(-1).expand_as(altered_positions)] = -1e5
    altered = extractor(
        image.detach(), altered_spectra, altered_positions, mask
    )
    torch.testing.assert_close(reference.detach(), altered, atol=1e-6, rtol=1e-6)

    weights = torch.linspace(
        -1.0, 1.0, reference.shape[-1], device=reference.device
    )
    (reference * weights).sum().backward()
    spectral_mask = mask[:, None, :, None].expand_as(spectra)
    position_mask = mask.unsqueeze(-1).expand_as(positions)
    assert torch.count_nonzero(spectra.grad[~spectral_mask]) == 0
    assert torch.count_nonzero(positions.grad[~position_mask]) == 0
    assert torch.count_nonzero(spectra.grad[spectral_mask]) > 0
    assert torch.count_nonzero(positions.grad[position_mask]) > 0


def test_stage3_mask_and_shape_errors_are_explicit():
    extractor = _extractor().eval()
    image, spectra, positions = _inputs()
    with pytest.raises(TypeError, match="bool"):
        extractor(image, spectra, positions, torch.ones(2, 5))
    with pytest.raises(ValueError, match="shape"):
        extractor(image, spectra, positions, torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="same fiber count"):
        extractor(image, spectra, positions[:, :4])
    with pytest.raises(ValueError, match="at least one"):
        extractor(image, spectra, positions, torch.zeros(2, 5, dtype=torch.bool))


def test_factory_and_ccl_use_stage3_with_nine_targets_and_oracle_context(
    monkeypatch,
):
    monkeypatch.setattr(networks, "ImgCNN", TinyImageEncoder)
    backbone = build_feature_extractor(nspec=5)
    assert isinstance(backbone, Stage3FeatureExtractor)

    model = CCLPretrain(projector_dim=16)
    image, spectra, positions = _inputs(batch_size=4)
    labels = torch.rand(4, 9) * 2.0 - 1.0
    context = {
        "rmag_true": torch.linspace(18.0, 21.0, 4),
        "image_snr": torch.linspace(5.0, 1000.0, 4),
        "central_halpha_snr": torch.linspace(1.0, 200.0, 4),
    }
    loss = model(image, spectra, positions, labels, context)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.backbone.image_fiber_attention.in_proj_weight.grad is not None


def test_stage3_state_dict_round_trip():
    original = _extractor().eval()
    torch.manual_seed(999)
    restored = Stage3FeatureExtractor(
        nspec=5, img_net=TinyImageEncoder()
    ).eval()
    inputs = _inputs(batch_size=1)
    assert not torch.allclose(original(*inputs), restored(*inputs))
    restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
    torch.testing.assert_close(original(*inputs), restored(*inputs))
