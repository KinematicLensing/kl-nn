"""Contracts for the fixed-shape, three-branch feature extractor."""

import copy

import pytest
import torch
from torch import nn

import config
import train
from networks import (
    FEATURE_DIM,
    IMAGE_FEATURE_DIM,
    METADATA_FEATURE_DIM,
    SPECTRAL_FEATURE_DIM,
    ImageSpectrumFilmFusion,
    JointSpecCNN,
    KLNPE,
    MetadataMLP,
    SimpleFusionFeatureExtractor,
    build_feature_extractor,
)


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(1, IMAGE_FEATURE_DIM)

    def forward(self, image):
        return self.projection(image.mean(dim=(-2, -1)))


class TinySpectralEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(5 * 64, SPECTRAL_FEATURE_DIM)

    def forward(self, spectra):
        return self.projection(spectra.flatten(start_dim=1))


class TinyMetadataEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(13, METADATA_FEATURE_DIM)

    def forward(self, metadata):
        return self.projection(metadata)


def _context(batch_size=2):
    return {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size),
        "image_snr": torch.linspace(50.0, 500.0, batch_size),
        "central_halpha_snr": torch.linspace(20.0, 120.0, batch_size),
    }


def _inputs(batch_size=2, *, requires_grad=False):
    generator = torch.Generator().manual_seed(20260811)
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
        5,
        64,
        generator=generator,
        requires_grad=requires_grad,
    )
    positions = torch.randn(
        batch_size,
        5,
        2,
        generator=generator,
        requires_grad=requires_grad,
    )
    return image, spectra, positions, _context(batch_size)


def _extractor(seed=17):
    torch.manual_seed(seed)
    return SimpleFusionFeatureExtractor(
        nspec=5,
        img_net=TinyImageEncoder(),
        spec_net=TinySpectralEncoder(),
        metadata_net=TinyMetadataEncoder(),
    )


def _clone_context(context):
    return {name: value.clone() for name, value in context.items()}


def test_only_three_catalog_scalars_are_accepted():
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    leaked = _clone_context(context)
    leaked["halpha_flux_true"] = torch.ones(image.shape[0])
    with pytest.raises(ValueError, match="halpha_flux_true"):
        extractor(image, spectra, positions, leaked)


def test_feature_dimensions_and_fixed_shape_output():
    assert IMAGE_FEATURE_DIM == 512
    assert SPECTRAL_FEATURE_DIM == 512
    assert METADATA_FEATURE_DIM == 128
    assert FEATURE_DIM == 1152
    assert MetadataMLP(nspec=5).input_dim == 13

    output = _extractor().eval()(*_inputs())
    assert output.shape == (2, FEATURE_DIM)
    assert torch.isfinite(output).all()


def test_joint_spectral_cnn_requires_five_by_sixty_four_and_returns_512():
    encoder = JointSpecCNN(nspec=5).eval()
    with torch.inference_mode():
        output = encoder(torch.randn(2, 1, 5, 64))
    assert output.shape == (2, SPECTRAL_FEATURE_DIM)
    assert torch.isfinite(output).all()

    with pytest.raises(ValueError, match=r"5, 64"):
        encoder(torch.randn(2, 1, 4, 64))
    with pytest.raises(ValueError, match=r"5, 64"):
        encoder(torch.randn(2, 1, 5, 63))


def test_joint_spectral_cnn_keeps_sixteen_wavelength_bins_before_the_final_kernel():
    encoder = JointSpecCNN(nspec=5)
    pools = [module for module in encoder.cnn_spec if isinstance(module, nn.MaxPool2d)]
    assert len(pools) == 2
    assert encoder.pooled_wavelength_count == 16
    last_conv = [module for module in encoder.cnn_spec if isinstance(module, nn.Conv2d)][-1]
    assert last_conv.kernel_size == (5, 16)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (
            lambda image, spectra, positions: (
                image[:, :0],
                spectra,
                positions,
            ),
            "image must have shape",
        ),
        (
            lambda image, spectra, positions: (
                image,
                spectra[:, :, :4],
                positions,
            ),
            "spectra must have shape",
        ),
        (
            lambda image, spectra, positions: (
                image,
                spectra[..., :-1],
                positions,
            ),
            "spectra must have shape",
        ),
        (
            lambda image, spectra, positions: (
                image,
                spectra,
                positions[:, :4],
            ),
            "fiber_positions must have shape",
        ),
    ],
)
def test_extractor_rejects_noncanonical_observation_shapes(replacement, message):
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    invalid = replacement(image, spectra, positions)
    with pytest.raises(ValueError, match=message):
        extractor(*invalid, context)


def test_fixed_fiber_mask_requires_all_five_fibers():
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    full_mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.inference_mode():
        reference = extractor(image, spectra, positions, context)
        masked = extractor(
            image, spectra, positions, context, fiber_mask=full_mask
        )
    torch.testing.assert_close(reference, masked)

    with pytest.raises(TypeError, match="bool"):
        extractor(
            image,
            spectra,
            positions,
            context,
            fiber_mask=torch.ones(2, 5),
        )
    with pytest.raises(ValueError, match="shape"):
        extractor(
            image,
            spectra,
            positions,
            context,
            fiber_mask=torch.ones(2, 4, dtype=torch.bool),
        )
    partial_mask = full_mask.clone()
    partial_mask[0, -1] = False
    with pytest.raises(ValueError, match="requires all configured fibers"):
        extractor(
            image, spectra, positions, context, fiber_mask=partial_mask
        )


def test_joint_spectral_normalization_is_globally_scale_invariant():
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    with torch.inference_mode():
        reference = extractor(image, spectra, positions, context)
        scaled = extractor(image, 7.0 * spectra, positions, context)
    torch.testing.assert_close(reference, scaled, atol=2e-6, rtol=2e-6)


def test_joint_spectral_branch_preserves_relative_fiber_strength_and_order():
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    relative_change = spectra.clone()
    relative_change[:, :, 0] *= 4.0
    permutation = torch.tensor([3, 1, 4, 0, 2])

    with torch.inference_mode():
        reference = extractor(image, spectra, positions, context)
        changed = extractor(image, relative_change, positions, context)
        permuted = extractor(
            image, spectra[:, :, permutation], positions, context
        )
    assert not torch.allclose(reference, changed, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(reference, permuted, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("rmag_true", torch.tensor([19.0, 19.5])),
        ("image_snr", torch.tensor([250.0, 750.0])),
        ("central_halpha_snr", torch.tensor([60.0, 180.0])),
    ],
)
def test_each_catalog_scalar_changes_metadata_features(field, replacement):
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    changed_context = _clone_context(context)
    changed_context[field] = replacement
    with torch.inference_mode():
        reference = extractor(image, spectra, positions, context)
        changed = extractor(image, spectra, positions, changed_context)
    assert not torch.allclose(
        reference[:, -METADATA_FEATURE_DIM:],
        changed[:, -METADATA_FEATURE_DIM:],
        atol=1e-7,
        rtol=1e-7,
    )


def test_fiber_positions_change_metadata_features():
    extractor = _extractor().eval()
    image, spectra, positions, context = _inputs()
    changed_positions = positions.clone()
    changed_positions[:, 0, 0] += 0.5
    with torch.inference_mode():
        reference = extractor(image, spectra, positions, context)
        changed = extractor(image, spectra, changed_positions, context)
    assert not torch.allclose(
        reference[:, -METADATA_FEATURE_DIM:],
        changed[:, -METADATA_FEATURE_DIM:],
        atol=1e-7,
        rtol=1e-7,
    )


def test_production_image_and_spectral_parameter_counts_are_same_order():
    extractor = SimpleFusionFeatureExtractor(nspec=5)
    image_parameters = sum(
        parameter.numel() for parameter in extractor.img_net.parameters()
    )
    spectral_parameters = sum(
        parameter.numel() for parameter in extractor.spec_net.parameters()
    )
    ratio = max(image_parameters, spectral_parameters) / min(
        image_parameters, spectral_parameters
    )
    assert ratio < 10.0


def test_all_three_feature_branches_receive_gradients():
    extractor = _extractor().train()
    output = extractor(*_inputs())
    weights = torch.linspace(-1.0, 1.0, FEATURE_DIM)
    (output * weights).sum().backward()

    for branch in (
        extractor.img_net,
        extractor.spec_net,
        extractor.metadata_net,
    ):
        gradients = [
            parameter.grad
            for parameter in branch.parameters()
            if parameter.requires_grad
        ]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_factory_builds_simple_fusion_extractor():
    backbone = build_feature_extractor(nspec=5)
    assert isinstance(backbone, SimpleFusionFeatureExtractor)
    assert backbone.output_dim == FEATURE_DIM


def test_simple_fusion_state_dict_round_trip():
    original = _extractor().eval()
    restored = _extractor(seed=999).eval()
    inputs = _inputs(batch_size=1)
    assert not torch.allclose(original(*inputs), restored(*inputs))
    restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)
    torch.testing.assert_close(original(*inputs), restored(*inputs))


def test_film_fusion_is_identity_at_initialization():
    fusion = ImageSpectrumFilmFusion()
    features = torch.randn(4, FEATURE_DIM)
    torch.testing.assert_close(fusion(features), features)


def test_film_fusion_rejects_wrong_feature_width():
    fusion = ImageSpectrumFilmFusion()
    with pytest.raises(ValueError, match="1152"):
        fusion(torch.randn(2, FEATURE_DIM - 1))


def test_film_fusion_leaves_metadata_unmodulated_and_mixes_image_from_spectrum():
    fusion = ImageSpectrumFilmFusion()
    nn.init.normal_(fusion.spectral_to_image.weight, std=0.05)
    nn.init.normal_(fusion.image_to_spectral.weight, std=0.05)
    features = torch.randn(3, FEATURE_DIM)
    fused = fusion(features)
    meta = slice(IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM, FEATURE_DIM)
    torch.testing.assert_close(fused[:, meta], features[:, meta])

    perturbed = features.clone()
    perturbed[:, IMAGE_FEATURE_DIM:IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM] += 1.5
    fused_perturbed = fusion(perturbed)
    assert not torch.allclose(
        fused[:, :IMAGE_FEATURE_DIM],
        fused_perturbed[:, :IMAGE_FEATURE_DIM],
    )


class _Tiny1152Extractor(nn.Module):
    output_dim = FEATURE_DIM

    def forward(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        del spectra, fiber_positions, observation_context, fiber_mask
        return image.new_zeros(image.shape[0], FEATURE_DIM)


def test_klnpe_fusion_remains_trainable_when_the_extractor_is_frozen():
    model = KLNPE(feature_extractor=_Tiny1152Extractor())
    for parameter in model.feature_extractor.parameters():
        parameter.requires_grad = False
    assert all(parameter.requires_grad for parameter in model.fusion.parameters())
    groups = train._npe_optimizer_parameters(
        model,
        {
            "initial_learning_rate": 3e-4,
            "non_theta_learning_rate": 2e-4,
            "theta_learning_rate": 1e-4,
        },
    )
    shared_ids = {id(parameter) for parameter in groups[0]["params"]}
    fusion_ids = {id(parameter) for parameter in model.fusion.parameters()}
    assert fusion_ids <= shared_ids
    extractor_ids = {id(parameter) for parameter in model.feature_extractor.parameters()}
    assert extractor_ids.isdisjoint(shared_ids)


def test_klnpe_can_disable_image_spectrum_fusion(monkeypatch):
    monkeypatch.setitem(config.train, "use_image_spectrum_fusion", False)
    model = KLNPE(feature_extractor=_Tiny1152Extractor())
    assert isinstance(model.fusion, nn.Identity)
