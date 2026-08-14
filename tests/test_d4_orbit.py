import copy

import pytest
import torch
from torch import nn

from data import (
    D4_ELEMENTS,
    D4_INVERSES,
    apply_d4_to_datavector,
    transform_d4_feature_blocks,
    transform_d4_fiber_mask,
)
from networks import (
    BACKBONE_TYPES,
    CCLPretrain,
    D4EquivariantCCLProjector,
    D4OrbitFeatureExtractor,
    build_feature_extractor,
)


SCALAR_CHANNELS = 512
SPIN1_CHANNELS = 256
SPIN2_CHANNELS = 256
OUTPUT_DIM = SCALAR_CHANNELS + SPIN1_CHANNELS + SPIN2_CHANNELS


def _asymmetric_inputs(batch_size=2, *, requires_grad=False):
    image = torch.arange(
        batch_size * 25, dtype=torch.float32
    ).reshape(batch_size, 1, 5, 5)
    image = (image / 17.0).requires_grad_(requires_grad)
    spectra = torch.arange(
        batch_size * 5 * 8, dtype=torch.float32
    ).reshape(batch_size, 1, 5, 8)
    spectra = ((spectra + 1.0) / 23.0).requires_grad_(requires_grad)
    positions = torch.tensor(
        [
            [[1.7, 0.2], [-1.3, -0.4], [0.1, -0.2], [-0.3, 1.4], [0.6, -1.1]],
            [[1.2, -0.7], [-1.8, 0.3], [-0.2, 0.4], [0.8, 1.6], [-0.5, -1.3]],
        ],
        dtype=torch.float32,
    )[:batch_size]
    positions = positions.clone().requires_grad_(requires_grad)
    return image, spectra, positions


def _mask(batch_size=2):
    return torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, True]],
        dtype=torch.bool,
    )[:batch_size]


class TinyBackbone(nn.Module):
    """Cheap orientation-sensitive backbone with a train-mode BatchNorm."""

    output_dim = OUTPUT_DIM

    def __init__(self, *, record_inputs=False):
        super().__init__()
        self.record_inputs = bool(record_inputs)
        self.call_count = 0
        self.last_inputs = None
        self.batch_norm = nn.BatchNorm1d(16)
        self.projection = nn.Linear(16, self.output_dim)

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        self.call_count += 1
        if self.record_inputs:
            self.last_inputs = (
                image.detach().clone(),
                spectra.detach().clone(),
                fiber_positions.detach().clone(),
                None if fiber_mask is None else fiber_mask.detach().clone(),
            )

        if fiber_mask is None:
            fiber_mask = torch.ones(
                spectra.shape[0], spectra.shape[2],
                device=spectra.device,
                dtype=torch.bool,
            )
        observed = fiber_mask.to(spectra.dtype)
        spectral_values = spectra[:, 0] * observed.unsqueeze(-1)
        position_values = fiber_positions * observed.unsqueeze(-1)
        probes = torch.stack(
            (
                image[:, 0, 0, 0],
                image[:, 0, 0, -1],
                image[:, 0, -1, 0],
                image[:, 0, 1, 3],
                image[:, 0].mean(dim=(-2, -1)),
                spectral_values[:, :, 0].sum(dim=1),
                spectral_values[:, :, 1].sum(dim=1),
                spectral_values[:, 0, 2],
                spectral_values[:, 3, 3],
                spectral_values.mean(dim=(-2, -1)),
                position_values[..., 0].sum(dim=1),
                position_values[..., 1].sum(dim=1),
                position_values[:, 0, 0],
                position_values[:, 3, 1],
                (spectral_values[:, :, 0] * position_values[..., 0]).sum(dim=1),
                (spectral_values[:, :, 1] * position_values[..., 1]).sum(dim=1),
            ),
            dim=-1,
        )
        return self.projection(self.batch_norm(probes))


class ConstantBackbone(nn.Module):
    output_dim = OUTPUT_DIM

    def __init__(self):
        super().__init__()
        values = torch.linspace(-0.75, 1.25, self.output_dim)
        self.constant = nn.Parameter(values)

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        return self.constant.unsqueeze(0).expand(image.shape[0], -1)


class PermutationInvariantBackbone(nn.Module):
    """Set probe that remains sensitive to spectrum-position association."""

    output_dim = OUTPUT_DIM

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(12, self.output_dim)

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        if fiber_mask is None:
            fiber_mask = torch.ones(
                spectra.shape[0], spectra.shape[2],
                device=spectra.device,
                dtype=torch.bool,
            )
        observed = fiber_mask.to(spectra.dtype)
        spectral_values = spectra[:, 0] * observed.unsqueeze(-1)
        position_values = fiber_positions * observed.unsqueeze(-1)
        line0 = spectral_values[:, :, 0]
        line1 = spectral_values[:, :, 1]
        probes = torch.stack(
            (
                image[:, 0, 0, 0],
                image[:, 0, 1, 3],
                image[:, 0].mean(dim=(-2, -1)),
                spectral_values[..., 0].sum(dim=1),
                spectral_values[..., 1].sum(dim=1),
                spectral_values[..., 2].sum(dim=1),
                position_values[..., 0].sum(dim=1),
                position_values[..., 1].sum(dim=1),
                (line0 * position_values[..., 0]).sum(dim=1),
                (line0 * position_values[..., 1]).sum(dim=1),
                (line1 * position_values[..., 0]).sum(dim=1),
                (line1 * position_values[..., 1]).sum(dim=1),
            ),
            dim=-1,
        )
        return self.projection(probes)


def _orbit(backbone):
    return D4OrbitFeatureExtractor(
        nspec=5,
        base_backbone=backbone,
        scalar_channels=SCALAR_CHANNELS,
        spin1_channels=SPIN1_CHANNELS,
        spin2_channels=SPIN2_CHANNELS,
    )


def _act(features, element, scalar_channels=2, spin1_channels=4, spin2_channels=4):
    return transform_d4_feature_blocks(
        features,
        element,
        scalar_channels=scalar_channels,
        spin1_channels=spin1_channels,
        spin2_channels=spin2_channels,
    )


def test_feature_generators_have_authoritative_spin_actions():
    features = torch.tensor(
        [[2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]],
        dtype=torch.float64,
    )

    r90 = _act(features, "r90")
    vertical_reflection = _act(features, "v")

    torch.testing.assert_close(
        r90,
        torch.tensor(
            [[2.0, 3.0, 2.0, -1.0, 4.0, -3.0, -5.0, -6.0, -7.0, -8.0]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        vertical_reflection,
        torch.tensor(
            [[2.0, 3.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]],
            dtype=torch.float64,
        ),
    )


def test_feature_action_satisfies_d4_relations_and_all_inverses():
    generator = torch.Generator().manual_seed(812)
    features = torch.randn((3, 10), generator=generator, dtype=torch.float64)

    rotated = features
    for _ in range(4):
        rotated = _act(rotated, "r90")
    torch.testing.assert_close(rotated, features)
    torch.testing.assert_close(_act(_act(features, "v"), "v"), features)
    reflected_rotation = _act(_act(_act(features, "v"), "r90"), "v")
    torch.testing.assert_close(reflected_rotation, _act(features, "r270"))

    for element in D4_ELEMENTS:
        transformed = _act(features, element)
        restored = _act(transformed, D4_INVERSES[element])
        torch.testing.assert_close(restored, features)
        torch.testing.assert_close(
            torch.linalg.vector_norm(transformed, dim=-1),
            torch.linalg.vector_norm(features, dim=-1),
        )


def test_fiber_mask_uses_the_same_reflection_pairing_and_round_trips():
    mask = _mask()
    assert torch.equal(transform_d4_fiber_mask(mask, "r90"), mask)
    assert torch.equal(
        transform_d4_fiber_mask(mask, "v"),
        mask[:, [0, 1, 2, 4, 3]],
    )
    for element in D4_ELEMENTS:
        restored = transform_d4_fiber_mask(
            transform_d4_fiber_mask(mask, element),
            D4_INVERSES[element],
        )
        assert torch.equal(restored, mask)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_orbit_features_are_exactly_d4_equivariant(element):
    torch.manual_seed(31)
    model = _orbit(TinyBackbone()).eval()
    image, spectra, positions = _asymmetric_inputs()
    reference = model(image, spectra, positions)
    transformed_data = apply_d4_to_datavector(
        image, spectra, fp=positions, element=element
    )

    transformed_features = model(
        transformed_data[0], transformed_data[1], transformed_data[3]
    )
    expected = model.transform_features(reference, element)

    torch.testing.assert_close(transformed_features, expected, atol=2e-5, rtol=2e-5)
    mapped_back = model.transform_features(
        transformed_features, D4_INVERSES[element]
    )
    torch.testing.assert_close(mapped_back, reference, atol=2e-5, rtol=2e-5)


def test_orbit_builds_one_complete_multimodal_8b_batch_in_train_mode():
    torch.manual_seed(41)
    backbone = TinyBackbone(record_inputs=True)
    model = _orbit(backbone).train()
    image, spectra, positions = _asymmetric_inputs()
    mask = _mask()

    model(image, spectra, positions, mask)

    assert backbone.call_count == 1
    assert backbone.batch_norm.num_batches_tracked.item() == 1
    recorded_image, recorded_spectra, recorded_positions, recorded_mask = (
        backbone.last_inputs
    )
    assert recorded_image.shape[0] == len(D4_ELEMENTS) * image.shape[0]
    assert recorded_spectra.shape[0] == recorded_image.shape[0]
    assert recorded_positions.shape[0] == recorded_image.shape[0]
    assert recorded_mask.shape[0] == recorded_image.shape[0]

    # Match by the deliberately asymmetric image, without assuming whether the
    # implementation stores the orbit group-major or sample-major.
    for element in D4_ELEMENTS:
        expected = apply_d4_to_datavector(
            image, spectra, fp=positions, element=element
        )
        expected_mask = transform_d4_fiber_mask(mask, element)
        for batch_index in range(image.shape[0]):
            matches = torch.all(
                recorded_image == expected[0][batch_index], dim=(1, 2, 3)
            ).nonzero(as_tuple=False).flatten()
            assert matches.numel() == 1
            orbit_index = int(matches.item())
            torch.testing.assert_close(
                recorded_spectra[orbit_index], expected[1][batch_index]
            )
            torch.testing.assert_close(
                recorded_positions[orbit_index], expected[3][batch_index]
            )
            assert torch.equal(recorded_mask[orbit_index], expected_mask[batch_index])


@pytest.mark.parametrize("element", ("r90", "v"))
def test_train_mode_batchnorm_does_not_break_generator_equivariance(element):
    torch.manual_seed(51)
    reference_model = _orbit(TinyBackbone()).train()
    transformed_model = copy.deepcopy(reference_model).train()
    image, spectra, positions = _asymmetric_inputs()
    mask = _mask()
    transformed = apply_d4_to_datavector(
        image, spectra, fp=positions, element=element
    )
    transformed_mask = transform_d4_fiber_mask(mask, element)

    reference = reference_model(image, spectra, positions, mask)
    actual = transformed_model(
        transformed[0], transformed[1], transformed[3], transformed_mask
    )
    expected = reference_model.transform_features(reference, element)

    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_constant_backbone_cannot_create_a_fixed_spin_axis():
    model = _orbit(ConstantBackbone()).eval()
    image, spectra, positions = _asymmetric_inputs(batch_size=1)

    features = model(image, spectra, positions)

    torch.testing.assert_close(
        features[:, :SCALAR_CHANNELS],
        model.base_backbone.constant[:SCALAR_CHANNELS].unsqueeze(0),
    )
    torch.testing.assert_close(
        features[:, SCALAR_CHANNELS:],
        torch.zeros_like(features[:, SCALAR_CHANNELS:]),
        atol=1e-7,
        rtol=0.0,
    )


def test_orbit_is_differentiable_through_every_modality_and_backbone():
    torch.manual_seed(61)
    backbone = TinyBackbone()
    model = _orbit(backbone).eval()
    image, spectra, positions = _asymmetric_inputs(requires_grad=True)

    features = model(image, spectra, positions)
    weights = torch.linspace(-1.0, 1.0, features.shape[-1])
    (features * weights).sum().backward()

    for gradient in (
        image.grad,
        spectra.grad,
        positions.grad,
        backbone.batch_norm.weight.grad,
        backbone.projection.weight.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_joint_fiber_permutation_is_invariant_but_association_matters():
    torch.manual_seed(71)
    model = _orbit(PermutationInvariantBackbone()).eval()
    image, spectra, positions = _asymmetric_inputs()
    mask = _mask()
    permutation = torch.tensor([4, 2, 0, 3, 1])

    reference = model(image, spectra, positions, mask)
    joint_permutation = model(
        image,
        spectra[:, :, permutation],
        positions[:, permutation],
        mask[:, permutation],
    )
    spectra_only = model(image, spectra[:, :, permutation], positions, mask)

    torch.testing.assert_close(reference, joint_permutation, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(reference, spectra_only, atol=1e-5, rtol=1e-5)


def test_orbit_state_dict_round_trip_and_stage4_selector():
    assert "stage4_d4" in BACKBONE_TYPES
    torch.manual_seed(81)
    original = _orbit(TinyBackbone()).eval()
    torch.manual_seed(82)
    restored = _orbit(TinyBackbone()).eval()
    inputs = _asymmetric_inputs(batch_size=1)
    assert not torch.allclose(original(*inputs), restored(*inputs))

    restored.load_state_dict(copy.deepcopy(original.state_dict()), strict=True)

    torch.testing.assert_close(original(*inputs), restored(*inputs))




def test_stage4_ccl_projector_preserves_d4_actions_and_cosine_geometry():
    torch.manual_seed(86)
    projector = D4EquivariantCCLProjector(output_dim=128).eval()
    features = torch.randn(4, OUTPUT_DIM)
    projected = projector(features)
    normalized = torch.nn.functional.normalize(projected, dim=-1)
    reference_gram = normalized @ normalized.T

    for element in D4_ELEMENTS:
        transformed_features = transform_d4_feature_blocks(
            features,
            element,
            scalar_channels=SCALAR_CHANNELS,
            spin1_channels=SPIN1_CHANNELS,
            spin2_channels=SPIN2_CHANNELS,
        )
        actual = projector(transformed_features)
        expected = transform_d4_feature_blocks(
            projected,
            element,
            scalar_channels=projector.output_scalar_channels,
            spin1_channels=projector.output_spin1_channels,
            spin2_channels=projector.output_spin2_channels,
        )
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        actual_normalized = torch.nn.functional.normalize(actual, dim=-1)
        torch.testing.assert_close(
            actual_normalized @ actual_normalized.T,
            reference_gram,
            atol=2e-5,
            rtol=2e-5,
        )

    ccl_model = CCLPretrain(backbone=_orbit(TinyBackbone()))
    assert isinstance(ccl_model.projector, D4EquivariantCCLProjector)


def test_production_stage4_factory_forward_is_finite():
    torch.manual_seed(91)
    model = build_feature_extractor("stage4_d4", nspec=5).eval()
    image = torch.randn(1, 1, 48, 48)
    spectra = torch.randn(1, 1, 5, 64)
    positions = torch.tensor(
        [[[1.0, 0.2], [-1.0, -0.2], [0.0, 0.0], [-0.3, 1.0], [0.3, -1.0]]]
    )

    with torch.no_grad():
        features = model(image, spectra, positions)

    assert isinstance(model, D4OrbitFeatureExtractor)
    assert features.shape == (1, model.output_dim)
    assert torch.isfinite(features).all()
