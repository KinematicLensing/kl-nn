import itertools
import math

import pytest
import torch
from torch import nn

from data import (
    D4_ELEMENTS,
    D4_INVERSES,
    TFCalculator,
    apply_d4_to_datavector,
    transform_d4_parameters,
)
from networks import D4OrbitFeatureExtractor, KLNPE
from train import sample_density


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
NFEATURES = len(FEATURE_NAMES)
FEATURE_DIM = 1024


class ProbeBackbone(nn.Module):
    """Small asymmetric multimodal map for the exact orbit wrapper."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(81226)
        self.projection = nn.Parameter(
            torch.randn(12, FEATURE_DIM, generator=generator) / math.sqrt(12)
        )

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del fiber_mask
        spectral = spectra[:, 0]
        line0 = spectral[..., 0]
        line1 = spectral[..., 1]
        probes = torch.stack(
            (
                image[:, 0, 0, 1],
                image[:, 0, 2, 4],
                image[:, 0].mean(dim=(-2, -1)),
                spectral[..., 0].sum(dim=1),
                spectral[..., 1].sum(dim=1),
                spectral[..., 2].sum(dim=1),
                fiber_positions[..., 0].sum(dim=1),
                fiber_positions[..., 1].sum(dim=1),
                (line0 * fiber_positions[..., 0]).sum(dim=1),
                (line0 * fiber_positions[..., 1]).sum(dim=1),
                (line1 * fiber_positions[..., 0]).sum(dim=1),
                (line1 * fiber_positions[..., 1]).sum(dim=1),
            ),
            dim=-1,
        )
        return probes @ self.projection


class RecordingOrbitExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.orbit = D4OrbitFeatureExtractor(
            nspec=5,
            base_backbone=ProbeBackbone(),
        )
        self.calls = 0

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        self.calls += 1
        return self.orbit(image, spectra, fiber_positions, fiber_mask)

    def transform_features(self, features, element):
        return self.orbit.transform_features(features, element)


class RecordingGaussianFlow(nn.Module):
    """A deliberately coordinate-dependent conditional density."""

    def __init__(self):
        super().__init__()
        self.context_scale = nn.Parameter(
            torch.linspace(0.05, 0.35, NFEATURES)
        )
        self.input_scale = nn.Parameter(
            torch.linspace(0.7, 1.4, NFEATURES)
        )
        self.log_prob_calls = 0
        self.sample_calls = 0
        self.last_log_prob_shapes = None

    def score(self, inputs, context):
        center = context[:, :NFEATURES] * self.context_scale
        residual = (inputs - center) * self.input_scale
        return -residual.square().sum(dim=-1) + 0.01 * context[:, 17]

    def log_prob(self, inputs, context):
        self.log_prob_calls += 1
        self.last_log_prob_shapes = (tuple(inputs.shape), tuple(context.shape))
        return self.score(inputs, context)

    def sample(self, num_samples, context):
        self.sample_calls += 1
        center = context[:, :NFEATURES] * self.context_scale
        return center[:, None, :].expand(-1, num_samples, -1).clone()


class PresetBranchFlow(RecordingGaussianFlow):
    """Return either the same vector or a covariant vector in each branch."""

    def __init__(self, template, *, covariant):
        super().__init__()
        self.register_buffer("template", torch.as_tensor(template).float())
        self.covariant = bool(covariant)

    def sample(self, num_samples, context):
        self.sample_calls += 1
        if context.shape[0] != len(D4_ELEMENTS):
            raise AssertionError("sampling test expects one galaxy and eight branches")
        rows = []
        for element in D4_ELEMENTS:
            value = (
                transform_d4_parameters(
                    self.template.unsqueeze(0),
                    element,
                    feature_names=FEATURE_NAMES,
                )[0]
                if self.covariant
                else self.template
            )
            rows.append(value.expand(num_samples, -1))
        return torch.stack(rows, dim=0)


class SequenceFlow(nn.Module):
    def __init__(self, branch_scores):
        super().__init__()
        scores = torch.as_tensor(branch_scores, dtype=torch.float32)
        if scores.ndim != 2 or scores.shape[1] != len(D4_ELEMENTS):
            raise ValueError("branch_scores must have shape (batch, 8)")
        self.register_buffer("branch_scores", scores)
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.log_prob_calls = 0

    def log_prob(self, inputs, context):
        del context
        self.log_prob_calls += 1
        expected = self.branch_scores.T.reshape(-1)
        if inputs.shape[0] != expected.numel():
            raise AssertionError((inputs.shape, expected.shape))
        return expected.to(inputs) + self.anchor * inputs.sum(dim=-1)


def _model(feature_extractor=None, flow=None, *, mode=1):
    """Construct only the parts of KLNPE exercised by these unit tests."""
    model = KLNPE.__new__(KLNPE)
    nn.Module.__init__(model)
    model.bs = 2
    model.nfeatures = NFEATURES
    model.nspecs = 5
    model.mode = mode
    model.posterior_symmetry = "d4"
    model.feature_names = tuple(FEATURE_NAMES)
    model.feature_extractor = feature_extractor or RecordingOrbitExtractor()
    model.layer_norm = nn.LayerNorm(FEATURE_DIM)
    model.flow = flow or RecordingGaussianFlow()
    model.vcirc_idx = FEATURE_NAMES.index("vcirc")
    model.vcirc_dex = 0.1
    model.vcirc_log_scale = 0.1 * math.log(10.0)
    model.vcirc_min = 60.0
    model.vcirc_max = 540.0
    model.vcirc_jac = 0.5 * (model.vcirc_max - model.vcirc_min)
    model.tf_calc = TFCalculator(slope=-7.22, intercept=36.0)
    return model


def _datavector(batch_size=2):
    generator = torch.Generator().manual_seed(441)
    image = torch.randn(batch_size, 1, 7, 7, generator=generator)
    spectra = torch.randn(batch_size, 1, 5, 6, generator=generator)
    positions = torch.tensor(
        [
            [
                [1.0, 0.2],
                [-1.0, -0.2],
                [0.0, 0.0],
                [-0.3, 1.1],
                [0.3, -1.1],
            ]
        ],
        dtype=torch.float32,
    ).expand(batch_size, -1, -1).clone()
    targets = torch.tensor(
        [
            [0.13, -0.07, 0.31, -0.4, 0.2, -0.1, 0.5, -0.6],
            [-0.09, 0.16, -0.62, 0.3, -0.5, 0.4, -0.2, 0.7],
        ],
        dtype=torch.float32,
    )[:batch_size]
    return image, spectra, targets, positions


def test_parameter_action_handles_shuffled_features_and_arbitrary_leading_dims():
    names = ["vcirc", "g2", "theta_int", "hlr", "g1", "sini", "v0", "rscale"]
    parameters = torch.tensor(
        [[[0.2, -0.3, 0.75, 0.1, 0.4, -0.2, 0.6, -0.5]]],
        dtype=torch.float64,
    ).expand(2, 3, -1)

    r90 = transform_d4_parameters(parameters, "r90", feature_names=names)
    reflection = transform_d4_parameters(parameters, "v", feature_names=names)

    expected_r90 = parameters.clone()
    expected_r90[..., 1] = 0.3
    expected_r90[..., 4] = -0.4
    expected_r90[..., 2] = 0.25
    expected_reflection = parameters.clone()
    expected_reflection[..., 1] = 0.3
    expected_reflection[..., 2] = -0.75
    torch.testing.assert_close(r90, expected_r90)
    torch.testing.assert_close(reflection, expected_reflection)


def test_parameter_action_satisfies_d4_inverses_and_closure():
    _, _, targets, _ = _datavector()
    orbit = [transform_d4_parameters(targets, g) for g in D4_ELEMENTS]
    for element in D4_ELEMENTS:
        transformed = transform_d4_parameters(targets, element)
        restored = transform_d4_parameters(transformed, D4_INVERSES[element])
        torch.testing.assert_close(restored, targets)
        assert torch.all((-1.0 <= transformed[:, 2]) & (transformed[:, 2] < 1.0))
    for first, second in itertools.product(D4_ELEMENTS, repeat=2):
        composed = transform_d4_parameters(
            transform_d4_parameters(targets, first), second
        )
        assert any(torch.allclose(composed, member) for member in orbit)


def test_parameter_only_action_matches_complete_datavector_action():
    image, spectra, targets, positions = _datavector()
    for element in D4_ELEMENTS:
        complete = apply_d4_to_datavector(
            image, spectra, targets, positions, element=element
        )[2]
        parameter_only = transform_d4_parameters(targets, element)
        torch.testing.assert_close(parameter_only, complete)


def test_contexts_transform_raw_features_before_layer_norm():
    model = _model()
    features = torch.linspace(-3.0, 5.0, 2 * FEATURE_DIM).reshape(2, FEATURE_DIM)

    actual = model._d4_contexts_from_features(features)
    expected = torch.stack(
        [
            model.layer_norm(model.feature_extractor.transform_features(features, g))
            for g in D4_ELEMENTS
        ]
    )
    wrong_shortcut = torch.stack(
        [
            model.feature_extractor.transform_features(model.layer_norm(features), g)
            for g in D4_ELEMENTS
        ]
    )

    assert actual.shape == (len(D4_ELEMENTS), 2, FEATURE_DIM)
    torch.testing.assert_close(actual, expected)
    assert torch.max(torch.abs(actual - wrong_shortcut)) > 1e-2


def test_training_uses_mean_branch_log_prob_but_density_uses_logsumexp():
    ell = torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]])
    image, spectra, targets, positions = _datavector(batch_size=1)
    flow = SequenceFlow(ell)
    model = _model(flow=flow, mode=1)
    features = model.feature_extractor(image, spectra, positions)

    actual_branches = model._d4_branch_log_prob_from_features(features, targets)
    torch.testing.assert_close(actual_branches, ell)
    training_loss = model(image, spectra, targets, positions)
    density = model.posterior_log_prob(image, spectra, targets, positions)

    expected_density = torch.logsumexp(ell, dim=1) - math.log(len(D4_ELEMENTS))
    torch.testing.assert_close(training_loss, -ell.mean())
    torch.testing.assert_close(density, expected_density)
    assert not torch.allclose(-training_loss.unsqueeze(0), density)


@pytest.mark.parametrize("element", D4_ELEMENTS)
def test_symmetrized_posterior_is_exactly_d4_equivariant(element):
    image, spectra, targets, positions = _datavector()
    model = _model().eval()
    reference = model.posterior_log_prob(image, spectra, targets, positions)
    transformed = apply_d4_to_datavector(
        image, spectra, targets, positions, element=element
    )

    actual = model.posterior_log_prob(
        transformed[0], transformed[1], transformed[2], transformed[3]
    )

    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=3e-5)


def test_posterior_vectorizes_branches_and_backpropagates_everywhere():
    image, spectra, targets, positions = _datavector()
    targets.requires_grad_()
    extractor = RecordingOrbitExtractor()
    flow = RecordingGaussianFlow()
    model = _model(feature_extractor=extractor, flow=flow)

    log_prob = model.posterior_log_prob(image, spectra, targets, positions)
    log_prob.sum().backward()

    assert extractor.calls == 1
    assert flow.log_prob_calls == 1
    assert flow.last_log_prob_shapes == (
        (len(D4_ELEMENTS) * image.shape[0], NFEATURES),
        (len(D4_ELEMENTS) * image.shape[0], FEATURE_DIM),
    )
    for gradient in (
        targets.grad,
        extractor.orbit.base_backbone.projection.grad,
        flow.context_scale.grad,
        flow.input_scale.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_logsumexp_density_is_stable_for_very_negative_branch_scores():
    ell = -10000.0 - torch.arange(8, dtype=torch.float32).unsqueeze(0)
    image, spectra, targets, positions = _datavector(batch_size=1)
    model = _model(flow=SequenceFlow(ell))

    actual = model.posterior_log_prob(image, spectra, targets, positions)
    expected = torch.logsumexp(ell.double(), dim=1) - math.log(8)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.double(), expected, atol=5e-4, rtol=0.0)


def test_balanced_sampling_cancels_a_fixed_detector_frame_shear():
    template = torch.tensor([0.24, -0.11, 0.2, -0.4, 0.3, -0.2, 0.1, 0.7])
    model = _model(flow=PresetBranchFlow(template, covariant=False))
    image, spectra, _, positions = _datavector(batch_size=1)

    samples = model.sample(image, spectra, 16, positions)

    assert samples.shape == (1, 16, NFEATURES)
    torch.testing.assert_close(
        samples[0, :, :2].mean(dim=0),
        torch.zeros(2),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(samples[..., 5], template[5].expand(1, 16))


def test_balanced_sampling_preserves_a_covariant_nonzero_signal():
    template = torch.tensor([0.14, -0.08, 0.37, -0.4, 0.3, -0.2, 0.1, 0.7])
    flow = PresetBranchFlow(template, covariant=True)
    model = _model(flow=flow)
    image, spectra, _, positions = _datavector(batch_size=1)

    samples = model.sample(image, spectra, 24, positions)

    assert flow.sample_calls == 1
    torch.testing.assert_close(
        samples,
        template.expand_as(samples),
        atol=1e-6,
        rtol=0.0,
    )


def test_sampling_rejects_an_unbalanced_orbit_sample_count():
    model = _model()
    image, spectra, _, positions = _datavector(batch_size=1)

    with pytest.raises(ValueError, match="divisible by 8"):
        model.sample(image, spectra, 10, positions)


def test_returned_sample_scores_are_full_mixture_log_probabilities():
    template = torch.tensor([0.12, -0.09, 0.27, -0.4, 0.3, -0.2, 0.1, 0.7])
    flow = PresetBranchFlow(template, covariant=True)
    model = _model(flow=flow)
    image, spectra, _, positions = _datavector(batch_size=1)

    samples, log_prob = model.sample(
        image, spectra, 16, positions, return_log_prob=True
    )
    features = model.feature_extractor(image, spectra, positions)
    contexts = model._d4_contexts_from_features(features)
    expected = []
    for sample in samples[0]:
        orbit = torch.stack(
            [transform_d4_parameters(sample.unsqueeze(0), g)[0] for g in D4_ELEMENTS]
        )
        branch_scores = flow.score(orbit, contexts[:, 0])
        expected.append(torch.logsumexp(branch_scores, dim=0) - math.log(8))

    assert log_prob.shape == (16,)
    torch.testing.assert_close(log_prob, torch.stack(expected))


def test_posterior_mean_is_d4_equivariant_and_circular_at_theta_seam():
    model = _model()
    samples = torch.tensor(
        [[
            [0.12, -0.07, 0.98, -0.4, 0.3, -0.2, 0.1, 0.7],
            [0.08, -0.03, -0.98, -0.2, 0.1, 0.2, -0.1, 0.5],
        ]],
        dtype=torch.float32,
    )
    reference = model.posterior_mean(samples)
    assert abs(abs(reference[0, 2].item()) - 1.0) < 1e-5

    for element in D4_ELEMENTS:
        transformed_samples = transform_d4_parameters(samples, element)
        actual = model.posterior_mean(transformed_samples)
        expected = transform_d4_parameters(reference, element)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0.0)


def test_tf_training_weight_is_applied_after_each_galaxys_group_mean():
    ell = torch.tensor(
        [
            [0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
            [-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0],
        ]
    )
    image, spectra, targets, positions = _datavector()
    mag = torch.tensor([20.3, 22.1])
    snr = torch.tensor([80.0, 17.0])
    model = _model(flow=SequenceFlow(ell), mode=2)

    weights = model._compute_tf_weights(targets, mag, snr)
    actual = model(image, spectra, targets, positions, mag=mag, snr=snr)

    expected = -(weights * ell.mean(dim=1)).mean()
    torch.testing.assert_close(actual, expected)


def test_tf_weights_are_identical_on_every_d4_branch():
    model = _model(mode=2)
    _, _, targets, _ = _datavector()
    mag = torch.tensor([20.3, 22.1])
    snr = torch.tensor([80.0, 17.0])
    reference = model._compute_tf_weights(targets, mag, snr)

    for element in D4_ELEMENTS:
        transformed = transform_d4_parameters(targets, element)
        actual = model._compute_tf_weights(transformed, mag, snr)
        torch.testing.assert_close(actual, reference)


def test_legacy_rot90_cancellation_is_rejected_for_d4_posterior():
    class D4Model:
        posterior_symmetry = "d4"

    with pytest.raises(ValueError, match="redundant.*D4"):
        sample_density(
            D4Model(),
            [],
            8,
            apply_add_noise_cancellation=True,
        )


def test_theta_action_canonicalizes_but_is_not_injective_off_circle_support():
    first = torch.zeros(1, NFEATURES)
    second = first.clone()
    first[:, 2] = 0.2
    second[:, 2] = 2.2

    canonical_first = transform_d4_parameters(first, "e")
    canonical_second = transform_d4_parameters(second, "e")

    torch.testing.assert_close(canonical_first, canonical_second, atol=1e-6, rtol=0.0)
    assert -1.0 <= canonical_second[0, 2] < 1.0

