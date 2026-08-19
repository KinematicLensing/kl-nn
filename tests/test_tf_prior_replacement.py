import math
import types

import pytest
import torch
from torch import nn

from data import TFCalculator
from networks import KLNPE


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


def _bare_npe(mode=1):
    model = KLNPE.__new__(KLNPE)
    nn.Module.__init__(model)
    model.nfeatures = len(FEATURE_NAMES)
    model.feature_names = list(FEATURE_NAMES)
    model.theta_idx = 2
    model.vcirc_idx = 5
    model.vcirc_min = 60.0
    model.vcirc_max = 540.0
    model.vcirc_dex = 0.1
    model.tf_calc = TFCalculator(slope=-7.22, intercept=36.0, scatter=0.1)
    model.mode = mode
    model.posterior_symmetry = "none"
    model.flow_type = "affine"
    return model


def _normalized_vcirc(vcirc, model):
    return 2.0 * (vcirc - model.vcirc_min) / (
        model.vcirc_max - model.vcirc_min
    ) - 1.0


def test_truncated_tf_density_is_normalized_in_physical_velocity():
    model = _bare_npe()
    velocity = torch.linspace(
        model.vcirc_min,
        model.vcirc_max,
        40_001,
        dtype=torch.float64,
    )
    magnitude = model.tf_calc.vcirc_to_mag(300.0)

    density = model.tf_prior_log_prob(
        velocity,
        magnitude,
        0.15,
    ).exp()

    integral = torch.trapezoid(density, velocity)
    torch.testing.assert_close(
        integral,
        torch.tensor(1.0, dtype=torch.float64),
        rtol=2e-6,
        atol=2e-6,
    )
    outside = model.tf_prior_log_prob(
        torch.tensor([59.0, 541.0], dtype=torch.float64),
        magnitude,
        0.15,
    )
    assert torch.isneginf(outside).all()


def test_tf_density_includes_base10_velocity_jacobian():
    model = _bare_npe()
    velocity = torch.tensor(240.0, dtype=torch.float64)
    magnitude = torch.tensor(
        model.tf_calc.vcirc_to_mag(300.0), dtype=torch.float64
    )
    magnitude_sigma = torch.tensor(0.2, dtype=torch.float64)
    slope = model.tf_calc.slope
    mean_log10 = (magnitude - model.tf_calc.intercept) / slope
    sigma_log10 = torch.sqrt(
        torch.tensor(model.vcirc_dex**2, dtype=torch.float64)
        + (magnitude_sigma / slope).square()
    )
    standard_normal = torch.distributions.Normal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(1.0, dtype=torch.float64),
    )
    lower = (math.log10(model.vcirc_min) - mean_log10) / sigma_log10
    upper = (math.log10(model.vcirc_max) - mean_log10) / sigma_log10
    truncation_mass = standard_normal.cdf(upper) - standard_normal.cdf(lower)
    standardized = (torch.log10(velocity) - mean_log10) / sigma_log10
    expected = (
        torch.exp(-0.5 * standardized.square())
        / math.sqrt(2.0 * math.pi)
        / sigma_log10
        / truncation_mass
        / velocity
        / math.log(10.0)
    )

    actual = model.tf_prior_log_prob(
        velocity,
        magnitude,
        magnitude_sigma,
    ).exp()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_prior_replacement_weights_are_per_galaxy_pi_tf_over_uniform_pi0():
    model = _bare_npe()
    candidate_count = 257
    samples = torch.zeros((2, candidate_count, model.nfeatures), dtype=torch.float64)
    samples[..., model.vcirc_idx] = torch.linspace(-1.0, 1.0, candidate_count)
    magnitudes = torch.tensor(
        [
            model.tf_calc.vcirc_to_mag(150.0),
            model.tf_calc.vcirc_to_mag(400.0),
        ],
        dtype=torch.float64,
    )
    magnitude_sigma = torch.tensor([0.1, 0.2], dtype=torch.float64)

    # The new path must never estimate/divide by the learned posterior KDE.
    def forbidden_kde(*args, **kwargs):
        raise AssertionError("prior replacement must not call the legacy KDE")

    model._kde_log_density_1d = forbidden_kde
    weights, log_ratio, diagnostics = model.tf_prior_replacement_weights(
        samples,
        magnitudes,
        magnitude_sigma,
    )
    velocity = model.vcirc_min + 0.5 * (
        samples[..., model.vcirc_idx] + 1.0
    ) * (model.vcirc_max - model.vcirc_min)
    log_tf = model.tf_prior_log_prob(
        velocity,
        magnitudes[:, None],
        magnitude_sigma[:, None],
    )

    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(
        log_ratio,
        log_tf + math.log(model.vcirc_max - model.vcirc_min),
    )
    expected_ess = weights.sum(dim=1).square() / weights.square().sum(dim=1)
    torch.testing.assert_close(diagnostics["effective_sample_size"], expected_ess)
    torch.testing.assert_close(
        diagnostics["effective_sample_fraction"],
        expected_ess / candidate_count,
    )
    torch.testing.assert_close(
        diagnostics["max_normalized_weight"], weights.max(dim=1).values
    )
    peak_velocity = velocity.gather(1, weights.argmax(dim=1, keepdim=True)).squeeze(1)
    assert peak_velocity[0] < peak_velocity[1]


def test_prior_replacement_resamples_complete_joint_rows():
    model = _bare_npe()
    candidate_count = 96
    samples = torch.empty((2, candidate_count, model.nfeatures), dtype=torch.float64)
    for galaxy in range(2):
        for candidate in range(candidate_count):
            samples[galaxy, candidate] = torch.arange(
                model.nfeatures, dtype=torch.float64
            ) + 1000 * galaxy + 10 * candidate
    velocity = torch.linspace(70.0, 530.0, candidate_count, dtype=torch.float64)
    samples[..., model.vcirc_idx] = _normalized_vcirc(velocity, model)
    magnitudes = torch.tensor(
        [model.tf_calc.vcirc_to_mag(160.0), model.tf_calc.vcirc_to_mag(390.0)],
        dtype=torch.float64,
    )
    magnitude_sigma = torch.tensor([0.12, 0.12], dtype=torch.float64)
    _, source_log_ratio, _ = model.tf_prior_replacement_weights(
        samples, magnitudes, magnitude_sigma
    )

    torch.manual_seed(1729)
    resampled, selected_log_ratio = model._apply_tf_prior_replacement(
        samples,
        magnitudes,
        magnitude_sigma,
    )

    assert resampled.shape == samples.shape
    assert selected_log_ratio.shape == samples.shape[:2]
    for galaxy in range(samples.shape[0]):
        for draw in range(candidate_count):
            matches = torch.all(
                samples[galaxy] == resampled[galaxy, draw], dim=1
            )
            assert int(matches.sum()) == 1
            source_index = int(torch.nonzero(matches)[0])
            torch.testing.assert_close(
                selected_log_ratio[galaxy, draw],
                source_log_ratio[galaxy, source_index],
            )
    assert hasattr(model, "last_tf_inference_diagnostics")


class _ZeroFeatureExtractor(nn.Module):
    def forward(self, image, spectra, fiber_positions):
        return torch.zeros(
            (image.shape[0], 1024), device=image.device, dtype=image.dtype
        )


class _CandidateFlow(nn.Module):
    def __init__(self, candidates):
        super().__init__()
        self.register_buffer("candidates", candidates)

    def sample(self, num_samples, context):
        assert context.shape[0] == 1
        assert num_samples == self.candidates.shape[1]
        return self.candidates.clone()

    def log_prob(self, samples, context):
        return -0.25 * samples.square().sum(dim=-1)


def _sampleable_npe(mode=1, candidate_count=64):
    model = _bare_npe(mode=mode)
    candidates = torch.zeros((1, candidate_count, model.nfeatures))
    candidates[0, :, 0] = torch.linspace(-0.8, 0.8, candidate_count)
    candidates[0, :, 1] = 0.5 * candidates[0, :, 0]
    candidates[0, :, model.vcirc_idx] = torch.linspace(-0.9, 0.9, candidate_count)
    model.feature_extractor = _ZeroFeatureExtractor()
    model.layer_norm = nn.Identity()
    model.flow = _CandidateFlow(candidates)
    return model, candidates


def _dummy_observation():
    return (
        torch.zeros((1, 1, 8, 8)),
        torch.zeros((1, 1, 5, 16)),
        torch.zeros((1, 5, 2)),
    )


def test_mode1_sample_selector_returns_tf_corrected_scores():
    model, raw_candidates = _sampleable_npe(mode=1)
    image, spectra, fiber_positions = _dummy_observation()
    magnitude = model.tf_calc.vcirc_to_mag(280.0)
    magnitude_sigma = 0.15

    raw_samples, raw_scores = model.sample(
        image,
        spectra,
        raw_candidates.shape[1],
        fiber_positions,
        return_log_prob=True,
    )
    torch.testing.assert_close(raw_samples, raw_candidates)
    torch.testing.assert_close(
        raw_scores,
        model.flow.log_prob(
            raw_candidates[0],
            torch.zeros((raw_candidates.shape[1], 1024)),
        ),
    )

    torch.manual_seed(55)
    samples, scores = model.sample(
        image,
        spectra,
        raw_candidates.shape[1],
        fiber_positions,
        mag=magnitude,
        mag_sigma=magnitude_sigma,
        tf_inference="prior_replacement",
        return_log_prob=True,
    )
    velocity = model.vcirc_min + 0.5 * (
        samples[0, :, model.vcirc_idx] + 1.0
    ) * (model.vcirc_max - model.vcirc_min)
    expected = model.flow.log_prob(
        samples[0], torch.zeros((samples.shape[1], 1024))
    ) + model.tf_prior_log_prob(
        velocity, magnitude, magnitude_sigma
    ) + math.log(model.vcirc_max - model.vcirc_min)

    torch.testing.assert_close(scores, expected)
    for row in samples[0]:
        assert torch.any(torch.all(raw_candidates[0] == row, dim=1))


def test_prior_replacement_requires_mode1_and_magnitude_uncertainty():
    image, spectra, fiber_positions = _dummy_observation()
    mode1, candidates = _sampleable_npe(mode=1)
    with pytest.raises(ValueError, match="both mag and mag_sigma"):
        mode1.sample(
            image,
            spectra,
            candidates.shape[1],
            fiber_positions,
            mag=20.0,
            tf_inference="prior_replacement",
        )

    mode2, candidates = _sampleable_npe(mode=2)
    with pytest.raises(ValueError, match=r"mode-?1"):
        mode2.sample(
            image,
            spectra,
            candidates.shape[1],
            fiber_positions,
            mag=20.0,
            mag_sigma=0.1,
            tf_inference="prior_replacement",
        )


def test_legacy_mode2_sampling_route_remains_available():
    model, raw_candidates = _sampleable_npe(mode=2)
    image, spectra, fiber_positions = _dummy_observation()
    calls = []

    def fake_legacy_resampling(self, samples, mag, snr):
        calls.append((mag, snr))
        return samples.flip(1), torch.full(
            (samples.shape[1],), 0.25, dtype=samples.dtype
        )

    model._apply_tf_resampling = types.MethodType(fake_legacy_resampling, model)
    samples, scores = model.sample(
        image,
        spectra,
        raw_candidates.shape[1],
        fiber_positions,
        mag=19.0,
        snr=25.0,
        return_log_prob=True,
    )

    torch.testing.assert_close(samples, raw_candidates.flip(1))
    expected = model.flow.log_prob(
        samples[0], torch.zeros((samples.shape[1], 1024))
    ) + 0.25
    torch.testing.assert_close(scores, expected)
    assert calls == [(19.0, 25.0)]
