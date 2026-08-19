import copy
import math

import pytest
import torch
from torch import nn

import config
from data import D4_ELEMENTS
from networks import DEFAULT_OBSERVATION_CONTEXT_FIELDS, KLNPE


FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
NFEATURES = len(FEATURE_NAMES)
VISUAL_FEATURES = 1024
OBSERVATION_FEATURES = len(DEFAULT_OBSERVATION_CONTEXT_FIELDS)


class _TinyEquivariantExtractor(nn.Module):
    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        scalar = image.mean(dim=tuple(range(1, image.ndim)))[:, None]
        return scalar.expand(-1, VISUAL_FEATURES)

    def transform_features(self, features, element):
        branch = D4_ELEMENTS.index(element)
        return features + branch


class _RecordingFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_prob_contexts = []
        self.sample_contexts = []

    def log_prob(self, inputs, context):
        self.log_prob_contexts.append(context.detach().clone())
        return inputs.sum(dim=-1) * 0.0 + context.sum(dim=-1) * 0.0

    def sample(self, num_samples, context):
        self.sample_contexts.append(context.detach().clone())
        return context.new_zeros(context.shape[0], num_samples, NFEATURES)


def _bare_model(*, version=2, posterior_symmetry="none"):
    model = KLNPE.__new__(KLNPE)
    nn.Module.__init__(model)
    model.mode = 1
    model.nfeatures = NFEATURES
    model.feature_names = FEATURE_NAMES
    model.theta_idx = FEATURE_NAMES.index("theta_int")
    model.posterior_symmetry = posterior_symmetry
    model.flow_type = "affine"
    model.observation_model_version = version
    model.observation_context_fields = (
        DEFAULT_OBSERVATION_CONTEXT_FIELDS if version == 2 else ()
    )
    model.observation_context_features = OBSERVATION_FEATURES if version == 2 else 0
    model.flow_context_features = VISUAL_FEATURES + model.observation_context_features
    model.observation_rmag_midpoint = 19.2
    model.observation_rmag_half_range = 4.2
    model.observation_quality_log_midpoint = 0.5 * (
        math.log(3.0) + math.log(100.0)
    )
    model.observation_quality_log_half_range = 0.5 * math.log(100.0 / 3.0)
    if version == 2:
        model.register_buffer("image_noise_sigma", torch.tensor(0.125))
        model.register_buffer("spectral_reference_line_norm", torch.tensor(2.5))
    model.feature_extractor = _TinyEquivariantExtractor()
    model.layer_norm = nn.Identity()
    model.flow = _RecordingFlow()
    return model


def _observed_context(batch_size=2):
    return {
        "rmag_obs": torch.linspace(18.4, 20.1, batch_size),
        "rmag_sigma": torch.linspace(0.08, 0.16, batch_size),
        "image_snr": torch.linspace(8.0, 40.0, batch_size),
        "spectral_reference_quality": torch.linspace(5.0, 50.0, batch_size),
        "spectral_noise_scale": torch.linspace(0.2, 0.8, batch_size),
    }


def _dummy_datavector(batch_size=2):
    return (
        torch.zeros((batch_size, 1, 6, 6)),
        torch.zeros((batch_size, 1, 5, 12)),
        torch.zeros((batch_size, NFEATURES)),
        torch.zeros((batch_size, 5, 2)),
    )


def test_v2_mapping_and_tensor_contexts_follow_archived_field_order():
    model = _bare_model(version=2)
    raw = torch.zeros((2, VISUAL_FEATURES))
    observed = _observed_context()

    mapping_result = model._prepare_observation_context(observed, 2, raw)
    tensor = torch.stack(
        [observed[name] for name in DEFAULT_OBSERVATION_CONTEXT_FIELDS], dim=-1
    )
    tensor_result = model._prepare_observation_context(tensor, 2, raw)

    five_sigma_mag_error = (2.5 / math.log(10.0)) / 5.0
    expected = tensor.clone()
    expected[:, 0] = (tensor[:, 0] - 19.2) / 4.2
    expected[:, 1] = torch.log10(tensor[:, 1] / five_sigma_mag_error)
    expected[:, 2] = torch.log10(tensor[:, 2] / 5.0)
    expected[:, 3] = (
        torch.log(tensor[:, 3]) - model.observation_quality_log_midpoint
    ) / model.observation_quality_log_half_range
    expected[:, 4] = torch.log10(tensor[:, 4] / 2.5)

    torch.testing.assert_close(mapping_result, tensor_result)
    torch.testing.assert_close(mapping_result, expected)


@pytest.mark.parametrize(
    "latent_name",
    ("rmag_true", "halpha_flux_true"),
)
def test_v2_context_is_required_and_latent_simulator_metadata_is_rejected(
    latent_name,
):
    model = _bare_model(version=2)
    raw = torch.zeros((2, VISUAL_FEATURES))
    with pytest.raises(ValueError, match="required.*model_version=2"):
        model._prepare_observation_context(None, 2, raw)

    leaked = _observed_context()
    leaked[latent_name] = torch.tensor([18.5, 20.0])
    with pytest.raises(
        ValueError,
        match=rf"must not contain latent.*{latent_name}",
    ):
        model._prepare_observation_context(leaked, 2, raw)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda context: context.pop("image_snr"), "missing=.*image_snr"),
        (
            lambda context: context.__setitem__("unexpected", torch.ones(2)),
            "extra=.*unexpected",
        ),
        (
            lambda context: context.__setitem__(
                "spectral_noise_scale", torch.tensor([0.2, 0.0])
            ),
            "must all be positive",
        ),
        (
            lambda context: context.__setitem__(
                "rmag_obs", torch.tensor([19.0, torch.nan])
            ),
            "only finite values",
        ),
    ],
)
def test_invalid_v2_context_mapping_is_rejected(mutator, message):
    model = _bare_model(version=2)
    context = _observed_context()
    mutator(context)

    with pytest.raises(ValueError, match=message):
        model._prepare_observation_context(
            context, 2, torch.zeros((2, VISUAL_FEATURES))
        )


def test_legacy_context_stays_1024_wide_and_rejects_new_metadata():
    model = _bare_model(version=1)
    raw = torch.randn((2, VISUAL_FEATURES))

    prepared = model._prepare_observation_context(None, 2, raw)
    flow_context = model._flow_context_from_features(raw, prepared)

    assert prepared is None
    assert flow_context.shape == (2, VISUAL_FEATURES)
    torch.testing.assert_close(flow_context, raw)
    with pytest.raises(ValueError, match="unavailable.*model_version=1"):
        model._prepare_observation_context(_observed_context(), 2, raw)


def test_v2_observed_scalars_are_identical_on_all_d4_branches():
    model = _bare_model(version=2, posterior_symmetry="d4")
    raw = torch.arange(2 * VISUAL_FEATURES, dtype=torch.float32).reshape(
        2, VISUAL_FEATURES
    )
    prepared = model._prepare_observation_context(
        _observed_context(), 2, raw
    )

    contexts = model._d4_contexts_from_features(raw, prepared)

    assert contexts.shape == (
        len(D4_ELEMENTS),
        2,
        VISUAL_FEATURES + OBSERVATION_FEATURES,
    )
    torch.testing.assert_close(
        contexts[..., -OBSERVATION_FEATURES:],
        prepared.unsqueeze(0).expand(len(D4_ELEMENTS), -1, -1),
    )
    assert not torch.equal(contexts[0, :, :-OBSERVATION_FEATURES], contexts[1, :, :-OBSERVATION_FEATURES])


def test_public_forward_posterior_and_sample_propagate_v2_context():
    model = _bare_model(version=2, posterior_symmetry="none")
    image, spectra, targets, positions = _dummy_datavector(batch_size=2)
    observed = _observed_context(batch_size=2)
    expected = model._prepare_observation_context(observed, 2, image)

    loss = model(
        image,
        spectra,
        targets,
        positions,
        observation_context=observed,
    )
    assert torch.isfinite(loss)
    torch.testing.assert_close(
        model.flow.log_prob_contexts[-1][:, -OBSERVATION_FEATURES:], expected
    )

    density = model.posterior_log_prob(
        image,
        spectra,
        targets,
        positions,
        observation_context=observed,
    )
    assert density.shape == (2,)
    torch.testing.assert_close(
        model.flow.log_prob_contexts[-1][:, -OBSERVATION_FEATURES:], expected
    )

    one_observed = {name: value[:1] for name, value in observed.items()}
    samples = model.sample(
        image[:1],
        spectra[:1],
        6,
        positions[:1],
        observation_context=one_observed,
    )
    assert samples.shape == (1, 6, NFEATURES)
    torch.testing.assert_close(
        model.flow.sample_contexts[-1][:, -OBSERVATION_FEATURES:], expected[:1]
    )


def test_public_d4_paths_keep_observed_scalars_invariant():
    model = _bare_model(version=2, posterior_symmetry="d4")
    image, spectra, targets, positions = _dummy_datavector(batch_size=1)
    observed = _observed_context(batch_size=1)
    prepared = model._prepare_observation_context(observed, 1, image)
    expected = prepared.expand(len(D4_ELEMENTS), -1)

    model(
        image,
        spectra,
        targets,
        positions,
        observation_context=observed,
    )
    torch.testing.assert_close(
        model.flow.log_prob_contexts[-1][:, -OBSERVATION_FEATURES:], expected
    )

    model.posterior_log_prob(
        image,
        spectra,
        targets,
        positions,
        observation_context=observed,
    )
    torch.testing.assert_close(
        model.flow.log_prob_contexts[-1][:, -OBSERVATION_FEATURES:], expected
    )

    samples = model.sample(
        image,
        spectra,
        len(D4_ELEMENTS),
        positions,
        observation_context=observed,
    )
    assert samples.shape == (1, len(D4_ELEMENTS), NFEATURES)
    torch.testing.assert_close(
        model.flow.sample_contexts[-1][:, -OBSERVATION_FEATURES:], expected
    )


def test_initialized_v1_and_v2_models_have_versioned_context_state():
    original = copy.deepcopy(config.MODEL_CONFIG)
    models = []
    try:
        for version in (1, 2):
            configured = copy.deepcopy(original)
            configured.observation.model_version = version
            configured.flow.flow_type = "affine"
            configured.flow.num_layers = 1
            configured.train.feature_number = NFEATURES
            configured.train.feature_names = list(FEATURE_NAMES)
            configured.train.posterior_symmetry = "none"
            config.set_model_config(configured)
            models.append(
                KLNPE(
                    feature_extractor=_TinyEquivariantExtractor(),
                    mode=1,
                    batch_size=2,
                    nfeatures=NFEATURES,
                    nspec=5,
                    posterior_symmetry="none",
                )
            )
    finally:
        config.set_model_config(original)

    legacy, current = models
    assert legacy.flow_context_features == VISUAL_FEATURES
    assert current.flow_context_features == VISUAL_FEATURES + OBSERVATION_FEATURES
    assert "image_noise_sigma" not in legacy.state_dict()
    assert "spectral_reference_line_norm" not in legacy.state_dict()
    assert "image_noise_sigma" in current.state_dict()
    assert "spectral_reference_line_norm" in current.state_dict()
