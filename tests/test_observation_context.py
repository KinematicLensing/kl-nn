import pytest
import torch
from torch import nn

import config
from networks import FEATURE_DIM, KLNPE, ORACLE_CONTEXT_FIELDS, OracleContextNormalizer


TARGET_COUNT = len(config.TARGET_NAMES)


class _TinyFeatureExtractor(nn.Module):
    output_dim = FEATURE_DIM

    def __init__(self):
        super().__init__()
        self.observation_contexts = []

    def forward(
        self,
        image,
        spectra,
        fiber_positions,
        observation_context,
        fiber_mask=None,
    ):
        del spectra, fiber_positions, fiber_mask
        self.observation_contexts.append(observation_context)
        scalar = image.mean(dim=tuple(range(1, image.ndim)), keepdim=False)
        return scalar[:, None].expand(-1, FEATURE_DIM)


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
        return context.new_zeros(context.shape[0], num_samples, TARGET_COUNT)


def _oracle_context(batch_size=3):
    return {
        "rmag_true": torch.linspace(15.0, 23.4, batch_size),
        "image_snr": torch.linspace(5.0, 1000.0, batch_size),
        "central_halpha_snr": torch.linspace(1.0, 200.0, batch_size),
    }


def _dummy_datavector(batch_size=2):
    return (
        torch.arange(batch_size * 16, dtype=torch.float32).reshape(
            batch_size, 1, 4, 4
        ),
        torch.zeros((batch_size, 5, 12)),
        torch.zeros((batch_size, TARGET_COUNT)),
        torch.zeros((batch_size, 5, 2)),
    )


def test_oracle_context_schema_is_exact_and_independent():
    assert ORACLE_CONTEXT_FIELDS == (
        "rmag_true",
        "image_snr",
        "central_halpha_snr",
    )
    assert tuple(config.ORACLE_CONTEXT_FIELDS) == ORACLE_CONTEXT_FIELDS

    with pytest.raises(ValueError, match="exactly the independent oracle"):
        OracleContextNormalizer(context_fields=reversed(ORACLE_CONTEXT_FIELDS))
    with pytest.raises(ValueError, match="exactly the independent oracle"):
        OracleContextNormalizer(
            context_fields=(*ORACLE_CONTEXT_FIELDS, "halpha_flux_true")
        )


def test_mapping_and_tensor_contexts_share_field_order_and_normalization():
    normalizer = OracleContextNormalizer()
    reference = torch.zeros((3, FEATURE_DIM), dtype=torch.float64)
    mapping = {
        "rmag_true": torch.tensor([15.0, 19.2, 23.4]),
        "image_snr": torch.tensor([5.0, 502.5, 1000.0]),
        "central_halpha_snr": torch.tensor([1.0, 100.5, 200.0]),
    }
    tensor = torch.stack([mapping[name] for name in ORACLE_CONTEXT_FIELDS], dim=-1)

    from_mapping = normalizer(mapping, 3, reference)
    from_tensor = normalizer(tensor, 3, reference)
    expected = torch.tensor(
        [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )

    assert from_mapping.dtype == reference.dtype
    torch.testing.assert_close(from_mapping, from_tensor)
    torch.testing.assert_close(from_mapping, expected, atol=5e-7, rtol=5e-7)


def test_scalar_and_singleton_columns_expand_to_batch():
    normalizer = OracleContextNormalizer()
    reference = torch.zeros((2, FEATURE_DIM))

    result = normalizer(
        {
            "rmag_true": 19.2,
            "image_snr": torch.tensor([[502.5], [502.5]]),
            "central_halpha_snr": 100.5,
        },
        2,
        reference,
    )

    torch.testing.assert_close(result, torch.zeros((2, 3)), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("context", "message"),
    [
        (None, "required"),
        (
            {"rmag_true": [18.0, 19.0]},
            "missing=.*central_halpha_snr.*image_snr",
        ),
        (
            {
                **_oracle_context(2),
                "spectral_reference_quality": torch.ones(2),
            },
            "extra=.*spectral_reference_quality",
        ),
        (
            {
                **_oracle_context(2),
                "halpha_flux_true": torch.ones(2),
            },
            "posterior target.*halpha_flux_true",
        ),
        (
            {
                "rmag_true": torch.tensor([18.0, torch.nan]),
                "image_snr": torch.ones(2),
                "central_halpha_snr": torch.ones(2),
            },
            "finite",
        ),
        (
            {
                "rmag_true": torch.tensor([18.0, 19.0]),
                "image_snr": torch.tensor([5.0, 0.0]),
                "central_halpha_snr": torch.ones(2),
            },
            "must be positive",
        ),
        (
            {
                "rmag_true": torch.tensor([18.0, 19.0]),
                "image_snr": torch.ones(2),
                "central_halpha_snr": torch.tensor([1.0, 0.0]),
            },
            "must be positive",
        ),
        (
            {
                "rmag_true": torch.ones(3),
                "image_snr": torch.ones(2),
                "central_halpha_snr": torch.ones(3),
            },
            r"shape \(2,\)",
        ),
        (torch.ones((2, 2)), r"field order.*shape \(2, 3\)"),
    ],
)
def test_invalid_oracle_context_is_rejected(context, message):
    normalizer = OracleContextNormalizer()

    with pytest.raises(ValueError, match=message):
        normalizer(context, 2, torch.zeros((2, FEATURE_DIM)))


def test_invalid_context_normalization_bounds_are_rejected():
    with pytest.raises(ValueError, match="rmag bounds"):
        OracleContextNormalizer(rmag_min=20.0, rmag_max=20.0)
    with pytest.raises(ValueError, match="image S/N bounds"):
        OracleContextNormalizer(image_snr_min=0.0)
    with pytest.raises(ValueError, match="central H-alpha S/N bounds"):
        OracleContextNormalizer(central_halpha_snr_min=0.0)


def test_public_klnpe_routes_catalog_context_through_backbone_only():
    flow = _RecordingFlow()
    extractor = _TinyFeatureExtractor()
    model = KLNPE(feature_extractor=extractor, flow=flow)
    image, spectra, targets, positions = _dummy_datavector(batch_size=2)
    oracle = _oracle_context(batch_size=2)

    loss = model(
        image,
        spectra,
        targets,
        positions,
        observation_context=oracle,
    )
    assert torch.isfinite(loss)
    assert extractor.observation_contexts[-1] is oracle
    assert flow.log_prob_contexts[-1].shape == (2, FEATURE_DIM)

    density = model.posterior_log_prob(
        image,
        spectra,
        targets,
        positions,
        observation_context=oracle,
    )
    assert density.shape == (2,)
    assert extractor.observation_contexts[-1] is oracle
    assert flow.log_prob_contexts[-1].shape == (2, FEATURE_DIM)

    samples = model.sample(
        image,
        spectra,
        5,
        fiber_positions=positions,
        observation_context=oracle,
    )
    assert samples.shape == (2, 5, TARGET_COUNT)
    assert extractor.observation_contexts[-1] is oracle
    assert flow.sample_contexts[-1].shape == (2, FEATURE_DIM)
