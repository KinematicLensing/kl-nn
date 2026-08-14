"""Integration regressions for the periodic ``theta_int`` NPE flow."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

import config
from circular_spline import CircularAutoregressiveRationalQuadraticSpline
from networks import (
    ConditionalNormalWithCircularTheta,
    KLNPE,
    PeriodicThetaFlow,
)
from nflows.flows.base import Flow
from nflows.transforms.permutations import Permutation


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
THETA_INDEX = FEATURE_NAMES.index("theta_int")


class ConstantContextEncoder(nn.Module):
    def __init__(self, output_features):
        super().__init__()
        self.output_features = int(output_features)

    def forward(self, context):
        return context.new_zeros(context.shape[0], self.output_features)


class TinyFeatureExtractor(nn.Module):
    """Cheap 1024-vector fixture satisfying both plain and D4 KLNPE APIs."""

    def forward(self, image, spectra, fiber_positions, fiber_mask=None):
        del spectra, fiber_positions, fiber_mask
        return image.mean(dim=tuple(range(1, image.ndim)), keepdim=False)[:, None].expand(
            -1, 1024
        )

    def transform_features(self, features, element):
        del element
        return features


class FixedSampleFlow(nn.Module):
    """Minimal flow fixture returning a prescribed posterior sample tensor."""

    def __init__(self, samples):
        super().__init__()
        self.register_buffer("samples", samples)

    def sample(self, num_samples, context=None):
        assert context is not None
        assert self.samples.shape[:2] == (context.shape[0], num_samples)
        return self.samples.to(device=context.device, dtype=context.dtype).clone()


def _sampling_only_circular_model(samples):
    """Build only the KLNPE state exercised by ``_draw_flow_samples``."""
    model = KLNPE.__new__(KLNPE)
    nn.Module.__init__(model)
    model.flow_type = "circular_rqs"
    model.feature_names = list(FEATURE_NAMES)
    model.nfeatures = len(FEATURE_NAMES)
    model.theta_idx = THETA_INDEX
    model.flow = FixedSampleFlow(samples)
    return model


def _load_training_entrypoint():
    path = Path(__file__).resolve().parents[1] / "arch" / "[scr]_train_model.py"
    spec = importlib.util.spec_from_file_location("circular_flow_training_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_splines_nonidentity(model, seed):
    """Exercise a learned map instead of the deliberately exact initialization."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CircularAutoregressiveRationalQuadraticSpline):
                module.autoregressive_net.final_layer.weight.normal_(
                    mean=0.0, std=0.01, generator=generator
                )
                module.autoregressive_net.final_layer.bias.normal_(
                    mean=0.0, std=0.03, generator=generator
                )


@pytest.fixture
def circular_model_factory():
    original = copy.deepcopy(config.MODEL_CONFIG)
    made_models = []

    def make(*, posterior_symmetry="none", num_layers=2):
        configured = copy.deepcopy(original)
        configured.flow.flow_type = "circular_rqs"
        configured.flow.num_layers = num_layers
        configured.flow.num_bins = 8
        configured.train.feature_number = len(FEATURE_NAMES)
        configured.train.feature_names = list(FEATURE_NAMES)
        configured.train.posterior_symmetry = posterior_symmetry
        config.set_model_config(configured)
        model = KLNPE(
            feature_extractor=TinyFeatureExtractor(),
            mode=1,
            batch_size=2,
            nfeatures=len(FEATURE_NAMES),
            nspec=5,
            posterior_symmetry=posterior_symmetry,
        ).eval()
        made_models.append(model)
        return model

    try:
        yield make
    finally:
        del made_models[:]
        config.set_model_config(original)


def test_hybrid_base_has_gaussian_times_uniform_circle_density_and_support():
    base = ConditionalNormalWithCircularTheta(
        features=4,
        context_encoder=ConstantContextEncoder(6),
    )
    context = torch.zeros(3, 5, dtype=torch.float64)
    inputs = torch.tensor(
        [
            [0.2, -0.4, 0.1, -1.0],
            [-0.3, 0.5, 0.7, 0.999],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    actual = base.log_prob(inputs, context=context)
    expected = (
        -0.5 * inputs[:2, :-1].square().sum(dim=-1)
        - 1.5 * math.log(2.0 * math.pi)
        - math.log(2.0)
    )

    torch.testing.assert_close(actual[:2], expected)
    assert actual[2].item() == -math.inf


def test_hybrid_base_samples_have_context_shape_and_compact_theta_support():
    torch.manual_seed(2701)
    base = ConditionalNormalWithCircularTheta(
        features=4,
        context_encoder=ConstantContextEncoder(6),
    )
    context = torch.zeros(3, 5)

    samples = base.sample(257, context=context)

    assert samples.shape == (3, 257, 4)
    assert torch.isfinite(samples).all()
    assert torch.all(samples[..., -1] >= -1.0)
    assert torch.all(samples[..., -1] < 1.0)


def test_old_flow_config_without_new_keys_restores_affine_defaults():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["flow"].pop("flow_type", None)
    payload["flow"].pop("num_bins", None)

    restored = config.ModelConfig.from_dict(payload)

    assert restored.flow.flow_type == "affine"
    assert restored.flow.num_bins == 8


def test_training_cli_records_circular_flow_and_rejects_it_for_pretraining():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            [
                "--train-type",
                "train",
                "--flow-type",
                "circular_rqs",
                "--flow-num-bins",
                "12",
            ]
        )
        entrypoint.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert restored.flow.flow_type == "circular_rqs"
        assert restored.flow.num_bins == 12

        pretrain_args = entrypoint.parse_args(
            ["--train-type", "pretrain", "--flow-type", "circular_rqs"]
        )
        with pytest.raises(ValueError, match="only valid for NPE training"):
            entrypoint.apply_overrides(pretrain_args)
    finally:
        config.set_model_config(original)


def test_production_circular_flow_keeps_theta_last_through_every_mixing_step(
    circular_model_factory,
):
    model = circular_model_factory(num_layers=3)

    assert isinstance(model.flow, PeriodicThetaFlow)
    transforms = model.transform._transforms
    permutations = [item for item in transforms if isinstance(item, Permutation)]
    splines = [
        item
        for item in transforms
        if isinstance(item, CircularAutoregressiveRationalQuadraticSpline)
    ]

    # The sole physical-to-internal permutation moves theta_int from index 2
    # to index 7. Every subsequent permutation may mix only the seven linear
    # variables and must leave the final circular coordinate fixed.
    assert permutations[0]._permutation.tolist() == [0, 1, 3, 4, 5, 6, 7, 2]
    for permutation in permutations[1:]:
        assert permutation._permutation[-1].item() == len(FEATURE_NAMES) - 1
        assert sorted(permutation._permutation[:-1].tolist()) == list(range(7))
    assert len(splines) == 3
    for spline in splines:
        assert spline.tails == ["linear"] * 7 + ["circular"]


def test_production_circular_flow_roundtrips_and_has_finite_input_gradients(
    circular_model_factory,
):
    torch.manual_seed(2702)
    model = circular_model_factory(num_layers=2).double()
    _make_splines_nonidentity(model, seed=27022)
    context = torch.randn(3, 1024, dtype=torch.float64)
    inputs = torch.tensor(
        [
            [0.1, -0.2, -0.999, 0.3, -0.4, 0.5, -0.6, 0.7],
            [-0.4, 0.5, 0.0, -0.6, 0.7, -0.8, 0.9, -0.1],
            [0.7, -0.6, 0.999, 0.5, -0.4, 0.3, -0.2, 0.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    noise = model.flow.transform_to_noise(inputs, context=context)
    restored, inverse_logdet = model.transform.inverse(noise, context=context)
    log_prob = model.flow.log_prob(inputs, context=context)
    gradient = torch.autograd.grad(log_prob.sum(), inputs)[0]

    torch.testing.assert_close(restored, inputs, atol=1e-9, rtol=1e-9)
    assert torch.isfinite(inverse_logdet).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(gradient).all()


def test_production_circular_flow_density_and_samples_respect_theta_seam(
    circular_model_factory,
):
    torch.manual_seed(2703)
    model = circular_model_factory(num_layers=2).double()
    _make_splines_nonidentity(model, seed=27033)
    context = torch.randn(2, 1024, dtype=torch.float64)
    common = torch.tensor(
        [0.1, -0.2, 0.0, 0.3, -0.4, 0.5, -0.6, 0.7],
        dtype=torch.float64,
    ).expand(2, -1).clone()
    common[0, THETA_INDEX] = -1.0
    common[1, THETA_INDEX] = 1.0

    repeated_context = context[:1].expand(2, -1)
    endpoint_log_prob = model.flow.log_prob(common, context=repeated_context)
    near_seam = common.clone()
    near_seam[0, THETA_INDEX] = -1.0 + 1e-7
    near_seam[1, THETA_INDEX] = 1.0 - 1e-7
    near_seam_log_prob = model.flow.log_prob(near_seam, context=repeated_context)
    samples = model.flow.sample(129, context=context)

    torch.testing.assert_close(endpoint_log_prob[0], endpoint_log_prob[1])
    torch.testing.assert_close(
        near_seam_log_prob[0], near_seam_log_prob[1], atol=1e-5, rtol=1e-6
    )
    assert samples.shape == (2, 129, len(FEATURE_NAMES))
    theta = samples[..., THETA_INDEX]
    assert torch.isfinite(samples).all()
    assert torch.all(theta >= -1.0)
    assert torch.all(theta < 1.0)


def test_d4_posterior_log_density_identifies_theta_seam(circular_model_factory):
    torch.manual_seed(2704)
    model = circular_model_factory(posterior_symmetry="d4", num_layers=1).double()
    raw_features = torch.randn(2, 1024, dtype=torch.float64)
    parameters = torch.tensor(
        [
            [0.1, -0.2, -1.0, 0.3, -0.4, 0.5, -0.6, 0.7],
            [0.1, -0.2, 1.0, 0.3, -0.4, 0.5, -0.6, 0.7],
        ],
        dtype=torch.float64,
    )

    actual = model._d4_mixture_log_prob_from_features(raw_features[:1].expand(2, -1), parameters)

    torch.testing.assert_close(actual[0], actual[1], atol=1e-10, rtol=1e-10)


def test_legacy_affine_setup_remains_an_ordinary_flow():
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        configured = copy.deepcopy(original)
        configured.flow.flow_type = "affine"
        configured.flow.num_layers = 1
        config.set_model_config(configured)
        model = KLNPE(
            feature_extractor=TinyFeatureExtractor(),
            mode=1,
            batch_size=2,
            nfeatures=len(FEATURE_NAMES),
            nspec=5,
            posterior_symmetry="none",
        )

        assert type(model.flow) is Flow
        assert not isinstance(model.flow, PeriodicThetaFlow)
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_draw_flow_samples_canonicalizes_circle_endpoint_and_ulp_excursions(dtype):
    """The two seam representatives and roundoff beside them are valid."""
    one = torch.tensor(1.0, dtype=dtype)
    minus_one = torch.tensor(-1.0, dtype=dtype)
    positive_inward = torch.nextafter(one, torch.tensor(-math.inf, dtype=dtype))
    positive_outward = torch.nextafter(one, torch.tensor(math.inf, dtype=dtype))
    negative_outward = torch.nextafter(
        minus_one, torch.tensor(-math.inf, dtype=dtype)
    )
    theta = torch.stack(
        (
            minus_one,
            positive_inward,
            one,
            positive_outward,
            negative_outward,
        )
    )
    prescribed = torch.zeros(
        1, theta.numel(), len(FEATURE_NAMES), dtype=dtype
    )
    prescribed[0, :, THETA_INDEX] = theta
    model = _sampling_only_circular_model(prescribed)
    context = torch.zeros(1, 1024, dtype=dtype)

    actual = model._draw_flow_samples(
        theta.numel(),
        context,
        sample_id="boundary-roundoff",
        canonical_theta=False,
    )

    outside = (theta < -1.0) | (theta >= 1.0)
    expected = torch.where(
        outside,
        torch.remainder(theta + 1.0, 2.0) - 1.0,
        theta,
    )
    torch.testing.assert_close(actual[0, :, THETA_INDEX], expected)
    assert actual[0, 1, THETA_INDEX] == positive_inward
    assert torch.all(actual[..., THETA_INDEX] >= -1.0)
    assert torch.all(actual[..., THETA_INDEX] < 1.0)


def test_draw_flow_samples_wraps_all_finite_circle_representatives():
    """Even distant finite representatives are valid angles modulo one turn."""
    theta = torch.tensor((1.05, -1.05, 5.25, -4.75))
    prescribed = torch.zeros(1, theta.numel(), len(FEATURE_NAMES))
    prescribed[0, :, THETA_INDEX] = theta
    model = _sampling_only_circular_model(prescribed)
    context = torch.zeros(1, 1024)

    actual = model._draw_flow_samples(
        theta.numel(),
        context,
        sample_id="noncanonical-circle-representatives",
        canonical_theta=False,
    )

    expected = torch.remainder(theta + 1.0, 2.0) - 1.0
    torch.testing.assert_close(actual[0, :, THETA_INDEX], expected)
    assert torch.all(actual[..., THETA_INDEX] >= -1.0)
    assert torch.all(actual[..., THETA_INDEX] < 1.0)
