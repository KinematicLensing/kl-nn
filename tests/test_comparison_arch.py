"""Contracts for the additive position-query comparison architecture."""

import copy

import pytest
import torch

import config
import train
from comparison_arch import (
    COMPARISON_FEATURE_DIM,
    BoundedDiagonalGaussian,
    ComparisonFactorizedFlow,
    ComparisonFeatureExtractor,
    ComparisonKLNPE,
    PositionQueryFiberPool,
    SharedFiberSpectrumEncoder,
)
from networks import BoundedHybridCircularFlow, ImgCNN, JointSpecCNN


@pytest.fixture(autouse=True)
def _restore_model_config():
    original = copy.deepcopy(config.MODEL_CONFIG)
    yield
    config.set_model_config(original)


def _context(batch_size=2):
    return {
        "rmag_true": torch.linspace(18.0, 21.0, batch_size),
        "image_snr": torch.linspace(50.0, 500.0, batch_size),
        "central_halpha_snr": torch.linspace(20.0, 120.0, batch_size),
    }


def _inputs(batch_size=2, n_fibers=5, image_size=48):
    generator = torch.Generator().manual_seed(20260906)
    image = torch.randn(
        batch_size, 1, image_size, image_size, generator=generator
    )
    spectra = torch.randn(batch_size, 1, n_fibers, 64, generator=generator)
    positions = torch.randn(batch_size, n_fibers, 2, generator=generator)
    return image, spectra, positions, _context(batch_size)


def _normalized_targets(batch_size=2):
    generator = torch.Generator().manual_seed(17)
    values = torch.rand(batch_size, 9, generator=generator) * 1.8 - 0.9
    values[:, :2].clamp_(-0.2, 0.2)
    return values


def test_shared_fiber_encoder_reuses_one_1d_tower():
    extractor = ComparisonFeatureExtractor().eval()
    towers = [
        module
        for module in extractor.modules()
        if isinstance(module, SharedFiberSpectrumEncoder)
    ]
    assert len(towers) == 1
    image, spectra, positions, context = _inputs()
    fiber0 = spectra[:, :, 0, :]
    fiber1 = spectra[:, :, 1, :]
    stacked = torch.cat((fiber0, fiber1), dim=0)
    with torch.inference_mode():
        encoded = extractor.spec_net(stacked)
        first = extractor.spec_net(fiber0)
        second = extractor.spec_net(fiber1)
    torch.testing.assert_close(encoded[: image.shape[0]], first)
    torch.testing.assert_close(encoded[image.shape[0] :], second)


def test_position_queries_are_the_attention_queries():
    pool = PositionQueryFiberPool().eval()
    tokens = torch.randn(2, 5, 256)
    positions = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    recorded = {}
    original = pool.attention.forward

    def _wrapped(query, key, value, **kwargs):
        recorded["query"] = query
        recorded["key"] = key
        recorded["value"] = value
        recorded["key_padding_mask"] = kwargs.get("key_padding_mask")
        return original(query, key, value, **kwargs)

    pool.attention.forward = _wrapped
    pool(tokens, positions, mask)
    torch.testing.assert_close(recorded["query"], pool.query(positions))
    torch.testing.assert_close(recorded["key"], tokens)
    torch.testing.assert_close(recorded["value"], tokens)
    torch.testing.assert_close(recorded["key_padding_mask"], ~mask)


def test_paired_fiber_permutation_is_a_set_noop():
    pool = PositionQueryFiberPool().eval()
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn(2, 5, 256, generator=generator)
    positions = torch.randn(2, 5, 2, generator=generator)
    mask = torch.ones(2, 5, dtype=torch.bool)
    perm = torch.tensor([2, 0, 4, 1, 3])
    with torch.inference_mode():
        reference = pool(tokens, positions, mask)
        paired = pool(tokens[:, perm], positions[:, perm], mask[:, perm])
    torch.testing.assert_close(reference, paired, atol=1e-5, rtol=1e-5)


def test_masked_fiber_token_does_not_affect_pooled_features():
    pool = PositionQueryFiberPool().eval()
    positions = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 4] = False
    tokens = torch.ones(2, 5, 256)
    tokens[:, 4] = 50.0
    altered = tokens.clone()
    altered[:, 4] = -50.0
    used = torch.ones(2, 5, 256)
    used[:, 0] = 50.0
    used[:, 4] = 50.0
    with torch.inference_mode():
        reference = pool(tokens, positions, mask)
        unused = pool(altered, positions, mask)
        shifted = pool(used, positions, mask)
    torch.testing.assert_close(reference, unused, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(reference, shifted, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n_fibers", (3, 5))
def test_comparison_extractor_accepts_variable_fiber_counts(n_fibers):
    extractor = ComparisonFeatureExtractor().eval()
    image, spectra, positions, context = _inputs(n_fibers=n_fibers)
    with torch.inference_mode():
        features = extractor(image, spectra, positions, context)
    assert features.shape == (image.shape[0], COMPARISON_FEATURE_DIM)


def test_bounded_diagonal_gaussian_support_and_samples():
    head = BoundedDiagonalGaussian(8, features=2)
    context = torch.randn(4, 8)
    interior = 0.1 * torch.randn(4, 2)
    log_prob = head.log_prob(interior, context)
    assert torch.isfinite(log_prob).all()
    exterior = torch.tensor(
        [[1.5, 0.0], [-1.2, 0.1], [0.0, 2.0], [1.0, 0.0]],
        dtype=context.dtype,
    )
    assert torch.isneginf(head.log_prob(exterior, context)).all()
    samples = head.sample(11, context)
    assert samples.shape == (4, 11, 2)
    assert bool((samples.abs() < 1.0).all())


def test_comparison_model_does_not_instantiate_concat_towers():
    model = ComparisonKLNPE().eval()
    assert not any(
        isinstance(module, (ImgCNN, JointSpecCNN)) for module in model.modules()
    )
    image, spectra, positions, context = _inputs(batch_size=2)
    targets = _normalized_targets(2)
    loss = model(
        image,
        spectra,
        targets,
        fiber_positions=positions,
        observation_context=context,
    )
    assert torch.isfinite(loss)
    assert "g_log_prob_mean" in model.last_training_diagnostics
    samples = model.sample(
        image,
        spectra,
        4,
        fiber_positions=positions,
        observation_context=context,
    )
    assert samples.shape == (2, 4, 9)
    assert bool((samples[..., :2].abs() < 1.0).all())


def test_comparison_optimizer_groups_are_nonempty_and_disjoint():
    model = ComparisonKLNPE()
    groups = train._npe_optimizer_parameters(model, config.train)
    names = [group["group_name"] for group in groups]
    assert names == ["shared", "non_theta_flow", "theta_transform"]
    ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    assert len(ids) == len(set(ids))
    assert all(len(group["params"]) > 0 for group in groups)


def test_comparison_channels_last_is_disabled():
    stage = {"channels_last": True, "architecture": "comparison"}
    assert train._use_channels_last(stage) is False
    joint = {"channels_last": True, "architecture": "comparison_joint"}
    assert train._use_channels_last(joint) is False
    concat = {"channels_last": True, "architecture": "concat"}
    assert train._use_channels_last(concat) is True


def test_comparison_default_flow_is_factorized_gaussian():
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.architecture = "comparison"
    config.set_model_config(configured)
    model = ComparisonKLNPE()
    assert isinstance(model.flow, ComparisonFactorizedFlow)
    assert model.flow.features == 9


def test_comparison_joint_uses_nine_target_hybrid_flow():
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.architecture = "comparison_joint"
    config.set_model_config(configured)
    model = ComparisonKLNPE()
    assert isinstance(model.flow, BoundedHybridCircularFlow)
    assert not isinstance(model.flow, ComparisonFactorizedFlow)
    assert model.flow.features == 9
    assert model.flow.theta_index == 2
    groups = train._npe_optimizer_parameters(model, config.train)
    assert [group["group_name"] for group in groups] == [
        "shared",
        "non_theta_flow",
        "theta_transform",
    ]
    assert all(len(group["params"]) > 0 for group in groups)
    image, spectra, positions, context = _inputs(batch_size=2)
    loss = model(
        image,
        spectra,
        _normalized_targets(2),
        fiber_positions=positions,
        observation_context=context,
    )
    assert torch.isfinite(loss)
    assert "non_theta_log_prob_mean" in model.last_training_diagnostics
    assert "g_log_prob_mean" not in model.last_training_diagnostics
    samples = model.sample(
        image,
        spectra,
        4,
        fiber_positions=positions,
        observation_context=context,
    )
    assert samples.shape == (2, 4, 9)
    assert bool((samples.abs() < 1.0).all())


def test_load_model_resolves_comparison_class_from_snapshot(tmp_path):
    import model_registry

    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.model_name = "comparison-load"
    configured.train.architecture = "comparison"
    model_registry.save_model_artifacts(
        configured,
        configs_root=str(tmp_path / "configs"),
        networks_root=str(tmp_path / "networks"),
    )
    model = ComparisonKLNPE()
    checkpoint = tmp_path / "models" / "comparison-load" / "comparison-loadbest"
    checkpoint.parent.mkdir(parents=True)
    torch.save(model.state_dict(), checkpoint)
    restored = train.load_model(
        ComparisonKLNPE,
        path=str(checkpoint),
        model_name="comparison-load",
        networks_root=str(tmp_path / "networks"),
    )
    assert type(restored).__name__ == "ComparisonKLNPE"
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
