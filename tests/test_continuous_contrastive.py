import math

import pytest
import torch

from networks import ContinuousContrastiveLoss


def _loss(nfeatures=9):
    return ContinuousContrastiveLoss(
        temperature=0.1,
        sigma_label=0.5,
        d_cutoff=2.0,
        label_scales=[1.0] * nfeatures,
        theta_idx=2,
        distance_reduction="mean",
    )


def test_theta_distance_is_continuous_across_periodic_seam():
    labels = torch.zeros((3, 9), dtype=torch.float32)
    labels[0, 2] = -0.99
    labels[1, 2] = 0.99
    labels[2, 2] = 0.0

    distance = _loss().pairwise_label_distance_sq(labels)

    expected_seam_distance = (0.02 ** 2) / labels.shape[1]
    assert distance[0, 1].item() == pytest.approx(expected_seam_distance, abs=1e-7)
    assert distance[0, 1] < distance[0, 2]


def test_theta_values_separated_by_pi_remain_distinct():
    labels = torch.zeros((2, 9), dtype=torch.float32)
    labels[1, 2] = 1.0

    distance = _loss().pairwise_label_distance_sq(labels)

    assert distance[0, 1].item() == pytest.approx(1.0 / labels.shape[1])


def test_fixed_pair_distance_does_not_depend_on_other_batch_members():
    pair = torch.tensor(
        [
            [0.1, -0.2, 0.95, 0.0, 0.1, 0.2, 0.3, 0.4, -0.5],
            [-0.2, 0.3, -0.95, 0.2, -0.1, 0.0, 0.5, -0.4, 0.6],
        ],
        dtype=torch.float32,
    )
    extended = torch.cat((pair, torch.full((1, 9), 0.75)), dim=0)

    pair_distance = _loss().pairwise_label_distance_sq(pair)[0, 1]
    extended_distance = _loss().pairwise_label_distance_sq(extended)[0, 1]

    torch.testing.assert_close(pair_distance, extended_distance)


def test_continuous_contrastive_loss_is_finite_and_differentiable():
    generator = torch.Generator().manual_seed(123)
    z = torch.randn((16, 32), generator=generator, requires_grad=True)
    labels = torch.rand((16, 9), generator=generator) * 2.0 - 1.0

    value = _loss()(z, labels)
    value.backward()

    assert math.isfinite(value.item())
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_invalid_label_scale_configuration_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        ContinuousContrastiveLoss(label_scales=[1.0, 0.0], theta_idx=1)



def test_continuous_contrastive_loss_supports_float16_logits():
    generator = torch.Generator().manual_seed(321)
    z = torch.randn((8, 16), generator=generator, dtype=torch.float16)
    z.requires_grad_(True)
    labels = torch.rand((8, 9), generator=generator, dtype=torch.float16) * 2 - 1

    value = _loss()(z, labels)
    value.backward()

    assert torch.isfinite(value)
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_continuous_contrastive_loss_rejects_singleton_batch():
    with pytest.raises(ValueError, match="at least two"):
        _loss()(torch.ones((1, 4)), torch.zeros((1, 9)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_kernel_hyperparameters_are_rejected(value):
    with pytest.raises(ValueError, match="positive and finite"):
        ContinuousContrastiveLoss(temperature=value)

def test_target_diagnostics_identify_uniform_embeddings():
    generator = torch.Generator().manual_seed(777)
    labels = torch.rand((32, 9), generator=generator) * 2.0 - 1.0
    z = torch.ones((32, 16), dtype=torch.float32, requires_grad=True)
    loss_fn = ContinuousContrastiveLoss(
        temperature=0.1,
        sigma_label=0.15,
        d_cutoff=0.40,
        label_scales=[1.0] * 9,
        theta_idx=2,
        distance_reduction="mean",
    )

    value, diagnostics = loss_fn(z, labels, return_diagnostics=True)
    value.backward()

    assert set(diagnostics) == {
        "target_entropy",
        "uniform_baseline",
        "effective_positives",
        "target_mass",
        "excess_loss",
    }
    assert diagnostics["target_entropy"] <= diagnostics["uniform_baseline"]
    assert 1.0 <= diagnostics["effective_positives"] <= labels.shape[0] - 1
    assert 0.0 < diagnostics["target_mass"] <= 1.0
    torch.testing.assert_close(value.detach(), diagnostics["uniform_baseline"])
    torch.testing.assert_close(
        value.detach(),
        diagnostics["target_entropy"] + diagnostics["excess_loss"],
    )
    assert all(not metric.requires_grad for metric in diagnostics.values())
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
