import numpy as np
import torch

from map_laplace import (
    laplace_covariance_from_log_prob,
    tf_log_prior_ratio_torch,
    truncated_tf_log_prob_torch,
)
from tf_prior import TFPrior, tf_log_prior_ratio, truncated_tf_log_prob


def test_torch_tf_log_prob_matches_numpy():
    prior = TFPrior()
    rng = np.random.default_rng(0)
    rmag = rng.uniform(16.0, 22.0, size=32)
    mean_log10 = (rmag - prior.intercept) / prior.slope
    vcirc = np.clip(10.0 ** (mean_log10 + 0.05 * rng.normal(size=32)), 70.0, 500.0)
    numpy_lp = truncated_tf_log_prob(vcirc, rmag, prior)
    torch_lp = truncated_tf_log_prob_torch(
        torch.as_tensor(vcirc), torch.as_tensor(rmag), prior
    )
    np.testing.assert_allclose(torch_lp.numpy(), numpy_lp, rtol=1e-10, atol=1e-10)
    numpy_ratio = tf_log_prior_ratio(vcirc, rmag, prior)
    torch_ratio = tf_log_prior_ratio_torch(
        torch.as_tensor(vcirc), torch.as_tensor(rmag), prior
    )
    np.testing.assert_allclose(
        torch_ratio.numpy(), numpy_ratio, rtol=1e-10, atol=1e-10
    )


def test_torch_tf_log_prob_is_neg_inf_off_support():
    prior = TFPrior()
    vcirc = torch.tensor([10.0, 200.0, 800.0])
    rmag = torch.tensor([20.0, 20.0, 20.0])
    log_prob = truncated_tf_log_prob_torch(vcirc, rmag, prior)
    assert torch.isneginf(log_prob[0])
    assert torch.isfinite(log_prob[1])
    assert torch.isneginf(log_prob[2])


def test_laplace_recovers_known_shear_block_of_quadratic():
    precision = torch.diag(
        torch.tensor(
            [25.0, 16.0, 4.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float64,
        )
    )
    precision[0, 5] = precision[5, 0] = 2.0
    expected = torch.linalg.inv(precision)[:2, :2]

    def log_prob(theta):
        return -0.5 * theta @ precision.to(dtype=theta.dtype) @ theta

    covariance, ok = laplace_covariance_from_log_prob(
        log_prob, torch.zeros(9, dtype=torch.float64)
    )
    assert ok
    np.testing.assert_allclose(covariance, expected.numpy(), rtol=1e-6, atol=1e-8)


def test_laplace_marks_non_positive_hessian_invalid():
    def log_prob(theta):
        return 0.5 * torch.square(theta).sum()

    covariance, ok = laplace_covariance_from_log_prob(
        log_prob, torch.zeros(9, dtype=torch.float64)
    )
    assert not ok
    assert not np.isfinite(covariance).any()
