import pytest
import torch

circular = pytest.importorskip("circular_spline")


def test_circular_spline_roundtrip():
    transform = circular.CircularAutoregressiveRationalQuadraticSpline(
        num_input_channels=1,
        num_blocks=2,
        num_hidden_channels=16,
        ind_circ=[0],
        num_bins=8,
        tail_bound=1.0,
        identity_init=False,
    )
    x = torch.linspace(-0.9, 0.9, 11, dtype=torch.float32).unsqueeze(1)
    y, logdet = transform.forward(x)
    x_inv, logdet_inv = transform.inverse(y)

    assert torch.allclose(x, x_inv, atol=1e-4, rtol=1e-4)
    assert torch.allclose(logdet + logdet_inv, torch.zeros_like(logdet), atol=1e-4, rtol=1e-4)


def test_circular_spline_logdet_continuity_at_bounds():
    transform = circular.CircularAutoregressiveRationalQuadraticSpline(
        num_input_channels=1,
        num_blocks=2,
        num_hidden_channels=16,
        ind_circ=[0],
        num_bins=8,
        tail_bound=1.0,
        identity_init=False,
    )
    edge_inputs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    _, logdet = transform.forward(edge_inputs)

    assert torch.allclose(logdet[0], logdet[1], atol=1e-4, rtol=1e-4)


def test_mixed_circular_spline_roundtrip_with_context_and_linear_tails():
    """Exercise the mixed-tail path used by the joint eight-parameter flow."""
    torch.manual_seed(7)
    transform = circular.CircularAutoregressiveRationalQuadraticSpline(
        num_input_channels=4,
        num_blocks=2,
        num_hidden_channels=16,
        ind_circ=[3],
        num_context_channels=5,
        num_bins=8,
        tail_bound=1.0,
        identity_init=False,
    ).double().eval()
    # Linear coordinates outside the spline interval must retain identity tails,
    # while the circular coordinate always remains on its compact support.
    inputs = torch.tensor(
        [
            [-0.7, 0.2, 1.3, 0.999],
            [0.4, -1.2, 0.1, -0.999],
        ],
        dtype=torch.float64,
    )
    context = torch.randn(2, 5, dtype=torch.float64)

    outputs, logdet = transform(inputs, context=context)
    restored, inverse_logdet = transform.inverse(outputs, context=context)

    torch.testing.assert_close(restored, inputs, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        logdet + inverse_logdet,
        torch.zeros_like(logdet),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(outputs[0, 2], inputs[0, 2])
    torch.testing.assert_close(outputs[1, 1], inputs[1, 1])
    assert torch.all((-1.0 <= outputs[:, 3]) & (outputs[:, 3] <= 1.0))


def test_mixed_circular_spline_reports_full_autograd_jacobian():
    """Guard the change-of-variables term, not only forward/inverse agreement."""
    torch.manual_seed(11)
    transform = circular.CircularAutoregressiveRationalQuadraticSpline(
        num_input_channels=4,
        num_blocks=2,
        num_hidden_channels=16,
        ind_circ=[3],
        num_context_channels=3,
        num_bins=8,
        tail_bound=1.0,
        identity_init=False,
    ).double().eval()
    point = torch.tensor(
        [0.2, -0.4, 0.3, 0.1], dtype=torch.float64, requires_grad=True
    )
    context = torch.randn(1, 3, dtype=torch.float64)

    def forward_one(vector):
        return transform(vector.unsqueeze(0), context=context)[0].squeeze(0)

    jacobian = torch.autograd.functional.jacobian(forward_one, point)
    sign, expected_logdet = torch.linalg.slogdet(jacobian)
    _, actual_logdet = transform(point.unsqueeze(0), context=context)

    assert sign > 0
    torch.testing.assert_close(
        actual_logdet[0], expected_logdet, atol=1e-10, rtol=1e-10
    )


def test_theta_last_mixed_spline_is_continuous_across_full_seam():
    """Catch MADE coordinates downstream of theta breaking circular topology."""
    torch.manual_seed(19)
    transform = circular.CircularAutoregressiveRationalQuadraticSpline(
        num_input_channels=4,
        num_blocks=2,
        num_hidden_channels=16,
        ind_circ=[3],
        num_context_channels=3,
        num_bins=8,
        tail_bound=1.0,
    ).double().eval()
    context = torch.randn(1, 3, dtype=torch.float64).expand(2, -1)
    seam = torch.tensor(
        [
            [0.2, -0.4, 0.3, -1.0],
            [0.2, -0.4, 0.3, 1.0],
        ],
        dtype=torch.float64,
    )

    outputs, logdet = transform(seam, context=context)

    # Endpoints represent the same angle, so neither the non-angular map nor
    # the density Jacobian may distinguish them.
    torch.testing.assert_close(outputs[0, :-1], outputs[1, :-1], atol=1e-10, rtol=0)
    torch.testing.assert_close(logdet[0], logdet[1], atol=1e-10, rtol=0)
    torch.testing.assert_close(outputs[0, -1], outputs[1, -1], atol=1e-10, rtol=0)
    torch.testing.assert_close(
        outputs[:, -1], torch.full_like(outputs[:, -1], -1.0), atol=1e-10, rtol=0
    )
