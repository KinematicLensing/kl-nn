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
    )
    edge_inputs = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    _, logdet = transform.forward(edge_inputs)

    assert torch.allclose(logdet[0], logdet[1], atol=1e-4, rtol=1e-4)
