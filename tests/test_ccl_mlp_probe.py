import torch

from diagnostics.ccl_mlp_probe import (
    choose_indices,
    encode_probe_targets,
    evaluate_probe,
    MLPProbe,
    fit_mlp_probe,
)


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


def test_choose_indices_is_seeded_sorted_and_unique():
    first = choose_indices(100, 20, seed=17)
    second = choose_indices(100, 20, seed=17)

    assert (first == second).all()
    assert (first[:-1] < first[1:]).all()


def test_mlp_probe_fits_a_nonlinear_target():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn((384, 4), generator=generator)
    targets = torch.stack(
        (
            features[:, 0] * features[:, 1],
            torch.relu(features[:, 2]) + 0.25 * features[:, 3].square(),
        ),
        dim=1,
    )

    predictions, losses = fit_mlp_probe(
        features[:320],
        targets[:320],
        features[320:],
        hidden_dims=(64, 32),
        epochs=120,
        batch_size=64,
        learning_rate=3e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=11,
    )

    assert predictions.shape == targets[320:].shape
    assert torch.isfinite(predictions).all()
    assert losses[-1] < 0.15 * losses[0]


def test_mlp_probe_contains_nonlinear_hidden_layers():
    probe = MLPProbe(input_dim=8, output_dim=3, hidden_dims=(16, 12))

    assert sum(isinstance(layer, torch.nn.ReLU) for layer in probe.network) == 2
    assert probe(torch.zeros((5, 8))).shape == (5, 3)


def test_probe_metrics_handle_theta_as_a_circular_target():
    generator = torch.Generator().manual_seed(9)
    labels = torch.rand((64, 8), generator=generator) * 2.0 - 1.0
    encoded, encoded_names = encode_probe_targets(labels, FEATURE_NAMES)

    metrics = evaluate_probe(labels, encoded, FEATURE_NAMES, encoded_names)

    assert metrics["g1"]["r2"] == 1.0
    assert metrics["g2"]["rmse"] == 0.0
    assert metrics["theta_int"]["circular_rmse"] < 1e-7
    assert metrics["theta_int"]["mean_cosine_alignment"] > 0.999999
