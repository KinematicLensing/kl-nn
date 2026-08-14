import copy
import importlib.util
from pathlib import Path

import pytest
import torch

import config
from data import apply_d4_to_datavector
from train import make_npe_training_batch


def _load_training_entrypoint():
    entrypoint_path = (
        Path(__file__).resolve().parents[1] / "arch" / "[scr]_train_model.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "stage4_npe_training_entrypoint",
        entrypoint_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _datavector(batch_size=2):
    image = torch.arange(
        batch_size * 9,
        dtype=torch.float32,
    ).reshape(batch_size, 1, 3, 3)
    spectra = torch.arange(
        batch_size * 10,
        dtype=torch.float32,
    ).reshape(batch_size, 1, 5, 2)
    labels = torch.tensor(
        [
            [0.10, -0.20, 0.75, -0.4, 0.3, -0.2, 0.1, 0.9],
            [-0.30, 0.40, -0.80, 0.2, -0.1, 0.6, -0.5, 0.7],
        ],
        dtype=torch.float32,
    )[:batch_size]
    fiber_positions = torch.tensor(
        [
            [
                [2.0, 0.5],
                [-2.0, -0.5],
                [0.25, -0.1],
                [-0.5, 2.0],
                [0.5, -2.0],
            ],
            [
                [1.5, -0.25],
                [-1.5, 0.25],
                [-0.1, 0.2],
                [0.75, 1.25],
                [-0.75, -1.25],
            ],
        ],
        dtype=torch.float32,
    )[:batch_size]
    snr = torch.tensor([12.5, 87.0], dtype=torch.float32)[:batch_size]
    return image, spectra, labels, fiber_positions, snr


def test_old_train_config_defaults_to_legacy_rotation_and_no_posterior_symmetry():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["train"].pop("use_rot90_counterpart", None)
    payload["train"].pop("posterior_symmetry", None)

    restored = config.ModelConfig.from_dict(payload)

    assert restored.train.use_rot90_counterpart is True
    assert restored.train.posterior_symmetry == "none"


def test_stage4_npe_cli_persists_d4_symmetry_without_rot90_counterpart():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            [
                "--train-type",
                "train",
                "--backbone-type",
                "stage4_d4",
                "--posterior-symmetry",
                "d4",
                "--no-rot90-counterpart",
            ]
        )
        stage = entrypoint.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert stage.backbone_type == "stage4_d4"
        assert stage.posterior_symmetry == "d4"
        assert stage.use_rot90_counterpart is False
        assert restored.train.backbone_type == "stage4_d4"
        assert restored.train.posterior_symmetry == "d4"
        assert restored.train.use_rot90_counterpart is False
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize(
    "arguments",
    (
        (
            "--backbone-type",
            "legacy",
            "--posterior-symmetry",
            "d4",
            "--no-rot90-counterpart",
        ),
        (
            "--backbone-type",
            "stage3",
            "--posterior-symmetry",
            "d4",
            "--no-rot90-counterpart",
        ),
        (
            "--backbone-type",
            "stage4_d4",
            "--posterior-symmetry",
            "none",
            "--no-rot90-counterpart",
        ),
        (
            "--backbone-type",
            "stage4_d4",
            "--posterior-symmetry",
            "d4",
            "--rot90-counterpart",
        ),
    ),
)
def test_npe_cli_rejects_incompatible_backbone_symmetry_and_rotation(arguments):
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(("--train-type", "train", *arguments))
        with pytest.raises(ValueError, match="D4|d4|rot90|symmetry"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_legacy_npe_batch_appends_exact_canonical_rot90_counterpart():
    image, spectra, labels, fiber_positions, snr = _datavector()
    rotated = apply_d4_to_datavector(
        image,
        spectra,
        labels,
        fiber_positions,
        element="r90",
    )

    actual = make_npe_training_batch(
        image,
        spectra,
        labels,
        fiber_positions,
        snr,
        use_rot90_counterpart=True,
    )
    expected = (
        torch.cat((image, rotated[0]), dim=0),
        torch.cat((spectra, rotated[1]), dim=0),
        torch.cat((labels, rotated[2]), dim=0),
        torch.cat((fiber_positions, rotated[3]), dim=0),
        torch.cat((snr, snr), dim=0),
    )

    assert len(actual) == len(expected) == 5
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_d4_npe_batch_uses_original_datavector_only():
    original = _datavector()

    actual = make_npe_training_batch(
        *original,
        use_rot90_counterpart=False,
    )

    assert len(actual) == len(original) == 5
    for actual_tensor, original_tensor in zip(actual, original):
        torch.testing.assert_close(actual_tensor, original_tensor)
