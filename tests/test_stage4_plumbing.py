import copy
import importlib.util
from pathlib import Path

import pytest
import torch

import config
from data import apply_d4_to_datavector
from train import make_ccl_pretrain_views


def _load_training_entrypoint():
    entrypoint_path = (
        Path(__file__).resolve().parents[1] / "arch" / "[scr]_train_model.py"
    )
    module_spec = importlib.util.spec_from_file_location(
        "stage4_training_entrypoint",
        entrypoint_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _datavector():
    image = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    spectra = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    labels = torch.tensor(
        [[0.10, -0.20, 0.75, -0.4, 0.3, -0.2, 0.1, 0.9]],
        dtype=torch.float32,
    )
    fiber_positions = torch.tensor(
        [
            [
                [2.0, 0.5],
                [-2.0, -0.5],
                [0.25, -0.1],
                [-0.5, 2.0],
                [0.5, -2.0],
            ]
        ],
        dtype=torch.float32,
    )
    return image, spectra, labels, fiber_positions


def _assert_datavectors_equal(actual, expected):
    assert len(actual) == len(expected) == 4
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_old_model_config_without_rot90_counterpart_defaults_true():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["pretrain"].pop("use_rot90_counterpart")

    restored = config.ModelConfig.from_dict(payload)

    assert restored.pretrain.use_rot90_counterpart is True


def test_stage4_cli_serializes_disabled_rot90_counterpart():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            [
                "--train-type",
                "pretrain",
                "--backbone-type",
                "stage4_d4",
                "--no-rot90-counterpart",
            ]
        )
        stage = entrypoint.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert stage.backbone_type == "stage4_d4"
        assert stage.use_rot90_counterpart is False
        assert restored.pretrain.backbone_type == "stage4_d4"
        assert restored.pretrain.use_rot90_counterpart is False
        assert restored.train.backbone_type == "stage4_d4"
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize(
    ("backbone_type", "rot90_flag"),
    (
        ("legacy", "--no-rot90-counterpart"),
        ("stage3", "--no-rot90-counterpart"),
        ("stage4_d4", "--rot90-counterpart"),
    ),
)
def test_cli_rejects_incompatible_rot90_counterpart_selector(
    backbone_type,
    rot90_flag,
):
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            [
                "--train-type",
                "pretrain",
                "--backbone-type",
                backbone_type,
                rot90_flag,
            ]
        )
        with pytest.raises(ValueError, match="rot90 counterpart"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_ccl_pretrain_views_without_counterpart_returns_only_original():
    original = _datavector()

    views = make_ccl_pretrain_views(
        img=original[0],
        spec=original[1],
        fid=original[2],
        fp=original[3],
        use_rot90_counterpart=False,
    )

    assert len(views) == 1
    _assert_datavectors_equal(views[0], original)


def test_ccl_pretrain_views_with_counterpart_uses_exact_canonical_r90():
    original = _datavector()
    expected_rotated = apply_d4_to_datavector(*original, element="r90")

    views = make_ccl_pretrain_views(
        img=original[0],
        spec=original[1],
        fid=original[2],
        fp=original[3],
        use_rot90_counterpart=True,
    )

    assert len(views) == 2
    _assert_datavectors_equal(views[0], original)
    _assert_datavectors_equal(views[1], expected_rotated)
