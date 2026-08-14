import copy
import importlib.util
from pathlib import Path

import pytest

import config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_training_entrypoint():
    entrypoint_path = REPO_ROOT / "arch" / "[scr]_train_model.py"
    module_spec = importlib.util.spec_from_file_location(
        "circular_flow_training_entrypoint",
        entrypoint_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_old_model_config_without_flow_selector_defaults_to_affine():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["flow"].pop("flow_type", None)
    payload["flow"].pop("num_bins", None)

    restored = config.ModelConfig.from_dict(payload)

    assert restored.flow.flow_type == "affine"
    assert restored.flow.num_bins == 8


def test_npe_cli_persists_circular_flow_settings():
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
    finally:
        config.set_model_config(original)


def test_flow_options_are_npe_only():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            ["--train-type", "pretrain", "--flow-type", "circular_rqs"]
        )
        with pytest.raises(ValueError, match="only valid for NPE training"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_npe_cli_rejects_too_few_spline_bins():
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
                "--flow-type",
                "circular_rqs",
                "--flow-num-bins",
                "1",
            ]
        )
        with pytest.raises(ValueError, match="at least 2"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_stage5_launcher_is_a_controlled_circular_flow_run():
    launcher = (REPO_ROOT / "arch" / "train_npe_stage5_circular.slurm").read_text()

    expected_fragments = (
        '#SBATCH --time=2:00:00',
        'EPOCHS="${EPOCHS:-20}"',
        'BATCH_SIZE="${BATCH_SIZE:-50}"',
        'MODE="${MODE:-2}"',
        'SEED="${SEED:-42}"',
        'FLOW_TYPE="${FLOW_TYPE:-circular_rqs}"',
        'FLOW_NUM_BINS="${FLOW_NUM_BINS:-8}"',
        'PRETRAINED_NAME="${PRETRAINED_NAME:-CNN-SetAttn-D4_CCL_stage4_fixedgeo_s42_43420237}"',
        'PRETRAIN_FROM="${PRETRAIN_FROM:-14}"',
        '--backbone-type stage4_d4',
        '--posterior-symmetry d4',
        '--no-rot90-counterpart',
        '--flow-type "${FLOW_TYPE}"',
        '--flow-num-bins "${FLOW_NUM_BINS}"',
        '--deterministic',
        '--no-compile',
        'tf_analysis_stage4_mode1_10k.slurm',
    )
    for fragment in expected_fragments:
        assert fragment in launcher
