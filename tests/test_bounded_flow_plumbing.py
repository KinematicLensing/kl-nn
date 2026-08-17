"""Configuration and launcher contracts for the bounded hybrid flow."""

import copy
import importlib.util
from pathlib import Path

import pytest

import config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_training_entrypoint():
    entrypoint_path = REPO_ROOT / "arch" / "[scr]_train_model.py"
    module_spec = importlib.util.spec_from_file_location(
        "bounded_flow_training_entrypoint",
        entrypoint_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _bounded_arguments(*extra):
    return [
        "--train-type",
        "train",
        "--backbone-type",
        "stage4_d4",
        "--posterior-symmetry",
        "d4",
        "--no-rot90-counterpart",
        "--flow-type",
        "bounded_hybrid_circular",
        *extra,
    ]


def test_archived_config_defaults_bounded_logit_limit_without_changing_flow():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    original_flow_type = payload["flow"]["flow_type"]
    payload["flow"].pop("bounded_logit_limit", None)

    restored = config.ModelConfig.from_dict(payload)

    assert restored.flow.bounded_logit_limit == 10.0
    assert restored.flow.flow_type == original_flow_type


def test_bounded_npe_cli_persists_compact_flow_and_stability_settings():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            _bounded_arguments(
                "--flow-num-layers",
                "4",
                "--flow-num-bins",
                "8",
                "--bounded-logit-limit",
                "9",
                "--theta-num-layers",
                "1",
                "--theta-logit-limit",
                "10",
                "--initial-learning-rate",
                "0.0003",
                "--affine-learning-rate",
                "0.0003",
                "--theta-learning-rate",
                "0.0001",
            )
        )

        stage = entrypoint.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert restored.flow.flow_type == "bounded_hybrid_circular"
        assert restored.flow.num_layers == 4
        assert restored.flow.num_bins == 8
        assert restored.flow.bounded_logit_limit == 9.0
        assert restored.flow.theta_num_layers == 1
        assert restored.flow.theta_logit_limit == 10.0
        assert stage.affine_learning_rate == 3e-4
        assert stage.theta_learning_rate == 1e-4
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize("value", ("0", "-1", "nan", "inf"))
def test_bounded_npe_cli_rejects_nonpositive_or_nonfinite_logit_limit(value):
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            _bounded_arguments("--bounded-logit-limit", value)
        )
        with pytest.raises(ValueError, match="positive and finite"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_bounded_npe_cli_rejects_zero_flow_layers():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            _bounded_arguments("--flow-num-layers", "0")
        )
        with pytest.raises(ValueError, match="flow-num-layers.*at least 1"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_bounded_options_are_npe_only():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            ["--train-type", "pretrain", "--bounded-logit-limit", "10"]
        )
        with pytest.raises(ValueError, match="only valid for NPE training"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_effective_run_logs_all_bounded_architecture_controls():
    source = (REPO_ROOT / "arch" / "[scr]_train_model.py").read_text()

    for fragment in (
        'f"flow_type={config.MODEL_CONFIG.flow.flow_type}, "',
        'f"flow_num_layers={config.MODEL_CONFIG.flow.num_layers}, "',
        'f"flow_num_bins={config.MODEL_CONFIG.flow.num_bins}, "',
        'f"bounded_logit_limit={config.MODEL_CONFIG.flow.bounded_logit_limit}, "',
        'f"theta_num_layers={config.MODEL_CONFIG.flow.theta_num_layers}, "',
        'f"theta_logit_limit={config.MODEL_CONFIG.flow.theta_logit_limit}, "',
    ):
        assert fragment in source


def test_stage6_bounded_launcher_records_the_controlled_run():
    launcher = (
        REPO_ROOT / "arch" / "train_npe_stage6_bounded_hybrid.slurm"
    ).read_text()

    expected_fragments = (
        '#SBATCH --gpus=v100-32:4',
        'EPOCHS="${EPOCHS:-20}"',
        'BATCH_SIZE="${BATCH_SIZE:-50}"',
        'MODE="${MODE:-2}"',
        'SEED="${SEED:-42}"',
        'FLOW_TYPE="${FLOW_TYPE:-bounded_hybrid_circular}"',
        'FLOW_NUM_LAYERS="${FLOW_NUM_LAYERS:-4}"',
        'FLOW_NUM_BINS="${FLOW_NUM_BINS:-8}"',
        'BOUNDED_LOGIT_LIMIT="${BOUNDED_LOGIT_LIMIT:-10}"',
        'THETA_NUM_LAYERS="${THETA_NUM_LAYERS:-1}"',
        'THETA_LOGIT_LIMIT="${THETA_LOGIT_LIMIT:-10}"',
        'INITIAL_LR="${INITIAL_LR:-0.0003}"',
        'AFFINE_LR="${AFFINE_LR:-0.0003}"',
        'THETA_LR="${THETA_LR:-0.0001}"',
        'SCHEDULER_TYPE="${SCHEDULER_TYPE:-warmup_cosine}"',
        'WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"',
        'MIN_LR="${MIN_LR:-0.00001}"',
        'TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m_fixedfiber/}"',
        'VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m_fixedfiber/}"',
        "--backbone-type stage4_d4",
        "--posterior-symmetry d4",
        "--no-rot90-counterpart",
        '--flow-type "${FLOW_TYPE}"',
        '--flow-num-layers "${FLOW_NUM_LAYERS}"',
        '--flow-num-bins "${FLOW_NUM_BINS}"',
        '--bounded-logit-limit "${BOUNDED_LOGIT_LIMIT}"',
        '--theta-num-layers "${THETA_NUM_LAYERS}"',
        '--theta-logit-limit "${THETA_LOGIT_LIMIT}"',
        '--affine-learning-rate "${AFFINE_LR}"',
        '--theta-learning-rate "${THETA_LR}"',
        '--fixed-validation-streams',
        '--no-context-norm-trainable',
        '--deterministic',
        '--no-compile',
    )
    for fragment in expected_fragments:
        assert fragment in launcher
