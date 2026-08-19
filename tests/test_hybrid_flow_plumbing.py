import copy
import importlib.util
from pathlib import Path

import pytest

import config


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_training_entrypoint():
    entrypoint_path = REPO_ROOT / "arch" / "[scr]_train_model.py"
    module_spec = importlib.util.spec_from_file_location(
        "hybrid_flow_training_entrypoint",
        entrypoint_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_archived_config_defaults_preserve_legacy_training_behavior():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    for name in (
        "scheduler_type",
        "warmup_epochs",
        "min_learning_rate",
        "fixed_validation_streams",
        "context_norm_trainable",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "gradient_clip_norm",
        "affine_learning_rate",
        "theta_learning_rate",
    ):
        payload["train"].pop(name, None)
    payload["flow"].pop("theta_num_layers", None)
    payload["flow"].pop("theta_logit_limit", None)

    restored = config.ModelConfig.from_dict(payload)

    assert restored.train.scheduler_type == "plateau"
    assert restored.train.warmup_epochs == 0
    assert restored.train.min_learning_rate == 1e-6
    assert restored.train.fixed_validation_streams is False
    assert restored.train.context_norm_trainable is True
    assert restored.train.early_stopping_patience is None
    assert restored.train.early_stopping_min_delta == 0.0
    assert restored.train.gradient_clip_norm == 1.0
    assert restored.train.affine_learning_rate is None
    assert restored.train.theta_learning_rate is None
    assert restored.flow.theta_num_layers == 1
    assert restored.flow.theta_logit_limit == 10.0


def test_hybrid_npe_cli_persists_flow_and_stability_settings():
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
                "hybrid_circular",
                "--flow-num-bins",
                "8",
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
                "--scheduler-type",
                "warmup_cosine",
                "--warmup-epochs",
                "2",
                "--min-learning-rate",
                "0.00001",
                "--fixed-validation-streams",
                "--no-context-norm-trainable",
                "--early-stopping-patience",
                "5",
                "--early-stopping-min-delta",
                "0.001",
                "--gradient-clip-norm",
                "1.0",
            ]
        )

        stage = entrypoint.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert restored.flow.flow_type == "hybrid_circular"
        assert restored.flow.num_bins == 8
        assert restored.flow.theta_num_layers == 1
        assert restored.flow.theta_logit_limit == 10.0
        assert stage.initial_learning_rate == 3e-4
        assert stage.affine_learning_rate == 3e-4
        assert stage.theta_learning_rate == 1e-4
        assert stage.scheduler_type == "warmup_cosine"
        assert stage.warmup_epochs == 2
        assert stage.min_learning_rate == 1e-5
        assert stage.fixed_validation_streams is True
        assert stage.context_norm_trainable is False
        assert stage.early_stopping_patience == 5
        assert stage.early_stopping_min_delta == 1e-3
        assert stage.gradient_clip_norm == 1.0
        assert restored.train.to_dict() == stage.to_dict()
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize(
    "arguments",
    (
        ("--theta-num-layers", "1"),
        ("--theta-logit-limit", "10"),
        ("--scheduler-type", "warmup_cosine"),
        ("--gradient-clip-norm", "1"),
    ),
)
def test_hybrid_and_stability_options_are_npe_only(arguments):
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(("--train-type", "pretrain", *arguments))
        with pytest.raises(ValueError, match="only valid for NPE training"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_fixed_validation_streams_are_available_to_pretraining():
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            ("--train-type", "pretrain", "--fixed-validation-streams")
        )
        stage = entrypoint.apply_overrides(args)
        assert stage.fixed_validation_streams is True
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize(
    "option",
    ("--affine-learning-rate", "--theta-learning-rate"),
)
def test_branch_learning_rates_require_hybrid_flow(option):
    entrypoint = _load_training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = entrypoint.parse_args(
            [
                "--train-type",
                "train",
                "--flow-type",
                "affine",
                option,
                "0.0001",
            ]
        )
        with pytest.raises(ValueError, match="hybrid_circular"):
            entrypoint.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_stage5_hybrid_launcher_records_the_controlled_run():
    launcher = (REPO_ROOT / "arch" / "train_npe_stage5_hybrid.slurm").read_text()

    expected_fragments = (
        '#SBATCH --gpus=v100-32:4',
        'EPOCHS="${EPOCHS:-20}"',
        'BATCH_SIZE="${BATCH_SIZE:-50}"',
        'MODE="${MODE:-2}"',
        'SEED="${SEED:-42}"',
        'FLOW_TYPE="${FLOW_TYPE:-hybrid_circular}"',
        'THETA_NUM_LAYERS="${THETA_NUM_LAYERS:-1}"',
        'THETA_LOGIT_LIMIT="${THETA_LOGIT_LIMIT:-10}"',
        'INITIAL_LR="${INITIAL_LR:-0.0003}"',
        'AFFINE_LR="${AFFINE_LR:-0.0003}"',
        'THETA_LR="${THETA_LR:-0.0001}"',
        'SCHEDULER_TYPE="${SCHEDULER_TYPE:-warmup_cosine}"',
        'WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"',
        'MIN_LR="${MIN_LR:-0.00001}"',
        'EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-5}"',
        'EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.001}"',
        'TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_1m_fixedfiber/}"',
        'VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_1m_fixedfiber/}"',
        "--backbone-type stage4_d4",
        "--posterior-symmetry d4",
        "--no-rot90-counterpart",
        '--flow-type "${FLOW_TYPE}"',
        '--theta-num-layers "${THETA_NUM_LAYERS}"',
        '--theta-logit-limit "${THETA_LOGIT_LIMIT}"',
        '--affine-learning-rate "${AFFINE_LR}"',
        '--theta-learning-rate "${THETA_LR}"',
        '--scheduler-type "${SCHEDULER_TYPE}"',
        '--fixed-validation-streams',
        '--no-context-norm-trainable',
        '--early-stopping-patience "${EARLY_STOPPING_PATIENCE}"',
        '--gradient-clip-norm "${GRADIENT_CLIP_NORM}"',
        '--deterministic',
        '--no-compile',
    )
    for fragment in expected_fragments:
        assert fragment in launcher
