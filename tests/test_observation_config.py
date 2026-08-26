import copy
import importlib.util
from pathlib import Path

import pytest

import config


def _training_entrypoint():
    path = Path(__file__).resolve().parents[1] / "arch" / "train_model.py"
    spec = importlib.util.spec_from_file_location("current_training_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _restore_model_config():
    original = copy.deepcopy(config.MODEL_CONFIG)
    yield
    config.set_model_config(original)


def test_current_config_json_roundtrip_is_exact(tmp_path):
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.data.size = 321
    configured.test.size = 23
    configured.pretrain.model_name = "current-ccl"
    configured.train.model_name = "current-npe"
    configured.train.non_theta_learning_rate = 2.0e-4
    configured.train.theta_learning_rate = 7.0e-5
    configured.flow.num_layers = 6
    configured.observation.image_snr_min = 6.0
    configured.observation.central_halpha_snr_min = 2.0

    path = tmp_path / "current.json"
    configured.to_json(str(path))
    restored = config.ModelConfig.from_json(str(path))

    assert restored.to_dict() == configured.to_dict()
    assert tuple(restored.train.feature_names) == config.TARGET_NAMES
    assert tuple(restored.par_ranges) == config.TARGET_NAMES
    assert tuple(restored.observation.context_fields) == config.ORACLE_CONTEXT_FIELDS


def test_current_target_ranges_and_transforms_are_immutable():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["par_ranges"]["vcirc"] = [50.0, 550.0]
    with pytest.raises(ValueError, match="immutable"):
        config.ModelConfig.from_dict(payload)

    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["observation"]["halpha_flux_max"] *= 0.9
    with pytest.raises(ValueError, match="H-alpha bounds"):
        config.ModelConfig.from_dict(payload)

    assert tuple(config.TARGET_TRANSFORMS) == config.TARGET_NAMES
    assert config.TARGET_TRANSFORMS["halpha_flux_true"] == "log10"
    assert set(config.TARGET_TRANSFORMS.values()) == {"identity", "log10"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rmag_min", 16.0),
        ("rmag_max", 23.0),
        ("image_reference_psf_fwhm_arcsec", 1.1),
        ("image_pixel_scale_arcsec", 0.3),
        ("center_fiber_index", 1),
        ("center_exposure_s", 200.0),
        ("offset_exposure_s", 500.0),
    ],
)
def test_generator_metadata_is_immutable(field, value):
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["observation"][field] = value
    with pytest.raises(ValueError, match="fixed by simulator schema v3"):
        config.ModelConfig.from_dict(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("observation"),
        lambda payload: payload.__setitem__("unexpected", True),
        lambda payload: payload["data"].pop("size"),
        lambda payload: payload["observation"].pop("schema_version"),
        lambda payload: payload["observation"].__setitem__("unexpected", True),
    ],
)
def test_missing_and_extra_current_config_fields_are_rejected(mutate):
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    mutate(payload)

    with pytest.raises((TypeError, ValueError)):
        config.ModelConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feature_names", list(reversed(config.TARGET_NAMES))),
        ("feature_names", list(config.TARGET_NAMES[:-1])),
    ],
)
def test_target_schema_must_be_exact_and_ordered(field, value):
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload["train"][field] = value

    with pytest.raises(ValueError, match="nine-target schema"):
        config.ModelConfig.from_dict(payload)


def test_pretrain_cli_applies_only_shared_and_ccl_overrides():
    module = _training_entrypoint()
    args = module.parse_args(
        [
            "--stage",
            "pretrain",
            "--train-data",
            "/tmp/current-train",
            "--valid-data",
            "/tmp/current-valid",
            "--train-size",
            "100",
            "--valid-size",
            "20",
            "--model-name",
            "current-ccl",
            "--model-root",
            "/tmp/current-shared/models",
            "--epochs",
            "9",
            "--batch-size",
            "7",
            "--seed",
            "31415",
            "--deterministic",
            "--no-compile",
            "--amp",
            "--amp-dtype",
            "float16",
            "--fixed-validation-streams",
            "--initial-learning-rate",
            "0.0004",
            "--weight-decay",
            "0.0002",
            "--ccl-sigma-label",
            "0.12",
            "--ccl-d-cutoff",
            "0.35",
        ]
    )

    stage = module.apply_overrides(args)
    spawned = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

    assert stage is config.MODEL_CONFIG.pretrain
    assert spawned.data.data_dir == "/tmp/current-train"
    assert spawned.test.data_dir == "/tmp/current-valid"
    assert spawned.data.size == 100
    assert spawned.test.size == 20
    assert spawned.pretrain.model_name == "current-ccl"
    assert spawned.pretrain.model_path == "/tmp/current-shared/models"
    assert spawned.train.model_path == "/tmp/current-shared/models"
    assert spawned.pretrain.epoch_number == 9
    assert spawned.pretrain.batch_size == 7
    assert spawned.pretrain.seed == 31415
    assert spawned.pretrain.deterministic is True
    assert spawned.pretrain.use_compile is False
    assert spawned.pretrain.use_amp is True
    assert spawned.pretrain.amp_dtype == "float16"
    assert spawned.pretrain.fixed_validation_streams is True
    assert spawned.pretrain.initial_learning_rate == pytest.approx(4.0e-4)
    assert spawned.pretrain.weight_decay == pytest.approx(2.0e-4)
    assert spawned.pretrain.ccl_sigma_label == pytest.approx(0.12)
    assert spawned.pretrain.ccl_d_cutoff == pytest.approx(0.35)


def test_npe_cli_applies_only_posterior_overrides():
    module = _training_entrypoint()
    args = module.parse_args(
        [
            "--stage",
            "npe",
            "--model-name",
            "current-npe",
            "--pretrained-name",
            "current-ccl",
            "--pretrain-from",
            "8",
            "--compile",
            "--amp",
            "--amp-dtype",
            "bfloat16",
            "--flow-num-layers",
            "7",
            "--flow-num-bins",
            "12",
            "--theta-num-layers",
            "2",
            "--theta-logit-limit",
            "8",
            "--bounded-logit-limit",
            "9",
            "--non-theta-learning-rate",
            "0.0002",
            "--theta-learning-rate",
            "0.00005",
            "--scheduler-type",
            "cosine",
            "--warmup-epochs",
            "3",
            "--min-learning-rate",
            "0.000001",
            "--no-feature-norm-trainable",
            "--early-stopping-patience",
            "6",
            "--early-stopping-min-delta",
            "0.0001",
            "--gradient-clip-norm",
            "2.5",
        ]
    )

    stage = module.apply_overrides(args)
    spawned = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

    assert stage is config.MODEL_CONFIG.train
    assert spawned.train.model_name == "current-npe"
    assert spawned.train.pretrained_name == "current-ccl"
    assert spawned.train.pretrain_from == 8
    assert spawned.train.use_compile is True
    assert spawned.train.use_amp is True
    assert spawned.train.amp_dtype == "bfloat16"
    assert spawned.flow.num_layers == 7
    assert spawned.flow.num_bins == 12
    assert spawned.flow.theta_num_layers == 2
    assert spawned.flow.theta_logit_limit == pytest.approx(8.0)
    assert spawned.flow.bounded_logit_limit == pytest.approx(9.0)
    assert spawned.train.non_theta_learning_rate == pytest.approx(2.0e-4)
    assert spawned.train.theta_learning_rate == pytest.approx(5.0e-5)
    assert spawned.train.scheduler_type == "cosine"
    assert spawned.train.warmup_epochs == 3
    assert spawned.train.min_learning_rate == pytest.approx(1.0e-6)
    assert spawned.train.feature_norm_trainable is False
    assert spawned.train.early_stopping_patience == 6
    assert spawned.train.early_stopping_min_delta == pytest.approx(1.0e-4)
    assert spawned.train.gradient_clip_norm == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--stage", "pretrain", "--flow-num-layers", "2"], "NPE-only"),
        (["--stage", "npe", "--ccl-sigma-label", "0.2"], "CCL options"),
        (["--stage", "pretrain", "--batch-size", "0"], "batch size"),
        (["--stage", "npe", "--flow-num-bins", "1"], "bins at least two"),
        (["--stage", "npe", "--theta-learning-rate", "0"], "theta_learning_rate"),
        (["--stage", "npe", "--pretrain-from", "-1"], "pretrain-from"),
        (["--stage", "npe", "--pretrain-from", "last"], "pretrain-from"),
    ],
)
def test_cli_rejects_cross_stage_and_invalid_values(argv, message):
    module = _training_entrypoint()
    args = module.parse_args(argv)

    with pytest.raises(ValueError, match=message):
        module.apply_overrides(args)


def test_unknown_cli_flags_are_rejected_without_abbreviation():
    module = _training_entrypoint()

    with pytest.raises(SystemExit):
        module.parse_args(["--stage", "npe", "--unknown-option", "value"])


def test_best_pretraining_checkpoint_is_the_current_default_and_cli_value():
    assert config.MODEL_CONFIG.train.pretrain_from == "best"
    module = _training_entrypoint()
    stage = module.apply_overrides(
        module.parse_args(["--stage", "npe", "--pretrain-from", "best"])
    )
    assert stage.pretrain_from == "best"
