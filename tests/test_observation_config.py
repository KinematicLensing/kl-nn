import copy
import importlib.util
from pathlib import Path

import config
import pytest
from networks import DEFAULT_OBSERVATION_CONTEXT_FIELDS


def _training_entrypoint():
    path = Path(__file__).resolve().parents[1] / "arch" / "[scr]_train_model.py"
    spec = importlib.util.spec_from_file_location("observation_training_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archived_config_without_observation_block_defaults_to_legacy():
    payload = copy.deepcopy(config.MODEL_CONFIG.to_dict())
    payload.pop("observation")

    restored = config.ModelConfig.from_dict(payload)

    assert restored.observation == config.ObservationConfig()
    assert restored.observation.model_version == 1
    assert restored.observation.fiber_layout == "image_axis"
    assert restored.observation.halpha_flux_min * 1.0e16 == pytest.approx(1.2)
    assert restored.observation.halpha_flux_max * 1.0e16 == pytest.approx(301.43)
    assert restored.observation.halpha_flux_distribution == "uniform"
    assert restored.observation.halpha_flux_units == "erg s^-1 cm^-2"
    assert restored.observation.context_fields == list(
        DEFAULT_OBSERVATION_CONTEXT_FIELDS
    )


def test_observation_config_round_trips_and_syncs_legacy_global():
    original = copy.deepcopy(config.MODEL_CONFIG)
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.observation = config.ObservationConfig(
        model_version=2,
        rmag_min=16.0,
        rmag_max=23.0,
        halpha_flux_min=2.0e-16,
        halpha_flux_max=250.0e-16,
        halpha_flux_distribution="uniform",
        image_band="r",
        image_depth_5sigma_mag=23.5,
        spectral_quality_min=2.0,
        spectral_quality_max=80.0,
        spectral_quality_distribution="log_uniform",
        spectral_units="counts",
        center_fiber_index=2,
        center_exposure_s=200.0,
        offset_exposure_s=700.0,
    )

    restored = config.ModelConfig.from_dict(configured.to_dict())
    try:
        config.set_model_config(restored)
        assert restored.observation == configured.observation
        assert config.observation == configured.observation.to_dict()
    finally:
        config.set_model_config(original)


def test_v2_cli_overrides_round_trip_to_spawned_worker_config():
    module = _training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = module.parse_args(
            [
                "--train-type",
                "train",
                "--mode",
                "1",
                "--observation-model-version",
                "2",
                "--fiber-layout",
                "image_axis",
                "--rmag-min",
                "16.0",
                "--rmag-max",
                "23.0",
                "--halpha-flux-min",
                "2.0e-16",
                "--halpha-flux-max",
                "250.0e-16",
                "--halpha-flux-distribution",
                "uniform",
                "--image-band",
                "r",
                "--image-depth-5sigma-mag",
                "23.5",
                "--image-reference-psf-fwhm-arcsec",
                "0.9",
                "--image-pixel-scale-arcsec",
                "0.25",
                "--spectral-quality-min",
                "2.5",
                "--spectral-quality-max",
                "90.0",
                "--spectral-quality-distribution",
                "log_uniform",
                "--spectral-units",
                "counts",
                "--center-fiber-index",
                "2",
                "--center-exposure-s",
                "180",
                "--offset-exposure-s",
                "600",
            ]
        )
        stage = module.apply_overrides(args)
        restored = config.ModelConfig.from_dict(config.MODEL_CONFIG.to_dict())

        assert stage.mode == 1
        assert restored.observation == config.ObservationConfig(
            model_version=2,
            fiber_layout="image_axis",
            rmag_min=16.0,
            rmag_max=23.0,
            halpha_flux_min=2.0e-16,
            halpha_flux_max=250.0e-16,
            halpha_flux_distribution="uniform",
            image_band="r",
            image_depth_5sigma_mag=23.5,
            image_reference_psf_fwhm_arcsec=0.9,
            image_pixel_scale_arcsec=0.25,
            spectral_quality_min=2.5,
            spectral_quality_max=90.0,
            spectral_quality_distribution="log_uniform",
            spectral_units="counts",
            center_fiber_index=2,
            center_exposure_s=180.0,
            offset_exposure_s=600.0,
        )
    finally:
        config.set_model_config(original)


def test_v2_training_requires_unweighted_mode1_base_posterior():
    module = _training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = module.parse_args(
            [
                "--train-type",
                "train",
                "--mode",
                "2",
                "--observation-model-version",
                "2",
            ]
        )
        with pytest.raises(ValueError, match="broad base posterior.*mode 1"):
            module.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_pretraining_accepts_fixed_validation_streams():
    module = _training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = module.parse_args(
            ["--train-type", "pretrain", "--fixed-validation-streams"]
        )
        stage = module.apply_overrides(args)

        assert stage is config.MODEL_CONFIG.pretrain
        assert stage.fixed_validation_streams is True
        assert config.MODEL_CONFIG.train.fixed_validation_streams is False
    finally:
        config.set_model_config(original)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--rmag-min", "24", "--rmag-max", "23"], "rmag bounds"),
        (["--halpha-flux-min", "0"], "H-alpha flux bounds"),
        (
            [
                "--halpha-flux-min",
                "2e-14",
                "--halpha-flux-max",
                "1e-14",
            ],
            "H-alpha flux bounds",
        ),
        (
            ["--spectral-quality-min", "10", "--spectral-quality-max", "5"],
            "spectral-quality bounds",
        ),
        (["--center-fiber-index", "5"], "outside the configured fibers"),
        (["--center-exposure-s", "0"], "exposure times"),
        (
            ["--image-reference-psf-fwhm-arcsec", "0"],
            "image-reference-psf-fwhm",
        ),
        (["--image-pixel-scale-arcsec", "nan"], "image-pixel-scale"),
    ],
)
def test_invalid_observation_cli_ranges_are_rejected(options, message):
    module = _training_entrypoint()
    original = copy.deepcopy(config.MODEL_CONFIG)
    try:
        args = module.parse_args(["--train-type", "pretrain", *options])
        with pytest.raises(ValueError, match=message):
            module.apply_overrides(args)
    finally:
        config.set_model_config(original)


def test_unknown_fiber_layout_cli_value_is_rejected_by_parser():
    module = _training_entrypoint()
    with pytest.raises(SystemExit):
        module.parse_args(
            ["--train-type", "train", "--fiber-layout", "diagonal"]
        )
