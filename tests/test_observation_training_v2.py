import numpy as np
import pytest
import torch

import config
import train
from data import app_mag_to_snr
from networks import DEFAULT_OBSERVATION_CONTEXT_FIELDS


def test_v2_observation_levels_use_magnitude_forward_model_and_independent_stream():
    rmag = torch.tensor([17.0, 19.0, 21.0, 23.0], dtype=torch.float64)
    generator = torch.Generator().manual_seed(404)

    image_snr, spectral_quality = train.build_v2_observation_levels(
        rmag,
        image_depth_5sigma_mag=23.4,
        spectral_quality_min=3.0,
        spectral_quality_max=100.0,
        spectral_quality_distribution="log_uniform",
        spectral_generator=generator,
    )

    torch.testing.assert_close(
        image_snr,
        app_mag_to_snr(rmag, band="r", depth_5sigma_mag=23.4),
    )
    assert torch.all(image_snr[:-1] > image_snr[1:])

    unit_draw = torch.rand(
        rmag.shape,
        dtype=rmag.dtype,
        generator=torch.Generator().manual_seed(404),
    )
    expected_quality = 10 ** (
        unit_draw * (np.log10(100.0) - np.log10(3.0)) + np.log10(3.0)
    )
    torch.testing.assert_close(spectral_quality, expected_quality)
    assert torch.all((spectral_quality >= 3.0) & (spectral_quality <= 100.0))


def test_spectral_quality_draw_does_not_depend_on_true_magnitude():
    first_magnitudes = torch.tensor([16.0, 18.0, 20.0, 22.0])
    second_magnitudes = torch.tensor([22.5, 17.5, 21.5, 19.5])

    first_image_snr, first_quality = train.build_v2_observation_levels(
        first_magnitudes,
        spectral_generator=torch.Generator().manual_seed(919),
    )
    second_image_snr, second_quality = train.build_v2_observation_levels(
        second_magnitudes,
        spectral_generator=torch.Generator().manual_seed(919),
    )

    assert not torch.equal(first_image_snr, second_image_snr)
    torch.testing.assert_close(first_quality, second_quality)


def test_uniform_spectral_quality_draw_uses_requested_bounds():
    rmag = torch.full((32,), 20.0, dtype=torch.float64)
    _, quality = train.build_v2_observation_levels(
        rmag,
        spectral_quality_min=4.0,
        spectral_quality_max=12.0,
        spectral_quality_distribution="uniform",
        spectral_generator=torch.Generator().manual_seed(55),
    )
    expected = 4.0 + 8.0 * torch.rand(
        rmag.shape,
        dtype=rmag.dtype,
        generator=torch.Generator().manual_seed(55),
    )

    torch.testing.assert_close(quality, expected)


@pytest.mark.parametrize(
    ("rmag", "kwargs", "message"),
    [
        ([20.0, np.nan], {}, "rmag_true.*finite"),
        ([20.0], {"spectral_quality_min": 0.0}, "positive and increasing"),
        (
            [20.0],
            {"spectral_quality_min": 10.0, "spectral_quality_max": 10.0},
            "positive and increasing",
        ),
        (
            [20.0],
            {"spectral_quality_distribution": "normal"},
            "unsupported spectral-quality distribution",
        ),
    ],
)
def test_invalid_v2_observation_levels_are_rejected(rmag, kwargs, message):
    with pytest.raises(ValueError, match=message):
        train.build_v2_observation_levels(rmag, **kwargs)


def _metadata_trainer():
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.observation_model_version = 2
    trainer.expected_fiber_layout = "galaxy_axis"
    return trainer


def _valid_record():
    return {
        "rmag_true": torch.tensor(20.25),
        "halpha_flux_true": torch.tensor(4.2e-15),
        "observation_model_version": torch.tensor(2, dtype=torch.int16),
        "fiber_layout": torch.tensor(1, dtype=torch.int8),
        "image_band_code": torch.tensor(0, dtype=torch.int8),
        "target_line_code": torch.tensor(0, dtype=torch.int8),
        "spectral_units_code": torch.tensor(0, dtype=torch.int8),
        "center_fiber_index": torch.tensor(2, dtype=torch.int8),
        "center_exposure_s": torch.tensor(180.0),
        "offset_exposure_s": torch.tensor(600.0),
        "image_reference_psf_fwhm_arcsec": torch.tensor(1.0),
        "image_pixel_scale_arcsec": torch.tensor(0.2637),
    }


def test_v2_record_metadata_is_validated_before_training(monkeypatch):
    monkeypatch.setitem(config.observation, "rmag_min", 16.0)
    monkeypatch.setitem(config.observation, "rmag_max", 23.0)

    result = _metadata_trainer()._load_v2_record_metadata(
        _valid_record(), split="training", record_index=7
    )

    assert result == pytest.approx(20.25)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda record: record.pop("rmag_true"),
            "missing LMDB metadata.*rmag_true",
        ),
        (
            lambda record: record.pop("halpha_flux_true"),
            "missing LMDB metadata.*halpha_flux_true",
        ),
        (
            lambda record: record.__setitem__(
                "observation_model_version", torch.tensor(1)
            ),
            "does not match.*v1",
        ),
        (
            lambda record: record.__setitem__("fiber_layout", torch.tensor(0)),
            "fiber layout.*does not match",
        ),
        (
            lambda record: record.__setitem__("rmag_true", torch.tensor(np.nan)),
            "non-finite rmag_true",
        ),
        (
            lambda record: record.__setitem__(
                "halpha_flux_true", torch.tensor(np.nan)
            ),
            "non-finite halpha_flux_true",
        ),
        (
            lambda record: record.__setitem__("rmag_true", torch.tensor(24.0)),
            "outside configured",
        ),
        (
            lambda record: record.__setitem__(
                "halpha_flux_true", torch.tensor(4.0e-14)
            ),
            "halpha_flux_true=.*outside configured",
        ),
        (
            lambda record: record.__setitem__("fiber_layout", torch.tensor([1, 1])),
            "must be scalar",
        ),
        (
            lambda record: record.__setitem__(
                "image_reference_psf_fwhm_arcsec", torch.tensor(0.5)
            ),
            "image_reference_psf_fwhm_arcsec.*does not match",
        ),
    ],
)
def test_invalid_v2_record_metadata_is_rejected(monkeypatch, mutator, message):
    monkeypatch.setitem(config.observation, "rmag_min", 16.0)
    monkeypatch.setitem(config.observation, "rmag_max", 23.0)
    record = _valid_record()
    mutator(record)

    with pytest.raises(ValueError, match=message):
        _metadata_trainer()._load_v2_record_metadata(
            record, split="validation", record_index=3
        )


def _trainer_with_observed_arrays(version=2):
    trainer = train.NPETrainer.__new__(train.NPETrainer)
    trainer.observation_model_version = version
    for split, offset in (("train", 0.0), ("valid", 100.0)):
        setattr(trainer, f"rmag_{split}", torch.arange(4.0) + 900.0 + offset)
        setattr(trainer, f"RMAG_OBS_{split}", torch.arange(4.0) + 18.0 + offset)
        setattr(trainer, f"RMAG_SIGMA_{split}", torch.arange(4.0) + 0.1 + offset)
        setattr(trainer, f"SNR_{split}", torch.arange(4.0) + 5.0 + offset)
        setattr(
            trainer,
            f"IMAGE_SNR_OBS_{split}",
            torch.arange(4.0) + 4.5 + offset,
        )
        setattr(
            trainer,
            f"SPEC_QUALITY_{split}",
            torch.arange(4.0) + 10.0 + offset,
        )
        setattr(
            trainer,
            f"SPEC_NOISE_SCALE_{split}",
            torch.arange(4.0) + 0.01 + offset,
        )
    return trainer


def test_trainer_selects_only_observed_scalars_and_duplicates_them_in_lockstep():
    trainer = _trainer_with_observed_arrays()
    batch_ids = torch.tensor([3, 1])

    context = trainer._observation_context_for_batch(
        batch_ids, split="train", duplicate=False
    )
    duplicated = trainer._observation_context_for_batch(
        batch_ids, split="train", duplicate=True
    )

    assert tuple(context) == DEFAULT_OBSERVATION_CONTEXT_FIELDS
    assert "rmag_true" not in context
    assert "halpha_flux_true" not in context
    expected_sources = {
        "rmag_obs": trainer.RMAG_OBS_train,
        "rmag_sigma": trainer.RMAG_SIGMA_train,
        "image_snr": trainer.IMAGE_SNR_OBS_train,
        "spectral_reference_quality": trainer.SPEC_QUALITY_train,
        "spectral_noise_scale": trainer.SPEC_NOISE_SCALE_train,
    }
    for name, source in expected_sources.items():
        torch.testing.assert_close(context[name], source[batch_ids])
        torch.testing.assert_close(
            duplicated[name], torch.cat((source[batch_ids], source[batch_ids]))
        )

    nominal_snr = trainer.SNR_train[batch_ids]
    assert not torch.equal(context["image_snr"], nominal_snr)
    before = {name: value.clone() for name, value in context.items()}
    trainer.SNR_train.add_(10_000.0)
    after = trainer._observation_context_for_batch(
        batch_ids, split="train", duplicate=False
    )
    for name in DEFAULT_OBSERVATION_CONTEXT_FIELDS:
        torch.testing.assert_close(after[name], before[name])

    assert _trainer_with_observed_arrays(
        version=1
    )._observation_context_for_batch(batch_ids, split="train") is None


class _ContextRecordingNPE(torch.nn.Module):
    mode = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(
        self,
        image,
        spectra,
        targets,
        fp=None,
        observation_context=None,
    ):
        del spectra, targets, fp
        self.calls.append(observation_context)
        return self.weight.square() + 0.0 * image.sum()


class _DDPLikeWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_npe_run_batch_forwards_observation_context_in_train_and_validation():
    trainer = train.NPETrainer.__new__(train.NPETrainer)
    core = _ContextRecordingNPE()
    trainer.model = _DDPLikeWrapper(core)
    trainer.optimizer = torch.optim.SGD(core.parameters(), lr=0.01)
    trainer.device = torch.device("cpu")
    trainer.amp_dtype = torch.float16
    trainer.use_amp = False
    trainer.gradient_clip_norm = 1.0
    trainer.invalid_loss_count = 0
    trainer.invalid_gradient_count = 0
    trainer._preclip_grad_norm_sum = 0.0
    trainer._preclip_grad_norm_count = 0
    trainer._preclip_grad_norm_max = 0.0
    trainer._all_ranks_true = lambda value: bool(value)
    trainer._capture_training_diagnostics = lambda: None
    context = _trainer_with_observed_arrays()._observation_context_for_batch(
        torch.tensor([0, 2]), split="valid"
    )
    image = torch.zeros((2, 1, 6, 6))
    spectra = torch.zeros((2, 1, 5, 12))
    targets = torch.zeros((2, 8))
    positions = torch.zeros((2, 5, 2))

    validation_loss = trainer._run_batch(
        image,
        spectra,
        targets,
        "valid",
        fp=positions,
        observation_context=context,
    )
    training_loss = trainer._run_batch(
        image,
        spectra,
        targets,
        "train",
        fp=positions,
        observation_context=context,
    )

    assert torch.isfinite(validation_loss)
    assert torch.isfinite(training_loss)
    assert len(core.calls) == 2
    for forwarded in core.calls:
        assert forwarded is context
        assert "rmag_true" not in forwarded
        assert "halpha_flux_true" not in forwarded
