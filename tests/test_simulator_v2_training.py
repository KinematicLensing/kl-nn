import inspect

import pytest
import torch

import train
from data import app_mag_to_snr


def _v2_record(
    *,
    rmag_true=20.0,
    halpha_flux_true=4.2e-15,
    version=2,
    layout_code=1,
):
    return {
        "rmag_true": torch.tensor(rmag_true),
        "halpha_flux_true": torch.tensor(halpha_flux_true),
        "observation_model_version": torch.tensor(version),
        "fiber_layout": torch.tensor(layout_code),
        "image_band_code": torch.tensor(0),
        "target_line_code": torch.tensor(0),
        "spectral_units_code": torch.tensor(0),
        "center_fiber_index": torch.tensor(2),
        "center_exposure_s": torch.tensor(180.0),
        "offset_exposure_s": torch.tensor(600.0),
        "image_reference_psf_fwhm_arcsec": torch.tensor(1.0),
        "image_pixel_scale_arcsec": torch.tensor(0.2637),
    }


def _metadata_trainer():
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.observation_model_version = 2
    trainer.expected_fiber_layout = "galaxy_axis"
    return trainer


def test_v2_levels_are_forward_magnitude_snr_and_independent_spectral_draws():
    magnitudes = torch.linspace(16.0, 23.0, 128)
    reversed_magnitudes = torch.flip(magnitudes, dims=(0,))
    kwargs = {
        "image_band": "r",
        "image_depth_5sigma_mag": 23.4,
        "spectral_quality_min": 3.0,
        "spectral_quality_max": 100.0,
        "spectral_quality_distribution": "log_uniform",
    }

    image_snr, spectral_quality = train.build_v2_observation_levels(
        magnitudes,
        spectral_generator=torch.Generator().manual_seed(404),
        **kwargs,
    )
    repeated_snr, repeated_quality = train.build_v2_observation_levels(
        magnitudes,
        spectral_generator=torch.Generator().manual_seed(404),
        **kwargs,
    )
    reversed_snr, reversed_quality = train.build_v2_observation_levels(
        reversed_magnitudes,
        spectral_generator=torch.Generator().manual_seed(404),
        **kwargs,
    )

    torch.testing.assert_close(
        image_snr,
        app_mag_to_snr(magnitudes, band="r", depth_5sigma_mag=23.4),
    )
    torch.testing.assert_close(image_snr, repeated_snr)
    torch.testing.assert_close(spectral_quality, repeated_quality)
    torch.testing.assert_close(spectral_quality, reversed_quality)
    assert not torch.equal(image_snr, reversed_snr)
    assert bool((spectral_quality >= 3.0).all())
    assert bool((spectral_quality <= 100.0).all())


def test_v2_metadata_loader_accepts_matching_scalar_lmdb_metadata(monkeypatch):
    monkeypatch.setitem(train.config.observation, "rmag_min", 15.0)
    monkeypatch.setitem(train.config.observation, "rmag_max", 23.4)
    monkeypatch.setitem(train.config.observation, "halpha_flux_min", 1.2e-16)
    monkeypatch.setitem(train.config.observation, "halpha_flux_max", 301.43e-16)
    magnitude = _metadata_trainer()._load_v2_record_metadata(
        _v2_record(rmag_true=20.25), split="training", record_index=17
    )
    assert magnitude == pytest.approx(20.25)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"rmag_true": torch.tensor(20.0)}, "missing LMDB metadata"),
        (_v2_record(version=1), "does not match.*v1"),
        (_v2_record(layout_code=0), "fiber layout.*does not match"),
        (_v2_record(rmag_true=float("nan")), "non-finite rmag_true"),
        (
            _v2_record(halpha_flux_true=float("nan")),
            "non-finite halpha_flux_true",
        ),
        (_v2_record(rmag_true=24.0), "outside configured"),
        (
            _v2_record(halpha_flux_true=4.0e-14),
            "halpha_flux_true=.*outside configured",
        ),
    ],
)
def test_v2_metadata_loader_rejects_inconsistent_lmdb_metadata(
    monkeypatch, record, message
):
    monkeypatch.setitem(train.config.observation, "rmag_min", 15.0)
    monkeypatch.setitem(train.config.observation, "rmag_max", 23.4)
    monkeypatch.setitem(train.config.observation, "halpha_flux_min", 1.2e-16)
    monkeypatch.setitem(train.config.observation, "halpha_flux_max", 301.43e-16)
    with pytest.raises(ValueError, match=message):
        _metadata_trainer()._load_v2_record_metadata(
            record, split="validation", record_index=9
        )


def test_v2_spectrum_dispatch_uses_independent_quality_and_config(monkeypatch):
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.observation_model_version = 2
    trainer.spectral_reference_line_norm = torch.tensor(2.5)
    trainer.use_channels_last = False
    trainer.device = torch.device("cpu")
    clean = torch.zeros((3, 1, 5, 16))
    image_snr = torch.tensor([5.0, 10.0, 20.0])
    spectral_quality = torch.tensor([7.0, 11.0, 19.0])
    captured = {}

    def fake_spectral_noise(data, quality, reference, **kwargs):
        captured.update(
            data=data,
            quality=quality,
            reference=reference,
            kwargs=kwargs,
        )
        return data + 1.0

    monkeypatch.setattr(train, "apply_spectral_noise", fake_spectral_noise)
    result = trainer._apply_spectrum_noise(
        clean,
        image_snr,
        spectral_quality=spectral_quality,
        randgen=torch.Generator().manual_seed(2),
    )
    torch.testing.assert_close(result, clean + 1.0)
    torch.testing.assert_close(captured["quality"], spectral_quality)
    torch.testing.assert_close(captured["reference"], torch.tensor(2.5))
    assert captured["kwargs"]["center_fiber_index"] == train.config.observation[
        "center_fiber_index"
    ]
    assert captured["kwargs"]["spectral_units"] == train.config.observation[
        "spectral_units"
    ]


def test_v2_image_dispatch_uses_only_checkpointed_global_sigma(monkeypatch):
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.observation_model_version = 2
    trainer.image_noise_sigma = torch.tensor(0.125)
    trainer.use_channels_last = False
    trainer.device = torch.device("cpu")
    clean = torch.zeros((3, 1, 8, 8))
    captured = {}

    def fake_fixed_noise(data, sigma, **kwargs):
        captured.update(data=data, sigma=sigma, kwargs=kwargs)
        return data + 2.0

    monkeypatch.setattr(train, "apply_fixed_gaussian_image_noise", fake_fixed_noise)
    monkeypatch.setattr(
        train,
        "apply_noise",
        lambda *args, **kwargs: pytest.fail("v2 called segmented legacy noise"),
    )
    result = trainer._apply_noise(
        clean,
        torch.tensor([5.0, 50.0, 500.0]),
        maxs=torch.tensor([1.0, 100.0, 10_000.0]),
        randgen=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(result, clean + 2.0)
    torch.testing.assert_close(captured["sigma"], torch.tensor(0.125))
    assert "maxs" not in captured["kwargs"]


def test_missing_legacy_noise_max_cache_is_safe_for_v2_batches():
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.use_noise_cache_maxs = True
    batch_ids = torch.tensor([2, 0])

    assert trainer._noise_cache_for_batch(None, batch_ids) is None
    cache = torch.tensor([10.0, 20.0, 30.0])
    torch.testing.assert_close(
        trainer._noise_cache_for_batch(cache, batch_ids),
        torch.tensor([30.0, 10.0]),
    )

    trainer.use_noise_cache_maxs = False
    assert trainer._noise_cache_for_batch(cache, batch_ids) is None


def test_v1_spectrum_dispatch_preserves_legacy_shared_snr_path(monkeypatch):
    trainer = train.Trainer.__new__(train.Trainer)
    trainer.observation_model_version = 1
    clean = torch.zeros((2, 1, 5, 8))
    snr = torch.tensor([10.0, 30.0])
    sentinel = clean + 4.0
    captured = {}

    def fake_legacy_noise(data, passed_snr, **kwargs):
        captured.update(data=data, snr=passed_snr, kwargs=kwargs)
        return sentinel

    trainer._apply_noise = fake_legacy_noise
    monkeypatch.setattr(
        train,
        "apply_spectral_noise",
        lambda *args, **kwargs: pytest.fail("v1 called v2 spectral noise"),
    )
    result = trainer._apply_spectrum_noise(
        clean,
        snr,
        spectral_quality=None,
        maxs=torch.ones(2),
        randgen=torch.Generator().manual_seed(5),
    )
    assert result is sentinel
    torch.testing.assert_close(captured["snr"], snr)
    torch.testing.assert_close(captured["kwargs"]["maxs"], torch.ones(2))


@pytest.mark.parametrize(
    ("trainer_class", "method_name"),
    [
        (train.FETrainer, "_trainFunc"),
        (train.FETrainer, "_validFunc"),
        (train.NPETrainer, "_trainFunc"),
        (train.NPETrainer, "_validFunc"),
    ],
)
def test_every_training_loop_routes_spectra_through_versioned_dispatch(
    trainer_class, method_name
):
    source = inspect.getsource(getattr(trainer_class, method_name))
    assert "self._apply_spectrum_noise(" in source
    assert source.count("self._noise_cache_for_batch(") == 2


@pytest.mark.parametrize("method_name", ["_trainFunc", "_validFunc"])
def test_v2_npe_does_not_forward_snr_as_a_tf_training_weight(method_name):
    source = inspect.getsource(getattr(train.NPETrainer, method_name))
    assert "model_snr = None if self.observation_model_version == 2 else snr" in source
