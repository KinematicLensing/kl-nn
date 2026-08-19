import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import config
from data import app_mag_to_snr
from networks import DEFAULT_OBSERVATION_CONTEXT_FIELDS
from train import sample_density


FEATURE_NAMES = (
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_analysis_module():
    path = REPO_ROOT / "arch" / "[scr]_tf_analysis.py"
    spec = importlib.util.spec_from_file_location("tf_analysis_v2_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_simulator_v2_launcher_uses_safe_mode1_unpaired_defaults():
    launcher_path = (
        REPO_ROOT / "arch" / "tf_analysis_simulator_v2_10k.slurm"
    )
    launcher = launcher_path.read_text()
    assert launcher_path.stat().st_mode & 0o111

    assert "#SBATCH --array=1-10" in launcher
    assert 'MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME' in launcher
    assert 'DATASET="${DATASET:-small_1m_simv2_galaxyaxis_halpha}"' in launcher
    assert (
        'SAMPLE_SET="${SAMPLE_SET:-samples_small_1m_simv2_galaxyaxis_halpha.csv}"'
        in launcher
    )
    assert (
        'SAMPLE_SET="${SAMPLE_SET:-samples_small_1m_simv2_galaxyaxis_halpha.csv}"'
        in launcher
    )
    assert 'TF_INFERENCE="${TF_INFERENCE:-prior_replacement}"' in launcher
    assert 'NGALS="${NGALS:-1000}"' in launcher
    assert 'NSAMPLES="${NSAMPLES:-5000}"' in launcher
    assert 'NPARTS="${NPARTS:-10}"' in launcher
    assert "--mode 1" in launcher
    assert '--tf-inference "${TF_INFERENCE}"' in launcher
    assert "--matched-group-size 1" in launcher
    assert "--no-conform-to-tf" in launcher
    assert "--no-cancel-add-noise" in launcher
    assert 1000 * 10 == 10_000
    assert 5000 % 8 == 0


def _v2_config():
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.observation.model_version = 2
    configured.observation.fiber_layout = "galaxy_axis"
    configured.train.mode = 1
    configured.train.feature_number = len(FEATURE_NAMES)
    configured.train.feature_names = list(FEATURE_NAMES)
    configured.train.posterior_symmetry = "none"
    configured.train.use_compile = False
    configured.train.use_amp = False
    configured.train.channels_last = False
    return configured


class _TinyV2Dataset:
    def __init__(
        self,
        size=2,
        *,
        rmag_values=None,
        halpha_flux_values=None,
        matched_clean_group_size=1,
    ):
        if rmag_values is None:
            rmag_values = [19.0 + index for index in range(size)]
        if halpha_flux_values is None:
            halpha_flux_values = [4.2e-15] * size
        if len(rmag_values) != size:
            raise ValueError("rmag_values must have one entry per record")
        if len(halpha_flux_values) != size:
            raise ValueError("halpha_flux_values must have one entry per record")
        self.records = []
        for index in range(size):
            clean_index = index // matched_clean_group_size
            self.records.append(
                {
                    "img": torch.ones((1, 6, 6)) * (clean_index + 1),
                    "spec": torch.ones((1, 5, 12)) * clean_index,
                    "fid_pars": torch.zeros(8),
                    "fib_pos": torch.zeros((5, 2)),
                    "rmag_true": torch.tensor(rmag_values[index]),
                    "halpha_flux_true": torch.tensor(
                        halpha_flux_values[index]
                    ),
                    "observation_model_version": torch.tensor(2),
                    "fiber_layout": torch.tensor(1),
                    "image_band_code": torch.tensor(0),
                    "target_line_code": torch.tensor(0),
                    "spectral_units_code": torch.tensor(0),
                    "center_fiber_index": torch.tensor(2),
                    "center_exposure_s": torch.tensor(180.0),
                    "offset_exposure_s": torch.tensor(600.0),
                    "image_reference_psf_fwhm_arcsec": torch.tensor(1.0),
                    "image_pixel_scale_arcsec": torch.tensor(0.2637),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class _RecordingV2Posterior:
    observation_model_version = 2
    posterior_symmetry = "none"

    def __init__(
        self,
        *,
        mode=1,
        reference_line_norm=2.5,
        image_noise_sigma=0.125,
    ):
        self.mode = mode
        self.spectral_reference_line_norm = torch.tensor(reference_line_norm)
        self.image_noise_sigma = torch.tensor(image_noise_sigma)
        self.calls = []

    def eval(self):
        return self

    def sample(
        self,
        image,
        spectra,
        nsamples,
        fp=None,
        return_log_prob=False,
        **kwargs,
    ):
        self.calls.append(
            {
                "image": image.detach().clone(),
                "spectra": spectra.detach().clone(),
                "fp": fp,
                **kwargs,
            }
        )
        samples = torch.zeros((1, nsamples, len(FEATURE_NAMES)))
        if kwargs.get("tf_inference") == "prior_replacement":
            sample_id = float(kwargs.get("sample_id", 0))
            self.last_tf_inference_diagnostics = {
                "effective_sample_size": torch.tensor([3.0 + sample_id]),
                "effective_sample_fraction": torch.tensor([0.75]),
                "max_normalized_weight": torch.tensor([0.4]),
                "candidate_log_normalizer": torch.tensor([-1.2 + sample_id]),
            }
        if return_log_prob:
            return samples, torch.linspace(-1.0, 0.0, nsamples)
        return samples


def test_analysis_stream_seeds_are_documented_reproducible_and_independent():
    module = _load_analysis_module()

    first = module.analysis_stream_seeds(9123)
    repeated = module.analysis_stream_seeds(9123)

    assert tuple(first) == (
        "image_noise",
        "spectral_noise",
        "magnitude_observation",
        "spectral_quality",
    )
    assert first == repeated
    assert len(set(first.values())) == len(first)
    assert first == {
        name: 9123 + offset
        for name, offset in module.ANALYSIS_STREAM_OFFSETS.items()
    }


@pytest.mark.parametrize(
    ("cli", "message"),
    [
        (["--mode", "2"], "mode-1 base posterior"),
        (
            ["--mode", "1", "--conform-to-tf"],
            "forbids --conform-to-tf",
        ),
        (
            ["--mode", "1", "--cached-snrs-path", "legacy.npy"],
            "does not accept --cached-snrs-path",
        ),
    ],
)
def test_v2_analysis_cli_rejects_legacy_choices(
    monkeypatch, cli, message
):
    module = _load_analysis_module()
    monkeypatch.setattr(sys, "argv", ["tf_analysis.py", *cli])
    args = module.parse_args()

    with pytest.raises(ValueError, match=message):
        module.validate_analysis_observation_args(
            args,
            {"model_version": 2},
            train_mode=1,
        )


def test_v2_analysis_cli_accepts_matched_observation_groups(monkeypatch):
    module = _load_analysis_module()
    monkeypatch.setattr(
        sys,
        "argv",
        ["tf_analysis.py", "--mode", "1", "--matched-group-size", "2"],
    )
    args = module.parse_args()

    version, tf_inference = module.validate_analysis_observation_args(
        args,
        {"model_version": 2},
        train_mode=1,
    )

    assert version == 2
    assert tf_inference is None
    assert args.matched_group_size == 2


def test_tf_prior_replacement_cli_is_v2_only(monkeypatch):
    module = _load_analysis_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tf_analysis.py",
            "--mode",
            "1",
            "--tf-inference",
            "prior_replacement",
        ],
    )
    args = module.parse_args()

    version, tf_inference = module.validate_analysis_observation_args(
        args, {"model_version": 2}, train_mode=1
    )
    assert version == 2
    assert tf_inference == "prior_replacement"
    with pytest.raises(ValueError, match="requires observation model v2"):
        module.validate_analysis_observation_args(
            args, {"model_version": 1}, train_mode=1
        )


def test_v2_observation_provenance_is_complete_and_json_safe():
    module = _load_analysis_module()
    observation = _v2_config().observation.to_dict()
    stream_seeds = module.analysis_stream_seeds(27)

    provenance = module.build_observation_provenance(
        observation,
        observation_model_version=2,
        checkpoint_image_noise_sigma=0.125,
        checkpoint_reference_line_norm=2.75,
        stream_seeds=stream_seeds,
    )

    assert provenance == {
        "model_version": 2,
        "fiber_layout": "galaxy_axis",
        "metadata_source": (
            "validated LMDB v2 observation and instrument schema"
        ),
        "context_fields": list(DEFAULT_OBSERVATION_CONTEXT_FIELDS),
        "image_band": observation["image_band"],
        "target_line": observation["target_line"],
        "halpha_flux_min": observation["halpha_flux_min"],
        "halpha_flux_max": observation["halpha_flux_max"],
        "halpha_flux_distribution": observation["halpha_flux_distribution"],
        "halpha_flux_units": observation["halpha_flux_units"],
        "image_depth_5sigma_mag": observation["image_depth_5sigma_mag"],
        "image_reference_psf_fwhm_arcsec": observation.get(
            "image_reference_psf_fwhm_arcsec"
        ),
        "image_pixel_scale_arcsec": observation.get(
            "image_pixel_scale_arcsec"
        ),
        "image_depth_calibration": "Gaussian-PSF-equivalent",
        "image_reference_noise_equivalent_pixels": (
            module.gaussian_psf_noise_equivalent_pixels(
                observation["image_reference_psf_fwhm_arcsec"],
                observation["image_pixel_scale_arcsec"],
            )
        ),
        "checkpoint_image_noise_sigma": 0.125,
        "spectral_quality_min": observation["spectral_quality_min"],
        "spectral_quality_max": observation["spectral_quality_max"],
        "spectral_quality_distribution": observation[
            "spectral_quality_distribution"
        ],
        "spectral_units": observation["spectral_units"],
        "center_fiber_index": observation["center_fiber_index"],
        "center_exposure_s": observation["center_exposure_s"],
        "offset_exposure_s": observation["offset_exposure_s"],
        "snr_cache_semantics": "observed_catalog_flux_snr",
        "pixel_noise_target_snr_cached": False,
        "checkpoint_spectral_reference_line_norm": 2.75,
        "rng_stream_seeds": stream_seeds,
        "tf_prior_replacement_diagnostics": {},
    }
    json.dumps(provenance)
    assert "rmag_true" not in provenance["context_fields"]


def test_analysis_snr_cache_is_version_aware_and_never_exposes_v2_target():
    module = _load_analysis_module()
    latent_target = np.asarray([10.0, 20.0])
    observed = np.asarray([9.25, 21.5])

    np.testing.assert_array_equal(
        module.resolve_analysis_snr_cache(
            latent_target,
            {"image_snr": observed},
            observation_model_version=2,
        ),
        observed,
    )
    np.testing.assert_array_equal(
        module.resolve_analysis_snr_cache(
            latent_target,
            None,
            observation_model_version=1,
        ),
        latent_target,
    )
    with pytest.raises(RuntimeError, match="observed image_snr"):
        module.resolve_analysis_snr_cache(
            latent_target,
            {},
            observation_model_version=2,
        )


def test_tf_diagnostic_summary_is_finite_and_json_safe():
    module = _load_analysis_module()
    metadata = {
        "tf_effective_sample_size": np.asarray([2.0, 4.0, 8.0]),
        "tf_effective_sample_fraction": np.asarray([0.2, 0.4, 0.8]),
        "tf_max_normalized_weight": np.asarray([0.5, 0.25, 0.125]),
        "tf_candidate_log_normalizer": np.asarray([-3.0, -2.0, -1.0]),
    }

    summary = module.summarize_tf_diagnostic_arrays(metadata)

    assert summary["tf_effective_sample_size"] == {
        "min": 2.0,
        "median": 4.0,
        "mean": pytest.approx(14.0 / 3.0),
        "max": 8.0,
    }
    assert set(summary) == set(module.TF_DIAGNOSTIC_ARRAY_TYPES)
    json.dumps(summary)


def test_sample_density_v2_rejects_mode2():
    dataset = _TinyV2Dataset(size=2)
    with pytest.raises(ValueError, match="mode-1 base posterior"):
        sample_density(
            _RecordingV2Posterior(mode=2),
            dataset,
            3,
            matched_group_size=1,
        )


def test_sample_density_v2_pairs_all_observation_streams_within_groups():
    original = copy.deepcopy(config.MODEL_CONFIG)
    config.set_model_config(_v2_config())
    try:
        dataset = _TinyV2Dataset(
            size=4,
            rmag_values=[19.0, 19.0, 20.0, 20.0],
            halpha_flux_values=[3.0e-15, 3.0e-15, 5.0e-15, 5.0e-15],
            matched_clean_group_size=2,
        )
        model = _RecordingV2Posterior(mode=1, reference_line_norm=2.5)
        samples, nominal_image_snr, metadata = sample_density(
            model,
            dataset,
            3,
            matched_group_size=2,
            noise_seed=101,
            spectral_noise_seed=211,
            magnitude_seed=307,
            spectral_quality_seed=401,
            return_observation_metadata=True,
            channels_last=False,
        )
    finally:
        config.set_model_config(original)

    assert samples.shape == (4, 3, len(FEATURE_NAMES))
    assert len(model.calls) == 4
    np.testing.assert_allclose(nominal_image_snr[[0, 2]], [
        nominal_image_snr[1],
        nominal_image_snr[3],
    ])
    for name in (
        "image_snr",
        "spectral_quality",
        "spectral_noise_scale",
        "rmag_obs",
        "rmag_sigma",
    ):
        values = np.asarray(metadata[name])
        np.testing.assert_allclose(values[[0, 2]], values[[1, 3]])

    for left, right in ((0, 1), (2, 3)):
        torch.testing.assert_close(
            model.calls[left]["image"], model.calls[right]["image"]
        )
        torch.testing.assert_close(
            model.calls[left]["spectra"], model.calls[right]["spectra"]
        )
        for field in DEFAULT_OBSERVATION_CONTEXT_FIELDS:
            torch.testing.assert_close(
                model.calls[left]["observation_context"][field],
                model.calls[right]["observation_context"][field],
            )


def test_sample_density_v2_rejects_mismatched_group_metadata():
    original = copy.deepcopy(config.MODEL_CONFIG)
    config.set_model_config(_v2_config())
    try:
        mismatched_rmag = _TinyV2Dataset(
            size=2,
            rmag_values=[19.0, 19.1],
            matched_clean_group_size=2,
        )
        with pytest.raises(
            ValueError, match="must share the same archived rmag_true"
        ):
            sample_density(
                _RecordingV2Posterior(mode=1),
                mismatched_rmag,
                3,
                matched_group_size=2,
            )

        mismatched_halpha = _TinyV2Dataset(
            size=2,
            rmag_values=[19.0, 19.0],
            halpha_flux_values=[3.0e-15, 4.0e-15],
            matched_clean_group_size=2,
        )
        with pytest.raises(
            ValueError, match="must share the same archived halpha_flux_true"
        ):
            sample_density(
                _RecordingV2Posterior(mode=1),
                mismatched_halpha,
                3,
                matched_group_size=2,
            )

        matched_rmag = _TinyV2Dataset(
            size=2,
            rmag_values=[19.0, 19.0],
            matched_clean_group_size=2,
        )
        with pytest.raises(ValueError, match="must share one spectral_quality"):
            sample_density(
                _RecordingV2Posterior(mode=1),
                matched_rmag,
                3,
                matched_group_size=2,
                spectral_quality=torch.tensor([10.0, 11.0]),
            )
    finally:
        config.set_model_config(original)


def test_sample_density_v2_requires_complete_matched_groups():
    with pytest.raises(ValueError, match="divide the dataset size"):
        sample_density(
            _RecordingV2Posterior(mode=1),
            _TinyV2Dataset(size=3),
            3,
            matched_group_size=2,
        )


def test_sample_density_v2_returns_observed_metadata_and_no_latent_context():
    original = copy.deepcopy(config.MODEL_CONFIG)
    configured = _v2_config()
    config.set_model_config(configured)
    try:
        dataset = _TinyV2Dataset(size=2)
        model = _RecordingV2Posterior(reference_line_norm=2.5)
        rmag_true = torch.tensor([19.0, 20.0])
        expected_snr = app_mag_to_snr(
            rmag_true,
            band=config.observation["image_band"],
            depth_5sigma_mag=config.observation["image_depth_5sigma_mag"],
        )
        samples, log_prob, image_snr, metadata = sample_density(
            model,
            dataset,
            4,
            snr=expected_snr,
            rmag_true=rmag_true,
            tf_inference="prior_replacement",
            image_randgen=torch.Generator().manual_seed(101),
            spectral_randgen=torch.Generator().manual_seed(211),
            magnitude_randgen=torch.Generator().manual_seed(307),
            spectral_quality_randgen=torch.Generator().manual_seed(401),
            return_log_prob=True,
            return_observation_metadata=True,
            channels_last=False,
        )
    finally:
        config.set_model_config(original)

    assert samples.shape == (2, 4, len(FEATURE_NAMES))
    assert log_prob.shape == (2, 4)
    np.testing.assert_allclose(image_snr, expected_snr.numpy())
    assert {
        "image_snr",
        "spectral_quality",
        "spectral_noise_scale",
        "rmag_obs",
        "rmag_sigma",
        "spectral_reference_line_norm",
    }.issubset(metadata)
    assert set(metadata) >= set(
        _load_analysis_module().TF_DIAGNOSTIC_ARRAY_TYPES
    )
    assert metadata["spectral_reference_line_norm"] == pytest.approx(2.5)
    np.testing.assert_allclose(
        metadata["spectral_noise_scale"],
        2.5 / metadata["spectral_quality"],
    )
    assert not np.array_equal(metadata["image_snr"], image_snr)
    np.testing.assert_allclose(
        metadata["tf_effective_sample_size"], [3.0, 4.0]
    )
    np.testing.assert_allclose(
        metadata["tf_effective_sample_fraction"], [0.75, 0.75]
    )
    assert len(model.calls) == len(dataset)
    for index, call in enumerate(model.calls):
        context = call["observation_context"]
        assert tuple(context) == DEFAULT_OBSERVATION_CONTEXT_FIELDS
        assert "rmag_true" not in context
        assert "halpha_flux_true" not in context
        assert call["snr"] is None
        assert call["tf_inference"] == "prior_replacement"
        torch.testing.assert_close(call["mag"], context["rmag_obs"])
        torch.testing.assert_close(call["mag_sigma"], context["rmag_sigma"])
        assert float(context["image_snr"]) == pytest.approx(
            metadata["image_snr"][index]
        )


def test_tf_analysis_main_writes_v2_manifest_and_metadata_files(
    tmp_path, monkeypatch
):
    module = _load_analysis_module()
    configured = _v2_config()
    original = copy.deepcopy(config.MODEL_CONFIG)
    shared = tmp_path / "shared"
    dataset_path = shared / "datasets" / "tiny_v2"
    sample_path = shared / "samples" / "tiny.csv"
    dataset_path.mkdir(parents=True)
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text("ID\n0\n")
    dataset = _TinyV2Dataset(size=1)
    model = _RecordingV2Posterior(reference_line_norm=2.75)
    captured = {}

    def fake_sample_density(posterior, subset, nsamples, **kwargs):
        captured.update(kwargs)
        assert posterior is model
        assert len(subset) == 1
        metadata = {
            "image_snr": np.asarray([17.0], dtype=np.float32),
            "spectral_quality": np.asarray([23.0], dtype=np.float32),
            "spectral_noise_scale": np.asarray([2.75 / 23.0], dtype=np.float32),
            "rmag_obs": np.asarray([19.1], dtype=np.float32),
            "rmag_sigma": np.asarray([0.08], dtype=np.float32),
            "spectral_reference_line_norm": 2.75,
            "tf_effective_sample_size": np.asarray([3.0], dtype=np.float64),
            "tf_effective_sample_fraction": np.asarray([0.75], dtype=np.float64),
            "tf_max_normalized_weight": np.asarray([0.4], dtype=np.float64),
            "tf_candidate_log_normalizer": np.asarray([-1.2], dtype=np.float64),
        }
        return (
            np.zeros((1, nsamples, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((1, nsamples), dtype=np.float32),
            np.asarray([999.0], dtype=np.float32),
            metadata,
        )

    monkeypatch.setattr(module, "BASE_SHARED_DIR", str(shared))
    monkeypatch.setattr(module, "BASE_DATASETS_DIR", str(shared / "datasets"))
    monkeypatch.setattr(module, "BASE_SAMPLES_DIR", str(shared / "samples"))
    monkeypatch.setattr(module, "load_model_config", lambda *args, **kwargs: configured)
    monkeypatch.setattr(module, "load_model", lambda **kwargs: model)
    monkeypatch.setattr(module, "sample_density", fake_sample_density)
    monkeypatch.setattr(module.pxt, "TorchDataset", lambda path: dataset)
    monkeypatch.setattr(module, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(module, "now_utc_iso", lambda: "2026-08-17T00:00:00+00:00")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tf_analysis.py",
            "--ngals",
            "1",
            "--nsamples",
            "3",
            "--nparts",
            "1",
            "--stem",
            "v2_model",
            "--epoch",
            "4",
            "--dataset",
            "tiny_v2",
            "--sample-set",
            "tiny.csv",
            "--mode",
            "1",
            "--tf-inference",
            "prior_replacement",
            "--network-source",
            "current",
            "--cache-tag",
            "probe",
            "--seed",
            "123",
            "--no-compile",
            "--no-amp",
            "--no-channels-last",
            "--no-inference-mode",
        ],
    )
    try:
        module.main()
    finally:
        config.set_model_config(original)

    result_root = (
        shared
        / "cache"
        / "v2_model"
        / "tiny_v2_tf_prior_replacement_probe"
    )
    manifest_path = result_root / "meta" / "part0of1.json"
    manifest = json.loads(manifest_path.read_text())
    expected_streams = module.analysis_stream_seeds(123)

    assert manifest["observation"]["model_version"] == 2
    assert manifest["observation"]["fiber_layout"] == "galaxy_axis"
    assert manifest["observation"]["halpha_flux_min"] == pytest.approx(
        1.2e-16,
        rel=1.0e-12,
        abs=0.0,
    )
    assert manifest["observation"]["halpha_flux_max"] == pytest.approx(
        301.43e-16,
        rel=1.0e-12,
        abs=0.0,
    )
    assert manifest["observation"]["halpha_flux_distribution"] == "uniform"
    assert manifest["observation"]["halpha_flux_units"] == "erg s^-1 cm^-2"
    assert manifest["observation"]["context_fields"] == list(
        DEFAULT_OBSERVATION_CONTEXT_FIELDS
    )
    assert manifest["observation"][
        "checkpoint_spectral_reference_line_norm"
    ] == pytest.approx(2.75)
    assert manifest["observation"]["rng_stream_seeds"] == expected_streams
    assert manifest["observation"]["snr_cache_semantics"] == (
        "observed_catalog_flux_snr"
    )
    assert manifest["observation"]["pixel_noise_target_snr_cached"] is False
    assert manifest["args"]["tf_inference"] == "prior_replacement"
    assert manifest["args"]["partition_seed"] == 123
    for name in (
        "snr",
        "image_snr",
        "spectral_quality",
        "spectral_noise_scale",
        "rmag_obs",
        "rmag_sigma",
        "tf_effective_sample_size",
        "tf_effective_sample_fraction",
        "tf_max_normalized_weight",
        "tf_candidate_log_normalizer",
    ):
        assert name in manifest["paths"]
        assert (result_root / manifest["paths"][name]).is_file()

    np.testing.assert_array_equal(
        np.load(result_root / manifest["paths"]["snr"]),
        np.asarray([17.0], dtype=np.float32),
    )

    diagnostic_summary = manifest["observation"][
        "tf_prior_replacement_diagnostics"
    ]
    assert diagnostic_summary["tf_effective_sample_size"]["median"] == 3.0
    assert diagnostic_summary["tf_candidate_log_normalizer"]["mean"] == -1.2

    assert captured["tf_inference"] == "prior_replacement"
    assert captured["matched_group_size"] == 1
    assert captured["noise_seed"] == expected_streams["image_noise"]
    assert captured["spectral_noise_seed"] == expected_streams["spectral_noise"]
    assert captured["magnitude_seed"] == expected_streams["magnitude_observation"]
    assert captured["spectral_quality_seed"] == expected_streams["spectral_quality"]
    generator_seeds = {
        captured[name].initial_seed()
        for name in (
            "image_randgen",
            "spectral_randgen",
            "magnitude_randgen",
            "spectral_quality_randgen",
        )
    }
    assert generator_seeds == set(expected_streams.values())
