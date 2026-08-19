import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
from data_generate.observation_schema import HALPHA_FLUX_TRUE_COLUMN
from data_generate.generate_fits_wrapper import (
    SIMULATION_PARAMETERS,
    build_generate_command,
)
from data_generate.make_database import (
    extract_fiducial_parameters,
    merge_shards,
    normalize_sample_table,
)
from data_generate import make_shear_response_samples


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sample_row(sample_id=17):
    values = {
        "g1": 0.01,
        "g2": -0.02,
        "theta_int": 0.3,
        "sini": 0.7,
        "v0": 2.0,
        "vcirc": 250.0,
        "rscale": 0.8,
        "hlr": 1.2,
    }
    return {"ID": sample_id, **values}


def test_v2_wrapper_forwards_two_nuisances_version_and_fiber_layout():
    row = pd.Series(
        {
            **_sample_row(),
            "rmag_true": 20.125,
            "halpha_flux_true": 4.2e-15,
            "fiber_layout": "galaxy_axis",
            "observation_model_version": 2,
        }
    )

    command = build_generate_command(
        row,
        part=4,
        dataset="v2_pilot",
        low_psf=True,
    )

    assert command[0] == sys.executable
    assert "-n=4" in command
    assert "-d=v2_pilot" in command
    assert "-ID=17" in command
    assert SIMULATION_PARAMETERS == (
        "g1",
        "g2",
        "theta_int",
        "sini",
        "v0",
        "vcirc",
        "rscale",
        "hlr",
    )
    for name in SIMULATION_PARAMETERS:
        assert f"-{name}={float(row[name])}" in command
    assert "--observation-model-version=2" in command
    assert "--rmag-true=20.125" in command
    assert "--halpha-flux-true=4.2e-15" in command
    assert "--fiber-layout=galaxy_axis" in command
    assert "--low_psf" in command


def test_generate_fits_consumes_and_archives_both_v2_nuisances():
    source = (REPO_ROOT / "data_generate" / "generate_fits.py").read_text()

    assert "'--rmag-true'" in source
    assert "'--halpha-flux-true'" in source
    assert "rmag_true=rmag_true" in source
    assert "halpha_flux_true=halpha_flux_true" in source
    assert "datavector.header[FITS_RMAG_TRUE_KEY] = rmag_true" in source
    assert (
        "datavector.header[FITS_HALPHA_FLUX_TRUE_KEY] = halpha_flux_true"
        in source
    )


def test_legacy_wrapper_accepts_old_index_column_and_omitted_metadata():
    values = _sample_row(sample_id=23)
    values["Unnamed: 0"] = values.pop("ID")

    command = build_generate_command(
        pd.Series(values),
        part=1,
        dataset="legacy",
    )

    assert "-ID=23" in command
    assert "--observation-model-version=1" in command
    assert not any(item.startswith("--rmag-true=") for item in command)
    assert not any(item.startswith("--halpha-flux-true=") for item in command)
    assert "--fiber-layout=image_axis" in command


def test_normalization_and_fid_extraction_exclude_observation_metadata():
    targets = list(config.MODEL_CONFIG.par_ranges)
    rows = []
    for row_index in range(3):
        row = {
            name: lower + (upper - lower) * row_index / 2
            for name, (lower, upper) in config.MODEL_CONFIG.par_ranges.items()
        }
        rows.append(
            {
                "rmag_true": 18.0 + row_index,
                "halpha_flux_true": (10.0 + row_index) * 1.0e-16,
                "fiber_layout": "galaxy_axis",
                "observation_model_version": 2,
                "unrelated_metadata": 100 + row_index,
                **row,
            }
        )
    samples = pd.DataFrame(rows)

    normalized = normalize_sample_table(samples, config.MODEL_CONFIG.par_ranges)
    fids = extract_fiducial_parameters(
        normalized,
        [2, 0],
        targets,
    )

    assert fids.shape == (2, 8)
    assert fids.dtype == np.float32
    np.testing.assert_allclose(fids[0], np.ones(8))
    np.testing.assert_allclose(fids[1], -np.ones(8))
    np.testing.assert_array_equal(normalized["rmag_true"], [18.0, 19.0, 20.0])
    np.testing.assert_array_equal(
        normalized[HALPHA_FLUX_TRUE_COLUMN],
        np.asarray([10.0, 11.0, 12.0]) * 1.0e-16,
    )
    np.testing.assert_array_equal(
        normalized["observation_model_version"], [2, 2, 2]
    )
    assert (normalized["fiber_layout"] == "galaxy_axis").all()
    np.testing.assert_array_equal(normalized["unrelated_metadata"], [100, 101, 102])


def test_fid_extraction_requires_every_named_target():
    samples = pd.DataFrame([_sample_row()]).drop(columns="vcirc")

    with pytest.raises(ValueError, match="missing inference targets"):
        extract_fiducial_parameters(
            samples,
            [0],
            config.MODEL_CONFIG.par_ranges,
        )


def test_merge_refuses_missing_shards_before_opening_output(tmp_path, monkeypatch):
    base = tmp_path / "merged"
    first_shard = tmp_path / "merged_shard_0_of_2"
    first_shard.mkdir()

    def fail_writer(**kwargs):
        pytest.fail(f"Writer must not open when a shard is missing: {kwargs}")

    monkeypatch.setattr("data_generate.make_database.px.Writer", fail_writer)
    with pytest.raises(FileNotFoundError, match="expected shards are missing"):
        merge_shards(str(base), num_shards=2, chunk_size=8)

    assert first_shard.is_dir()
    assert not base.exists()


def test_matched_shear_samples_preserve_v2_metadata(tmp_path, monkeypatch):
    source = pd.DataFrame(
        [
            {
                **_sample_row(sample_id=0),
                "rmag_true": 19.25,
                "halpha_flux_true": 3.1e-15,
                "fiber_layout": "galaxy_axis",
                "observation_model_version": 2,
            },
            {
                **_sample_row(sample_id=1),
                "theta_int": -0.8,
                "vcirc": 410.0,
                "rmag_true": 22.1,
                "halpha_flux_true": 7.4e-15,
                "fiber_layout": "image_axis",
                "observation_model_version": 2,
            },
        ]
    )
    input_path = tmp_path / "base.csv"
    output_path = tmp_path / "matched.csv"
    manifest_path = tmp_path / "manifest.csv"
    source.to_csv(input_path, index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_shear_response_samples.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
            "--nbase",
            "2",
            "--delta-g",
            "0.01",
            "--seed",
            "31",
        ],
    )

    make_shear_response_samples.main()

    matched = pd.read_csv(output_path)
    manifest = pd.read_csv(manifest_path)
    joined = manifest.merge(
        matched[[
            "ID",
            "rmag_true",
            "halpha_flux_true",
            "fiber_layout",
            "observation_model_version",
        ]],
        on="ID",
        validate="one_to_one",
    )
    assert len(joined) == 10
    assert (joined.groupby("base_id").size() == 5).all()
    for _, group in joined.groupby("base_id"):
        assert group["rmag_true"].nunique() == 1
        assert group[HALPHA_FLUX_TRUE_COLUMN].nunique() == 1
        assert group["fiber_layout"].nunique() == 1
        assert group["observation_model_version"].nunique() == 1
        source_row = int(group["source_row"].iloc[0])
        assert group["rmag_true"].iloc[0] == pytest.approx(
            source.loc[source_row, "rmag_true"]
        )
        assert group[HALPHA_FLUX_TRUE_COLUMN].iloc[0] == pytest.approx(
            source.loc[source_row, HALPHA_FLUX_TRUE_COLUMN],
            rel=1.0e-12,
            abs=0.0,
        )
        assert group["fiber_layout"].iloc[0] == source.loc[
            source_row, "fiber_layout"
        ]
        assert group["observation_model_version"].iloc[0] == 2


def test_simulator_v2_generation_database_launchers_are_syntax_valid_and_paired():
    launcher_names = (
        "generate_simulator_v2.slurm",
        "make_database_simulator_v2.slurm",
        "merge_database_simulator_v2.slurm",
    )
    launchers = {
        name: (REPO_ROOT / "data_generate" / name).read_text()
        for name in launcher_names
    }

    for name in launcher_names:
        subprocess.run(
            ["bash", "-n", str(REPO_ROOT / "data_generate" / name)],
            check=True,
        )

    common_defaults = (
        'SAMPLE_NAME="${SAMPLE_NAME:-valid_1m_simv2_galaxyaxis_halpha}"',
        'DATASET_NAME="${DATASET_NAME:-valid_1m_simv2_galaxyaxis_halpha}"',
        'TOTAL="${TOTAL:-100000}"',
        'CHUNK_SIZE="${CHUNK_SIZE:-2000}"',
    )
    for text in launchers.values():
        for expected in common_defaults:
            assert expected in text

    generate = launchers["generate_simulator_v2.slurm"]
    assert "#SBATCH --array=1-50" in generate
    assert "--observation-model-version 2" in generate
    assert "--fiber-layout galaxy_axis" in generate
    assert 'SAMPLE_FILE="samples_${SAMPLE_NAME}.csv"' in generate

    make = launchers["make_database_simulator_v2.slurm"]
    assert "#SBATCH --array=1-50" in make
    assert '--shard_idx="${SHARD_INDEX}"' in make
    assert '--num_shards="${PART_COUNT}"' in make

    merge = launchers["merge_database_simulator_v2.slurm"]
    assert "#SBATCH --array" not in merge
    assert '--num_shards="${PART_COUNT}"' in merge
    assert "--merge" in merge


def test_simulator_v2_training_launchers_fix_validation_observations():
    expected_prefixes = {
        "pretrain_ccl_simulator_v2.slurm": (
            'MODEL_PREFIX="CNN-SetAttn-D4_CCL_simv2_galaxyaxis_halpha"'
        ),
        "train_npe_simulator_v2_affine.slurm": (
            'MODEL_PREFIX="CNN-SetAttn-D4-affine_simv2_galaxyaxis_halpha"'
        ),
    }
    for name, expected_prefix in expected_prefixes.items():
        path = REPO_ROOT / "arch" / name
        text = path.read_text()
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert expected_prefix in text
        assert "datasets/valid_1m_simv2_galaxyaxis_halpha/" in text
        assert "datasets/small_1m_simv2_galaxyaxis_halpha/" in text
        assert "--observation-model-version 2" in text
        assert "--fiber-layout galaxy_axis" in text
        assert "--fixed-validation-streams" in text


def test_simulator_v2_runbook_preserves_array_dependencies():
    runbook = (
        REPO_ROOT / "repo_report" / "SIMULATOR_V2_DESIGN.md"
    ).read_text()

    assert "--dependency=afterok:${FITS_JOB_ID}" in runbook
    assert "--dependency=afterok:${DB_JOB_ID}" in runbook
    assert "--array=1-5" in runbook
    assert "refuses to open an output database" in runbook


def test_shear_response_launchers_use_halpha_v2_mode1_workflow():
    generator = (
        REPO_ROOT / "data_generate" / "generate_shear_response.slurm"
    )
    database = (
        REPO_ROOT / "data_generate" / "make_shear_response_database.slurm"
    )
    merger = (
        REPO_ROOT / "data_generate" / "merge_shear_response_database.slurm"
    )
    inference = REPO_ROOT / "arch" / "shear_response_inference.slurm"
    for path in (generator, database, merger, inference):
        subprocess.run(["bash", "-n", str(path)], check=True)

    generator_text = generator.read_text()
    assert "samples_shear_response_simv2_galaxyaxis_halpha_5k.csv" in generator_text
    assert "--observation-model-version 2" in generator_text
    assert "--fiber-layout galaxy_axis" in generator_text
    for path in (database, merger):
        text = path.read_text()
        assert "shear_response_simv2_galaxyaxis_halpha_5k" in text

    inference_text = inference.read_text()
    assert 'MODE:-1' in inference_text
    assert '--tf-inference="${TF_INFERENCE:-none}"' in inference_text
    assert "shear_response_simv2_galaxyaxis_halpha_5k" in inference_text

    runbook = (
        REPO_ROOT / "repo_report" / "SHEAR_RESPONSE_RUNBOOK.md"
    ).read_text()
    assert "samples_valid_1m_simv2_galaxyaxis_halpha.csv" in runbook
    assert "halpha_flux_true" in runbook
