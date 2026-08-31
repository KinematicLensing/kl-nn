import csv
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest

from data_generate import check_fits_generation
from data_generate import generate_fits_wrapper
from data_generate.generation_integrity import (
    FITS_SCIENCE_ROW_FINGERPRINT_KEY,
    FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY,
    FITS_SCIENCE_ROW_ID_KEY,
    SCIENCE_ROW_FINGERPRINT_VERSION,
    SCIENCE_ROW_FLOAT_COLUMNS,
    atomic_write_simulator_v3_fits,
    simulator_v3_fits_completion_error,
    simulator_v3_science_row_fingerprint,
)
from data_generate import observation_schema as schema


ROOT = Path(__file__).resolve().parents[1]


def test_fits_launcher_has_recovery_safe_runtime_and_log_contract():
    launcher = (ROOT / "data_generate/generate_simulator_v3.slurm").read_text()
    assert "#SBATCH --time=12:00:00" in launcher
    assert "generate_simulator_v3_%A_%a.out" in launcher
    assert "--skip-existing" in launcher
    assert "SLURM_ARRAY_JOB_ID" in launcher


def _row(sample_id: int) -> dict[str, float | int | str]:
    return {
        "ID": sample_id,
        "g1": 0.01,
        "g2": -0.02,
        "theta_int": 0.3,
        "sini": 0.7,
        "v0": 2.0,
        "vcirc": 250.0,
        "rscale": 0.8,
        "hlr": 1.2,
        schema.RMAG_TRUE_COLUMN: 20.125 + sample_id,
        schema.HALPHA_FLUX_TRUE_COLUMN: 4.2e-15 + sample_id * 1e-17,
        schema.IMAGE_SNR_COLUMN: 240.0 + sample_id,
        schema.CENTRAL_HALPHA_SNR_COLUMN: 37.0 + sample_id,
        schema.FIBER_LAYOUT_COLUMN: schema.GALAXY_AXIS_FIBER_LAYOUT,
        schema.OBSERVATION_MODEL_VERSION_COLUMN: schema.OBSERVATION_MODEL_VERSION,
    }


def _write_valid_fits(
    path: Path,
    row: dict[str, float | int | str],
    *,
    include_science_identity: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    central_flux = float(row[schema.HALPHA_FLUX_TRUE_COLUMN])
    header = fits.Header()
    header["OBSNUM"] = 6
    header[schema.FITS_OBSERVATION_MODEL_VERSION_KEY] = schema.OBSERVATION_MODEL_VERSION
    header[schema.FITS_PHOTOMETRY_BAND_KEY] = schema.PHOTOMETRY_BAND
    header[schema.FITS_TARGET_LINE_KEY] = schema.TARGET_EMISSION_LINE
    header[schema.FITS_FIBER_LAYOUT_KEY] = schema.GALAXY_AXIS_FIBER_LAYOUT
    header[schema.FITS_SPECTRAL_UNITS_KEY] = schema.SPECTRAL_UNITS
    header[schema.FITS_CENTER_FIBER_INDEX_KEY] = schema.CENTER_FIBER_INDEX
    header[schema.FITS_CENTER_EXPOSURE_KEY] = schema.CENTER_EXPOSURE_S
    header[schema.FITS_OFFSET_EXPOSURE_KEY] = schema.OFFSET_EXPOSURE_S
    header[schema.FITS_IMAGE_PSF_FWHM_KEY] = schema.IMAGE_REFERENCE_PSF_FWHM_ARCSEC
    header[schema.FITS_IMAGE_PIXEL_SCALE_KEY] = schema.IMAGE_PIXEL_SCALE_ARCSEC
    header[schema.FITS_RMAG_TRUE_KEY] = float(row[schema.RMAG_TRUE_COLUMN])
    header[schema.FITS_HALPHA_FLUX_TRUE_KEY] = central_flux
    header[schema.FITS_IMAGE_SNR_KEY] = float(row[schema.IMAGE_SNR_COLUMN])
    header[schema.FITS_CENTER_HALPHA_SNR_KEY] = float(
        row[schema.CENTRAL_HALPHA_SNR_COLUMN]
    )
    header[schema.FITS_HALPHA_FLUX_UNITS_KEY] = schema.HALPHA_FLUX_UNITS
    header[schema.FITS_HALPHA_FLUX_SEMANTICS_KEY] = schema.HALPHA_FLUX_SEMANTICS
    header[schema.FITS_HALPHA_FLUX_TRANSFORM_KEY] = schema.HALPHA_FLUX_TRANSFORM
    header[schema.FITS_HALPHA_FLUX_API_VERSION_KEY] = schema.HALPHA_FLUX_API_VERSION
    header[schema.FITS_HALPHA_TOTAL_FLUX_KEY] = central_flux * 2.0
    header[schema.FITS_CENTER_HALPHA_APERTURE_KEY] = 0.5
    if include_science_identity:
        header[FITS_SCIENCE_ROW_ID_KEY] = int(row["ID"])
        header[FITS_SCIENCE_ROW_FINGERPRINT_VERSION_KEY] = (
            SCIENCE_ROW_FINGERPRINT_VERSION
        )
        header[FITS_SCIENCE_ROW_FINGERPRINT_KEY] = (
            simulator_v3_science_row_fingerprint(int(row["ID"]), row)
        )
    hdus = [fits.PrimaryHDU(header=header)]
    hdus.extend(fits.ImageHDU(np.zeros(61, dtype=np.float32)) for _ in range(5))
    hdus.append(fits.ImageHDU(np.zeros((48, 48), dtype=np.float32)))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _metadata(row):
    columns = (
        schema.RMAG_TRUE_COLUMN,
        schema.HALPHA_FLUX_TRUE_COLUMN,
        schema.IMAGE_SNR_COLUMN,
        schema.CENTRAL_HALPHA_SNR_COLUMN,
    )
    return {name: float(row[name]) for name in columns}


def _identity(row):
    sample_id = int(row["ID"])
    return {
        "expected_sample_id": sample_id,
        "expected_row_fingerprint": simulator_v3_science_row_fingerprint(
            sample_id,
            row,
        ),
    }


def test_full_completion_check_rejects_truncation_and_wrong_row_metadata(tmp_path):
    row = _row(0)
    path = tmp_path / "gal_0.fits"
    _write_valid_fits(path, row)
    assert simulator_v3_fits_completion_error(
        path,
        expected_metadata=_metadata(row),
        **_identity(row),
    ) is None

    wrong_row = dict(row)
    wrong_row[schema.IMAGE_SNR_COLUMN] = float(row[schema.IMAGE_SNR_COLUMN]) + 1.0
    assert "image_snr" in simulator_v3_fits_completion_error(
        path,
        expected_metadata=_metadata(wrong_row),
        **_identity(row),
    )

    payload = path.read_bytes()
    path.write_bytes(payload[:-1])
    assert "complete FITS block" in simulator_v3_fits_completion_error(path)


@pytest.mark.parametrize("column", SCIENCE_ROW_FLOAT_COLUMNS)
def test_full_completion_check_rejects_any_changed_science_value(tmp_path, column):
    original = _row(0)
    path = tmp_path / "gal_0.fits"
    _write_valid_fits(path, original)
    changed = dict(original)
    changed[column] = float(changed[column]) + 0.125

    error = simulator_v3_fits_completion_error(path, **_identity(changed))
    assert error is not None
    assert FITS_SCIENCE_ROW_FINGERPRINT_KEY in error


def test_full_completion_check_requires_identity_and_matching_id(tmp_path):
    row = _row(4)
    legacy_path = tmp_path / "legacy.fits"
    _write_valid_fits(legacy_path, row, include_science_identity=False)
    assert "missing required ROWID" in simulator_v3_fits_completion_error(
        legacy_path,
        **_identity(row),
    )

    current_path = tmp_path / "current.fits"
    _write_valid_fits(current_path, row)
    wrong_id = dict(row)
    wrong_id["ID"] = 5
    assert "ROWID=4; expected 5" in simulator_v3_fits_completion_error(
        current_path,
        **_identity(wrong_id),
    )


def test_atomic_writer_preserves_old_final_when_new_output_is_invalid(tmp_path):
    row = _row(0)
    output = tmp_path / "gal_0.fits"
    _write_valid_fits(output, row)
    original = output.read_bytes()

    def invalid_writer(path):
        path.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="Refusing to publish"):
        atomic_write_simulator_v3_fits(output, invalid_writer)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob(".gal_0.fits.*.tmp.fits"))


def test_wrapper_resume_skips_valid_and_regenerates_missing_or_invalid(
    tmp_path, monkeypatch
):
    sample_root = tmp_path / "samples"
    fits_root = tmp_path / "fits"
    sample_root.mkdir()
    rows = [_row(0), _row(1), _row(2)]
    pd.DataFrame(rows).to_csv(sample_root / "sample.csv", index=False)
    _write_valid_fits(fits_root / "dataset/part_1/gal_0.fits", rows[0])
    invalid_path = fits_root / "dataset/part_1/gal_2.fits"
    invalid_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_path.write_bytes(b"")
    generated: list[int] = []

    def fake_run(command, check):
        assert check is True
        sample_id = int(
            next(item.split("=", 1)[1] for item in command if item.startswith("-ID="))
        )
        generated.append(sample_id)
        _write_valid_fits(
            fits_root / f"dataset/part_1/gal_{sample_id}.fits",
            rows[sample_id],
        )

    monkeypatch.setattr(generate_fits_wrapper, "SAMPLE_ROOT", str(sample_root))
    monkeypatch.setattr(generate_fits_wrapper, "FITS_ROOT", str(fits_root))
    monkeypatch.setattr(generate_fits_wrapper.subprocess, "run", fake_run)
    generate_fits_wrapper.main(
        [
            "-i=0",
            "-j=3",
            "-n=1",
            "-s=sample.csv",
            "-d=dataset",
            "--skip-existing",
        ]
    )
    assert generated == [1, 2]


def test_wrapper_resume_regenerates_legacy_and_changed_science_rows(
    tmp_path, monkeypatch
):
    sample_root = tmp_path / "samples"
    fits_root = tmp_path / "fits"
    sample_root.mkdir()
    old_row = _row(0)
    changed_row = {**old_row, "g1": float(old_row["g1"]) + 0.05}
    current_row = _row(1)
    rows = [changed_row, current_row]
    pd.DataFrame(rows).to_csv(sample_root / "sample.csv", index=False)
    _write_valid_fits(fits_root / "dataset/part_1/gal_0.fits", old_row)
    _write_valid_fits(
        fits_root / "dataset/part_1/gal_1.fits",
        current_row,
        include_science_identity=False,
    )
    generated: list[int] = []

    def fake_run(command, check):
        assert check is True
        sample_id = int(
            next(item.split("=", 1)[1] for item in command if item.startswith("-ID="))
        )
        generated.append(sample_id)
        _write_valid_fits(
            fits_root / f"dataset/part_1/gal_{sample_id}.fits",
            rows[sample_id],
        )

    monkeypatch.setattr(generate_fits_wrapper, "SAMPLE_ROOT", str(sample_root))
    monkeypatch.setattr(generate_fits_wrapper, "FITS_ROOT", str(fits_root))
    monkeypatch.setattr(generate_fits_wrapper.subprocess, "run", fake_run)
    generate_fits_wrapper.main(
        [
            "-i=0",
            "-j=2",
            "-n=1",
            "-s=sample.csv",
            "-d=dataset",
            "--skip-existing",
        ]
    )
    assert generated == [0, 1]
    for sample_id, row in enumerate(rows):
        assert simulator_v3_fits_completion_error(
            fits_root / f"dataset/part_1/gal_{sample_id}.fits",
            expected_metadata=_metadata(row),
            **_identity(row),
        ) is None


def test_checksum_reports_exact_incomplete_parts(tmp_path, capsys):
    sample = tmp_path / "sample.csv"
    rows = [_row(index) for index in range(3)]
    pd.DataFrame(rows).to_csv(sample, index=False)
    fits_dir = tmp_path / "fits"
    for sample_id, part in ((0, 1), (1, 1), (2, 2)):
        path = fits_dir / f"part_{part}/gal_{sample_id}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_valid_fits(path, rows[sample_id])
    report = tmp_path / "complete.tsv"
    common = [
        "--sample",
        str(sample),
        "--fits-dir",
        str(fits_dir),
        "--chunk-size",
        "2",
        "--report",
        str(report),
    ]
    assert check_fits_generation.main(common) == 0

    (fits_dir / "part_1/gal_1.fits").write_bytes(b"")
    (fits_dir / "part_2/gal_2.fits").rename(fits_dir / "part_1/gal_2.fits")
    assert check_fits_generation.main(common) == 1
    assert "Incomplete array parts: 1-2" in capsys.readouterr().out
    with report.open(newline="") as handle:
        failures = list(csv.DictReader(handle, delimiter="\t"))
    assert {failure["status"] for failure in failures} == {
        "invalid",
        "missing",
        "unexpected",
    }


def test_checksum_deep_verification_rejects_stale_science_row(tmp_path):
    sample = tmp_path / "sample.csv"
    row = _row(0)
    pd.DataFrame([row]).to_csv(sample, index=False)
    fits_dir = tmp_path / "fits"
    _write_valid_fits(fits_dir / "part_1/gal_0.fits", row)
    report = tmp_path / "complete.tsv"
    common = [
        "--sample",
        str(sample),
        "--fits-dir",
        str(fits_dir),
        "--chunk-size",
        "1",
        "--verify-fits",
        "--report",
        str(report),
    ]
    assert check_fits_generation.main(common) == 0

    changed = {**row, "vcirc": float(row["vcirc"]) + 1.0}
    pd.DataFrame([changed]).to_csv(sample, index=False)
    assert check_fits_generation.main(common) == 1
    with report.open(newline="") as handle:
        failures = list(csv.DictReader(handle, delimiter="\t"))
    assert len(failures) == 1
    assert failures[0]["status"] == "invalid"
    assert FITS_SCIENCE_ROW_FINGERPRINT_KEY in failures[0]["detail"]
