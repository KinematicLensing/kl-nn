import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm

from arch.tf_prior import TFPrior
from data_generate.desi_test_set_sampling import (
    GENERATION_MANIFEST_SCHEMA,
    SOURCE_PROVENANCE_COLUMNS,
    write_generation_manifest,
)
from data_generate.latin_hypercube import (
    PARAMETER_LIMITS,
    generate_test_set_samples,
    parse_args,
)
from data_generate.make_database import (
    propagate_generation_manifest,
    validate_generation_manifest,
)
from data_generate.observation_schema import (
    CENTRAL_HALPHA_SNR_COLUMN,
    DEFAULT_HALPHA_LOG10_FLUX_RANGE,
    FIBER_LAYOUT_COLUMN,
    HALPHA_FLUX_TRUE_COLUMN,
    IMAGE_SNR_COLUMN,
    OBSERVATION_MODEL_VERSION_COLUMN,
    RMAG_TRUE_COLUMN,
)


def _write_catalog(path: Path) -> dict[str, np.ndarray]:
    columns = {
        "targetid": np.arange(1000, 1012, dtype=np.int64),
        "z": np.linspace(0.1, 0.6, 12),
        "rmag": np.asarray(
            (18.0, 19.0, 14.9, 21.0, 21.0, 21.0, 22.0, 23.4, np.nan, 20, 20, 20)
        ),
        "hlr": np.asarray((1.0, 6.0, 2.0, 0.1, 2.0, 0.05, 4.0, 5.0, 2, 2, 2, np.nan)),
        "img_snr": np.asarray((10, 20, 30, 10, 30, 30, 1000, 10, 30, np.inf, 30, 30)),
        "halpha_snr": np.asarray((2, 3, 4, 4, np.nan, 4, 150, 1, 4, 4, 151, 4)),
        # Row 0 has zero weight and row 7 has NaN.  Both remain eligible because
        # this field is provenance, not a sampling or downstream analysis weight.
        "xu_effective_weight": np.asarray((0, 1, 1, 1, 1, 1, 0.2, np.nan, 1, 1, 1, 1)),
    }
    fits_columns = []
    for name, values in columns.items():
        format_code = "K" if name == "targetid" else "D"
        fits_columns.append(fits.Column(name=name, format=format_code, array=values))
    hdu = fits.BinTableHDU.from_columns(fits_columns, name="SELECTION")
    hdu.header["XUSAMPLE"] = 1
    hdu.header["SELMODE"] = "fixture"
    hdu.header["DENSPASS"] = True
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return columns


def test_catalog_test_set_preserves_joint_rows_filters_hlr_and_conforms_tf(tmp_path):
    catalog = tmp_path / "xu_sample_1_fullfootprint.fits"
    source = _write_catalog(catalog)
    samples, manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=1401,
        catalog_block_size=3,
    )
    repeated, repeated_manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=1401,
        catalog_block_size=5,
    )
    pd.testing.assert_frame_equal(samples, repeated)
    assert repeated_manifest["catalog_sampling"]["eligible_row_count"] == 4

    np.testing.assert_array_equal(
        np.sort(samples["source_catalog_row"]),
        np.asarray((0, 3, 6, 7)),
    )
    for row in samples.itertuples(index=False):
        source_row = row.source_catalog_row
        assert row.rmag_true == source["rmag"][source_row]
        assert row.image_snr == source["img_snr"][source_row]
        assert row.central_halpha_snr == source["halpha_snr"][source_row]
        assert row.source_targetid == source["targetid"][source_row]
        assert row.source_catalog_z == source["z"][source_row]
        assert row.source_hlr_raw == source["hlr"][source_row]
        assert row.hlr == source["hlr"][source_row]
    assert samples.loc[samples.source_catalog_row == 0, "source_xu_effective_weight"].item() == 0
    assert np.isnan(
        samples.loc[
            samples.source_catalog_row == 7,
            "source_xu_effective_weight",
        ].item()
    )

    count = len(samples)
    cosi = np.sqrt(np.maximum(0.0, 1.0 - samples["sini"].to_numpy() ** 2))
    np.testing.assert_array_equal(np.sort(np.floor(count * cosi).astype(int)), np.arange(count))
    prior = TFPrior()
    magnitude = samples[RMAG_TRUE_COLUMN].to_numpy()
    mean = (magnitude - prior.intercept) / prior.slope
    lower = (np.log10(prior.vcirc_min) - mean) / prior.scatter_dex
    upper = (np.log10(prior.vcirc_max) - mean) / prior.scatter_dex
    standardized = (np.log10(samples["vcirc"].to_numpy()) - mean) / prior.scatter_dex
    tf_quantile = truncnorm.cdf(standardized, lower, upper)
    np.testing.assert_array_equal(
        np.sort(np.floor(count * tf_quantile).astype(int)), np.arange(count)
    )
    log_flux = np.log10(samples[HALPHA_FLUX_TRUE_COLUMN])
    low, high = DEFAULT_HALPHA_LOG10_FLUX_RANGE
    flux_strata = np.floor(count * (log_flux - low) / (high - low)).astype(int)
    np.testing.assert_array_equal(np.sort(flux_strata), np.arange(count))

    assert tuple(samples.columns[:14]) == (
        *PARAMETER_LIMITS,
        RMAG_TRUE_COLUMN,
        HALPHA_FLUX_TRUE_COLUMN,
        IMAGE_SNR_COLUMN,
        CENTRAL_HALPHA_SNR_COLUMN,
        FIBER_LAYOUT_COLUMN,
        OBSERVATION_MODEL_VERSION_COLUMN,
    )
    assert tuple(samples.columns[14:]) == SOURCE_PROVENANCE_COLUMNS
    assert manifest["analysis_mode"] == "test_set"
    assert manifest["population"] == "tf_conformed_catalog"
    assert manifest["redshift"] == manifest["simulation_redshift"] == 0.3
    assert manifest["sample_count"] == 4
    assert manifest["catalog_sampling"]["support_counts"]["hlr_support"] == 9
    assert manifest["catalog_sampling"]["eligibility"]["hlr"] == {
        "finite": True,
        "minimum": 0.1,
        "maximum": 5.0,
        "bounds": "inclusive",
    }
    assert manifest["catalog_sampling"]["eligibility"]["image_snr"] == {
        "finite": True,
        "minimum": 10.0,
        "maximum": 1000.0,
    }
    assert manifest["catalog_sampling"]["eligibility"]["halpha_snr"] == {
        "finite": True,
        "minimum": 1.0,
        "maximum": 150.0,
    }
    assert manifest["parameter_sampling"]["inclination"] == {
        "distribution": "cosi_uniform_0_1_latin_hypercube",
        "transform": "sini=sqrt(1-cosi**2)",
    }
    assert "eligible_hlr_capped_count" not in manifest["catalog_sampling"]
    assert "selected_hlr_capped_count" not in manifest["catalog_sampling"]
    assert (
        manifest["catalog_sampling"]["xu_effective_weight_policy"]
        == "provenance_only_not_sampling_weight"
    )
    assert set(manifest["tf"]) == {
        "slope",
        "intercept",
        "scatter_dex",
        "vcirc_min",
        "vcirc_max",
    }


def test_manifest_sidecar_is_hashed_and_propagates_to_dataset(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _write_catalog(catalog)
    samples, manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=31,
        catalog_block_size=4,
    )
    sample_path = tmp_path / "test_population.csv"
    samples.to_csv(sample_path, index_label="ID")
    manifest_path = write_generation_manifest(manifest, sample_path)
    assert manifest_path == sample_path.with_suffix(".manifest.json")
    payload = json.loads(manifest_path.read_text())
    assert payload["schema"] == GENERATION_MANIFEST_SCHEMA
    assert payload["sample_table"]["path"] == str(sample_path.resolve())
    assert len(payload["sample_table"]["sha256"]) == 64
    assert validate_generation_manifest(sample_path, 4) == manifest_path

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    installed = propagate_generation_manifest(manifest_path, dataset)
    assert installed == dataset / "manifest.json"
    assert installed.read_bytes() == manifest_path.read_bytes()

    sample_path.write_text(sample_path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_generation_manifest(sample_path, 4)


def test_database_rejects_legacy_hlr_cap_manifest(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _write_catalog(catalog)
    samples, manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=32,
        catalog_block_size=4,
    )
    sample_path = tmp_path / "test_population.csv"
    samples.to_csv(sample_path, index_label="ID")
    manifest_path = write_generation_manifest(manifest, sample_path)
    payload = json.loads(manifest_path.read_text())
    payload["catalog_sampling"]["eligibility"]["hlr"] = {
        "finite": True,
        "minimum": 0.1,
        "maximum_policy": "cap_after_selection",
        "cap": 5.0,
    }
    payload["catalog_sampling"]["selected_hlr_capped_count"] = 1
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(
        ValueError,
        match="catalog_sampling.eligibility.hlr",
    ):
        validate_generation_manifest(sample_path, 4)


def test_database_rejects_stale_snr_eligibility(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _write_catalog(catalog)
    samples, manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=34,
        catalog_block_size=4,
    )
    sample_path = tmp_path / "test_population.csv"
    samples.to_csv(sample_path, index_label="ID")
    manifest_path = write_generation_manifest(manifest, sample_path)
    payload = json.loads(manifest_path.read_text())
    payload["catalog_sampling"]["eligibility"]["image_snr"][
        "minimum"
    ] = 5.0
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="eligibility.image_snr"):
        validate_generation_manifest(sample_path, 4)


def test_database_rejects_non_cosi_generation_manifest(tmp_path):
    catalog = tmp_path / "catalog.fits"
    _write_catalog(catalog)
    samples, manifest = generate_test_set_samples(
        4,
        catalog=str(catalog),
        seed=33,
        catalog_block_size=4,
    )
    sample_path = tmp_path / "test_population.csv"
    samples.to_csv(sample_path, index_label="ID")
    manifest_path = write_generation_manifest(manifest, sample_path)
    payload = json.loads(manifest_path.read_text())
    payload["parameter_sampling"]["inclination"] = {
        "distribution": "sini_uniform_0_1_latin_hypercube",
        "transform": "identity",
    }
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="parameter_sampling.inclination"):
        validate_generation_manifest(sample_path, 4)


def test_test_set_cli_requires_catalog_and_explicit_output():
    with pytest.raises(SystemExit):
        parse_args(["--test-set"])
    with pytest.raises(SystemExit):
        parse_args(["--test-set", "--catalog", "catalog.fits"])
    with pytest.raises(SystemExit):
        parse_args(["--catalog", "catalog.fits"])
    args = parse_args(
        [
            "--test-set",
            "--catalog",
            "catalog.fits",
            "--output",
            "test.csv",
        ]
    )
    assert args.test_set
    assert args.nsamples == 100_000
    assert args.output == "test.csv"


def test_production_launcher_uses_canonical_cut_names_and_seeds():
    launcher = (
        Path(__file__).resolve().parents[1]
        / "data_generate"
        / "generate_desi_test_sets.slurm"
    ).read_text()
    assert 'BASE_SEED="${BASE_SEED:-42000}"' in launcher
    assert 'SAMPLE_CUTS=(1 3 5)' in launcher
    assert 'PRODUCTION_TOTAL="100000"' in launcher
    assert 'REQUESTED_TOTAL="${TOTAL:-${PRODUCTION_TOTAL}}"' in launcher
    assert '[[ "${REQUESTED_TOTAL}" != "${PRODUCTION_TOTAL}" ]]' in launcher
    assert '[[ -n "${SAMPLE_NAME:-}" ]]' in launcher
    assert 'CANONICAL_SAMPLE_NAME="test_100k_simv3_cosi_xu${CUT}_tf"' in launcher
    assert 'SAMPLE_NAME="${SAMPLE_NAME:-' not in launcher
    assert 'SEED="$((BASE_SEED + CUT))"' in launcher
