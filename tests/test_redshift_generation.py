import numpy as np
import pandas as pd
import pytest

from data_generate.generate_fits_wrapper import build_generate_command
from data_generate.generation_integrity import simulator_v3_science_row_fingerprint
from data_generate.observation_schema import (
    HALPHA_REST_WAVELENGTH_NM,
    halpha_observed_sigma,
    halpha_window,
)

SCAN_REDSHIFTS = np.round(np.arange(0.14, 0.5001, 0.04), 2)


def _row(sample_id=17, **extra):
    row = {
        "ID": sample_id,
        "g1": 0.01,
        "g2": -0.02,
        "theta_int": 0.3,
        "sini": 0.7,
        "v0": 2.0,
        "vcirc": 250.0,
        "rscale": 0.8,
        "hlr": 1.2,
        "rmag_true": 20.125,
        "halpha_flux_true": 4.2e-15,
        "image_snr": 240.0,
        "central_halpha_snr": 37.0,
        "fiber_layout": "measured_axis",
        "observation_model_version": 3,
    }
    row.update(extra)
    return row


def test_production_redshift_reproduces_the_old_window_and_width():
    assert halpha_window(0.3) == (851.0, 855.81)
    assert halpha_observed_sigma(0.3) == 0.065


@pytest.mark.parametrize("z", SCAN_REDSHIFTS)
def test_window_follows_the_line_and_keeps_61_pixels(z):
    lo, hi = halpha_window(z)
    assert len(np.arange(lo, hi, 0.08)) == 61
    # The line sits at the same place in the window at every redshift.
    line = HALPHA_REST_WAVELENGTH_NM * (1.0 + z)
    assert line - lo == pytest.approx(HALPHA_REST_WAVELENGTH_NM * 1.3 - 851.0, abs=1e-9)
    assert halpha_observed_sigma(z) == pytest.approx(0.05 * (1.0 + z))


def test_wrapper_omits_redshift_flags_for_production_rows():
    command = build_generate_command(pd.Series(_row()), part=1, dataset="x")
    assert not any(item.startswith(("--redshift", "--targeting-seed")) for item in command)
    command = build_generate_command(pd.Series(_row(redshift=0.3)), part=1, dataset="x")
    assert not any(item.startswith("--redshift") for item in command)


def test_wrapper_forwards_redshift_and_targeting_seed():
    row = _row(redshift=0.42, targeting_seed=900_000_123)
    command = build_generate_command(pd.Series(row), part=1, dataset="x")
    assert "--redshift=0.42" in command
    assert "--targeting-seed=900000123" in command


def test_fingerprint_unchanged_for_default_redshift_and_seed():
    base = simulator_v3_science_row_fingerprint(17, _row())
    assert simulator_v3_science_row_fingerprint(17, _row(redshift=0.3)) == base
    assert simulator_v3_science_row_fingerprint(17, _row(targeting_seed=900_000_017)) == base
    assert simulator_v3_science_row_fingerprint(17, _row(redshift=0.3, targeting_seed=None)) == base


def test_fingerprint_tells_redshifts_and_seeds_apart():
    base = simulator_v3_science_row_fingerprint(17, _row())
    z42 = simulator_v3_science_row_fingerprint(17, _row(redshift=0.42))
    z46 = simulator_v3_science_row_fingerprint(17, _row(redshift=0.46))
    seeded = simulator_v3_science_row_fingerprint(17, _row(targeting_seed=900_000_005))
    assert len({base, z42, z46, seeded}) == 4


def test_redshift_scan_rows_share_every_input_but_redshift():
    from data_generate.make_redshift_scan_samples import build_scan

    source = pd.DataFrame([_row(sample_id=i, g1=0.001 * i) for i in range(7)])
    samples, manifest = build_scan(source, nbase=3, redshifts=[0.14, 0.3, 0.5], seed=1)
    assert list(samples["ID"]) == list(range(9))
    assert list(manifest["ID"]) == list(range(9))
    assert list(samples["redshift"]) == [0.14] * 3 + [0.3] * 3 + [0.5] * 3
    inputs = samples.drop(columns=["ID", "redshift"]).to_numpy()
    for k in (1, 2):
        assert (inputs[3 * k:3 * k + 3] == inputs[:3]).all()
    # Targeting seed is the source row's production seed.
    assert list(samples["targeting_seed"][:3]) == [900_000_000 + i for i in manifest["source_id"][:3]]
    assert (samples["fiber_layout"] == "measured_axis").all()
