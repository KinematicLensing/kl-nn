"""The Sersic-profile option of the generator leaves production rows unchanged."""

import pandas as pd
import pytest

from data_generate.generate_fits_wrapper import build_generate_command
from data_generate.generation_integrity import simulator_v3_science_row_fingerprint
from data_generate.observation_schema import SIMULATION_SERSIC_N, validate_sersic_n

SCAN_SERSIC_N = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)


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


def test_production_profile_is_exponential():
    assert SIMULATION_SERSIC_N == 1.0


@pytest.mark.parametrize("n", SCAN_SERSIC_N)
def test_scan_values_are_valid(n):
    assert validate_sersic_n(n) == n


@pytest.mark.parametrize("n", (0.0, 0.2, 7.0, float("nan")))
def test_out_of_range_sersic_n_is_rejected(n):
    with pytest.raises(ValueError):
        validate_sersic_n(n)


def test_wrapper_omits_sersic_flag_for_exponential_rows():
    for row in (_row(), _row(sersic_n=1.0)):
        command = build_generate_command(pd.Series(row), part=1, dataset="x")
        assert not any(item.startswith("--sersic-n") for item in command)


def test_wrapper_forwards_sersic_n():
    command = build_generate_command(pd.Series(_row(sersic_n=1.75)), part=1, dataset="x")
    assert "--sersic-n=1.75" in command


def test_fingerprint_unchanged_for_exponential_and_distinct_otherwise():
    base = simulator_v3_science_row_fingerprint(17, _row())
    assert simulator_v3_science_row_fingerprint(17, _row(sersic_n=1.0)) == base
    prints = {simulator_v3_science_row_fingerprint(17, _row(sersic_n=n)) for n in SCAN_SERSIC_N}
    assert len(prints) == len(SCAN_SERSIC_N)
    assert base in prints
