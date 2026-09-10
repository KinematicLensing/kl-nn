"""Tests for the hybrid Fisher stencil sample builder and bound math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_generate import make_fisher_stencil_samples as builder
from data_generate.make_fisher_stencil_samples import HALPHA_FLUX_TRUE_COLUMN


ROOT = Path(__file__).resolve().parents[1]


def _source_row(**overrides):
    row = {
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
        "fiber_layout": "galaxy_axis",
        "observation_model_version": 3,
    }
    row.update(overrides)
    return row


def _source_table(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append(
            _source_row(
                g1=float(rng.uniform(-0.1, 0.1)),
                g2=float(rng.uniform(-0.1, 0.1)),
                sini=float(rng.uniform(0.0, 1.0)),
                vcirc=float(rng.uniform(80.0, 500.0)),
                hlr=float(rng.uniform(0.2, 4.5)),
                halpha_flux_true=float(10 ** rng.uniform(-16.5, -14.5)),
                image_snr=float(rng.uniform(20.0, 400.0)),
                central_halpha_snr=float(rng.uniform(3.0, 80.0)),
            )
        )
    return pd.DataFrame(rows)


def test_sini_edge_is_one_sided_not_out_of_bounds():
    side, plus, minus = builder.parameter_sides("sini", 0.99, ("additive", 0.02))
    assert side == "minus_only"
    assert not np.isfinite(plus)
    assert minus == pytest.approx(0.97)
    assert builder.in_bounds("sini", minus)


def test_g_edge_is_one_sided():
    side, plus, minus = builder.parameter_sides("g1", 0.095, ("additive", 0.01))
    assert side == "minus_only"
    assert minus == pytest.approx(0.085)


def test_theta_wraps_instead_of_clipping():
    plus = builder.apply_step("theta_int", np.pi - 0.01, 1, ("additive", 0.05))
    assert plus < 0.0
    assert builder.in_bounds("theta_int", plus)


def test_full_catalog_center_keeps_native_shear_and_halpha_steps_metadata(
    tmp_path, monkeypatch
):
    source = pd.DataFrame(
        [
            _source_row(g1=0.04, g2=-0.03, sini=0.99, vcirc=62.0),
            _source_row(g1=-0.02, g2=0.01, sini=0.4, vcirc=200.0),
        ]
    )
    samples, manifest = builder.build_full_catalog(
        source,
        np.asarray([0, 1], dtype=np.int64),
        steps=builder.DEFAULT_STEPS,
        n_double_delta=0,
    )
    joined = manifest.merge(samples, on="ID", validate="one_to_one")
    center0 = joined[(joined.base_id == 0) & (joined.state == "center")].iloc[0]
    assert center0["g1"] == pytest.approx(0.04)
    assert center0["g2"] == pytest.approx(-0.03)
    sini_states = set(
        joined.loc[(joined.base_id == 0) & (joined.stepped == "sini"), "state"]
    )
    assert "sini_plus" not in sini_states
    assert "sini_minus" in sini_states
    assert (joined.loc[joined.stepped == "sini", "sini"] <= 1.0).all()
    vcirc_states = set(joined.loc[(joined.base_id == 0) & (joined.stepped == "vcirc"), "state"])
    assert "vcirc_minus" not in vcirc_states
    halpha_plus = joined[joined.state == "halpha_flux_true_plus"].iloc[0]
    assert halpha_plus[HALPHA_FLUX_TRUE_COLUMN] == pytest.approx(
        4.2e-15 * 10**0.02, rel=1e-12
    )


def test_g5_catalog_forces_zero_shear():
    source = _source_table(n=8)
    samples, manifest = builder.build_g5_catalog(
        source, np.arange(3), delta_g=0.01
    )
    joined = manifest.merge(samples, on="ID")
    for _, group in joined.groupby("base_id"):
        assert len(group) == 5
        zero = group[group.state == "zero"].iloc[0]
        assert zero["g1"] == 0.0
        assert zero["g2"] == 0.0
        assert group.hlr.nunique() == 1


def test_cli_writes_disjoint_catalogs(tmp_path, monkeypatch):
    source = _source_table(n=80, seed=3)
    source_path = tmp_path / "source.csv"
    source.to_csv(source_path, index=False)
    full_out = tmp_path / "full.csv"
    full_man = tmp_path / "full_manifest.csv"
    g5_out = tmp_path / "g5.csv"
    g5_man = tmp_path / "g5_manifest.csv"
    builder.main(
        [
            "--input",
            str(source_path),
            "--full-output",
            str(full_out),
            "--full-manifest",
            str(full_man),
            "--g5-output",
            str(g5_out),
            "--g5-manifest",
            str(g5_man),
            "--n-full",
            "8",
            "--n-g5",
            "12",
            "--n-double-delta",
            "2",
            "--seed",
            "2718",
        ]
    )
    full_man_df = pd.read_csv(full_man)
    g5_man_df = pd.read_csv(g5_man)
    assert set(full_man_df.source_row).isdisjoint(set(g5_man_df.source_row))
    assert full_man_df.base_id.nunique() == 8
    assert g5_man_df.base_id.nunique() == 12
    n_2d = full_man_df.state.str.endswith("_2d").sum()
    assert n_2d >= 4
    full_samples = pd.read_csv(full_out)
    assert len(full_samples) <= 8 * 19 + 2 * 4


def _fisher_module():
    import importlib.util

    path = ROOT / "arch" / "diagnostics" / "fisher_shear_bound.py"
    spec = importlib.util.spec_from_file_location("fisher_shear_bound_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fisher_math_recovers_known_gaussian_signal():
    fisher = _fisher_module()
    rng = np.random.default_rng(0)
    signal = rng.normal(size=32)
    inv_var = np.full(32, 4.0)
    dmu = np.zeros((9, 32), dtype=np.float64)
    dmu[0] = signal
    dmu[1] = rng.normal(size=32)
    info = fisher.fisher_matrix(dmu, inv_var)
    expected_00 = 4.0 * float(signal @ signal)
    assert info[0, 0] == pytest.approx(expected_00, rel=1e-12)
    info[2:, 2:] = np.eye(7)
    _, _, sigma_cr = fisher.profile_shear_block(info)
    assert np.isfinite(sigma_cr)
    m, response = fisher.shrinkage_m_r(0.039)
    expected_m = -(0.039**2) / (0.039**2 + fisher.SIGMA_PI**2)
    assert m == pytest.approx(expected_m, rel=1e-12)
    assert response == pytest.approx(1.0 + m, rel=1e-12)


def test_snr_rescaling_is_quadratic():
    fisher = _fisher_module()
    unit = np.eye(9)
    twice = fisher.scale_fisher(unit, unit, image_snr=2.0, spec_snr=1.0)
    spec_twice = fisher.scale_fisher(unit, unit, image_snr=1.0, spec_snr=2.0)
    np.testing.assert_allclose(twice, 4.0 * unit + unit)
    np.testing.assert_allclose(spec_twice, unit + 4.0 * unit)


def test_onesided_derivative_matches_twosided_on_linear_signal():
    fisher = _fisher_module()
    plus = np.array([2.0, 4.0])
    minus = np.array([0.0, 0.0])
    center = np.array([1.0, 2.0])
    two = fisher.finite_difference(center, plus=plus, minus=minus, delta=1.0, side="two_sided")
    one = fisher.finite_difference(center, plus=plus, minus=None, delta=1.0, side="plus_only")
    np.testing.assert_allclose(two, one)
    np.testing.assert_allclose(two, np.array([1.0, 2.0]))
