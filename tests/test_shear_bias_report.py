from copy import deepcopy
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest


def _load_report():
    path = (
        Path(__file__).resolve().parents[1]
        / "arch"
        / "diagnostics"
        / "shear_bias_report.py"
    )
    spec = importlib.util.spec_from_file_location("shear_bias_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_report()


def _metric_row():
    return {
        "estimator": "Mean",
        "component": "g1",
        "c": 2.5e-4,
        "c_se": 3.0e-5,
        "low_m": -1.7e-2,
        "low_m_se": 2.0e-3,
        "cubic_m": 2.6e-2,
        "cubic_m_se": 3.0e-3,
        "cubic_q": -7.25,
        "cubic_q_se": 0.5,
        "spin2": 4.0e-5,
        "spin4": 1.2e-4,
        "sini_cuts": {"0.5": (-1.3e-2, 2.0e-3, 1234)},
    }


def test_metrics_table_scales_additive_and_multiplicative_values_only():
    row = _metric_row()
    original = deepcopy(row)

    table = report.metrics_table([row])

    assert "10<sup>4</sup> mean c" in table
    assert "10<sup>2</sup> low-|g| m" in table
    assert "10<sup>2</sup> cubic m" in table
    assert "cubic q (unscaled)" in table
    assert "10<sup>4</sup> spin-2 amp" in table
    assert "2.500 ± 0.300" in table
    assert "-1.700 ± 0.200" in table
    assert "2.600 ± 0.300" in table
    assert "-7.250e+00 ± 5.000e-01" in table
    assert "<td>0.400</td><td>1.200</td>" in table
    assert row == original


def test_cuts_table_scales_slope_and_uncertainty():
    row = _metric_row()
    original = deepcopy(row)

    table = report.cuts_table([row])

    assert "10<sup>2</sup> low-|g| m" in table
    assert "-1.300 ± 0.200" in table
    assert "1,234" in table
    assert row == original


def test_component_metrics_remain_in_physical_units():
    truth = np.array([-0.015, -0.005, 0.005, 0.015])
    estimate = truth + 2.0e-4 + 3.0e-2 * truth

    metrics = report.component_metrics(truth, estimate, low_g=0.02)

    np.testing.assert_allclose(metrics["c"], 2.0e-4, atol=1e-14)
    np.testing.assert_allclose(metrics["low_c"], 2.0e-4, atol=1e-14)
    np.testing.assert_allclose(metrics["low_m"], 3.0e-2, atol=1e-14)
    np.testing.assert_allclose(metrics["cubic_m"], 3.0e-2, atol=1e-12)
    np.testing.assert_allclose(metrics["cubic_q"], 0.0, atol=1e-10)


def _assert_errorbar_in_physical_units(axis, index, x, residual, bins):
    _, expected_y, expected_error = report.binned(x, residual, bins)
    np.testing.assert_allclose(axis.lines[index].get_ydata(), expected_y)
    segments = axis.collections[index].get_segments()
    bounds = np.asarray([[segment[0, 1], segment[1, 1]] for segment in segments])
    np.testing.assert_allclose(
        bounds,
        np.column_stack([expected_y - expected_error, expected_y + expected_error]),
    )


def test_bias_figure_uses_physical_residuals_and_errors(monkeypatch):
    truth = np.zeros((8, 4), dtype=float)
    truth[:, 0] = np.linspace(-0.02, 0.02, len(truth))
    truth[:, 1] = np.linspace(0.02, -0.02, len(truth))
    truth[:, 2] = np.linspace(-np.pi, np.pi, len(truth))
    map_residual = np.linspace(1.0e-4, 4.0e-4, len(truth))
    mean_residual = -np.linspace(2.0e-4, 5.0e-4, len(truth))
    map_estimate = truth.copy()
    mean_estimate = truth.copy()
    map_estimate[:, :2] += map_residual[:, None]
    mean_estimate[:, :2] += mean_residual[:, None]
    case = {
        "case": "model:dataset",
        "truth": truth,
        "estimates": {"MAP": map_estimate, "Mean": mean_estimate},
    }
    figures = []

    def capture_figure(fig):
        figures.append(fig)
        return "data:image/png;base64,test"

    monkeypatch.setattr(report, "fig_data_uri", capture_figure)

    try:
        assert report.bias_figure(case, bins=2) == "data:image/png;base64,test"
        truth_axis = figures[0].axes[0]
        theta_axis = figures[0].axes[2]
        _assert_errorbar_in_physical_units(
            truth_axis, 0, truth[:, 0], map_residual, bins=2
        )
        _assert_errorbar_in_physical_units(
            truth_axis, 1, truth[:, 0], mean_residual, bins=2
        )
        _assert_errorbar_in_physical_units(
            theta_axis, 0, truth[:, 2], map_residual, bins=2
        )
        _assert_errorbar_in_physical_units(
            theta_axis, 1, truth[:, 2], mean_residual, bins=2
        )
        assert truth_axis.get_ylabel() == "estimate - truth"
        assert theta_axis.get_ylabel() == "estimate - truth"
    finally:
        for fig in figures:
            plt.close(fig)



def _galaxy_to_image_clockwise(g_plus, g_cross, theta):
    cos2 = np.cos(2.0 * theta)
    sin2 = np.sin(2.0 * theta)
    return (
        g_plus * cos2 - g_cross * sin2,
        g_plus * sin2 + g_cross * cos2,
    )


def test_clockwise_galaxy_transform_known_quarter_turn():
    g_plus, g_cross = report.img_to_galaxy_clockwise(
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([np.pi / 4.0, 0.0]),
    )

    np.testing.assert_allclose(g_plus, [1.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(g_cross, [0.0, 0.0], atol=1e-15)


def test_galaxy_frame_metrics_recover_physical_additive_and_response():
    n = 80
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    true_plus = np.linspace(-0.018, 0.018, n)
    true_cross = 0.017 * np.sin(np.linspace(0.0, 4.0 * np.pi, n))
    true_g1, true_g2 = _galaxy_to_image_clockwise(true_plus, true_cross, theta)
    truth = np.zeros((n, 8), dtype=float)
    truth[:, 0] = true_g1
    truth[:, 1] = true_g2
    truth[:, 2] = theta
    truth[:, 3] = np.linspace(0.05, 0.95, n)
    truth[:, 5] = np.linspace(100.0, 300.0, n)
    truth[:, 7] = np.linspace(0.2, 2.0, n)

    estimated_plus = true_plus + 2.5e-4 + 0.03 * true_plus
    estimated_cross = true_cross - 1.5e-4 - 0.02 * true_cross
    estimated_g1, estimated_g2 = _galaxy_to_image_clockwise(
        estimated_plus, estimated_cross, theta
    )
    estimate = truth.copy()
    estimate[:, 0] = estimated_g1
    estimate[:, 1] = estimated_g2
    # Make the sini error exactly track the E residual to test the correlation.
    estimate[:, 3] += estimated_plus - true_plus

    rows, correlations = report.galaxy_frame_diagnostics(
        {"truth": truth, "estimates": {"Mean": estimate}}, low_g=0.02
    )
    plus = next(row for row in rows if row["component"] == "g+ (E)")
    cross = next(row for row in rows if row["component"] == "gx (B)")
    np.testing.assert_allclose(plus["low_c"], 2.5e-4, atol=1e-14)
    np.testing.assert_allclose(plus["low_m"], 0.03, atol=1e-12)
    np.testing.assert_allclose(cross["low_c"], -1.5e-4, atol=1e-14)
    np.testing.assert_allclose(cross["low_m"], -0.02, atol=1e-12)
    sini_plus = next(
        row
        for row in correlations
        if row["component"] == "g+ (E)" and row["nuisance"] == "sini"
    )
    np.testing.assert_allclose(sini_plus["correlation"], 1.0, atol=1e-14)
    assert sini_plus["n"] == n


def test_galaxy_frame_figure_uses_physical_residuals_and_errors(monkeypatch):
    n = 8
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)
    sini = np.linspace(0.1, 0.9, n)
    plus_residual = np.linspace(1.0e-4, 4.0e-4, n)
    cross_residual = -np.linspace(2.0e-4, 5.0e-4, n)
    truth = np.zeros((n, 8), dtype=float)
    truth[:, 2] = theta
    truth[:, 3] = sini
    estimate = truth.copy()
    estimate[:, 0], estimate[:, 1] = _galaxy_to_image_clockwise(
        plus_residual, cross_residual, theta
    )
    figures = []

    def capture_figure(fig):
        figures.append(fig)
        return "data:image/png;base64,test"

    monkeypatch.setattr(report, "fig_data_uri", capture_figure)
    try:
        uri = report.galaxy_frame_figure(
            {
                "case": "model:dataset",
                "truth": truth,
                "estimates": {"Mean": estimate},
            },
            bins=2,
        )
        assert uri == "data:image/png;base64,test"
        _assert_errorbar_in_physical_units(
            figures[0].axes[0], 0, sini, plus_residual, bins=2
        )
        _assert_errorbar_in_physical_units(
            figures[0].axes[1], 0, sini, cross_residual, bins=2
        )
        assert figures[0].axes[0].get_ylabel() == "estimate - truth"
        assert figures[0].axes[1].get_ylabel() == "estimate - truth"
    finally:
        for fig in figures:
            plt.close(fig)



def test_coverage_metrics_handles_theta_interval_across_wrap_seam():
    truth = np.array([[-3.10], [0.0], [0.10]])
    summary = np.array(
        [
            [[2.90], [3.10], [-3.00]],
            [[2.90], [3.10], [-3.00]],
            [[-0.20], [0.00], [0.20]],
        ]
    )

    row = report.coverage_metrics(
        truth, summary, feature_names=("theta_int",)
    )[0]

    assert row["n"] == 3
    np.testing.assert_allclose(row["coverage"], 2.0 / 3.0)
    np.testing.assert_allclose(
        row["coverage_se"], np.sqrt((2.0 / 3.0) * (1.0 / 3.0) / 3.0)
    )


def test_load_case_max_galaxies_uses_identical_combined_prefix(monkeypatch, tmp_path):
    n = 5
    truth = np.arange(n * 8, dtype=float).reshape(n, 8)
    snr = np.arange(n, dtype=float)
    map_all = np.arange(2 * n * 8, dtype=float).reshape(2, n, 8)
    mean_all = np.arange(2 * n * 3 * 8, dtype=float).reshape(2, n, 3, 8)

    def fake_load_concat(directory, axis):
        return {
            "truth": truth,
            "snr": snr,
            "map_estimates": map_all,
            "mean_estimates": mean_all,
        }[directory.name]

    monkeypatch.setattr(report, "load_concat", fake_load_concat)
    case = report.load_case(
        tmp_path, "model:dataset", mode=1, max_galaxies=3
    )

    np.testing.assert_array_equal(case["truth"], truth[:3])
    np.testing.assert_array_equal(case["snr"], snr[:3])
    np.testing.assert_array_equal(case["estimates"]["MAP"], map_all[1, :3])
    np.testing.assert_array_equal(
        case["mean_summary"], mean_all[1, :3]
    )
    with pytest.raises(ValueError, match="must be positive"):
        report.load_case(tmp_path, "model:dataset", mode=0, max_galaxies=0)


def test_load_case_detects_in_support_summary_and_retention(monkeypatch, tmp_path):
    n = 5
    root = tmp_path / "model" / "dataset"
    (root / "in_support_mean_estimates").mkdir(parents=True)
    (root / "in_support_retention").mkdir()
    truth = np.arange(n * 8, dtype=float).reshape(n, 8)
    snr = np.arange(n, dtype=float)
    map_all = np.zeros((1, n, 8))
    mean_all = np.zeros((1, n, 3, 8))
    support_all = np.full((1, n, 3, 8), 7.0)
    retention_all = np.linspace(0.5, 0.9, n)[None]

    def fake_load_concat(directory, axis):
        return {
            "truth": truth,
            "snr": snr,
            "map_estimates": map_all,
            "mean_estimates": mean_all,
            "in_support_mean_estimates": support_all,
            "in_support_retention": retention_all,
        }[directory.name]

    monkeypatch.setattr(report, "load_concat", fake_load_concat)
    case = report.load_case(tmp_path, "model:dataset", mode=0, max_galaxies=3)

    np.testing.assert_array_equal(
        case["estimates"]["In-support Mean"], support_all[0, :3, 1]
    )
    np.testing.assert_array_equal(
        case["summaries"]["In-support Mean"], support_all[0, :3]
    )
    np.testing.assert_array_equal(case["support_retention"], retention_all[0, :3])
    assert case["support_manifest"] == root / "in_support_meta" / "manifest.json"


def test_retention_and_coverage_tables_label_truncated_estimator():
    retention = report.retention_table(np.array([0.0, 0.5, 0.75, 1.0]))
    assert "In-support Mean" in retention
    assert "56.25%" in retention
    assert "Galaxies with zero draws" in retention

    row = report.coverage_metrics(
        np.array([[0.0], [0.0]]),
        np.array([[[-1.0], [0.0], [1.0]], [[-1.0], [0.0], [1.0]]]),
        feature_names=("g1",),
    )[0]
    row["estimator"] = "In-support Mean"
    table = report.coverage_table([row])
    assert "<th>Estimator</th>" in table
    assert "In-support Mean" in table


def test_bias_figure_styles_in_support_estimator_without_key_error(monkeypatch):
    n = 8
    truth = np.zeros((n, 8))
    truth[:, 0] = np.linspace(-0.02, 0.02, n)
    truth[:, 1] = np.linspace(0.02, -0.02, n)
    truth[:, 2] = np.linspace(-np.pi, np.pi, n)
    estimate = truth.copy()
    figures = []
    monkeypatch.setattr(
        report,
        "fig_data_uri",
        lambda fig: figures.append(fig) or "data:image/png;base64,test",
    )
    try:
        report.bias_figure(
            {
                "case": "model:dataset",
                "truth": truth,
                "estimates": {"In-support Mean": estimate},
            },
            bins=2,
        )
        assert figures[0].axes[0].lines[0].get_color() == "tab:green"
    finally:
        for figure in figures:
            plt.close(figure)
