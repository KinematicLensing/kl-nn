from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def _write_posterior_part(
    root,
    index,
    total,
    samples,
    *,
    truth=None,
    cancel_add_noise=False,
):
    sample_dir = root / "sample"
    truth_dir = root / "truth"
    meta_dir = root / "meta"
    sample_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    label = f"part{index}of{total}"
    np.save(sample_dir / f"{label}.npy", samples)
    if truth is None:
        truth = np.zeros(
            (samples.shape[1], samples.shape[-1]), dtype=np.float32
        )
    np.save(truth_dir / f"{label}.npy", truth)
    (meta_dir / f"{label}.json").write_text(
        '{"args": {"cancel_add_noise": '
        + ("true" if cancel_add_noise else "false")
        + "}}"
    )


def test_load_shear_pit_values_streams_parts_selects_mode_and_prefix(
    monkeypatch, tmp_path
):
    """Prior-matched ranks stream the selected mode and common case prefix."""
    root = tmp_path / "model" / "dataset"
    draws = 5
    features = 8
    edges = report.SHEAR_PP_BIN_EDGES
    truth = np.zeros((6, features), dtype=np.float64)
    truth[:, :2] = [
        [-0.1, 0.0],
        [edges[1], -0.05],
        [0.0, 0.05],
        [edges[2], np.nan],
        [0.1, -0.1],
        [0.0, 0.0],  # Deliberately beyond the loaded five-galaxy prefix.
    ]
    parts = []
    for _ in range(2):
        # Mode 0 is an out-of-support sentinel. Selecting it accidentally would
        # give zero retained draws for every component.
        values = np.full((2, 3, draws, features), 99.0, dtype=np.float64)
        parts.append(values)

    # Component-wise filtering is intentional: g1 and g2 use the bin selected
    # by their own truths and do not require the other component to be in-bin.
    parts[0][1, 0, :, 0] = [-0.1, -0.09, -0.04, edges[1], np.nan]
    parts[0][1, 1, :, 0] = [edges[1], -0.02, 0.0, edges[2], np.nan]
    parts[0][1, 2, :, 0] = [edges[1], -0.01, 0.0, 0.01, edges[2]]
    parts[1][1, 0, :, 0] = [edges[2], 0.05, 0.1, edges[2] - 1e-6, 0.11]
    parts[1][1, 1, :, 0] = [edges[2], 0.05, 0.1, np.nan, 0.11]

    # The first galaxy has no middle-bin g2 draws. The fourth has a nonfinite
    # truth. Both must produce NaN ranks and retained count zero.
    parts[0][1, 0, :, 1] = [-0.09, -0.05, 0.05, 0.09, np.nan]
    parts[0][1, 1, :, 1] = [-0.09, -0.05, -0.04, 0.0, np.nan]
    parts[0][1, 2, :, 1] = [0.04, 0.05, 0.08, 0.0, np.nan]
    parts[1][1, 0, :, 1] = [0.0, 0.0, 0.0, 0.0, 0.0]
    parts[1][1, 1, :, 1] = [-0.1, -0.05, -0.04, edges[1], np.nan]

    for index, values in enumerate(parts):
        start = 3 * index
        _write_posterior_part(
            root,
            index,
            2,
            values,
            truth=truth[start : start + 3],
        )

    # Five truths deliberately stop in the middle of the second sample part.
    case = {"root": root, "truth": truth[:5], "case": "model:dataset"}
    original_load = report.np.load
    posterior_loads = []

    def audited_load(path, *args, **kwargs):
        if Path(path).parent.name in {"sample", "truth"}:
            posterior_loads.append(
                (Path(path).parent.name, Path(path).name, kwargs.get("mmap_mode"))
            )
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(report.np, "load", audited_load)
    pits = report.load_shear_pit_values(case, mode=1, block_size=2)

    # Finite-sample midpoint rank after same-bin filtering:
    # (n_less + 0.5*n_equal + 0.5) / (n_retained + 1).
    np.testing.assert_allclose(pits["g1"], [0.25, 0.25, 0.6, 0.25, 0.75])
    np.testing.assert_array_equal(pits["g1_retained"], [3, 3, 4, 3, 3])
    assert np.isnan(pits["g2"][[0, 3]]).all()
    np.testing.assert_allclose(pits["g2"][[1, 2, 4]], [0.5, 0.5, 0.25])
    np.testing.assert_array_equal(pits["g2_retained"], [0, 3, 3, 0, 3])
    assert posterior_loads == [
        ("sample", "part0of2.npy", "r"),
        ("truth", "part0of2.npy", "r"),
        ("sample", "part1of2.npy", "r"),
        ("truth", "part1of2.npy", "r"),
    ]


@pytest.mark.parametrize("metadata", ["missing", "cancelled"])
def test_load_shear_pit_values_rejects_unverifiable_or_cancelled_samples(
    metadata, tmp_path
):
    root = tmp_path / "model" / "dataset"
    samples = np.zeros((1, 2, 3, 8), dtype=np.float32)
    _write_posterior_part(
        root, 0, 1, samples, cancel_add_noise=(metadata == "cancelled")
    )
    if metadata == "missing":
        (root / "meta" / "part0of1.json").unlink()
    case = {"root": root, "truth": np.zeros((2, 8)), "case": "model:dataset"}

    with pytest.raises(report.PosteriorPPUnavailable):
        report.load_shear_pit_values(case, mode=0)


def test_quantile_binned_filters_nonfinite_pairs_and_returns_physical_errors():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, np.nan])
    y = 1.0e-4 + 2.0e-5 * x
    y[-1] = 9.0

    center, mean, error = report.quantile_binned(x, y, bins=2)

    np.testing.assert_allclose(center, [1.5, 5.5])
    np.testing.assert_allclose(mean, [1.3e-4, 2.1e-4])
    expected_se = np.std([1.0e-4, 1.2e-4, 1.4e-4, 1.6e-4], ddof=1) / 2.0
    np.testing.assert_allclose(error, [expected_se, expected_se])


def test_nuisance_correlation_figure_is_mean_only_and_uses_physical_residuals(
    monkeypatch,
):
    n = 24
    truth = np.zeros((n, 8), dtype=float)
    truth[:, 2] = 0.0  # Image and galaxy-frame components coincide.
    mean_estimate = truth.copy()
    map_estimate = truth.copy()
    nuisance_errors = {
        "sini": np.linspace(-0.2, 0.2, n),
        "vcirc": np.linspace(-30.0, 30.0, n),
        "hlr": np.linspace(0.3, -0.3, n),
    }
    for name, values in nuisance_errors.items():
        mean_estimate[:, report.NUISANCE_INDICES[name]] += values
        map_estimate[:, report.NUISANCE_INDICES[name]] += 10.0 * values
    plus_residual = np.linspace(-4.0e-4, 4.0e-4, n)
    cross_residual = 2.0e-4 * np.cos(np.linspace(0.0, 2.0 * np.pi, n))
    mean_estimate[:, 0] = plus_residual
    mean_estimate[:, 1] = cross_residual
    # A visibly different MAP must not add curves to this Mean-only diagnostic.
    map_estimate[:, 0] = 0.1
    map_estimate[:, 1] = -0.1
    case = {
        "case": "model:dataset",
        "truth": truth,
        "estimates": {"MAP": map_estimate, "Mean": mean_estimate},
    }
    figures = []
    monkeypatch.setattr(
        report,
        "fig_data_uri",
        lambda fig: figures.append(fig) or "data:image/png;base64,test",
    )

    try:
        assert (
            report.nuisance_correlation_figure(case, bins=4)
            == "data:image/png;base64,test"
        )
        assert len(figures[0].axes) == 3
        for axis, (name, nuisance_error) in zip(
            figures[0].axes, nuisance_errors.items()
        ):
            # ``errorbar`` places its public label on the container rather than
            # the central Line2D; the first two lines are the g+ and gx curves,
            # followed by the horizontal and vertical zero references.
            plotted = axis.lines[:2]
            assert len(axis.containers) == 2
            expected_x, expected_plus, expected_plus_se = report.quantile_binned(
                nuisance_error, plus_residual, bins=4
            )
            _, expected_cross, expected_cross_se = report.quantile_binned(
                nuisance_error, cross_residual, bins=4
            )
            np.testing.assert_allclose(plotted[0].get_xdata(), expected_x)
            np.testing.assert_allclose(plotted[0].get_ydata(), expected_plus)
            np.testing.assert_allclose(plotted[1].get_xdata(), expected_x)
            np.testing.assert_allclose(plotted[1].get_ydata(), expected_cross)
            segments = axis.collections
            assert len(segments) == 2
            plus_bounds = np.asarray(
                [[segment[0, 1], segment[1, 1]] for segment in segments[0].get_segments()]
            )
            cross_bounds = np.asarray(
                [[segment[0, 1], segment[1, 1]] for segment in segments[1].get_segments()]
            )
            np.testing.assert_allclose(
                plus_bounds,
                np.column_stack(
                    [expected_plus - expected_plus_se, expected_plus + expected_plus_se]
                ),
            )
            np.testing.assert_allclose(
                cross_bounds,
                np.column_stack(
                    [expected_cross - expected_cross_se, expected_cross + expected_cross_se]
                ),
            )
            assert "estimate - truth" in axis.get_xlabel()
            assert axis.get_ylabel() == "shear estimate - truth"
            assert name in axis.get_xlabel()
    finally:
        for figure in figures:
            plt.close(figure)


def test_posterior_pp_figure_shows_prior_matched_bins_and_retention(monkeypatch):
    edges = np.linspace(-0.1, 0.1, 4)
    # Include every bin edge exactly, values immediately below the interior
    # edges, and values outside the plotted shear range. This makes the
    # half-open/closed boundary convention observable in the plotted curves.
    g1_truth = np.array(
        [
            edges[0],
            edges[1] - 1.0e-6,
            edges[1],
            0.0,
            edges[2] - 1.0e-6,
            edges[2],
            edges[3],
            edges[0] - 1.0e-3,
            edges[3] + 1.0e-3,
        ]
    )
    truth = np.zeros((len(g1_truth), 8), dtype=float)
    truth[:, 0] = g1_truth
    truth[:, 1] = g1_truth[::-1]
    pits = {
        "g1": np.array(
            [0.12, 0.21, np.nan, 0.44, 0.52, 0.63, 0.78, np.nan, np.nan]
        ),
        "g2": np.array(
            [np.nan, np.nan, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28, np.nan]
        ),
        "g1_retained": np.array([5, 7, 0, 9, 11, 13, 15, 0, 0]),
        "g2_retained": np.array([0, 0, 21, 23, 25, 27, 29, 31, 0]),
    }
    figures = []
    monkeypatch.setattr(
        report,
        "fig_data_uri",
        lambda fig: figures.append(fig) or "data:image/png;base64,test",
    )

    try:
        assert (
            report.posterior_pp_figure(pits, truth)
            == "data:image/png;base64,test"
        )
        axes = figures[0].axes
        assert len(axes) == 2
        assert "Prior-matched conditional" in figures[0]._suptitle.get_text()
        for component_index, (axis, component) in enumerate(
            zip(axes, ("g1", "g2"))
        ):
            component_truth = truth[:, component_index]
            retained = pits[f"{component}_retained"]
            masks = [
                (component_truth >= edges[0]) & (component_truth < edges[1]),
                (component_truth >= edges[1]) & (component_truth < edges[2]),
                (component_truth >= edges[2]) & (component_truth <= edges[3]),
            ]
            empirical_lines = [
                line for line in axis.lines if line.get_label() != "Uniform"
            ]
            assert len(empirical_lines) == 3
            assert len(axis.collections) == 3  # one DKW band per shear bin
            for bin_index, (line, mask) in enumerate(zip(empirical_lines, masks)):
                finite_rank = mask & np.isfinite(pits[component])
                expected_x, expected_y, expected_distance = (
                    report.posterior_pp_curve(pits[component][mask])
                )
                np.testing.assert_allclose(line.get_xdata(), expected_x)
                np.testing.assert_allclose(line.get_ydata(), expected_y)
                assert f"N={finite_rank.sum():,}" in line.get_label()
                assert f"KS D={expected_distance:.3f}" in line.get_label()
                assert f"{edges[bin_index]:.4f}" in line.get_label()
                assert f"{edges[bin_index + 1]:.4f}" in line.get_label()
                label = line.get_label().lower()
                expected_median = np.median(retained[finite_rank])
                assert f"median retained={expected_median:,.0f}" in label
                zero_retained = np.count_nonzero(mask & (retained == 0))
                assert f"zero retained={zero_retained:,}" in label

            assert axis.get_xlim()[0] <= 0.0 and axis.get_xlim()[1] >= 1.0
            assert axis.get_ylim()[0] <= 0.0 and axis.get_ylim()[1] >= 1.0
            reference_lines = [
                line
                for line in axis.lines
                if line.get_label() == "Uniform"
                and np.array_equal(line.get_xdata(), line.get_ydata())
            ]
            assert len(reference_lines) == 1

        # Exact interior boundaries go to the bin on their right, while the
        # outer endpoints are included and out-of-range objects are excluded.
        assert [
            np.count_nonzero(
                (truth[:, 0] >= edges[index])
                & (
                    truth[:, 0] <= edges[index + 1]
                    if index == 2
                    else truth[:, 0] < edges[index + 1]
                )
            )
            for index in range(3)
        ] == [2, 3, 2]
    finally:
        for figure in figures:
            plt.close(figure)


def test_main_describes_prior_matched_conditional_pp(monkeypatch, tmp_path):
    output = tmp_path / "report.html"
    args = SimpleNamespace(
        case=["model:dataset"],
        cache_root=tmp_path,
        mode=0,
        max_galaxies=None,
        low_g=0.02,
        bins=2,
        output=output,
    )
    truth = np.zeros((2, 8), dtype=float)
    case = {
        "case": "model:dataset",
        "root": tmp_path / "model" / "dataset",
        "truth": truth,
        "estimates": {"Mean": truth.copy()},
        "summaries": {"Mean": np.zeros((2, 3, 8), dtype=float)},
        "support_retention": None,
        "support_manifest": None,
    }
    pits = {
        "g1": np.array([0.25, 0.75]),
        "g2": np.array([0.25, 0.75]),
        "g1_retained": np.array([10, 10]),
        "g2_retained": np.array([10, 10]),
    }
    monkeypatch.setattr(report, "parse_args", lambda: args)
    monkeypatch.setattr(report, "load_case", lambda *unused, **also_unused: case)
    monkeypatch.setattr(report, "compute_metrics", lambda *unused: [])
    monkeypatch.setattr(
        report, "galaxy_frame_diagnostics", lambda *unused: ([], [])
    )
    monkeypatch.setattr(report, "coverage_metrics", lambda *unused: [])
    monkeypatch.setattr(report, "load_shear_pit_values", lambda *unused: pits)
    monkeypatch.setattr(
        report, "posterior_pp_figure", lambda *unused: "data:image/png;base64,pp"
    )
    for function_name in (
        "bias_figure",
        "galaxy_frame_figure",
        "nuisance_correlation_figure",
    ):
        monkeypatch.setattr(
            report,
            function_name,
            lambda *unused: "data:image/png;base64,diagnostic",
        )

    report.main()

    document = output.read_text().lower()
    assert "prior-matched conditional posterior p-p diagnostic" in document
    assert "restricted to the same true-shear interval" in document
    assert "renormal" in document
    assert "zero retained draws" in document
    assert "need not be uniform even for an exact bayesian posterior" not in document


def test_shear_pp_bin_masks_snap_all_float32_edges_to_right_hand_bins():
    edges = report.SHEAR_PP_BIN_EDGES.astype(np.float32)
    true_shear = np.array(
        [-0.101, edges[0], edges[1], 0.0, edges[2], edges[3], 0.101],
        dtype=np.float32,
    )

    masks = report.shear_pp_bin_masks(true_shear)

    assert [np.flatnonzero(mask).tolist() for mask in masks] == [
        [1],
        [2, 3],
        [4, 5],
    ]
