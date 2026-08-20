import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _report():
    path = (
        Path(__file__).resolve().parents[1]
        / "arch" / "diagnostics" / "shear_response_report.py"
    )
    spec = importlib.util.spec_from_file_location("shear_response_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matched(nbase=6, delta=0.01):
    report = _report()
    rows, truth, estimate, rmag, ratio = [], [], [], [], []
    response = np.asarray([[1.03, 0.02], [-0.01, 0.98]])
    additive = np.asarray([2e-4, -1e-4])
    state_shear = {
        "zero": (0.0, 0.0), "g1_plus": (delta, 0.0),
        "g1_minus": (-delta, 0.0), "g2_plus": (0.0, delta),
        "g2_minus": (0.0, -delta),
    }
    identifier = 0
    for base in range(nbase):
        nuisance = [
            0.2 + 0.01 * base,
            0.65,
            5.0,
            180.0 + base,
            0.8,
            1.2,
            2.0e-14,
        ]
        for state in report.STATE_ORDER:
            shear = np.asarray(state_shear[state])
            rows.append(
                {
                    "ID": identifier,
                    "base_id": base,
                    "state": state,
                    "g1": shear[0],
                    "g2": shear[1],
                }
            )
            truth.append([*shear, *nuisance])
            estimate.append([*(additive + response @ shear), *nuisance])
            rmag.append(20.0 + 0.01 * base)
            ratio.append(float(base))
            identifier += 1
    return (
        pd.DataFrame(rows),
        np.asarray(truth),
        np.asarray(estimate),
        np.asarray(rmag),
        np.asarray(ratio),
    )


def test_matched_tf_ratio_must_be_equal_inside_all_five_states():
    report = _report()
    manifest, truth, estimate, rmag, ratio = _matched()
    cube, true_cube, base_ratio, _ = report.build_matched_cubes(
        manifest, truth, estimate, rmag, ratio
    )
    np.testing.assert_array_equal(base_ratio, np.arange(6.0))
    ratio[3] += 1e-6
    with pytest.raises(ValueError, match="identical within every five-state"):
        report.build_matched_cubes(manifest, truth, estimate, rmag, ratio)


def test_weighted_response_recovers_known_matrix_and_additive():
    report = _report()
    manifest, truth, estimate, rmag, ratio = _matched(nbase=10)
    cube, true_cube, base_ratio, _ = report.build_matched_cubes(
        manifest, truth, estimate, rmag, ratio
    )
    weight = report.normalize_log_weights(base_ratio)
    result = report.analyze_response(
        cube, true_cube, weight, calibration_fraction=0.5, seed=7
    )
    np.testing.assert_allclose(
        result["calibration_response"], [[1.03, 0.02], [-0.01, 0.98]], atol=1e-13
    )
    np.testing.assert_allclose(result["calibration_additive"], [2e-4, -1e-4])
    np.testing.assert_allclose(result["corrected_holdout_response"], np.eye(2), atol=1e-13)
    np.testing.assert_allclose(result["corrected_holdout_additive"], [0.0, 0.0], atol=1e-13)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("nuisance", "non-shear nuisance truths"),
        ("halpha", "halpha_flux_true differs"),
        ("rmag", "rmag_true differs"),
        ("cross_shear", "expected zero/g1\\+/g1-/g2\\+/g2- shear stencil"),
        ("manifest_shear", "manifest g1/g2 do not match"),
        ("identifier", "manifest IDs must be contiguous"),
    ],
)
def test_matched_response_rejects_corrupt_group_contract(corrupt, message):
    report = _report()
    manifest, truth, estimate, rmag, ratio = _matched()
    if corrupt == "nuisance":
        truth[1, 5] += 1.0
    elif corrupt == "halpha":
        truth[1, 8] *= 1.1
    elif corrupt == "rmag":
        rmag[1] += 0.01
    elif corrupt == "cross_shear":
        truth[1, 1] = 0.002
        manifest.loc[1, "g2"] = 0.002
    elif corrupt == "manifest_shear":
        manifest.loc[1, "g1"] += 0.001
    elif corrupt == "identifier":
        manifest.loc[1, "ID"] = manifest.loc[0, "ID"]
    with pytest.raises(ValueError, match=message):
        report.build_matched_cubes(
            manifest, truth, estimate, rmag, ratio
        )


def test_cli_requires_an_explicit_posterior_source():
    report = _report()
    args = report.parse_args([
        "--cache-root", "cache", "--manifest", "manifest.csv",
        "--output", "result.json", "--posterior-source", "tf_target",
    ])
    assert args.posterior_source == "tf_target"


def test_tf_importance_summary_reports_candidate_and_population_ess():
    report = _report()
    manifest, _, _, _, ratio = _matched(nbase=6)
    base_weight = report.normalize_log_weights(np.arange(6.0))
    nrows = len(manifest)
    summary = report.summarize_importance_sampling(
        manifest,
        base_weight,
        np.linspace(100.0, 200.0, nrows),
        np.linspace(0.2, 0.4, nrows),
        np.linspace(0.01, 0.03, nrows),
    )

    assert summary["matched_base_galaxies"] == 6
    assert summary["posterior_rows"] == 30
    assert summary["population_effective_sample_size"] == pytest.approx(
        1.0 / np.sum(base_weight**2)
    )
    assert summary["posterior_candidate_ess"]["minimum"] == pytest.approx(100.0)
    assert summary["posterior_candidate_ess_fraction"]["maximum"] == pytest.approx(0.4)
    assert summary["posterior_candidate_max_weight"]["maximum"] == pytest.approx(0.03)
