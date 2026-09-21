from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from config import TARGET_NAMES
from diagnostics.hop_boundary_npe import (
    ABS_G_BIN_WIDTH,
    G02_NPE,
    G02_VAL,
    N_REALIZATIONS,
    N_SAMPLES,
    REPORT_DIR,
    SNR_FACTOR,
    abs_g_bin_index,
    abs_g_overlap,
    apply_shared_noise,
    average_realizations,
    catalog_geometry,
    choose_ellipticity_cut,
    circular_residual,
    clip_snr,
    compare_groups,
    distribution_summary,
    group_metrics,
    json_safe,
    label_distributions,
    linear_slope,
    match_abs_g_counts,
    parse_args,
    selected_mask,
)


def _physical_row(*, g1=0.0, g2=0.0, theta_int=0.0, cosi=1.0, hlr=1.0):
    row = np.zeros(len(TARGET_NAMES), dtype=np.float64)
    names = list(TARGET_NAMES)
    row[names.index("g1")] = g1
    row[names.index("g2")] = g2
    row[names.index("theta_int")] = theta_int
    row[names.index("cosi")] = cosi
    row[names.index("hlr")] = hlr
    return row


def test_cli_defaults_match_the_g02_plan():
    args = parse_args([])
    assert args.model_name == G02_NPE
    assert args.data_dir == G02_VAL
    assert args.report_dir == REPORT_DIR
    assert args.n_realizations == N_REALIZATIONS
    assert args.n_samples == N_SAMPLES
    assert args.n_realizations == 4
    assert args.n_samples == 2048
    assert SNR_FACTOR == pytest.approx(1.2)


def test_choose_ellipticity_cut_keeps_the_tightest_viable_threshold():
    ellipticity = np.array(
        [0.05] * 40 + [0.05] * 40 + [0.085] * 10 + [0.085] * 10
    )
    hop_class = np.array(
        ["unhopped"] * 40 + ["hopped"] * 40 + ["unhopped"] * 10 + ["hopped"] * 10
    )
    assert choose_ellipticity_cut(ellipticity, hop_class) == pytest.approx(0.08)


def test_choose_ellipticity_cut_relaxes_when_a_group_is_too_small():
    ellipticity = np.concatenate(
        [np.full(40, 0.05), np.full(10, 0.05), np.full(40, 0.095)]
    )
    hop_class = np.array(["unhopped"] * 40 + ["hopped"] * 10 + ["hopped"] * 40)
    assert choose_ellipticity_cut(ellipticity, hop_class) == pytest.approx(0.10)


def test_selected_mask_drops_knife_edge_rows():
    ellipticity = np.array([0.01, 0.01, 0.20])
    hop_class = np.array(["hopped", "knife_edge", "unhopped"])
    mask = selected_mask(ellipticity, hop_class, 0.08)
    assert mask.tolist() == [True, False, False]


def test_abs_g_overlap_is_the_shared_interval():
    abs_g = np.array([0.02, 0.05, 0.12, 0.08, 0.18])
    hop_class = np.array(["unhopped", "unhopped", "unhopped", "hopped", "hopped"])
    low, high = abs_g_overlap(abs_g, hop_class)
    assert low == pytest.approx(0.08)
    assert high == pytest.approx(0.12)


def test_abs_g_overlap_is_none_when_the_ranges_miss():
    abs_g = np.array([0.02, 0.03, 0.15, 0.18])
    hop_class = np.array(["unhopped", "unhopped", "hopped", "hopped"])
    assert abs_g_overlap(abs_g, hop_class) is None


def test_match_abs_g_counts_equalizes_each_bin():
    abs_g = np.array(
        [0.012, 0.015, 0.018, 0.021, 0.052, 0.053, 0.054, 0.055, 0.056]
    )
    hop_class = np.array(
        [
            "unhopped",
            "unhopped",
            "hopped",
            "hopped",
            "unhopped",
            "unhopped",
            "unhopped",
            "hopped",
            "hopped",
        ]
    )
    keep = match_abs_g_counts(abs_g, hop_class, width=0.01, seed=0)
    kept_g = abs_g[keep]
    kept_class = hop_class[keep]
    bins = abs_g_bin_index(kept_g, width=0.01)
    for bin_id in np.unique(bins):
        in_bin = bins == bin_id
        n_hopped = int(np.sum(in_bin & (kept_class == "hopped")))
        n_unhopped = int(np.sum(in_bin & (kept_class == "unhopped")))
        assert n_hopped == n_unhopped
    assert ABS_G_BIN_WIDTH == pytest.approx(0.01)
    # The 0.01 bin has 2 vs 1; the 0.05 bin has 3 vs 2. Matched N is 2*(1+2).
    assert int(keep.sum()) == 6


def test_match_abs_g_counts_is_empty_when_no_bin_is_shared():
    abs_g = np.array([0.02, 0.03, 0.15, 0.16])
    hop_class = np.array(["unhopped", "unhopped", "hopped", "hopped"])
    keep = match_abs_g_counts(abs_g, hop_class, width=0.01, seed=0)
    assert not np.any(keep)


def test_match_abs_g_counts_is_deterministic():
    abs_g = np.array([0.051, 0.052, 0.053, 0.054, 0.055, 0.056])
    hop_class = np.array(
        ["unhopped", "unhopped", "unhopped", "unhopped", "hopped", "hopped"]
    )
    first = match_abs_g_counts(abs_g, hop_class, seed=7)
    second = match_abs_g_counts(abs_g, hop_class, seed=7)
    third = match_abs_g_counts(abs_g, hop_class, seed=8)
    assert first.tolist() == second.tolist()
    assert int(first.sum()) == 4
    assert first.tolist() != third.tolist()


def test_linear_slope_recovers_a_known_line():
    truth = np.linspace(-0.2, 0.2, 21)
    prediction = 0.8 * truth + 0.01
    slope, intercept = linear_slope(truth, prediction)
    assert slope == pytest.approx(0.8, abs=1e-12)
    assert intercept == pytest.approx(0.01, abs=1e-12)


def test_circular_residual_wraps_through_pi():
    residual = circular_residual(np.array([math.pi - 0.1]), np.array([-math.pi + 0.1]))
    assert residual[0] == pytest.approx(0.2, abs=1e-12)


def test_group_metrics_reports_bias_width_and_m():
    n = 8
    truth = {
        "g1": np.linspace(-0.1, 0.1, n),
        "g2": np.zeros(n),
        "theta_int": np.zeros(n),
        "sini": np.full(n, 0.4),
        "hlr": np.full(n, 1.2),
        "abs_g": np.abs(np.linspace(-0.1, 0.1, n)),
        "ellipticity": np.full(n, 0.05),
    }
    estimate = {
        "g1": 0.9 * truth["g1"] + 0.01,
        "g2": np.full(n, 0.002),
        "theta_int": np.full(n, 0.1),
        "sigma_g1": np.full(n, 0.03),
        "sigma_g2": np.full(n, 0.04),
    }
    metrics = group_metrics(truth, estimate)
    assert metrics["n"] == n
    assert metrics["g1_bias"] == pytest.approx(np.mean(estimate["g1"] - truth["g1"]))
    assert metrics["g1_m"] == pytest.approx(-0.1, abs=1e-12)
    assert metrics["g2_bias"] == pytest.approx(0.002)
    assert metrics["sigma_g1"] == pytest.approx(0.03)
    assert metrics["mean_cos_theta"] == pytest.approx(math.cos(0.1))


def test_catalog_geometry_flags_a_round_unhopped_and_a_hopped_disk():
    physical = np.vstack(
        [
            _physical_row(g1=0.0, g2=0.0, theta_int=0.3, cosi=1.0),
            _physical_row(g1=0.16, g2=0.0, theta_int=0.5 * np.pi, cosi=np.sqrt(0.75)),
        ]
    )
    geometry = catalog_geometry(physical, TARGET_NAMES)
    assert geometry["ellipticity"][0] == pytest.approx(0.0, abs=1e-12)
    assert geometry["hop_class"][0] == "unhopped"
    assert geometry["hop_class"][1] == "hopped"
    assert geometry["sini"][1] == pytest.approx(0.5, abs=1e-12)


def test_compare_groups_and_overlap_keep_only_shared_mass():
    rng = np.random.default_rng(0)
    n = 12
    hop_class = np.array(["unhopped"] * 6 + ["hopped"] * 6)
    truth = {
        "g1": np.concatenate([np.linspace(-0.04, 0.04, 6), np.linspace(-0.12, 0.12, 6)]),
        "g2": np.zeros(n),
        "theta_int": np.zeros(n),
        "sini": np.full(n, 0.3),
        "hlr": np.full(n, 1.0),
        "abs_g": np.concatenate([np.linspace(0.01, 0.05, 6), np.linspace(0.04, 0.16, 6)]),
        "ellipticity": np.full(n, 0.06),
        "hop_class": hop_class,
    }
    estimate = {
        "g1": truth["g1"] + 0.01,
        "g2": rng.normal(scale=0.001, size=n),
        "theta_int": np.zeros(n),
        "sigma_g1": np.full(n, 0.02),
        "sigma_g2": np.full(n, 0.02),
    }
    compared = compare_groups(truth, estimate)
    assert compared["unhopped"]["n"] == 6
    assert compared["hopped"]["n"] == 6
    low, high = abs_g_overlap(truth["abs_g"], truth["hop_class"])
    overlap_mask = (truth["abs_g"] >= low) & (truth["abs_g"] <= high)
    overlap = compare_groups(
        {key: value[overlap_mask] for key, value in truth.items()},
        {key: value[overlap_mask] for key, value in estimate.items()},
    )
    assert overlap["unhopped"]["n"] >= 1
    assert overlap["hopped"]["n"] >= 1
    assert overlap["unhopped"]["mean_abs_g"] <= compared["hopped"]["mean_abs_g"]


def test_label_distributions_split_by_hop_class():
    truth = {
        "abs_g": np.array([0.02, 0.03, 0.15]),
        "sini": np.array([0.2, 0.3, 0.4]),
        "hlr": np.array([1.0, 1.1, 2.0]),
        "ellipticity": np.array([0.04, 0.05, 0.06]),
        "hop_class": np.array(["unhopped", "unhopped", "hopped"]),
    }
    summary = label_distributions(truth)
    assert summary["unhopped"]["abs_g"]["n"] == 2
    assert summary["hopped"]["abs_g"]["mean"] == pytest.approx(0.15)
    empty = distribution_summary(np.array([]))
    assert empty["n"] == 0
    assert math.isnan(empty["mean"])


def test_json_safe_turns_nan_and_paths_into_json_values():
    payload = json_safe({"cut": float("nan"), "path": REPORT_DIR, "tuple": (1, 2)})
    assert payload["cut"] is None
    assert payload["path"] == str(REPORT_DIR)
    assert payload["tuple"] == [1, 2]


def test_clip_snr_stays_inside_observation_bounds():
    assert clip_snr(5.0, 10.0, 100.0) == pytest.approx(10.0)
    assert clip_snr(50.0, 10.0, 100.0) == pytest.approx(50.0)
    assert clip_snr(500.0, 10.0, 100.0) == pytest.approx(100.0)


def test_average_realizations_uses_a_circular_mean_for_position_angle():
    rows = [
        {
            "g1": np.array([0.1, 0.2]),
            "g2": np.array([0.0, 0.0]),
            "theta_int": np.array([math.pi - 0.1, 0.0]),
            "sigma_g1": np.array([0.02, 0.03]),
            "sigma_g2": np.array([0.02, 0.03]),
        },
        {
            "g1": np.array([0.3, 0.4]),
            "g2": np.array([0.0, 0.0]),
            "theta_int": np.array([-math.pi + 0.1, 0.2]),
            "sigma_g1": np.array([0.04, 0.05]),
            "sigma_g2": np.array([0.04, 0.05]),
        },
    ]
    averaged = average_realizations(rows)
    assert averaged["g1"][0] == pytest.approx(0.2)
    wrapped = averaged["theta_int"][0]
    assert abs(abs(wrapped) - math.pi) < 0.05


def test_shared_noise_reuses_one_unit_draw(monkeypatch):
    import diagnostics.hop_boundary_npe as hop_mod

    monkeypatch.setattr(
        hop_mod.config,
        "observation",
        {
            "center_fiber_index": 2,
            "center_exposure_s": 180.0,
            "offset_exposure_s": 600.0,
            "spectral_units": "counts",
        },
        raising=False,
    )
    image = torch.stack(
        [torch.ones(1, 8, 8), 2.0 * torch.ones(1, 8, 8)],
        dim=0,
    )
    spec = torch.ones(2, 1, 5, 12)
    spec[1] *= 2.0
    spec[..., 6] += 4.0
    tensors = {"img": image, "spec": spec}
    noisy_image, noisy_spec = apply_shared_noise(
        tensors,
        image_snr=20.0,
        spec_snr=10.0,
        image_seed=7,
        spec_seed=11,
        device=torch.device("cpu"),
    )
    again_image, again_spec = apply_shared_noise(
        tensors,
        image_snr=20.0,
        spec_snr=10.0,
        image_seed=7,
        spec_seed=11,
        device=torch.device("cpu"),
    )
    assert torch.allclose(noisy_image, again_image)
    assert torch.allclose(noisy_spec, again_spec)
    residual = noisy_image - image
    scale = residual[1] / residual[0]
    assert torch.allclose(scale, torch.full_like(scale, 2.0), rtol=1e-5, atol=1e-5)
