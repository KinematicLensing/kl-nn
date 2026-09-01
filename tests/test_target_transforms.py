import numpy as np
import pandas as pd
import pytest
import torch

import config
from arch.utils import (
    denormalization_logabsdet,
    denormalize,
    normalize_targets,
)
from data_generate.make_database import normalize_sample_table


@pytest.mark.parametrize("backend", ("numpy", "torch"))
def test_log_halpha_normalization_and_physical_round_trip(backend):
    physical = np.asarray(
        [
            [0.1, 1.0e-17],
            [2.55, np.sqrt(1.0e-17 * 1.0e-14)],
            [5.0, 1.0e-14],
        ],
        dtype=np.float64,
    )
    ranges = {
        "hlr": [0.1, 5.0],
        "halpha_flux_true": [1.0e-17, 1.0e-14],
    }
    transforms = {"hlr": "identity", "halpha_flux_true": "log10"}
    values = torch.from_numpy(physical) if backend == "torch" else physical
    normalized = normalize_targets(
        values,
        ranges,
        target_transforms=transforms,
    )
    expected = np.asarray([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    actual = normalized.detach().numpy() if backend == "torch" else normalized
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
    restored = denormalize(
        normalized,
        ranges,
        target_transforms=transforms,
    )
    actual = restored.detach().numpy() if backend == "torch" else restored
    np.testing.assert_allclose(actual, physical, rtol=2e-14, atol=0.0)


def test_log_halpha_denormalization_jacobian_matches_finite_difference():
    ranges = {"halpha_flux_true": [1.0e-17, 1.0e-14]}
    transforms = {"halpha_flux_true": "log10"}
    normalized = np.asarray([[-0.4], [0.2], [0.8]], dtype=np.float64)
    analytic = np.exp(
        denormalization_logabsdet(
            normalized,
            ranges,
            target_transforms=transforms,
        )
    )
    epsilon = 1e-6
    upper = denormalize(
        normalized + epsilon,
        ranges,
        target_transforms=transforms,
    )[:, 0]
    lower = denormalize(
        normalized - epsilon,
        ranges,
        target_transforms=transforms,
    )[:, 0]
    numerical = (upper - lower) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-9, atol=0.0)


def test_database_normalization_uses_registry_and_preserves_metadata():
    rows = []
    for flux, expected in (
        (1.0e-17, -1.0),
        (np.sqrt(1.0e-31), 0.0),
        (1.0e-14, 1.0),
    ):
        row = {
            name: 0.5 * (low + high)
            for name, (low, high) in config.par_ranges.items()
        }
        row["sini"] = np.sqrt(1.0 - row.pop("cosi") ** 2)
        row["halpha_flux_true"] = flux
        row["image_snr"] = 317.0
        row["expected"] = expected
        rows.append(row)
    physical = pd.DataFrame(rows)
    normalized = normalize_sample_table(physical, config.par_ranges)
    np.testing.assert_allclose(
        normalized["halpha_flux_true"], physical["expected"], atol=1e-12
    )
    np.testing.assert_array_equal(normalized["image_snr"], physical["image_snr"])
    np.testing.assert_array_equal(normalized["expected"], physical["expected"])


def test_database_materializes_and_normalizes_cosi_from_simulator_sini():
    base = {
        name: 0.5 * (low + high)
        for name, (low, high) in config.par_ranges.items()
        if name != "cosi"
    }
    physical = pd.DataFrame([base, base, base])
    physical["sini"] = [1.0, np.sqrt(0.75), 0.0]

    normalized = normalize_sample_table(physical, config.par_ranges)

    np.testing.assert_allclose(normalized["cosi"], [-1.0, 0.0, 1.0], atol=1e-15)


@pytest.mark.parametrize("bad_sini", (-0.1, 1.1, np.nan))
def test_database_rejects_invalid_simulator_sini(bad_sini):
    row = {
        name: 0.5 * (low + high)
        for name, (low, high) in config.par_ranges.items()
        if name != "cosi"
    }
    row["sini"] = bad_sini
    with pytest.raises(ValueError, match="sini values"):
        normalize_sample_table(pd.DataFrame([row]), config.par_ranges)


@pytest.mark.parametrize("bad", (0.0, -1.0, np.nan))
def test_log_target_normalization_rejects_nonpositive_or_nonfinite_flux(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        normalize_targets(
            np.asarray([[bad]]),
            {"halpha_flux_true": [1.0e-17, 1.0e-14]},
            target_transforms={"halpha_flux_true": "log10"},
        )
