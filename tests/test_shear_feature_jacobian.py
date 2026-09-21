from __future__ import annotations

import numpy as np
import pytest

from diagnostics.shear_feature_jacobian import (
    FEATURE_DIM,
    IMAGE_FEATURE_DIM,
    METADATA_FEATURE_DIM,
    SPECTRAL_FEATURE_DIM,
    branch_delta_metrics,
    cosine,
    pack_product,
    parse_args,
    split_features,
)


def test_pack_product_pads_spectra_and_adds_a_channel():
    product = {
        "image": np.ones((48, 48), dtype=np.float64),
        "spectra": np.arange(5 * 61, dtype=np.float64).reshape(5, 61),
        "positions": np.arange(10, dtype=np.float64).reshape(5, 2),
    }
    packed = pack_product(product)

    assert packed["img"].shape == (1, 48, 48)
    assert packed["spec"].shape == (1, 5, 64)
    assert packed["fib_pos"].shape == (5, 2)
    assert packed["spec"][0, 0, :61] == pytest.approx(product["spectra"][0])
    assert packed["spec"][0, :, 61:].sum() == 0.0


def test_pack_product_rejects_overlong_spectra():
    product = {
        "image": np.zeros((48, 48)),
        "spectra": np.zeros((5, 65)),
        "positions": np.zeros((5, 2)),
    }
    with pytest.raises(ValueError, match="exceeds wavelength_count"):
        pack_product(product)


def test_split_features_uses_architecture_slices():
    features = np.arange(FEATURE_DIM, dtype=np.float64)
    parts = split_features(features)

    assert parts["image"].shape == (IMAGE_FEATURE_DIM,)
    assert parts["spectral"].shape == (SPECTRAL_FEATURE_DIM,)
    assert parts["metadata"].shape == (METADATA_FEATURE_DIM,)
    assert parts["concat"].shape == (FEATURE_DIM,)
    assert parts["spectral"][0] == IMAGE_FEATURE_DIM
    assert parts["metadata"][0] == IMAGE_FEATURE_DIM + SPECTRAL_FEATURE_DIM


def test_cosine_of_known_pairs():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)


def test_branch_delta_metrics_isolates_a_spectral_sign_flip():
    rng = np.random.default_rng(0)
    image = rng.normal(size=IMAGE_FEATURE_DIM)
    spec = rng.normal(size=SPECTRAL_FEATURE_DIM)
    meta = rng.normal(size=METADATA_FEATURE_DIM)
    small_lo = np.concatenate([image, spec, meta])
    small_hi = np.concatenate([image + 0.1 * image, spec + 0.1 * spec, meta])
    large_lo = small_lo.copy()
    large_hi = np.concatenate([image + 0.1 * image, spec - 0.1 * spec, meta])

    metrics = branch_delta_metrics(small_lo, small_hi, large_lo, large_hi)

    assert metrics["image"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["spectral"]["cosine"] == pytest.approx(-1.0, abs=1e-6)
    assert metrics["metadata"]["cosine"] == pytest.approx(1.0)


def test_cli_defaults_point_at_stage_six():
    args = parse_args([])
    assert args.report_dir.name == "06_feature_jacobian"
    assert args.data_space_json.name == "report.json"
    assert args.checkpoint_suffix == "best"
