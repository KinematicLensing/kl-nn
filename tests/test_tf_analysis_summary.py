import importlib.util
import math
from pathlib import Path

import numpy as np


def _load_tf_analysis():
    path = Path(__file__).resolve().parents[1] / "arch" / "[scr]_tf_analysis.py"
    spec = importlib.util.spec_from_file_location("tf_analysis_summary_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_posterior_summary_uses_circular_theta_mean_at_seam():
    module = _load_tf_analysis()
    feature_names = [
        "g1",
        "g2",
        "theta_int",
        "sini",
        "v0",
        "vcirc",
        "rscale",
        "hlr",
    ]
    samples = np.zeros((2, len(feature_names)), dtype=np.float64)
    samples[:, 0] = (0.1, 0.3)
    samples[:, 2] = (math.pi - 0.05, -math.pi + 0.05)

    summary = module.summarize_posterior_samples(samples, feature_names)

    np.testing.assert_allclose(summary[1, 0], 0.2)
    assert abs(abs(summary[1, 2]) - math.pi) < 1e-12
    circular_residuals = np.arctan2(
        np.sin(summary[:, 2] - summary[1, 2]),
        np.cos(summary[:, 2] - summary[1, 2]),
    )
    assert np.max(np.abs(circular_residuals)) < 0.051


def test_posterior_summary_validates_shape():
    module = _load_tf_analysis()

    try:
        module.summarize_posterior_samples(np.zeros((3, 2, 1)), ["g1"])
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("invalid sample shape should be rejected")



def test_partition_range_preserves_provenance_under_profiling():
    module = _load_tf_analysis()

    assert module.resolve_partition_range(0, 1000, 10000) == (0, 1000)
    assert module.resolve_partition_range(7, 1000, 10000) == (7000, 8000)


def test_partition_range_rejects_invalid_or_overflowing_ranges():
    module = _load_tf_analysis()

    for args in ((0, 0, 10000), (10, 1000, 10000)):
        try:
            module.resolve_partition_range(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid partition range should fail: {args}")
