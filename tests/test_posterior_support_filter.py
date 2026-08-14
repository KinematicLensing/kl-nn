import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_filter():
    path = (
        Path(__file__).resolve().parents[1]
        / "arch"
        / "diagnostics"
        / "posterior_support_filter.py"
    )
    spec = importlib.util.spec_from_file_location("posterior_support_filter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


support_filter = _load_filter()

FEATURES = [
    "g1",
    "g2",
    "theta_int",
    "sini",
    "v0",
    "vcirc",
    "rscale",
    "hlr",
]
BOUNDS = np.asarray(
    [
        [-0.1, 0.1],
        [-0.1, 0.1],
        [-np.pi, np.pi],
        [0.0, 1.0],
        [-30.0, 30.0],
        [60.0, 540.0],
        [0.1, 2.0],
        [0.1, 3.0],
    ]
)


def _valid_draw(theta=0.0):
    return np.asarray([0.0, 0.0, theta, 0.5, 0.0, 200.0, 1.0, 1.0])


def test_joint_filter_rejects_any_parameter_and_summarizes_theta_circularly():
    draws = np.stack(
        [
            _valid_draw(3.12),
            _valid_draw(-3.12),
            _valid_draw(3.10),
            _valid_draw(0.0),
            _valid_draw(0.0),
        ]
    )
    draws[3, 0] = 0.1001
    draws[4, 5] = 540.001

    summary, retention, diagnostics = support_filter.summarize_partition(
        draws[None, None], FEATURES, BOUNDS
    )

    np.testing.assert_allclose(retention, [[0.6]])
    assert abs(summary[0, 0, 1, 2]) > 3.0
    np.testing.assert_allclose(summary[0, 0, 1, 0], 0.0)
    assert diagnostics["jointly_retained_count_by_mode"] == [3]
    assert diagnostics["per_feature_rejected_count_by_mode"][0][0] == 1
    assert diagnostics["per_feature_rejected_count_by_mode"][0][5] == 1


def test_empty_support_returns_nan_summary():
    draws = _valid_draw()[None, None, None]
    draws[0, 0, 0, 3] = np.nan

    summary, retention, diagnostics = support_filter.summarize_partition(
        draws, FEATURES, BOUNDS
    )

    assert np.isnan(summary).all()
    np.testing.assert_array_equal(retention, [[0.0]])
    assert diagnostics["zero_retained_galaxies_by_mode"] == [1]


def test_process_cache_writes_compact_provenance_without_modifying_samples(tmp_path):
    cache = tmp_path / "cache" / "example_model" / "example_dataset"
    sample_dir = cache / "sample"
    sample_dir.mkdir(parents=True)
    parts = []
    originals = []
    for index in range(2):
        values = np.tile(_valid_draw(), (1, 2, 4, 1)).astype(np.float32)
        values[0, index, 0, index] = 999.0
        path = sample_dir / f"part{index}of2.npy"
        np.save(path, values)
        parts.append(path)
        originals.append(path.read_bytes())

    config_root = tmp_path / "configs"
    config_root.mkdir()
    config_path = config_root / "cfg_example_model.json"
    config_path.write_text(
        json.dumps(
            {
                "train": {"feature_names": FEATURES},
                "par_ranges": {
                    name: [float(low), float(high)]
                    for name, (low, high) in zip(FEATURES, BOUNDS)
                },
            }
        )
    )

    manifest = support_filter.process_cache(cache, configs_root=config_root)

    assert [path.read_bytes() for path in parts] == originals
    assert len(manifest["source_parts"]) == 2
    assert manifest["inclusive_bounds"]["vcirc"] == [60.0, 540.0]
    assert len(manifest["archived_config_sha256"]) == 64
    for index in range(2):
        summary = np.load(
            cache / support_filter.SUMMARY_DIR / f"part{index}of2.npy"
        )
        retention = np.load(
            cache / support_filter.RETENTION_DIR / f"part{index}of2.npy"
        )
        assert summary.shape == (1, 2, 3, 8)
        assert retention.shape == (1, 2)
        np.testing.assert_allclose(np.sort(retention[0]), [0.75, 1.0])
        part_meta = json.loads(
            (cache / support_filter.META_DIR / f"part{index}of2.json").read_text()
        )
        assert part_meta["source_sample_part"] == f"sample/part{index}of2.npy"
        assert part_meta["joint_rule"].startswith("retain iff every feature")
    assert (cache / support_filter.META_DIR / "manifest.json").is_file()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        support_filter.process_cache(cache, configs_root=config_root)
