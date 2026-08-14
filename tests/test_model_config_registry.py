import json
import sys
from pathlib import Path
from types import ModuleType

import config
import model_registry


def test_model_config_json_roundtrip(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    config.MODEL_CONFIG.to_json(str(cfg_path), indent=2)

    loaded = config.ModelConfig.from_json(str(cfg_path))

    assert loaded.train.model_name == config.MODEL_CONFIG.train.model_name
    assert loaded.train.feature_names == config.MODEL_CONFIG.train.feature_names
    assert loaded.par_ranges["vcirc"] == config.MODEL_CONFIG.par_ranges["vcirc"]

    parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert "train" in parsed
    assert "model_name" in parsed["train"]


def test_save_model_artifacts_writes_json_and_network_snapshot(tmp_path):
    cfg_root = tmp_path / "configs"
    net_root = tmp_path / "networks"
    src_network = tmp_path / "networks_src.py"
    src_network.write_text("class ForkCNN:\n    pass\n", encoding="utf-8")

    artifacts = model_registry.save_model_artifacts(
        config.MODEL_CONFIG,
        model_name="unit-test-model",
        configs_root=str(cfg_root),
        networks_root=str(net_root),
        networks_source_path=str(src_network),
    )

    config_path = Path(artifacts["config_path"])
    network_path = Path(artifacts["network_path"])
    assert config_path.is_file()
    assert network_path.is_file()
    assert "unit-test-model" in config_path.name
    assert "unit-test-model" in network_path.name
    assert "class ForkCNN" in network_path.read_text(encoding="utf-8")


def test_circular_spline_helper_is_snapshotted_and_used_when_loading(
    tmp_path, monkeypatch
):
    cfg_root = tmp_path / "configs"
    net_root = tmp_path / "snapshots"
    source_root = tmp_path / "source"
    source_root.mkdir()
    src_network = source_root / "networks.py"
    src_helper = source_root / "circular_spline.py"
    src_network.write_text(
        "from circular_spline import HelperMarker\n",
        encoding="utf-8",
    )
    src_helper.write_text(
        "class HelperMarker:\n" "    origin = 'snapshotted'\n",
        encoding="utf-8",
    )

    artifacts = model_registry.save_model_artifacts(
        config.MODEL_CONFIG,
        model_name="model-with-hyphens and spaces",
        configs_root=str(cfg_root),
        networks_root=str(net_root),
        networks_source_path=str(src_network),
    )

    helper_path = Path(artifacts["circular_spline_path"])
    assert helper_path.is_file()
    assert helper_path.read_text(encoding="utf-8") == src_helper.read_text(
        encoding="utf-8"
    )

    live_helper = ModuleType("circular_spline")
    live_marker = type("HelperMarker", (), {"origin": "live"})
    live_helper.HelperMarker = live_marker
    monkeypatch.setitem(sys.modules, "circular_spline", live_helper)

    loaded = model_registry.load_networks_module_for_model(
        "model-with-hyphens and spaces",
        networks_root=str(net_root),
    )

    assert loaded.HelperMarker.origin == "snapshotted"
    assert loaded.HelperMarker is not live_marker
    assert loaded.HelperMarker.__module__.isidentifier()
    assert sys.modules["circular_spline"] is live_helper


def test_legacy_network_snapshot_without_helper_uses_existing_import_behavior(
    tmp_path, monkeypatch
):
    net_root = tmp_path / "snapshots"
    net_root.mkdir()
    snapshot = Path(
        model_registry.get_network_snapshot_path("legacy", networks_root=str(net_root))
    )
    snapshot.write_text("from circular_spline import VALUE\n", encoding="utf-8")

    live_helper = ModuleType("circular_spline")
    live_helper.VALUE = "legacy-live-value"
    monkeypatch.setitem(sys.modules, "circular_spline", live_helper)

    loaded = model_registry.load_networks_module_for_model(
        "legacy", networks_root=str(net_root)
    )

    assert loaded.VALUE == "legacy-live-value"
    assert sys.modules["circular_spline"] is live_helper
