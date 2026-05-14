import json
from pathlib import Path

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
