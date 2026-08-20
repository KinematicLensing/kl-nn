import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import config
import model_registry


ROOT = Path(__file__).resolve().parents[1]


def test_current_model_config_snapshot_roundtrips_exactly(tmp_path):
    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.model_name = "snapshot-model"
    config_root = tmp_path / "configs"
    config_root.mkdir()
    path = Path(
        model_registry.get_model_config_path(
            "snapshot-model", configs_root=str(config_root)
        )
    )
    configured.to_json(str(path), indent=2)

    loaded = model_registry.load_model_config(
        "snapshot-model", configs_root=str(config_root)
    )

    assert loaded.to_dict() == configured.to_dict()
    assert json.loads(path.read_text(encoding="utf-8")) == configured.to_dict()


def test_save_artifacts_snapshots_config_network_and_circular_helper(
    tmp_path, monkeypatch
):
    config_root = tmp_path / "configs"
    network_root = tmp_path / "snapshots"
    source_root = tmp_path / "source"
    source_root.mkdir()
    network_source = source_root / "networks.py"
    helper_source = source_root / "circular_spline.py"
    network_source.write_text(
        "from circular_spline import ORIGIN\nSNAPSHOT_ORIGIN = ORIGIN\n",
        encoding="utf-8",
    )
    helper_source.write_text("ORIGIN = 'snapshotted'\n", encoding="utf-8")

    configured = copy.deepcopy(config.MODEL_CONFIG)
    configured.train.model_name = "model-with-hyphens and spaces"
    artifacts = model_registry.save_model_artifacts(
        configured,
        model_name=configured.train.model_name,
        configs_root=str(config_root),
        networks_root=str(network_root),
        networks_source_path=str(network_source),
    )

    assert set(artifacts) == {
        "model_name",
        "config_path",
        "network_path",
        "circular_spline_path",
    }
    assert Path(artifacts["network_path"]).read_text(
        encoding="utf-8"
    ) == network_source.read_text(encoding="utf-8")
    assert Path(artifacts["circular_spline_path"]).read_text(
        encoding="utf-8"
    ) == helper_source.read_text(encoding="utf-8")
    assert config.ModelConfig.from_json(artifacts["config_path"]).to_dict() == (
        configured.to_dict()
    )

    live_helper = ModuleType("circular_spline")
    live_helper.ORIGIN = "live"
    monkeypatch.setitem(sys.modules, "circular_spline", live_helper)
    loaded = model_registry.load_networks_module_for_model(
        configured.train.model_name, networks_root=str(network_root)
    )

    assert loaded.SNAPSHOT_ORIGIN == "snapshotted"
    assert loaded.__name__.isidentifier()
    assert sys.modules["circular_spline"] is live_helper


def test_full_snapshot_loads_from_package_root_and_restores_import_state(tmp_path):
    code = r'''
import sys
from pathlib import Path

import arch.config as config
import arch.model_registry as registry

root = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
arch_dir = str(repo_root / "arch")
assert arch_dir not in sys.path
aliases = ("config", "data", "utils", "circular_spline")
before_path = list(sys.path)
before_aliases = {name: sys.modules.get(name) for name in aliases}

registry.save_model_artifacts(
    config.MODEL_CONFIG,
    model_name="package-root",
    configs_root=str(root / "configs"),
    networks_root=str(root / "networks"),
)
loaded = registry.load_networks_module_for_model(
    "package-root", networks_root=str(root / "networks")
)
assert loaded.KLNPE.__name__ == "KLNPE"
assert sys.path == before_path
for name in aliases:
    assert sys.modules.get(name) is before_aliases[name]

snapshot = Path(
    registry.get_network_snapshot_path(
        "package-root", networks_root=str(root / "networks")
    )
)
snapshot.write_text(
    "import config\nimport data\nimport utils\n"
    "from circular_spline import rational_quadratic_spline\n"
    "raise RuntimeError('snapshot boom')\n",
    encoding="utf-8",
)
try:
    registry.load_networks_module_for_model(
        "package-root", networks_root=str(root / "networks")
    )
except RuntimeError as error:
    assert str(error) == "snapshot boom"
else:
    raise AssertionError("broken snapshot unexpectedly loaded")
assert sys.path == before_path
for name in aliases:
    assert sys.modules.get(name) is before_aliases[name]
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path), str(ROOT)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_saving_current_network_import_without_helper_fails_closed(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    network_source = source_root / "networks.py"
    network_source.write_text(
        "from circular_spline import missing\n", encoding="utf-8"
    )

    with pytest.raises(FileNotFoundError, match="helper source"):
        model_registry.save_model_artifacts(
            config.MODEL_CONFIG,
            model_name="incomplete",
            configs_root=str(tmp_path / "configs"),
            networks_root=str(tmp_path / "snapshots"),
            networks_source_path=str(network_source),
        )


def test_loading_snapshot_without_circular_helper_fails_closed(tmp_path):
    network_root = tmp_path / "snapshots"
    network_root.mkdir()
    snapshot = Path(
        model_registry.get_network_snapshot_path(
            "incomplete", networks_root=str(network_root)
        )
    )
    snapshot.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="incomplete.*circular-spline"):
        model_registry.load_networks_module_for_model(
            "incomplete", networks_root=str(network_root)
        )


def test_missing_current_config_and_network_snapshots_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="Current-schema model config"):
        model_registry.load_model_config(
            "missing", configs_root=str(tmp_path / "configs")
        )
    with pytest.raises(FileNotFoundError, match="Current networks snapshot"):
        model_registry.load_networks_module_for_model(
            "missing", networks_root=str(tmp_path / "snapshots")
        )
