from __future__ import annotations

import importlib.util
import os
import shutil
from os.path import basename, dirname, isfile, join
from types import ModuleType

import config

DEFAULT_SHARED_ROOT = "/ocean/projects/phy250048p/shared"
DEFAULT_CONFIGS_ROOT = join(DEFAULT_SHARED_ROOT, "configs")
DEFAULT_NETWORKS_ROOT = join(DEFAULT_SHARED_ROOT, "networks")


def get_model_config_path(model_name: str, configs_root: str = DEFAULT_CONFIGS_ROOT) -> str:
    return join(configs_root, f"cfg_{model_name}.json")


def get_network_snapshot_path(model_name: str, networks_root: str = DEFAULT_NETWORKS_ROOT) -> str:
    return join(networks_root, f"networks_{model_name}.py")


def load_model_config(
    model_name: str,
    *,
    configs_root: str = DEFAULT_CONFIGS_ROOT,
    allow_fallback_current: bool = True,
) -> config.ModelConfig:
    path = get_model_config_path(model_name, configs_root=configs_root)
    if isfile(path):
        return config.ModelConfig.from_json(path)

    if allow_fallback_current and config.train["model_name"] == model_name:
        return config.MODEL_CONFIG

    raise FileNotFoundError(
        f"Model config JSON not found for '{model_name}' at {path}. "
        "Train/snapshot the model first or provide a config path explicitly."
    )


def save_model_artifacts(
    model_config: config.ModelConfig,
    *,
    train_type: str = 'train',
    model_name: str | None = None,
    configs_root: str = DEFAULT_CONFIGS_ROOT,
    networks_root: str = DEFAULT_NETWORKS_ROOT,
    networks_source_path: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    train_config = model_config.train if train_type == 'train' else model_config.pretrain
    effective_model_name = model_name or train_config.model_name
    if not effective_model_name:
        raise ValueError("model_name cannot be empty")

    os.makedirs(configs_root, exist_ok=True)
    os.makedirs(networks_root, exist_ok=True)

    config_path = get_model_config_path(effective_model_name, configs_root=configs_root)
    network_path = get_network_snapshot_path(effective_model_name, networks_root=networks_root)
    source_path = networks_source_path or join(dirname(__file__), "networks.py")

    if not overwrite:
        for path in (config_path, network_path):
            if isfile(path):
                raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

    model_config.to_json(config_path, indent=2)
    shutil.copy2(source_path, network_path)

    return {
        "model_name": effective_model_name,
        "config_path": config_path,
        "network_path": network_path,
    }


def load_networks_module_for_model(
    model_name: str,
    *,
    networks_root: str = DEFAULT_NETWORKS_ROOT,
) -> ModuleType:
    snapshot_path = get_network_snapshot_path(model_name, networks_root=networks_root)
    if not isfile(snapshot_path):
        raise FileNotFoundError(f"Archived networks file not found: {snapshot_path}")

    module_name = f"_archived_networks_{model_name}"
    spec = importlib.util.spec_from_file_location(module_name, snapshot_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {snapshot_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_model_name_from_checkpoint_path(checkpoint_path: str | None) -> str | None:
    if checkpoint_path is None:
        return None
    parent = basename(dirname(checkpoint_path.rstrip("/")))
    return parent or None
