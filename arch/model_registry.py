from __future__ import annotations

import importlib.util
import hashlib
import os
import re
import shutil
import sys
from os.path import basename, dirname, isfile, join
from types import ModuleType

import config

DEFAULT_SHARED_ROOT = "/ocean/projects/phy250048p/shared"
DEFAULT_CONFIGS_ROOT = join(DEFAULT_SHARED_ROOT, "configs")
DEFAULT_NETWORKS_ROOT = join(DEFAULT_SHARED_ROOT, "networks")

_CIRCULAR_SPLINE_IMPORT_RE = re.compile(
    r"(?m)^\s*(?:from\s+circular_spline\s+import\b|import\s+circular_spline\b)"
)
_MISSING_MODULE = object()


def get_model_config_path(
    model_name: str, configs_root: str = DEFAULT_CONFIGS_ROOT
) -> str:
    return join(configs_root, f"cfg_{model_name}.json")


def get_network_snapshot_path(
    model_name: str, networks_root: str = DEFAULT_NETWORKS_ROOT
) -> str:
    return join(networks_root, f"networks_{model_name}.py")


def get_circular_spline_snapshot_path(
    model_name: str,
    networks_root: str = DEFAULT_NETWORKS_ROOT,
) -> str:
    return join(networks_root, f"circular_spline_{model_name}.py")


def _archived_module_name(prefix: str, snapshot_path: str) -> str:
    """Return a deterministic, import-safe name for an artifact module."""
    digest = hashlib.sha256(os.path.abspath(snapshot_path).encode("utf-8")).hexdigest()[
        :20
    ]
    return f"_archived_{prefix}_{digest}"


def _load_module_from_path(module_name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name, _MISSING_MODULE)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is _MISSING_MODULE:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


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
    train_type: str = "train",
    model_name: str | None = None,
    configs_root: str = DEFAULT_CONFIGS_ROOT,
    networks_root: str = DEFAULT_NETWORKS_ROOT,
    networks_source_path: str | None = None,
    circular_spline_source_path: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    train_config = (
        model_config.train if train_type == "train" else model_config.pretrain
    )
    effective_model_name = model_name or train_config.model_name
    if not effective_model_name:
        raise ValueError("model_name cannot be empty")

    os.makedirs(configs_root, exist_ok=True)
    os.makedirs(networks_root, exist_ok=True)

    config_path = get_model_config_path(effective_model_name, configs_root=configs_root)
    network_path = get_network_snapshot_path(
        effective_model_name, networks_root=networks_root
    )
    source_path = networks_source_path or join(dirname(__file__), "networks.py")
    with open(source_path, "r", encoding="utf-8") as source_file:
        network_source = source_file.read()

    snapshots_circular_spline = bool(_CIRCULAR_SPLINE_IMPORT_RE.search(network_source))
    circular_source_path = circular_spline_source_path or join(
        dirname(source_path), "circular_spline.py"
    )
    circular_snapshot_path = get_circular_spline_snapshot_path(
        effective_model_name, networks_root=networks_root
    )
    if snapshots_circular_spline and not isfile(circular_source_path):
        raise FileNotFoundError(
            "The networks source imports circular_spline, but its helper source "
            f"was not found at {circular_source_path}."
        )

    if not overwrite:
        artifact_paths = [config_path, network_path]
        if snapshots_circular_spline:
            artifact_paths.append(circular_snapshot_path)
        for path in artifact_paths:
            if isfile(path):
                raise FileExistsError(
                    f"Refusing to overwrite existing artifact: {path}"
                )

    model_config.to_json(config_path, indent=2)
    shutil.copy2(source_path, network_path)
    if snapshots_circular_spline:
        shutil.copy2(circular_source_path, circular_snapshot_path)

    artifacts = {
        "model_name": effective_model_name,
        "config_path": config_path,
        "network_path": network_path,
    }
    if snapshots_circular_spline:
        artifacts["circular_spline_path"] = circular_snapshot_path
    return artifacts


def load_networks_module_for_model(
    model_name: str,
    *,
    networks_root: str = DEFAULT_NETWORKS_ROOT,
) -> ModuleType:
    snapshot_path = get_network_snapshot_path(model_name, networks_root=networks_root)
    if not isfile(snapshot_path):
        raise FileNotFoundError(f"Archived networks file not found: {snapshot_path}")

    helper_path = get_circular_spline_snapshot_path(
        model_name, networks_root=networks_root
    )
    previous_circular_spline = _MISSING_MODULE
    if isfile(helper_path):
        helper_name = _archived_module_name("circular_spline", helper_path)
        helper_module = _load_module_from_path(helper_name, helper_path)
        previous_circular_spline = sys.modules.get("circular_spline", _MISSING_MODULE)
        sys.modules["circular_spline"] = helper_module

    # Model names routinely contain hyphens and other punctuation, so derive
    # an import-safe module name from the artifact path.
    module_name = _archived_module_name("networks", snapshot_path)
    try:
        return _load_module_from_path(module_name, snapshot_path)
    finally:
        if isfile(helper_path):
            if previous_circular_spline is _MISSING_MODULE:
                sys.modules.pop("circular_spline", None)
            else:
                sys.modules["circular_spline"] = previous_circular_spline


def infer_model_name_from_checkpoint_path(checkpoint_path: str | None) -> str | None:
    if checkpoint_path is None:
        return None
    parent = basename(dirname(checkpoint_path.rstrip("/")))
    return parent or None
