"""The JSON path config (paths.json) is the only place machine paths live."""

import json
import re
import subprocess
from pathlib import Path

import pytest

import klnn_paths

ROOT = Path(__file__).resolve().parents[1]


def test_every_key_resolves_without_templates():
    resolved = klnn_paths.load()
    assert resolved["repo_root"] == str(ROOT)
    for key, value in resolved.items():
        assert "{" not in value, (key, value)
    assert klnn_paths.path("datasets") == Path(resolved["shared_root"]) / "datasets"
    assert klnn_paths.path("kl_tools_root") == ROOT.parent / "kl-tools"
    assert klnn_paths.value("conda_env") == json.loads((ROOT / "paths.json").read_text())["conda_env"]


def test_environment_override_wins(monkeypatch):
    klnn_paths.load.cache_clear()
    monkeypatch.setenv("KLNN_PATH_SHARED_ROOT", "/elsewhere/shared")
    try:
        assert klnn_paths.path("models") == Path("/elsewhere/shared/models")
    finally:
        monkeypatch.delenv("KLNN_PATH_SHARED_ROOT")
        klnn_paths.load.cache_clear()


def test_unknown_key_is_an_error():
    with pytest.raises(KeyError):
        klnn_paths.path("no_such_key")


def test_shell_helper_exports_every_key():
    keys = [k for k in json.loads((ROOT / "paths.json").read_text()) if not k.startswith("_")]
    script = f'set -u; source "{ROOT}/klnn_paths.sh"; ' + "; ".join(
        f'echo "{k}=${{KLNN_{k.upper()}}}"' for k in keys
    )
    out = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"}).stdout
    shell = dict(line.split("=", 1) for line in out.splitlines())
    expected = klnn_paths.load()
    for key in keys:
        assert shell[key] == expected[key], key


def test_no_hardcoded_machine_paths_outside_paths_json():
    pattern = re.compile(r"/ocean/|/jet/home|\.conda/envs")
    offenders = []
    for path in ROOT.rglob("*"):
        if path.suffix not in {".py", ".sh", ".slurm"} or ".git" in path.parts:
            continue
        if path.name in {"test_klnn_paths.py", "klnn_paths.sh"}:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_slurm_launchers_leave_log_paths_to_submit_wrapper():
    for launcher in ROOT.rglob("*.slurm"):
        text = launcher.read_text(encoding="utf-8")
        assert not re.search(r"^#SBATCH\s+(--output|-o)\b", text, re.M), launcher
        assert 'source "${KLNN_REPO_ROOT}/klnn_paths.sh"' in text, launcher
    subprocess.run(["bash", "-n", str(ROOT / "submit.sh")], check=True)
