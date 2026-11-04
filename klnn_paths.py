"""Single source of machine-specific paths for kl-nn and kl-nn-analysis.

All paths live in ``paths.json`` next to this file. Resolution order for a key:

1. environment variable ``KLNN_PATH_<KEY>`` (upper case), e.g. ``KLNN_PATH_MODELS``;
2. ``paths.local.json`` (optional, gitignored) next to this file;
3. ``paths.json``.

Values may reference other keys as ``{key}``; ``{repo_root}`` is the directory of this file.

Python:  ``from klnn_paths import path; path("models")``
Shell:   ``python klnn_paths.py models``  (or ``klnn_path models`` after sourcing klnn_paths.sh)
"""
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent
_CONFIG = REPO_ROOT / "paths.json"
_LOCAL_CONFIG = REPO_ROOT / "paths.local.json"
_REF = re.compile(r"\{([A-Za-z0-9_]+)\}")
_NON_PATH_KEYS = {"conda_env"}


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {k: v for k, v in data.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def load() -> Dict[str, str]:
    """Return every key resolved to a string (templates expanded, overrides applied)."""
    raw = _read(_CONFIG)
    raw.update(_read(_LOCAL_CONFIG))
    for key in list(raw):
        env = os.environ.get(f"KLNN_PATH_{key.upper()}")
        if env:
            raw[key] = env
    raw["repo_root"] = str(REPO_ROOT)

    resolved = {}  # type: Dict[str, str]

    def resolve(key: str, stack: Tuple[str, ...] = ()) -> str:
        if key in resolved:
            return resolved[key]
        if key not in raw:
            raise KeyError(f"unknown path key {key!r} (see {_CONFIG})")
        if key in stack:
            raise ValueError(f"circular reference in {_CONFIG}: {' -> '.join(stack + (key,))}")
        value = _REF.sub(lambda m: resolve(m.group(1), stack + (key,)), str(raw[key]))
        if key not in _NON_PATH_KEYS:
            value = os.path.normpath(os.path.expanduser(value))
        resolved[key] = value
        return value

    for key in raw:
        resolve(key)
    return resolved


def path(key: str) -> Path:
    """Return the configured path for ``key`` as a Path."""
    return Path(load()[key])


def value(key: str) -> str:
    """Return the configured value for ``key`` as a string (for non-path keys such as conda_env)."""
    return load()[key]


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("--all", "-a"):
        for k, v in sorted(load().items()):
            print(f"{k}={v}")
    elif len(sys.argv) == 2:
        try:
            print(value(sys.argv[1]))
        except KeyError as exc:
            print(exc, file=sys.stderr)
            sys.exit(2)
    else:
        print("usage: python klnn_paths.py KEY | --all", file=sys.stderr)
        sys.exit(2)
