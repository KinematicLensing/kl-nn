import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCH = ROOT / "arch"

for path in (ROOT, ARCH):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
