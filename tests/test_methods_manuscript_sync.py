from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_methods_manuscript_matches_implementation_sources():
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "repo_reports" / "check_methods_manuscript.py"),
            "--check",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_methods_manuscript_retains_required_sections():
    manuscript = (
        REPOSITORY_ROOT / "repo_reports" / "METHODS_MANUSCRIPT.md"
    ).read_text(encoding="utf-8")
    required = (
        "## Generating DESI-like image and spectra",
        "## Neural Network Architecture",
        "## References",
    )
    required_method_details = (
        r"\bm{D}(\bm{\theta})",
        "This on-the-fly construction has two practical motivations.",
        "simulation controls",
        "diagnostics of example difficulty",
        r"\widehat M_I",
        r"\widehat M_S",
    )
    assert all(manuscript.count(heading) == 1 for heading in required)
    assert "klnn-methods-source-sha256: PENDING" not in manuscript
    assert all(detail in manuscript for detail in required_method_details)


def test_overleaf_tex_matches_markdown_manuscript():
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "repo_reports" / "render_methods_tex.py"),
            "--check",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

