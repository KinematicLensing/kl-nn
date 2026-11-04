from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "shared_job_scratch.sh"
LAUNCHERS = (
    ROOT / "data_generate" / "generate_desi_test_sets.slurm",
    ROOT / "data_generate" / "generate_simulator_v3.slurm",
    ROOT / "data_generate" / "make_database_simulator_v3.slurm",
    ROOT / "data_generate" / "merge_database_simulator_v3.slurm",
    ROOT / "arch" / "cache_posteriors.slurm",
)


def test_shared_scratch_helper_is_syntax_valid_and_scoped():
    subprocess.run(["bash", "-n", str(HELPER)], check=True)
    text = HELPER.read_text(encoding="utf-8")
    assert 'source "$(dirname "${BASH_SOURCE[0]}")/klnn_paths.sh"' in text
    assert 'readonly KLNN_SHARED_TMP_ROOT="${KLNN_SHARED_TMP}"' in text
    assert 'SHARED_JOB_TMPDIR="${KLNN_SHARED_TMP_ROOT}/klnn-' in text
    assert 'export TMPDIR="${SHARED_JOB_TMPDIR}"' in text
    assert 'export TMP="${SHARED_JOB_TMPDIR}"' in text
    assert 'export TEMP="${SHARED_JOB_TMPDIR}"' in text
    assert 'trap _cleanup_shared_job_scratch EXIT' in text
    assert 'rm -rf -- "${scratch}"' in text
    assert '"${scratch}" != "${KLNN_SHARED_TMP_ROOT}"/klnn-*' in text


def test_every_xu_pipeline_launcher_installs_shared_scratch_cleanup():
    source = 'source "${KLNN_REPO_ROOT}/shared_job_scratch.sh"'
    root_default = 'KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"'
    paths = 'source "${KLNN_REPO_ROOT}/klnn_paths.sh"'
    for launcher in LAUNCHERS:
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text(encoding="utf-8")
        assert text.count(root_default) == 1
        assert text.count(paths) == 1
        assert text.count(source) == 1
        assert text.count("setup_shared_job_scratch ") == 1
