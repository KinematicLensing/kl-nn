#!/bin/bash
# Submit the shared-η likelihood-stack campaign: select 128 shrunk Means,
# then the 16-noise stack versus inverse-variance control, plus an independent
# inner/outer 16-84% coverage split on the cached g01 catalog.
set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
SEED="${SEED:-42}"
GPU_EXCLUDE="${GPU_EXCLUDE:-v005}"
CASE="${CASE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded}"
MODEL_NAME="${MODEL_NAME:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_select.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi

mkdir -p \
    "${REPORT_ROOT}/00_select" \
    "${REPORT_ROOT}/01_stack" \
    "${REPORT_ROOT}/02_coverage"

SELECT_JOB=$(sbatch --parsable \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},CASE=${CASE},REPORT_DIR=${REPORT_ROOT}/00_select,SEED=${SEED},OVERWRITE=1" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_select.slurm")
echo "SELECT_JOB=${SELECT_JOB}"

STACK_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" \
    --dependency=afterok:${SELECT_JOB} \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},MODEL_NAME=${MODEL_NAME},DATA_DIR=${DATA_DIR},REPORT_ROOT=${REPORT_ROOT},GALAXIES=${REPORT_ROOT}/00_select/galaxies.npz,REPORT_DIR=${REPORT_ROOT}/01_stack,SEED=${SEED},OVERWRITE=1" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack.slurm")
echo "STACK_JOB=${STACK_JOB}"

INDEX_JOB=$(sbatch --parsable --dependency=afterok:${STACK_JOB} \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},REPORT_ROOT=${REPORT_ROOT}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_write_index.slurm")
echo "INDEX_JOB=${INDEX_JOB}"

STATUS="${REPORT_ROOT}/STATUS.txt"
{
    echo "likelihood-stack campaign"
    echo "========================="
    echo
    echo "Submitted $(date -R)"
    echo "CASE=${CASE}"
    echo "MODEL_NAME=${MODEL_NAME}"
    echo
    echo "SELECT_JOB=${SELECT_JOB}"
    echo "STACK_JOB=${STACK_JOB}"
    echo "INDEX_JOB=${INDEX_JOB}"
    echo
    echo "00_select  128 galaxies from the more-shrunk half of |g|>0.02"
    echo "01_stack   16 independent noises; IVW Means vs shared-parameter MAP"
    echo
    echo "Diagnostic only. Does not replace catalog multiplicative-bias tables."
} > "${STATUS}"

COVERAGE_JOB=$(sbatch --parsable \
    --export="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},CASE=${CASE},REPORT_DIR=${REPORT_ROOT}/02_coverage,OVERWRITE=1" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_coverage.slurm")
echo "COVERAGE_JOB=${COVERAGE_JOB}"
{
    echo
    echo "COVERAGE_JOB=${COVERAGE_JOB}  submitted $(date -R)"
    echo "02_coverage  16-84% coverage on |g|<0.05 vs |g|>0.05"
} >> "${STATUS}"

echo "Wrote ${STATUS}"

module load anaconda3 >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    KLNN_CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
    set +u
    conda activate kl-nn
    set -u
    python "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_write_index.py" \
        --report-root "${REPORT_ROOT}"
fi
