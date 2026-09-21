#!/bin/bash
# Submit likelihood-stack stage 03: frozen-encoder 2D NRE train, then xu3 infer.
# Appends job ids to STATUS.txt. Does not overwrite 00-02.
set -euo pipefail

KLNN_REPO_ROOT="${KLNN_REPO_ROOT:-/jet/home/xwang30/kl-nn}"
REPORT_ROOT="${REPORT_ROOT:-/ocean/projects/phy250048p/shared/reports/likelihood-stack}"
SEED="${SEED:-42}"
GPU_EXCLUDE="${GPU_EXCLUDE:-v005}"
PARENT_NPE="${PARENT_NPE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
NRE_NAME="${NRE_NAME:-CNN-CNN-Meta-nre2d-simv3-cosi-r90_valid100k_frozen_s42_fibrepair}"
TRAIN_DATA="${TRAIN_DATA:-/ocean/projects/phy250048p/shared/datasets/valid_100k_simv3_cosi}"
VALID_DATA="${VALID_DATA:-/ocean/projects/phy250048p/shared/datasets/small_10k_simv3_cosi}"
DATA_DIR="${DATA_DIR:-/ocean/projects/phy250048p/shared/datasets/test_100k_simv3_cosi_xu3_tf}"
CASE="${CASE:-CNN-CNN-Meta-bounded-hybrid-simv3-cosi-r90_valid100k_frozen_s42_fibrepair:test_100k_simv3_cosi_xu3_tf_testset_tfweighted_v2_10k_s42_righthanded}"

if [[ ! -f "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_train.py" ]]; then
    echo "KLNN_REPO_ROOT does not identify a KL-NN checkout: ${KLNN_REPO_ROOT}" >&2
    exit 2
fi
if [[ "${NRE_NAME}" == "${PARENT_NPE}" ]]; then
    echo "NRE_NAME must differ from PARENT_NPE" >&2
    exit 2
fi

mkdir -p "${REPORT_ROOT}/03_nre"

TRAIN_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},PARENT_NPE=${PARENT_NPE},NRE_NAME=${NRE_NAME},TRAIN_DATA=${TRAIN_DATA},VALID_DATA=${VALID_DATA},REPORT_ROOT=${REPORT_ROOT},SEED=${SEED},OVERWRITE=1"
INFER_EXPORT="KLNN_REPO_ROOT=${KLNN_REPO_ROOT},PARENT_NPE=${PARENT_NPE},NRE_NAME=${NRE_NAME},DATA_DIR=${DATA_DIR},CASE=${CASE},REPORT_ROOT=${REPORT_ROOT},REPORT_DIR=${REPORT_ROOT}/03_nre,SEED=${SEED},OVERWRITE=1"

NRE_TRAIN_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" \
    --export="${TRAIN_EXPORT}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_train.slurm")
echo "NRE_TRAIN_JOB=${NRE_TRAIN_JOB}"

NRE_INFER_JOB=$(sbatch --parsable --exclude="${GPU_EXCLUDE}" \
    --dependency=afterok:${NRE_TRAIN_JOB} \
    --export="${INFER_EXPORT}" \
    "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_nre_infer.slurm")
echo "NRE_INFER_JOB=${NRE_INFER_JOB}"

STATUS="${REPORT_ROOT}/STATUS.txt"
{
    echo
    echo "NRE submitted $(date -R)"
    echo "PARENT_NPE=${PARENT_NPE}"
    echo "NRE_NAME=${NRE_NAME}"
    echo "NRE_TRAIN_JOB=${NRE_TRAIN_JOB}"
    echo "NRE_INFER_JOB=${NRE_INFER_JOB}  afterok:${NRE_TRAIN_JOB}"
    echo "03_nre train  frozen 2D NRE head on valid_100k; new nre2d dir only"
    echo "03_nre infer  xu3 MAP / coverage / Bernstein vs cached Means"
} >> "${STATUS}"
echo "Wrote ${STATUS}"

module load anaconda3 >/dev/null 2>&1 || true
if command -v conda >/dev/null 2>&1; then
    KLNN_CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${KLNN_CONDA_BASE}/etc/profile.d/conda.sh"
    set +eu
    conda activate kl-nn
    activate_status=$?
    set -eu
    if [[ "${activate_status}" -eq 0 ]]; then
        python "${KLNN_REPO_ROOT}/arch/diagnostics/likelihood_stack_write_index.py" \
            --report-root "${REPORT_ROOT}"
    fi
fi
